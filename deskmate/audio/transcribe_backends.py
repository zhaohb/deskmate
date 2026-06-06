"""Pluggable Whisper inference backends.

Two backends sit behind a common :class:`TranscriptionBackend` interface so the
orchestrator (:class:`~deskmate.audio.transcribe.WhisperTranscriber`) stays
backend-agnostic:

- ``onnx_cpu``       → :class:`FasterWhisperBackend` (faster-whisper, CTranslate2).
- ``openvino_genai`` → :class:`WhisperGenAIBackend` (OpenVINO GenAI, NPU/GPU/CPU).

Each backend owns *only* model loading and a single-clip ``transcribe`` call.
Audio gating (RMS), VAD segmentation, clip writing and language resolution all
live in the orchestrator and are shared across backends. Adding a third backend
means adding one class here and one entry in :data:`BACKENDS` — no edits to the
orchestrator.

A backend that fails to import its deps or load its model reports the failure
via ``load_error_code`` / ``user_hint`` and leaves ``available`` False; the
orchestrator decides whether to fall back.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..logger import get
from .pipeline_status import classify_model_load_error

logger = get("audio.transcribe.backend")

# Whisper translation is always disabled: we transcribe in the source language.
WHISPER_TRANSLATE = False

# Hallucination-suppression thresholds for faster-whisper.
WHISPER_NO_SPEECH_THRESHOLD = 0.6
WHISPER_LOG_PROB_THRESHOLD = -2.0
WHISPER_COMPRESSION_RATIO_THRESHOLD = 2.4

ZH_INITIAL_PROMPT = "以下是普通话简体中文内容。"


@dataclass
class RawSegment:
    """One decoded span as returned by a backend, before offset/language fixup.

    The orchestrator adds the VAD ``base_offset`` and resolves the final
    language, so backends only report clip-relative times and the raw detected
    language (or None)."""

    text: str
    start: float
    end: float


@dataclass
class TranscribeResult:
    """A backend's output for one clip: raw segments + detected language."""

    segments: list[RawSegment]
    detected_language: str | None


@dataclass
class LoadStatus:
    """Outcome of a backend load attempt, surfaced to the user on failure."""

    available: bool
    error_code: str | None = None
    error_detail: str | None = None
    user_hint: str | None = None

    @classmethod
    def ok(cls) -> LoadStatus:
        return cls(available=True)

    @classmethod
    def fail(cls, code: str, detail: str, hint: str) -> LoadStatus:
        return cls(available=False, error_code=code, error_detail=detail, user_hint=hint)


def _set_translate(transcribe_kwargs: dict, translate: bool) -> None:
    """Map a translate flag onto the faster-whisper task.

    translate=False -> task="transcribe" (keep source language)
    translate=True  -> task="translate"  (translate to English; unused here)
    """
    transcribe_kwargs["task"] = "translate" if translate else "transcribe"


class TranscriptionBackend(ABC):
    """A Whisper inference engine. One instance owns one loaded model.

    ``languages`` is the configured language list (see WhisperTranscriber): a
    single entry forces that language; a single ``"zh"`` also injects the
    Mandarin initial prompt to steer Simplified output.
    """

    name: str = "abstract"

    def __init__(self, model_size: str, languages: list[str]) -> None:
        self.model_size = model_size
        self.languages = languages
        self._model = None

    @abstractmethod
    def load(self) -> LoadStatus:
        """Import deps and load the model. Sets ``self._model`` on success."""

    @abstractmethod
    def transcribe(self, wav_path: str, *, vad_filter: bool) -> TranscribeResult:
        """Transcribe one (already VAD-clipped) wav file.

        ``vad_filter`` lets the orchestrator request the engine's own VAD when
        Silero pre-segmentation was unavailable; backends without an internal
        VAD may ignore it.
        """


class FasterWhisperBackend(TranscriptionBackend):
    """``onnx_cpu`` — faster-whisper (CTranslate2 + ONNX Runtime)."""

    name = "onnx_cpu"

    def __init__(
        self,
        model_size: str,
        languages: list[str],
        *,
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        super().__init__(model_size, languages)
        self.device = device
        self.compute_type = compute_type

    def load(self) -> LoadStatus:
        try:
            from faster_whisper import WhisperModel  # noqa: PLC0415
        except ImportError:
            return LoadStatus.fail(
                "missing_deps",
                "faster-whisper not installed",
                'Install audio extras: pip install -e ".[audio,vad]"',
            )
        try:
            from ..model_status import loading, whisper_cached  # noqa: PLC0415

            cached = whisper_cached(self.model_size)
            with loading(
                f"Whisper ({self.model_size})",
                cached=cached,
                detail=f"backend=onnx_cpu, device={self.device}, compute={self.compute_type}",
            ):
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            logger.info("Whisper loaded with onnx_cpu backend (faster-whisper)")
            return LoadStatus.ok()
        except Exception as exc:  # noqa: BLE001
            code, hint = classify_model_load_error(exc)
            logger.warning("faster-whisper init failed: %s (%s)", exc, hint)
            return LoadStatus.fail(code, str(exc), hint)

    def transcribe(self, wav_path: str, *, vad_filter: bool) -> TranscribeResult:
        transcribe_kwargs: dict = {
            "beam_size": 1,
            "vad_filter": vad_filter,
            "word_timestamps": False,
            "no_speech_threshold": WHISPER_NO_SPEECH_THRESHOLD,
            "log_prob_threshold": WHISPER_LOG_PROB_THRESHOLD,
            "compression_ratio_threshold": WHISPER_COMPRESSION_RATIO_THRESHOLD,
        }
        _set_translate(transcribe_kwargs, WHISPER_TRANSLATE)

        if len(self.languages) == 1:
            transcribe_kwargs["language"] = self.languages[0]
            if self.languages[0] == "zh":
                transcribe_kwargs["initial_prompt"] = ZH_INITIAL_PROMPT
        # len == 0 → auto-detect; len > 1 → detect then constrain (orchestrator).

        segments, info = self._model.transcribe(wav_path, **transcribe_kwargs)

        raw = [
            RawSegment(
                text=s.text or "",
                start=float(s.start) if s.start is not None else 0.0,
                end=float(s.end) if s.end is not None else 0.0,
            )
            for s in segments
        ]
        return TranscribeResult(segments=raw, detected_language=getattr(info, "language", None))


# GenAI WhisperPipeline expects 16 kHz mono audio.
GENAI_SAMPLE_RATE = 16000


class WhisperGenAIBackend(TranscriptionBackend):
    """``openvino_genai`` — OpenVINO GenAI ``WhisperPipeline``.

    Uses Intel's official ``openvino_genai`` package. The pipeline runs the
    whole Whisper graph (encoder + decoder, with internal static-shape handling
    and sequential chunking for >30 s audio) on a single OpenVINO ``device``.

    Validated on an Intel Core Ultra X7 358H (AI Boost NPU + Arc iGPU): NPU,
    GPU and CPU all transcribe correctly. NPU/GPU inference (~0.7 s/clip) beats
    CPU (~2 s); NPU's first compile is slow (~3 min) but ``CACHE_DIR`` persists
    the compiled blob so later starts load in ~3 s. If the chosen device can't
    load, we fall back to CPU before giving up.

    Models are GenAI-format OpenVINO IR (encoder/decoder/tokenizer ``.xml`` +
    ``.bin``). ``model`` is either a local directory or a ModelScope model id
    (auto-downloaded on first use, e.g. ``OpenVINO/whisper-medium-int8-ov``).
    """

    name = "openvino_genai"

    # Devices we try, in order, when the requested one fails to load. The
    # requested device is tried first; CPU is the universal terminal fallback.
    _DEVICE_FALLBACK = "CPU"

    def __init__(
        self,
        model: str,
        languages: list[str],
        *,
        device: str = "NPU",
        cache_dir: str | None = None,
    ) -> None:
        # `model` here is a ModelScope id or local IR path, not a size tier.
        super().__init__(model, languages)
        self.model_ref = model
        self.device = (device or "NPU").upper()
        self.cache_dir = cache_dir
        self._active_device: str | None = None

    def _resolve_model_dir(self) -> str:
        """Return a local IR directory, downloading from ModelScope if needed."""
        from pathlib import Path  # noqa: PLC0415

        if Path(self.model_ref).expanduser().exists():
            return str(Path(self.model_ref).expanduser())
        # Treat as a ModelScope model id.
        from modelscope import snapshot_download  # noqa: PLC0415

        return snapshot_download(self.model_ref)

    def load(self) -> LoadStatus:
        try:
            import openvino_genai as ov_genai  # noqa: PLC0415
        except ImportError:
            return LoadStatus.fail(
                "missing_deps",
                "openvino-genai not installed",
                'Install OpenVINO extras: pip install -e ".[audio-openvino]"',
            )
        try:
            model_dir = self._resolve_model_dir()
        except Exception as exc:  # noqa: BLE001
            code, hint = classify_model_load_error(exc)
            logger.warning("GenAI model resolve/download failed: %s (%s)", exc, hint)
            return LoadStatus.fail(code, str(exc), hint)

        from ..model_status import loading  # noqa: PLC0415

        # Try the requested device, then CPU. Pass CACHE_DIR so the (slow) NPU
        # compile is persisted and later starts are fast.
        kwargs: dict = {}
        if self.cache_dir:
            from pathlib import Path  # noqa: PLC0415

            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            kwargs["CACHE_DIR"] = self.cache_dir

        devices = [self.device]
        if self.device != self._DEVICE_FALLBACK:
            devices.append(self._DEVICE_FALLBACK)

        last_exc: Exception | None = None
        for dev in devices:
            try:
                with loading(
                    f"Whisper ({self.model_ref})",
                    cached=bool(self.cache_dir),
                    detail=f"backend=openvino_genai, device={dev}",
                ):
                    self._model = ov_genai.WhisperPipeline(model_dir, dev, **kwargs)
                self._active_device = dev
                # Warm up: the very first inference right after an NPU compile can
                # raise a transient "roi_end <= max_dim" before the device settles.
                # A throwaway run on 1 s of silence absorbs it so the first real
                # clip succeeds.
                self._warmup()
                if dev != self.device:
                    logger.warning(
                        "openvino_genai device %r unavailable; using %r", self.device, dev
                    )
                logger.info("Whisper loaded with openvino_genai backend (device=%s)", dev)
                return LoadStatus.ok()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning("openvino_genai load on %s failed: %s", dev, exc)

        code, hint = classify_model_load_error(last_exc or RuntimeError("load failed"))
        return LoadStatus.fail(code, str(last_exc), hint)

    def _warmup(self) -> None:
        """Run one throwaway inference on silence to settle the device.

        Best-effort: a failure here is non-fatal (the real call retries), so we
        only log it."""
        try:
            silence = [0.0] * 16000  # 1 s @ 16 kHz
            self._model.generate(silence, task="transcribe")
        except Exception as exc:  # noqa: BLE001
            logger.debug("openvino_genai warmup failed (non-fatal): %s", exc)

    def transcribe(self, wav_path: str, *, vad_filter: bool) -> TranscribeResult:
        # GenAI handles its own decoding; Silero pre-segmentation upstream
        # already bounds the clip, so vad_filter is intentionally ignored.
        raw_speech = _read_audio_16k(wav_path)
        if raw_speech is None or len(raw_speech) == 0:
            return TranscribeResult(segments=[], detected_language=None)

        gen_kwargs: dict = {
            "task": "translate" if WHISPER_TRANSLATE else "transcribe",
            "return_timestamps": True,
        }
        language = None
        if len(self.languages) == 1:
            language = self.languages[0]
            # GenAI expects the special-token form, e.g. "<|zh|>". Forcing the
            # language already steers Simplified-Chinese output, so we do NOT add
            # ZH_INITIAL_PROMPT here: on the NPU, an initial_prompt makes the
            # pipeline raise "roi_end <= max_dim" and the clip fails entirely.
            gen_kwargs["language"] = f"<|{language}|>"

        result = self._model.generate(raw_speech, **gen_kwargs)

        # result.chunks carries (start_ts, end_ts, text); fall back to the flat
        # string when timestamps are absent.
        chunks = getattr(result, "chunks", None)
        if chunks:
            raw = [
                RawSegment(
                    text=ch.text or "",
                    start=float(ch.start_ts) if ch.start_ts is not None else 0.0,
                    end=float(ch.end_ts) if ch.end_ts is not None else 0.0,
                )
                for ch in chunks
            ]
        else:
            raw = [RawSegment(text=str(result), start=0.0, end=0.0)]
        return TranscribeResult(segments=raw, detected_language=language)


def _read_audio_16k(wav_path: str):
    """Read a wav file as a mono float32 list resampled to 16 kHz, or None."""
    try:
        import numpy as np  # type: ignore[import-untyped]  # noqa: PLC0415
        import soundfile as sf  # type: ignore[import-untyped]  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        logger.warning("soundfile/numpy unavailable for GenAI audio read: %s", exc)
        return None
    try:
        audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to read %s: %s", wav_path, exc)
        return None
    if getattr(audio, "ndim", 1) == 2:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if int(sr) != GENAI_SAMPLE_RATE and audio.size:
        duration = len(audio) / float(sr)
        target_len = max(1, int(duration * GENAI_SAMPLE_RATE))
        src_x = np.linspace(0.0, duration, num=len(audio), endpoint=False)
        tgt_x = np.linspace(0.0, duration, num=target_len, endpoint=False)
        audio = np.interp(tgt_x, src_x, audio).astype(np.float32)
    return audio.tolist()


# Registry: backend name → factory. Adding a backend is a one-line change here
# plus the class above; the orchestrator never needs editing.
BACKENDS: dict[str, type[TranscriptionBackend]] = {
    FasterWhisperBackend.name: FasterWhisperBackend,
    WhisperGenAIBackend.name: WhisperGenAIBackend,
}

# The fallback chain when a preferred backend is unavailable. openvino_genai
# degrades to onnx_cpu; onnx_cpu is terminal.
FALLBACK_BACKEND: dict[str, str | None] = {
    "openvino_genai": "onnx_cpu",
    "onnx_cpu": None,
}
