"""Whisper transcription orchestrator.

This module owns the backend-agnostic transcription pipeline — RMS gating,
Silero VAD pre-segmentation, per-clip writing, language resolution and offset
fixup — and delegates the actual inference to a pluggable
:class:`~deskmate.audio.transcribe_backends.TranscriptionBackend`:

- ``onnx_cpu``       → faster-whisper (CTranslate2 + ONNX Runtime)
- ``openvino_genai`` → OpenVINO GenAI WhisperPipeline (NPU/GPU/CPU)

The backend is selected from config (``[audio] whisper_backend``); if the
preferred backend can't load, the orchestrator falls back along
:data:`~deskmate.audio.transcribe_backends.FALLBACK_BACKEND`.

Returns time-aligned segments so the caller can persist multiple
`audio_transcriptions` rows per chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from ..logger import get

# Re-exported for backward compatibility (tests / external callers import these
# names from this module).
from .transcribe_backends import (  # noqa: F401
    BACKENDS,
    FALLBACK_BACKEND,
    WHISPER_COMPRESSION_RATIO_THRESHOLD,
    WHISPER_LOG_PROB_THRESHOLD,
    WHISPER_NO_SPEECH_THRESHOLD,
    WHISPER_TRANSLATE,
    ZH_INITIAL_PROMPT,
    FasterWhisperBackend,
    TranscriptionBackend,
    WhisperGenAIBackend,
    _set_translate,
)
from .vad import SileroVAD, SpeechSegment

logger = get("audio.transcribe")

# Skip near-silent audio below this RMS energy (Whisper hallucinates on silence).
MIN_RMS_ENERGY = 0.015

# Minimum VAD clip duration; shorter clips produce garbage Chinese fragments
MIN_CLIP_DURATION_S = 1.0


@dataclass
class TranscriptSegment:
    text: str
    start_time: float
    end_time: float
    language: str | None = None
    speaker_id: int | None = None


class WhisperTranscriber:
    """Backend-agnostic transcription orchestrator.

    Owns VAD segmentation, RMS gating, clip writing and language resolution;
    delegates inference to a :class:`TranscriptionBackend` chosen by ``backend``
    (``"onnx_cpu"`` | ``"openvino_genai"``). On load failure it walks the
    :data:`FALLBACK_BACKEND` chain before giving up.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        *,
        vad_threshold: float = 0.5,
        vad_min_segment_ms: int = 300,
        vad_padding_ms: int = 200,
        compute_type: str = "int8",
        languages: list[str] | None = None,
        backend: str = "onnx_cpu",
        openvino_genai_model: str = "OpenVINO/whisper-medium-int8-ov",
        openvino_device: str = "NPU",
        openvino_cache_dir: str | None = None,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.requested_backend = backend  # what config asked for
        # openvino_genai backend settings
        self.openvino_genai_model = openvino_genai_model
        self.openvino_device = openvino_device
        self.openvino_cache_dir = openvino_cache_dir
        self.vad_min_segment_ms = vad_min_segment_ms
        self.vad_padding_ms = vad_padding_ms
        # Language handling:
        #   [] = auto-detect (Whisper decides)
        #   ["zh"] = force Simplified Chinese (skip detection, always use "zh")
        #   ["zh", "en"] = constrained detect (must be one of these)
        self.languages: list[str] = [l for l in (languages or []) if l]
        self._backend: TranscriptionBackend | None = None
        self.load_error_code: str | None = None
        self.load_error_detail: str | None = None
        self.user_hint: str | None = None
        self._available = self._load_backend_chain(backend)
        self._vad = SileroVAD(threshold=vad_threshold)
        if not self._available and self.user_hint:
            logger.warning("transcription unavailable: %s", self.user_hint)

    @property
    def available(self) -> bool:
        return self._available

    @property
    def backend(self) -> str:
        """The backend actually in use (may differ from requested after fallback)."""
        return self._backend.name if self._backend else self.requested_backend

    def set_languages(self, languages: list[str]) -> list[str]:
        """Hot-swap the transcription language list — no model reload.

        ``languages`` is read per clip (not at model load), so updating it takes
        effect on the next audio chunk. Safe to call from another thread: we
        assign a fresh list to both the orchestrator and the active backend, so
        the audio loop only ever sees a complete old or new list, never a torn
        one. Returns the normalized list that is now in effect.
        """
        normalized = [l for l in (languages or []) if l]
        self.languages = normalized
        if self._backend is not None:
            self._backend.languages = normalized
        logger.info("transcription languages updated to %r", normalized)
        return normalized

    def _build_backend(self, name: str) -> TranscriptionBackend:
        """Instantiate a backend by name, wiring backend-specific options."""
        cls = BACKENDS[name]
        if cls is FasterWhisperBackend:
            return cls(
                self.model_size,
                self.languages,
                device=self.device,
                compute_type=self.compute_type,
            )
        if cls is WhisperGenAIBackend:
            return cls(
                self.openvino_genai_model,
                self.languages,
                device=self.openvino_device,
                cache_dir=self.openvino_cache_dir or None,
            )
        return cls(self.model_size, self.languages)

    def _load_backend_chain(self, name: str | None) -> bool:
        """Try ``name``, then walk FALLBACK_BACKEND until one loads or all fail."""
        while name is not None:
            if name not in BACKENDS:
                logger.warning("unknown whisper backend %r; using onnx_cpu", name)
                name = "onnx_cpu"
            backend = self._build_backend(name)
            status = backend.load()
            if status.available:
                self._backend = backend
                self.load_error_code = None
                self.load_error_detail = None
                self.user_hint = None
                if name != self.requested_backend:
                    logger.warning(
                        "whisper backend %r unavailable; using %r instead",
                        self.requested_backend, name,
                    )
                return True
            # Record the failure, then try the fallback.
            self.load_error_code = status.error_code
            self.load_error_detail = status.error_detail
            self.user_hint = status.user_hint
            nxt = FALLBACK_BACKEND.get(name)
            if nxt is not None:
                logger.warning("whisper backend %r failed to load; trying %r", name, nxt)
            name = nxt
        return False

    def transcribe(self, wav_path: Path) -> tuple[str, str | None]:
        """Backwards-compat one-shot transcription. Returns (text, language)."""
        segs = self.transcribe_segments(wav_path)
        if not segs:
            return "", None
        return " ".join(s.text for s in segs).strip(), segs[0].language

    def transcribe_segments(self, wav_path: Path) -> list[TranscriptSegment]:
        """Return per-segment transcription with start/end times in seconds.
        Each `TranscriptSegment` matches one row in `audio_transcriptions`."""
        if not self._available or self._backend is None:
            return []

        speech_segments = self._speech_segments(wav_path)
        if speech_segments is None:
            return self._transcribe_file(wav_path, base_offset=0.0, vad_filter=True)
        if not speech_segments:
            return []

        # Whisper's own VAD is disabled here because Silero has already split
        # the audio. This keeps DB rows aligned with the VAD segment boundaries.
        out: list[TranscriptSegment] = []
        try:
            clip_source = self._load_clip_source(wav_path)
            with TemporaryDirectory(prefix="pc_assistant_vad_") as tmp:
                for idx, speech in enumerate(speech_segments):
                    clip = self._write_clip(
                        wav_path,
                        speech,
                        Path(tmp) / f"segment_{idx}.wav",
                        clip_source=clip_source,
                    )
                    if clip is None:
                        continue
                    if speech.duration_s < MIN_CLIP_DURATION_S:
                        continue
                    out.extend(
                        self._transcribe_file(
                            clip,
                            base_offset=speech.start_s,
                            vad_filter=False,
                        )
                    )
            return out
        except Exception as exc:  # noqa: BLE001
            logger.warning("segmented transcribe failed for %s: %s", wav_path.name, exc)
            return self._transcribe_file(wav_path, base_offset=0.0, vad_filter=True)

    def _resolve_language(self, detected: str | None) -> str | None:
        """Resolve the effective language:

        - 1 configured language → always force that language
        - 0 configured languages → auto-detect (return detected or None)
        - >1 configured languages → constrained: only accept if detected is
          in the list, otherwise fall back to the first configured language
        """
        if len(self.languages) == 1:
            return self.languages[0]
        if not self.languages:
            return detected
        if detected and detected in self.languages:
            return detected
        return self.languages[0]

    @staticmethod
    def _audio_rms(wav_path: Path) -> float | None:
        try:
            import numpy as np  # type: ignore[import-untyped]  # noqa: PLC0415
            import soundfile as sf  # type: ignore[import-untyped]  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            return None
        try:
            audio, _ = sf.read(str(wav_path), dtype="float32", always_2d=False)
        except Exception:  # noqa: BLE001
            return None
        if getattr(audio, "ndim", 1) == 2:
            audio = audio.mean(axis=1)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            return 0.0
        return float(np.sqrt((audio * audio).mean()))

    def _transcribe_file(
        self,
        wav_path: Path,
        *,
        base_offset: float,
        vad_filter: bool,
    ) -> list[TranscriptSegment]:
        """Backend-agnostic single-clip transcription.

        Applies shared RMS gating, delegates raw inference to the active
        backend, then resolves the language and shifts each segment by
        ``base_offset`` to recover absolute (recording-relative) times.
        """
        try:
            rms = self._audio_rms(wav_path)
            if rms is not None and rms < MIN_RMS_ENERGY:
                logger.debug(
                    "skip whisper for %s: RMS %.6f < %.6f",
                    wav_path.name, rms, MIN_RMS_ENERGY,
                )
                return []

            result = self._backend.transcribe(str(wav_path), vad_filter=vad_filter)
            language = self._resolve_language(result.detected_language)

            out: list[TranscriptSegment] = []
            for seg in result.segments:
                text = (seg.text or "").strip()
                if not text:
                    continue
                out.append(TranscriptSegment(
                    text=text,
                    start_time=base_offset + seg.start,
                    end_time=base_offset + seg.end,
                    language=language,
                ))
            return out
        except Exception as exc:  # noqa: BLE001
            logger.warning("transcribe failed for %s: %s", wav_path.name, exc)
            return []

    def _speech_segments(self, wav_path: Path) -> list[SpeechSegment] | None:
        """Return VAD speech segments, or None when audio loading failed."""
        try:
            import numpy as np  # type: ignore[import-untyped]  # noqa: PLC0415
            import soundfile as sf  # type: ignore[import-untyped]  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            logger.warning("soundfile/numpy unavailable; falling back to Whisper VAD: %s", exc)
            return None

        try:
            audio, sample_rate = sf.read(str(wav_path), dtype="float32", always_2d=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to read audio for VAD (%s): %s", wav_path.name, exc)
            return None

        if getattr(audio, "ndim", 1) == 2:
            audio = audio.mean(axis=1)
        audio = np.asarray(audio, dtype="float32")
        if audio.size == 0:
            return []

        vad_audio = audio
        if int(sample_rate) != self._vad.sampling_rate:
            vad_audio = _resample_linear(audio, int(sample_rate), self._vad.sampling_rate)

        duration_s = len(audio) / float(sample_rate)
        segments = self._vad.split_recording(
            vad_audio,
            min_seg_ms=self.vad_min_segment_ms,
            padding_ms=self.vad_padding_ms,
        )
        clamped = [_clamp_segment(seg, duration_s) for seg in segments]
        return [seg for seg in clamped if seg.duration_s > 0]

    @staticmethod
    def _load_clip_source(source: Path):
        """Read the source WAV once so multiple VAD clips don't repeatedly hit disk."""
        try:
            import soundfile as sf  # type: ignore[import-untyped]  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            logger.warning("soundfile unavailable; cannot cache VAD source: %s", exc)
            return None
        try:
            audio, sample_rate = sf.read(str(source), dtype="float32", always_2d=False)
            return audio, sample_rate
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to cache VAD source: %s", exc)
            return None

    @staticmethod
    def _write_clip(
        source: Path,
        segment: SpeechSegment,
        target: Path,
        *,
        clip_source=None,
    ) -> Path | None:
        try:
            import soundfile as sf  # type: ignore[import-untyped]  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            logger.warning("soundfile unavailable; cannot write VAD clip: %s", exc)
            return None
        try:
            if clip_source is None:
                audio, sample_rate = sf.read(str(source), dtype="float32", always_2d=False)
            else:
                audio, sample_rate = clip_source
            start = max(0, int(segment.start_s * sample_rate))
            end = min(len(audio), int(segment.end_s * sample_rate))
            if end <= start:
                return None
            sf.write(str(target), audio[start:end], sample_rate)
            return target
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to write VAD clip: %s", exc)
            return None

    @property
    def vad(self) -> SileroVAD:
        return self._vad


def _resample_linear(audio, source_rate: int, target_rate: int):
    if source_rate == target_rate:
        return audio
    import numpy as np  # type: ignore[import-untyped]  # noqa: PLC0415

    duration = len(audio) / float(source_rate)
    target_len = max(1, int(duration * target_rate))
    source_x = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    target_x = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(target_x, source_x, audio).astype("float32")


def _clamp_segment(segment: SpeechSegment, duration_s: float) -> SpeechSegment:
    return SpeechSegment(
        start_s=max(0.0, min(segment.start_s, duration_s)),
        end_s=max(0.0, min(segment.end_s, duration_s)),
    )
