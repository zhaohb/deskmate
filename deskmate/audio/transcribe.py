"""Whisper transcription via `faster-whisper`.

This module returns time-aligned segments so the caller can persist multiple
`audio_transcriptions` rows per chunk. Silero VAD pre-segmentation is applied
before sending to Whisper.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from ..logger import get
from .vad import SileroVAD, SpeechSegment

logger = get("audio.transcribe")

# screenpipe: params.set_translate(false) in whisper/batch.rs (always false, not configurable)
WHISPER_TRANSLATE = False

# screenpipe: MIN_RMS_ENERGY in whisper/batch.rs — skip near-silent audio (Whisper hallucinates)
MIN_RMS_ENERGY = 0.015

# Minimum VAD clip duration; shorter clips produce garbage Chinese fragments
MIN_CLIP_DURATION_S = 1.0

# screenpipe whisper/batch.rs hallucination thresholds (faster-whisper equivalents)
WHISPER_NO_SPEECH_THRESHOLD = 0.6
WHISPER_LOG_PROB_THRESHOLD = -2.0
WHISPER_COMPRESSION_RATIO_THRESHOLD = 2.4

ZH_INITIAL_PROMPT = "以下是普通话简体中文内容。"


def _set_translate(transcribe_kwargs: dict, translate: bool) -> None:
    """Mirror whisper_rs FullParams::set_translate for faster-whisper.

    screenpipe: params.set_translate(false) → task="transcribe"
    screenpipe: params.set_translate(true)  → task="translate" (not used in screenpipe)
    """
    transcribe_kwargs["task"] = "translate" if translate else "transcribe"


@dataclass
class TranscriptSegment:
    text: str
    start_time: float
    end_time: float
    language: str | None = None
    speaker_id: int | None = None


class WhisperTranscriber:
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
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.vad_min_segment_ms = vad_min_segment_ms
        self.vad_padding_ms = vad_padding_ms
        # Language handling identical to screenpipe:
        #   [] = auto-detect (Whisper decides)
        #   ["zh"] = force Simplified Chinese (skip detection, always use "zh")
        #   ["zh", "en"] = constrained detect (must be one of these)
        self.languages: list[str] = [l for l in (languages or []) if l]
        self._model = None
        self._available = self._try_load()
        self._vad = SileroVAD(threshold=vad_threshold)

    @property
    def available(self) -> bool:
        return self._available

    def _try_load(self) -> bool:
        try:
            from faster_whisper import WhisperModel  # noqa: PLC0415
        except ImportError:
            logger.warning("faster-whisper not installed; transcription disabled")
            return False
        try:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("faster-whisper init failed: %s", exc)
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
        if not self._available or self._model is None:
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
        """Resolve the effective language following screenpipe's logic:

        - 1 configured language → always force that language
        - 0 configured languages → auto-detect (return detected or None)
        - >1 configured languages → constrained: only accept if detected is
          in the list, otherwise fall back to the first configured language

        This mirrors screenpipe's detect_language() in
        crates/screenpipe-audio/src/transcription/whisper/detect_language.rs
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
        try:
            rms = self._audio_rms(wav_path)
            if rms is not None and rms < MIN_RMS_ENERGY:
                logger.debug(
                    "skip whisper for %s: RMS %.6f < %.6f",
                    wav_path.name, rms, MIN_RMS_ENERGY,
                )
                return []

            # Build kwargs matching screenpipe's Whisper params:
            # - set_language(lang): force or constrain language
            # - set_translate(false): transcribe only, no translation
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
                # Force single language (screenpipe: languages.len() == 1)
                transcribe_kwargs["language"] = self.languages[0]
                if self.languages[0] == "zh":
                    transcribe_kwargs["initial_prompt"] = ZH_INITIAL_PROMPT
            elif len(self.languages) > 1:
                # Let faster-whisper detect, then we constrain below
                pass
            # else: [] = fully automatic (screenpipe: languages.is_empty())

            segments, info = self._model.transcribe(
                str(wav_path), **transcribe_kwargs,
            )

            detected_lang = getattr(info, "language", None)
            language = self._resolve_language(detected_lang)

            out: list[TranscriptSegment] = []
            for s in segments:
                text = (s.text or "").strip()
                if not text:
                    continue
                out.append(TranscriptSegment(
                    text=text,
                    start_time=base_offset + (float(s.start) if s.start is not None else 0.0),
                    end_time=base_offset + (float(s.end) if s.end is not None else 0.0),
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
