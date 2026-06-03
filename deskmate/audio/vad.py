"""Silero VAD wrapper for speech segmentation.

We use the `silero-vad` PyPI package which ships a self-contained ONNX/Torch
model and a small Python API. This module exposes:

- `SileroVAD(threshold=0.5, sampling_rate=16000)` — load once, reuse.
- `vad.iter_speech_chunks(pcm: np.ndarray) -> Iterator[(start_s, end_s)]`.
- `vad.split_recording(pcm, min_seg_ms=300, padding_ms=200) -> List[Segment]`.

If `silero-vad` isn't installed we transparently fall back to an energy-based
heuristic so the pipeline still functions; the segments will be coarser but
shaped identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from ..logger import get

logger = get("audio.vad")


@dataclass
class SpeechSegment:
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)


class SileroVAD:
    """Silero VAD wrapper. Loads model lazily."""

    def __init__(self, *, threshold: float = 0.5, sampling_rate: int = 16000) -> None:
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self._model = None
        self._get_speech_timestamps = None
        self._backend = "uninitialized"

    def _ensure_model(self) -> bool:
        if self._model is not None or self._backend == "energy":
            return True
        try:
            from silero_vad import load_silero_vad, get_speech_timestamps  # type: ignore[import-not-found]

            from ..model_status import loading  # noqa: PLC0415

            # The silero-vad pip package bundles the model file, so this is a
            # local load rather than a download.
            with loading("Silero VAD", cached=True):
                self._model = load_silero_vad()
            self._get_speech_timestamps = get_speech_timestamps
            self._backend = "silero"
            logger.info("silero-vad loaded (sampling_rate=%d)", self.sampling_rate)
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("silero-vad not available (%s); falling back to energy VAD", exc)
            self._backend = "energy"
            return True

    @property
    def backend(self) -> str:
        return self._backend

    def iter_speech_chunks(self, pcm) -> Iterator[SpeechSegment]:
        self._ensure_model()
        if self._backend == "silero":
            try:
                import numpy as np  # type: ignore[import-untyped]
                import torch  # type: ignore[import-not-found]

                if not isinstance(pcm, np.ndarray):
                    pcm = np.asarray(pcm, dtype="float32")
                tensor = torch.from_numpy(pcm.astype("float32"))
                ts = self._get_speech_timestamps(  # type: ignore[misc]
                    tensor, self._model, sampling_rate=self.sampling_rate,
                    threshold=self.threshold,
                )
                for s in ts:
                    yield SpeechSegment(
                        start_s=float(s["start"]) / self.sampling_rate,
                        end_s=float(s["end"]) / self.sampling_rate,
                    )
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("silero inference failed: %s — energy fallback", exc)
                self._backend = "energy"

        yield from _energy_vad(pcm, self.sampling_rate)

    def split_recording(
        self,
        pcm,
        *,
        min_seg_ms: int = 300,
        padding_ms: int = 200,
    ) -> list[SpeechSegment]:
        """Return contiguous speech segments with the requested padding.
        Segments shorter than `min_seg_ms` are discarded."""
        pad_s = padding_ms / 1000.0
        min_s = min_seg_ms / 1000.0
        segs: list[SpeechSegment] = []
        for s in self.iter_speech_chunks(pcm):
            if s.duration_s < min_s:
                continue
            segs.append(SpeechSegment(start_s=max(0.0, s.start_s - pad_s), end_s=s.end_s + pad_s))
        return segs


def _energy_vad(pcm, sampling_rate: int) -> Iterator[SpeechSegment]:
    """Tiny RMS-based fallback. ~30 ms hops; voiced if RMS > -45 dBFS."""
    try:
        import numpy as np  # type: ignore[import-untyped]
    except Exception:  # noqa: BLE001
        return iter(())

    if not isinstance(pcm, np.ndarray):
        pcm = np.asarray(pcm, dtype="float32")
    if pcm.ndim == 2:
        pcm = pcm.mean(axis=1)

    hop = max(1, int(sampling_rate * 0.030))
    rms_threshold = 10 ** (-45 / 20)
    in_seg = False
    seg_start = 0.0
    last_voiced = 0.0
    out: list[SpeechSegment] = []
    for start in range(0, len(pcm) - hop, hop):
        chunk = pcm[start:start + hop]
        rms = float((chunk ** 2).mean() ** 0.5) if chunk.size else 0.0
        voiced = rms > rms_threshold
        t = start / sampling_rate
        if voiced:
            if not in_seg:
                seg_start = t
                in_seg = True
            last_voiced = t + hop / sampling_rate
        elif in_seg and (t - last_voiced) > 0.5:
            out.append(SpeechSegment(start_s=seg_start, end_s=last_voiced))
            in_seg = False
    if in_seg:
        out.append(SpeechSegment(start_s=seg_start, end_s=last_voiced))
    yield from out
