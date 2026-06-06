"""Audio capture via `sounddevice`. Supports microphone and (best-effort)
WASAPI loopback for system audio on Windows.

The implementation uses `sounddevice`, which wraps PortAudio. WASAPI loopback
requires PortAudio v19.7+ or the `pyaudiowpatch` variant. If loopback isn't
available, recording degrades gracefully to microphone-only."""

from __future__ import annotations

import queue
import threading
import wave
from datetime import datetime, timezone
from pathlib import Path

from .. import paths
from ..logger import get

logger = get("audio.capture")


class AudioRecorder:
    """Continuously records `chunk_seconds` slices to WAV files and notifies a
    consumer queue. Caller can subscribe via `take_next_chunk()`."""

    def __init__(
        self,
        *,
        sample_rate: int = 16000,
        chunk_seconds: int = 30,
        microphone: bool = True,
        loopback: bool = True,
        chunk_mode: str = "fixed",
        endpoint_silence_ms: int = 700,
        endpoint_max_chunk_s: float = 8.0,
        endpoint_min_chunk_s: float = 1.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.want_mic = microphone
        self.want_loopback = loopback
        # "fixed" => accumulate chunk_seconds then emit (legacy).
        # "endpoint" => emit per utterance when a pause is detected (low latency).
        self.chunk_mode = chunk_mode
        self.endpoint_silence_ms = endpoint_silence_ms
        self.endpoint_max_chunk_s = endpoint_max_chunk_s
        self.endpoint_min_chunk_s = endpoint_min_chunk_s
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._chunks: queue.Queue[tuple[str, Path, int]] = queue.Queue(maxsize=128)
        self.capture_active = False

    def start(self) -> None:
        self.capture_active = False
        try:
            import sounddevice as sd  # noqa: F401, PLC0415
        except ImportError:
            logger.warning("sounddevice not installed; audio disabled")
            return

        paths.ensure_dirs()
        self._stop.clear()
        if self.want_mic:
            self._threads.append(threading.Thread(target=self._record_loop, args=("mic", None), name="audio-mic", daemon=True))
        if self.want_loopback:
            self._threads.append(threading.Thread(target=self._record_loop, args=("loopback", "loopback"), name="audio-loopback", daemon=True))
        for t in self._threads:
            t.start()
        self.capture_active = bool(self._threads)

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()
        self.capture_active = False

    def take_next_chunk(self, timeout: float | None = None) -> tuple[str, Path, int] | None:
        try:
            return self._chunks.get(timeout=timeout)
        except queue.Empty:
            return None

    def _record_loop(self, label: str, mode: str | None) -> None:
        try:
            import numpy as np  # noqa: PLC0415
            import sounddevice as sd  # noqa: PLC0415
        except ImportError as exc:
            logger.warning("audio deps missing for %s: %s", label, exc)
            return

        kwargs: dict = {"samplerate": self.sample_rate, "channels": 1, "dtype": "int16"}
        if mode == "loopback":
            try:
                import pyaudiowpatch as pyaudio  # noqa: F401, PLC0415
                # If pyaudiowpatch is available, prefer it for WASAPI loopback.
                self._record_loop_pyaudiowpatch(label)
                return
            except ImportError:
                # Try sounddevice's WASAPI exclusive loopback hint, available in newer builds.
                try:
                    hostapi = next(i for i, h in enumerate(sd.query_hostapis()) if "WASAPI" in h["name"])  # type: ignore[index]
                    default_out = sd.query_hostapis(hostapi)["default_output_device"]  # type: ignore[index]
                    extra_settings = _sounddevice_loopback_settings(sd)
                    if extra_settings is None:
                        logger.warning(
                            "WASAPI loopback unavailable: install pyaudiowpatch or use a sounddevice build with loopback support"
                        )
                        return
                    kwargs["device"] = default_out
                    kwargs["extra_settings"] = extra_settings
                except Exception as exc:  # noqa: BLE001
                    logger.warning("WASAPI loopback unavailable: %s", exc)
                    return

        # endpoint mode reads in smaller increments for finer pause resolution.
        read_seconds = 0.25 if self.chunk_mode == "endpoint" else 0.5
        read_frames = max(1, int(self.sample_rate * read_seconds))
        chunk_samples = self.sample_rate * self.chunk_seconds
        endpoint = self._make_endpoint_detector(self.sample_rate)
        buffer: list = []
        try:
            with sd.InputStream(**kwargs) as stream:
                while not self._stop.is_set():
                    data, _ = stream.read(read_frames)
                    if endpoint is not None:
                        arr = endpoint.add(data.copy())
                        if arr is not None:
                            self._emit(label, arr)
                        continue
                    buffer.append(data.copy())
                    total = sum(len(b) for b in buffer)
                    if total >= chunk_samples:
                        arr = np.concatenate(buffer)[:chunk_samples]
                        buffer = [np.concatenate(buffer)[chunk_samples:]] if total > chunk_samples else []
                        self._emit(label, arr)
        except Exception as exc:  # noqa: BLE001
            logger.warning("audio loop %s failed: %s", label, exc)


    def _record_loop_pyaudiowpatch(self, label: str) -> None:
        try:
            import numpy as np  # noqa: PLC0415
            import pyaudiowpatch as pyaudio  # noqa: PLC0415
        except ImportError:
            return
        pa = pyaudio.PyAudio()
        try:
            default = pa.get_default_wasapi_loopback()
        except Exception as exc:  # noqa: BLE001
            logger.warning("no WASAPI loopback device: %s", exc); pa.terminate(); return
        rate = int(default["defaultSampleRate"])
        channels = int(default["maxInputChannels"]) or 2
        stream = None
        buf = bytearray()
        target_bytes = rate * channels * 2 * self.chunk_seconds
        read_seconds = 0.25 if self.chunk_mode == "endpoint" else 0.5
        read_frames = max(1, int(rate * read_seconds))
        # Endpoint detection runs on mono float audio at the loopback's own rate.
        endpoint = self._make_endpoint_detector(rate)
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=rate,
                input=True,
                input_device_index=default["index"],
                frames_per_buffer=read_frames,
            )
            while not self._stop.is_set():
                raw = stream.read(read_frames, exception_on_overflow=False)
                if endpoint is not None:
                    frame = np.frombuffer(raw, dtype=np.int16)
                    if channels > 1:
                        frame = frame.reshape(-1, channels).mean(axis=1).astype(np.int16)
                    arr = endpoint.add(frame)
                    if arr is not None:
                        self._emit(label, arr, sample_rate=rate)
                    continue
                buf.extend(raw)
                if len(buf) >= target_bytes:
                    arr = np.frombuffer(buf[:target_bytes], dtype=np.int16)
                    buf = bytearray(buf[target_bytes:])
                    if channels > 1:
                        arr = arr.reshape(-1, channels).mean(axis=1).astype(np.int16)
                    self._emit(label, arr, sample_rate=rate)
        except Exception as exc:  # noqa: BLE001
            logger.warning("loopback loop failed: %s", exc)
        finally:
            _close_pyaudio_stream(stream)
            pa.terminate()

    def _make_endpoint_detector(self, sample_rate: int) -> "_EndpointBuffer | None":
        """Build the per-utterance endpoint detector for `endpoint` chunk mode.

        Returns None in `fixed` mode so the caller keeps the legacy fixed-window
        path. The detector accumulates frames and signals an emit when a spoken
        utterance is followed by a pause (or the max-chunk safety cap is hit)."""
        if self.chunk_mode != "endpoint":
            return None
        return _EndpointBuffer(
            sample_rate=sample_rate,
            silence_ms=self.endpoint_silence_ms,
            max_chunk_s=self.endpoint_max_chunk_s,
            min_chunk_s=self.endpoint_min_chunk_s,
        )

    def _emit(self, label: str, samples, *, sample_rate: int | None = None) -> None:
        import numpy as np  # noqa: PLC0415
        sr = sample_rate or self.sample_rate
        ts = datetime.now(timezone.utc).astimezone().strftime("%Y%m%dT%H%M%S")
        out = paths.audio_dir() / f"{ts}_{label}.wav"
        with wave.open(str(out), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(np.asarray(samples, dtype=np.int16).tobytes())
        duration_ms = int(len(samples) * 1000 / sr)
        try:
            self._chunks.put_nowait((label, out, duration_ms))
        except queue.Full:
            logger.warning("audio chunk queue full; dropping %s", out.name)


class _EndpointBuffer:
    """Streaming utterance endpointer for low-latency `endpoint` chunk mode.

    Frames are fed in via :meth:`add`. The buffer tracks whether it has heard
    speech and how long the trailing silence has run, using a cheap RMS energy
    gate (the same dBFS floor the energy-VAD fallback uses). When an utterance
    is followed by a pause of ``silence_ms`` — or the buffer reaches
    ``max_chunk_s`` — :meth:`add` returns the accumulated int16 mono samples and
    resets. Otherwise it returns ``None``.

    Pure energy gating here keeps the audio thread light; the heavyweight Silero
    VAD still runs downstream on the emitted chunk to split/clean segments. We
    only need "is the speaker pausing right now?" — energy answers that well and
    cheaply at 0.25s granularity.
    """

    # ~ -45 dBFS, matching the energy-VAD fallback in vad.py.
    _RMS_THRESHOLD = 10 ** (-45 / 20)

    def __init__(
        self,
        *,
        sample_rate: int,
        silence_ms: int,
        max_chunk_s: float,
        min_chunk_s: float,
    ) -> None:
        self.sample_rate = sample_rate
        self.silence_samples = int(sample_rate * silence_ms / 1000.0)
        self.max_samples = int(sample_rate * max_chunk_s)
        self.min_samples = int(sample_rate * min_chunk_s)
        self._frames: list = []
        self._total = 0
        self._heard_speech = False
        self._trailing_silence = 0  # consecutive silent samples at the tail

    def add(self, frame):
        """Append one read frame; return a finished utterance (int16 array) or None."""
        import numpy as np  # noqa: PLC0415

        arr = np.asarray(frame).reshape(-1)
        n = len(arr)
        if n == 0:
            return None
        self._frames.append(arr)
        self._total += n

        voiced = self._rms(arr) > self._RMS_THRESHOLD
        if voiced:
            self._heard_speech = True
            self._trailing_silence = 0
        else:
            self._trailing_silence += n

        # End of utterance: heard speech, then a long-enough pause.
        ended = self._heard_speech and self._trailing_silence >= self.silence_samples
        # Safety cap: emit a long monologue even without a pause.
        capped = self._total >= self.max_samples

        if not (ended or capped):
            return None

        # If we somehow only buffered silence/too little, don't emit a fragment —
        # keep accumulating (this also rolls "嗯"/"对" into the next utterance).
        if self._heard_speech and self._total >= self.min_samples:
            out = np.concatenate(self._frames).astype(np.int16)
            self._reset()
            return out
        if capped:
            # All-silence buffer hit the cap: drop it so silence isn't transcribed.
            self._reset()
        return None

    @staticmethod
    def _rms(arr) -> float:
        import numpy as np  # noqa: PLC0415

        if arr.size == 0:
            return 0.0
        # int16 → normalize to [-1, 1] for a dBFS-comparable RMS.
        x = arr.astype("float32") / 32768.0
        return float(np.sqrt((x * x).mean()))

    def _reset(self) -> None:
        self._frames = []
        self._total = 0
        self._heard_speech = False
        self._trailing_silence = 0


def _sounddevice_loopback_settings(sd) -> object | None:
    """Return sounddevice WASAPI loopback settings when this build supports it."""
    try:
        import inspect  # noqa: PLC0415

        params = inspect.signature(sd.WasapiSettings).parameters
    except Exception as exc:  # noqa: BLE001
        logger.debug("could not inspect sounddevice WasapiSettings: %s", exc)
        return None
    if "loopback" not in params:
        return None
    return sd.WasapiSettings(loopback=True)  # type: ignore[attr-defined]


def _close_pyaudio_stream(stream: object | None) -> None:
    """Best-effort cleanup for PyAudio streams that may already be closed."""
    if stream is None:
        return
    for method_name in ("stop_stream", "close"):
        method = getattr(stream, method_name, None)
        if method is None:
            continue
        try:
            method()
        except OSError as exc:
            logger.debug("pyaudio stream %s skipped: %s", method_name, exc)
        except Exception as exc:  # noqa: BLE001
            logger.debug("pyaudio stream %s failed during cleanup: %s", method_name, exc)
