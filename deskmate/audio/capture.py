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
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.want_mic = microphone
        self.want_loopback = loopback
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._chunks: queue.Queue[tuple[str, Path, int]] = queue.Queue(maxsize=128)

    def start(self) -> None:
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

    def stop(self) -> None:
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads.clear()

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

        chunk_samples = self.sample_rate * self.chunk_seconds
        buffer: list = []
        try:
            with sd.InputStream(**kwargs) as stream:
                while not self._stop.is_set():
                    data, _ = stream.read(int(self.sample_rate * 0.5))
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
        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=rate,
                input=True,
                input_device_index=default["index"],
                frames_per_buffer=int(rate * 0.5),
            )
            while not self._stop.is_set():
                buf.extend(stream.read(int(rate * 0.5), exception_on_overflow=False))
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
