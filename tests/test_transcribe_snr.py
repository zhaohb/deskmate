"""SNR-relative speech gating in the transcription orchestrator (⭐5)."""

from __future__ import annotations

from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")

from deskmate.audio.transcribe import (  # noqa: E402
    MIN_RMS_ENERGY,
    MIN_SNR_DB,
    WhisperTranscriber,
)

_SR = 16000


def _write(path: Path, samples) -> Path:
    sf.write(str(path), np.asarray(samples, dtype="float32"), _SR)
    return path


def test_silence_is_dropped(tmp_path: Path) -> None:
    wav = _write(tmp_path / "silence.wav", np.zeros(_SR))
    snr = WhisperTranscriber._audio_snr_db(wav)
    assert snr is not None
    rms, _snr_db = snr
    assert rms < MIN_RMS_ENERGY


def test_uniform_noise_has_low_snr(tmp_path: Path) -> None:
    # Stationary white noise has no speech peaks → low SNR → dropped.
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 0.02, _SR).astype("float32")
    wav = _write(tmp_path / "noise.wav", noise)
    snr = WhisperTranscriber._audio_snr_db(wav)
    assert snr is not None
    _rms, snr_db = snr
    assert snr_db < MIN_SNR_DB


def test_quiet_speech_above_quiet_floor_is_kept(tmp_path: Path) -> None:
    # A low-amplitude tone burst over a very quiet floor: the old absolute
    # MIN_RMS_ENERGY=0.015 gate would likely drop this soft signal, but its SNR
    # is high, so the relative gate keeps it.
    n = _SR
    floor = (np.random.default_rng(1).normal(0.0, 0.0008, n)).astype("float32")
    t = np.arange(n) / _SR
    burst = np.zeros(n, dtype="float32")
    seg = slice(int(0.3 * n), int(0.7 * n))
    burst[seg] = (0.03 * np.sin(2 * np.pi * 220 * t[seg])).astype("float32")
    wav = _write(tmp_path / "quiet_speech.wav", floor + burst)
    snr = WhisperTranscriber._audio_snr_db(wav)
    assert snr is not None
    rms, snr_db = snr
    assert snr_db >= MIN_SNR_DB
    # And it sits below the old absolute floor — proving the regression it fixes.
    assert rms < 0.015
