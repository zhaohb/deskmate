"""User-facing audio / transcription pipeline status for API and UI."""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..config import AudioConfig, Config
    from .transcribe import WhisperTranscriber


def classify_model_load_error(exc: BaseException) -> tuple[str, str]:
    """Return (error_code, short_user_hint)."""
    text = f"{type(exc).__name__}: {exc}".lower()
    if "ssl" in text or "certificate" in text or "cert_" in text:
        return (
            "model_download_ssl",
            "Whisper model download failed (SSL certificate). Install certifi and set "
            "SSL_CERT_FILE, or pre-download the model (see logs).",
        )
    if "connection" in text or "timeout" in text or "network" in text or "connect" in text:
        return (
            "model_download_network",
            "Whisper model download failed (network). Check proxy/VPN or set HF_ENDPOINT "
            "to a mirror, then pre-download the model.",
        )
    if "local disk" in text or "snapshot folder" in text or "not find" in text:
        return (
            "model_not_cached",
            "Whisper model is not on disk yet. Pre-download: "
            "python -m huggingface_hub.cli download Systran/faster-whisper-<size>",
        )
    return (
        "model_load_failed",
        f"Whisper failed to load ({type(exc).__name__}). See ~/.deskmate/logs/deskmate.log",
    )


def whisper_model_repo(model_size: str) -> str:
    return f"Systran/faster-whisper-{model_size}"


def model_cache_dir(model_size: str) -> Path:
    return Path.home() / ".cache" / "huggingface" / "hub" / f"models--Systran--faster-whisper-{model_size}"


def model_cached(model_size: str) -> bool:
    root = model_cache_dir(model_size)
    if not root.is_dir():
        return False
    return any(root.iterdir())


def missing_audio_deps() -> list[str]:
    missing: list[str] = []
    if find_spec("sounddevice") is None:
        missing.append("sounddevice")
    if find_spec("faster_whisper") is None:
        missing.append("faster-whisper")
    return missing


def build_audio_status(
    cfg: Config,
    *,
    transcriber: WhisperTranscriber | None = None,
    capture_active: bool | None = None,
    transcript_count: int = 0,
) -> dict[str, Any]:
    """Summarize why transcripts may be empty."""
    audio: AudioConfig = cfg.audio
    if not audio.enabled:
        return {
            "enabled": False,
            "transcription_ready": False,
            "error_code": "audio_disabled",
            "hint": "Audio is disabled. Set [audio] enabled = true in ~/.deskmate/config.toml and restart.",
            "whisper_model": audio.whisper_model,
            "model_cached": False,
            "missing_deps": [],
            "capture_active": False,
        }

    missing = missing_audio_deps()
    if missing:
        return {
            "enabled": True,
            "transcription_ready": False,
            "error_code": "missing_deps",
            "hint": "Install audio extras: pip install -e \".[audio,vad]\" (missing: "
            + ", ".join(missing)
            + ").",
            "whisper_model": audio.whisper_model,
            "model_cached": model_cached(audio.whisper_model),
            "missing_deps": missing,
            "capture_active": bool(capture_active),
        }

    ready = bool(transcriber and transcriber.available)
    error_code: str | None = None
    hint: str | None = None

    if not ready:
        error_code = transcriber.load_error_code if transcriber else None
        hint = transcriber.user_hint if transcriber else None
        if not model_cached(audio.whisper_model):
            error_code = error_code or "model_not_cached"
            hint = hint or (
                f"Download Whisper model ({whisper_model_repo(audio.whisper_model)}), e.g. "
                "python -m huggingface_hub.cli download "
                f"{whisper_model_repo(audio.whisper_model)}"
            )
        else:
            error_code = error_code or "model_load_failed"
            hint = hint or "Whisper could not start. See ~/.deskmate/logs/deskmate.log"
    elif transcript_count == 0:
        error_code = "waiting_for_audio"
        hint = (
            "Transcription is ready. Allow microphone access, speak for 30+ seconds "
            f"(chunks are {audio.chunk_seconds}s), then refresh."
        )
        if capture_active is False:
            error_code = "capture_inactive"
            hint = (
                "Whisper is ready but microphone/loopback capture did not start. "
                'Install pip install -e ".[audio]" and check Windows microphone privacy.'
            )

    return {
        "enabled": True,
        "transcription_ready": ready,
        "error_code": error_code,
        "hint": hint,
        "whisper_model": audio.whisper_model,
        "model_repo": whisper_model_repo(audio.whisper_model),
        "model_cached": model_cached(audio.whisper_model),
        "model_cache_path": str(model_cache_dir(audio.whisper_model)),
        "missing_deps": [],
        "capture_active": bool(capture_active),
        "load_error_detail": transcriber.load_error_detail if transcriber else None,
    }
