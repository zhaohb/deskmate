"""Configuration for pc_assistant.

Reads `~/.pc_assistant/config.toml` (auto-created with defaults on first run).
Environment variables prefixed with `PCA_` override file values.

Ollama settings for Ask / pipe apps live under ``[ollama]``; ``OLLAMA_BASE``,
``OLLAMA_MODEL``, and ``OLLAMA_CHAT_TIMEOUT`` still override the file when set.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from . import paths

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


class CaptureConfig(BaseModel):
    enabled: bool = True
    # Event-driven capture is the primary path; heartbeat is fallback when disabled.
    event_driven: bool = True
    heartbeat_seconds: int = 60
    min_capture_gap_seconds: float = 1.0
    debounce_seconds: float = 1.5
    min_capture_interval_ms: int = 200
    idle_capture_interval_ms: int = 30_000
    capture_on_keystroke: bool = False
    capture_on_clipboard: bool = False
    record_input_events: bool = True
    ui_event_batch_size: int = 50
    ui_event_batch_timeout_ms: int = 1000
    scroll_stop_delay_ms: int = 300
    include_screenshot: bool = True
    screenshot_max_width: int = 1920
    screenshot_jpeg_quality: int = 80
    all_monitors: bool = False
    pause_extraction_on_input_ms: int = 800
    # Legacy heartbeat when event_driven=false
    adaptive_fps_floor: bool = True


class A11yConfig(BaseModel):
    enabled: bool = True
    ax_depth: int = 60
    ax_max_nodes: int = 5000
    text_input_debounce_seconds: float = 0.3
    capture_clicks: bool = True
    capture_keystrokes: bool = True
    capture_clipboard: bool = True
    capture_mouse_move: bool = False


class OcrConfig(BaseModel):
    engine: Literal["off", "winrt", "tesseract"] = "winrt"
    # WinRT: zh-CN → zh-Hans; include en-US for mixed UI (bookmarks, URLs)
    languages: list[str] = ["zh-CN", "en-US"]
    tesseract_cmd: str | None = None


class AudioConfig(BaseModel):
    enabled: bool = False
    sample_rate: int = 16000
    chunk_seconds: int = 30
    loopback: bool = True
    microphone: bool = True
    # Default tier uses large-v3-turbo; small is a balanced default for Chinese
    whisper_model: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    vad_threshold: float = 0.5
    # Avoid very short clips; 300ms fragments cause Whisper hallucinations
    vad_min_segment_ms: int = 1000
    vad_padding_ms: int = 200
    speaker_recognition: bool = False
    # Language codes for transcription (ISO 639-1). [] = auto-detect all languages.
    languages: list[str] = ["zh"]


class RedactConfig(BaseModel):
    enabled: bool = False
    rules: list[str] = ["email", "credit_card", "phone_cn", "phone_us", "ipv4", "ssn"]
    # ONNX-based PII detector (optional). If `onnx_model_path` is set, a
    # background reconciler will re-scan stored text with this model and
    # populate the `redacted_text` / `redacted_text_json` columns.
    onnx_model_path: str | None = None
    onnx_tokenizer_path: str | None = None
    onnx_threshold: float = 0.5


class FilterConfig(BaseModel):
    ignored_apps: list[str] = ["1password", "bitwarden", "keepassxc", "lastpass", "lockapp", "logonui"]
    ignored_windows: list[str] = []
    included_windows: list[str] = []
    ignore_incognito: bool = True


class OllamaConfig(BaseModel):
    """Local Ollama endpoint for Ask and pipe apps."""

    base: str = "http://127.0.0.1:11434"
    model: str = "qwen3_8b_ov:v1"
    chat_timeout: int = 600


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 3030


class OutlookConfig(BaseModel):
    client_id: str = ""
    tenant: str = "common"
    redirect_uri: str | None = None
    scopes: list[str] = ["offline_access", "User.Read", "Mail.Read", "Mail.Send"]


class GmailConfig(BaseModel):
    client_id: str = ""
    client_secret: str = ""
    redirect_uri: str | None = None
    scopes: list[str] = [
        "openid",
        "email",
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.send",
    ]


class RetentionConfig(BaseModel):
    frame_days: int = 30
    audio_days: int = 30
    db_max_mb: int = 4000


class Config(BaseSettings):
    """Top-level config object."""

    model_config = SettingsConfigDict(env_prefix="PCA_", env_nested_delimiter="__")

    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    a11y: A11yConfig = Field(default_factory=A11yConfig)
    ocr: OcrConfig = Field(default_factory=OcrConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    redact: RedactConfig = Field(default_factory=RedactConfig)
    filters: FilterConfig = Field(default_factory=FilterConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    outlook: OutlookConfig = Field(default_factory=OutlookConfig)
    gmail: GmailConfig = Field(default_factory=GmailConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)


def _read_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as f:
        return tomllib.load(f)


def load() -> Config:
    """Load config from disk, merge env overrides, write defaults if missing."""
    paths.ensure_dirs()
    cfg_path = paths.config_path()
    data = _read_toml(cfg_path)
    cfg = Config(**data) if data else Config()
    if not cfg_path.exists():
        cfg_path.write_text(_render_default_toml(), encoding="utf-8")
    return cfg


def _render_default_toml() -> str:
    return """# pc_assistant configuration. See pc_assistant/config.py for all fields.

[capture]
enabled = true
include_screenshot = true
all_monitors = false
heartbeat_seconds = 60
screenshot_max_width = 1920
screenshot_jpeg_quality = 80

[a11y]
enabled = true
ax_depth = 60
capture_clicks = true
capture_keystrokes = true
capture_clipboard = true

[ocr]
engine = "winrt"     # winrt | tesseract | off
languages = ["zh-CN", "en-US"]

[audio]
enabled = false
whisper_model = "small"
device = "cpu"
languages = ["zh"]
vad_min_segment_ms = 1000

[redact]
enabled = false

[filters]
ignored_apps  = ["1password", "bitwarden", "keepassxc", "lastpass", "lockapp", "logonui"]
ignore_incognito = true

[ollama]
# Used by Ask and apps/agent.py. OLLAMA_* env vars override these values.
base = "http://127.0.0.1:11434"
model = "qwen3_8b_ov:v1"
chat_timeout = 600

[server]
host = "127.0.0.1"
port = 3030

[outlook]
# Register a Microsoft Entra public client and add this redirect URI:
# http://127.0.0.1:3030/connections/outlook/oauth/callback
client_id = ""
tenant = "common"

[gmail]
# Register a Google OAuth client and add this redirect URI:
# http://127.0.0.1:3030/connections/gmail/oauth/callback
client_id = ""
client_secret = ""

[retention]
frame_days = 30
audio_days = 30
db_max_mb = 4000
"""
