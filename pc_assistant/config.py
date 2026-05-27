"""Configuration for pc_assistant.

Reads `~/.pc_assistant/config.toml` (auto-created with defaults on first run).
Environment variables prefixed with `PCA_` override file values.
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
    heartbeat_seconds: int = 60
    min_capture_gap_seconds: float = 1.0
    debounce_seconds: float = 1.5
    include_screenshot: bool = True
    screenshot_max_width: int = 1920
    screenshot_jpeg_quality: int = 80
    all_monitors: bool = False
    pause_extraction_on_input_ms: int = 800
    # When true, the adaptive FPS loop will never go *below* `heartbeat_seconds`.
    # When false, ActivityFeed fully controls the capture rate.
    adaptive_fps_floor: bool = True


class A11yConfig(BaseModel):
    enabled: bool = True
    ax_depth: int = 60
    ax_max_nodes: int = 5000
    text_input_debounce_seconds: float = 5.0
    capture_clicks: bool = True
    capture_keystrokes: bool = True
    capture_clipboard: bool = True


class OcrConfig(BaseModel):
    engine: Literal["off", "winrt", "tesseract"] = "winrt"
    languages: list[str] = ["en-US"]
    tesseract_cmd: str | None = None


class AudioConfig(BaseModel):
    enabled: bool = False
    sample_rate: int = 16000
    chunk_seconds: int = 30
    loopback: bool = True
    microphone: bool = True
    whisper_model: str = "base"
    device: str = "cpu"
    compute_type: str = "int8"
    vad_threshold: float = 0.5
    vad_min_segment_ms: int = 300
    vad_padding_ms: int = 200
    speaker_recognition: bool = False


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


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 3030


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
    server: ServerConfig = Field(default_factory=ServerConfig)
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
languages = ["en-US"]

[audio]
enabled = false
whisper_model = "base"
device = "cpu"

[redact]
enabled = false

[filters]
ignored_apps  = ["1password", "bitwarden", "keepassxc", "lastpass", "lockapp", "logonui"]
ignore_incognito = true

[server]
host = "127.0.0.1"
port = 3030

[retention]
frame_days = 30
audio_days = 30
db_max_mb = 4000
"""
