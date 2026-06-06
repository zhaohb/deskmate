"""Configuration for DeskMate.

Reads `~/.deskmate/config.toml` (auto-created with defaults on first run).
Environment variables prefixed with `DESKMATE_` override file values.

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
    text_input_debounce_seconds: float = 1.0
    capture_clicks: bool = True
    capture_keystrokes: bool = True
    capture_clipboard: bool = True
    capture_mouse_move: bool = False
    # P1: persist normalized accessibility nodes into the `elements` table.
    # Off by default (gradual rollout); when on, each "new content" frame writes
    # up to `elements_max_rows_per_frame` rows flattened from its UIA tree.
    persist_elements: bool = False
    elements_max_rows_per_frame: int = 300


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
    # Inference backend:
    #   onnx_cpu:       faster-whisper (CTranslate2 + ONNX Runtime). Default, most
    #                   compatible; uses `whisper_model` + `device`/`compute_type`.
    #   openvino_genai: OpenVINO GenAI WhisperPipeline. Runs on NPU/GPU/CPU; uses
    #                   `openvino_genai_model` + `openvino_device`.
    whisper_backend: Literal["onnx_cpu", "openvino_genai"] = "onnx_cpu"
    # device/compute_type apply to the onnx_cpu (faster-whisper) backend only.
    device: str = "cpu"
    compute_type: str = "int8"
    # --- openvino_genai backend settings ---
    # ModelScope model id (auto-downloaded on first run) or a local path to a
    # GenAI-format OpenVINO IR directory (encoder/decoder/tokenizer .xml+.bin).
    openvino_genai_model: str = "OpenVINO/whisper-medium-int8-ov"
    # OpenVINO device for openvino_genai: NPU | GPU | CPU | AUTO. NPU is fastest
    # here but its first compile is slow; results are cached so later starts are
    # fast. Falls back to CPU if the chosen device can't load.
    openvino_device: str = "NPU"
    # Directory for OpenVINO's compiled-model cache (CACHE_DIR). Empty =>
    # ~/.deskmate/ov_cache. Avoids recompiling the model on every start.
    openvino_cache_dir: str = ""
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


class SearchConfig(BaseModel):
    """Semantic / hybrid search settings.

    Semantic search is opt-in: it requires the optional ``fastembed`` extra and
    a one-time embedding-model download. When disabled (or unavailable) search
    transparently falls back to the FTS5 keyword index.
    """

    semantic_enabled: bool = False
    # ONNX embedding model resolved by fastembed. 384-dim, CPU-friendly.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Reciprocal Rank Fusion constant. Larger -> flatter rank weighting.
    rrf_k: int = 60
    # Max vectors scored per content type for a single semantic query.
    candidate_pool: int = 5000
    # Index new content from the daemon in the background.
    auto_index: bool = True
    # Rows embedded per indexing batch.
    index_batch: int = 64
    # Skip content shorter than this many characters.
    min_chars: int = 12


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


class HabitsConfig(BaseModel):
    """Settings for the additive habits module (mining + proactive suggestions)."""

    enabled: bool = True
    # How often the watcher evaluates "current activity vs. learned routine".
    tick_interval_min: int = 5
    # How often habit_profiles are re-mined from frames.
    mine_interval_hours: int = 24
    # Look-back window for mining routines.
    mine_lookback_days: int = 30
    # A (slot, category) must occur on at least this share of days to count as a habit.
    min_frequency: float = 0.5
    # Minimum distinct days backing a habit before it is trusted.
    min_sample_days: int = 3
    # Local "start-end" hours during which no notification is sent.
    quiet_hours: str = "22-8"
    # Hard cap on proactive notifications per day.
    daily_quota: int = 5
    # Whether to attempt a native Windows toast (falls back to UI inbox).
    toast_enabled: bool = True


class FusionConfig(BaseModel):
    """Additive context-fusion + capture-control module.

    The fusion bus subscribes to the in-process event bus and projects every
    signal into the unified ``context_events`` timeline. Per-source recording
    and global pause are controlled at runtime via the ``capture_control``
    table (see the capture-control API), not here; this config only governs
    whether the subsystem runs at all and which low-sensitivity window/focus
    events are worth persisting.
    """

    enabled: bool = True
    # Project window focus / title / value-change events into the timeline too.
    record_window_events: bool = True
    # Max characters of any text payload kept in the unified timeline.
    summary_max_chars: int = 200


class TrainingConfig(BaseModel):
    """Opt-in LoRA fine-tuning module (additive).

    Mines supervised (input, output) pairs from existing local data
    (``habit_suggestions`` the user marked useful, successful
    ``pipe_executions`` and the unified ``context_events`` timeline) and
    fine-tunes a small local causal LM with LoRA/QLoRA adapters. Heavy ML deps
    (torch/transformers/peft) live in the optional ``[training]`` extra; nothing
    here runs unless explicitly invoked via the CLI (``deskmate train-lora``)
    or the ``/training/lora`` API.
    """

    enabled: bool = True
    # Base model to adapt (HuggingFace id or local path).
    model_name: str = "Qwen/Qwen3-0.6B"
    # Adapter output directory. Empty => ~/.deskmate/checkpoints/lora.
    output_dir: str = ""

    # Data mining
    # Note: ``timeline`` (typed text / clipboard / transcript echoes) is
    # intentionally excluded by default — those pairs echo raw user input and
    # make poor SFT targets. It can still be opted into explicitly.
    sources: list[str] = Field(
        default_factory=lambda: ["habits", "pipes", "behavior", "ask"]
    )
    min_feedback: int = 1
    min_chars: int = 8
    limit_per_source: int = 2000
    max_pairs: int = 5000

    # LoRA / training hyperparameters
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])
    num_epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-5
    max_seq_length: int = 2048
    use_4bit: bool = False


class Config(BaseSettings):
    """Top-level config object."""

    model_config = SettingsConfigDict(env_prefix="DESKMATE_", env_nested_delimiter="__")

    capture: CaptureConfig = Field(default_factory=CaptureConfig)
    a11y: A11yConfig = Field(default_factory=A11yConfig)
    ocr: OcrConfig = Field(default_factory=OcrConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    redact: RedactConfig = Field(default_factory=RedactConfig)
    filters: FilterConfig = Field(default_factory=FilterConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    outlook: OutlookConfig = Field(default_factory=OutlookConfig)
    gmail: GmailConfig = Field(default_factory=GmailConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)
    habits: HabitsConfig = Field(default_factory=HabitsConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)

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
    return """# DeskMate configuration. See deskmate/config.py for all fields.

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
persist_elements = false

[ocr]
engine = "winrt"     # winrt | tesseract | off
languages = ["zh-CN", "en-US"]

[audio]
enabled = false
whisper_model = "small"            # onnx_cpu backend model tier
whisper_backend = "onnx_cpu"       # onnx_cpu (faster-whisper) | openvino_genai (OpenVINO NPU/GPU/CPU)
device = "cpu"                     # faster-whisper device (onnx_cpu backend only)
# openvino_genai backend: ModelScope id (auto-downloaded) or local IR dir
openvino_genai_model = "OpenVINO/whisper-medium-int8-ov"
openvino_device = "NPU"            # NPU | GPU | CPU | AUTO (openvino_genai backend only)
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
