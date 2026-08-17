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
    # rapidocr: PP-OCR mobile via OpenVINO CPU — best for Chinese + small text.
    # winrt: Windows.Media.Ocr. tesseract: pytesseract. off: disabled.
    engine: Literal["off", "winrt", "tesseract", "rapidocr"] = "winrt"
    # WinRT: zh-CN → zh-Hans; include en-US for mixed UI (bookmarks, URLs).
    # (rapidocr ignores this — its PP-OCR model is already zh+en.)
    languages: list[str] = ["zh-CN", "en-US"]
    tesseract_cmd: str | None = None


class AudioConfig(BaseModel):
    enabled: bool = False
    sample_rate: int = 16000
    chunk_seconds: int = 30
    # Chunking strategy for how raw audio is sliced into chunks for transcription:
    #   fixed:    accumulate exactly `chunk_seconds` then emit (legacy, high latency
    #             but cheap — one transcription per 30s).
    #   endpoint: emit as soon as a speech utterance ends (VAD detects a pause), so
    #             a sentence is transcribed/translated ~1-4s after it is spoken.
    #             This is the low-latency path for live translation.
    chunk_mode: Literal["fixed", "endpoint"] = "fixed"
    # endpoint mode: a silence gap (ms) at the tail of the rolling buffer marks the
    # end of an utterance and triggers an emit. Smaller => lower latency, choppier.
    endpoint_silence_ms: int = 700
    # endpoint mode: force-emit when the buffer reaches this many seconds even if the
    # speaker hasn't paused, so a long monologue still streams out.
    endpoint_max_chunk_s: float = 8.0
    # endpoint mode: don't emit utterances shorter than this; carry them into the
    # next chunk so "嗯"/"对" fragments aren't transcribed/translated on their own.
    endpoint_min_chunk_s: float = 1.0
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
    # --- live translation (opt-in) ---
    # When enabled, each transcript segment is translated by the local Ollama LLM
    # into `translate_target_lang` and pushed to the UI in real time. Off by
    # default; turning it on adds one LLM call per spoken utterance.
    translate_enabled: bool = False
    # Target language for translation (ISO 639-1, e.g. "zh", "en", "ja").
    translate_target_lang: str = "zh"
    # Skip translation when the transcript's detected language already equals the
    # target language (no point translating zh→zh).
    translate_skip_if_same: bool = True
    # Latency/quality trade-off for live translation. Maps onto the endpoint
    # silence threshold above when chunk_mode="endpoint":
    #   fast:     ~400ms pause  — lowest latency, choppier segments
    #   balanced: ~700ms pause  — a natural clause (default)
    #   quality:  ~1000ms pause — fuller sentences, best translation
    translate_latency_mode: Literal["fast", "balanced", "quality"] = "balanced"
    # How many preceding utterances to feed the translator as context (improves
    # pronoun/terminology consistency without re-translating them). 0 disables.
    translate_context_window: int = 2
    # Ollama model for translation. Empty => use the global [ollama] model.
    translate_model: str = ""


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
    # Ask the model to run its thinking/reasoning pass before answering and
    # before choosing tools. Improves quality; the reasoning is returned in a
    # separate field so it doesn't leak into the answer. Costs extra latency and
    # tokens — turn off on slow hardware.
    think: bool = True


class ModelServiceConfig(BaseModel):
    """How DeskMate obtains and launches the local Ollama *service*.

    Distinct from :class:`OllamaConfig`, which is the *connection* (where Ask /
    apps talk). This section governs provisioning/lifecycle: which backend
    binary to use, where a user-supplied OpenVINO ``ollama.exe`` lives, the
    custom model pull source (``OLLAMA_REGISTRY``), and whether to start the
    service automatically. The service is launched on the host:port parsed from
    ``[ollama] base`` so both views agree on one endpoint.
    """

    # "official"  -> auto-download the official Ollama build (GitHub release).
    # "openvino"  -> use a user-supplied OpenVINO ollama.exe (zhaohb/ollama_openvino).
    backend: Literal["official", "openvino"] = "official"
    # OpenVINO build: absolute path to the prebuilt ollama.exe the user obtained.
    ollama_exe_path: str = ""
    # Optional direct download URL for the OpenVINO ollama.exe (informational).
    ollama_exe_url: str = ""
    # Custom model pull source -> injected as OLLAMA_REGISTRY when launching.
    registry: str = ""
    # Allow pulling from an HTTP (non-TLS) registry. Self-hosted registries are
    # commonly plain HTTP, so this defaults to True; Ollama otherwise assumes
    # HTTPS for /api/pull and silently fails against an http:// registry.
    pull_insecure: bool = True
    # Selected GenAI runtime DLL dir put on PATH at launch (empty => newest found
    # under the download dir's runtime/). Set when the user picks a version.
    genai_runtime_dir: str = ""
    # Download URL for the OpenVINO GenAI runtime zip. Empty => the built-in
    # default (GENAI_RUNTIME_URL). The user can point at another version.
    genai_url: str = ""
    # Where OpenVINO downloads (ollama.exe + GenAI runtime) land. Empty =>
    # ~/.deskmate/bin/ollama-openvino. The user can point this anywhere.
    download_dir: str = ""
    # Start the service automatically when the daemon boots (opt-in).
    auto_start: bool = False
    # Stop a DeskMate-launched Ollama service when DeskMate exits. Default on, so
    # the service's lifetime is tied to DeskMate. Turn off to keep Ollama running
    # in the background after DeskMate closes. Only affects a service DeskMate
    # started (a PID file we own); an external Ollama is never touched.
    stop_on_exit: bool = True


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
    # Runaway BACKSTOP, not a daily throttle. Day-to-day pacing is owned by each
    # rule's own cooldown_min (semantic, per-rule) — the philosophy is "if a rule
    # is due and not in cooldown, let it fire". This shared cap exists only to
    # contain a misbehaving rule (e.g. one that hits every single tick) from
    # spamming the user. Set high enough that a normal day never reaches it:
    # hourly break nudges + a handful of others stay well under 30.
    daily_quota: int = 30
    # Whether to attempt a native Windows toast (falls back to UI inbox).
    toast_enabled: bool = True
    # Hold proactive nudges while the user is presumably busy — in a meeting, an
    # app is full-screen (presentation / video), or Windows Focus Assist is on.
    # Fail-open: if presence can't be determined, reminders still fire.
    respect_presence: bool = True
    # Language for reminder text: "zh" or "en". Rules carry both; this picks one.
    reminder_lang: str = "zh"


class PowerConfig(BaseModel):
    """Battery-saver: eco-throttle background workers onto E/LPE-cores on battery.

    Zero-invasive — a PowerManager thread tags workers by name via thread-level
    EcoQoS; existing worker code is untouched. No-op on AC / non-Windows.
    """

    enabled: bool = True
    # How often to re-check AC/battery state and re-tag workers (seconds).
    poll_seconds: float = 15.0


class LearningConfig(BaseModel):
    """Frame-level learning session detector (MeetingDetector-style FSM)."""

    enabled: bool = True
    # Seconds without a learning signal before the open session is closed.
    end_grace_seconds: float = 180.0
    # Open a new session only at/above this confidence.
    start_confidence: float = 0.75
    # Keep an open session alive at/above this (lower than start).
    keep_confidence: float = 0.60
    # On grace-close, queue a user-learning recap in the background.
    auto_recap_on_end: bool = True
    auto_recap_hours: float = 8.0
    # While a meeting is active, do not open/keep learning sessions.
    pause_during_meeting: bool = True
    # Use recent audio transcripts (loopback/mic) as lecture cues — enables
    # "video speech sounds like a class" detection. Requires [audio] enabled.
    use_audio_cues: bool = True
    audio_lookback_seconds: float = 90.0

    # User-curated "this is always studying" list — the escape hatch for the
    # keyword heuristics.
    #
    # Video sites (bilibili / YouTube) are only *candidates*: they need a
    # lecture-like title or on-screen text to open a session, because otherwise
    # every entertainment video would become a study log. That gate is tuned for
    # school-flavoured wording (课程/讲义/tutorial/lecture), so a genuine
    # technical talk — a conference session, a release walkthrough, a vendor
    # channel — scores 0 and is silently rejected.
    #
    # No keyword list fixes that in general, so the user names the sources they
    # trust instead. Each entry is matched case-insensitively against the URL,
    # the URL host, the window title, and captured on-screen text; a hit means
    # "learning", skipping the lecture-score gate entirely. Examples:
    #   "docs.openvino.ai"          → a whole documentation domain
    #   "space.bilibili.com/123456" → one creator's space URL
    #   "OpenVINO中文社区"           → a channel name as it appears on screen
    always_learning: list[str] = []


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
    # Base model to adapt — a HuggingFace id or local HF-format path (NOT an
    # Ollama/OpenVINO model; LoRA training loads via transformers.from_pretrained).
    model_name: str = "Qwen/Qwen3-0.6B"
    # Adapter output directory. Empty => ~/.deskmate/checkpoints/lora.
    output_dir: str = ""

    # Data mining
    # Note: ``timeline`` (typed text / clipboard / transcript echoes) is
    # intentionally excluded by default — those pairs echo raw user input and
    # make poor SFT targets. It can still be opted into explicitly.
    sources: list[str] = Field(
        default_factory=lambda: ["habits", "apps", "pipes", "behavior", "ask", "profile"]
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
    model_service: ModelServiceConfig = Field(default_factory=ModelServiceConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    outlook: OutlookConfig = Field(default_factory=OutlookConfig)
    gmail: GmailConfig = Field(default_factory=GmailConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)
    habits: HabitsConfig = Field(default_factory=HabitsConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    power: PowerConfig = Field(default_factory=PowerConfig)

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


def set_audio_languages(languages: list[str]) -> None:
    """Persist ``[audio] languages`` to config.toml, preserving everything else.

    We avoid a full TOML round-trip (which would drop the file's comments) by
    rewriting just the ``languages = [...]`` line inside the ``[audio]`` table,
    or inserting it if absent. Best-effort: a malformed/missing file is created
    from defaults first.
    """
    import re  # noqa: PLC0415

    cfg_path = paths.config_path()
    if not cfg_path.exists():
        cfg_path.write_text(_render_default_toml(), encoding="utf-8")

    text = cfg_path.read_text(encoding="utf-8")
    # TOML array of the (already-validated) language strings.
    rendered = "[" + ", ".join(f'"{l}"' for l in languages) + "]"
    new_line = f"languages = {rendered}"

    lines = text.splitlines()
    in_audio = False
    audio_start = -1
    replaced = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_audio = stripped == "[audio]"
            if in_audio:
                audio_start = i
            continue
        if in_audio and re.match(r"^\s*languages\s*=", line):
            # Preserve any trailing comment after the value.
            comment = ""
            if "#" in line:
                comment = "  " + line[line.index("#"):].rstrip()
            lines[i] = new_line + comment
            replaced = True
            break

    if not replaced:
        if audio_start >= 0:
            lines.insert(audio_start + 1, new_line)
        else:
            lines.append("[audio]")
            lines.append(new_line)

    cfg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_toml_value(value: object) -> str:
    """Render a Python scalar/list as a TOML literal (bool/str/number/list).

    Strings become TOML *basic strings* with the characters TOML requires
    escaped — critically the backslash, so Windows paths like
    ``C:\\Users\\…\\ollama.exe`` round-trip instead of being misread as escape
    sequences (``\\U`` → "Invalid hex value") that corrupt the file.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return _toml_basic_string(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_render_toml_value(v) for v in value) + "]"
    return str(value)


def _toml_basic_string(value: str) -> str:
    """Quote ``value`` as a TOML basic string, escaping per the TOML spec."""
    escaped = (
        value.replace("\\", "\\\\")  # backslash first, so we don't double-escape
        .replace('"', '\\"')
        .replace("\b", "\\b")
        .replace("\t", "\\t")
        .replace("\n", "\\n")
        .replace("\f", "\\f")
        .replace("\r", "\\r")
    )
    return f'"{escaped}"'


def set_config_value(section: str, key: str, value: object) -> None:
    """Persist a single ``[section] <key> = <value>`` to config.toml in place.

    Rewrites just that one line inside the named table — or inserts it (creating
    the table if absent) — so every comment and unrelated setting in the file is
    preserved. Handles bool/str/number and lists. Best-effort: a missing file is
    created from defaults first. This is the generic writer behind the Settings
    UI; :func:`set_audio_value` is a thin ``[audio]`` wrapper kept for callers.
    """
    import re  # noqa: PLC0415

    cfg_path = paths.config_path()
    if not cfg_path.exists():
        cfg_path.write_text(_render_default_toml(), encoding="utf-8")

    new_line = f"{key} = {_render_toml_value(value)}"
    lines = cfg_path.read_text(encoding="utf-8").splitlines()
    header = f"[{section}]"
    in_section = False
    section_start = -1
    replaced = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_section = stripped == header
            if in_section:
                section_start = i
            continue
        if in_section and re.match(rf"^\s*{re.escape(key)}\s*=", line):
            comment = ""
            if "#" in line:
                comment = "  " + line[line.index("#"):].rstrip()
            lines[i] = new_line + comment
            replaced = True
            break

    if not replaced:
        if section_start >= 0:
            lines.insert(section_start + 1, new_line)
        else:
            lines.append(header)
            lines.append(new_line)

    cfg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def set_audio_value(key: str, value: object) -> None:
    """Persist a single ``[audio] <key> = <value>`` (see :func:`set_config_value`)."""
    set_config_value("audio", key, value)


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
engine = "winrt"     # rapidocr | winrt | tesseract | off
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
# --- live translation (opt-in) ---
chunk_mode = "fixed"               # fixed (30s, legacy) | endpoint (low-latency, per-utterance)
translate_enabled = false          # translate each utterance via local Ollama
translate_target_lang = "zh"       # ISO 639-1 target language
translate_latency_mode = "balanced"  # fast | balanced | quality

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

[model_service]
# How DeskMate downloads & launches the local Ollama service (Model Service page).
backend = "official"          # official (auto-download) | openvino (user-supplied exe)
ollama_exe_path = ""          # openvino: path to ollama.exe (downloaded or your own)
registry = ""                 # custom model source -> OLLAMA_REGISTRY at launch
genai_runtime_dir = ""        # openvino: selected GenAI runtime dir (blank = newest)
genai_url = ""                # openvino: GenAI runtime zip URL (blank = built-in default)
download_dir = ""             # openvino: where exe + runtime download (blank = default)
auto_start = false            # start the service automatically when the daemon boots

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

[learning]
# Frame-level learning session detector (like meeting detection).
enabled = true
end_grace_seconds = 180.0
start_confidence = 0.75
keep_confidence = 0.60
auto_recap_on_end = true
auto_recap_hours = 8.0
pause_during_meeting = true
use_audio_cues = true
audio_lookback_seconds = 90.0

[power]
# Battery saver: on battery, push background workers (semantic index, redaction,
# screen capture/OCR, retention) onto efficient cores via thread-level EcoQoS.
# Zero-invasive and no-op on AC / non-Windows. Ask keeps performance cores.
enabled = true
poll_seconds = 15.0
"""
