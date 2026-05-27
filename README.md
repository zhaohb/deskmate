# pc_assistant

Local-first PC activity recorder for **Windows**. It captures screenshots,
accessibility text, UI events, clipboard activity and optional audio
transcription into a local SQLite database, then exposes the data through a
REST API, browser UI and MCP server.

## Requirements

- Windows 10/11.
- Python 3.10+.
- 建议使用 PowerShell。
- 如果要使用 Tesseract OCR，需要先安装 Tesseract 并确保 `tesseract.exe`
  在 `PATH` 中。
- 如果要录音转写，需要可用的麦克风或 WASAPI loopback 设备。

## Install

```powershell
cd c:\hongbo\UX\pc_assistant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[ocr-tesseract,audio,mcp]"
```

可选安装项：

```powershell
# Windows 原生 OCR
pip install -e ".[ocr-winrt]"

# Silero VAD
pip install -e ".[vad]"

# 说话人识别（会下载较大的模型）
pip install -e ".[speaker]"

# ONNX PII 检测
pip install -e ".[redact-onnx]"

# Windows DirectML ONNX
pip install -e ".[redact-onnx-dml]"

# 全量常用能力
pip install -e ".[full,mcp]"
```

## Quick Start

```powershell
# 启动录制服务并自动打开浏览器 UI
pc-assistant ui
```

打开 <http://127.0.0.1:3030/ui>。

第一次运行会自动创建 `%USERPROFILE%\.pc_assistant\config.toml` 和本地数据目录。

## Commands

### 启动浏览器 UI

```powershell
pc-assistant ui
```

等价于启动 HTTP API、录制守护进程，并打开 `/ui`。默认地址：

```text
http://127.0.0.1:3030/ui
```

指定监听地址：

```powershell
pc-assistant ui --host 127.0.0.1 --port 3030
```

只看已有数据，不启动录制守护进程：

```powershell
pc-assistant ui --no-run-daemon
```

### 启动 HTTP API

```powershell
pc-assistant serve
```

默认会同时启动录制守护进程。API 地址：

```text
http://127.0.0.1:3030
```

常用 API：

```text
GET  /health
GET  /search?q=keyword&content_type=all
GET  /frames?limit=20
GET  /frames/{frame_id}
GET  /frames/{frame_id}/image
GET  /events/recent?limit=50
GET  /audio/list?limit=20
GET  /config
POST /capture
```

### 只启动录制守护进程

```powershell
pc-assistant record
```

该命令只负责采集，不启动 HTTP API。适合把 API 和采集拆成两个进程运行：

```powershell
pc-assistant record
pc-assistant serve --no-run-daemon
```

### 手动捕获一次

```powershell
pc-assistant capture-once
```

会立即截屏、读取当前窗口 UIA 树、执行 OCR，并写入数据库。

### 命令行搜索

先启动 API：

```powershell
pc-assistant serve
```

再搜索：

```powershell
pc-assistant search "kubernetes"
pc-assistant search "Cursor" --limit 20
pc-assistant search "meeting" --app-name Teams
```

### MCP Server

先启动 API：

```powershell
pc-assistant serve
```

再启动 MCP stdio server：

```powershell
pc-assistant mcp
```

如果 API 不是默认地址，可以用环境变量指定：

```powershell
$env:PC_ASSISTANT_API = "http://127.0.0.1:3030"
pc-assistant mcp
```

## Browser UI

`pc_assistant` ships a lightweight browser UI served directly by FastAPI, so
there is no Node build step:

```powershell
pc-assistant ui
```

Open <http://127.0.0.1:3030/ui>. The UI includes:

- **总览**: health, schema version, adaptive FPS parameters, latest frames and events.
- **搜索**: query OCR/UI/audio content through the `/search` API.
- **时间线**: inspect recent frames, image preview and frame metadata.
- **事件**: browse UI events from keyboard/mouse/clipboard/window focus.
- **配置**: current config and monitor list.

## Configuration

配置文件位置：

```text
%USERPROFILE%\.pc_assistant\config.toml
```

最常改的配置项：

```toml
[capture]
enabled = true
heartbeat_seconds = 60
include_screenshot = true
all_monitors = false
screenshot_max_width = 1920
screenshot_jpeg_quality = 80

[a11y]
enabled = true
ax_depth = 60
ax_max_nodes = 5000
capture_clicks = true
capture_keystrokes = true
capture_clipboard = true

[ocr]
engine = "winrt"      # winrt | tesseract | off
languages = ["en-US"]

[audio]
enabled = false
loopback = true
microphone = true
whisper_model = "base"
device = "cpu"
compute_type = "int8"
vad_threshold = 0.5
vad_min_segment_ms = 300
vad_padding_ms = 200
speaker_recognition = false

[redact]
enabled = false
rules = ["email", "credit_card", "phone_cn", "phone_us", "ipv4", "ssn"]
onnx_model_path = null
onnx_tokenizer_path = null

[filters]
ignored_apps = ["1password", "bitwarden", "keepassxc", "lastpass", "lockapp", "logonui"]
ignored_windows = []
included_windows = []
ignore_incognito = true

[server]
host = "127.0.0.1"
port = 3030
```

也可以用环境变量覆盖配置，格式为 `PCA_` + 嵌套字段名：

```powershell
$env:PCA_SERVER__PORT = "4040"
$env:PCA_AUDIO__ENABLED = "true"
pc-assistant ui
```

## Data Directory

Data lives under `%USERPROFILE%\.pc_assistant\`:

```
data.db          SQLite (frames, ocr_text, accessibility, ui_events, transcripts) + FTS5
frames\          JPEG snapshots per monitor
videos\          registered/imported MP4 video chunks
audio\           WAV chunks when audio capture is enabled
logs\            rolling logs
pipes\           optional local pipe files
config.toml      user config (created on first run)
```

清空本地数据时，先停止程序，再删除该目录或其中的 `data.db` / `frames\` /
`videos\` / `audio\`。

## Common Workflows

### 只记录屏幕和 UI 文本

```toml
[capture]
enabled = true
include_screenshot = true

[a11y]
enabled = true

[ocr]
engine = "winrt"

[audio]
enabled = false
```

启动：

```powershell
pc-assistant ui
```

### 启用音频转写

安装音频依赖：

```powershell
pip install -e ".[audio,vad]"
```

修改配置：

```toml
[audio]
enabled = true
loopback = true
microphone = true
whisper_model = "base"
device = "cpu"
compute_type = "int8"
vad_threshold = 0.5
```

启动：

```powershell
pc-assistant ui
```

### 启用文本脱敏

正则脱敏：

```toml
[redact]
enabled = true
rules = ["email", "credit_card", "phone_cn", "phone_us", "ipv4", "ssn"]
```

ONNX 二次识别（模型需自己提供）：

```toml
[redact]
enabled = true
onnx_model_path = "C:\\models\\pii-detector.onnx"
onnx_tokenizer_path = "C:\\models\\tokenizer.json"
```

### 忽略敏感应用或窗口

```toml
[filters]
ignored_apps = ["1password", "bitwarden", "keepassxc"]
ignored_windows = ["Private", "Confidential"]
ignore_incognito = true
```

### 创建一个本地 Pipe

在 `%USERPROFILE%\.pc_assistant\pipes\demo.md` 中写入：

```markdown
---
name: demo
interval_seconds: 300
runtime: none
permissions:
  read_db: true
---

This pipe is recorded on schedule.
```

当前版本会把调度和手动运行记录到 `pipe_executions`，并支持
`runtime: python`、`runtime: js` / `javascript` 和 `runtime: none`。运行时会创建
每次执行独立的 `output/{execution_id}` 目录，并通过环境变量传入
`PC_ASSISTANT_PIPE_CONTEXT`、`PC_ASSISTANT_OUTPUT_DIR`、`PC_ASSISTANT_API`；
如果 pipe 权限允许读取数据库，还会传入 `PC_ASSISTANT_DB`。

## What it records

- Event-driven **screenshots** per monitor (event = focus change / click / value change / heartbeat).
- **Accessibility tree** of the focused window via UI Automation (paired with the screenshot).
- **Window / app metadata**, click + keystroke aggregates, **clipboard** copies.
- Optional **audio** (microphone + WASAPI loopback) transcribed locally via `faster-whisper`.
- **Regex PII redaction** on indexed text (off by default).

## Feature Set

- **DB schema** stores `frames`, `ocr_text`, `frame_accessibility`,
  `ui_events`, `audio_chunks`, `audio_transcriptions`, `speakers`,
  `speaker_embeddings`, `meetings`, `meeting_transcript_segments`, `tags`,
  `frame_tags`, `memories`, `pipe_executions` and FTS5 indexes. Schema
  version is recorded in `_pca_migrations`.
- **OCR output** stores full text plus word-level JSON with normalized
  `left` / `top` / `width` / `height` coordinates and confidence fields.
- **Accessibility tree** includes `control_type`, `automation_id`,
  `class_name`, `bounds:
  {x,y,width,height}`, `is_enabled`, `is_focused`, `is_keyboard_focusable`,
  `help_text`, `is_password`, `is_selected`, `is_expanded`,
  `accelerator_key`, `access_key`, `localized_control_type`, `on_screen`,
  `depth`, `children`). UIA tree walker requests a UIAutomation
  `CacheRequest` when available so property reads are batched.
- **ActivityFeed** drives the heartbeat. The daemon adapts capture FPS
  (200ms → 2000ms) based on keyboard/mouse activity.
- **Audio** integrates **Silero VAD** + per-segment Whisper transcription
  (one `audio_transcriptions` row per VAD segment, with `start_time` /
  `end_time` / `speaker_id`). Optional **speaker identification** via
  pyannote → speechbrain → spectral fallback, centroid stored on the
  `speakers` row.
- **Redact** gains an **ONNX detector** (CPU + DirectML providers) and an
  **async reconciler** that backfills `redacted_text` / `redacted_text_json`
  / `redacted_transcription` columns. Model is bring-your-own; absent →
  regex path only.
- **Pixel-level frame redaction** for `/frames/{id}/image?redact_pii=true`
  derives OCR word boxes that overlap PII spans and covers them with solid
  black rectangles.
- **Video chunks** have a canonical storage root at
  `%USERPROFILE%\.pc_assistant\videos\YYYY-MM-DD\`. `/video-chunks/path`
  returns the next import/recording path, `/video-chunks/register` records
  metadata in SQLite, and frames can reference `video_chunk_id` + `offset_index`.
- **REST API** exposes `/health`, `/search` (returns
  `{data: ContentItem[], pagination}`), `/frames`, `/frames/{id}/image`,
  `/frames/{id}/text`, `/frames/{id}/context`, `/video-chunks`,
  `/audio/list`, `/speakers/search`, `/speakers/{id}/name`, `/meetings`,
  `/meetings/{id}/transcript`, `/tags/*`, `/memories`,
  `/pipes`, `/pipes/{name}/run`, `/pipes/{name}/executions`, `/raw_sql`,
  `/add`, `/monitors`, `/capture`, `/workflow/classify`, `/activity/params`,
  `/events/recent`, `/events/stream`. Endpoints we don't implement
  (`/transcribe`) return `501`.
- **Pipes** loader (`.md` files with YAML frontmatter) + scheduler that
  executes `python` / `js` / `none` runtimes with execution-scoped output
  directories, timeout handling and permission context.
- **Workflow classifier** heuristic with optional HTTP backend
  (`WORKFLOW_CLASSIFIER` env).

## Known Gaps

These are optional areas that require external assets or major additional
engineering work:

- **ONNX PII model** is **not bundled** — set
  `redact.onnx_model_path` to your own model.
- **Speaker diarization** uses pyannote/speechbrain only if the user
  installs the optional `[speaker]` extra (license click-through).
- **Pipe runtime** does not include screenpipe's Pi/LLM agent; local
  `python` / `js` pipes are supported, while remote LLM routing must be built
  by the pipe itself.
- **Continuous video recording** is not bundled yet. The video chunk API and
  storage path are ready for an encoder/importer, while the default recorder
  still writes event-driven JPEG snapshots.
- **macOS / Linux** capture paths are out of scope.
- **Cloud sync, vault, desktop wrapper, browser extension and business
  connectors** are not implemented.

## Status

Windows-only local activity recorder focused on capture, storage, search,
browser UI and MCP access.
