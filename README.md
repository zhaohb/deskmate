# DeskMate

<div style="text-align:center;">
  <img src="./imgs/deskmate.png" alt="DeskMate" width="900" height="600">
</div>

DeskMate is a **local-first** desktop activity recorder for **Windows**. It captures
your screenshots, on-screen text (OCR / accessibility tree), keyboard/mouse/clipboard
events, and optional audio transcription into a SQLite database **on your own machine**,
then lets you use that data through a browser UI, a REST API, and an **MCP server**
that plugs straight into AI coding agents like **Claude Code** and **Cursor** —
so your assistant can search your activity, ask natural-language questions, and
run auto-generated reports (see [MCP server](#-mcp-server-use-deskmate-from-ai-agents)).

> **All your data stays on your own computer — nothing is uploaded to the cloud.**

---

## 🚀 Quick Start (5 minutes — just get it running)

If you just want to try it out, follow these 4 steps. **You don't need to understand
anything below this section.**

### Step 1 — Install Python (if you don't have it)
You need **Python 3.10 or newer**.
- If it's not installed, get it from <https://www.python.org/downloads/> and **check
  "Add Python to PATH"** during installation.
- Verify: open PowerShell and run `python --version` — you should see `Python 3.10.x`
  or higher.

### Step 2 — Get the code and create an isolated environment
```powershell
git clone <repo-url>
cd deskmate

# Create an isolated Python environment (keeps your system clean)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
> Every time you open a new terminal to use DeskMate, `cd` into the folder and run
> `.\.venv\Scripts\Activate.ps1` again to activate the environment. You'll know it
> worked when you see `(.venv)` at the start of your prompt.

### Step 3 — Install (recommended default combo)
```powershell
pip install -e ".[ocr-winrt,audio,vad,mcp]"
```
This installs: **core features + built-in Windows OCR + audio transcription + voice
activity detection + MCP**. That's enough for most people.
(What each `[...]` means and what else you can add → see [Install features as needed](#-install-features-as-needed-submodule-dependencies).)

### Step 4 — Launch
```powershell
deskmate ui
```
Your browser opens at **http://127.0.0.1:3030/ui** and DeskMate starts recording.

On first run it auto-creates a config and data folder under
`C:\Users\<your-username>\.deskmate\`. **You're up and running.**

> To stop: press `Ctrl + C` in the terminal where `deskmate ui` is running.

---

## 🧩 Install features as needed (submodule dependencies)

DeskMate splits the "heavy" features into **optional modules (extras)** — install only
what you need, not everything at once. The install format is always:
```powershell
pip install -e ".[module-name]"          # install one
pip install -e ".[module1,module2]"      # install several (comma-separated, NO spaces)
```

| Module | What it adds | When you need it / notes |
|--------|--------------|--------------------------|
| `ocr-winrt` | Built-in Windows OCR | **Recommended default**, no extra downloads |
| `ocr-rapidocr` | PP-OCR (via OpenVINO) | **Best for Chinese / small UI text**, models bundled |
| `ocr-tesseract` | Tesseract OCR | Requires [Tesseract](https://github.com/tesseract-ocr/tesseract) installed and on PATH |
| `audio` | Recording + Whisper transcription | Needs a mic or system audio (WASAPI). Default transcription backend |
| `audio-openvino` | Whisper on Intel NPU/GPU/CPU | Accelerated on Intel Core Ultra (NPU) or Arc GPU, see below |
| `vad` | Voice activity detection (Silero VAD) | Skips silent segments to save compute; install alongside `audio` |
| `speaker` | Speaker diarization | Tells "who said what" |
| `redact-onnx` | PII redaction | Optional privacy-protection model |
| `semantic` | Semantic (vector) search | Search by meaning, not just keywords |
| `pipes` | Scheduled tasks (YAML) | Run reports automatically on a schedule |
| `mcp` | MCP server | Integrate with AI agents / Claude, etc. |
| `training` | Local LoRA fine-tuning | Fine-tune a small model on your own data, see [training docs](docs/16-learning-training.md) |
| `full` | Most common extras at once | = tesseract + audio + vad + redact + semantic + pipes. **Excludes** OpenVINO & training |
| `dev` | Dev tools (pytest, ruff) | Only needed to modify code / run tests |

> Beginner tip: start with the default combo from Step 3. Add a module later when you
> decide you want a feature (e.g. Chinese OCR, semantic search) — re-running
> `pip install -e ".[...]"` is safe.
>
> Battery Saver / Power Manager is built in and uses Windows APIs through the
> standard library; it does **not** need an extra package.

### Install from a built wheel (instead of the source checkout)

If you have the built `.whl` (see [Building a wheel](#-building-a-wheel-for-distribution) below)
instead of the source tree, use the same `[extras]` syntax on the wheel file:

```powershell
:: core only
pip install deskmate-0.1.0-py3-none-any.whl

:: recommended combo (same as Step 3)
pip install "deskmate-0.1.0-py3-none-any.whl[ocr-winrt,audio,vad,mcp]"
```

**Install (almost) everything in one go** — every optional feature at once. Note
there is no single "all" extra, so list them explicitly. This pulls in the heavy
deps (PyTorch, OpenVINO, ONNX Runtime, Whisper, RapidOCR…), so expect a large download:

```powershell
pip install "deskmate-0.1.0-py3-none-any.whl[ocr-tesseract,ocr-rapidocr,ocr-winrt,audio,audio-openvino,vad,speaker,redact-onnx,semantic,mcp,pipes]"
```

> This deliberately omits `training` (LoRA fine-tuning — only needed if you train
> models; see [training docs](docs/16-learning-training.md)), `redact-onnx-dml` (the
> DirectML build of `redact-onnx` — pick one, not both), and `dev` (test tooling).
> The same works on the source checkout — just swap the wheel filename for `-e "."`,
> e.g. `pip install -e ".[ocr-winrt,audio,vad,speaker,semantic,mcp,pipes]"`.
>
> ⚠️ Most extras pull Windows-only / GPU-specific wheels and large model runtimes.
> Don't install everything unless you actually need it — the [recommended combo](#step-3--install-recommended-default-combo)
> is enough for most users.

### 📦 Building a wheel (for distribution)

To produce a distributable wheel (e.g. to install on another machine):

```powershell
pip install build
python -m build --wheel        :: output: dist\deskmate-0.1.0-py3-none-any.whl
```

The wheel bundles all of DeskMate's own code **and** the built-in apps
(`deskmate/apps/`), but **not** third-party dependencies — those are recorded as
requirements and fetched by `pip install` from PyPI. For an **offline** install,
pre-download everything into a folder first:

```powershell
pip download "deskmate-0.1.0-py3-none-any.whl[ocr-winrt,audio,vad,mcp]" -d wheelhouse
:: then on the offline machine:
pip install --no-index --find-links wheelhouse "deskmate[ocr-winrt,audio,vad,mcp]"
```

### 🧠 Installing the `training` extra (LoRA fine-tuning)

The `training` extra (`torch` + `unsloth` + `transformers` + `peft` + `accelerate`)
is **deliberately left out** of the "everything" command above, because the right
`torch` build depends on your accelerator. A pip `[extra]` can't pin a
hardware-specific wheel (the Intel XPU `torch` lives on PyTorch's own index, not
PyPI). Below, `WHL` = `deskmate-0.1.0-py3-none-any.whl`; on a source checkout use
`-e "."` instead.

**Intel Arc / Core-Ultra iGPU (XPU):** the XPU `torch` wheel **and** an
oneAPI/Level-Zero/MSVC toolchain are required — there are several non-obvious
gotchas, so this path is scripted. Install the extra, then run the setup script:
```powershell
pip install "WHL[training]"
scripts\setup-intel-xpu.bat          :: pulls torch 2.10.0+xpu + Level-Zero SDK headers
```
Full walkthrough (oneAPI, MSVC, the toolchain gotchas) → [training docs](docs/16-learning-training.md).

**CPU only** (no GPU — slow, for validating the pipeline on a tiny base like
`Qwen/Qwen3-0.6B`, not real ≥1B runs):
```powershell
pip install "WHL[training]"          :: the default torch is already CPU-only
```

> Verify XPU is live after setup:
> ```powershell
> python -c "import torch; print(torch.__version__, getattr(torch, 'xpu', None) is not None and torch.xpu.is_available())"
> :: expect:  2.10.0+xpu True
> ```

---

## 🤖 Want Q&A / auto-reports? Install Ollama first (optional but recommended)

DeskMate's **natural-language Q&A (Ask)** and the **LLM apps under My Apps** (daily
recap, todo extraction, meeting summaries…) need a local LLM server, **Ollama**. If you
don't use these, you can **skip this section** — recording and search work fine without it.

> **💡 Recommended: use the Ollama OpenVINO build on Intel hardware**
> If you're on an Intel CPU / Arc GPU / Core Ultra (NPU), we strongly recommend
> **Ollama-OV (OpenVINO backend)** — it runs models on Intel CPU/GPU/NPU, faster and
> more power-efficient:
> 👉 <https://github.com/zhaohb/ollama_openvino>
> . On regular GPUs / other platforms, the official
> Ollama works fine.

**Option A — Ollama OpenVINO build (recommended on Intel)**
1. Follow <https://github.com/zhaohb/ollama_openvino> to get `ollama.exe` and start the
   service (see that repo's README).
2. Import a model in **OpenVINO IR format** as described there (get it from HuggingFace /
   ModelScope → write a `Modelfile` with `ModelBackend "OpenVINO"` → `ollama create <name> -f Modelfile`).
3. Put the model name into `config.toml` (below).

**Option B — Official Ollama (other platforms / simplest)**
1. Install Ollama: <https://ollama.com/download> (it runs a local service in the
   background at `http://127.0.0.1:11434`).
2. Pull a model: `ollama pull qwen3`.

**Final step for both options** — in `C:\Users\<your-username>\.deskmate\config.toml`,
set the model name to the one you actually have:
```toml
[ollama]
base = "http://127.0.0.1:11434"
model = "qwen3_8b_ov:v1"   # OpenVINO build: the name you imported it as.
                           # Official build: the name you pulled (e.g. qwen3).
```
Then restart `deskmate ui` — you can now use Ask on the Home page and run reports under **My Apps**.

---

## ⚡ Advanced features (read when you need them)

These are more specialized capabilities - **beginners can ignore them for now** and
come back when needed:

| Feature | One-liner | Docs |
|---------|-----------|------|
| OpenVINO-accelerated transcription | Run Whisper faster on Intel NPU/GPU | [docs/04-audio.md](docs/04-audio.md#whisper-backends) |
| Live speech translation | Translate as you speak, shown in the UI | [docs/18-live-translation.md](docs/18-live-translation.md) |
| Video-call detection | Auto-detect Teams/Zoom/Meet + meeting notes | [docs/09-meeting-workflow.md](docs/09-meeting-workflow.md) |
| Gmail / Outlook integration | Search real mailboxes in Ask and apps | [docs/11-connections.md](docs/11-connections.md) |
| Battery Saver | On battery, move background AI work and user-selected apps onto efficient cores; the SPA view and live status text are localized in Chinese/English | [docs/22-power-manager.md](docs/22-power-manager.md) |
| Local LoRA fine-tuning | Train a small model on your data (incl. Intel iGPU) | [docs/16-learning-training.md](docs/16-learning-training.md) |
| All technical design docs | Architecture of every module | [docs/README.md](docs/README.md) |

**OpenVINO Whisper quick setup** (Intel devices):
```powershell
pip install -e ".[audio-openvino,vad]"
```
Then in `config.toml`:
```toml
[audio]
whisper_backend = "openvino_genai"   # default is "onnx_cpu"
openvino_device = "NPU"              # NPU | GPU | CPU | AUTO
```
The model is auto-downloaded from ModelScope on first use; if the chosen device fails to
load, it automatically falls back to CPU.

**LoRA training (Intel GPU)** has a one-click setup script — see
[scripts/setup-intel-xpu.bat](scripts/setup-intel-xpu.bat) and [docs/16](docs/16-learning-training.md).

---

## Requirements

- **Windows 10 / 11**
- **Python 3.10+**
- Optional, by feature:
  - **OCR**: `ocr-winrt` needs nothing extra; `ocr-tesseract` needs Tesseract on PATH;
    `ocr-rapidocr` just needs the extra (best for Chinese)
  - **Audio**: a microphone, or a system-audio loopback (WASAPI) device
  - **OpenVINO acceleration**: an Intel CPU; an Intel Core Ultra (NPU) or Arc/iGPU
    unlocks NPU/GPU inference
  - **Q&A & apps**: a local [Ollama](http://127.0.0.1:11434) server
  - **LoRA training**: a GPU is recommended (CPU works but is slow)

---

## Everyday usage

### Browser UI
```powershell
deskmate ui                  # record + API + auto-open /ui
deskmate ui --no-run-daemon  # view existing data only, don't start new recording
```

| Page | What it does |
|------|--------------|
| Home | Health status, recent activity, **Ask** (natural-language queries) |
| Apps | Run built-in and plugin LLM apps; configure schedules and time ranges |
| Email | Connect Gmail / Outlook and review mailbox integration status |
| Timeline | Browse the screenshot timeline |
| Transcripts | Audio transcriptions |
| Translate | Live speech translation controls and translated transcript stream |
| Events | Keyboard / mouse / clipboard / window-focus events |
| Todos | Action items extracted from email + meetings |
| Meetings | Detected video calls, transcripts, one-click summary |
| Capture | Capture-control view for recording sources and runtime toggles |
| Training | Local LoRA fine-tuning |
| Model Service | Start/stop local model backends and inspect service logs |
| Diagnostics | Environment self-checks for common setup issues |
| Reminders | Proactive habit reminders, acknowledgement, and history |
| Battery Saver | EcoQoS battery saver for background AI workers and selected apps; UI text is localized in Chinese/English |
| Settings | Config and monitors |

### CLI
```powershell
deskmate serve          # HTTP API only (records by default)
deskmate record         # recorder only
deskmate capture-once   # capture one frame manually
deskmate search "query" # keyword search via the API
deskmate mcp            # MCP server for AI agents (see "MCP server" below; API must be running)
```
Split API and recorder into two processes:
```powershell
deskmate record
deskmate serve --no-run-daemon
```

### LLM apps (My Apps)
Built-in apps live in `deskmate/apps/` (and ship inside the installed package);
your own apps go in `~/.deskmate/apps/plugins/`. Run them from the **My Apps**
page or the CLI. Full list, examples, and how to write your own in
[deskmate/apps/README.md](deskmate/apps/README.md).

| App | Purpose |
|-----|---------|
| `todo-list` | Unified checkbox todos from **email + meetings** |
| `meeting-summary` | Summarize the meeting that just ended |
| `email-digest` | Inbox overview |
| `email-compose` | Draft / reply via Gmail or Outlook |
| `day-recap` / `time-breakdown` / `ai-habits` | Daily recap / time split / AI-tool usage habits |
| `user-profile` / `habit-report` | Multi-day user profile / routine (default: last 7 days) |

> These apps need both **Ollama and the DeskMate API** running.

---

## Configuration

Config file: `C:\Users\<your-username>\.deskmate\config.toml` (auto-created on first run).

Main sections: `[capture]`, `[a11y]`, `[ocr]`, `[audio]`, `[ollama]`, `[redact]`,
`[filters]`, `[server]`

Override with environment variables (prefix `DESKMATE_`):
```powershell
$env:DESKMATE_SERVER__PORT = "4040"
$env:DESKMATE_AUDIO__ENABLED = "true"
```

## Where your data lives

```
C:\Users\<your-username>\.deskmate\
├── config.toml      # configuration
├── data.db          # SQLite database (with full-text search)
├── frames\          # screenshots
├── audio\           # audio chunks (when enabled)
├── videos\          # video chunks
├── checkpoints\     # LoRA training artifacts (when training)
├── apps\            # LLM app output reports; apps\plugins\ for your own apps
└── logs\            # logs
```
**To wipe and start over**: stop DeskMate, then delete `data.db` and the folders above.

## API

Default base URL: **http://127.0.0.1:3030**. Full endpoint list is in the running
server's `/docs` (OpenAPI). Common ones:
```
GET  /health        GET  /search?q=...      POST /ask
GET  /frames        POST /capture           GET  /meetings
GET  /todos         POST /todos             PATCH /todos/{id}
```

---

## 🔌 MCP server (use DeskMate from AI agents)

DeskMate ships an [MCP](https://modelcontextprotocol.io) stdio server so other AI
agents (Claude Desktop, IDE assistants, custom clients) can query your local
activity **and** drive DeskMate's Q&A and report apps. It's a thin proxy over the
local HTTP API, so **the DeskMate API must already be running** (`deskmate serve`
or `deskmate ui`).

**Install & launch**
```powershell
pip install -e ".[mcp]"     # or add `mcp` to your extras combo
deskmate serve              # in one terminal — the API on :3030
deskmate mcp                # in another — the MCP stdio server
```

**Tools exposed** (all prefixed `deskmate_` so they don't collide with other MCP servers):

| Tool | Kind | What it does |
|------|------|--------------|
| `deskmate_search` | read | Full-text search over frames (OCR + a11y text), UI events, audio transcripts |
| `deskmate_recent_frames` | read | List the latest captured frames |
| `deskmate_recent_events` | read | List latest UI events (clicks, focus, key text, clipboard) |
| `deskmate_health` | read | Daemon liveness + counters |
| `deskmate_capture_once` | action | Trigger a paired capture now |
| `deskmate_ask` | action | Natural-language Q&A: an LLM searches your local context and answers (slow) |
| `deskmate_list_apps` | read | List the report apps (day-recap, standup-update, todo-list, meeting-summary, email-compose, …) |
| `deskmate_run_app` | action | Run a report app and return its result (slow; params vary per app) |
| `deskmate_list_app_outputs` | read | List an app's past run outputs |
| `deskmate_get_app_output` | read | Fetch one output file (markdown/json) of a past run |

> `deskmate_ask` and `deskmate_run_app` invoke a local LLM, so they need **Ollama +
> the API** running and can take tens of seconds to minutes.

### 💡 Why connect it to Claude Code / Cursor?

Once DeskMate is registered, **your coding assistant knows what you've been doing
locally** — no copy-pasting context. While you work in Claude Code or Cursor you can ask:

**deskmate_ask** — natural-language Q&A over your activity
- "Use deskmate to recap the last hour: which apps I used, how long on each, and what I was doing — as a timeline."
- "Ask deskmate: between 2pm and 4pm today, what task was I mainly working on, and which files/windows were involved?"
- "From my local activity record (deskmate), what was I last reading about in the browser?"

**deskmate_search** — full-text search over OCR / UI / audio
- "Use deskmate to search my screen OCR, UI text and audio transcripts for 'recursive doubling', list the hits by time with the app/window they appeared in."
- "Use deskmate to find frames in the last 3 hours whose OCR text contains 'error' or 'exception', and tell me which app each was in."
- "Use deskmate to find which .py files I opened in VS Code today."

**deskmate_list_apps / run_app** — report apps
- "Use deskmate to list all available report apps and what each one does."
- "Use deskmate to run the time-breakdown report for the last hour."
- "Use deskmate to run day-recap and summarize what I did today."
- "Use deskmate to run standup-update: what I did, what's next, and any blockers."

**deskmate_recent_events / recent_frames** — raw activity stream
- "Use deskmate to show my last 50 UI events (clicks/focus/typing/clipboard) and summarize what I was just doing."
- "Use deskmate to list the last 20 captured frames and tell me which apps/windows I've been switching between."

**deskmate_health / capture_once** — status & instant capture
- "Use deskmate to check whether the recorder daemon is healthy and how many frames/transcripts it has captured."
- "Use deskmate to capture the screen right now and tell me what's on it."

**deskmate_list_app_outputs / get_app_output** — past reports
- "Use deskmate to list day-recap's past report runs and show me the contents of the latest one."

The assistant reads your **local activity record** on demand and answers in-context,
entirely on your machine.

### Register the server

Point `command` at the Python env that has the `mcp` extra installed.

**Claude Code** (CLI):
```bash
claude mcp add deskmate -e DESKMATE_API=http://127.0.0.1:3030 \
  -- C:/path/to/env/python.exe -m deskmate mcp
# verify:  claude mcp list      (should show deskmate ✓ connected)
```

**Cursor** — add to `~/.cursor/mcp.json` (global) or `<project>/.cursor/mcp.json`:
```json
{
  "mcpServers": {
    "deskmate": {
      "command": "C:/path/to/env/python.exe",
      "args": ["-m", "deskmate", "mcp"],
      "env": { "DESKMATE_API": "http://127.0.0.1:3030" }
    }
  }
}
```

**Claude Desktop** — same block in `claude_desktop_config.json`.

> After editing config, **fully restart** the client — MCP servers are only loaded
> at startup.

### Notes

- **`DESKMATE_API`** (optional) overrides the API base URL; defaults to
  `http://127.0.0.1:3030`.
- **Slow tools / timeouts:** `deskmate_ask` and `deskmate_run_app` invoke a local
  LLM and can take minutes. Raise the server-side budget with
  `DESKMATE_MCP_ASK_TIMEOUT` / `DESKMATE_MCP_RUN_APP_TIMEOUT` (seconds, in the
  server's `env`), and raise the **client** tool timeout with `MCP_TOOL_TIMEOUT`
  (milliseconds; e.g. `600000` for 10 min) — both must be long enough.
- **Behind a corporate proxy?** No action needed — the MCP server talks only to
  the local API and deliberately ignores `HTTP(S)_PROXY`, so `127.0.0.1` traffic
  is never routed through a proxy.
- The MCP server and the daemon are **independent processes** coupled only by the
  API, so the MCP server can run from a different environment than the daemon —
  it just needs the `mcp` extra.

---

## FAQ

- **Install error / a feature is missing?** Most likely the matching extra isn't
  installed. Go back to [Install features as needed](#-install-features-as-needed-submodule-dependencies)
  and add that one.
- **Ask / My Apps does nothing or errors?** Check that Ollama is running
  (`ollama list` should list models) and that `[ollama] model` in `config.toml` matches
  a model you actually have. On Intel hardware, use the
  [Ollama OpenVINO build](https://github.com/zhaohb/ollama_openvino) (the default model
  name `qwen3_8b_ov:v1` is meant for it).
- **`deskmate` not found in a new terminal?** You didn't activate the virtual
  environment. `cd` into the project folder, then run `.\.venv\Scripts\Activate.ps1`.
- **Want the internals / finer config of a feature?** It's all in [docs/](docs/README.md),
  numbered by module.

## License

**PolyForm Noncommercial License 1.0.0** — see [LICENSE](LICENSE).

DeskMate is free to use, copy, modify, and share **for any noncommercial
purpose**. Commercial use is **not** permitted under this license; for a
commercial license, contact the author.
