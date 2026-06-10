# DeskMate

<div style="text-align:center;">
  <img src="./imgs/deskmate.png" alt="DeskMate" width="900" height="600">
</div>

DeskMate is a **local-first** desktop activity recorder for **Windows**. It captures
your screenshots, on-screen text (OCR / accessibility tree), keyboard/mouse/clipboard
events, and optional audio transcription into a SQLite database **on your own machine**,
then lets you use that data through a browser UI, a REST API, and an MCP server
(search, natural-language Q&A, auto-generated reports, and more).

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
| `notify` | Windows desktop notifications | Used by reminder features |
| `pipes` | Scheduled tasks (YAML) | Run reports automatically on a schedule |
| `mcp` | MCP server | Integrate with AI agents / Claude, etc. |
| `training` | Local LoRA fine-tuning | Fine-tune a small model on your own data, see [training docs](docs/16-learning-training.md) |
| `full` | Most common extras at once | = tesseract + audio + vad + redact + semantic + notify + pipes. **Excludes** OpenVINO & training |
| `dev` | Dev tools (pytest, ruff) | Only needed to modify code / run tests |

> Beginner tip: start with the default combo from Step 3. Add a module later when you
> decide you want a feature (e.g. Chinese OCR, semantic search) — re-running
> `pip install -e ".[...]"` is safe.

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
> The model name in DeskMate's default config, `qwen3_8b_ov:v1` (note the `_ov` suffix),
> is meant for this OpenVINO build. On regular GPUs / other platforms, the official
> Ollama works fine.

**Option A — Ollama OpenVINO build (recommended on Intel)**
1. Follow <https://github.com/zhaohb/ollama_openvino> to get `ollama.exe` and start the
   service (note it needs env settings like `GODEBUG=cgocheck=0`; see that repo's README).
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

These are more specialized capabilities — **beginners can ignore them for now** and
come back when needed:

| Feature | One-liner | Docs |
|---------|-----------|------|
| OpenVINO-accelerated transcription | Run Whisper faster on Intel NPU/GPU | [docs/04-audio.md](docs/04-audio.md#whisper-backends) |
| Live speech translation | Translate as you speak, shown in the UI | [docs/18-live-translation.md](docs/18-live-translation.md) |
| Video-call detection | Auto-detect Teams/Zoom/Meet + meeting notes | [docs/09-meeting-workflow.md](docs/09-meeting-workflow.md) |
| Gmail / Outlook integration | Search real mailboxes in Ask and apps | [docs/11-connections.md](docs/11-connections.md) |
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
| Timeline | Browse the screenshot timeline |
| Events | Keyboard / mouse / clipboard / window-focus events |
| Transcripts | Audio transcriptions |
| Todos | Action items extracted from email + meetings |
| Meetings | Detected video calls, transcripts, one-click summary |
| Training | Local LoRA fine-tuning |
| My Apps | Run LLM apps; connect Gmail/Outlook in Settings |
| Settings | Config and monitors |

### CLI
```powershell
deskmate serve          # HTTP API only (records by default)
deskmate record         # recorder only
deskmate capture-once   # capture one frame manually
deskmate search "query" # keyword search via the API
deskmate mcp            # MCP server (API must be running)
```
Split API and recorder into two processes:
```powershell
deskmate record
deskmate serve --no-run-daemon
```

### LLM apps (My Apps)
Apps live in `apps/`. Run them from the **My Apps** page or the CLI. Full list and
examples in [apps/README.md](apps/README.md).

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
├── apps\            # output reports from LLM apps
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

MIT
