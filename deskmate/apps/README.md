# DeskMate apps

Local LLM apps driven by `pipe.md` prompts. Each app follows this execution model:

1. Read `pipe.md` (YAML frontmatter + prompt body)
2. Prepend context header (time range, timezone, API base)
3. Send to LLM (Ollama) with `SKILL.md` as system knowledge
4. **day-recap** / **ai-habits**: model calls `activity_summary` / `search` in a loop (tool-driven)
5. Other pipes may pre-fetch API data when `pipe.md` instructs it
6. LLM generates the report in the exact format specified by `pipe.md`

Output is written to: `%USERPROFILE%\.deskmate\apps\<app-name>\output\<timestamp>\`

## Prerequisites

- **Ollama** running at `http://127.0.0.1:11434` with a model loaded
- **DeskMate API** running at `http://127.0.0.1:3030`

## Apps

### video-export

Exports recent screenshots as a video clip. The LLM calls `POST /frames/export`
as instructed by `pipe.md`.

```cmd
cd /d C:\hongbo\UX\deskmate
conda activate support_qwen3.5
python deskmate\apps\video-export\app.py --minutes 5 --verbose
```

### day-recap

LLM queries screen and audio data, then generates a structured day recap with
Summary, Accomplishments, Key Moments, Unfinished Work, Patterns, and Next step.

```cmd
python deskmate\apps\day-recap\app.py --hours 16 --verbose
```

### ai-habits

LLM searches for AI tool usage (ChatGPT, Claude, Copilot, Cursor, Gemini,
Perplexity) and generates an analysis report. The agent runs one search per
tool in Python and only the tools with verified hits are passed to the model
(it cannot fabricate usage for tools that were never opened).

```cmd
python deskmate\apps\ai-habits\app.py --hours 24 --verbose
```

### meeting-summary

Finds the meeting that just ended, summarizes its transcript (key topics,
decisions, action items), and patches the summary back onto the meeting record
(appends under a `## Summary` heading in the note, and refreshes a generic
title). Skips the write-back when there is nothing worth saving. Pass
`--meeting-id <id>` to summarize a specific meeting instead of the latest — this
is what the **Meetings** page "Summarize" button uses.

```cmd
python deskmate\apps\meeting-summary\app.py --verbose
python deskmate\apps\meeting-summary\app.py --meeting-id 42 --verbose
```

### standup-update

Generates a short standup update (Yesterday / Today / Blockers) from the supplied
time window. Uses the same rich prefetch as day-recap (timeline, key texts,
edited files, top-app searches) plus any meetings in range, then a single LLM
pass — copy-paste ready; aim for concrete bullets with timestamps, not generic
phrases.

```cmd
python deskmate\apps\standup-update\app.py --hours 24 --verbose
```

### time-breakdown

Breaks down the supplied time range by application, category, and project, with
a productivity score. Python pre-computes per-app and per-category minutes from
`activity_summary` (so apps like Cursor are not dropped to 0 min), then the LLM
writes the four sections from that evidence plus timeline/edited files.

```cmd
python deskmate\apps\time-breakdown\app.py --hours 12 --verbose
```

### ai-prompt-journal

Captures every prompt the user typed into AI tools (ChatGPT, Claude, Gemini,
Perplexity, Grok, DeepSeek, Copilot, Cursor, local models, etc.) over the
supplied window (default 1 hour) and appends only the genuinely new prompts to
a daily markdown journal at
`%USERPROFILE%\.deskmate\apps\ai-prompt-journal\journal\YYYY-MM-DD.md`.
Run manually from the My Apps UI or CLI, or set a custom schedule in **My Apps → Schedule** (interval or daily time). User schedules are stored in ``~/.deskmate/apps/schedules.json``.

```cmd
python deskmate\apps\ai-prompt-journal\app.py --hours 1 --verbose
```

### email-digest

Summarizes inbox activity from two sources: Gmail OAuth / Gmail API and Outlook
OAuth / Microsoft Graph when accounts are connected, plus local screen / UI
recordings for Outlook, Thunderbird, Windows Mail, Mailspring, Mailbird, eM
Client and webmail tabs (Gmail, Outlook web, Outlook 365, QQ Mail, 163 Mail,
Yahoo Mail). Lists tools used, top senders / threads, drafts in progress,
action items and patterns.

```cmd
python deskmate\apps\email-digest\app.py --hours 24 --verbose
```

### todo-list (Todo List Assistant)

Builds ONE unified todolist from two evidence sources: **email** (the same
prefetch as `email-digest` — Gmail / Outlook OAuth messages + per-tool OCR / UI
search across 9 mail client / webmail types) AND **meetings** (detected video
calls in the window + their transcripts, so spoken action items become todos).
Output is GitHub-style checkboxes
`- [ ] <task> — from <sender / meeting> — due <date> — source: <email:<tool> | meeting:<name>> — priority: <H/M/L>`,
grouped by source, ending with a `Suggested Next Action`. Deduplicates a task
that shows up in both an email and a meeting. Does **not** require a search
keyword — scans the whole time window automatically.

Besides the markdown report, the app parses each checkbox into a structured
row and persists it to the `todos` table via `POST /todos` (deduplicated by a
stable key, so re-runs update rather than duplicate). These show up on the
**Todos** page in the UI, where you can check items off or delete them. Pass
`--no-store` to skip database persistence and only write the markdown.

```cmd
python deskmate\apps\todo-list\app.py --hours 24 --verbose
```

### email-compose

LLM drafts an email (or a reply) for a connected Gmail or Outlook account.
Optionally fetches a source message via `--reply-to <id>` to ground the draft.
The script writes Subject + Body + up to 2 alternatives + a Send Preview to
disk for review; pass `--send` to actually deliver via the provider's API.

```cmd
:: New draft (review only)
python deskmate\apps\email-compose\app.py --provider gmail --to a@b.com --intent "需要本周确认预算" --verbose

:: Reply, then actually send
python deskmate\apps\email-compose\app.py --provider outlook --to a@b.com --intent "我同意方案 A" --reply-to AAMkAGI... --send --verbose
```

## When to use which app

| Your moment | App | Why |
|---|---|---|
| Morning standup in Slack/Teams | `standup-update` | Yesterday / Today / Blockers, ≤150 words |
| Clean my inbox / plan my day | `todo-list` | One checkbox list from email + meetings, no keyword needed |
| Write a new business email | `email-compose` | LLM draft + 2 alternatives, then `--send` |
| Reply to a specific message | `email-compose --reply-to <id>` | Grounds the draft in the source message |
| A meeting just ended | `meeting-summary` (auto on `meeting_ended`) | Patches summary back to the meeting record |
| Capture what I asked AI in a time window | `ai-prompt-journal` (manual) | Appends new prompts to today's journal |
| End-of-day reflection | `day-recap` | Accomplishments + key moments + unfinished work |
| Audit my email time today | `email-digest` | Tool minutes + senders + drafts + todos + patterns |
| Weekly time breakdown | `time-breakdown` | App / category / project + productivity score |
| Weekly AI usage habits | `ai-habits` | Per-tool time + effectiveness, no fabricated tools |
| Record the last few minutes | `video-export` | Screenshots → mp4 |

## Write your own app (plugin)

DeskMate discovers apps from **two** locations, so you can add your own without
touching the install tree:

1. **Built-in apps** — this folder (`<install>/apps/`), shipped with DeskMate.
2. **User plugins** — `~/.deskmate/apps/plugins/` (`%USERPROFILE%\.deskmate\apps\plugins\`
   on Windows). This survives upgrades and needs no source checkout. A plugin
   whose folder name matches a built-in **shadows** the built-in.

An app is just a folder with two files:

```
~/.deskmate/apps/plugins/my-report/
├── pipe.md     # YAML frontmatter + the prompt/report instructions
└── app.py      # CLI entry point (python app.py [--hours N | --start ... --end ...])
```

**`pipe.md`** — frontmatter drives the My Apps UI and scheduling:

```markdown
---
title: My Report
description: One-line summary shown in the UI
icon: "📊"
schedule: manual        # or "every 2h" / handled per-user in the UI
enabled: true
---
Write a concise report of what I worked on. Use ONLY the data provided…
```

**`app.py`** — the simplest app reuses the shared runner (≈15 lines); copy
[day-recap/app.py](day-recap/app.py) as a template. Import the shared modules by
their package path:

```python
from deskmate.apps.agent import run_agent
from deskmate.apps.common import add_agent_time_args, agent_time_kwargs_from_args, output_dir, run_cli, write_markdown
```

It calls `run_agent(PIPE_MD, …)`, which for an unrecognized app name falls back
to a generic prefetch (`/activity-summary`) + single-shot report — so a brand-new
plugin works with no changes to `agent.py`. Output lands in
`~/.deskmate/apps/<name>/output/<timestamp>/`.

> A user plugin can import `deskmate.apps.*` only when DeskMate is installed in
> the same environment (so `deskmate` is on `sys.path`) — which is the normal
> case, since the daemon launches each app with its own interpreter.

After adding the folder, **restart the daemon** (discovery runs at startup). Your
app then appears in the My Apps UI, can be scheduled there, and is runnable via
`POST /apps/<name>/run`. To customize behavior beyond the generic report, give
your `app.py` its own logic (query the DeskMate API at `DESKMATE_API` or open the
DB via `deskmate.apps.common` helpers) instead of delegating to `run_agent`.

> Folder names are sanitized on lookup — a plugin name can't contain path
> separators or `..`, so it can never resolve outside its app root.

## Configuration

Ollama settings can live in `~/.deskmate/config.toml` (or `%USERPROFILE%\.deskmate\config.toml` on Windows):

```toml
[ollama]
base = "http://127.0.0.1:11434"
model = "qwen3.5_4b_ov:v1"
chat_timeout = 600
```

| Setting | Default | Description |
|---|---|---|
| `[ollama].base` | `http://127.0.0.1:11434` | Ollama API endpoint |
| `[ollama].model` | `qwen3_8b_ov:v1` | Model to use |
| `[ollama].chat_timeout` | `600` | `/api/chat` timeout (seconds) |

Environment overrides (take precedence over the file):

| Env variable | Same as |
|---|---|
| `OLLAMA_BASE` | `[ollama].base` |
| `OLLAMA_MODEL` | `[ollama].model` |
| `OLLAMA_CHAT_TIMEOUT` | `[ollama].chat_timeout` |
| `DESKMATE_API` | — (default `http://127.0.0.1:3030`) |
| `MAX_TOOL_ROUNDS` | — (default `12`) |

Pydantic env form: `DESKMATE_ollama__model=my-model:tag` (also overrides the file).

Override the model for one run: `python deskmate\apps\day-recap\app.py --model other-model:tag`

## Architecture

```
pipe.md (prompt)
    │
    ▼
agent.py (runner)  ──→  Ollama /api/chat (with tools)
    │                         │
    │  ◄── tool_calls ────────┘
    │
    ▼
execute_tool()  ──→  DeskMate /search API
    │
    │  ──→ results back to model
    │
    ▼
model generates final report (markdown)
```

The `pipe.md` is a prompt; the agent runner manages the tool-calling loop, and
the model autonomously decides what data to query and how to present it.

Both the app runner (`apps/agent.py`) and the in-app **Ask** agent
(`deskmate/engine/ask.py`) share one engine, `deskmate/engine/llm.py`,
for the proxy-bypassing HTTP transport and the `/api/chat` call. Each agent
keeps its own orchestration and its own `OLLAMA_BASE` / `OLLAMA_MODEL` module
settings (so `--model` still works by overriding `agent.OLLAMA_MODEL`).
