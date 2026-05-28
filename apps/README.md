# pc_assistant apps

Local LLM apps driven by `pipe.md` prompts. Each app follows this execution model:

1. Read `pipe.md` (YAML frontmatter + prompt body)
2. Prepend context header (time range, timezone, API base)
3. Send to LLM (Ollama) with `SKILL.md` as system knowledge
4. **day-recap** / **ai-habits**: model calls `activity_summary` / `search` in a loop (tool-driven)
5. Other pipes may pre-fetch API data when `pipe.md` instructs it
6. LLM generates the report in the exact format specified by `pipe.md`

Output is written to: `%USERPROFILE%\.pc_assistant\apps\<app-name>\output\<timestamp>\`

## Prerequisites

- **Ollama** running at `http://127.0.0.1:11434` with a model loaded
- **pc_assistant API** running at `http://127.0.0.1:3030`

## Apps

### video-export

Exports recent screenshots as a video clip. The LLM calls `POST /frames/export`
as instructed by `pipe.md`.

```cmd
cd /d C:\hongbo\UX\pc_assistant
conda activate support_qwen3.5
python apps\video-export\app.py --minutes 5 --verbose
```

### day-recap

LLM queries screen and audio data, then generates a structured day recap with
Summary, Accomplishments, Key Moments, Unfinished Work, Patterns, and Next step.

```cmd
python apps\day-recap\app.py --hours 16 --verbose
```

### ai-habits

LLM searches for AI tool usage (ChatGPT, Claude, Copilot, Cursor, Gemini,
Perplexity) and generates an analysis report. The agent runs one search per
tool in Python and only the tools with verified hits are passed to the model
(it cannot fabricate usage for tools that were never opened).

```cmd
python apps\ai-habits\app.py --hours 24 --verbose
```

### meeting-summary

Finds the meeting that just ended, summarizes its transcript (key topics,
decisions, action items), and patches the summary back onto the meeting record
(appends under a `## Summary` heading in the note, and refreshes a generic
title). Skips the write-back when there is nothing worth saving.

```cmd
python apps\meeting-summary\app.py --verbose
```

### standup-update

Generates a short standup update (Yesterday / Today / Blockers) from the last
24 hours of recorded activity. Copy-paste ready for a team standup; capped at
~150 words.

```cmd
python apps\standup-update\app.py --hours 24 --verbose
```

### time-breakdown

Breaks down the supplied time range by application, category (coding /
meetings / browsing / writing / communication / other), and project, then
computes a productivity score from `activity_summary` minutes.

```cmd
python apps\time-breakdown\app.py --hours 12 --verbose
```

### ai-prompt-journal

Captures every prompt the user typed into AI tools (ChatGPT, Claude, Gemini,
Perplexity, Grok, DeepSeek, Copilot, Cursor, local models, etc.) over the
supplied window (default 1 hour) and appends only the genuinely new prompts to
a daily markdown journal at
`%USERPROFILE%\.pc_assistant\apps\ai-prompt-journal\journal\YYYY-MM-DD.md`.
Designed to be scheduled hourly.

```cmd
python apps\ai-prompt-journal\app.py --hours 1 --verbose
```

### email-digest

Summarizes inbox activity from two sources: Gmail OAuth / Gmail API and Outlook
OAuth / Microsoft Graph when accounts are connected, plus local screen / UI
recordings for Outlook, Thunderbird, Windows Mail, Mailspring, Mailbird, eM
Client and webmail tabs (Gmail, Outlook web, Outlook 365, QQ Mail, 163 Mail,
Yahoo Mail). Lists tools used, top senders / threads, drafts in progress,
action items and patterns.

```cmd
python apps\email-digest\app.py --hours 24 --verbose
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

```cmd
python apps\todo-list\app.py --hours 24 --verbose
```

### email-compose

LLM drafts an email (or a reply) for a connected Gmail or Outlook account.
Optionally fetches a source message via `--reply-to <id>` to ground the draft.
The script writes Subject + Body + up to 2 alternatives + a Send Preview to
disk for review; pass `--send` to actually deliver via the provider's API.

```cmd
:: New draft (review only)
python apps\email-compose\app.py --provider gmail --to a@b.com --intent "需要本周确认预算" --verbose

:: Reply, then actually send
python apps\email-compose\app.py --provider outlook --to a@b.com --intent "我同意方案 A" --reply-to AAMkAGI... --send --verbose
```

## When to use which app

| Your moment | App | Why |
|---|---|---|
| Morning standup in Slack/Teams | `standup-update` | Yesterday / Today / Blockers, ≤150 words |
| Clean my inbox / plan my day | `todo-list` | One checkbox list from email + meetings, no keyword needed |
| Write a new business email | `email-compose` | LLM draft + 2 alternatives, then `--send` |
| Reply to a specific message | `email-compose --reply-to <id>` | Grounds the draft in the source message |
| A meeting just ended | `meeting-summary` (auto on `meeting_ended`) | Patches summary back to the meeting record |
| Capture what I asked AI this hour | `ai-prompt-journal` (auto every 1h) | Appends new prompts to today's journal |
| End-of-day reflection | `day-recap` | Accomplishments + key moments + unfinished work |
| Audit my email time today | `email-digest` | Tool minutes + senders + drafts + todos + patterns |
| Weekly time breakdown | `time-breakdown` | App / category / project + productivity score |
| Weekly AI usage habits | `ai-habits` | Per-tool time + effectiveness, no fabricated tools |
| Record the last few minutes | `video-export` | Screenshots → mp4 |

## Configuration

| Env variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE` | `http://127.0.0.1:11434` | Ollama API endpoint |
| `OLLAMA_MODEL` | `qwen3_8b_ov:v1` | Model to use |
| `PC_ASSISTANT_API` | `http://127.0.0.1:3030` | pc_assistant API base |
| `MAX_TOOL_ROUNDS` | `12` | Max tool-calling rounds per run |

Override the model per run: `python apps\day-recap\app.py --model other-model:tag`

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
execute_tool()  ──→  pc_assistant /search API
    │
    │  ──→ results back to model
    │
    ▼
model generates final report (markdown)
```

The `pipe.md` is a prompt; the agent runner manages the tool-calling loop, and
the model autonomously decides what data to query and how to present it.
