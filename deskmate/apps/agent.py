"""LLM agent runner for DeskMate pipe apps.

Reads a pipe.md, follows its search instructions by pre-executing API calls,
then sends the real results to the LLM for report generation.

Key principle: the pipe.md IS the logic. We parse its instructions and execute
them in Python (since a 4B model can't do multi-step tool calling), then hand
the results to the model.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from deskmate.console import echo_stderr
from deskmate.engine import llm
from deskmate.engine.activity_summary import format_summary_for_agent
from deskmate.engine.day_recap_context import (
    calendar_days_in_range,
    format_search_items,
    format_ts_local,
    range_spans_calendar_days,
)

from .common import normalize_capture_text
from .learning_slice import (
    build_learning_sessions,
    filter_learning_edited_files,
    filter_learning_key_texts,
    format_learning_bundle,
    merge_transcript_paragraphs,
    select_spanning as _select_spanning,
)
from deskmate.learning_memory import build_learning_enrichment


def _todo_list_email_evidence(
    start_iso: str,
    end_iso: str,
    *,
    verbose: bool = False,
    limit_per_tool: int = 3,
) -> tuple[str, list[str]]:
    """Email prefetch for todo-list; per-day sections when range spans multiple days."""
    if not range_spans_calendar_days(start_iso, end_iso):
        text, verified = _do_email_digest_prefetch(
            start_iso, end_iso, verbose=verbose, limit_per_tool=limit_per_tool,
        )
        return text, verified

    blocks: list[str] = []
    verified_all: list[str] = []
    for day, day_start, day_end in calendar_days_in_range(start_iso, end_iso):
        day_text, verified = _do_email_digest_prefetch(
            day_start, day_end, verbose=verbose, limit_per_tool=limit_per_tool,
        )
        blocks.append(f"### {day.isoformat()} ({day_start} → {day_end})\n\n{day_text}")
        for v in verified:
            if v not in verified_all:
                verified_all.append(v)
    return "\n\n".join(blocks), verified_all

@dataclass(frozen=True)
class PipeToolConfig:
    max_search: int
    search_limit: int
    max_summary: int = 2
    max_rounds: int = 12
    require_summary_first: bool = False
    min_search_before_report: int = 0
    num_predict: int = 4096


PIPE_TOOL_CONFIG: dict[str, PipeToolConfig] = {
    "ai-habits": PipeToolConfig(
        max_search=6,
        search_limit=5,
        max_summary=1,
        max_rounds=14,
        require_summary_first=False,
        min_search_before_report=6,
        num_predict=4096,
    ),
    # standup-update: activity_summary first, then limited follow-up searches.
    # then at most 2 follow-up searches (limit=5) to confirm blockers / tasks.
    "standup-update": PipeToolConfig(
        max_search=2,
        search_limit=5,
        max_summary=1,
        max_rounds=8,
        require_summary_first=True,
        min_search_before_report=0,
        num_predict=2048,
    ),
    # time-breakdown: activity_summary first, then
    # up to 4 follow-up searches (limit=5) to disambiguate app categories/topics.
    "time-breakdown": PipeToolConfig(
        max_search=4,
        search_limit=5,
        max_summary=1,
        max_rounds=10,
        require_summary_first=True,
        min_search_before_report=0,
        num_predict=3072,
    ),
}

# Pipes that let the model drive tool calls in a loop. Small models are unreliable
# at multi-step tool calling; most pipes use Python prefetch + single-shot instead.
TOOL_DRIVEN_PIPES: frozenset[str] = frozenset()

AI_HABITS_APP_NAMES = (
    "ChatGPT", "Claude", "Copilot", "Cursor", "Gemini", "Perplexity",
)

# Display label → Windows process names (and optional OCR keyword for browser tabs)
AI_TOOL_TARGETS: dict[str, dict[str, Any]] = {
    "ChatGPT": {"app_names": ["ChatGPT.exe", "chatgpt.exe"], "q_fallback": "ChatGPT"},
    "Claude": {"app_names": ["Claude.exe", "claude.exe"], "q_fallback": "Claude"},
    "Copilot": {
        "app_names": ["Copilot.exe", "copilot.exe", "GitHubCopilot.exe", "Code.exe"],
        "q_fallback": "Copilot",
    },
    "Cursor": {"app_names": ["Cursor.exe"], "q_fallback": None},
    "Gemini": {"app_names": ["chrome.exe", "msedge.exe"], "q_fallback": "Gemini"},
    "Perplexity": {"app_names": ["chrome.exe", "msedge.exe"], "q_fallback": "Perplexity"},
}

# Display label → Windows email process names (and optional OCR/URL keyword for
# web-mail tabs). Gmail and Outlook also have OAuth backends; the email-digest
# app merges those structured sources with local screen/UI evidence.
EMAIL_TOOL_TARGETS: dict[str, dict[str, Any]] = {
    "Outlook": {
        "app_names": ["OUTLOOK.EXE", "Outlook.exe", "outlook.exe", "olk.exe"],
        "q_fallback": "Outlook",
    },
    "Thunderbird": {
        "app_names": ["thunderbird.exe", "Thunderbird.exe"],
        "q_fallback": "Thunderbird",
    },
    "Windows Mail": {
        "app_names": ["HxOutlook.exe", "HxMail.exe", "HxAccounts.exe"],
        "q_fallback": None,
    },
    "Mailspring": {
        "app_names": ["Mailspring.exe", "mailspring.exe"],
        "q_fallback": "Mailspring",
    },
    "Mailbird": {
        "app_names": ["Mailbird.exe", "mailbird.exe"],
        "q_fallback": "Mailbird",
    },
    "eM Client": {
        "app_names": ["MailClient.exe", "eM Client.exe"],
        "q_fallback": "eM Client",
    },
    "Gmail (web)": {
        "app_names": ["chrome.exe", "msedge.exe", "firefox.exe", "brave.exe"],
        "q_fallback": "mail.google.com",
    },
    "Outlook (web)": {
        "app_names": ["chrome.exe", "msedge.exe", "firefox.exe", "brave.exe"],
        "q_fallback": "outlook.live.com",
    },
    "Outlook 365 (web)": {
        "app_names": ["chrome.exe", "msedge.exe", "firefox.exe", "brave.exe"],
        "q_fallback": "outlook.office",
    },
    "QQ Mail (web)": {
        "app_names": ["chrome.exe", "msedge.exe", "firefox.exe", "brave.exe"],
        "q_fallback": "mail.qq.com",
    },
    "163 Mail (web)": {
        "app_names": ["chrome.exe", "msedge.exe", "firefox.exe", "brave.exe"],
        "q_fallback": "mail.163.com",
    },
    "Yahoo Mail (web)": {
        "app_names": ["chrome.exe", "msedge.exe", "firefox.exe", "brave.exe"],
        "q_fallback": "mail.yahoo.com",
    },
}

OLLAMA_BASE, OLLAMA_MODEL, _OLLAMA_CHAT_TIMEOUT = llm.resolve_ollama_settings()
API_BASE = os.environ.get("DESKMATE_API", "http://127.0.0.1:3030")
MAX_TOOL_ROUNDS = int(os.environ.get("MAX_TOOL_ROUNDS", "8"))

SKILL_PATH = Path(__file__).with_name("SKILL.md")

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# Regex to extract "for each tool separately: X, Y, Z" from pipe.md
_APP_FILTER_RE = re.compile(
    r"app_name\s+filter\s+for\s+each\s+tool\s+separately:\s*(.+)",
    re.IGNORECASE,
)
# Regex to detect "POST /frames/export" instruction in pipe.md
_FRAMES_EXPORT_RE = re.compile(r"POST\s+/frames/export", re.IGNORECASE)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search DeskMate screen captures, audio transcriptions and UI events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "FTS search query (optional)"},
                    "content_type": {"type": "string", "enum": ["all", "ocr", "audio", "ui", "element"]},
                    "role": {"type": "string", "description": "With content_type=element, filter UI controls by UIA role"},
                    "app_name": {"type": "string", "description": "Filter by app process name"},
                    "start_time": {"type": "string", "description": "ISO 8601 start time"},
                    "end_time": {"type": "string", "description": "ISO 8601 end time"},
                    "limit": {"type": "integer", "description": "Max results to return"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "activity_summary",
            "description": (
                "Broad activity overview for a time range — apps with minutes, windows, "
                "key_texts, timeline, snippets, edited_files, audio. Call this FIRST for "
                "day recap / habits before targeted search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "string", "description": "ISO 8601 start (required)"},
                    "end_time": {"type": "string", "description": "ISO 8601 end (required)"},
                    "app_name": {"type": "string", "description": "Optional app filter"},
                    "q": {"type": "string", "description": "Optional keyword filter"},
                },
                "required": ["start_time", "end_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "frames_export",
            "description": "Export recent screenshots as a video clip via POST /frames/export.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "string", "description": "ISO 8601 start time"},
                    "end_time": {"type": "string", "description": "ISO 8601 end time"},
                    "fps": {"type": "number", "description": "Output FPS (default 1.0)"},
                    "limit": {"type": "integer", "description": "Max frames to export"},
                },
                "required": ["start_time", "end_time"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "timeline",
            "description": (
                "Unified, time-ordered stream FUSING all capture sources into one feed: "
                "screen frames, audio transcripts, input (clicks / typed text), clipboard "
                "and window focus/title changes. Each row carries source, kind, app, a short "
                "summary and a confidence score. Use for strongly time-ordered, cross-source "
                "questions ('what happened step by step', 'what did I copy', 'what did I type "
                "during the meeting'). Prefer 'search' for keywords, 'activity_summary' for stats."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "string", "description": "ISO 8601 start (optional)"},
                    "end_time": {"type": "string", "description": "ISO 8601 end (optional)"},
                    "sources": {
                        "type": "string",
                        "description": "Comma-separated subset: screen,audio,input,clipboard,window. Omit for all.",
                    },
                    "limit": {"type": "integer", "description": "Max events, newest first (default 100, max 1000)."},
                },
                "required": [],
            },
        },
    },
]


# ── HTTP helpers (http.client to bypass proxy) ───────────────────────────

# Transport primitives are shared with the Ask agent via the engine module.
_raw_request = llm.raw_request
_http_get = llm.http_get
_http_post = llm.http_post
_http_patch = llm.http_patch


# ── tool-driven session (model chooses API calls) ─────────────────────────

@dataclass
class ToolSession:
    start_iso: str
    end_iso: str
    pipe_name: str = ""
    search_calls: int = 0
    summary_calls: int = 0
    format_for_llm: bool = False


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            return json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            return {}
    return raw if isinstance(raw, dict) else {}


def _fill_time_range(args: dict[str, Any], session: ToolSession) -> dict[str, Any]:
    out = dict(args)
    out.setdefault("start_time", session.start_iso)
    out.setdefault("end_time", session.end_iso)
    return out


def _pipe_config(pipe_name: str) -> PipeToolConfig | None:
    return PIPE_TOOL_CONFIG.get(pipe_name)


def _check_search_budget(session: ToolSession) -> str | None:
    cfg = _pipe_config(session.pipe_name)
    if not cfg:
        return None
    if session.search_calls >= cfg.max_search:
        return (
            f"Search budget exhausted ({cfg.max_search} calls max for {session.pipe_name}). "
            "Write the report from data you already have."
        )
    return None


def _execute_ai_habits_search(args: dict[str, Any], session: ToolSession) -> str:
    """Search one AI tool with correct process names; never imply usage on 0 hits."""
    label = (args.get("app_name") or args.get("q") or "").strip()
    target = AI_TOOL_TARGETS.get(label)
    if not target:
        for key in AI_TOOL_TARGETS:
            if key.lower() == label.lower():
                target = AI_TOOL_TARGETS[key]
                label = key
                break
    if not target:
        return (
            f"Unknown AI tool label '{label}'. "
            f"Use app_name one of: {', '.join(AI_HABITS_APP_NAMES)}."
        )

    cfg = _pipe_config("ai-habits")
    limit = int(args.get("limit") or (cfg.search_limit if cfg else 5))
    base = {k: v for k, v in args.items() if k in ("start_time", "end_time")}
    all_items: list[dict[str, Any]] = []
    tried: list[str] = []

    for proc in target["app_names"]:
        tried.append(proc)
        params = "&".join(
            f"{k}={quote(str(v))}"
            for k, v in {
                **base,
                "content_type": "all",
                "app_name": proc,
                "limit": limit,
                "semantic": True,
            }.items()
            if v is not None
        )
        result = _http_get(f"{API_BASE}/search?{params}")
        all_items.extend(result.get("data", []) if isinstance(result, dict) else [])

    q_fb = target.get("q_fallback")
    if q_fb and len(format_search_items(all_items)) == 0:
        tried.append(f"q={q_fb}")
        params = "&".join(
            f"{k}={quote(str(v))}"
            for k, v in {
                **base,
                "content_type": "all",
                "q": q_fb,
                "limit": limit,
                "semantic": True,
            }.items()
            if v is not None
        )
        result = _http_get(f"{API_BASE}/search?{params}")
        all_items.extend(result.get("data", []) if isinstance(result, dict) else [])

    lines = format_search_items(all_items, max_text=450)
    hit_count = len(lines)
    header = (
        f"### {label} | tried: {', '.join(tried)} | substantive_hits={hit_count}\n"
    )
    if hit_count == 0:
        return (
            f"{header}"
            f"NO USAGE RECORDED for {label}. "
            f"Do NOT list {label} in 'AI Tools Used' or assign any minutes.\n"
        )
    return header + "\n".join(lines)


def _check_summary_budget(session: ToolSession) -> str | None:
    cfg = _pipe_config(session.pipe_name)
    if not cfg:
        return None
    if session.summary_calls >= cfg.max_summary:
        return (
            f"activity_summary budget exhausted ({cfg.max_summary} calls max). "
            "Use prior results or /search."
        )
    return None


# ── tool dispatch ─────────────────────────────────────────────────────────

def execute_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    session: ToolSession | None = None,
) -> str:
    args = _fill_time_range(arguments, session) if session else arguments
    try:
        if name == "activity_summary":
            if session and session.pipe_name == "ai-habits" and (
                args.get("app_name") or args.get("q")
            ):
                return (
                    "For ai-habits: call activity_summary with ONLY start_time and end_time "
                    "(no app_name, no q). Use per-tool search for each AI product."
                )
            if session:
                blocked = _check_summary_budget(session)
                if blocked:
                    return blocked
                session.summary_calls += 1
            params = "&".join(
                f"{k}={quote(str(v))}"
                for k, v in args.items()
                if v is not None and k in ("start_time", "end_time", "app_name", "q")
            )
            rich = (
                "&include_recording=true&include_snippets=true&include_guidance=true"
                "&max_snippets=12&max_snippet_chars=800"
            )
            summary = _http_get(f"{API_BASE}/activity-summary?{params}{rich}")
            if session and session.format_for_llm:
                return format_summary_for_agent(summary)
            text = json.dumps(summary, ensure_ascii=False)
        elif name == "search":
            if session:
                blocked = _check_search_budget(session)
                if blocked:
                    return blocked
                session.search_calls += 1
                cfg = _pipe_config(session.pipe_name)
                if cfg:
                    args.setdefault("limit", cfg.search_limit)
            # Prefer hybrid recall; the API ignores this unless semantic search
            # is enabled in config, so it's a safe default.
            args.setdefault("semantic", True)
            if session and session.pipe_name == "ai-habits" and session.format_for_llm:
                clean = dict(args)
                clean.pop("q", None)  # per-tool search uses process names, not generic q
                return _execute_ai_habits_search(clean, session)
            params = "&".join(
                f"{k}={quote(str(v))}" for k, v in args.items() if v is not None
            )
            result = _http_get(f"{API_BASE}/search?{params}")
            if session and session.format_for_llm:
                items = result.get("data", []) if isinstance(result, dict) else []
                lines = format_search_items(items, max_text=450)
                label = args.get("app_name") or "search"
                if not lines:
                    return (
                        f"### {label} | substantive_hits=0\n"
                        f"NO USAGE RECORDED. Do NOT list {label} in the report.\n"
                    )
                return f"### {label} | substantive_hits={len(lines)}\n" + "\n".join(lines)
            text = json.dumps(result, ensure_ascii=False)
        elif name == "frames_export":
            result = _http_post(f"{API_BASE}/frames/export", args)
            text = json.dumps(result, ensure_ascii=False)
        elif name == "timeline":
            valid_sources = {"screen", "audio", "input", "clipboard", "window"}
            try:
                limit = int(args.get("limit") or 100)
            except (TypeError, ValueError):
                limit = 100
            limit = max(1, min(limit, 1000))
            raw_sources = str(args.get("sources") or "").strip()
            sources = [
                s.strip().lower()
                for s in raw_sources.split(",")
                if s.strip().lower() in valid_sources
            ]
            params = [f"limit={limit}"]
            if args.get("start_time"):
                params.append(f"since={quote(str(args['start_time']))}")
            if args.get("end_time"):
                params.append(f"until={quote(str(args['end_time']))}")
            if sources:
                params.append(f"sources={quote(','.join(sources))}")
            result = _http_get(f"{API_BASE}/timeline/unified?{'&'.join(params)}")
            text = json.dumps(result, ensure_ascii=False)
        else:
            text = json.dumps({"error": f"unknown tool: {name}"}, ensure_ascii=False)
    except Exception as exc:
        text = json.dumps({"error": str(exc)}, ensure_ascii=False)
    if len(text) > 14000:
        text = text[:14000] + "\n...(truncated)"
    return text


# ── pipe.md instruction parser ────────────────────────────────────────────

def _extract_per_tool_names(pipe_body: str) -> list[str]:
    """Parse pipe.md for 'app_name filter for each tool separately: X, Y, Z'.

    Returns a list of tool names to search, or [] if no such instruction.
    Parse per-app search instructions from pipe.md.
    """
    m = _APP_FILTER_RE.search(pipe_body)
    if not m:
        return []
    raw = m.group(1)
    raw = raw.split(".")[0]  # stop at sentence boundary
    names = [n.strip().rstrip(",") for n in raw.split(",")]
    return [n for n in names if n and len(n) < 30]


# ── pre-execute searches ──────────────────────────────────────────────────

def _do_per_tool_searches(
    tool_names: list[str],
    start: str,
    end: str,
    limit_per_search: int = 5,
    verbose: bool = False,
    *,
    include_date: bool = False,
) -> str:
    """Execute one /search per tool name and format results.

    One /search per app name, e.g. app_name=Cursor.exe&limit=5&start_time=...
    """
    sections: list[str] = []

    for tool_name in tool_names:
        url = (
            f"{API_BASE}/search"
            f"?app_name={quote(tool_name)}"
            f"&content_type=all"
            f"&limit={limit_per_search}"
            f"&start_time={quote(start)}"
            f"&end_time={quote(end)}"
        )
        try:
            result = _http_get(url)
            items = result.get("data", [])
        except Exception as exc:
            items = []
            if verbose:
                echo_stderr(f"  [search] {tool_name}: error {exc}")

        if verbose:
            echo_stderr(f"  [search] {tool_name}: {len(items)} results")

        if not items:
            sections.append(
                f"### Search: app_name={tool_name}\n"
                f"Results: 0 items found.\n"
            )
            continue

        max_text = 450 if limit_per_search >= 10 else 200
        lines = format_search_items(items, max_text=max_text, include_date=include_date)

        sections.append(
            f"### Search: app_name={tool_name}\n"
            f"Results: {len(items)} items ({len(lines)} after noise filter).\n"
            + ("\n".join(lines) if lines else "  (no substantive matches)")
        )

    return "\n\n".join(sections)


def _do_frames_export(start: str, end: str, verbose: bool = False) -> str:
    """Pre-execute POST /frames/export when pipe.md requests it.

    The pipe.md says: "Use the POST /frames/export endpoint with the time range
    and fps=1.0." — we execute this directly before the LLM writes the report.
    """
    body = {
        "start_time": start,
        "end_time": end,
        "fps": 1.0,
        "limit": 1000,
    }
    if verbose:
        echo_stderr("  [export] POST /frames/export fps=1.0 ...")
    try:
        result = _http_post(f"{API_BASE}/frames/export", body, timeout=60)
    except Exception as exc:
        if verbose:
            echo_stderr(f"  [export] error: {exc}")
        return (
            f"### POST /frames/export — FAILED\n"
            f"Error: {exc}\n\n"
            f"The DeskMate API may not be running or the time range has no frames."
        )

    success = result.get("success", False)
    file_path = result.get("file_path", "")
    frame_count = result.get("frame_count", 0)
    manifest_path = result.get("manifest_path", "")
    reason = result.get("reason", "")

    if verbose:
        echo_stderr(f"  [export] success={success}, frames={frame_count}, path={file_path}")

    lines = ["### POST /frames/export — Result\n"]
    lines.append(f"- success: {success}")
    lines.append(f"- frame_count: {frame_count}")
    if file_path:
        lines.append(f"- file_path: `{file_path}`")
    if manifest_path:
        lines.append(f"- manifest_path: `{manifest_path}`")
    if reason:
        lines.append(f"- reason: {reason}")
    lines.append(f"- time_range: {start} to {end}")
    lines.append("- fps: 1.0")
    try:
        frame_count_int = int(frame_count)
    except (TypeError, ValueError):
        frame_count_int = 0
    if frame_count_int > 500:
        lines.append(
            "- suggestion: Large export detected; use a shorter --minutes range "
            "or lower fps (for example 0.5) to reduce file size."
        )

    return "\n".join(lines)


def _fetch_activity_summary(
    start: str,
    end: str,
    verbose: bool = False,
    *,
    rich: bool = False,
) -> dict[str, Any]:
    extra = ""
    if rich:
        extra = "&max_snippets=12&max_snippet_chars=900&max_memories=8"
    try:
        return _http_get(
            f"{API_BASE}/activity-summary"
            f"?start_time={quote(start)}&end_time={quote(end)}"
            f"&include_recording=true&include_snippets=true&include_guidance=true"
            f"{extra}"
        )
    except Exception as exc:
        if verbose:
            echo_stderr(f"  [prefetch] error: {exc}")
        return {}


def _do_broad_search(start: str, end: str, verbose: bool = False) -> str:
    """Fetch /activity-summary — aggregated activity bundle."""
    summary = _fetch_activity_summary(start, end, verbose=verbose)
    if not summary:
        return "(Failed to fetch activity data from DeskMate API.)"

    if verbose:
        audio_n = int((summary.get("audio_summary") or {}).get("segment_count") or 0)
        echo_stderr(
            f"  [prefetch] summary status={summary.get('data_status')} "
            f"apps={len(summary.get('apps', []))} "
            f"key_texts={len(summary.get('key_texts', []))} "
            f"snippets={len(summary.get('snippets') or [])} audio={audio_n}",        )

    return format_summary_for_agent(summary)


def _do_content_search(
    start: str,
    end: str,
    *,
    limit: int = 20,
    app_name: str | None = None,
    q: str | None = None,
    content_type: str = "all",
    verbose: bool = False,
) -> list[dict[str, Any]]:
    params = [
        f"content_type={quote(content_type or 'all')}",
        f"limit={limit}",
        f"start_time={quote(start)}",
        f"end_time={quote(end)}",
        "semantic=true",
    ]
    if app_name:
        params.append(f"app_name={quote(app_name)}")
    if q:
        params.append(f"q={quote(q)}")
    url = f"{API_BASE}/search?{'&'.join(params)}"
    try:
        result = _http_get(url)
        items = result.get("data", [])
        if verbose:
            label = app_name or q or content_type or "all"
            echo_stderr(f"  [search] {label}: {len(items)} raw")
        return items
    except Exception as exc:
        if verbose:
            echo_stderr(f"  [search] error: {exc}")
        return []


def _recap_project_keywords(summary: dict[str, Any]) -> list[str]:
    """Extract project/topic tokens from edited files and window titles."""
    skip = {"new tab", "google chrome", "microsoft edge", "settings", "task manager"}
    found: list[str] = []
    seen: set[str] = set()
    for ef in summary.get("edited_files") or []:
        path = (ef.get("path") or "").replace("\\", "/")
        stem = path.rsplit("/", 1)[-1]
        if stem and len(stem) > 4 and stem.lower() not in seen:
            seen.add(stem.lower())
            found.append(stem)
    for w in summary.get("windows") or []:
        title = (w.get("window_name") or "").strip()
        if not title or len(title) < 6:
            continue
        head = title.split(" - ")[0].strip()
        low = head.lower()
        if low in skip or low in seen or len(head) < 5:
            continue
        seen.add(low)
        found.append(head)
    return found[:4]


def _data_sufficiency_warning(summary: dict[str, Any]) -> str:
    """Return a data-quality warning banner when the range is too thin to recap.

    Without this, a recorder that started at 5pm (or a near-empty range) still
    gets a full LLM report, which reads as "you rested all day" rather than "we
    barely captured anything". We surface the gap explicitly so the model states
    its limits instead of confidently narrating thin air. Returns "" when there
    is enough evidence to recap normally.
    """
    status = summary.get("data_status", "unknown")
    apps = [a for a in (summary.get("apps") or []) if float(a.get("minutes") or 0) > 0]
    total_minutes = sum(float(a.get("minutes") or 0) for a in apps)
    audio_count = int((summary.get("audio_summary") or {}).get("segment_count") or 0)
    key_texts = summary.get("key_texts") or []
    edited = summary.get("edited_files") or []

    # Plenty of evidence — no warning needed.
    if status == "ok" and (total_minutes >= 15 or audio_count >= 3 or len(key_texts) >= 5):
        return ""

    reasons: list[str] = []
    if status == "not_recording":
        reasons.append("DeskMate was not recording for most/all of this range")
    elif status in ("no_capture_in_range", "empty_but_recording", "unknown"):
        reasons.append("almost no screen/audio was captured in this range")
    if total_minutes < 15:
        reasons.append(f"only ~{total_minutes:.0f} min of app activity was tracked")
    if audio_count == 0:
        reasons.append("no audio was transcribed")
    if not edited and not key_texts:
        reasons.append("no edited files or substantive on-screen text were found")

    if not reasons:
        return ""

    return (
        "### ⚠️ DATA QUALITY WARNING — limited evidence\n"
        "The captured data for this range is sparse: "
        + "; ".join(reasons)
        + ".\nThis usually means the recorder was off, just started, or the device "
        "was idle — NOT that the user did nothing. In the report you MUST state up "
        "front that coverage was limited and that conclusions are partial. Do NOT "
        "imply the user rested or was unproductive; only describe what little the "
        "data actually shows.\n"
    )


def _do_day_recap_prefetch(start: str, end: str, verbose: bool = False) -> str:
    """Rich prefetch: activity-summary + up to 5 supplemental /search calls."""
    multi_day = range_spans_calendar_days(start, end)
    include_date = multi_day

    summary = _fetch_activity_summary(start, end, verbose=verbose, rich=True)
    if not summary:
        return "(Failed to fetch activity data from DeskMate API.)"

    sections: list[str] = []
    warning = _data_sufficiency_warning(summary)
    if warning:
        if verbose:
            echo_stderr("  [day-recap] data sufficiency: LOW — injecting warning banner")
        sections.append(warning)
    if multi_day:
        day_slices = calendar_days_in_range(start, end)
        sections.append(
            f"### Multi-day range ({len(day_slices)} calendar day(s))\n"
            f"Report MUST use one `## YYYY-MM-DD` section per day below.\n"
            f"Timestamps in this bundle include the calendar date."
        )
        sections.append(
            "### Range-wide activity (totals across all days)\n\n"
            + format_summary_for_agent(summary, include_date=True)
        )
        # Most-recent day first so that if the bundle is later truncated to fit
        # the model budget, the OLDEST days are the ones dropped — not the days
        # the user most likely cares about (optimization note 9.6).
        for day, day_start, day_end in reversed(day_slices):
            day_summary = _fetch_activity_summary(day_start, day_end, verbose=verbose, rich=True)
            if not day_summary:
                continue
            sections.append(
                f"### Day {day.isoformat()} ({day_start} → {day_end})\n\n"
                + format_summary_for_agent(day_summary, include_date=True)
            )
    else:
        sections.append(format_summary_for_agent(summary, include_date=include_date))

    apps = sorted(
        summary.get("apps") or [],
        key=lambda a: float(a.get("minutes") or 0),
        reverse=True,
    )
    top_apps = [a.get("name", "") for a in apps[:4] if a.get("name")]
    max_searches = 5
    search_limit = 10
    searches_left = max_searches

    if top_apps and searches_left > 0:
        n = min(len(top_apps), searches_left - 1)
        if verbose:
            echo_stderr(f"  [day-recap] app searches ({n}): {top_apps[:n]}")
        extra = _do_per_tool_searches(
            top_apps[:n], start, end,
            limit_per_search=search_limit,
            verbose=verbose,
            include_date=include_date,
        )
        if extra.strip():
            sections.append("### Supplemental searches (top apps by minutes)\n\n" + extra)
        searches_left -= n

    if searches_left > 0:
        if multi_day:
            for day, day_start, day_end in calendar_days_in_range(start, end):
                if searches_left <= 0:
                    break
                broad = _do_content_search(
                    day_start, day_end, limit=12, verbose=verbose,
                )
                lines = format_search_items(broad, max_text=450, include_date=True)
                if lines:
                    sections.append(
                        f"### Broad search — {day.isoformat()}\n"
                        f"Results: {len(broad)} raw, {len(lines)} kept.\n"
                        + "\n".join(lines)
                    )
                searches_left -= 1
        else:
            broad = _do_content_search(
                start, end, limit=25, verbose=verbose,
            )
            lines = format_search_items(broad, max_text=500, include_date=include_date)
            if verbose:
                echo_stderr(f"  [day-recap] broad search: {len(lines)} substantive")
            if lines:
                sections.append(
                    "### Broad search (all apps, substantive content only)\n"
                    f"Results: {len(broad)} raw, {len(lines)} kept.\n"
                    + "\n".join(lines)
                )
            searches_left -= 1

    keywords = _recap_project_keywords(summary)
    for kw in keywords:
        if searches_left <= 0:
            break
        items = _do_content_search(start, end, limit=10, q=kw, verbose=verbose)
        lines = format_search_items(items, max_text=400, include_date=include_date)
        if lines:
            sections.append(
                f"### Topic search: {kw}\n"
                + "\n".join(lines)
            )
        searches_left -= 1

    if verbose:
        echo_stderr(
            f"  [day-recap] context sections={len(sections)} multi_day={multi_day} "
            f"timeline={len(summary.get('timeline') or [])} "
            f"key_texts={len(summary.get('key_texts') or [])}",        )
    return "\n\n".join(sections)


_APP_DISPLAY_NAMES: dict[str, str] = {
    "cursor.exe": "Cursor",
    "code.exe": "VS Code",
    "chrome.exe": "Chrome",
    "msedge.exe": "Edge",
    "firefox.exe": "Firefox",
    "explorer.exe": "File Explorer",
    "windowsterminal.exe": "Windows Terminal",
    "powershell.exe": "PowerShell",
    "cmd.exe": "Command Prompt",
    "teams.exe": "Microsoft Teams",
    "zoom.exe": "Zoom",
}


def _friendly_app_name(process_name: str) -> str:
    key = (process_name or "").strip().lower()
    return _APP_DISPLAY_NAMES.get(key, process_name or "Unknown")


_CODING_PROCS = frozenset({
    "cursor.exe", "code.exe", "devenv.exe", "windowsterminal.exe", "powershell.exe",
    "cmd.exe", "wt.exe", "idea64.exe", "pycharm64.exe",
})
_BROWSING_PROCS = frozenset({
    "chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe",
})
_MEETING_PROCS = frozenset({
    "teams.exe", "zoom.exe", "webex.exe", "slack.exe", "discord.exe",
})
_COMM_PROCS = frozenset({
    "outlook.exe", "thunderbird.exe", "mailbird.exe", "mailspring.exe",
})


def _app_category(process_name: str) -> str:
    key = (process_name or "").strip().lower()
    if key in _CODING_PROCS:
        return "coding"
    if key in _BROWSING_PROCS:
        return "browsing"
    if key in _MEETING_PROCS:
        return "meetings"
    if key in _COMM_PROCS:
        return "communication"
    if "word" in key or "excel" in key or "powerpnt" in key or "notepad" in key:
        return "writing"
    return "other"


def _format_time_breakdown_stats(summary: dict[str, Any]) -> str:
    """Deterministic minutes/percentages so the model cannot drop apps like Cursor."""
    apps = [
        a for a in (summary.get("apps") or [])
        if float(a.get("minutes") or 0) > 0
    ]
    if not apps:
        return "### Pre-computed totals\n(no app minutes in range — say so in the report)"

    total = sum(float(a.get("minutes") or 0) for a in apps)
    lines = [
        "### Pre-computed application minutes (copy these durations — do not invent or round to 0)",
        f"TOTAL tracked active time: {total:.1f} min",
        "",
    ]
    for a in sorted(apps, key=lambda x: float(x.get("minutes") or 0), reverse=True):
        raw = a.get("name") or "?"
        minutes = float(a.get("minutes") or 0)
        pct = round(100 * minutes / total, 1) if total else 0
        lines.append(
            f"- {_friendly_app_name(raw)} (`{raw}`): {minutes:.1f} min ({pct}%)"
        )

    cat_totals: dict[str, float] = {}
    for a in apps:
        cat = _app_category(a.get("name") or "")
        cat_totals[cat] = cat_totals.get(cat, 0.0) + float(a.get("minutes") or 0)
    lines.append("")
    lines.append("### Pre-computed category minutes (sum of apps above)")
    for cat, minutes in sorted(cat_totals.items(), key=lambda x: x[1], reverse=True):
        pct = round(100 * minutes / total, 1) if total else 0
        lines.append(f"- {cat}: {minutes:.1f} min ({pct}%)")

    focused = cat_totals.get("coding", 0) + cat_totals.get("writing", 0)
    unfocused = cat_totals.get("browsing", 0) + cat_totals.get("other", 0)
    score = round(100 * focused / total, 1) if total else 0
    lines.append("")
    lines.append(
        f"### Pre-computed productivity score\n"
        f"- focused (coding+writing): {focused:.1f} min\n"
        f"- unfocused (browsing+other): {unfocused:.1f} min\n"
        f"- score = focused/total: {score}%"
    )

    edited = summary.get("edited_files") or []
    if edited:
        lines.append("")
        lines.append("### Files touched (use for By Project)")
        for ef in edited[:20]:
            lines.append(f"- {ef.get('path', '')} ({ef.get('frame_count', 0)} captures)")

    return "\n".join(lines)


def _do_time_breakdown_prefetch(start: str, end: str, verbose: bool = False) -> str:
    """Rich activity-summary + deterministic minute tables for time-breakdown."""
    summary = _fetch_activity_summary(start, end, verbose=verbose, rich=True)
    if not summary:
        return "(Failed to fetch activity data from DeskMate API.)"

    sections = [
        _format_time_breakdown_stats(summary),
        format_summary_for_agent(summary),
    ]

    apps = sorted(
        summary.get("apps") or [],
        key=lambda a: float(a.get("minutes") or 0),
        reverse=True,
    )
    top_apps = [a.get("name", "") for a in apps[:3] if a.get("name") and float(a.get("minutes") or 0) > 0]
    if top_apps:
        if verbose:
            echo_stderr(f"  [time-breakdown] supplemental searches: {top_apps}")
        extra = _do_per_tool_searches(
            top_apps, start, end, limit_per_search=8, verbose=verbose,
        )
        if extra.strip():
            sections.append(
                "### Window/title detail (top apps by minutes)\n\n" + extra
            )

    if verbose:
        echo_stderr(
            f"  [time-breakdown] apps={len(apps)} timeline={len(summary.get('timeline') or [])}",        )
    return "\n\n".join(sections)


def _do_standup_prefetch(start: str, end: str, verbose: bool = False) -> str:
    """Same rich context as day-recap plus meetings for a concrete standup."""
    sections = [_do_day_recap_prefetch(start, end, verbose=verbose)]
    meeting_text, meeting_names = _do_meeting_todos_prefetch(start, end, verbose=verbose)
    if meeting_names:
        sections.append(f"### Meetings in range ({len(meeting_names)})\n\n{meeting_text}")
    if verbose:
        echo_stderr(
            f"  [standup-update] meetings={len(meeting_names)}",        )
    return "\n\n".join(sections)


def _format_habit_profiles_for_profile(verbose: bool = False) -> str:
    """Fetch learned habit profiles (作息规律) as a compact text block.

    The profile app uses this for the "工作习惯与节奏" dimension — it's the
    one signal that already encodes WHEN the user is active and with WHAT, so
    the model doesn't have to guess rhythm from raw frames.
    """
    try:
        payload = _http_get(f"{API_BASE}/habits/profile")
    except Exception as exc:  # noqa: BLE001
        if verbose:
            echo_stderr(f"  [user-profile] habits/profile error: {exc}")
        return ""
    rows = (payload.get("rows") if isinstance(payload, dict) else None) or []
    if not rows:
        return "### 作息规律 (habit profile)\n\n(暂无已学习的作息规律 — 录制天数可能还太少)"
    # Strongest routines first; cap so we don't blow the prompt budget.
    rows = sorted(rows, key=lambda r: float(r.get("frequency") or 0), reverse=True)[:24]

    def _slot_label(slot: int) -> str:
        h, half = divmod(int(slot), 2)
        return f"{h:02d}:{'30' if half else '00'}"

    lines = []
    for r in rows:
        lines.append(
            f"- {r.get('day_type', '?')} {_slot_label(r.get('slot') or 0)} · "
            f"{r.get('category', '?')} · 常用 {r.get('top_app') or '—'} · "
            f"均 {r.get('avg_minutes', 0)} 分钟 · 频率 {r.get('frequency', 0)} "
            f"({r.get('sample_days', 0)} 天)"
        )
    return "### 作息规律 (habit profile — when/what, already mined)\n\n" + "\n".join(lines)


def _do_user_profile_prefetch(start: str, end: str, verbose: bool = False) -> str:
    """Gather all four profile dimensions: behavior, rhythm, meetings, email.

    Behavioral activity (apps/files/sites/topics) comes from the same rich
    day-recap prefetch; rhythm from the mined habit profiles; collaboration from
    detected meetings plus connected mailboxes (best-effort — absent sources are
    simply omitted so the model states the gap rather than fabricating)."""
    sections = [_do_day_recap_prefetch(start, end, verbose=verbose)]

    habit_text = _format_habit_profiles_for_profile(verbose=verbose)
    if habit_text:
        sections.append(habit_text)

    meeting_text, meeting_names = _do_meeting_todos_prefetch(start, end, verbose=verbose)
    if meeting_names:
        sections.append(f"### Meetings in range ({len(meeting_names)})\n\n{meeting_text}")

    try:
        email_text, email_verified = _do_email_digest_prefetch(start, end, verbose=verbose)
    except Exception as exc:  # noqa: BLE001
        email_text, email_verified = "", []
        if verbose:
            echo_stderr(f"  [user-profile] email prefetch error: {exc}")
    if email_verified:
        sections.append(f"### Email activity ({', '.join(email_verified)})\n\n{email_text}")
    else:
        sections.append(
            "### Email activity\n\n(未连接 Gmail/Outlook，且未发现邮件客户端使用 — "
            "协作维度仅能依据会议与屏幕记录)"
        )

    if verbose:
        echo_stderr(
            f"  [user-profile] meetings={len(meeting_names)} "
            f"email_sources={len(email_verified)} habits={'y' if habit_text else 'n'}"
        )
    return "\n\n".join(sections)


# Lecture audio is the PRIMARY source for 讲解重点 / 理解要点, so it gets the
# largest single share of user-learning's 22k prompt budget, leaving the rest for
# sessions, courseware OCR, key texts and the pre-computed structure.
#
# The budget is in CHARACTERS. Lines are the wrong unit: Whisper emits ~14
# characters per row for Chinese and 3-4x that for English, so a line cap tuned
# on one language silently discards most of a lecture in the other. A measured
# 35-minute talk ran 630 rows / 8,698 chars — an earlier 260-line cap dropped 59%
# of it while using barely a third of the character budget.
#
# The line cap survives only as a runaway guard for pathological transcripts
# (thousands of near-empty rows); it should never be what binds.
_AUDIO_MAX_LINES = 2000
_AUDIO_MAX_CHARS = 13000


def _merge_manual_sessions(
    derived: list[dict[str, Any]],
    start: str,
    end: str,
    *,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Let user-declared sessions override the heuristics for their own span.

    ``build_learning_sessions`` re-derives sessions from the activity summary by
    running the same classifier the live detector uses. That is the right default
    for automatic detection, but it is wrong for a session the user started by
    hand: the whole declared span is study time by definition, including the
    parts where nothing on screen looked like studying (reading on paper,
    thinking, taking notes elsewhere). Re-deriving would report a fraction of it.

    Manual spans therefore replace any derived session falling inside them, and
    the derived ones outside are kept as-is.
    """
    try:
        from deskmate.learning_memory.store import LearningStore  # noqa: PLC0415

        manual = LearningStore().list_manual_sessions(start, end)
    except Exception as exc:  # noqa: BLE001
        if verbose:
            echo_stderr(f"  [user-learning] manual-session lookup failed: {exc}")
        return derived
    if not manual:
        return derived

    spans = [
        (str(m.get("started_at") or ""), str(m.get("ended_at") or "") or end)
        for m in manual
    ]

    def _inside(row: dict[str, Any]) -> bool:
        began = str(row.get("started_at") or "")
        return any(lo <= began <= hi for lo, hi in spans if lo)

    kept = [row for row in derived if not _inside(row)]
    absorbed = [row for row in derived if _inside(row)]
    for m in manual:
        # Say plainly that this span is the user's word, not an inference — the
        # report cites `why_learning`, and "the user marked it" is the strongest
        # reason available.
        m.setdefault("reason", "")
        m["reason"] = "user-declared study session (manual start/end)"
        # Inherit what the replaced spans observed. A manual session only
        # accumulates apps/urls while some automatic signal is also firing, so
        # one spent entirely on hands-on work would otherwise name no
        # application at all — and the courseware-OCR pass, which searches by
        # app, would have nothing to search.
        for field in ("apps", "urls", "queries", "topics", "concepts"):
            merged_vals = list(m.get(field) or [])
            for row in absorbed:
                for v in row.get(field) or []:
                    if v and v not in merged_vals:
                        merged_vals.append(v)
            m[field] = merged_vals
    if verbose:
        echo_stderr(
            f"  [user-learning] manual sessions: {len(manual)} "
            f"(replaced {len(derived) - len(kept)} derived)"
        )
    return manual + kept


def _full_transcript_text(start: str, end: str, *, verbose: bool = False) -> str:
    """The entire transcript for a window, merged into readable paragraphs.

    Unbudgeted on purpose. What reaches the prompt is bounded by what a local
    model can read in one call; what gets kept should be bounded by nothing, so
    the session stays fully searchable and quotable afterwards — and so a later
    pass can summarize it in chunks rather than from a sample.
    """
    try:
        from deskmate.db.manager import DatabaseManager  # noqa: PLC0415

        rows = DatabaseManager().transcripts_in_range(start, end)
    except Exception as exc:  # noqa: BLE001
        if verbose:
            echo_stderr(f"  [user-learning] full transcript read failed: {exc}")
        return ""

    def _fmt(ts: str, text: str) -> str | None:
        body = " ".join((text or "").split())
        return f"{(ts or '')[11:19]}  {body}".strip() if body else None

    return "\n\n".join(line for line, _ in merge_transcript_paragraphs(rows, _fmt))


def _collect_learning_audio_bits(
    start: str,
    end: str,
    summary: dict[str, Any],
    *,
    verbose: bool = False,
) -> tuple[list[str], dict[str, Any]]:
    """Pull lecture audio for the window, OLDEST FIRST and span-complete.

    Returns ``(lines, stats)``; ``stats`` feeds the coverage disclosure in
    ``format_learning_bundle`` so a partial transcript is never presented as a
    complete one.

    Why not ``/search``: that path is ``ORDER BY timestamp DESC LIMIT n``, so a
    capped audio query returns the *most recent* n utterances — for an 8-hour
    default window over a 1-hour class that is the tail of the class (or of
    whatever was spoken afterwards), not the lecture. Reading the transcript
    table directly in ascending time order gives the whole span in teaching
    order, which is what 讲解重点 and ``learning_lecture_items.ordinal`` need.
    The old search path stays as a fallback for callers without DB access.
    """
    bits: list[str] = []
    seen: set[str] = set()
    # total=0 means "unknown" (the /search fallback cannot count the window).
    stats: dict[str, Any] = {
        "total": 0, "included": 0, "truncated": False,
        "source": "db", "ordered": True,
    }

    def _fmt(ts: str, text: str, speaker: str = "") -> str | None:
        clean = " ".join((text or "").split())
        if len(clean) < 8:
            return None
        key = clean[:160].lower()
        if key in seen:
            return None
        seen.add(key)
        # Clock time only. A full ISO stamp costs 25 characters against a row of
        # speech averaging 14 — two thirds of the audio budget spent restating
        # the date. HH:MM:SS still locates the moment in the session.
        prefix = (ts or "")[11:19]
        if speaker:
            prefix = f"{prefix} [{speaker}]".strip()
        body = clean[:1200]
        return f"{prefix}: {body}".strip(": ").strip() if prefix else body

    rows: list[dict[str, Any]] = []
    try:
        from deskmate.db.manager import DatabaseManager  # noqa: PLC0415

        db = DatabaseManager()
        rows = db.transcripts_in_range(start, end)
        stats["total"] = db.count_transcripts_in_range(start, end)
    except Exception as exc:  # noqa: BLE001
        if verbose:
            echo_stderr(f"  [user-learning] transcript DB read failed ({exc}); using /search")
        rows = []

    if rows:
        paragraphs = merge_transcript_paragraphs(rows, _fmt)
        all_rows = sum(n for _, n in paragraphs)
        # Character budget decides; the line cap is only a runaway guard. Both
        # keep the span (even sampling), never the head or tail alone.
        kept = _select_spanning(paragraphs, _AUDIO_MAX_LINES)
        while kept and sum(len(x[0]) for x in kept) > _AUDIO_MAX_CHARS:
            kept = _select_spanning(kept, max(1, int(len(kept) * 0.8)))

        bits = [line for line, _ in kept]
        kept_rows = sum(n for _, n in kept)
        # Coverage is reported in transcript rows at both ends, so a lecture that
        # fits entirely reads as complete however it was paragraphed.
        stats["total"] = max(stats["total"], all_rows)
        stats["included"] = kept_rows
        stats["truncated"] = kept_rows < all_rows
        if verbose:
            echo_stderr(
                f"  [user-learning] audio: {kept_rows}/{stats['total']} rows in "
                f"{len(bits)} paragraphs (chronological, "
                f"{'sampled across span' if stats['truncated'] else 'complete'})"
            )
        return bits, stats

    # ── fallback: original /search + summary path ────────────────────────────
    # Reached only when the DB is unreachable. This path is recency-ordered and
    # cannot count the window, so it reports total=0 ("unknown") and
    # ordered=False rather than implying complete, in-order coverage.
    stats["source"] = "search"
    stats["ordered"] = False
    collected: list[tuple[str, str]] = []   # (sort_ts, line)

    for item in _do_content_search(
        start, end, limit=_AUDIO_MAX_LINES, content_type="audio", verbose=verbose,
    ):
        c = item.get("content") or {}
        ts = str(c.get("timestamp") or "")
        line = _fmt(
            ts,
            str(c.get("transcription") or c.get("text") or ""),
            str(c.get("speaker") or c.get("speaker_name") or ""),
        )
        if line:
            collected.append((ts, line))

    audio = summary.get("audio_summary") or {}
    for t in audio.get("top_transcriptions") or []:
        if isinstance(t, dict):
            ts = str(t.get("timestamp") or "")
            line = _fmt(
                ts,
                str(t.get("transcription") or t.get("text") or ""),
                str(t.get("speaker") or ""),
            )
        elif isinstance(t, str):
            ts, line = "", _fmt("", t)
        else:
            ts, line = "", None
        if line:
            collected.append((ts, line))

    for snip in summary.get("snippets") or []:
        if (snip.get("source") or "") != "audio":
            continue
        ts = str(snip.get("timestamp") or "")
        line = _fmt(ts, str(snip.get("text") or ""))
        if line:
            collected.append((ts, line))

    # Best-effort chronological order: timestamped lines first (ISO sorts
    # correctly), undated ones appended rather than interleaved arbitrarily.
    dated = sorted((c for c in collected if c[0]), key=lambda c: c[0])
    undated = [c for c in collected if not c[0]]
    bits = [line for _, line in dated] + [line for _, line in undated]
    bits = _select_spanning(bits, _AUDIO_MAX_LINES)
    stats["included"] = len(bits)
    stats["truncated"] = len(bits) >= _AUDIO_MAX_LINES
    return bits, stats


def _collect_courseware_ocr_lines(
    start: str,
    end: str,
    sessions: list[dict[str, Any]],
    *,
    verbose: bool = False,
) -> list[str]:
    """Extra OCR from courseware / browser study apps for slide text."""
    app_names: list[str] = []
    for s in sessions:
        if s.get("kind") not in {"courseware_view", "material_query", "study_other"}:
            continue
        for a in s.get("apps") or []:
            if a and a not in app_names:
                app_names.append(a)
    if not app_names:
        # Fall back to any session apps — still better than nothing for slides.
        for s in sessions:
            for a in s.get("apps") or []:
                if a and a not in app_names:
                    app_names.append(a)

    lines: list[str] = []
    # Backfilled/manual sessions may only carry a declared time span and title,
    # with no detected app metadata. Search the exact span without an app filter
    # so their screen OCR is not silently omitted from the recap.
    search_apps: list[str | None] = app_names[:4] or [None]
    for app in search_apps:
        search_limit = 12 if app else 35
        items = _do_content_search(
            start, end, limit=search_limit, app_name=app, content_type="ocr", verbose=verbose,
        )
        formatted = format_search_items(items, max_text=550)
        for fl in formatted:
            lines.append(fl)
        if len(lines) >= 35:
            break
    return lines[:35]


# Last enrichment payload from user-learning prefetch (for app.py sidecar).
G_LEARNING_ENRICHMENT: dict[str, Any] = {}


def _do_user_learning_prefetch(start: str, end: str, verbose: bool = False) -> str:
    """Detect learning sessions and slice screen/audio evidence to that subset.

    Pipeline:
      1) rich activity-summary for the window
      2) rule-based learning session merge (courseware / query / code / problem)
      3) heavy audio transcript pull (primary for 讲解重点)
      4) courseware OCR pull (slides / docs)
      5) learning-related key_texts + study files
      6) focused topic searches
    """
    # Request more snippets than day-recap so lecture audio/OCR survive capping.
    try:
        summary = _http_get(
            f"{API_BASE}/activity-summary"
            f"?start_time={quote(start)}&end_time={quote(end)}"
            f"&include_recording=true&include_snippets=true&include_guidance=true"
            f"&max_snippets=12&max_snippet_chars=1200&max_memories=8"
        )
    except Exception as exc:  # noqa: BLE001
        if verbose:
            echo_stderr(f"  [user-learning] activity-summary error: {exc}")
        summary = _fetch_activity_summary(start, end, verbose=verbose, rich=True)
    if not summary:
        return "(Failed to fetch activity data from DeskMate API.)"

    sessions = _merge_manual_sessions(build_learning_sessions(summary), start, end, verbose=verbose)
    # Spans the user declared by hand. Everything on screen inside one counts as
    # study evidence — the searches and experiments they ran against the lecture
    # are the point of the session, and the classifier alone would drop them.
    declared_spans = [
        (str(s.get("started_at") or ""), str(s.get("ended_at") or "") or end)
        for s in sessions
        if "user-declared" in str(s.get("reason") or "")
    ]
    key_texts = filter_learning_key_texts(
        summary.get("key_texts") or [], limit=80, declared_spans=declared_spans,
    )
    edited = filter_learning_edited_files(summary.get("edited_files") or [], limit=30)

    audio_bits: list[str] = []
    audio_stats: dict[str, Any] = {}
    courseware_ocr: list[str] = []
    if sessions:
        audio_bits, audio_stats = _collect_learning_audio_bits(
            start, end, summary, verbose=verbose,
        )
        courseware_ocr = _collect_courseware_ocr_lines(
            start, end, sessions, verbose=verbose,
        )

    bundle = format_learning_bundle(
        sessions=sessions,
        key_texts=key_texts,
        edited_files=edited,
        audio_bits=audio_bits,
        range_start=start,
        range_end=end,
        courseware_ocr_lines=courseware_ocr,
        audio_stats=audio_stats,
    )

    sections = [bundle]

    # Concept extraction + lecture structure + SM-2 queue (persisted locally).
    G_LEARNING_ENRICHMENT.clear()

    # Keep the COMPLETE transcript alongside the report, separate from the
    # trimmed copy that went into the prompt. A local model cannot read three
    # hours of speech in one call, but that is a limit on what can be summarized
    # — not a reason to lose the material.
    if audio_stats.get("total"):
        G_LEARNING_ENRICHMENT["full_transcript"] = {
            "range_start": start,
            "range_end": end,
            "rows": audio_stats.get("total", 0),
            "rows_in_prompt": audio_stats.get("included", 0),
            "text": _full_transcript_text(start, end, verbose=verbose),
        }
    if sessions:
        key_blobs = [
            str(row.get("text") or "")
            for row in key_texts
            if row.get("text")
        ]
        enrichment = build_learning_enrichment(
            audio_bits=audio_bits,
            courseware_ocr_lines=courseware_ocr,
            key_text_blobs=key_blobs,
            sessions=sessions,
            persist=True,
            verbose=verbose,
        )
        if enrichment.get("prompt_block"):
            sections.append(enrichment["prompt_block"])
        # Stash for callers that want the JSON sidecar (optional).
        G_LEARNING_ENRICHMENT.update(enrichment)

        queries: list[str] = []
        for s in sessions:
            for q in s.get("queries") or []:
                if q and q not in queries:
                    queries.append(q)
        for q in queries[:4]:
            items = _do_content_search(start, end, limit=10, q=q, verbose=verbose)
            lines = format_search_items(items, max_text=450)
            if lines:
                sections.append(f"### Topic search: {q}\n" + "\n".join(lines))

        # Compact contrast only (avoid drowning lecture evidence).
        apps = summary.get("apps") or []
        if apps:
            top = sorted(apps, key=lambda a: float(a.get("minutes") or 0), reverse=True)[:8]
            contrast = ["### Full-window top apps (contrast — NOT study evidence)"]
            for a in top:
                contrast.append(
                    f"- {a.get('name')}: {float(a.get('minutes') or 0):.1f} min"
                )
            sections.append("\n".join(contrast))

    if verbose:
        echo_stderr(
            f"  [user-learning] sessions={len(sessions)} "
            f"key_texts={len(key_texts)} edited={len(edited)} "
            f"audio_bits={len(audio_bits)} courseware_ocr={len(courseware_ocr)}"
        )
    return "\n\n".join(sections)


BROWSER_PROCS = frozenset({
    "chrome.exe", "msedge.exe", "brave.exe", "firefox.exe", "opera.exe",
})


def _keyword_in_item(item: dict[str, Any], kw: str) -> bool:
    """True if the tool name appears in the URL or window title (not raw OCR).

    Checking only the browser URL / window title avoids false positives from the
    tool name merely appearing in on-screen text (e.g. while editing this very
    report, or reading an article that mentions the tool). A real Gemini /
    Perplexity / ChatGPT session shows up as gemini.google.com / perplexity.ai /
    chatgpt.com in the URL and as the product name in the tab title.
    """
    c = item.get("content", {}) or {}
    blob = " ".join(
        str(c.get(k, "") or "")
        for k in ("browser_url", "window_name", "window_title")
    ).lower()
    return kw.lower() in blob


def _items_time_span(items: list[dict[str, Any]]) -> str:
    """Return 'first–last' clock-time span for a tool's hits, or ''."""
    stamps = sorted(
        s for s in ((it.get("content") or {}).get("timestamp", "") for it in items) if s
    )
    if not stamps:
        return ""
    first = format_ts_local(stamps[0])
    last = format_ts_local(stamps[-1])
    return f"{first}–{last}" if first != last else first


def _usage_intensity(hit_count: int) -> str:
    """Coarse AI-tool usage strength from substantive evidence count."""
    if hit_count <= 0:
        return "none"
    if hit_count <= 3:
        return "light"
    if hit_count <= 10:
        return "moderate"
    return "heavy"


def _do_ai_habits_prefetch(
    start: str,
    end: str,
    verbose: bool = False,
) -> tuple[str, list[str]]:
    """One /search per AI tool label; return formatted data + verified labels.

    "Verified" = labels with >=1 substantive hit after noise filtering. Dedicated
    apps (Cursor.exe, ChatGPT.exe, …) are verified by process name. Browser-hosted
    tools (Gemini, Perplexity, web ChatGPT/Claude) require the tool name to actually
    appear in screen/url text — generic Chrome/Edge activity does NOT count. The
    model is then told to list ONLY verified tools, so the small model can't
    fabricate usage for tools that were never opened.
    """
    sections: list[str] = []
    verified: list[str] = []
    for label, target in AI_TOOL_TARGETS.items():
        items: list[dict[str, Any]] = []
        tried: list[str] = []
        # 1. Dedicated (non-browser) process hits identify the tool directly.
        for proc in target["app_names"]:
            if proc.lower() in BROWSER_PROCS:
                continue
            tried.append(proc)
            items.extend(
                _do_content_search(start, end, limit=5, app_name=proc, verbose=verbose)
            )
        # 2. Keyword hits cover browser-hosted / web usage; require the tool name
        #    to literally appear so generic browsing isn't miscounted.
        kw = target.get("q_fallback")
        if kw:
            tried.append(f"q={kw}")
            raw = _do_content_search(start, end, limit=8, q=kw, verbose=verbose)
            items.extend(it for it in raw if _keyword_in_item(it, kw))
        lines = format_search_items(items, max_text=450)
        span = _items_time_span(items)
        if verbose:
            echo_stderr(
                f"  [ai-habits] {label}: {len(lines)} substantive "
                f"({span or 'no span'}) (tried {', '.join(tried)})",            )
        if lines:
            verified.append(label)
            hit_count = len(lines)
            header = (
                f"### {label} | substantive_hits={hit_count} "
                f"| usage_intensity={_usage_intensity(hit_count)}"
            )
            if span:
                header += f" | active window: {span}"
            sections.append(header + "\n" + "\n".join(lines))
        else:
            sections.append(
                f"### {label} | substantive_hits=0\nNO USAGE RECORDED for {label}.\n"
            )
    return "\n\n".join(sections), verified


def _days_since(since_iso: str | None) -> int:
    """Whole days from ``since_iso`` to now, floored at 1 (for Gmail newer_than)."""
    dt = _parse_iso_loose(since_iso) if since_iso else None
    if not dt:
        return 1
    delta = datetime.now().astimezone() - dt
    return max(1, int(delta.total_seconds() // 86400) + 1)


def _parse_msg_datetime(date_str: str | None) -> datetime | None:
    """Parse a message date from ISO 8601 (Graph) or RFC 2822 (Gmail)."""
    raw = (date_str or "").strip()
    if not raw or raw == "(no date)":
        return None
    dt = _parse_iso_loose(raw)
    if dt is None:
        try:
            dt = parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return dt


def _msg_date_within(
    date_str: str | None,
    since_iso: str | None,
    until_iso: str | None = None,
) -> bool:
    """True if ``date_str`` falls in [since_iso, until_iso] when bounds are set.

    Unparseable or missing dates are KEPT (return True) so a parsing quirk never
    silently drops a real message.
    """
    dt = _parse_msg_datetime(date_str)
    if dt is None:
        return True
    if since_iso:
        floor = _parse_iso_loose(since_iso)
        if floor and dt < floor:
            return False
    if until_iso:
        ceiling = _parse_iso_loose(until_iso)
        if ceiling and dt > ceiling:
            return False
    return True


def _fetch_outlook_oauth_messages(
    verbose: bool = False,
    *,
    since_iso: str | None = None,
    until_iso: str | None = None,
) -> tuple[str, bool]:
    """Fetch Microsoft Graph messages from connected Outlook accounts."""
    try:
        instances_payload = _http_get(f"{API_BASE}/connections/outlook/instances")
    except Exception as exc:
        if verbose:
            echo_stderr(f"  [email-digest] Outlook OAuth unavailable: {exc}")
        return "", False

    accounts = instances_payload.get("data", []) if isinstance(instances_payload, dict) else []
    sections: list[str] = []
    any_messages = False
    for account in accounts:
        if not isinstance(account, dict):
            continue
        instance = account.get("instance") or account.get("email")
        if not instance:
            continue
        try:
            payload = _http_get(
                f"{API_BASE}/connections/outlook/messages"
                f"?top=20&instance={quote(str(instance))}"
            )
        except Exception as exc:
            sections.append(
                f"### Outlook (OAuth) | account={instance} | graph_messages=0\n"
                f"Could not fetch Microsoft Graph messages: {exc}"
            )
            continue
        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        messages = data.get("messages", []) if isinstance(data, dict) else []
        lines: list[str] = []
        message_count = 0
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            date = msg.get("date") or "(no date)"
            if not _msg_date_within(date, since_iso, until_iso):
                continue
            message_count += 1
            subject = msg.get("subject") or "(no subject)"
            sender = msg.get("from") or "(unknown sender)"
            snippet = msg.get("snippet") or ""
            lines.append(f"- {date} | From: {sender} | Subject: {subject}")
            if snippet:
                lines.append(f"  Snippet: {snippet[:260]}")
        if lines:
            any_messages = True
            sections.append(
                f"### Outlook (OAuth) | account={instance} | graph_messages={message_count}\n"
                + "\n".join(lines)
            )
        else:
            sections.append(
                f"### Outlook (OAuth) | account={instance} | graph_messages=0\n"
                "NO GRAPH MAIL RECORDED for this account in the fetched window."
            )
    return "\n\n".join(sections), any_messages


def _fetch_gmail_oauth_messages(
    verbose: bool = False,
    *,
    since_iso: str | None = None,
    until_iso: str | None = None,
) -> tuple[str, bool]:
    """Fetch Gmail API messages from connected Gmail accounts."""
    try:
        instances_payload = _http_get(f"{API_BASE}/connections/gmail/instances")
    except Exception as exc:
        if verbose:
            echo_stderr(f"  [email-digest] Gmail OAuth unavailable: {exc}")
        return "", False

    accounts = instances_payload.get("data", []) if isinstance(instances_payload, dict) else []
    sections: list[str] = []
    any_messages = False
    # Constrain the Gmail query to the activity window so a stale inbox does not
    # surface weeks-old mail as if it were today's activity.
    date_filter = ""
    if since_iso:
        date_filter = f"&q={quote(f'newer_than:{_days_since(since_iso)}d')}"
    for account in accounts:
        if not isinstance(account, dict):
            continue
        instance = account.get("instance") or account.get("email")
        if not instance:
            continue
        try:
            listed = _http_get(
                f"{API_BASE}/connections/gmail/messages"
                f"?maxResults=10&instance={quote(str(instance))}{date_filter}"
            )
        except Exception as exc:
            sections.append(
                f"### Gmail (OAuth) | account={instance} | api_messages=0\n"
                f"Could not fetch Gmail messages: {exc}"
            )
            continue
        data = listed.get("data", {}) if isinstance(listed, dict) else {}
        message_refs = data.get("messages", []) if isinstance(data, dict) else []
        lines: list[str] = []
        message_count = 0
        for ref in message_refs[:10]:
            if not isinstance(ref, dict) or not ref.get("id"):
                continue
            try:
                detail = _http_get(
                    f"{API_BASE}/connections/gmail/messages/{quote(str(ref['id']))}"
                    f"?instance={quote(str(instance))}"
                )
            except Exception as exc:
                lines.append(f"- message_id={ref['id']} could not be read: {exc}")
                continue
            msg = detail.get("data", {}) if isinstance(detail, dict) else {}
            if not isinstance(msg, dict):
                continue
            date = msg.get("date") or "(no date)"
            if not _msg_date_within(date, since_iso, until_iso):
                continue
            message_count += 1
            subject = msg.get("subject") or "(no subject)"
            sender = msg.get("from") or "(unknown sender)"
            snippet = msg.get("snippet") or ""
            lines.append(f"- {date} | From: {sender} | Subject: {subject}")
            if snippet:
                lines.append(f"  Snippet: {snippet[:260]}")
        if message_count:
            any_messages = True
            sections.append(
                f"### Gmail (OAuth) | account={instance} | api_messages={message_count}\n"
                + "\n".join(lines)
            )
        else:
            sections.append(
                f"### Gmail (OAuth) | account={instance} | api_messages=0\n"
                "NO GMAIL API MAIL RECORDED for this account in the fetched window."
            )
    return "\n\n".join(sections), any_messages


def _cap_prefetch_text(text: str, max_chars: int = 12000) -> str:
    """Keep single-shot LLM prompts within a size local models can finish in time.

    When truncation happens, a WATERMARK is prepended at the TOP (not just the
    cut tail) so the model always sees that some evidence was dropped — otherwise
    it reads a partial bundle as complete and infers the missing days/sections
    were "idle" rather than "not shown" (optimization note 9.6).
    """
    if len(text) <= max_chars:
        return text
    dropped = len(text) - max_chars
    watermark = (
        "### ⚠️ EVIDENCE TRUNCATED\n"
        f"~{dropped} characters of evidence below were dropped to fit the model "
        "budget. Sections are ordered most-recent-first, so the OLDEST data was "
        "cut. Do NOT infer that any day or topic was idle merely because it is "
        "missing here — say coverage was truncated instead.\n\n"
    )
    budget = max_chars - len(watermark)
    if budget <= 0:
        return watermark
    return watermark + text[:budget] + "\n\n...(prefetch truncated for model time budget)"


def _do_email_digest_prefetch(
    start: str,
    end: str,
    verbose: bool = False,
    *,
    limit_per_tool: int = 5,
) -> tuple[str, list[str]]:
    """Per email-tool search; mirror of `_do_ai_habits_prefetch` for email apps.

    Dedicated mail clients (Outlook.exe, Thunderbird.exe, …) are matched by
    process name. Webmail tabs (Gmail / Outlook web / 163 / QQ / Yahoo …) only
    count when the mail host literally appears in the browser URL or window
    title — generic Chrome / Edge activity does NOT count as email usage, so
    the small model can't fabricate inbox time on top of plain browsing.
    """
    sections: list[str] = []
    verified: list[str] = []
    gmail_oauth_text, has_gmail_oauth_messages = _fetch_gmail_oauth_messages(
        verbose=verbose, since_iso=start, until_iso=end,
    )
    if gmail_oauth_text:
        sections.append(gmail_oauth_text)
    if has_gmail_oauth_messages:
        verified.append("Gmail (OAuth)")
    oauth_text, has_oauth_messages = _fetch_outlook_oauth_messages(
        verbose=verbose, since_iso=start, until_iso=end,
    )
    if oauth_text:
        sections.append(oauth_text)
    if has_oauth_messages:
        verified.append("Outlook (OAuth)")
    for label, target in EMAIL_TOOL_TARGETS.items():
        items: list[dict[str, Any]] = []
        tried: list[str] = []
        for proc in target["app_names"]:
            if proc.lower() in BROWSER_PROCS:
                continue
            tried.append(proc)
            items.extend(
                _do_content_search(
                    start, end, limit=limit_per_tool, app_name=proc, verbose=verbose,
                )
            )
        kw = target.get("q_fallback")
        if kw:
            tried.append(f"q={kw}")
            kw_limit = min(8, max(limit_per_tool + 2, limit_per_tool))
            raw = _do_content_search(start, end, limit=kw_limit, q=kw, verbose=verbose)
            items.extend(it for it in raw if _keyword_in_item(it, kw))
        lines = format_search_items(items, max_text=350 if limit_per_tool <= 3 else 450)
        span = _items_time_span(items)
        if verbose:
            echo_stderr(
                f"  [email-digest] {label}: {len(lines)} substantive "
                f"({span or 'no span'}) (tried {', '.join(tried)})",            )
        if lines:
            verified.append(label)
            header = f"### {label} | substantive_hits={len(lines)}"
            if span:
                header += f" | active window: {span}"
            sections.append(header + "\n" + "\n".join(lines))
        else:
            sections.append(
                f"### {label} | substantive_hits=0\nNO USAGE RECORDED for {label}.\n"
            )
    return "\n\n".join(sections), verified


# ── meeting summary ───────────────────────────────────────────────────────

_GENERIC_MEETING_TITLES = frozenset({
    "", "untitled", "untitled meeting", "meeting", "manual meeting",
    "zoom", "teams", "meet", "google meet", "new meeting",
    "standup", "daily standup", "weekly standup", "scrum",
    "1:1", "1-1", "one-on-one", "one on one",
    "sync", "weekly sync", "team sync", "planning", "retro",
    "retrospective", "checkpoint",
})


def _is_generic_meeting_title(title: str) -> bool:
    """True for auto/default meeting titles safe to replace with a summary title."""
    normalized = " ".join((title or "").strip().lower().split())
    return normalized in _GENERIC_MEETING_TITLES


def _fetch_latest_meeting(verbose: bool = False) -> dict[str, Any] | None:
    try:
        meetings = _http_get(f"{API_BASE}/meetings?limit=1")
    except Exception as exc:
        if verbose:
            echo_stderr(f"  [meeting] list error: {exc}")
        return None
    if isinstance(meetings, list) and meetings:
        return meetings[0]
    return None


def _fetch_meeting_by_id(meeting_id: int, verbose: bool = False) -> dict[str, Any] | None:
    try:
        data = _http_get(f"{API_BASE}/meetings/{meeting_id}")
    except Exception as exc:
        if verbose:
            echo_stderr(f"  [meeting] fetch id={meeting_id} error: {exc}")
        return None
    return data.get("meeting") if isinstance(data, dict) else None


def _fetch_meeting_transcript(meeting_id: int, verbose: bool = False) -> tuple[str, list[dict[str, Any]]]:
    try:
        data = _http_get(f"{API_BASE}/meetings/{meeting_id}/transcript")
    except Exception as exc:
        if verbose:
            echo_stderr(f"  [meeting] transcript error: {exc}")
        return "", []
    segments = data.get("segments") or [] if isinstance(data, dict) else []
    text = data.get("text", "") if isinstance(data, dict) else ""
    return text, segments


def _format_meeting_segments(segments: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        speaker = seg.get("speaker_name")
        if not speaker:
            sid = seg.get("speaker_id")
            speaker = f"Speaker {sid}" if sid is not None else "Unknown"
        lines.append(f"- {speaker}: {text}")
    return "\n".join(lines)


def _parse_iso_loose(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return dt


def _meeting_in_range(meeting: dict[str, Any], start_iso: str, end_iso: str) -> bool:
    """True if the meeting interval overlaps [start_iso, end_iso]."""
    m_start = _parse_iso_loose(meeting.get("started_at"))
    if not m_start:
        return False
    m_end = _parse_iso_loose(meeting.get("ended_at")) or datetime.now(m_start.tzinfo)
    rs = _parse_iso_loose(start_iso)
    re_ = _parse_iso_loose(end_iso)
    if rs and m_end <= rs:
        return False
    if re_ and m_start >= re_:
        return False
    return True


# ── Screen evidence for todos (OCR / chat / notes) — heavily guarded ─────────
#
# OCR is noisy: a page title, an article, code you're reading, or a menu label
# can all look task-ish to a model. To avoid polluting the todolist we do NOT
# hand the raw screen dump to the LLM. Instead we pre-filter to lines that match
# an EXPLICIT task shape, and hard-drop anything matching a noise pattern. The
# LLM then makes the final call on this already-narrow candidate set, with a
# prompt rule telling it to drop anything that isn't a personal task.

# Screen todos are gated in THREE tiers by ownership/direction signal, because a
# task-*shaped* line is not necessarily the USER's task (optimization note 7.8):
#
#   1. SELF markers — a TODO/FIXME note or a checklist item. These are
#      self-authored, so they're owned by the user — BUT only when NOT in a
#      browser (a `TODO:` seen on GitHub / Stack Overflow is third-party code
#      the user is merely reading, not their task).
#   2. DIRECTED asks — phrasing that is unambiguously aimed at the user
#      ("can you …", "please review …", "需要你 …", "帮我 …", "@you …"). Always
#      counts: the second person is baked into the pattern.
#   3. AMBIGUOUS directives — "请 …", "麻烦 …", "记得 …", "别忘了 …", "跟进 …".
#      A group message "请更新文档" addressed to someone ELSE matches these, so
#      they only count when a second-person / @-mention signal co-occurs on the
#      line (proving it's directed at the user).
#
# A bare deadline ("due today", "截止…") is NOT sufficient on its own anymore —
# a date on screen doesn't mean the user owns the task.
_SCREEN_SELF_PATTERNS = (
    re.compile(r"\b(todo|fixme|to-do|action item|action required)\b[:：]", re.I),
    re.compile(r"^\s*[-*]?\s*\[\s*[ xX]?\s*\]\s+\S"),          # checkbox line
    re.compile(r"^\s*(todo|fixme)\b", re.I),                    # leading TODO
)
_SCREEN_DIRECTED_PATTERNS = (
    re.compile(r"\b(can|could|would)\s+you\s+(please\s+)?\w+", re.I),
    re.compile(r"\bplease\s+(review|send|check|confirm|update|fix|finish|reply|sign)\b", re.I),
    re.compile(r"(需要你|帮我|请你|麻烦你)\s*\S"),
)
_SCREEN_AMBIGUOUS_PATTERNS = (
    re.compile(r"(请|麻烦|记得|别忘了|待办|跟进)\s*\S"),
)
# Second-person / @-mention signal that promotes an AMBIGUOUS directive to a
# user-owned task. Bare ASCII "u" is intentionally excluded (too noisy).
_SECOND_PERSON_RE = re.compile(r"(@\w+|\byou\b|\byour\b|\byourself\b|你|您)", re.I)
# Lines matching any of these are NEVER candidates — typical OCR noise that
# super­ficially resembles a task (code, search boxes, nav, marketing, etc.).
_SCREEN_NOISE_PATTERNS = (
    re.compile(r"^\s*(def|class|function|import|from|return|const|let|var|if|for|while|#include)\b"),
    re.compile(r"[{}<>;]\s*$"),                                  # code-ish line ends
    re.compile(r"\b(stack overflow|github|documentation|search results?|how to)\b", re.I),
    re.compile(r"^\s*(home|file|edit|view|settings?|menu|sign in|log ?in|subscribe|cookie)\b", re.I),
    re.compile(r"https?://\S+\s*$"),                             # a bare URL
    re.compile(r"^\s*\d+\s*(comments?|likes?|views?|赞|评论|播放)\b", re.I),
)
# Apps whose content is almost always personal tasks/chat. Used to LABEL the
# source app cleanly; not a filter (other apps can still surface a TODO: line).
_SCREEN_CHAT_APPS = (
    "wechat", "weixin", "slack", "teams", "telegram", "discord", "qq",
    "feishu", "lark", "dingtalk", "钉钉", "飞书", "微信",
)


def _looks_like_screen_task(text: str, *, is_browser: bool = False) -> bool:
    """True only when a screen line is, with high confidence, a task the USER
    owns — not merely task-shaped. See the tier comment above. ``is_browser``
    marks lines captured from a browser, where TODO/FIXME notes are almost
    always third-party code being read rather than the user's own work."""
    line = text.strip()
    if len(line) < 6 or len(line) > 240:
        return False
    if any(p.search(line) for p in _SCREEN_NOISE_PATTERNS):
        return False
    # Tier 1: self-authored note / checklist — owned, unless read in a browser.
    if any(p.search(line) for p in _SCREEN_SELF_PATTERNS):
        return not is_browser
    # Tier 2: phrasing explicitly aimed at the user.
    if any(p.search(line) for p in _SCREEN_DIRECTED_PATTERNS):
        return True
    # Tier 3: ambiguous directive — only if it's actually addressed to the user.
    if any(p.search(line) for p in _SCREEN_AMBIGUOUS_PATTERNS):
        return bool(_SECOND_PERSON_RE.search(line))
    return False


def _do_screen_todos_prefetch(
    start_iso: str, end_iso: str, verbose: bool = False, *, max_candidates: int = 25,
) -> tuple[str, list[str]]:
    """Surface EXPLICIT on-screen tasks (OCR / chat / notes) as todo candidates.

    Returns (data_text, app_names). app_names lists the apps that contributed at
    least one candidate line — empty when nothing passed the strict filter, in
    which case the LLM is told to emit no screen todos. We deliberately only
    forward lines that already look like a task; the model still does the final
    judgement, so this is a two-stage guard against OCR false positives.
    """
    try:
        items = _do_content_search(start_iso, end_iso, limit=120, verbose=verbose)
    except Exception as exc:  # noqa: BLE001
        if verbose:
            echo_stderr(f"  [todo-list] screen search error: {exc}")
        return "(no screen data — /search unavailable)", []

    candidates: list[str] = []
    seen: set[str] = set()
    apps: list[str] = []
    for item in items:
        c = item.get("content") or {}
        # OCR text or UI typed/clipboard text — both can carry an explicit task.
        raw = normalize_capture_text(c.get("text") or c.get("text_content") or "")
        if not raw:
            continue
        app = (c.get("app_name") or "").strip()
        is_browser = app.lower() in BROWSER_PROCS
        ts = format_ts_local(c.get("timestamp") or "")
        for ln in raw.splitlines():
            if not _looks_like_screen_task(ln, is_browser=is_browser):
                continue
            key = ln.strip().lower()[:80]
            if key in seen:
                continue
            seen.add(key)
            app_label = app or "screen"
            if app_label not in apps:
                apps.append(app_label)
            candidates.append(f"- [{ts}] {app_label}: {ln.strip()[:200]}")
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break

    if verbose:
        echo_stderr(f"  [todo-list] screen task candidates={len(candidates)} from {len(apps)} app(s)")
    if not candidates:
        return "(no explicit on-screen tasks detected in this time range)", []
    header = (
        "Candidate task lines extracted from screen content (each already matched "
        "a task pattern; confirm each is a real personal task before including):"
    )
    return header + "\n" + "\n".join(candidates), apps


def _do_meeting_todos_prefetch(
    start_iso: str, end_iso: str, verbose: bool = False,
) -> tuple[str, list[str]]:
    """Fetch meetings in range + their transcripts, formatted for the LLM.

    Returns (data_text, meeting_names). meeting_names lists meetings that
    actually had a transcript (i.e. could yield spoken action items).
    """
    try:
        meetings = _http_get(f"{API_BASE}/meetings?limit=20")
    except Exception as exc:
        if verbose:
            echo_stderr(f"  [todo-list] meetings list error: {exc}")
        return "(no meeting data — /meetings unavailable)", []

    if not isinstance(meetings, list):
        meetings = []
    in_range = [m for m in meetings if isinstance(m, dict) and _meeting_in_range(m, start_iso, end_iso)]
    if not in_range:
        return "(no video meetings detected in this time range)", []

    blocks: list[str] = []
    with_transcript: list[str] = []
    for m in in_range:
        mid = m.get("id")
        name = (m.get("name") or "Meeting").strip()
        start = m.get("started_at") or ""
        end = m.get("ended_at") or "(ongoing)"
        transcript_text, segments = _fetch_meeting_transcript(int(mid), verbose=verbose) if mid is not None else ("", [])
        # Keep only the tail of long transcripts so todo extraction stays fast.
        if len(segments) > 60:
            segments = segments[-60:]
        formatted = _format_meeting_segments(segments)
        if len(formatted) > 4000:
            formatted = formatted[-4000:]
            formatted = "...(transcript tail)\n" + formatted
        header = f"### Meeting: {name} (id={mid}) — {start} to {end}"
        if formatted.strip():
            with_transcript.append(name)
            blocks.append(f"{header}\n{formatted}")
        else:
            blocks.append(f"{header}\n- (no transcript captured for this meeting)")

    if verbose:
        echo_stderr(
            f"  [todo-list] meetings in range={len(in_range)}, "
            f"with transcript={len(with_transcript)}",        )
    return "\n\n".join(blocks), with_transcript


def _do_meeting_summary(
    *,
    skill_text: str,
    context_header: str,
    pipe_body: str,
    meeting_id: int | None = None,
    verbose: bool = False,
) -> str:
    """Summarize a meeting transcript and PATCH the summary back onto it.

    When ``meeting_id`` is given (e.g. the Meetings page "Summarize" button)
    that exact meeting is used; otherwise the most recent meeting is picked
    (the auto ``meeting_ended`` flow). Python performs the find / transcript /
    PATCH calls (the local model can't run curl), the LLM only writes the
    summary. Skips the PATCH when there is nothing worth saving.
    """
    if meeting_id is not None:
        meeting = _fetch_meeting_by_id(int(meeting_id), verbose=verbose)
    else:
        meeting = _fetch_latest_meeting(verbose=verbose)
    if not meeting:
        return "No meeting found to summarize. Start a meeting first, then run this app."

    meeting_id = int(meeting.get("id"))
    existing_note = (meeting.get("note") or "").strip()
    current_title = (meeting.get("name") or "").strip()
    start = meeting.get("started_at") or ""
    end = meeting.get("ended_at") or ""

    if verbose:
        echo_stderr(
            f"  [meeting] id={meeting_id} title={current_title!r} "
            f"start={start} end={end}",        )

    transcript_text, segments = _fetch_meeting_transcript(meeting_id, verbose=verbose)
    transcript = _format_meeting_segments(segments)

    if not transcript.strip() and start:
        # Fallback: search audio in the meeting window.
        if verbose:
            echo_stderr("  [meeting] empty transcript → audio search fallback")
        items = _do_content_search(
            start, end or datetime.now().astimezone().isoformat(),
            limit=40, verbose=verbose,
        )
        audio_lines = [
            line for line in format_search_items(items, max_text=450) if "AUDIO" in line
        ]
        transcript = "\n".join(audio_lines)

    if not transcript.strip():
        return (
            f"Meeting #{meeting_id} (\"{current_title or 'untitled'}\") has no transcript "
            f"to summarize — empty or no relevant audio captured. Skipped writing a summary."
        )

    system_prompt = (
        "You are a DeskMate meeting summarizer.\n\n"
        "RULES:\n"
        "1. Summarize ONLY from the transcript provided. NEVER invent decisions, names, or action items.\n"
        "2. Output a 5-8 word plain-english TITLE on the first line, prefixed exactly with 'TITLE: '.\n"
        "   No quotes, no 'meeting about' prefix.\n"
        "3. Then a blank line, then the summary body starting with '## Summary'.\n"
        "4. Cover: key topics and decisions (use bullet lists).\n"
        "5. Then a '## Action Items' section. Put EACH action item on its own line in EXACTLY "
        "this format (machine-parsed — keep the `|` separators and the `key:` labels):\n"
        "   - [ ] <task> | owner: <name or unknown> | due: <YYYY-MM-DD or none> | priority: <high|medium|low>\n"
        "   Set owner to the speaker who committed to the task (from `speaker_name` in the "
        "transcript) when clear, else 'unknown'. Infer priority from the transcript's urgency "
        "cues ('urgent/ASAP/今天' → high; a future date or 'this week' → medium; else low). "
        "Only list action items explicitly stated or clearly implied in the transcript. "
        "If there are none, write the single line 'NONE' under the heading.\n"
        "6. Be concise. If something is unclear in the transcript, omit it.\n\n"
        f"{skill_text}\n"
    )
    user_prompt = (
        f"{context_header}\n"
        f"## Meeting\n"
        f"- id: {meeting_id}\n"
        f"- current title: {current_title or '(none)'}\n"
        f"- time: {start} to {end or '(ongoing)'}\n\n"
        f"## Transcript\n{transcript}\n\n"
        f"---\n\n## Instructions\n\n{pipe_body}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = chat_ollama(messages, tools=None, num_predict=4096)
    content = strip_thinking(response.get("content", ""))

    new_title, summary_body = _split_title_and_summary(content)

    # Decide whether to refresh the title.
    title_to_set: str | None = None
    if new_title and _is_generic_meeting_title(current_title):
        title_to_set = new_title

    # Append the summary to the existing note under a "## Summary" heading.
    if "## Summary" in summary_body:
        summary_section = summary_body
    else:
        summary_section = f"## Summary\n{summary_body}"
    new_note = (existing_note + "\n\n" + summary_section).strip() if existing_note else summary_section

    patch_body: dict[str, Any] = {"note": new_note}
    if title_to_set:
        patch_body["name"] = title_to_set
    try:
        _http_patch(f"{API_BASE}/meetings/{meeting_id}", patch_body)
        if verbose:
            echo_stderr(
                f"  [meeting] patched #{meeting_id} "
                f"(title={'set' if title_to_set else 'kept'})",            )
        status = (
            f"_Saved to meeting #{meeting_id}"
            + (f" — title updated to \"{title_to_set}\"" if title_to_set else "")
            + "._"
        )
    except Exception as exc:
        if verbose:
            echo_stderr(f"  [meeting] patch error: {exc}")
        status = f"_(Could not write back to meeting #{meeting_id}: {exc})_"

    # Extract structured action items into the todos table (deduped per meeting).
    todo_count = 0
    try:
        action_items = _parse_action_items(summary_body)
        todo_count = _write_meeting_todos(
            meeting_id, title_to_set or current_title, action_items, verbose=verbose
        )
    except Exception as exc:  # noqa: BLE001
        if verbose:
            echo_stderr(f"  [meeting] action-item extraction error: {exc}")

    todo_note = f" · {todo_count} todo(s) extracted" if todo_count else ""
    return f"{summary_section}\n\n{status}{todo_note}"


# Order-independent, key-anchored field extraction. The task is everything up to
# the first ` | key:` segment; owner/due/priority are matched wherever they
# appear. This mirrors the todo-list app's pipe-delimited contract so the two
# todo-producing paths share one format (and one mental model).
_ACTION_TASK_RE = re.compile(r"^\s*[-*]\s*\[[ xX✓✔]?\]\s*(?P<task>.+?)\s*(?:\||$)")


def _action_field(line: str, key: str) -> str:
    """Pull `| key: value` from an action-item line, regardless of field order."""
    m = re.search(rf"\|\s*{key}\s*:\s*(?P<v>[^|]*)", line, re.I)
    return (m.group("v").strip() if m else "")


# Priority words → canonical H/M/L (shared intent with todo-list's _PRIORITY_MAP).
_ACTION_PRIORITY_MAP = {
    "high": "H", "medium": "M", "low": "L", "h": "H", "m": "M", "l": "L",
    "urgent": "H", "critical": "H", "normal": "M",
    "紧急": "H", "高": "H", "中": "M", "低": "L",
}


def _parse_action_items(summary_body: str) -> list[dict[str, str]]:
    """Extract `- [ ] task | owner: X | due: Y | priority: Z` lines from the
    Action Items section into structured dicts. Fields are key-anchored so order
    doesn't matter and a missing field never shifts the others. Returns [] when
    the section is NONE/empty."""
    items: list[dict[str, str]] = []
    in_section = False
    for line in summary_body.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("## action item"):
            in_section = True
            continue
        if in_section and stripped.startswith("## "):
            break  # next section
        if not in_section or not stripped or stripped.upper() == "NONE":
            continue
        m = _ACTION_TASK_RE.match(line)
        if not m:
            continue
        task = (m.group("task") or "").strip()
        if not task:
            continue
        owner = _action_field(line, "owner")
        due = _action_field(line, "due")
        if due.lower() in ("none", "n/a", ""):
            due = ""
        priority = _ACTION_PRIORITY_MAP.get(_action_field(line, "priority").lower(), "")
        items.append({"task": task, "owner": owner, "due": due, "priority": priority})
    return items


def _write_meeting_todos(
    meeting_id: int,
    meeting_name: str,
    items: list[dict[str, str]],
    *,
    verbose: bool = False,
) -> int:
    """Upsert extracted action items into the todos table via POST /todos.

    Deduped per meeting by a stable key so re-running the summary doesn't create
    duplicates. Returns the number of todos written."""
    if not items:
        return 0
    payload = []
    for it in items:
        task = it["task"]
        owner = (it.get("owner") or "").strip()
        has_owner = bool(owner) and owner.lower() != "unknown"
        # Owner (the responsible person) goes in source_ref — same slot the
        # todo-list path uses for an email sender — so both paths store the
        # "who" consistently instead of one jamming it into the task text. Fall
        # back to the meeting name when no owner was attributed.
        source_ref = owner if has_owner else (meeting_name or f"meeting #{meeting_id}")
        payload.append({
            "text": task,
            "source": "meeting",
            "source_ref": source_ref,
            "source_detail": f"meeting:{meeting_name}" if meeting_name else "meeting",
            "meeting_id": meeting_id,
            "priority": it.get("priority") or "",
            "due": it.get("due") or "",
            "origin_app": "meeting-summary",
            # Stable per-meeting+task key so repeated summaries upsert, not dup.
            "dedup_key": f"meeting:{meeting_id}:{task.lower()[:80]}",
        })
    try:
        resp = _http_post(f"{API_BASE}/todos", {"todos": payload})
        count = int(resp.get("count") or 0) if isinstance(resp, dict) else 0
        if verbose:
            echo_stderr(f"  [meeting] wrote {count} action item(s) to todos")
        return count
    except Exception as exc:  # noqa: BLE001
        if verbose:
            echo_stderr(f"  [meeting] todo write failed: {exc}")
        return 0


def _split_title_and_summary(content: str) -> tuple[str, str]:
    """Parse the model output 'TITLE: ...\\n\\n## Summary ...' into parts."""
    title = ""
    body = content.strip()
    lines = body.splitlines()
    if lines and lines[0].strip().upper().startswith("TITLE:"):
        title = lines[0].split(":", 1)[1].strip().strip('"').strip()
        body = "\n".join(lines[1:]).strip()
    return title, body


# ── context header ────────────────────────────────────────────────────────

def build_context_header(hours: float) -> tuple[str, str, str]:
    now = datetime.now().astimezone()
    start = now - timedelta(hours=hours)
    return _context_header(now, start.isoformat(), now.isoformat())


def build_context_header_range(start_time: str, end_time: str) -> tuple[str, str, str]:
    """Build context header from explicit ISO 8601 bounds."""
    now = datetime.now().astimezone()
    return _context_header(now, start_time, end_time)


def _context_header(now: datetime, start_iso: str, end_iso: str) -> tuple[str, str, str]:
    header = (
        f"## Context\n"
        f"- Current time: {now.isoformat()}\n"
        f"- Timezone: {now.tzname()}\n"
        f"- Time range: {start_iso} to {end_iso}\n"
        f"- API base: {API_BASE}\n"
    )
    return header, start_iso, end_iso


# ── pipe.md parser ────────────────────────────────────────────────────────

def parse_pipe_md(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    fm_match = _FRONTMATTER_RE.match(text)
    if not fm_match:
        return {}, text
    try:
        import yaml  # type: ignore[import-not-found]
        fm = yaml.safe_load(fm_match.group(1)) or {}
    except Exception:
        fm = {}
    body = text[fm_match.end():]
    return fm, body


strip_thinking = llm.strip_thinking


# ── Ollama chat ───────────────────────────────────────────────────────────

def chat_ollama(
    messages: list[dict],
    tools: list[dict] | None = None,
    *,
    num_predict: int = 4096,
    timeout: int | None = None,
) -> dict:
    # Reads module-level OLLAMA_MODEL at call time so apps can override it
    # via ``agent.OLLAMA_MODEL = args.model`` before running.
    return llm.chat_ollama(
        messages,
        tools,
        base=OLLAMA_BASE,
        model=OLLAMA_MODEL,
        num_predict=num_predict,
        timeout=timeout,
    )


# ── main agent loop ───────────────────────────────────────────────────────

def run_agent(
    pipe_md_path: Path,
    *,
    hours: float | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
    meeting_id: int | None = None,
    verbose: bool = False,
) -> str:
    """Run one pipe through the full agent loop.

    Flow:
    1. Read pipe.md to understand what to do
    2. Parse search instructions from pipe.md
    3. Execute those searches against the local API
    4. Send search results + pipe.md template to LLM
    5. LLM writes the report based on real data
    6. If LLM calls additional tools, execute and loop
    """

    skill_text = SKILL_PATH.read_text(encoding="utf-8") if SKILL_PATH.exists() else ""
    _, pipe_body = parse_pipe_md(pipe_md_path)
    if start_time and end_time:
        context_header, start_iso, end_iso = build_context_header_range(start_time, end_time)
    elif hours is not None:
        context_header, start_iso, end_iso = build_context_header(hours)
    else:
        raise ValueError("run_agent requires hours or both start_time and end_time")

    # Step 1: Parse pipe.md for specific instructions
    per_tool_names = _extract_per_tool_names(pipe_body)
    has_frames_export = bool(_FRAMES_EXPORT_RE.search(pipe_body))

    # Step 2: tool-driven pipes (ai-habits) — model calls APIs in a loop
    if pipe_md_path.parent.name in TOOL_DRIVEN_PIPES:
        return _run_tool_driven_agent(
            pipe_md_path,
            pipe_body=pipe_body,
            skill_text=skill_text,
            context_header=context_header,
            start_iso=start_iso,
            end_iso=end_iso,
            verbose=verbose,
        )

    # Meeting summary: find latest meeting, summarize transcript, PATCH it back.
    if pipe_md_path.parent.name == "meeting-summary":
        if verbose:
            echo_stderr("  [agent] meeting-summary → find + summarize + patch")
        return _do_meeting_summary(
            skill_text=skill_text,
            context_header=context_header,
            pipe_body=pipe_body,
            meeting_id=meeting_id,
            verbose=verbose,
        )

    # AI habits: deterministic per-tool prefetch + single-shot with an allowlist
    # so the small model can't fabricate usage for tools that were never opened.
    if pipe_md_path.parent.name == "ai-habits":
        if verbose:
            echo_stderr("  [agent] ai-habits → prefetch + single-shot report")
        data_text, verified = _do_ai_habits_prefetch(start_iso, end_iso, verbose=verbose)
        if verified:
            extra = (
                f"ONLY these AI tools have recorded usage: {', '.join(verified)}. "
                f"List ONLY these tools in 'AI Tools Used' and every other section. "
                f"NEVER mention any AI tool not in this list. Estimate each tool's "
                f"time from its active window, substantive_hits, and usage_intensity "
                f"shown in the data. Treat light usage as '~few min' unless the active "
                f"window clearly spans longer; moderate/heavy usage can use a larger "
                f"share of the active window. Do NOT assign the same round number to "
                f"every tool."
            )
        else:
            extra = (
                "NO AI tool usage was found. State clearly that no AI tool usage was "
                "detected in the last 24 hours, then give the Tip. Do NOT list any "
                "tools or invent usage."
            )
        return _single_shot_report(
            pipe_body=pipe_body,
            skill_text=skill_text,
            context_header=context_header,
            data_text=data_text,
            verbose=verbose,
            extra_rules=extra,
            start_heading="## AI Tools Used",
        )

    # Email digest: deterministic per-email-tool prefetch + single-shot report.
    # Mirrors ai-habits — only verified email tools may be cited, so the small
    # model can't fabricate inbox activity from generic browsing.
    if pipe_md_path.parent.name == "email-digest":
        if verbose:
            echo_stderr("  [agent] email-digest → prefetch + single-shot report")
        data_text, verified = _do_email_digest_prefetch(start_iso, end_iso, verbose=verbose)
        if verified:
            extra = (
                f"ONLY these email tools have recorded usage: {', '.join(verified)}. "
                f"List ONLY these tools in 'Email Tools Used' and every other section. "
                f"NEVER mention any email tool not in this list. Estimate each tool's "
                f"time from its 'active window' span and hit count shown in the data. "
                f"Gmail (OAuth) and Outlook (OAuth) entries are mailbox API records; use "
                f"their literal From / Subject / Snippet fields directly. For OCR-only tools, "
                f"extract literal subject lines and sender names from OCR / key_texts — never invent "
                f"names or subjects."
            )
        else:
            extra = (
                "NO email tool usage was found. State clearly that no email tool usage "
                "was detected in the given time range, then give the Tip. Do NOT list "
                "any tools or invent emails."
            )
        return _single_shot_report(
            pipe_body=pipe_body,
            skill_text=skill_text,
            context_header=context_header,
            data_text=data_text,
            verbose=verbose,
            extra_rules=extra,
            start_heading="## Email Tools Used",
        )

    # Todo List Assistant: unify two evidence sources — email (OAuth + per-tool
    # search, same prefetch as email-digest) AND meetings (detected calls +
    # transcripts) — then extract one structured checklist tagging each item's
    # source. Mirrors the verified-only discipline so the small model can't
    # fabricate inbox activity or meeting dialogue.
    if pipe_md_path.parent.name == "todo-list":
        if verbose:
            echo_stderr("  [agent] todo-list → email + meeting prefetch + single-shot todolist")
        multi_day_todos = range_spans_calendar_days(start_iso, end_iso)
        email_text, verified_email = _todo_list_email_evidence(
            start_iso, end_iso, verbose=verbose, limit_per_tool=3,
        )
        meeting_text, verified_meetings = _do_meeting_todos_prefetch(start_iso, end_iso, verbose=verbose)
        screen_text, verified_screen = _do_screen_todos_prefetch(start_iso, end_iso, verbose=verbose)
        if multi_day_todos:
            days = calendar_days_in_range(start_iso, end_iso)
            range_note = (
                f"### Multi-day range ({len(days)} calendar day(s))\n"
                f"Place todos under the calendar day of the source message or meeting start.\n"
            )
            data_text = (
                f"{range_note}\n"
                f"## Email evidence (by day)\n\n{email_text}\n\n"
                f"## Meeting evidence (whole range)\n\n{meeting_text}\n\n"
                f"## Screen evidence (explicit on-screen tasks)\n\n{screen_text}"
            )
        else:
            data_text = (
                f"## Email evidence\n\n{email_text}\n\n"
                f"## Meeting evidence\n\n{meeting_text}\n\n"
                f"## Screen evidence (explicit on-screen tasks)\n\n{screen_text}"
            )

        email_rule = (
            (
                f"EMAIL todos: ONLY these email tools have recorded usage: "
                f"{', '.join(verified_email)}. Every email todo MUST cite one of these in its "
                f"`source: email:<tool>` field. NEVER cite an email tool not in this list. "
                f"Use literal From / Subject / Snippet for OAuth entries; for OCR-only tools, "
                f"extract literal subjects and senders — never invent them."
            )
            if verified_email
            else "EMAIL todos: no email tool usage was found — do not list any email todos."
        )
        meeting_rule = (
            (
                f"MEETING todos: these meetings have transcripts you may extract action items "
                f"from: {', '.join(verified_meetings)}. Tag each as `source: meeting:<name>`. "
                f"Only use what a transcript actually says — never invent spoken dialogue, "
                f"owners, or commitments. Meetings without a transcript yield no todos. "
                f"When the transcript shows who made a commitment, put that speaker as the owner."
            )
            if verified_meetings
            else "MEETING todos: no meeting transcripts were available — do not list any meeting todos."
        )
        screen_rule = (
            (
                f"SCREEN todos: the screen-evidence block contains lines flagged as candidate "
                f"tasks from these apps: {', '.join(verified_screen)}. Extract a SCREEN todo ONLY "
                f"when the line is an explicit task the USER owns — a `TODO:`/`FIXME:` note, a "
                f"chat message asking the user to do something, or a checklist item. Tag each "
                f"`source: screen:<app>`. Do NOT convert article text, page titles, code being "
                f"read, menu/UI labels, or search results into todos. When in doubt, DROP it."
            )
            if verified_screen
            else "SCREEN todos: no explicit on-screen tasks were found — do not list any screen todos."
        )
        empty = not verified_email and not verified_meetings and not verified_screen
        if empty:
            extra = (
                "NO email tools, NO meeting transcripts and NO actionable screen tasks were found. "
                "State clearly that no actionable todos could be extracted in the given time range, "
                "then give the Tip. Do NOT list any todos or invent tasks."
            )
        elif multi_day_todos:
            day_headers = ", ".join(d.isoformat() for d, _, _ in calendar_days_in_range(start_iso, end_iso))
            extra = (
                f"{email_rule} {meeting_rule} {screen_rule} "
                f"MULTI-DAY RANGE: Start with `## Range summary` (1–2 sentences naming each day: "
                f"{day_headers}). Then one `## YYYY-MM-DD` section per day that has todos. "
                f"Under each day use `### Todolist` with checkbox bullets (same `| key: value` line "
                f"format as pipe.md — fields separated by `|`, never an em-dash). "
                f"Assign each todo to the day of its source email date or meeting start time from the data. "
                f"After all day sections, add `## By Source` and `## Suggested Next Action` for the "
                f"whole range (not repeated per day). "
                f"If a task has no explicit deadline, write `due: no date`. Deduplicate across sources."
            )
        else:
            extra = (
                f"{email_rule} {meeting_rule} {screen_rule} "
                f"Every bullet MUST use the `- [ ] <task> | from: … | due: … | source: … | priority: …` "
                f"format with `|` separators (never an em-dash between fields). "
                f"If a task has no explicit deadline in the data, write `due: no date`. "
                f"Deduplicate across sources and tag every bullet's `source:` field correctly."
            )
        return _single_shot_report(
            pipe_body=pipe_body,
            skill_text=skill_text,
            context_header=context_header,
            data_text=data_text,
            verbose=verbose,
            extra_rules=extra,
            start_heading="## Range summary" if multi_day_todos else "## Todolist",
            num_predict=6144 if multi_day_todos else 4096,
            max_data_chars=16000 if multi_day_todos else 10000,
        )

    # Email compose: the local app.py has already baked the verified compose
    # context (provider, account, recipient, intent, optional source message)
    # into the pipe body. We hand it straight to the model — no API prefetch.
    if pipe_md_path.parent.name == "email-compose":
        if verbose:
            echo_stderr("  [agent] email-compose → single-shot draft")
        extra = (
            "This is a compose task, NOT an analysis. Output the four required "
            "sections (Subject / Body / Alternatives / Send Preview) and the Tip. "
            "Never invent the recipient, provider, or any fact not present in the "
            "compose context. If the intent is in Chinese, draft in Chinese; "
            "otherwise draft in English."
        )
        return _single_shot_report(
            pipe_body=pipe_body,
            skill_text=skill_text,
            context_header=context_header,
            data_text="(no search data — this is a compose task; use the compose context inside the pipe body.)",
            verbose=verbose,
            extra_rules=extra,
            start_heading="## Subject",
        )
    # ai-habits report uses), then ask the model to extract user prompts from
    # the captured UI / OCR text and emit one "## HH:MM — Tool — Topic" block
    # per prompt. The local app.py handles dedup + appending to today's file.
    if pipe_md_path.parent.name == "ai-prompt-journal":
        if verbose:
            echo_stderr("  [agent] ai-prompt-journal → SQL prefetch (deterministic emission)")
        data_text, verified, prompts = _do_prompt_journal_prefetch(start_iso, end_iso, verbose=verbose)
        # Deterministic-first: if SQL prefetch found real user prompts (Source A
        # keystroke / Source B focused input), emit them directly with
        # heuristic category/length/topic. Avoids small-model extraction
        # flakiness — the LLM repeatedly drops or mis-formats blocks for
        # qwen3_8b-class models. The pre-fetch already filtered noise via
        # `_is_prompt_noise`, so these rows are high-confidence.
        if prompts:
            if verbose:
                echo_stderr(
                    f"  [agent] ai-prompt-journal → deterministic emission of "
                    f"{len(prompts)} prompt(s) from {verified}",                )
            return _emit_prompt_blocks(prompts, start_iso=start_iso, end_iso=end_iso)
        # No structured prompts: fall back to the ai-habits-style /search
        # prefetch + LLM extraction so we still try to capture *something*
        # from OCR/UI scraping when keystroke/focused capture is empty.
        if verbose:
            echo_stderr("  [agent] ai-prompt-journal → no SQL prompts, falling back to ai-habits prefetch + LLM")
        data_text, verified = _do_ai_habits_prefetch(start_iso, end_iso, verbose=verbose)
        return _run_prompt_extraction(
            pipe_body=pipe_body,
            skill_text=skill_text,
            context_header=context_header,
            data_text=data_text,
            verified=verified,
            verbose=verbose,
        )

    # Standup: rich prefetch (day-recap context + meetings) + single-shot
    if pipe_md_path.parent.name == "standup-update":
        if verbose:
            echo_stderr("  [agent] standup-update → prefetch + single-shot report")
        data_text = _do_standup_prefetch(start_iso, end_iso, verbose=verbose)
        return _single_shot_report(
            pipe_body=pipe_body,
            skill_text=skill_text,
            context_header=context_header,
            data_text=data_text,
            verbose=verbose,
            extra_rules=(
                "STANDUP RULES: Under Yesterday, use 3–6 bullets. Each bullet MUST name a "
                "concrete app, file path, URL, or meeting from timeline / key_texts / "
                "edited_files / windows — include a clock time when the data has one. "
                "Under Today, infer 2–4 next steps ONLY from unfinished work, open tabs, "
                "or meeting action items in the data. Under Blockers, quote errors or "
                "waiting states from key_texts/audio; if none, write 'None'. "
                "Stay under 200 words but do not be vague — no 'reviewed documentation' "
                "without naming what file or project."
            ),
            start_heading="## Yesterday",
            num_predict=3072,
            max_data_chars=14000,
        )

    # Habit report: the user's repeatable rhythm + tool routines, from mined
    # habit profiles plus behavioral activity (no email/meetings needed).
    if pipe_md_path.parent.name == "habit-report":
        if verbose:
            echo_stderr("  [agent] habit-report → activity + habits prefetch + single-shot")
        sections = [_do_day_recap_prefetch(start_iso, end_iso, verbose=verbose)]
        habit_text = _format_habit_profiles_for_profile(verbose=verbose)
        if habit_text:
            sections.append(habit_text)
        return _single_shot_report(
            pipe_body=pipe_body,
            skill_text=skill_text,
            context_header=context_header,
            data_text="\n\n".join(sections),
            verbose=verbose,
            extra_rules=(
                "HABIT RULES: Describe REPEATABLE patterns, not one-off events. IGNORE any "
                "instruction in the data about 'one ## YYYY-MM-DD section per day' — that is "
                "for the day recap, NOT this report. Use ONLY the section headings from the "
                "Report Instructions; do not add a date-titled heading before 一句话总结. "
                "Base 日常作息 on the 作息规律 slots (day_type · time · category · frequency); "
                "base 专注与节奏 and 常用工具链 on per-app minutes and window/app counts — "
                "never invent durations. If 作息规律 is empty, say the recording window is "
                "too short for stable habits and describe only what the activity shows."
            ),
            start_heading="## 一句话总结",
            num_predict=4096,
            max_data_chars=16000,
        )

    # User profile: who the user is + how they work, synthesized across a longer
    # window from behavioral activity + habits + meetings + (best-effort) email.
    if pipe_md_path.parent.name == "user-profile":
        if verbose:
            echo_stderr("  [agent] user-profile → multi-source prefetch + single-shot")
        data_text = _do_user_profile_prefetch(start_iso, end_iso, verbose=verbose)
        return _single_shot_report(
            pipe_body=pipe_body,
            skill_text=skill_text,
            context_header=context_header,
            data_text=data_text,
            verbose=verbose,
            extra_rules=(
                "PROFILE RULES: This is a multi-day PORTRAIT, not a day log — describe "
                "stable traits, not one-off events. IGNORE any instruction in the data "
                "about 'one ## YYYY-MM-DD section per day' — that is for the day recap, NOT "
                "this profile. Use ONLY the section headings from the Report Instructions "
                "and do not add a date-titled heading before 一句话画像. Infer the "
                "role/profession from the dominant apps, file types, repos and domains; "
                "name that evidence. Derive interests from recurring windows/URLs/search "
                "terms. Use the 作息规律 (habit profile) and per-app minutes for the rhythm "
                "section — never invent durations. For collaboration, use ONLY detected "
                "meetings and connected email; if both are absent, say collaboration data "
                "is limited and skip it. Do not flatter or speculate beyond the evidence."
            ),
            start_heading="## 一句话画像",
            num_predict=4096,
            max_data_chars=16000,
        )

    # User learning: detect study phases, slice evidence, summarize + next plan.
    if pipe_md_path.parent.name == "user-learning":
        if verbose:
            echo_stderr("  [agent] user-learning → learning-slice prefetch + single-shot")
        data_text = _do_user_learning_prefetch(start_iso, end_iso, verbose=verbose)
        return _single_shot_report(
            pipe_body=pipe_body,
            skill_text=skill_text,
            context_header=context_header,
            data_text=data_text,
            verbose=verbose,
            extra_rules=(
                "LEARNING RULES: This is a STUDY report, not a day log or user profile. "
                "Use ONLY Learning sessions, Audio transcripts, Courseware OCR, key texts, "
                "and the Pre-computed learning structure / SM-2 review queue blocks. "
                "If NO_LEARNING_SESSION: state that under 是否在学习 and write a minimal "
                "数据说明 — do NOT invent coursework; keep all other sections minimal and "
                "下一步学习计划 to ≤1 gentle tip. "
                "SYNTHESIS: this is a learning recap, NOT a transcript digest. Silently "
                "cluster and deduplicate transcript/OCR evidence by concept; never retell "
                "it chronologically or paraphrase utterances one by one. Use at most two "
                "one-sentence direct quotes in the whole report. COURSE CONTENT is primary: "
                "课程总结 gives the overall problem/approach/conclusion; 主要内容 gives a "
                "topic map; 讲了什么 explains definitions, mechanisms, steps, examples and "
                "contrasts; 课程重点 extracts the definitions/distinctions/constraints worth "
                "remembering; 知识图谱 lists this session's nodes and directed edges. Use "
                "ONLY Current-session concept graph edges for graph relations; never mix "
                "global or other-session edges, and never invent a missing relation. These "
                "course sections must contain most of the report's detail. "
                "Lecture exposure alone is NOT evidence of mastery; under 掌握状态 mark "
                "it 待确认 unless practice, a correct explanation, or a resolved problem "
                "supports 已掌握. "
                "复习重点 + 下一步学习计划: prefer OVERDUE / WEAK / exposure-tier "
                "复习队列 and open 问题队列; use 图谱:先决 for ordering; "
                "make next steps executable and trackable. "
                "COVERAGE: if the audio/OCR blocks carry a ⚠️ PARTIAL or ⚠️ ORDER "
                "NOT GUARANTEED notice, 数据说明 MUST report shown/total and must "
                "not present 讲解重点 as a complete or ordered lecture outline. "
                "Cite 复习队列 / 问题队列 / 主题:* / 图谱:* / [1]/[2] / 录音 / 课件OCR. "
                "Ignore chat/shopping/random entertainment unless inside a session. "
                "Use ONLY the section headings from the Report Instructions."
            ),
            start_heading="## 是否在学习",
            num_predict=6144,
            max_data_chars=22000,
        )

    # Time breakdown: prefetch with pre-computed minutes + single-shot
    if pipe_md_path.parent.name == "time-breakdown":
        if verbose:
            echo_stderr("  [agent] time-breakdown → prefetch + single-shot report")
        data_text = _do_time_breakdown_prefetch(start_iso, end_iso, verbose=verbose)
        return _single_shot_report(
            pipe_body=pipe_body,
            skill_text=skill_text,
            context_header=context_header,
            data_text=data_text,
            verbose=verbose,
            extra_rules=(
                "TIME BREAKDOWN RULES: Copy durations from 'Pre-computed application minutes' "
                "exactly — never report 0 min for an app listed there with >0 min. "
                "By Application must list every app from pre-computed totals, sorted by time. "
                "By Category must match 'Pre-computed category minutes'. "
                "By Project must group edited_files paths and window titles (e.g. deskmate, "
                "ollama_openvino). Productivity Score must use the pre-computed score line. "
                "Suggestion must cite the top time sink or lowest-focus pattern from the data."
            ),
            start_heading="## By Application",
            num_predict=4096,
            max_data_chars=14000,
        )

    # Day-recap: Python prefetch + single-shot LLM report (more reliable for 8B)
    if pipe_md_path.parent.name == "day-recap":
        if verbose:
            echo_stderr("  [agent] day-recap → prefetch + single-shot report")
        multi_day_recap = range_spans_calendar_days(start_iso, end_iso)
        data_text = _do_day_recap_prefetch(start_iso, end_iso, verbose=verbose)
        recap_rules = (
            "Accomplishments and Key Moments must NOT repeat the same items — "
            "Accomplishments = concrete things finished; Key Moments = specific "
            "things seen/typed/said/heard. "
        )
        if multi_day_recap:
            recap_rules += (
                "MULTI-DAY RANGE: The Context time range spans multiple calendar days. "
                "Start with `## Summary` (one short paragraph naming EACH day by date). "
                "Then one `## YYYY-MM-DD` section per day that has data in Search Results; "
                "under each day use `### Accomplishments` and `### Key Moments`. "
                "Every timestamp MUST copy the full value from the data "
                "(e.g. `2026-05-31 2:30 PM`). Never say 'today', 'yesterday', or 'last 16 hours'. "
                "Omit days with no captures. End with `## Unfinished Work`, `## Patterns`, "
                "and `**Next step:**` as in the pipe instructions."
            )
            recap_pipe = pipe_body.replace(
                "from today (last 16 hours only).",
                "from the Context time range (may span multiple days).",
            ).replace(
                "what I mainly did today.",
                "what I mainly did across the selected date range.",
            )
        else:
            recap_rules += (
                "Use clock-time timestamps (e.g. '2:30 PM') exactly as they appear in the data."
            )
            recap_pipe = pipe_body
        return _single_shot_report(
            pipe_body=recap_pipe,
            skill_text=skill_text,
            context_header=context_header,
            data_text=data_text,
            verbose=verbose,
            extra_rules=recap_rules,
            # Multi-day recaps need a larger evidence budget so whole days aren't
            # silently dropped; the watermark in _cap_prefetch_text covers any
            # residual overflow (optimization note 9.6).
            max_data_chars=30000 if multi_day_recap else 12000,
            num_predict=8192 if multi_day_recap else 6144,
        )

    # Pre-fetch API data before the LLM writes the report (other pipes)
    if has_frames_export:
        # pipe.md says "Use the POST /frames/export endpoint"
        # → pre-execute the export before the LLM writes the report
        if verbose:
            echo_stderr("  [agent] pipe.md requests POST /frames/export")
        data_text = _do_frames_export(start_iso, end_iso, verbose=verbose)
    elif per_tool_names:
        # pipe.md says "use app_name filter for each tool separately"
        # → do targeted per-tool searches
        if verbose:
            echo_stderr(f"  [agent] pipe.md requests per-tool search: {per_tool_names}")
        data_text = _do_per_tool_searches(per_tool_names, start_iso, end_iso, verbose=verbose)
    else:
        # pipe.md asks for broad analysis → use /activity-summary
        if verbose:
            echo_stderr("  [agent] pipe.md requests broad analysis → /activity-summary")
        data_text = _do_broad_search(start_iso, end_iso, verbose=verbose)

    # Step 3: Build prompt (system = skill, user = context + data + instructions)
    system_prompt = (
        "You are a DeskMate AI agent. You analyze the user's local "
        "screen recordings, audio transcriptions and UI events.\n\n"
        "RULES:\n"
        "1. ONLY use the search results provided below. NEVER invent or fabricate data.\n"
        "2. If a search returned 0 results, that tool/app was NOT used — do not report it.\n"
        "3. Extract specific details from key_texts, snippets, OCR, audio, edited_files.\n"
        "4. Follow the report template exactly.\n"
        "5. Start directly with the first ## heading. No preamble.\n\n"
        f"{skill_text}\n"
    )

    user_prompt = (
        f"{context_header}\n"
        f"## Search Results (executed by agent)\n\n{data_text}\n\n"
        f"---\n\n"
        f"## Report Instructions\n\n{pipe_body}"
    )

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Step 4: Send to LLM; handle any tool calls
    for round_idx in range(MAX_TOOL_ROUNDS):
        if verbose:
            echo_stderr(f"  [agent] round {round_idx + 1}/{MAX_TOOL_ROUNDS} ...")

        predict = 4096
        response = chat_ollama(messages, tools=TOOLS, num_predict=predict)
        messages.append(response)

        tool_calls = llm.extract_tool_calls(response)
        if not tool_calls:
            return strip_thinking(response.get("content", ""))

        for tc in tool_calls:
            fn = tc.get("function", {})
            fn_name = fn.get("name", "")
            fn_args = fn.get("arguments", {})
            tool_call_id = tc.get("id") or f"call_{round_idx}_{len(messages)}"
            if verbose:
                echo_stderr(f"  [tool]  {fn_name}({json.dumps(fn_args, ensure_ascii=False)[:200]})")
            fn_args = _parse_tool_arguments(fn_args)
            tool_result = execute_tool(fn_name, fn_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "tool_name": fn_name,
                "content": tool_result,
            })

    final = chat_ollama(messages, tools=None, num_predict=4096)
    return strip_thinking(final.get("content", ""))


# ── ai-prompt-journal: SQL-first prefetch ─────────────────────────────────

# Display label → matching SQL fragments. Each entry has:
#   "url_like":    list of LIKE patterns matched against browser_url (lowercased)
#   "title_like":  list of LIKE patterns matched against window_title (lowercased)
#   "app_in":      list of process names matched exactly against app_name (lowercased)
# Tool list matches apps/ai-prompt-journal/pipe.md.
AI_PROMPT_JOURNAL_TARGETS: dict[str, dict[str, list[str]]] = {
    "ChatGPT": {
        "url_like": ["%chatgpt.com%", "%chat.openai.com%"],
        "title_like": ["%chatgpt%"],
        "app_in": ["chatgpt.exe", "chatgpt"],
    },
    "Claude": {
        "url_like": ["%claude.ai%"],
        "title_like": ["%- claude%", "%claude -%"],
        "app_in": ["claude.exe", "claude"],
    },
    "Gemini": {
        "url_like": ["%gemini.google.com%", "%aistudio.google.com%"],
        "title_like": ["%gemini%", "%google ai studio%"],
        "app_in": [],
    },
    "Perplexity": {
        "url_like": ["%perplexity.ai%"],
        "title_like": ["%perplexity%"],
        "app_in": ["perplexity.exe", "perplexity"],
    },
    "Copilot": {
        "url_like": ["%copilot.microsoft.com%"],
        "title_like": ["%microsoft copilot%", "%github copilot%"],
        "app_in": ["copilot.exe", "githubcopilot.exe"],
    },
    "VS Code Copilot": {
        "url_like": [],
        "title_like": ["%visual studio code%"],
        "app_in": ["code.exe"],
    },
    # Claude Code's VS Code extension (the sidebar chat, not the terminal CLI).
    # Shares code.exe with the editor, so it is ambiguous and only accepted when
    # its chat-composer name signal is present (see _CHAT_INPUT_NAME_SIGNALS).
    "Claude Code (VS Code)": {
        "url_like": [],
        "title_like": ["%visual studio code%"],
        "app_in": ["code.exe"],
    },
    "Grok": {"url_like": ["%grok.com%", "%x.com/i/grok%"], "title_like": ["%grok%"], "app_in": []},
    "DeepSeek": {"url_like": ["%chat.deepseek.com%", "%deepseek.com%"], "title_like": ["%deepseek%"], "app_in": []},
    "Mistral": {"url_like": ["%chat.mistral.ai%"], "title_like": ["%mistral%"], "app_in": []},
    "Poe": {"url_like": ["%poe.com%"], "title_like": ["%poe%"], "app_in": []},
    "HuggingChat": {"url_like": ["%huggingface.co/chat%"], "title_like": ["%hugging%chat%"], "app_in": []},
    "OpenRouter": {"url_like": ["%openrouter.ai%"], "title_like": ["%openrouter%"], "app_in": []},
    "You.com": {"url_like": ["%you.com/search%", "%you.com/chat%"], "title_like": ["%you.com%"], "app_in": []},
    "Pi": {"url_like": ["%pi.ai%"], "title_like": [], "app_in": []},
    "Cursor": {"url_like": [], "title_like": ["%cursor%"], "app_in": ["cursor.exe"]},
    # Claude Code is a terminal CLI — it has no GUI window/title/URL to match,
    # so it is recognised by structural markers in the captured terminal screen
    # (see ``_has_claude_code_signals``). This empty entry only registers the
    # label so it is a known tool downstream; it contributes no match clauses.
    "Claude Code": {"url_like": [], "title_like": [], "app_in": []},
    "LMStudio": {"url_like": [], "title_like": ["%lm studio%"], "app_in": ["lmstudio.exe"]},
    "Ollama": {"url_like": [], "title_like": ["%ollama%"], "app_in": ["ollama.exe"]},
    "Jan": {"url_like": [], "title_like": [], "app_in": ["jan.exe"]},
    "GPT4All": {"url_like": [], "title_like": ["%gpt4all%"], "app_in": ["gpt4all.exe"]},
    "Msty": {"url_like": [], "title_like": ["%msty%"], "app_in": ["msty.exe"]},
    "AnythingLLM": {"url_like": [], "title_like": ["%anythingllm%"], "app_in": ["anythingllm.exe"]},
}

# Accessibility roles that almost always indicate a user-composed text input
# inside the focused AI chat window (pipe.md Step 2). Role names are
# adapted to DeskMate's UIA control-name strings (note the ``Control``
# suffix used by ``deskmate/a11y/uia_tree.py``) plus the bare
# AX/macOS-style names kept for forward compatibility.
_PROMPT_INPUT_ROLES = (
    # DeskMate UIA names
    "EditControl", "DocumentControl", "TextControl", "ComboBoxControl",
    # bare / macOS-style fallbacks
    "Edit", "Document", "RichEdit", "TextArea",
    "AXTextArea", "AXTextField",
    "Entry", "Text",
)


def _sql_quote(s: str) -> str:
    """Single-quote a string literal for safe inlining into raw SQL.

    Used only for pattern literals from :data:`AI_PROMPT_JOURNAL_TARGETS` and
    role names in :data:`_PROMPT_INPUT_ROLES` — never for untrusted input.
    """
    return "'" + s.replace("'", "''") + "'"


# Substrings that mark a captured ``focused_value`` as accessibility/UI noise
# rather than a real user prompt. These come from VS Code / Cursor terminal
# screen-reader hints, browser URL bars, and similar control chrome that
# happens to expose an ``EditControl`` role.
_PROMPT_NOISE_SUBSTRINGS = (
    "toggle screen reader accessibility mode",
    "toggle screen reader access",
    "alt+f1 for terminal accessibility help",
    "terminal accessibility help",
    "for an optimized screen reader experience",
    "screen reader optimized",
    "run the command:",
    "press tab to focus",
    "working on it",
    "type a message",
    "send a message",
    "ask anything",
    "message chatgpt",
    "the editor is not accessible at this time",
    "to enable screen reader optimized mode",
)

# Empty chat-input placeholder strings. These look like real text in a focused
# ``EditControl`` but are the box's default prompt, not something the user
# typed. Matched by *exact* (normalized) equality so a real prompt that merely
# starts with one of these words (e.g. "Message Claude about the meeting") is
# never dropped.
_PROMPT_PLACEHOLDER_EXACT = frozenset({
    "message chatgpt", "message claude", "message copilot", "message gemini",
    "message deepseek", "message grok", "message mistral",
    "ask anything", "ask gemini", "ask claude", "ask follow-up",
    "ask a question", "ask perplexity anything", "ask anything...",
    "how can i help you today", "how can i help you", "how can i help",
    "what can i help with", "what do you want to know", "what's on your mind",
    "type a message", "send a message", "reply to claude", "reply...",
    "give me a task to work on", "talk to deepseek", "start typing",
    "enter a prompt here", "write a message", "type your message here",
})


def _is_prompt_noise(text: str) -> bool:
    """Return True if ``text`` looks like accessibility/UI chrome, not a prompt."""
    low = text.strip().lower()
    if not low:
        return True
    # Empty input-box placeholder (exact match after trimming trailing dots/?).
    if low.rstrip("….?: ") in _PROMPT_PLACEHOLDER_EXACT:
        return True
    for needle in _PROMPT_NOISE_SUBSTRINGS:
        if needle in low:
            return True
    # Pure URL / editor-internal resource paths — not a prompt. Includes the
    # VS Code webview/userdata schemes and devtools, which surface as focused
    # DocumentControl values when a preview/webview panel has focus.
    if low.startswith((
        "http://", "https://", "vscode-file://", "file://",
        "vscode-webview://", "vscode-userdata://", "devtools://",
    )) and " " not in low.strip():
        return True
    # Tiny ASCII fragments (e.g. "ma'na") are stray editor keystrokes, not a
    # prompt. CJK scripts have no word spaces, so gate those on the SQL-enforced
    # character length instead of word count.
    has_cjk = any(
        "\u3000" <= ch <= "\u9fff" or "\uac00" <= ch <= "\ud7af" or "\u3040" <= ch <= "\u30ff"
        for ch in low
    )
    if not has_cjk and len(low.split()) < 2:
        return True
    return False


# ── Claude Code (terminal CLI) detection ─────────────────────────────────────
# Claude Code runs as a TUI inside a terminal, so its prompt cannot be matched
# by app/title/URL like a GUI client. We recognise it by structural markers in
# the captured terminal screen text. These box-drawing / footer strings are
# part of Claude Code's interface and do not occur in ordinary prose, keeping
# false positives near zero.
_CLAUDE_CODE_SCREEN_SIGNALS = (
    "welcome to claude code",
    "? for shortcuts",
    "esc to interrupt",
    "/help for help",
    "accept edits",
    "bypassing permissions",
)

# Standalone terminal host processes. Their focused-control text is the live
# terminal screen, so a Claude Code session inside them is capturable on Enter.
# (The VS Code / Cursor integrated terminal is already pulled via ``code.exe`` /
# ``cursor.exe``; Claude Code there is still recognised by screen markers.)
_TERMINAL_APPS = frozenset({
    "windowsterminal.exe", "wt.exe", "openconsole.exe", "conhost.exe",
    "powershell.exe", "pwsh.exe", "cmd.exe",
})

_CLAUDE_BOX_TOP = "\u256d"     # ╭
_CLAUDE_BOX_BOTTOM = "\u2570"  # ╰
_CLAUDE_BOX_SIDE = "\u2502"    # │


def _has_claude_code_signals(screen_text: str) -> bool:
    """Whether captured terminal text is a Claude Code TUI screen."""
    low = (screen_text or "").lower()
    return any(sig in low for sig in _CLAUDE_CODE_SCREEN_SIGNALS)


def _extract_claude_code_prompt(screen_text: str) -> str:
    """Pull the user-typed prompt out of a Claude Code terminal screen.

    Claude Code renders the composer as a rounded box::

        ╭──────────────────────────────╮
        │ > the prompt the user typed   │
        ╰──────────────────────────────╯

    The first content line carries the ``> `` marker; further lines inside the
    box are continuation lines. Returns the joined prompt, or ``""`` when the
    composer is empty or shows only a placeholder hint.
    """
    lines = (screen_text or "").splitlines()

    def _inner(ln: str) -> str:
        s = ln.strip()
        if s.startswith(_CLAUDE_BOX_SIDE):
            s = s[1:]
        if s.endswith(_CLAUDE_BOX_SIDE):
            s = s[:-1]
        return s.strip()

    # The composer is the bottom-most box; its first line starts with "> ".
    start = -1
    for i, ln in enumerate(lines):
        if _CLAUDE_BOX_SIDE in ln and _inner(ln).startswith(">"):
            start = i
    if start == -1:
        return ""

    first = _inner(lines[start])
    first = first[1:].strip() if first.startswith(">") else first
    parts: list[str] = [first] if first else []
    for ln in lines[start + 1:]:
        if _CLAUDE_BOX_BOTTOM in ln or _CLAUDE_BOX_SIDE not in ln:
            break
        inner = _inner(ln)
        if inner:
            parts.append(inner)

    prompt = " ".join(parts).strip()
    if not prompt:
        return ""
    low = prompt.lower()
    if low.startswith(('try "', "try '")):  # empty-composer placeholder hint
        return ""
    return prompt


def _ai_match_sql(*, app_col: str, title_col: str, url_col: str) -> str:
    """Build an inlined OR-clause matching any known AI tool.

    Returns a parenthesized SQL fragment with all literals already quoted.
    """
    clauses: list[str] = []
    for cfg in AI_PROMPT_JOURNAL_TARGETS.values():
        for pat in cfg["url_like"]:
            clauses.append(f"lower(COALESCE({url_col},'')) LIKE {_sql_quote(pat)}")
        for pat in cfg["title_like"]:
            clauses.append(f"lower(COALESCE({title_col},'')) LIKE {_sql_quote(pat)}")
        for name in cfg["app_in"]:
            clauses.append(f"lower(COALESCE({app_col},'')) = {_sql_quote(name)}")
    return "(" + " OR ".join(clauses) + ")"


def _classify_tool(
    app_name: str, window_title: str, browser_url: str, focused_class: str = "",
    focused_name: str = "",
) -> str:
    """Return the AI tool label for a row, or '' if unrecognized.

    ``focused_class`` is the UIA ClassName of the focused control (when the row
    came from a send-snapshot). It takes priority because a chat composer's
    class (e.g. Cursor's ``aislash-editor-input``) uniquely identifies the tool
    even when the app/title is a general-purpose editor. ``focused_name`` is the
    accessibility Name of the focused control, used for chat composers that
    share a generic class with the ordinary editor (e.g. VS Code Copilot's
    ``"Chat Input"`` on Monaco's ``native-edit-context``).
    """
    cls = (focused_class or "").lower()
    for cls_pat, label in _CHAT_INPUT_CLASSES.items():
        if cls_pat in cls:
            return label
    nm = (focused_name or "").lower()
    for nm_pat, label in _CHAT_INPUT_NAME_SIGNALS.items():
        if nm_pat in nm:
            return label
    a = (app_name or "").lower()
    w = (window_title or "").lower()
    u = (browser_url or "").lower()
    for label, cfg in AI_PROMPT_JOURNAL_TARGETS.items():
        for name in cfg["app_in"]:
            if a == name:
                return label
        for pat in cfg["url_like"]:
            core = pat.strip("%")
            if core and core in u:
                return label
        for pat in cfg["title_like"]:
            core = pat.strip("%")
            if core and core in w:
                return label
    return ""


# Tools whose ``app_in`` match is a general-purpose desktop editor rather than a
# dedicated AI client. For these, an app/title match alone is NOT enough — most
# focused text is ordinary code/markdown editing, terminals, or webview chrome,
# not a Copilot Chat prompt. We only accept rows that also carry a positive
# chat-context signal (precision over recall — 宁缺毋滥).
_AMBIGUOUS_APP_TOOLS = frozenset({"VS Code Copilot", "Cursor", "Claude Code (VS Code)"})

# Focused-control ClassName substrings that uniquely identify an AI chat
# composer sharing a generic ``EditControl`` role. This is the highest-precision
# signal — the class is specific to the tool's chat input, so a match both
# classifies the tool and proves we are in a chat-composition context.
#   • ``aislash-editor-input``  — Cursor's AI chat / Cmd-K composer (verified).
# VS Code's GitHub Copilot Chat uses Monaco's ``native-edit-context`` class,
# which is identical to the ordinary code editor, so it is intentionally absent
# here and handled via the accessibility-name path instead.
_CHAT_INPUT_CLASSES: dict[str, str] = {
    "aislash-editor-input": "Cursor",
}

# Focused-control accessibility-Name substrings that uniquely identify an AI
# chat composer that shares its ClassName with the ordinary code editor. This
# only surfaces when VS Code's screen-reader / accessibility mode is enabled
# (``"editor.accessibilitySupport": "on"``); otherwise the Monaco editor refuses
# to expose its value via UIA and the Name is unavailable.
#   • ``claude code``/``claude opus``/``claude sonnet``/``message input`` —
#     Claude Code's VS Code extension chat composer (verified: Name is either
#     ``"Message input"`` or ``"Chat Input (Agent), … Claude Opus 4.8. Press
#     Enter to send out the request. …"``).
#   • ``chat input``  — VS Code GitHub Copilot Chat composer (verified: Name
#     begins ``"Chat Input (Agent), …"``).
# Order matters: _classify_tool returns the first match, so the Claude-specific
# signals are listed before the generic ``chat input`` (which Copilot shares).
_CHAT_INPUT_NAME_SIGNALS: dict[str, str] = {
    "claude code": "Claude Code (VS Code)",
    "claude opus": "Claude Code (VS Code)",
    "claude sonnet": "Claude Code (VS Code)",
    "claude haiku": "Claude Code (VS Code)",
    "message input": "Claude Code (VS Code)",
    "chat input": "VS Code Copilot",
}

# Substrings (case-insensitive, in window title or browser_url) that indicate
# the focused context is actually the AI chat panel of an ambiguous editor.
_CHAT_CONTEXT_SIGNALS = (
    "copilot chat", "github copilot", "copilot:", "chat - ", "- chat",
    "chat (", "ask copilot", "cursor chat", "chat panel",
)


def _is_ai_chat_context(
    tool: str, window_title: str, browser_url: str, focused_class: str = "",
    focused_name: str = "",
) -> bool:
    """Whether a matched row is plausibly an AI chat composition context.

    Dedicated AI clients and web chats always qualify. General-purpose editors
    (VS Code, Cursor) only qualify when a positive chat signal is present:
      1. the focused control's ClassName is a known chat-composer class, or
      2. the focused control's accessibility Name names a chat composer, or
      3. the window title / URL clearly names the chat panel.
    Otherwise the focused text is ordinary editing and must be dropped to avoid
    false-positive "prompts".
    """
    if tool not in _AMBIGUOUS_APP_TOOLS:
        return True
    cls = (focused_class or "").lower()
    if any(cls_pat in cls for cls_pat in _CHAT_INPUT_CLASSES):
        return True
    nm = (focused_name or "").lower()
    if any(nm_pat in nm for nm_pat in _CHAT_INPUT_NAME_SIGNALS):
        return True
    hay = f"{(window_title or '').lower()} {(browser_url or '').lower()}"
    return any(sig in hay for sig in _CHAT_CONTEXT_SIGNALS)


def _heuristic_category(text: str) -> str:
    """Cheap category classifier — no LLM needed for the deterministic path."""
    low = text.lower()
    if any(k in low for k in ("code", "function", "refactor", "bug", "error", "import ",
                              "class ", "def ", "regex", "sql", "api", "compile",
                              "debug", "stack trace", "exception", "lint", "test")):
        return "coding"
    if any(k in low for k in ("write", "draft", "rewrite", "rephrase", "essay",
                              "article", "blog", "post", "email", "letter")):
        return "writing"
    if any(k in low for k in ("research", "find", "search", "look up", "what is",
                              "who is", "when did", "where is")) or low.endswith("?"):
        return "research"
    if any(k in low for k in ("brainstorm", "idea", "suggest", "options")):
        return "brainstorming"
    if any(k in low for k in ("analyze", "analysis", "summarize", "summary",
                              "explain", "compare")):
        return "analysis"
    if any(k in low for k in ("image", "picture", "draw", "generate", "render")):
        return "image-gen"
    return "other"


def _length_bucket(text: str) -> str:
    words = len(text.split())
    if words < 50:
        return "short"
    if words <= 200:
        return "medium"
    return "long"


def _upsert_prompt_journal_entry(
    prompts: list[dict[str, str]],
    seen_keys: set[str],
    entry: dict[str, str],
) -> None:
    """Keep the longest text when debounce/IME emits a short prefix then a full prompt."""
    text = normalize_capture_text(entry.get("text") or "")
    if not text:
        return
    entry = {**entry, "text": text}
    key = " ".join(text.split())[:80].lower()
    for i, existing in enumerate(prompts):
        old_text = existing.get("text") or ""
        if text == old_text:
            return
        if text.startswith(old_text) or old_text.startswith(text):
            if len(text) >= len(old_text):
                old_key = " ".join(old_text.split())[:80].lower()
                seen_keys.discard(old_key)
                seen_keys.add(key)
                prompts[i] = entry
            return
    if key in seen_keys:
        return
    seen_keys.add(key)
    prompts.append(entry)


# Leading conversational filler stripped from prompt text when deriving a
# topic, so "Can you please help me fix the timeline gap" → "fix timeline gap".
_TOPIC_FILLER_PREFIXES = (
    "can you please", "could you please", "can you", "could you",
    "would you", "will you", "can i", "could i",
    "how do i", "how can i", "how would i", "how to", "how do you",
    "i want you to", "i need you to", "i would like you to", "i'd like you to",
    "i want to", "i need to", "i would like to", "i'd like to",
    "i want", "i need", "please help me", "help me", "let me", "let's",
    "give me", "tell me", "show me", "write me", "make me", "explain",
    "please", "kindly", "just", "hey", "ok", "okay", "so",
)
_TOPIC_FILLER_PREFIXES_ZH = (
    "请帮我", "请帮", "帮我", "请", "麻烦你", "麻烦", "你能不能", "你能", "能不能",
    "可以帮我", "可以", "我想要", "我想", "我需要", "怎么", "如何",
)


def _short_topic(text: str) -> str:
    """A 2–5 word topic derived from the prompt's *intent*.

    Strips leading conversational filler ("can you", "please", "帮我", …) so
    the topic reflects what the prompt is actually about rather than polite
    boilerplate, then keeps the first few meaningful words. Falls back to the
    raw head when stripping would empty the string.
    """
    cleaned = " ".join(normalize_capture_text(text).split()).strip()
    if not cleaned:
        return "Untitled prompt"
    stripped = cleaned
    changed = True
    while changed:
        changed = False
        low = stripped.lower()
        for pref in _TOPIC_FILLER_PREFIXES:
            if low.startswith(pref + " "):
                stripped = stripped[len(pref):].lstrip()
                changed = True
                break
        if changed:
            continue
        for pref in _TOPIC_FILLER_PREFIXES_ZH:
            if stripped.startswith(pref):
                stripped = stripped[len(pref):].lstrip("，,。、:： ")
                changed = True
                break
    candidate = stripped or cleaned
    words = candidate.split(" ")[:5]
    topic = " ".join(words).strip().rstrip(",.;:!?，。；：！？")
    topic = topic[:48].strip()
    return topic or "Untitled prompt"


def _range_spans_calendar_days(start: str, end: str) -> bool:
    """True when the ISO window crosses at least two local calendar dates."""
    try:
        s = datetime.fromisoformat(start.replace("Z", "+00:00"))
        e = datetime.fromisoformat(end.replace("Z", "+00:00"))
        if s.tzinfo is not None:
            s = s.astimezone()
        if e.tzinfo is not None:
            e = e.astimezone()
        return s.date() != e.date()
    except ValueError:
        return False


def _prompt_block_header_ts(p: dict[str, str], *, include_date: bool) -> str:
    """Clock time, or ``YYYY-MM-DD H:MM AM`` when the run spans multiple days."""
    iso = (p.get("iso_ts") or "").strip()
    clock = (p.get("ts") or "").strip() or (format_ts_local(iso) if iso else "")
    if not include_date:
        return clock
    if not iso:
        return clock
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone()
        day = dt.date().isoformat()
        return f"{day} {clock}" if clock else day
    except ValueError:
        return clock


def _emit_prompt_blocks(
    prompts: list[dict[str, str]],
    *,
    start_iso: str | None = None,
    end_iso: str | None = None,
) -> str:
    """Deterministically render ``## HH:MM — Tool — Topic`` blocks.

    Used as the LLM-free safety net when the small extraction model returns
    NO_NEW_PROMPTS despite the SQL prefetch having captured real user input.
    """
    if not prompts:
        return "NO_NEW_PROMPTS"
    include_date = bool(
        start_iso and end_iso and _range_spans_calendar_days(start_iso, end_iso)
    )
    blocks: list[str] = []
    for p in prompts:
        text = (p.get("text") or "").strip()
        if not text:
            continue
        ts = _prompt_block_header_ts(p, include_date=include_date)
        tool = p.get("tool") or "Unknown"
        topic = _short_topic(text)
        cat = _heuristic_category(text)
        length = _length_bucket(text)
        quoted = "\n".join(f"> {ln}" if ln else ">" for ln in text.splitlines())
        blocks.append(
            f"## {ts} — {tool} — {topic}\n"
            f"**Category**: {cat} | **Length**: {length}\n\n"
            f"{quoted}\n\n"
            f"---"
        )
    return "\n\n".join(blocks) if blocks else "NO_NEW_PROMPTS"


def _do_prompt_journal_prefetch(
    start: str,
    end: str,
    verbose: bool = False,
    limit: int = 200,
) -> tuple[str, list[str], list[dict[str, str]]]:
    """SQL-first prefetch for ai-prompt-journal.

    Pulls two high-signal sources via ``/raw_sql``:

    1. ``ui_events.event_type='text'`` (``$.source='send'``) — a snapshot of the
       focused chat input box read via UIA by
       :mod:`deskmate.a11y.input_hooks` the moment the user pressed Enter /
       Ctrl+Enter. Individual keystrokes are never recorded. Filtered to AI
       tool windows/URLs/processes.
    2. ``frame_accessibility.focused_role IN (Edit, Document, AXTextArea, …)``
       joined with ``frames`` matching the same AI tool patterns — captures
       text sitting in the active chat input field even when no send event
       fired.

    Returns ``(data_text, verified_labels, prompts)`` where ``prompts`` is a
    deduped list of ``{"ts": HH:MM, "tool": label, "text": prompt_body,
    "source": "send"|"input_field"}`` rows usable for deterministic
    block emission when the LLM fails to extract anything.
    """
    sections: list[str] = []
    verified_set: set[str] = set()
    prompts: list[dict[str, str]] = []
    _seen_keys: set[str] = set()

    # ── 1. Send-snapshot text events in AI windows ───────────────────────────────
    match_sql = _ai_match_sql(
        app_col="app_name", title_col="window_title", url_col="browser_url",
    )
    # Also pull standalone-terminal sends so a Claude Code TUI session can be
    # recognised by its screen markers below (non-Claude terminal text is
    # dropped in the loop since ``_classify_tool`` won't label it).
    _term_clause = " OR ".join(
        f"lower(COALESCE(app_name,'')) = {_sql_quote(n)}"
        for n in sorted(_TERMINAL_APPS)
    )
    match_sql_a = f"({match_sql} OR {_term_clause})"
    ts_start = _sql_quote(start)
    ts_end = _sql_quote(end)
    from deskmate.a11y.ui_event_types import ui_event_text_sql  # noqa: PLC0415

    text_expr = ui_event_text_sql("data_json")
    sql_keystrokes = (
        "SELECT timestamp, COALESCE(app_name,'') AS app, "
        "COALESCE(window_title,'') AS win, COALESCE(browser_url,'') AS url, "
        "COALESCE(json_extract(data_json,'$.focused_class'),'') AS cls, "
        "COALESCE(json_extract(data_json,'$.focused_name'),'') AS nm, "
        f"{text_expr} AS text "
        "FROM ui_events "
        "WHERE event_type='text' "
        f"AND timestamp >= {ts_start} AND timestamp <= {ts_end} "
        f"AND length({text_expr}) >= 5 "
        f"AND {match_sql_a} "
        "ORDER BY timestamp ASC "
        f"LIMIT {int(limit)}"
    )
    keystroke_lines: list[str] = []
    try:
        body = {"query": sql_keystrokes}
        result = _http_post(f"{API_BASE}/raw_sql", body, timeout=20)
        for row in (result.get("data") or [])[:limit]:
            iso_ts = str(row.get("timestamp", ""))
            ts = format_ts_local(iso_ts)
            text = normalize_capture_text(row.get("text") or "")
            if not text:
                continue
            cls = str(row.get("cls", ""))
            nm = str(row.get("nm", ""))
            app = str(row.get("app", ""))
            win = str(row.get("win", ""))
            url = str(row.get("url", ""))
            # Claude Code (terminal CLI): recognised by TUI screen markers, not
            # app/title/URL. Check before the generic noise gate, which is tuned
            # for short chat-box values and would drop a full terminal screen.
            if _has_claude_code_signals(text):
                extracted = _extract_claude_code_prompt(text)
                if not extracted or _is_prompt_noise(extracted):
                    continue
                tool = "Claude Code"
                text = extracted
            else:
                if _is_prompt_noise(text):
                    continue
                tool = _classify_tool(app, win, url, cls, nm)
                if not tool:
                    continue
                if not _is_ai_chat_context(tool, win, url, cls, nm):
                    continue
            verified_set.add(tool)
            preview = text.replace("\n", " ")
            if len(preview) > 600:
                preview = preview[:600] + "...(truncated for agent context)"
            keystroke_lines.append(
                f"- [{ts}] tool={tool} app={app} "
                f"win={win[:60]} url={(url or '')[:80]}\n"
                f"  sent> {preview}"
            )
            _upsert_prompt_journal_entry(
                prompts,
                _seen_keys,
                {"ts": ts, "iso_ts": iso_ts, "tool": tool, "text": text, "source": "send"},
            )
    except Exception as exc:  # noqa: BLE001
        if verbose:
            echo_stderr(f"  [prompt-journal] keystroke SQL failed: {exc}")

    if keystroke_lines:
        sections.append(
            "### Source A — sent-prompt snapshots (highest confidence)\n"
            + "\n".join(keystroke_lines)
        )
    else:
        sections.append(
            "### Source A — sent-prompt snapshots (highest confidence)\n"
            "(no sent-prompt snapshots captured inside AI tool windows in this window)"
        )

    # ── 2. Focused-input accessibility snapshots ──────────────────────────
    role_clause = "(" + " OR ".join(
        f"fa.focused_role = {_sql_quote(r)}" for r in _PROMPT_INPUT_ROLES
    ) + ")"
    frames_match_sql = _ai_match_sql(
        app_col="f.app_name", title_col="f.window_name", url_col="f.browser_url",
    )
    sql_focused = (
        "SELECT f.timestamp, COALESCE(f.app_name,'') AS app, "
        "COALESCE(f.window_name,'') AS win, COALESCE(f.browser_url,'') AS url, "
        "COALESCE(fa.focused_role,'') AS role, "
        "COALESCE(fa.focused_value,'') AS val "
        "FROM frame_accessibility fa "
        "JOIN frames f ON fa.frame_id = f.id "
        f"WHERE f.timestamp >= {ts_start} AND f.timestamp <= {ts_end} "
        f"AND {role_clause} "
        "AND length(COALESCE(fa.focused_value,'')) >= 5 "
        f"AND {frames_match_sql} "
        "ORDER BY f.timestamp ASC "
        f"LIMIT {int(limit)}"
    )
    focused_lines: list[str] = []
    try:
        body = {"query": sql_focused}
        result = _http_post(f"{API_BASE}/raw_sql", body, timeout=20)
        for row in (result.get("data") or [])[:limit]:
            iso_ts = str(row.get("timestamp", ""))
            ts = format_ts_local(iso_ts)
            val = normalize_capture_text(row.get("val") or "")
            if not val or _is_prompt_noise(val):
                continue
            tool = _classify_tool(
                str(row.get("app", "")), str(row.get("win", "")), str(row.get("url", "")),
            )
            if not tool:
                continue
            if not _is_ai_chat_context(tool, str(row.get("win", "")), str(row.get("url", ""))):
                continue
            verified_set.add(tool)
            preview = val.replace("\n", " ")
            if len(preview) > 600:
                preview = preview[:600] + "...(truncated for agent context)"
            focused_lines.append(
                f"- [{ts}] tool={tool} role={row.get('role','')} "
                f"win={row.get('win','')[:60]}\n"
                f"  input_field> {preview}"
            )
            _upsert_prompt_journal_entry(
                prompts,
                _seen_keys,
                {"ts": ts, "iso_ts": iso_ts, "tool": tool, "text": val, "source": "input_field"},
            )
    except Exception as exc:  # noqa: BLE001
        if verbose:
            echo_stderr(f"  [prompt-journal] focused SQL failed: {exc}")

    if focused_lines:
        sections.append(
            "### Source B — focused input field snapshots (Edit/Document/TextArea roles)\n"
            + "\n".join(focused_lines)
        )

    # ── 3. Element-table input snapshots (P2; only when persist_elements on) ──
    # Covers input boxes that were NOT the focused control at capture time
    # (e.g. multi-pane chat UIs). Empty when the elements table is unused, so
    # this degrades gracefully to Sources A/B above.
    element_role_clause = "(" + " OR ".join(
        f"e.role = {_sql_quote(r)}" for r in _PROMPT_INPUT_ROLES
    ) + ")"
    sql_elements = (
        "SELECT f.timestamp, COALESCE(f.app_name,'') AS app, "
        "COALESCE(f.window_name,'') AS win, COALESCE(f.browser_url,'') AS url, "
        "COALESCE(e.role,'') AS role, COALESCE(e.value,'') AS val "
        "FROM elements e "
        "JOIN frames f ON e.frame_id = f.id "
        f"WHERE f.timestamp >= {ts_start} AND f.timestamp <= {ts_end} "
        f"AND {element_role_clause} "
        "AND length(COALESCE(e.value,'')) >= 5 "
        f"AND {frames_match_sql} "
        "ORDER BY f.timestamp ASC "
        f"LIMIT {int(limit)}"
    )
    element_lines: list[str] = []
    try:
        body = {"query": sql_elements}
        result = _http_post(f"{API_BASE}/raw_sql", body, timeout=20)
        for row in (result.get("data") or [])[:limit]:
            iso_ts = str(row.get("timestamp", ""))
            ts = format_ts_local(iso_ts)
            val = normalize_capture_text(row.get("val") or "")
            if not val or _is_prompt_noise(val):
                continue
            tool = _classify_tool(
                str(row.get("app", "")), str(row.get("win", "")), str(row.get("url", "")),
            )
            if not tool:
                continue
            if not _is_ai_chat_context(tool, str(row.get("win", "")), str(row.get("url", ""))):
                continue
            verified_set.add(tool)
            preview = val.replace("\n", " ")
            if len(preview) > 600:
                preview = preview[:600] + "...(truncated for agent context)"
            element_lines.append(
                f"- [{ts}] tool={tool} role={row.get('role','')} "
                f"win={row.get('win','')[:60]}\n"
                f"  input_field> {preview}"
            )
            _upsert_prompt_journal_entry(
                prompts,
                _seen_keys,
                {"ts": ts, "iso_ts": iso_ts, "tool": tool, "text": val, "source": "input_field"},
            )
    except Exception as exc:  # noqa: BLE001
        if verbose:
            echo_stderr(f"  [prompt-journal] elements SQL failed: {exc}")

    if element_lines:
        sections.append(
            "### Source B2 — input field snapshots from element table (any pane)\n"
            + "\n".join(element_lines)
        )

    verified = sorted(verified_set)
    if verbose:
        echo_stderr(
            f"  [prompt-journal] keystroke_lines={len(keystroke_lines)} "
            f"focused_lines={len(focused_lines)} element_lines={len(element_lines)} "
            f"verified={verified}",        )

    if not verified:
        return ("(no AI-tool activity detected in this time window via keystroke "
                "events or focused input snapshots)", [], [])

    return "\n\n".join(sections), verified, prompts


def _run_prompt_extraction(
    *,
    pipe_body: str,
    skill_text: str,
    context_header: str,
    data_text: str,
    verified: list[str],
    verbose: bool = False,
) -> str:
    """ai-prompt-journal: single-shot prompt extraction over per-tool data.

    Same evidence as the ai-habits prefetch is the most reliable source we
    have for "did the user actually use this AI tool". The model then has to
    pick out only the user-typed prompts (not AI responses) and emit one
    ``## HH:MM — Tool — Topic`` block per prompt — or the literal token
    ``NO_NEW_PROMPTS`` if nothing qualifies. The local app.py handles dedup
    against today's journal file.
    """
    if not verified:
        return "NO_NEW_PROMPTS"

    rules = (
        "You are a DeskMate prompt extraction agent.\n\n"
        "CRITICAL RULES:\n"
        "1. ONLY use the Per-AI-tool data below. NEVER invent prompts, timestamps, tools or topics.\n"
        "2. Every line below starting with '  sent> ' or '  input_field> ' IS A USER PROMPT BY CONSTRUCTION (Enter/send snapshot of the chat input box, or focused chat input). EMIT a block for it unless it is obviously AI-generated prose (multi-paragraph polished writing, fenced code blocks, or text starting with 'Sure!'/\"Here's\"/'Certainly'/'I'd be happy').\n"
        "3. When in doubt, INCLUDE the prompt — false positives are acceptable, missed prompts are not.\n"
        "4. Use only the tools that actually appear in the data: " + ", ".join(verified) + ". The tool label is the value after `tool=` on the same evidence line.\n"
        "5. Every HH:MM timestamp must come from a real entry in the data (use 24h clock derived from its timestamp).\n"
        "6. Output format is strict — one block per prompt, separated by a line containing only '---':\n"
        "       ## HH:MM — Tool — Topic\n"
        "       **Category**: category | **Length**: short|medium|long\n"
        "\n"
        "       > quoted prompt text (multi-line prompts: prefix every line with > )\n"
        "       \n"
        "       ---\n"
        "7. Category MUST be one of: coding, writing, research, brainstorming, analysis, conversation, image-gen, other.\n"
        "8. Length: short (<50 words), medium (50-200), long (200+).\n"
        "9. Topic MUST be a 2-5 word summary DERIVED FROM THE PROMPT CONTENT/INTENT. Never use a window title (e.g. `README.md`) or file path as the topic.\n"
        "10. Deduplicate by the first 80 chars of the prompt body.\n"
        "11. Output ONLY when there are zero `sent>`/`input_field>` evidence lines (or all are obviously AI prose): output exactly one line: NO_NEW_PROMPTS — and nothing else.\n"
        "12. No preamble. No trailing commentary. Either ## blocks or NO_NEW_PROMPTS.\n"
    )
    system_prompt = f"{rules}\n{skill_text}\n"

    user_prompt = (
        f"{context_header}\n"
        f"## Per-AI-tool data (pre-fetched by agent)\n\n{data_text}\n\n"
        f"---\n\n"
        f"## Extraction Instructions\n\n{pipe_body}\n\n"
        f"REMINDER: Every prompt block MUST reference text that literally appears in the data above. "
        f"If you cannot find a real user-typed prompt, output exactly: NO_NEW_PROMPTS"
    )

    if verbose:
        echo_stderr(
            f"  [prompt-journal] verified tools={verified}, "
            f"data lines={data_text.count(chr(10))}",        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response = chat_ollama(messages, tools=None, num_predict=6144)
    content = strip_thinking(response.get("content", "")).strip()

    if not content:
        return "NO_NEW_PROMPTS"
    if "NO_NEW_PROMPTS" in content and not content.lstrip().startswith("##"):
        return "NO_NEW_PROMPTS"
    return content


def _single_shot_report(
    *,
    pipe_body: str,
    skill_text: str,
    context_header: str,
    data_text: str,
    verbose: bool = False,
    extra_rules: str = "",
    start_heading: str = "## Summary",
    num_predict: int = 6144,
    max_data_chars: int = 12000,
) -> str:
    """Python prefetched all data → model writes report in ONE call (no tools).

    More reliable than multi-round tool-calling for <=8B models: the model
    sees all evidence upfront and just needs to write, not plan API calls.
    """
    rules = (
        "You are a DeskMate AI agent.\n\n"
        "CRITICAL RULES:\n"
        "1. ONLY cite data from the Search Results below. NEVER invent apps, files, timestamps, or text.\n"
        "2. If you cannot find evidence for something, DO NOT include it.\n"
        "3. Every timestamp must come from the data. Never guess '2:30 PM' or similar.\n"
        "4. Every file/app/window named in the report must appear in the data.\n"
        "5. Prefer quoting key_texts, timeline, or OCR excerpts verbatim.\n"
        "6. Merge repeated noise (e.g. 'Activate Windows') into one bullet.\n"
        "7. Use minutes from Apps section for time estimates — never guess durations.\n"
        f"8. Start directly with {start_heading}. No preamble.\n"
    )
    if extra_rules:
        rules += f"9. {extra_rules}\n"
    system_prompt = f"{rules}\n{skill_text}\n"

    capped = _cap_prefetch_text(data_text, max_data_chars)
    user_prompt = (
        f"{context_header}\n"
        f"## Search Results (pre-fetched by agent)\n\n{capped}\n\n"
        f"---\n\n"
        f"## Report Instructions\n\n{pipe_body}\n\n"
        f"REMINDER: Every fact in your report MUST reference data above. "
        f"If you write a file name, timestamp, or app — it must appear in Search Results. "
        f"Do NOT fabricate."
    )

    if verbose:
        data_lines = capped.count("\n")
        echo_stderr(
            f"  [single-shot] prompt chars={len(capped)} lines={data_lines}, "
            f"num_predict={num_predict}, timeout={_OLLAMA_CHAT_TIMEOUT}s",        )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    chat_timeout = _OLLAMA_CHAT_TIMEOUT
    try:
        response = chat_ollama(messages, tools=None, num_predict=num_predict, timeout=chat_timeout)
    except TimeoutError:
        raise RuntimeError(
            f"Ollama did not respond within {chat_timeout}s. "
            "Try a shorter time window (--hours 8), set OLLAMA_CHAT_TIMEOUT=900, "
            "or use a faster model."
        ) from None
    content = strip_thinking(response.get("content", ""))

    if not content.startswith("##"):
        messages.append(response)
        messages.append({
            "role": "user",
            "content": f"Start directly with {start_heading}. Only use data from Search Results above.",
        })
        try:
            response = chat_ollama(messages, tools=None, num_predict=num_predict, timeout=chat_timeout)
        except TimeoutError:
            raise RuntimeError(
                f"Ollama did not respond within {chat_timeout}s on the retry pass. "
                "Try --hours 8 or increase OLLAMA_CHAT_TIMEOUT."
            ) from None
        content = strip_thinking(response.get("content", ""))

    return content


def _meets_minimum_tools(session: ToolSession, cfg: PipeToolConfig) -> bool:
    if cfg.require_summary_first and session.summary_calls < 1:
        return False
    if cfg.min_search_before_report > 0 and session.search_calls < cfg.min_search_before_report:
        return False
    return True


def _tool_nudge_message(session: ToolSession, cfg: PipeToolConfig, pipe_name: str) -> str:
    if cfg.require_summary_first and session.summary_calls < 1:
        return (
            "Call activity_summary with start_time and end_time from Context. "
            "Do not write the report yet."
        )
    need = cfg.min_search_before_report - session.search_calls
    if pipe_name == "day-recap":
        return (
            f"Call search (limit={cfg.search_limit}) for the top apps by minutes from "
            f"activity_summary — at least {need} more search(es) with app_name set. "
            "Use OCR/audio hits in the report. Do not write the report yet."
        )
    if pipe_name == "ai-habits":
        return (
            f"Call search once per AI tool label (app_name only, no q): "
            f"{', '.join(AI_HABITS_APP_NAMES)}. "
            f"You still need {need} more search call(s). "
            "Skip tools that return NO USAGE RECORDED. Do not write the report yet."
        )
    return (
        f"Call search at least {need} more time(s) before writing the report."
    )


def _run_tool_driven_agent(
    pipe_md_path: Path,
    *,
    pipe_body: str,
    skill_text: str,
    context_header: str,
    start_iso: str,
    end_iso: str,
    verbose: bool = False,
) -> str:
    """Tool-driven loop: model chooses activity_summary / search via tool calls."""
    pipe_name = pipe_md_path.parent.name
    session = ToolSession(
        start_iso=start_iso,
        end_iso=end_iso,
        pipe_name=pipe_name,
        format_for_llm=True,
    )
    cfg = _pipe_config(pipe_name) or PipeToolConfig(5, 10)
    tool_rules = ""
    if pipe_name == "day-recap":
        tool_rules = (
            "TOOL-DRIVEN DAY RECAP:\n"
            "1. Call activity_summary FIRST with start_time/end_time from Context.\n"
            f"2. You may call search at most {cfg.max_search} times "
            f"(limit<={cfg.search_limit} each) for verbatim OCR/audio or specific apps.\n"
            "3. Do NOT write the report until activity_summary AND at least 2 search calls.\n"
            "4. If data_status is not ok, follow guidance.\n"
            "5. Only report verified OCR, key_texts, snippets, audio, edited_files — "
            "not bare window title changes.\n\n"
        )
    elif pipe_name == "ai-habits":
        apps = ", ".join(AI_HABITS_APP_NAMES)
        tool_rules = (
            "TOOL-DRIVEN AI HABITS:\n"
            f"1. Call search once per label (app_name only, NO q param): {apps}.\n"
            f"2. At most {cfg.max_search} search calls; each resolves to real process names.\n"
            "3. activity_summary: start_time + end_time only (never app_name/q filters).\n"
            "4. You must complete one search per tool label (6 searches) before the report.\n"
            "5. If tool output says NO USAGE RECORDED or substantive_hits=0, omit that tool entirely.\n"
            "6. Minutes only from activity_summary apps[] when process name matches — never guess.\n"
            "7. Web ChatGPT/Gemini/Perplexity often appear as chrome.exe — rely on search hits, not the label alone.\n\n"
        )
    elif pipe_name == "standup-update":
        tool_rules = (
            "TOOL-DRIVEN STANDUP UPDATE:\n"
            "1. Call activity_summary FIRST with start_time/end_time from Context.\n"
            f"2. You may call search at most {cfg.max_search} times (limit<={cfg.search_limit} each), "
            "only to confirm a specific blocker, unfinished task, file or PR.\n"
            "3. Do NOT invent apps, files, repos, or PRs that are not present in tool output.\n"
            "4. After the tools return, write the three-section standup (Yesterday / Today / Blockers) "
            "in under 150 words. Copy-paste ready.\n"
            "5. If no blockers were observed, write 'None' under Blockers.\n\n"
        )
    elif pipe_name == "time-breakdown":
        tool_rules = (
            "TOOL-DRIVEN TIME BREAKDOWN:\n"
            "1. Call activity_summary FIRST with start_time/end_time from Context.\n"
            f"2. You may call search at most {cfg.max_search} times (limit<={cfg.search_limit} each), "
            "only to verify the topic/project/category of a specific app when window titles are ambiguous.\n"
            "3. Compute percentages from the `minutes` field of activity_summary.apps[].\n"
            "4. Do NOT invent apps, projects, or files not present in tool output.\n"
            "5. Write all four sections (By Application / By Category / By Project / Productivity Score) "
            "plus the final **Suggestion:** line.\n\n"
        )
    system_prompt = (
        "You are a DeskMate AI agent with tools: activity_summary, search, frames_export.\n\n"
        f"{tool_rules}"
        "RULES:\n"
        "1. Use tools to fetch data — do not invent content.\n"
        "2. Follow the report template exactly when you have enough data.\n"
        "3. Start the final answer with the first ## heading. No preamble.\n\n"
        f"{skill_text}\n"
    )
    user_prompt = (
        f"{context_header}\n"
        f"- Use start_time and end_time from Context for every tool call.\n\n"
        f"## Report Instructions\n\n{pipe_body}"
    )
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    max_rounds = cfg.max_rounds
    predict = cfg.num_predict

    if verbose:
        echo_stderr(f"  [agent] {pipe_name} → tool-driven loop (max {max_rounds} rounds)")

    for round_idx in range(max_rounds):
        if verbose:
            echo_stderr(
                f"  [agent] round {round_idx + 1}/{max_rounds} "
                f"(summary={session.summary_calls}, search={session.search_calls})",            )
        response = chat_ollama(messages, tools=TOOLS, num_predict=predict)
        messages.append(response)
        tool_calls = llm.extract_tool_calls(response)
        if not tool_calls:
            content = strip_thinking(response.get("content", ""))
            if _meets_minimum_tools(session, cfg) and content.startswith("##"):
                return content
            if not _meets_minimum_tools(session, cfg):
                messages.append({
                    "role": "user",
                    "content": _tool_nudge_message(session, cfg, pipe_name),
                })
                continue
            if content.startswith("##"):
                return content
            continue
        for tc in tool_calls:
            fn = tc.get("function", {})
            fn_name = fn.get("name", "")
            tool_call_id = tc.get("id") or f"call_{round_idx}_{len(messages)}"
            fn_args = _parse_tool_arguments(fn.get("arguments", {}))
            if verbose:
                echo_stderr(
                    f"  [tool]  {fn_name}({json.dumps(fn_args, ensure_ascii=False)[:200]})",                )
            tool_result = execute_tool(fn_name, fn_args, session=session)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "tool_name": fn_name,
                "content": tool_result,
            })

        if not _meets_minimum_tools(session, cfg):
            messages.append({
                "role": "user",
                "content": _tool_nudge_message(session, cfg, pipe_name),
            })

    messages.append({
        "role": "user",
        "content": "Stop calling tools. Write the final markdown report now.",
    })
    final = chat_ollama(messages, tools=None, num_predict=predict)
    return strip_thinking(final.get("content", ""))
