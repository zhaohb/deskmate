"""LLM agent runner for pc_assistant pipe apps.

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
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_APPS_SRC = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_APPS_SRC) not in sys.path:
    sys.path.insert(0, str(_APPS_SRC))

from common import normalize_capture_text  # noqa: E402

from pc_assistant.engine import llm  # noqa: E402
from pc_assistant.engine.activity_summary import format_summary_for_agent  # noqa: E402
from pc_assistant.engine.day_recap_context import format_search_items, format_ts_local  # noqa: E402

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
    # standup-update: aligned with screenpipe — LLM calls activity_summary once
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
    # time-breakdown: aligned with screenpipe — activity_summary first, then
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

# Pipes that let the model drive tool calls in a loop. The local 8B model is
# unreliable at multi-step tool calling and tends to fabricate usage, so
# ai-habits/day-recap use deterministic Python prefetch + single-shot. The
# standup-update pipe is opted in here to align with screenpipe's tool-loop
# implementation, where the LLM autonomously calls `activity_summary` / `search`.
TOOL_DRIVEN_PIPES: frozenset[str] = frozenset({"standup-update", "time-breakdown"})

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
API_BASE = os.environ.get("PC_ASSISTANT_API", "http://127.0.0.1:3030")
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
            "description": "Search pc_assistant screen captures, audio transcriptions and UI events.",
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "FTS search query (optional)"},
                    "content_type": {"type": "string", "enum": ["all", "ocr", "audio", "ui"]},
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
            for k, v in {**base, "content_type": "all", "app_name": proc, "limit": limit}.items()
            if v is not None
        )
        result = _http_get(f"{API_BASE}/search?{params}")
        all_items.extend(result.get("data", []) if isinstance(result, dict) else [])

    q_fb = target.get("q_fallback")
    if q_fb and len(format_search_items(all_items)) == 0:
        tried.append(f"q={q_fb}")
        params = "&".join(
            f"{k}={quote(str(v))}"
            for k, v in {**base, "content_type": "all", "q": q_fb, "limit": limit}.items()
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
                print(f"  [search] {tool_name}: error {exc}", file=sys.stderr)

        if verbose:
            print(f"  [search] {tool_name}: {len(items)} results", file=sys.stderr)

        if not items:
            sections.append(
                f"### Search: app_name={tool_name}\n"
                f"Results: 0 items found.\n"
            )
            continue

        max_text = 450 if limit_per_search >= 10 else 200
        lines = format_search_items(items, max_text=max_text)

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
        print(f"  [export] POST /frames/export fps=1.0 ...", file=sys.stderr)
    try:
        result = _http_post(f"{API_BASE}/frames/export", body, timeout=60)
    except Exception as exc:
        if verbose:
            print(f"  [export] error: {exc}", file=sys.stderr)
        return (
            f"### POST /frames/export — FAILED\n"
            f"Error: {exc}\n\n"
            f"The pc_assistant API may not be running or the time range has no frames."
        )

    success = result.get("success", False)
    file_path = result.get("file_path", "")
    frame_count = result.get("frame_count", 0)
    manifest_path = result.get("manifest_path", "")
    reason = result.get("reason", "")

    if verbose:
        print(f"  [export] success={success}, frames={frame_count}, path={file_path}", file=sys.stderr)

    lines = [f"### POST /frames/export — Result\n"]
    lines.append(f"- success: {success}")
    lines.append(f"- frame_count: {frame_count}")
    if file_path:
        lines.append(f"- file_path: `{file_path}`")
    if manifest_path:
        lines.append(f"- manifest_path: `{manifest_path}`")
    if reason:
        lines.append(f"- reason: {reason}")
    lines.append(f"- time_range: {start} to {end}")
    lines.append(f"- fps: 1.0")

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
            print(f"  [prefetch] error: {exc}", file=sys.stderr)
        return {}


def _do_broad_search(start: str, end: str, verbose: bool = False) -> str:
    """Fetch /activity-summary — aggregated activity bundle."""
    summary = _fetch_activity_summary(start, end, verbose=verbose)
    if not summary:
        return "(Failed to fetch activity data from pc_assistant API.)"

    if verbose:
        audio_n = int((summary.get("audio_summary") or {}).get("segment_count") or 0)
        print(
            f"  [prefetch] summary status={summary.get('data_status')} "
            f"apps={len(summary.get('apps', []))} "
            f"key_texts={len(summary.get('key_texts', []))} "
            f"snippets={len(summary.get('snippets') or [])} audio={audio_n}",
            file=sys.stderr,
        )

    return format_summary_for_agent(summary)


def _do_content_search(
    start: str,
    end: str,
    *,
    limit: int = 20,
    app_name: str | None = None,
    q: str | None = None,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    params = [
        f"content_type=all",
        f"limit={limit}",
        f"start_time={quote(start)}",
        f"end_time={quote(end)}",
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
            label = app_name or q or "all"
            print(f"  [search] {label}: {len(items)} raw", file=sys.stderr)
        return items
    except Exception as exc:
        if verbose:
            print(f"  [search] error: {exc}", file=sys.stderr)
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


def _do_day_recap_prefetch(start: str, end: str, verbose: bool = False) -> str:
    """Rich prefetch: activity-summary + up to 5 supplemental /search calls."""
    summary = _fetch_activity_summary(start, end, verbose=verbose, rich=True)
    if not summary:
        return "(Failed to fetch activity data from pc_assistant API.)"

    sections = [format_summary_for_agent(summary)]

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
            print(f"  [day-recap] app searches ({n}): {top_apps[:n]}", file=sys.stderr)
        extra = _do_per_tool_searches(
            top_apps[:n], start, end,
            limit_per_search=search_limit,
            verbose=verbose,
        )
        if extra.strip():
            sections.append("### Supplemental searches (top apps by minutes)\n\n" + extra)
        searches_left -= n

    if searches_left > 0:
        broad = _do_content_search(
            start, end, limit=25, verbose=verbose,
        )
        lines = format_search_items(broad, max_text=500)
        if verbose:
            print(f"  [day-recap] broad search: {len(lines)} substantive", file=sys.stderr)
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
        lines = format_search_items(items, max_text=400)
        if lines:
            sections.append(
                f"### Topic search: {kw}\n"
                + "\n".join(lines)
            )
        searches_left -= 1

    if verbose:
        print(
            f"  [day-recap] context sections={len(sections)} "
            f"timeline={len(summary.get('timeline') or [])} "
            f"key_texts={len(summary.get('key_texts') or [])}",
            file=sys.stderr,
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
            print(
                f"  [ai-habits] {label}: {len(lines)} substantive "
                f"({span or 'no span'}) (tried {', '.join(tried)})",
                file=sys.stderr,
            )
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


def _fetch_outlook_oauth_messages(verbose: bool = False) -> tuple[str, bool]:
    """Fetch Microsoft Graph messages from connected Outlook accounts."""
    try:
        instances_payload = _http_get(f"{API_BASE}/connections/outlook/instances")
    except Exception as exc:
        if verbose:
            print(f"  [email-digest] Outlook OAuth unavailable: {exc}", file=sys.stderr)
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
            message_count += 1
            subject = msg.get("subject") or "(no subject)"
            sender = msg.get("from") or "(unknown sender)"
            date = msg.get("date") or "(no date)"
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


def _fetch_gmail_oauth_messages(verbose: bool = False) -> tuple[str, bool]:
    """Fetch Gmail API messages from connected Gmail accounts."""
    try:
        instances_payload = _http_get(f"{API_BASE}/connections/gmail/instances")
    except Exception as exc:
        if verbose:
            print(f"  [email-digest] Gmail OAuth unavailable: {exc}", file=sys.stderr)
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
            listed = _http_get(
                f"{API_BASE}/connections/gmail/messages"
                f"?maxResults=10&instance={quote(str(instance))}"
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
            message_count += 1
            subject = msg.get("subject") or "(no subject)"
            sender = msg.get("from") or "(unknown sender)"
            date = msg.get("date") or "(no date)"
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
    """Keep single-shot LLM prompts within a size local models can finish in time."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n...(prefetch truncated for model time budget)"


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
    gmail_oauth_text, has_gmail_oauth_messages = _fetch_gmail_oauth_messages(verbose=verbose)
    if gmail_oauth_text:
        sections.append(gmail_oauth_text)
    if has_gmail_oauth_messages:
        verified.append("Gmail (OAuth)")
    oauth_text, has_oauth_messages = _fetch_outlook_oauth_messages(verbose=verbose)
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
            print(
                f"  [email-digest] {label}: {len(lines)} substantive "
                f"({span or 'no span'}) (tried {', '.join(tried)})",
                file=sys.stderr,
            )
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
    "zoom", "teams", "meet", "google meet",
})


def _fetch_latest_meeting(verbose: bool = False) -> dict[str, Any] | None:
    try:
        meetings = _http_get(f"{API_BASE}/meetings?limit=1")
    except Exception as exc:
        if verbose:
            print(f"  [meeting] list error: {exc}", file=sys.stderr)
        return None
    if isinstance(meetings, list) and meetings:
        return meetings[0]
    return None


def _fetch_meeting_by_id(meeting_id: int, verbose: bool = False) -> dict[str, Any] | None:
    try:
        data = _http_get(f"{API_BASE}/meetings/{meeting_id}")
    except Exception as exc:
        if verbose:
            print(f"  [meeting] fetch id={meeting_id} error: {exc}", file=sys.stderr)
        return None
    return data.get("meeting") if isinstance(data, dict) else None


def _fetch_meeting_transcript(meeting_id: int, verbose: bool = False) -> tuple[str, list[dict[str, Any]]]:
    try:
        data = _http_get(f"{API_BASE}/meetings/{meeting_id}/transcript")
    except Exception as exc:
        if verbose:
            print(f"  [meeting] transcript error: {exc}", file=sys.stderr)
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
            print(f"  [todo-list] meetings list error: {exc}", file=sys.stderr)
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
        print(
            f"  [todo-list] meetings in range={len(in_range)}, "
            f"with transcript={len(with_transcript)}",
            file=sys.stderr,
        )
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
        print(
            f"  [meeting] id={meeting_id} title={current_title!r} "
            f"start={start} end={end}",
            file=sys.stderr,
        )

    transcript_text, segments = _fetch_meeting_transcript(meeting_id, verbose=verbose)
    transcript = _format_meeting_segments(segments)

    if not transcript.strip() and start:
        # Fallback: search audio in the meeting window.
        if verbose:
            print("  [meeting] empty transcript → audio search fallback", file=sys.stderr)
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
        "You are a pc_assistant meeting summarizer.\n\n"
        "RULES:\n"
        "1. Summarize ONLY from the transcript provided. NEVER invent decisions, names, or action items.\n"
        "2. Output a 5-8 word plain-english TITLE on the first line, prefixed exactly with 'TITLE: '.\n"
        "   No quotes, no 'meeting about' prefix.\n"
        "3. Then a blank line, then the summary body starting with '## Summary'.\n"
        "4. Cover: key topics, decisions, and action items (use bullet lists).\n"
        "5. Be concise. If something is unclear in the transcript, omit it.\n\n"
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
    if new_title and current_title.lower() in _GENERIC_MEETING_TITLES:
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
            print(
                f"  [meeting] patched #{meeting_id} "
                f"(title={'set' if title_to_set else 'kept'})",
                file=sys.stderr,
            )
        status = (
            f"_Saved to meeting #{meeting_id}"
            + (f" — title updated to \"{title_to_set}\"" if title_to_set else "")
            + "._"
        )
    except Exception as exc:
        if verbose:
            print(f"  [meeting] patch error: {exc}", file=sys.stderr)
        status = f"_(Could not write back to meeting #{meeting_id}: {exc})_"

    return f"{summary_section}\n\n{status}"


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
            print("  [agent] meeting-summary → find + summarize + patch", file=sys.stderr)
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
            print("  [agent] ai-habits → prefetch + single-shot report", file=sys.stderr)
        data_text, verified = _do_ai_habits_prefetch(start_iso, end_iso, verbose=verbose)
        if verified:
            extra = (
                f"ONLY these AI tools have recorded usage: {', '.join(verified)}. "
                f"List ONLY these tools in 'AI Tools Used' and every other section. "
                f"NEVER mention any AI tool not in this list. Estimate each tool's "
                f"time from its 'active window' span and hit count shown in the data "
                f"— do NOT assign the same round number to every tool, and if a tool "
                f"has only 1 hit say '~few min'."
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
            print("  [agent] email-digest → prefetch + single-shot report", file=sys.stderr)
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
            print("  [agent] todo-list → email + meeting prefetch + single-shot todolist", file=sys.stderr)
        email_text, verified_email = _do_email_digest_prefetch(
            start_iso, end_iso, verbose=verbose, limit_per_tool=3,
        )
        meeting_text, verified_meetings = _do_meeting_todos_prefetch(start_iso, end_iso, verbose=verbose)
        data_text = (
            f"## Email evidence\n\n{email_text}\n\n"
            f"## Meeting evidence\n\n{meeting_text}"
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
                f"owners, or commitments. Meetings without a transcript yield no todos."
            )
            if verified_meetings
            else "MEETING todos: no meeting transcripts were available — do not list any meeting todos."
        )
        empty = not verified_email and not verified_meetings
        extra = (
            (
                "NO email tools and NO meeting transcripts were found. State clearly that no "
                "actionable todos could be extracted in the given time range, then give the Tip. "
                "Do NOT list any todos or invent tasks."
            )
            if empty
            else (
                f"{email_rule} {meeting_rule} "
                f"If a task has no explicit deadline in the data, write `due no date`. "
                f"Deduplicate across sources and tag every bullet's `source:` field correctly."
            )
        )
        return _single_shot_report(
            pipe_body=pipe_body,
            skill_text=skill_text,
            context_header=context_header,
            data_text=data_text,
            verbose=verbose,
            extra_rules=extra,
            start_heading="## Todolist",
            num_predict=4096,
            max_data_chars=10000,
        )

    # Email compose: the local app.py has already baked the verified compose
    # context (provider, account, recipient, intent, optional source message)
    # into the pipe body. We hand it straight to the model — no API prefetch.
    if pipe_md_path.parent.name == "email-compose":
        if verbose:
            print("  [agent] email-compose → single-shot draft", file=sys.stderr)
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
            print("  [agent] ai-prompt-journal → SQL prefetch (deterministic emission)", file=sys.stderr)
        data_text, verified, prompts = _do_prompt_journal_prefetch(start_iso, end_iso, verbose=verbose)
        # Deterministic-first: if SQL prefetch found real user prompts (Source A
        # keystroke / Source B focused input), emit them directly with
        # heuristic category/length/topic. Avoids small-model extraction
        # flakiness — the LLM repeatedly drops or mis-formats blocks for
        # qwen3_8b-class models. The pre-fetch already filtered noise via
        # `_is_prompt_noise`, so these rows are high-confidence.
        if prompts:
            if verbose:
                print(
                    f"  [agent] ai-prompt-journal → deterministic emission of "
                    f"{len(prompts)} prompt(s) from {verified}",
                    file=sys.stderr,
                )
            return _emit_prompt_blocks(prompts)
        # No structured prompts: fall back to the ai-habits-style /search
        # prefetch + LLM extraction so we still try to capture *something*
        # from OCR/UI scraping when keystroke/focused capture is empty.
        if verbose:
            print("  [agent] ai-prompt-journal → no SQL prompts, falling back to ai-habits prefetch + LLM", file=sys.stderr)
        data_text, verified = _do_ai_habits_prefetch(start_iso, end_iso, verbose=verbose)
        return _run_prompt_extraction(
            pipe_body=pipe_body,
            skill_text=skill_text,
            context_header=context_header,
            data_text=data_text,
            verified=verified,
            verbose=verbose,
        )

    # Day-recap: Python prefetch + single-shot LLM report (more reliable for 8B)
    if pipe_md_path.parent.name == "day-recap":
        if verbose:
            print("  [agent] day-recap → prefetch + single-shot report", file=sys.stderr)
        data_text = _do_day_recap_prefetch(start_iso, end_iso, verbose=verbose)
        return _single_shot_report(
            pipe_body=pipe_body,
            skill_text=skill_text,
            context_header=context_header,
            data_text=data_text,
            verbose=verbose,
            extra_rules=(
                "Accomplishments and Key Moments must NOT repeat the same items — "
                "Accomplishments = concrete things finished; Key Moments = specific "
                "things seen/typed/said/heard. Use clock-time timestamps (e.g. '2:30 PM') "
                "exactly as they appear in the data."
            ),
        )

    # Pre-fetch API data before the LLM writes the report (other pipes)
    if has_frames_export:
        # pipe.md says "Use the POST /frames/export endpoint"
        # → pre-execute the export before the LLM writes the report
        if verbose:
            print("  [agent] pipe.md requests POST /frames/export", file=sys.stderr)
        data_text = _do_frames_export(start_iso, end_iso, verbose=verbose)
    elif per_tool_names:
        # pipe.md says "use app_name filter for each tool separately"
        # → do targeted per-tool searches
        if verbose:
            print(f"  [agent] pipe.md requests per-tool search: {per_tool_names}", file=sys.stderr)
        data_text = _do_per_tool_searches(per_tool_names, start_iso, end_iso, verbose=verbose)
    else:
        # pipe.md asks for broad analysis → use /activity-summary
        if verbose:
            print("  [agent] pipe.md requests broad analysis → /activity-summary", file=sys.stderr)
        data_text = _do_broad_search(start_iso, end_iso, verbose=verbose)

    # Step 3: Build prompt (system = skill, user = context + data + instructions)
    system_prompt = (
        "You are a pc_assistant AI agent. You analyze the user's local "
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
            print(f"  [agent] round {round_idx + 1}/{MAX_TOOL_ROUNDS} ...", file=sys.stderr)

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
                print(f"  [tool]  {fn_name}({json.dumps(fn_args, ensure_ascii=False)[:200]})", file=sys.stderr)
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


# ── ai-prompt-journal: screenpipe-aligned, SQL-first prefetch ─────────────

# Display label → matching SQL fragments. Each entry has:
#   "url_like":    list of LIKE patterns matched against browser_url (lowercased)
#   "title_like":  list of LIKE patterns matched against window_title (lowercased)
#   "app_in":      list of process names matched exactly against app_name (lowercased)
# Aligned with screenpipe assets/pipes/ai-prompt-journal/pipe.md tool list.
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
    "Grok": {"url_like": ["%grok.com%", "%x.com/i/grok%"], "title_like": ["%grok%"], "app_in": []},
    "DeepSeek": {"url_like": ["%chat.deepseek.com%", "%deepseek.com%"], "title_like": ["%deepseek%"], "app_in": []},
    "Mistral": {"url_like": ["%chat.mistral.ai%"], "title_like": ["%mistral%"], "app_in": []},
    "Poe": {"url_like": ["%poe.com%"], "title_like": ["%poe%"], "app_in": []},
    "HuggingChat": {"url_like": ["%huggingface.co/chat%"], "title_like": ["%hugging%chat%"], "app_in": []},
    "OpenRouter": {"url_like": ["%openrouter.ai%"], "title_like": ["%openrouter%"], "app_in": []},
    "You.com": {"url_like": ["%you.com/search%", "%you.com/chat%"], "title_like": ["%you.com%"], "app_in": []},
    "Pi": {"url_like": ["%pi.ai%"], "title_like": [], "app_in": []},
    "Cursor": {"url_like": [], "title_like": ["%cursor%"], "app_in": ["cursor.exe"]},
    "LMStudio": {"url_like": [], "title_like": ["%lm studio%"], "app_in": ["lmstudio.exe"]},
    "Ollama": {"url_like": [], "title_like": ["%ollama%"], "app_in": ["ollama.exe"]},
    "Jan": {"url_like": [], "title_like": [], "app_in": ["jan.exe"]},
    "GPT4All": {"url_like": [], "title_like": ["%gpt4all%"], "app_in": ["gpt4all.exe"]},
    "Msty": {"url_like": [], "title_like": ["%msty%"], "app_in": ["msty.exe"]},
    "AnythingLLM": {"url_like": [], "title_like": ["%anythingllm%"], "app_in": ["anythingllm.exe"]},
}

# Accessibility roles that almost always indicate a user-composed text input
# inside the focused AI chat window. Aligned with screenpipe pipe.md Step 2 but
# adapted to pc_assistant's UIA control-name strings (note the ``Control``
# suffix used by ``pc_assistant/a11y/uia_tree.py``) plus the bare
# AX/macOS-style names kept for forward compatibility.
_PROMPT_INPUT_ROLES = (
    # pc_assistant UIA names
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
    "alt+f1 for terminal accessibility help",
    "for an optimized screen reader experience",
    "screen reader optimized",
    "press tab to focus",
    "working on it",
    "type a message",
    "send a message",
    "ask anything",
    "message chatgpt",
    "the editor is not accessible at this time",
    "to enable screen reader optimized mode",
)


def _is_prompt_noise(text: str) -> bool:
    """Return True if ``text`` looks like accessibility/UI chrome, not a prompt."""
    low = text.strip().lower()
    if not low:
        return True
    for needle in _PROMPT_NOISE_SUBSTRINGS:
        if needle in low:
            return True
    # Pure URL / VS Code internal resource paths — not a prompt.
    if low.startswith(("http://", "https://", "vscode-file://", "file://")) and " " not in low.strip():
        return True
    return False


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


def _classify_tool(app_name: str, window_title: str, browser_url: str) -> str:
    """Return the AI tool label for a row, or '' if unrecognized."""
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


def _short_topic(text: str) -> str:
    """First few non-trivial words of the prompt, capitalized."""
    cleaned = " ".join(normalize_capture_text(text).split())
    # Drop trailing punctuation, take the first ~6 words.
    words = cleaned.split(" ")[:6]
    topic = " ".join(words).strip()
    topic = topic.rstrip(",.;:!?，。；：！？")
    return topic or "Untitled prompt"


def _emit_prompt_blocks(prompts: list[dict[str, str]]) -> str:
    """Deterministically render ``## HH:MM — Tool — Topic`` blocks.

    Used as the LLM-free safety net when the small extraction model returns
    NO_NEW_PROMPTS despite the SQL prefetch having captured real user input.
    """
    if not prompts:
        return "NO_NEW_PROMPTS"
    blocks: list[str] = []
    for p in prompts:
        text = (p.get("text") or "").strip()
        if not text:
            continue
        ts = p.get("ts") or ""
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
    """SQL-first prefetch aligned with screenpipe's ai-prompt-journal pipe.

    Pulls two high-signal sources via ``/raw_sql``:

    1. ``ui_events.event_type='text'`` — keystroke text emitted by
       :mod:`pc_assistant.a11y.input_hooks` after the user finished typing.
       Filtered to AI tool windows/URLs/processes.
    2. ``frame_accessibility.focused_role IN (Edit, Document, AXTextArea, …)``
       joined with ``frames`` matching the same AI tool patterns — captures
       text sitting in the active chat input field even when no flush event
       fired.

    Returns ``(data_text, verified_labels, prompts)`` where ``prompts`` is a
    deduped list of ``{"ts": HH:MM, "tool": label, "text": prompt_body,
    "source": "keystroke"|"input_field"}`` rows usable for deterministic
    block emission when the LLM fails to extract anything.
    """
    sections: list[str] = []
    verified_set: set[str] = set()
    prompts: list[dict[str, str]] = []
    _seen_keys: set[str] = set()

    # ── 1. Keystroke text events in AI windows ────────────────────────────
    match_sql = _ai_match_sql(
        app_col="app_name", title_col="window_title", url_col="browser_url",
    )
    ts_start = _sql_quote(start)
    ts_end = _sql_quote(end)
    sql_keystrokes = (
        "SELECT timestamp, COALESCE(app_name,'') AS app, "
        "COALESCE(window_title,'') AS win, COALESCE(browser_url,'') AS url, "
        "COALESCE(json_extract(data_json,'$.text'), '') AS text "
        "FROM ui_events "
        "WHERE event_type='text' "
        f"AND timestamp >= {ts_start} AND timestamp <= {ts_end} "
        "AND length(COALESCE(json_extract(data_json,'$.text'), '')) >= 5 "
        f"AND {match_sql} "
        "ORDER BY timestamp ASC "
        f"LIMIT {int(limit)}"
    )
    keystroke_lines: list[str] = []
    try:
        body = {"query": sql_keystrokes}
        result = _http_post(f"{API_BASE}/raw_sql", body, timeout=20)
        for row in (result.get("data") or [])[:limit]:
            ts = format_ts_local(str(row.get("timestamp", "")))
            text = normalize_capture_text(row.get("text") or "")
            if not text or _is_prompt_noise(text):
                continue
            tool = _classify_tool(
                str(row.get("app", "")), str(row.get("win", "")), str(row.get("url", "")),
            )
            if not tool:
                continue
            verified_set.add(tool)
            preview = text.replace("\n", " ")
            if len(preview) > 600:
                preview = preview[:600] + "...(truncated for agent context)"
            keystroke_lines.append(
                f"- [{ts}] tool={tool} app={row.get('app','')} "
                f"win={row.get('win','')[:60]} url={(row.get('url','') or '')[:80]}\n"
                f"  keystroke> {preview}"
            )
            _upsert_prompt_journal_entry(
                prompts,
                _seen_keys,
                {"ts": ts, "tool": tool, "text": text, "source": "keystroke"},
            )
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"  [prompt-journal] keystroke SQL failed: {exc}", file=sys.stderr)

    if keystroke_lines:
        sections.append(
            "### Source A — keystroke text events (highest confidence)\n"
            + "\n".join(keystroke_lines)
        )
    else:
        sections.append(
            "### Source A — keystroke text events (highest confidence)\n"
            "(no keystroke text events captured inside AI tool windows in this window)"
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
            ts = format_ts_local(str(row.get("timestamp", "")))
            val = normalize_capture_text(row.get("val") or "")
            if not val or _is_prompt_noise(val):
                continue
            tool = _classify_tool(
                str(row.get("app", "")), str(row.get("win", "")), str(row.get("url", "")),
            )
            if not tool:
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
                {"ts": ts, "tool": tool, "text": val, "source": "input_field"},
            )
    except Exception as exc:  # noqa: BLE001
        if verbose:
            print(f"  [prompt-journal] focused SQL failed: {exc}", file=sys.stderr)

    if focused_lines:
        sections.append(
            "### Source B — focused input field snapshots (Edit/Document/TextArea roles)\n"
            + "\n".join(focused_lines)
        )

    verified = sorted(verified_set)
    if verbose:
        print(
            f"  [prompt-journal] keystroke_lines={len(keystroke_lines)} "
            f"focused_lines={len(focused_lines)} verified={verified}",
            file=sys.stderr,
        )

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
        "You are a pc_assistant prompt extraction agent.\n\n"
        "CRITICAL RULES:\n"
        "1. ONLY use the Per-AI-tool data below. NEVER invent prompts, timestamps, tools or topics.\n"
        "2. Every line below starting with '  keystroke> ' or '  input_field> ' IS A USER PROMPT BY CONSTRUCTION (keyboard hook output or focused chat input). EMIT a block for it unless it is obviously AI-generated prose (multi-paragraph polished writing, fenced code blocks, or text starting with 'Sure!'/\"Here's\"/'Certainly'/'I'd be happy').\n"
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
        "11. Output ONLY when there are zero `keystroke>`/`input_field>` evidence lines (or all are obviously AI prose): output exactly one line: NO_NEW_PROMPTS — and nothing else.\n"
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
        print(
            f"  [prompt-journal] verified tools={verified}, "
            f"data lines={data_text.count(chr(10))}",
            file=sys.stderr,
        )

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
        "You are a pc_assistant AI agent.\n\n"
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
        print(
            f"  [single-shot] prompt chars={len(capped)} lines={data_lines}, "
            f"num_predict={num_predict}, timeout={_OLLAMA_CHAT_TIMEOUT}s",
            file=sys.stderr,
        )

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
            "TOOL-DRIVEN STANDUP UPDATE (screenpipe-aligned):\n"
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
            "TOOL-DRIVEN TIME BREAKDOWN (screenpipe-aligned):\n"
            "1. Call activity_summary FIRST with start_time/end_time from Context.\n"
            f"2. You may call search at most {cfg.max_search} times (limit<={cfg.search_limit} each), "
            "only to verify the topic/project/category of a specific app when window titles are ambiguous.\n"
            "3. Compute percentages from the `minutes` field of activity_summary.apps[].\n"
            "4. Do NOT invent apps, projects, or files not present in tool output.\n"
            "5. Write all four sections (By Application / By Category / By Project / Productivity Score) "
            "plus the final **Suggestion:** line.\n\n"
        )
    system_prompt = (
        "You are a pc_assistant AI agent with tools: activity_summary, search, frames_export.\n\n"
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
        print(f"  [agent] {pipe_name} → tool-driven loop (max {max_rounds} rounds)", file=sys.stderr)

    for round_idx in range(max_rounds):
        if verbose:
            print(
                f"  [agent] round {round_idx + 1}/{max_rounds} "
                f"(summary={session.summary_calls}, search={session.search_calls})",
                file=sys.stderr,
            )
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
                print(
                    f"  [tool]  {fn_name}({json.dumps(fn_args, ensure_ascii=False)[:200]})",
                    file=sys.stderr,
                )
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
