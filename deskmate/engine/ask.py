"""Ask agent — answer natural-language questions about recorded activity.

Flow (mirrors the agentic Ask pattern):
  1. User asks a question in plain Chinese / English
  2. LLM (Ollama) receives question + tool definitions + SKILL context
  3. LLM autonomously calls search / activity_summary tools via tool_calls
  4. Python executes the API calls, feeds results back
  5. LLM generates a final answer grounded in evidence
  6. Answer + tool call log returned to the caller
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote

from . import llm
from .llm import http_get as _http_get
from .. import paths

OLLAMA_BASE, OLLAMA_MODEL, _OLLAMA_CHAT_TIMEOUT = llm.resolve_ollama_settings()
MAX_ROUNDS = 8

# Sentinel the model prepends when a question is about DeskMate itself
# (capabilities, greeting, how-to) and so needs NO data lookup. Letting the
# model declare this — instead of us guessing from keywords — keeps the
# data-grounding gate from dropping a perfectly good capability answer, while
# still requiring tool evidence for questions about the user's recorded data.
_NO_LOOKUP_TAG = "[[NO_DATA_NEEDED]]"

_TRUNC_TAIL_RE = re.compile(r"(?:\.{2,}|…)\s*$")
_TRUNC_LEAD_RE = re.compile(r"^\s*(?:\.{2,}|…)\s*")
_SUMMARY_LEAD_RE = re.compile(
    r"^(\s*(?:\d+[\.\)]\s*)?)"
    r"((?:\*\*)?(?:摘要|Snippet|Summary|预览)(?:\*\*)?[：:])\s*"
    r"(?:\.{2,}|…)\s*",
    re.IGNORECASE,
)

# SKILL.md is shared infrastructure shipped with the built-in apps.
SKILL_PATH = paths.builtin_apps_dir() / "SKILL.md"

ASK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search screen captures (OCR), audio transcriptions and UI events. "
                "Use for specific keyword matches or targeted queries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "Search query (optional)"},
                    "content_type": {"type": "string", "enum": ["all", "ocr", "audio", "ui", "element"]},
                    "role": {"type": "string", "description": "With content_type=element, filter UI controls by UIA role (e.g. EditControl, ButtonControl, DocumentControl)"},
                    "app_name": {"type": "string", "description": "Filter by process name"},
                    "start_time": {"type": "string", "description": "ISO 8601 start"},
                    "end_time": {"type": "string", "description": "ISO 8601 end"},
                    "limit": {"type": "integer", "description": "Max results (default 10)"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_meetings",
            "description": (
                "List VIDEO CALLS detected by deskmate (Teams, Zoom, Google Meet, "
                "Webex, Slack Huddle, Discord voice, etc.) with start/end times and "
                "transcript segment counts. Use for questions about 会议 / 开会 / 视频会议 / "
                "参加了什么会 — NOT for browser tabs, Google Cloud console, or calendar apps "
                "unless they were an active call. Call this FIRST for meeting questions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {
                        "type": "string",
                        "description": "ISO 8601 — only meetings overlapping this instant or later.",
                    },
                    "end_time": {
                        "type": "string",
                        "description": "ISO 8601 — only meetings overlapping this instant or earlier.",
                    },
                    "limit": {"type": "integer", "description": "Max meetings (default 20)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "meeting_transcript",
            "description": (
                "Read the FULL transcript of ONE meeting by id (from list_meetings). "
                "Returns speaker-labeled segments and the joined text. Use this before "
                "summarizing a meeting, extracting action items / TODOs, or quoting what "
                "was said. Only meetings with transcript_length > 0 have content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "meeting_id": {
                        "type": "integer",
                        "description": "Meeting id returned by list_meetings.",
                    },
                },
                "required": ["meeting_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "activity_summary",
            "description": (
                "Broad activity overview — apps with time, windows, key_texts, "
                "timeline, snippets, edited_files, audio. "
                "Use for general 'what was I doing' questions — NOT for video-meeting "
                "questions (use list_meetings). Call first for time-range activity, then search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "string", "description": "ISO 8601 start (required)"},
                    "end_time": {"type": "string", "description": "ISO 8601 end (required)"},
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
                "screen frames, audio transcripts, input (clicks / typed text), clipboard, "
                "and window focus/title changes — each row carries source, kind, app, a short "
                "summary and a confidence score. Use this for STRONGLY time-ordered, "
                "cross-source questions like 'what did I do step by step between X and Y', "
                "'what did I copy / paste just now', or 'what did I type during the meeting'. "
                "Prefer 'search' for keyword lookups and 'activity_summary' for aggregate stats."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "string", "description": "ISO 8601 start (optional)"},
                    "end_time": {"type": "string", "description": "ISO 8601 end (optional)"},
                    "sources": {
                        "type": "string",
                        "description": (
                            "Comma-separated subset of sources to include: "
                            "screen,audio,input,clipboard,window. Omit for all sources."
                        ),
                    },
                    "limit": {"type": "integer", "description": "Max events, newest first (default 100, max 1000)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "email_search",
            "description": (
                "Search the user's CONNECTED mailbox (Gmail / Outlook) over OAuth and "
                "return real messages (subject, from, date, preview text). Use this for any "
                "question about emails received/sent, senders, or inbox content — NOT the "
                "screen 'search' tool. Returns message ids you can pass to email_read."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Mailbox search query. LEAVE EMPTY to get the most RECENT / LATEST "
                            "messages. Only set it to filter by keyword/sender/subject, e.g. "
                            "'from:alice', 'subject:report', 'invoice'. Do NOT pass words like "
                            "'recent', 'latest', 'is:recent' — use an empty query instead."
                        ),
                    },
                    "provider": {
                        "type": "string",
                        "enum": ["gmail", "outlook", "all"],
                        "description": "Which mailbox to search. Default 'all' connected accounts.",
                    },
                    "start_time": {
                        "type": "string",
                        "description": (
                            "ISO 8601 start of the received/sent time range. Required for "
                            "date-relative email questions like today, yesterday, recent, this week."
                        ),
                    },
                    "end_time": {
                        "type": "string",
                        "description": (
                            "ISO 8601 end of the received/sent time range. Required for "
                            "date-relative email questions like today, yesterday, recent, this week."
                        ),
                    },
                    "limit": {"type": "integer", "description": "Max messages to return (default 8)."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "email_read",
            "description": (
                "Read ONE full email message (including body) by id from a connected "
                "Gmail / Outlook account. Use after email_search to quote exact content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "message_id": {"type": "string", "description": "Message id returned by email_search."},
                    "provider": {
                        "type": "string",
                        "enum": ["gmail", "outlook"],
                        "description": "Which provider the message id belongs to.",
                    },
                    "instance": {"type": "string", "description": "Account email (optional; defaults to first)."},
                },
                "required": ["message_id", "provider"],
            },
        },
    },
]


def _normalize_iso(ts: str | None) -> str | None:
    """Ensure ISO timestamps include local timezone offset for DB comparison."""
    if not ts:
        return ts
    ts = ts.strip()
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return ts
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return dt.isoformat()


def _parse_iso(ts: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
        return dt
    except ValueError:
        return None


def _range_minutes(start: str, end: str) -> float | None:
    s, e = _parse_iso(start), _parse_iso(end)
    if not s or not e:
        return None
    return max(0.0, (e - s).total_seconds() / 60.0)


def _meeting_overlaps_range(
    meeting: dict[str, Any],
    start: str | None,
    end: str | None,
) -> bool:
    """True if meeting interval overlaps [start, end] (open-ended ends allowed)."""
    if not start and not end:
        return True
    m_start = _parse_iso(str(meeting.get("started_at") or ""))
    if not m_start:
        return False
    ended = meeting.get("ended_at")
    if ended:
        m_end = _parse_iso(str(ended))
    else:
        m_end = datetime.now(m_start.tzinfo)
    if not m_end:
        return False
    rs = _parse_iso(_normalize_iso(start) or "") if start else None
    re = _parse_iso(_normalize_iso(end) or "") if end else None
    if rs and m_end <= rs:
        return False
    if re and m_start >= re:
        return False
    return True


def _slim_meeting_row(meeting: dict[str, Any]) -> dict[str, Any]:
    meta = meeting.get("metadata")
    if isinstance(meta, str) and meta.strip():
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            meta = {}
    elif not isinstance(meta, dict):
        meta = {}
    profile = meta.get("profile_name") or meeting.get("name") or ""
    return {
        "id": meeting.get("id"),
        "name": meeting.get("name") or profile or "Meeting",
        "platform": profile,
        "started_at": meeting.get("started_at"),
        "ended_at": meeting.get("ended_at"),
        "segment_count": meeting.get("segment_count") or 0,
        "transcript_length": meeting.get("transcript_length") or 0,
        "note": (meeting.get("note") or "").strip() or None,
        "app_name": meta.get("app_name"),
        "window_title": meta.get("window_title"),
    }


def _execute_list_meetings(arguments: dict[str, Any], api_base: str) -> str:
    arguments = _lower_keys(arguments)
    start = arguments.get("start_time")
    end = arguments.get("end_time")
    if start:
        start = _normalize_iso(str(start))
    if end:
        end = _normalize_iso(str(end))
    try:
        limit = int(arguments.get("limit") or 20)
    except (TypeError, ValueError):
        limit = 20
    limit = max(1, min(limit, 50))

    try:
        status = _http_get(f"{api_base}/meetings/status")
    except Exception:
        status = {}
    active = status.get("meeting") if isinstance(status, dict) else None

    try:
        rows = _http_get(f"{api_base}/meetings?limit={limit}")
    except Exception as exc:
        return json.dumps({"error": str(exc)})

    if not isinstance(rows, list):
        rows = []

    filtered = [
        _slim_meeting_row(m)
        for m in rows
        if isinstance(m, dict) and _meeting_overlaps_range(m, start, end)
    ]

    payload: dict[str, Any] = {
        "definition": (
            "Video calls auto-detected when you were in Teams, Zoom, Meet, Webex, etc. "
            "This is NOT Google Calendar and NOT generic browser window titles."
        ),
        "range": {"start_time": start, "end_time": end},
        "meeting_count": len(filtered),
        "meetings": filtered,
    }
    if active and isinstance(active, dict):
        if not start and not end or _meeting_overlaps_range(active, start, end):
            payload["active_meeting"] = _slim_meeting_row(active)
    if not filtered and not payload.get("active_meeting"):
        payload["hint"] = (
            "No video meetings in this range. Do not invent meetings from Chrome or "
            "console window titles. Offer to check calendar email if the user expected "
            "scheduled meetings, or say recording may have missed a call."
        )
    return _json_for_llm(payload, max_chars=16000)


def _slim_segment(seg: dict[str, Any]) -> dict[str, Any]:
    return {
        "speaker": seg.get("speaker_name") or "Unknown",
        "start": seg.get("start_time"),
        "text": _strip_truncation_marks(str(seg.get("text") or "")),
    }


def _execute_meeting_transcript(arguments: dict[str, Any], api_base: str) -> str:
    arguments = _lower_keys(arguments)
    raw_id = arguments.get("meeting_id") or arguments.get("id")
    try:
        meeting_id = int(raw_id)
    except (TypeError, ValueError):
        return json.dumps({"error": "meeting_id is required and must be an integer"})

    try:
        detail = _http_get(f"{api_base}/meetings/{meeting_id}")
    except Exception as exc:
        return json.dumps({"error": str(exc), "meeting_id": meeting_id})
    meeting = detail.get("meeting") if isinstance(detail, dict) else None
    if not isinstance(meeting, dict):
        return json.dumps({"error": "meeting not found", "meeting_id": meeting_id})

    try:
        tx = _http_get(f"{api_base}/meetings/{meeting_id}/transcript")
    except Exception as exc:
        return json.dumps({"error": str(exc), "meeting_id": meeting_id})

    segments = tx.get("segments", []) if isinstance(tx, dict) else []
    slim_segments = [_slim_segment(s) for s in segments if isinstance(s, dict)]
    full_text = _strip_truncation_marks(str(tx.get("text") or "")) if isinstance(tx, dict) else ""

    payload: dict[str, Any] = {
        "meeting": _slim_meeting_row(meeting),
        "segment_count": len(slim_segments),
        "segments": slim_segments,
        "text": full_text,
    }
    if not slim_segments and not full_text:
        payload["hint"] = (
            "This meeting has no transcript (audio capture/transcription may have been off "
            "or no speech was recorded). Summarize from metadata only and say so; do not "
            "invent dialogue or action items."
        )
    return _json_for_llm(payload, max_chars=28000)


def _is_empty_summary(result: dict[str, Any]) -> bool:
    if not isinstance(result, dict):
        return True
    status = result.get("data_status") or ""
    if status in ("no_capture_in_range", "empty_but_recording"):
        return True
    return (
        not result.get("apps")
        and not result.get("snippets")
        and not result.get("key_texts")
        and (result.get("total_frames") or 0) == 0
    )


def _broaden_range(start: str, end: str, *, hours: float = 1.0) -> tuple[str, str]:
    s, e = _parse_iso(start), _parse_iso(end)
    if not s or not e:
        return start, end
    pad = timedelta(hours=hours)
    return (s - pad).isoformat(), (e + pad).isoformat()


_QUESTION_STOPWORDS = frozenset({
    "what", "when", "where", "who", "how", "did", "do", "i", "the", "a", "an",
    "in", "on", "at", "to", "of", "and", "or", "my", "me", "was", "were", "is",
    "are", "with", "for", "about", "show", "tell", "during", "between",
    "什么", "怎么", "做", "了", "的", "我", "在", "和", "跟", "干", "啥",
})


def _question_terms(question: str) -> set[str]:
    """Content tokens from the question used to score snippet relevance.

    Latin words are lowercased and stopword-filtered; CJK runs of ≥2 chars are
    kept whole (no word spaces) so an app/person/topic mentioned in the question
    can be matched against snippet text.
    """
    if not question:
        return set()
    terms: set[str] = set()
    for w in re.findall(r"[A-Za-z0-9_]+", question.lower()):
        if len(w) >= 2 and w not in _QUESTION_STOPWORDS:
            terms.add(w)
    for run in re.findall(r"[一-鿿]{2,}", question):
        if run not in _QUESTION_STOPWORDS:
            terms.add(run)
    return terms


def _snippet_relevance(item: Any, terms: set[str]) -> int:
    """Overlap score between question terms and a snippet's text/app/speaker."""
    if not terms or not isinstance(item, dict):
        return 0
    blob = " ".join(
        str(item.get(k, "") or "")
        for k in ("text", "app_name", "window_name", "speaker", "speaker_name",
                  "device", "transcription", "content")
    ).lower()
    if not blob:
        return 0
    return sum(1 for t in terms if t in blob)


def _rerank_by_question(items: list[Any], terms: set[str]) -> list[Any]:
    """Stable-sort items by descending question relevance (ties keep order).

    Reranks BEFORE truncation so the kept top-N is the most relevant slice, not
    an arbitrary one — e.g. "2-3pm with Alice in Slack" floats Slack/Alice
    snippets above unrelated apps instead of wasting the small model's context.
    """
    if not terms or len(items) <= 1:
        return items
    return sorted(items, key=lambda it: _snippet_relevance(it, terms), reverse=True)


def _slim_summary(
    result: dict[str, Any], *, max_snippets: int = 6, question: str = "",
) -> dict[str, Any]:
    """Reduce activity-summary payload size for LLM context.

    When ``question`` is given, snippets/key_texts/timeline are reranked by
    relevance to the question before truncation, so the kept slice is the most
    relevant one rather than whatever happened to come first.
    """
    if not isinstance(result, dict):
        return result
    out = dict(result)
    terms = _question_terms(question)
    snippets = out.get("snippets") or []
    if snippets:
        ranked = _rerank_by_question(snippets, terms)
        if len(ranked) > max_snippets:
            out["snippets"] = ranked[:max_snippets]
            out["snippets_truncated"] = True
        else:
            out["snippets"] = ranked
    key_texts = out.get("key_texts") or []
    if key_texts:
        ranked = _rerank_by_question(key_texts, terms)
        if len(ranked) > 12:
            out["key_texts"] = ranked[:12]
            out["key_texts_truncated"] = True
        else:
            out["key_texts"] = ranked
    timeline = out.get("timeline") or []
    if len(timeline) > 15:
        # Timeline stays chronological (it's a sequence); only truncate.
        out["timeline"] = timeline[:15]
        out["timeline_truncated"] = True
    return out


def _json_for_llm(obj: Any, *, max_chars: int = 12000, question: str = "") -> str:
    text = json.dumps(obj, ensure_ascii=False)
    if len(text) <= max_chars:
        return text
    if isinstance(obj, dict) and "broadened" in obj:
        obj = dict(obj)
        obj["broadened"] = _slim_summary(obj["broadened"], max_snippets=4, question=question)
        text = json.dumps(obj, ensure_ascii=False)
    if len(text) <= max_chars:
        return text
    if isinstance(obj, dict):
        obj = _slim_summary(obj, max_snippets=3, question=question)
        text = json.dumps(obj, ensure_ascii=False)
    if len(text) > max_chars:
        text = text[: max_chars - 20] + ',"truncated":true}'
    return text


def _strip_truncation_marks(text: str) -> str:
    """Remove Gmail/LLM truncation markers (... or …) at the start/end of a line."""
    if not text:
        return ""
    out = text.strip()
    while True:
        trimmed = _TRUNC_TAIL_RE.sub("", out).strip()
        if trimmed == out:
            break
        out = trimmed
    out = _TRUNC_LEAD_RE.sub("", out).strip()
    return out


def _email_preview_text(msg: dict[str, Any], *, max_chars: int = 8000) -> str:
    """Prefer full body over API snippet; strip provider truncation ellipses."""
    body = _strip_truncation_marks((msg.get("body") or "").strip())
    snippet = _strip_truncation_marks((msg.get("snippet") or "").strip())
    text = body if len(body) >= len(snippet) else snippet
    if len(text) > max_chars:
        return text[:max_chars]
    return text


def _parse_email_datetime(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None

    try:
        dt = parsedate_to_datetime(raw)
    except Exception:
        dt = None

    if dt is None:
        normalized = raw.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(normalized)
        except ValueError:
            dt = None

    if dt is None:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(raw, fmt)
                break
            except ValueError:
                continue

    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
    return dt.astimezone()


def _email_in_range(msg: dict[str, Any], start_time: str | None, end_time: str | None) -> bool:
    if not start_time and not end_time:
        return True

    msg_dt = _parse_email_datetime(msg.get("date"))
    if msg_dt is None:
        return False

    start_dt = _parse_email_datetime(start_time) if start_time else None
    end_dt = _parse_email_datetime(end_time) if end_time else None
    if start_dt and msg_dt < start_dt:
        return False
    if end_dt and msg_dt >= end_dt:
        return False
    return True


def _gmail_range_query(start_time: str | None, end_time: str | None) -> str:
    """Build Gmail date query terms. Gmail dates are day-granularity; before is exclusive."""
    parts: list[str] = []
    start_dt = _parse_email_datetime(start_time) if start_time else None
    end_dt = _parse_email_datetime(end_time) if end_time else None
    if start_dt:
        parts.append(f"after:{start_dt.strftime('%Y/%m/%d')}")
    if end_dt:
        end_date = end_dt.date()
        if end_dt.time() != datetime.min.time():
            end_date += timedelta(days=1)
        parts.append(f"before:{end_date.strftime('%Y/%m/%d')}")
    return " ".join(parts)


def _infer_question_time_range(question: str, now: datetime) -> tuple[str, str] | None:
    q = question.lower()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # "N hours/minutes ago" (English + Chinese) → a window around that offset.
    rel = _infer_relative_offset(question, q, now)
    if rel is not None:
        return rel

    if "昨天" in question or "yesterday" in q:
        start = today_start - timedelta(days=1)
        end = today_start
    elif "前天" in question or "day before yesterday" in q:
        start = today_start - timedelta(days=2)
        end = today_start - timedelta(days=1)
    elif "今天" in question or "今日" in question or "today" in q:
        start = today_start
        end = now
    elif "上午" in question or "morning" in q or "早上" in question or "今早" in question:
        # Morning of today: 00:00–12:00.
        start = today_start
        end = today_start + timedelta(hours=12)
    elif "下午" in question or "afternoon" in q:
        start = today_start + timedelta(hours=12)
        end = today_start + timedelta(hours=18)
    elif "晚上" in question or "evening" in q or "tonight" in q or "今晚" in question:
        start = today_start + timedelta(hours=18)
        end = today_start + timedelta(hours=24)
    elif "上周" in question or "上週" in question or "last week" in q:
        # ISO week starting Monday; "last week" = the 7 days of the prior week.
        this_week_start = today_start - timedelta(days=today_start.weekday())
        start = this_week_start - timedelta(days=7)
        end = this_week_start
    elif "这周" in question or "这週" in question or "本周" in question or "本週" in question or "this week" in q:
        this_week_start = today_start - timedelta(days=today_start.weekday())
        start = this_week_start
        end = now
    elif "上个月" in question or "上個月" in question or "last month" in q:
        first_this_month = today_start.replace(day=1)
        last_month_end = first_this_month
        prev = (first_this_month - timedelta(days=1)).replace(day=1)
        start = prev
        end = last_month_end
    elif any(token in question for token in ("刚才", "剛才", "最近", "近期")) or "recent" in q or "just now" in q:
        start = now - timedelta(minutes=30)
        end = now
    else:
        return None
    return start.isoformat(), end.isoformat()


# "3 hours ago", "30 minutes ago", "2 天前", "5 小时前", "20 分钟前".
_REL_OFFSET_RE = re.compile(
    r"(\d+)\s*(hours?|hrs?|minutes?|mins?|days?|小时|小時|分钟|分鐘|天)\s*(?:ago|前|之前)",
    re.IGNORECASE,
)


def _infer_relative_offset(question: str, q_lower: str, now: datetime) -> tuple[str, str] | None:
    """Parse "N <unit> ago" / "N <单位>前" into a focused window around that time.

    A relative offset names a point, not a span, so we return a ±30-minute
    window around it (±2h for day-scale offsets) so the tool query stays
    focused instead of scanning everything.
    """
    m = _REL_OFFSET_RE.search(question) or _REL_OFFSET_RE.search(q_lower)
    if not m:
        return None
    try:
        amount = int(m.group(1))
    except ValueError:
        return None
    unit = m.group(2).lower()
    if unit in ("天", "day", "days"):
        center = now - timedelta(days=amount)
        pad = timedelta(hours=2)
    elif unit in ("小时", "小時", "hour", "hours", "hr", "hrs"):
        center = now - timedelta(hours=amount)
        pad = timedelta(minutes=30)
    else:  # minutes
        center = now - timedelta(minutes=amount)
        pad = timedelta(minutes=15)
    start = center - pad
    end = min(now, center + pad)
    return start.isoformat(), end.isoformat()


def _clean_answer_display(text: str) -> str:
    """Remove ellipsis artifacts from the final answer shown in the UI."""
    lines: list[str] = []
    for line in text.splitlines():
        line = _SUMMARY_LEAD_RE.sub(r"\1\2", line)
        line = re.sub(r"^(\s*[-*•]\s*)(?:\.{2,}|…)\s*", r"\1", line)
        line = re.sub(r"^(\s*\d+\.\s+)(?:\.{2,}|…)\s*", r"\1", line)
        line = _TRUNC_TAIL_RE.sub("", line).rstrip()
        lines.append(line)
    return "\n".join(lines)


def _finalize_answer(text: str) -> str:
    # Strip the no-lookup sentinel as a safety net so it never leaks to the UI,
    # even if the model emits it somewhere other than the clean meta-answer path.
    cleaned = (text or "").replace(_NO_LOOKUP_TAG, "")
    return _clean_answer_display(llm.strip_tool_call_markup(cleaned))


def _no_data_answer(
    tool_log: list[dict[str, Any]],
    *,
    api_base: str,
    question: str,
) -> str:
    if not tool_log:
        return (
            "未能完成查询：模型没有调用任何数据工具就想直接作答，"
            "因此无法给出有依据的结果。请把问题说得更具体（例如带上时间范围："
            "「今天上午我在做什么」「最近半小时用了哪些应用」），或重试一次。"
            "若使用 Ollama OpenVINO + 较小模型（如 Qwen3.5 4B）反复出现此情况，"
            "可换用更大的模型以改善工具调用的稳定性。"
        )
    try:
        health = _http_get(f"{api_base}/health")
    except Exception:
        health = {}
    uptime = int(health.get("uptime_seconds") or 0)
    frames = int(health.get("frames") or 0)
    q = question.lower()
    asks_history = any(
        token in question or token in q
        for token in ("昨天", "昨日", "前天", "上周", "yesterday", "last week", "last month")
    )
    if asks_history and uptime < 86_400:
        minutes = max(1, uptime // 60)
        return (
            f"当前录制仅运行约 {minutes} 分钟（共 {frames} 条画面），"
            "还没有「昨天」或更早的屏幕记录。"
            "请保持 deskmate 在后台持续录制后再问历史问题；"
            "若只想看最近活动，可问「刚才在做什么」或「今天用了哪些应用」。"
        )
    return (
        "在所查询的时间范围内没有找到匹配的录屏数据。"
        "请确认录制未暂停，或换一个更近的时间范围再试。"
    )


# ─── grounding / anti-hallucination ──────────────────────────────────────────
# A tool result counts as "carrying real evidence" only if it has more than
# these structural markers. Empty searches, empty meeting lists, connection
# errors and no-mailbox notices all decode to objects that fail this test.
_EMPTY_RESULT_MARKERS = (
    '"result_count":0', '"result_count": 0',
    '"meeting_count":0', '"meeting_count": 0',
    '"event_count":0', '"event_count": 0',
    '"error"', '"no_mailbox_connected"',
    '"data_status":"no_capture', '"data_status": "no_capture',
)


def _result_has_evidence(result_text: str) -> bool:
    """Heuristic: does one tool result contain usable data (not empty/error)?"""
    if not result_text:
        return False
    low = result_text.strip()
    if low in ("{}", "[]", '""'):
        return False
    # An explicit empty/error marker with no offsetting data array.
    has_empty_marker = any(m in low for m in _EMPTY_RESULT_MARKERS)
    has_payload = any(
        key in low
        for key in ('"messages":[{', '"meetings":[{', '"events":[{',
                    '"segments":[{', '"data":[{', '"apps":[{', '"snippets":[{',
                    '"text":"', '"key_texts":[')
    )
    if has_payload:
        return True
    return not has_empty_marker


def _evidence_is_empty(tool_log: list[dict[str, Any]]) -> bool:
    """True when every tool call came back empty / errored — i.e. no grounding
    exists, so any substantive answer would be fabricated."""
    if not tool_log:
        return True
    return not any(_result_has_evidence(str(e.get("result") or "")) for e in tool_log)


# Year-prefixed and bare clock timestamps the model may cite, e.g.
# "2026-06-06T15:04", "15:04", "3:04 PM". We verify each against the evidence.
_ANSWER_TIME_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}"      # ISO datetime
    r"|\b\d{1,2}:\d{2}\s*(?:[AaPp][Mm])?"       # HH:MM or HH:MM AM/PM
)


def _verify_answer_grounding(answer: str, tool_log: list[dict[str, Any]]) -> list[str]:
    """Return clock/timestamps cited in the answer that are NOT present in any
    tool result — these are likely fabricated. Empty list = fully grounded.

    Conservative: only flags time references (the most common and most checkable
    hallucination); normalizes ``HH:MM`` so "3:04 PM" matches an evidence
    "15:04" only when the digits line up. We do not flag prose.
    """
    evidence = "\n".join(str(e.get("result") or "") for e in tool_log)
    if not evidence:
        return []
    suspicious: list[str] = []
    for m in _ANSWER_TIME_RE.findall(answer):
        token = m.strip()
        # Compare on the bare HH:MM digits; that string appears verbatim in
        # ISO timestamps inside tool results when the citation is real.
        hhmm = re.search(r"(\d{1,2}):(\d{2})", token)
        if not hhmm:
            continue
        h, mm = hhmm.group(1), hhmm.group(2)
        variants = {f"{int(h):02d}:{mm}", f"{int(h)}:{mm}"}
        if not any(v in evidence for v in variants):
            suspicious.append(token)
    return suspicious


def _connected_instances(api_base: str, provider: str) -> list[str]:
    """Return connected account emails for a provider, or [] if none/unreachable."""
    try:
        payload = _http_get(f"{api_base}/connections/{provider}/instances")
    except Exception:
        return []
    accounts = payload.get("data", []) if isinstance(payload, dict) else []
    out: list[str] = []
    for acc in accounts:
        if isinstance(acc, dict):
            value = acc.get("instance") or acc.get("email")
            if value:
                out.append(str(value))
    return out


def _email_search_provider(
    api_base: str,
    provider: str,
    query: str | None,
    limit: int,
    start_time: str | None = None,
    end_time: str | None = None,
) -> list[dict[str, Any]]:
    """Search one provider's mailbox and return normalized message dicts."""
    results: list[dict[str, Any]] = []
    for instance in _connected_instances(api_base, provider):
        params = [f"instance={quote(instance)}"]
        provider_query = query
        if provider == "gmail":
            range_query = _gmail_range_query(start_time, end_time)
            provider_query = " ".join(part for part in (query, range_query) if part).strip() or None
        if provider_query:
            params.append(f"q={quote(provider_query)}")
        if provider == "gmail":
            params.append(f"maxResults={limit}")
        else:
            params.append(f"top={limit}")
        try:
            listed = _http_get(f"{api_base}/connections/{provider}/messages?{'&'.join(params)}")
        except Exception as exc:
            results.append({"provider": provider, "account": instance, "error": str(exc)})
            continue
        data = listed.get("data", {}) if isinstance(listed, dict) else {}

        if provider == "outlook":
            messages = data.get("messages", []) if isinstance(data, dict) else []
            for msg in messages[:limit]:
                if not isinstance(msg, dict) or not msg.get("id"):
                    continue
                results.append({
                    "provider": "outlook",
                    "account": instance,
                    "id": msg.get("id"),
                    "from": msg.get("from") or "",
                    "subject": msg.get("subject") or "(no subject)",
                    "date": msg.get("date") or "",
                    "preview": _email_preview_text(msg),
                })
        else:  # gmail returns id refs; fetch each detail
            refs = data.get("messages", []) if isinstance(data, dict) else []
            for ref in refs[:limit]:
                if not isinstance(ref, dict) or not ref.get("id"):
                    continue
                try:
                    detail = _http_get(
                        f"{api_base}/connections/gmail/messages/{quote(str(ref['id']))}"
                        f"?instance={quote(instance)}"
                    )
                except Exception as exc:
                    results.append({"provider": "gmail", "account": instance,
                                    "id": ref.get("id"), "error": str(exc)})
                    continue
                msg = detail.get("data", {}) if isinstance(detail, dict) else {}
                if not isinstance(msg, dict):
                    continue
                results.append({
                    "provider": "gmail",
                    "account": instance,
                    "id": msg.get("id") or ref.get("id"),
                    "from": msg.get("from") or "",
                    "subject": msg.get("subject") or "(no subject)",
                    "date": msg.get("date") or "",
                    "preview": _email_preview_text(msg),
                })
    return [m for m in results if m.get("error") or _email_in_range(m, start_time, end_time)]


# Words the LLM tends to invent as a "query" when it really means "latest".
_RECENCY_NOISE = {
    "recent", "latest", "newest", "new", "is:recent", "recent emails",
    "latest emails", "most recent", "today", "unread emails",
}


def _lower_keys(arguments: dict[str, Any]) -> dict[str, Any]:
    """Small models sometimes capitalize arg names (e.g. 'Limit'); normalize."""
    return {str(k).lower(): v for k, v in arguments.items()}


# Window-title / OCR hints that a frame shows a webmail client. Used by the
# no-OAuth fallback to keep screen evidence scoped to mail activity rather than
# returning arbitrary screen text.
_WEBMAIL_HINTS = (
    "gmail", "收件箱", "inbox", "outlook", "mail.google", "outlook.live",
    "outlook.office", "新邮件", "compose", "撰写",
)


def _looks_like_mail_evidence(text: str) -> bool:
    """True when *text* looks like a webmail window/title.

    Only the leading slice is inspected: a real mail view announces itself in
    the window title (e.g. "收件箱 (383) … Gmail"), whereas an unrelated screen
    (VS Code, a docs page) may merely contain the word "Email" somewhere deep in
    its OCR text — e.g. DeskMate's own "Home Apps Email Timeline" nav bar — which
    must NOT count as the user reading mail.
    """
    low = text[:100].lower()
    return any(h in low for h in _WEBMAIL_HINTS)


def _email_search_fallback(
    api_base: str,
    query: str | None,
    start_time: str | None,
    end_time: str | None,
    *,
    checked_providers: list[str],
) -> str:
    """No mailbox connected: degrade to screen evidence (OCR + UI events).

    The user may have *viewed* webmail in a browser even without OAuth. We can't
    enumerate real messages, but OCR/UI capture often holds window titles and
    on-screen text proving which mail client / inbox was open and when. Return
    that as ``degraded`` evidence with an explicit caveat so the model answers
    from real records instead of refusing outright — while still nudging the
    user to connect OAuth for complete, message-level results.
    """
    # The /search FTS index does NOT parse boolean "A OR B"; issue one query per
    # term. With a user topic (e.g. "NVIDIA") query that directly; otherwise
    # sweep the webmail clients by name. Dedupe by (source, timestamp, text).
    queries = [query] if query else ["Gmail", "收件箱", "Inbox", "Outlook"]
    hits: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for content_type in ("ocr", "ui"):
        for q in queries:
            params = [f"content_type={content_type}", "limit=8", f"q={quote(q)}"]
            if start_time:
                params.append(f"start_time={quote(start_time)}")
            if end_time:
                params.append(f"end_time={quote(end_time)}")
            try:
                result = _http_get(f"{api_base}/search?{'&'.join(params)}")
            except Exception:
                continue
            for row in (result.get("data") if isinstance(result, dict) else None) or []:
                if not isinstance(row, dict):
                    continue
                content = row.get("content") or {}
                window_title = str(content.get("window_title") or "")
                text = str(content.get("text") or window_title)
                # Decide on the highest-signal field per source: UI events carry a
                # clean window_title ("收件箱 … Gmail"); OCR rows only have full text
                # whose leading slice holds the title. A user topic query trusts the
                # search ranking and skips the webmail-shape filter.
                probe = window_title if content_type == "ui" else text
                if not query and not _looks_like_mail_evidence(probe):
                    continue
                ts = str(content.get("timestamp") or row.get("timestamp") or "")
                # Dedupe aggressively: a single inbox view fires many UI events with
                # an identical title. Collapse on (source, window_title) for UI, and
                # on the text head for OCR, keeping the first (newest) occurrence.
                dedupe_on = window_title if content_type == "ui" else text[:80]
                key = (content_type, dedupe_on, "")
                if key in seen:
                    continue
                seen.add(key)
                hits.append({
                    "source": content_type,
                    "timestamp": ts or None,
                    "app_name": content.get("app_name"),
                    "window_title": window_title or None,
                    "text": _strip_truncation_marks(text)[:600],
                })

    return _json_for_llm({
        "degraded": True,
        # NOTE: deliberately avoid the literal "no_mailbox_connected" here — that
        # string is an empty-evidence marker (see _EMPTY_RESULT_MARKERS) and would
        # make the grounding guard discard this degraded result, overriding the
        # model's screen-based answer with the generic no-data message.
        "mailbox_status": "not_connected",
        "note": (
            "No Gmail/Outlook OAuth account is connected, so real messages cannot "
            "be listed. The evidence below is from SCREEN CAPTURE (OCR) and UI "
            "events — it may show which mail client/inbox was open and when, but "
            "NOT a reliable list of individual messages (senders/subjects). "
            "Answer from this screen evidence if relevant, state clearly that it "
            "is based on screen records and may be incomplete, and suggest the "
            "user connect Gmail/Outlook on the Email tab for full results."
        ),
        "range": {"start_time": start_time, "end_time": end_time},
        "checked_providers": checked_providers,
        "screen_evidence_count": len(hits),
        "screen_evidence": hits[:16],
    }, max_chars=20000)


def _execute_email_search(arguments: dict[str, Any], api_base: str) -> str:
    arguments = _lower_keys(arguments)
    provider = str(arguments.get("provider") or "all").strip().lower()
    query = arguments.get("query")
    query = str(query).strip() if query else None
    if query and query.lower() in _RECENCY_NOISE:
        query = None
    start_time = _normalize_iso(str(arguments.get("start_time"))) if arguments.get("start_time") else None
    end_time = _normalize_iso(str(arguments.get("end_time"))) if arguments.get("end_time") else None
    try:
        limit = int(arguments.get("limit") or 8)
    except (TypeError, ValueError):
        limit = 8
    limit = max(1, min(limit, 25))

    providers = ["gmail", "outlook"] if provider == "all" else [provider]
    providers = [p for p in providers if p in ("gmail", "outlook")]
    if not providers:
        return json.dumps({"error": f"unknown provider: {provider}"})

    connected = {p: _connected_instances(api_base, p) for p in providers}
    if not any(connected.values()):
        # Soft-degrade instead of refusing: the user may have viewed webmail in a
        # browser without OAuth, and OCR/UI capture can still show that activity.
        return _email_search_fallback(
            api_base, query, start_time, end_time, checked_providers=providers,
        )

    messages: list[dict[str, Any]] = []
    for p in providers:
        if connected.get(p):
            messages.extend(_email_search_provider(api_base, p, query, limit, start_time, end_time))

    real = [m for m in messages if m.get("id") and not m.get("error")]
    return _json_for_llm({
        "query": query,
        "range": {"start_time": start_time, "end_time": end_time},
        "providers_searched": [p for p in providers if connected.get(p)],
        "result_count": len(real),
        "messages": messages[: limit * len(providers)],
    }, max_chars=32000)


def _execute_email_read(arguments: dict[str, Any], api_base: str) -> str:
    arguments = _lower_keys(arguments)
    provider = str(arguments.get("provider") or "").strip().lower()
    message_id = str(arguments.get("message_id") or "").strip()
    instance = arguments.get("instance")
    if provider not in ("gmail", "outlook"):
        return json.dumps({"error": "provider must be gmail or outlook"})
    if not message_id:
        return json.dumps({"error": "message_id is required"})
    if not instance:
        accounts = _connected_instances(api_base, provider)
        if not accounts:
            return json.dumps({"error": f"no {provider} account connected"})
        instance = accounts[0]
    url = (
        f"{api_base}/connections/{provider}/messages/{quote(message_id)}"
        f"?instance={quote(str(instance))}"
    )
    try:
        payload = _http_get(url)
    except Exception as exc:
        return json.dumps({"error": str(exc), "message_id": message_id, "provider": provider})
    msg = payload.get("data", {}) if isinstance(payload, dict) else {}
    if isinstance(msg, dict):
        msg = dict(msg)
        if msg.get("body"):
            msg["body"] = _strip_truncation_marks(str(msg["body"]))[:8000]
        if msg.get("snippet"):
            msg["preview"] = _email_preview_text(msg)
    return _json_for_llm({"provider": provider, "account": instance, "message": msg})


def _execute_timeline(arguments: dict[str, Any], api_base: str) -> str:
    """Read the unified, time-ordered cross-source context feed."""
    arguments = _lower_keys(arguments)
    start = arguments.get("start_time")
    end = arguments.get("end_time")
    if start:
        start = _normalize_iso(str(start))
    if end:
        end = _normalize_iso(str(end))
    try:
        limit = int(arguments.get("limit") or 100)
    except (TypeError, ValueError):
        limit = 100
    limit = max(1, min(limit, 1000))

    valid_sources = {"screen", "audio", "input", "clipboard", "window"}
    raw_sources = str(arguments.get("sources") or "").strip()
    sources = [s.strip().lower() for s in raw_sources.split(",") if s.strip().lower() in valid_sources]

    params = [f"limit={limit}"]
    if start:
        params.append(f"since={quote(start)}")
    if end:
        params.append(f"until={quote(end)}")
    if sources:
        params.append(f"sources={quote(','.join(sources))}")

    try:
        result = _http_get(f"{api_base}/timeline/unified?{'&'.join(params)}")
    except Exception as exc:
        return json.dumps({"error": str(exc)})

    rows = (result.get("data") if isinstance(result, dict) else None) or []
    events = [
        {
            "ts": r.get("ts"),
            "source": r.get("source"),
            "kind": r.get("kind"),
            "app": r.get("app_name"),
            "window": r.get("window_title"),
            "summary": r.get("summary"),
            "confidence": r.get("confidence"),
        }
        for r in rows
        if isinstance(r, dict)
    ]
    return _json_for_llm({
        "range": {"start_time": start, "end_time": end},
        "sources": sources or "all",
        "event_count": len(events),
        "events": events,
    }, max_chars=24000)


def _execute_tool(name: str, arguments: dict[str, Any], api_base: str, *, question: str = "") -> str:
    """Execute a tool call against the local DeskMate API.

    ``question`` (the user's original ask) is used to rerank activity-summary
    snippets by relevance before truncation so the kept slice is on-topic.
    """
    try:
        if name == "email_search":
            return _execute_email_search(arguments, api_base)
        if name == "email_read":
            return _execute_email_read(arguments, api_base)
        if name == "list_meetings":
            return _execute_list_meetings(arguments, api_base)
        if name == "meeting_transcript":
            return _execute_meeting_transcript(arguments, api_base)
        if name == "timeline":
            return _execute_timeline(arguments, api_base)
        if name == "activity_summary":
            args = dict(arguments)
            if args.get("start_time"):
                args["start_time"] = _normalize_iso(str(args["start_time"]))
            if args.get("end_time"):
                args["end_time"] = _normalize_iso(str(args["end_time"]))

            params = "&".join(
                f"{k}={quote(str(v))}"
                for k, v in args.items()
                if v is not None and k in ("start_time", "end_time", "app_name", "q")
            )
            rich = "&include_snippets=true&include_guidance=true&max_snippets=8&max_snippet_chars=600"
            result = _http_get(f"{api_base}/activity-summary?{params}{rich}")

            start = args.get("start_time", "")
            end = args.get("end_time", "")
            span = _range_minutes(start, end) if start and end else None
            if _is_empty_summary(result) and span is not None and span <= 120:
                broad_start, broad_end = _broaden_range(start, end, hours=1.0)
                broad_args = {**args, "start_time": broad_start, "end_time": broad_end}
                broad_params = "&".join(
                    f"{k}={quote(str(v))}"
                    for k, v in broad_args.items()
                    if v is not None and k in ("start_time", "end_time", "app_name", "q")
                )
                broad_result = _http_get(f"{api_base}/activity-summary?{broad_params}{rich}")
                if not _is_empty_summary(broad_result):
                    result = {
                        "note": (
                            f"Original range {start} to {end} had no captured data. "
                            f"Automatically broadened to {broad_start} to {broad_end}."
                        ),
                        "original_range": {"start": start, "end": end, "empty": True},
                        "broadened": broad_result,
                    }

            text = _json_for_llm(result, question=question)
        elif name == "search":
            args = dict(arguments)
            if args.get("start_time"):
                args["start_time"] = _normalize_iso(str(args["start_time"]))
            if args.get("end_time"):
                args["end_time"] = _normalize_iso(str(args["end_time"]))
            args.setdefault("limit", 10)
            # Prefer hybrid recall; the API ignores this unless semantic search
            # is enabled in config, so it's a safe default.
            args.setdefault("semantic", True)
            params = "&".join(
                f"{k}={quote(str(v))}" for k, v in args.items() if v is not None
            )
            result = _http_get(f"{api_base}/search?{params}")

            start = args.get("start_time", "")
            end = args.get("end_time", "")
            span = _range_minutes(start, end) if start and end else None
            items = (result.get("data") if isinstance(result, dict) else None) or []
            if not items and span is not None and span <= 120 and start and end:
                broad_start, broad_end = _broaden_range(start, end, hours=1.0)
                broad_args = {**args, "start_time": broad_start, "end_time": broad_end}
                broad_params = "&".join(
                    f"{k}={quote(str(v))}" for k, v in broad_args.items() if v is not None
                )
                broad_result = _http_get(f"{api_base}/search?{broad_params}")
                broad_items = (broad_result.get("data") if isinstance(broad_result, dict) else None) or []
                if broad_items:
                    result = {
                        "note": (
                            f"Original range {start} to {end} had no matches. "
                            f"Automatically broadened to {broad_start} to {broad_end}."
                        ),
                        "original_range": {"start": start, "end": end, "empty": True},
                        "data": broad_items,
                        "pagination": broad_result.get("pagination"),
                    }

            text = _json_for_llm(result, question=question)
        else:
            text = json.dumps({"error": f"unknown tool: {name}"})
    except Exception as exc:
        text = json.dumps({"error": str(exc)})
    return text


# Per-round sampling temperature (optimization note 6.4). A small model run at a
# single low temperature gets over-confident about its FIRST (possibly wrong)
# tool choice and loops on it. Exploration rounds run a bit warmer to escape that
# rut; the final answer runs cooler for stable, grounded prose.
_EXPLORE_TEMPERATURE = 0.5
_FINAL_TEMPERATURE = 0.2


def _chat_ollama(
    messages: list[dict],
    tools: list[dict] | None = None,
    *,
    model: str | None = None,
    num_predict: int = 4096,
    temperature: float = _EXPLORE_TEMPERATURE,
) -> dict:
    return llm.chat_ollama(
        messages,
        tools,
        base=OLLAMA_BASE,
        model=model or OLLAMA_MODEL,
        num_predict=num_predict,
        temperature=temperature,
    )


def _grounded_final_answer(
    answer: str,
    tool_log: list[dict[str, Any]],
    *,
    api_base: str,
    question: str,
) -> str:
    """Gate a model answer through the grounding checks before returning it.

    1. No evidence at all → replace with the honest no-data message (a 4B model
       will otherwise pad the gap with plausible-sounding fabrication).
    2. Evidence exists but the answer cites timestamps absent from it → append a
       visible caution rather than silently trusting them.
    """
    if _evidence_is_empty(tool_log):
        return _no_data_answer(tool_log, api_base=api_base, question=question)
    suspicious = _verify_answer_grounding(answer, tool_log)
    if suspicious:
        shown = "、".join(suspicious[:5])
        answer = (
            f"{answer}\n\n"
            f"> ⚠️ 注意：回答中的时间 {shown} 未能在检索到的记录中核实，"
            f"可能不准确，请以下方“数据来源”为准。"
        )
    return answer


def run_ask(
    question: str,
    *,
    api_base: str = "http://127.0.0.1:3030",
    model: str | None = None,
) -> dict[str, Any]:
    """Run the Ask agent: question → tool calls → evidence-based answer.

    Returns {"answer": str, "tool_calls": [...], "error": str|None}.
    """
    skill_text = ""
    if SKILL_PATH.exists():
        skill_text = SKILL_PATH.read_text(encoding="utf-8")

    now = datetime.now().astimezone()
    inferred_email_range = _infer_question_time_range(question, now)
    recording_note = ""
    try:
        health = _http_get(f"{api_base}/health")
        uptime = int(health.get("uptime_seconds") or 0)
        frames = int(health.get("frames") or 0)
        recording_note = (
            f"Recording uptime: {uptime}s ({max(1, uptime // 60)} min), "
            f"frames captured: {frames}. "
            "If uptime is short, historical questions (yesterday / last week) will have no data.\n"
        )
    except Exception:
        pass
    context = (
        f"Current time: {now.isoformat()}\n"
        f"Timezone: {now.tzname()}\n"
        f"{recording_note}"
        f"Today: {now.strftime('%Y-%m-%d %A')}\n"
    )

    system_prompt = (
        "You are DeskMate — a local AI that answers questions about the user's "
        "screen activity, audio transcriptions, apps, files and meetings.\n\n"
        "TOOLS:\n"
        "- list_meetings: VIDEO CALLS (Teams, Zoom, Meet, Webex, …) detected during recording.\n"
        "- meeting_transcript: FULL transcript of one meeting by id (for summaries / action items).\n"
        "- activity_summary / search: recorded SCREEN, audio and UI activity (not mailbox).\n"
        "- email_search / email_read: CONNECTED Gmail / Outlook over OAuth.\n\n"
        "ROUTING — pick the right tool by topic:\n"
        "- Meeting / call questions (English: meeting, call, standup, sync; Chinese: 会议, "
        "开会, 参加了什么会, 视频会议, 通话, 腾讯会议, 飞书) => call list_meetings FIRST "
        "with today's start_time/end_time. NEVER list browser tabs or OAuth/console pages as "
        "meetings. If list_meetings is empty, say no video call was detected — do not substitute "
        "activity_summary window titles.\n"
        "- To SUMMARIZE a meeting or extract its TODOs / action items (会议总结, 会议纪要, "
        "待办, action items): call list_meetings to find the meeting id, then meeting_transcript "
        "to read the full transcript, then write the summary / action items grounded in it.\n"
        "- Email questions (邮件 / 邮箱 / inbox / sender) => email_search FIRST, then email_read.\n"
        "- General activity ('what did I do', apps, files) => activity_summary, then search.\n\n"
        "RULES:\n"
        "1. Use your tools to find evidence BEFORE answering. Never guess.\n"
        "2. Meeting questions => list_meetings. Report platform, start/end, transcript availability. "
        "An empty list means no detected video call, not 'no activity'. For 总结/待办/action items, "
        "read meeting_transcript first; if a meeting has no transcript, say so and summarize from "
        "metadata only — never fabricate dialogue or todos. Format action items as a checklist "
        "with owner + task when the transcript makes them clear.\n"
        "3. Mailbox questions => email_search (real Gmail/Outlook over OAuth). For date-relative "
        "mailbox questions (today, yesterday, recent, this week), ALWAYS pass start_time/end_time "
        "to email_search. To list recent mail, call email_search with an EMPTY query plus the "
        "time range. For a topic like NVIDIA, call email_search with query='NVIDIA' plus the "
        "time range if the user mentioned one.\n"
        "4. Do NOT answer mailbox or meeting questions using activity_summary window titles.\n"
        "5. Always pass start_time/end_time as ISO 8601 WITH timezone offset "
        f"(e.g. {now.strftime('%Y-%m-%d')}T10:00:00{now.strftime('%z')[:3]}:{now.strftime('%z')[3:]}).\n"
        "6. 'recent' = last 30 min. 'today' = since midnight local. "
        "'just now' = last 15 min.\n"
        "7. Hour references like '10 o'clock' mean that local hour today (10:00–10:59).\n"
        "8. If the exact hour has no data but a broadened range finds nearby activity, "
        "say clearly: 'No records for 10:00–10:59, but activity starts at 11:01…' — "
        "cite the actual timestamps.\n"
        "9. If there is truly no data in the range, say recording may have been paused "
        "and mention the nearest captured activity if tool results include it.\n"
        "10. ONLY report what the data shows. Never fabricate apps, files, timestamps, emails, "
        "senders, meetings or subjects. If email_search returns a result with "
        "\"degraded\": true (mailbox_status not_connected), NO OAuth mailbox is connected: "
        "answer from its `screen_evidence` (OCR / UI capture) if relevant — e.g. which mail "
        "client/inbox was open and when — but state explicitly that this is based on screen "
        "records and may be incomplete (you cannot list individual senders/subjects reliably), "
        "and suggest connecting Gmail/Outlook on the Email tab for full results. If "
        "screen_evidence is empty too, just say no mail activity was found on screen and "
        "suggest connecting the mailbox.\n"
        "11. When citing emails, copy the full `preview` field from email_search verbatim "
        "for 摘要/摘要内容. NEVER prefix or suffix with '...' or '…'. Never shorten with ellipsis.\n"
        "12. Answer in the user's language (Chinese or English). Be concise; use bullet points.\n"
        "13. GROUNDING: every factual claim (an app used, a file, what was said, a time) MUST "
        "come from a tool result in this conversation. Cite the supporting timestamp in "
        "parentheses, e.g. '编辑了 config.toml (15:04)'. If the tools returned no relevant "
        "data, reply that you found no matching records for the range — do NOT produce a "
        "plausible-sounding answer from prior knowledge. A confident answer with no tool "
        "evidence is a failure, not a success.\n"
        "14. NO-DATA QUESTIONS: if the question is about DeskMate ITSELF rather than the "
        "user's recorded data — greetings (你好/hi), who you are, what you can do, your "
        "features, or how to use you — answer it directly WITHOUT calling any tool, and begin "
        f"your reply with the exact tag {_NO_LOOKUP_TAG} on its own. Use this tag ONLY for "
        "such self/capability questions; any question about the user's activity, apps, files, "
        "meetings or mail needs real tool data and must NOT use it.\n\n"
        f"## Context\n{context}\n"
        f"{skill_text}\n"
    )

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    tool_log: list[dict[str, Any]] = []

    for round_idx in range(MAX_ROUNDS):
        response = _chat_ollama(messages, tools=ASK_TOOLS, model=model)
        messages.append(response)

        tool_calls = llm.extract_tool_calls(response)
        if not tool_calls:
            raw = response.get("content", "") or ""
            # The model self-declares (via the system-prompt tag) when a question
            # is about DeskMate itself and needs no data lookup — a greeting, "who
            # are you", "what can you do", how-to. Trust that signal: strip the tag
            # and return the answer directly, bypassing the data-grounding gate.
            if _NO_LOOKUP_TAG in raw:
                answer = _finalize_answer(raw.replace(_NO_LOOKUP_TAG, ""))
                if answer:
                    return {"answer": answer, "tool_calls": tool_log, "error": None}
            answer = _finalize_answer(raw)
            # Otherwise it's a data question: with NO tool data yet, a small model
            # sometimes replies with a generic blurb instead of calling a tool. On
            # the first round we nudge it to use its tools and retry (whether or not
            # it emitted prose); an ungrounded answer would just be dropped anyway.
            if not tool_log and round_idx == 0:
                messages.append({
                    "role": "user",
                    "content": (
                        "Please use your tools to look up data before answering. "
                        "For meetings use list_meetings; otherwise activity_summary or search."
                    ),
                })
                continue
            if answer:
                answer = _grounded_final_answer(
                    answer, tool_log, api_base=api_base, question=question
                )
                return {"answer": answer, "tool_calls": tool_log, "error": None}
            empty = _no_data_answer(tool_log, api_base=api_base, question=question)
            return {"answer": empty, "tool_calls": tool_log, "error": None}

        for tc in tool_calls:
            fn = tc.get("function", {})
            fn_name = fn.get("name", "")
            fn_args = fn.get("arguments", {})
            tool_call_id = tc.get("id") or f"call_{len(tool_log)}"
            if isinstance(fn_args, str):
                try:
                    fn_args = json.loads(fn_args) if fn_args.strip() else {}
                except json.JSONDecodeError:
                    fn_args = {}
            if (
                fn_name == "email_search"
                and inferred_email_range
                and not fn_args.get("start_time")
                and not fn_args.get("end_time")
            ):
                fn_args["start_time"], fn_args["end_time"] = inferred_email_range

            tool_result = _execute_tool(fn_name, fn_args, api_base, question=question)
            tool_log.append({
                "tool": fn_name,
                "args": fn_args,
                "result_length": len(tool_result),
                # Kept internally as the evidence pool for grounding checks; the
                # API strips this before returning to the UI (see api.run_ask).
                "result": tool_result,
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "tool_name": fn_name,
                "content": tool_result,
            })

    messages.append({
        "role": "user",
        "content": "Stop calling tools. Write your final answer now based on the data you have.",
    })
    final = _chat_ollama(messages, tools=None, model=model, temperature=_FINAL_TEMPERATURE)
    answer = _finalize_answer(final.get("content", ""))
    if answer:
        answer = _grounded_final_answer(
            answer, tool_log, api_base=api_base, question=question
        )
    else:
        answer = _no_data_answer(tool_log, api_base=api_base, question=question)
    return {"answer": answer, "tool_calls": tool_log, "error": None}
