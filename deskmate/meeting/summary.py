"""Summarize one meeting from its own time span.

The existing summary app finds "the meeting that just ended" and reads only the
transcript segments the detector managed to link. A meeting the user declared by
hand has no linked segments at all, so this works from the meeting's exact
start/end instead: speech in that window, plus the screen text captured
alongside it, which is where agendas, slides and shared documents live.

Deterministic structure, model-written content: the shape of the result never
depends on the model being reachable, and a failed call degrades to the evidence
counts rather than to an empty page.
"""

from __future__ import annotations

import re
from typing import Any

from ..logger import get

logger = get("meeting.summary")

_SYSTEM = (
    "You write the record of a meeting from its transcript and the text captured "
    "from the participants' screens. Return ONLY JSON:\n"
    '{"title":"5-8 word plain title","summary":"3-6 sentences on what the meeting '
    'was about and what came out of it","key_points":["…"],"decisions":["…"],'
    '"todos":[{"text":"action item","owner":"who, or empty","due":"when, or empty"}]}\n'
    "Rules:\n"
    "- Ground everything in the evidence; invent nothing.\n"
    "- `decisions` holds what was actually settled; leave it empty when nothing was.\n"
    "- `todos` holds concrete follow-up actions someone must do. An action needs a "
    "verb and an object — 'discussed the roadmap' is not a todo.\n"
    "- `owner` and `due` must be copied from the evidence. If nobody was named or "
    "no date was said, leave them empty — never guess a person or a date.\n"
    "- Screen text is context (slides, docs, tickets); speech is what was said.\n"
    "- Max 6 key points, 6 decisions, 8 todos.\n"
    "- Write in the SAME language as the evidence.\n"
)


def _clean(text: str, limit: int) -> str:
    return " ".join(str(text or "").split())[:limit]


def _spread(rows: list[str], n: int) -> list[str]:
    """Evenly sample so a long meeting isn't summarized from its opening minutes."""
    if len(rows) <= n:
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)]


def build_evidence(
    transcript_rows: list[dict[str, Any]],
    ocr_rows: list[dict[str, Any]],
    *,
    speech_budget: int = 9000,
    screen_budget: int = 2500,
) -> str:
    """Interleave what was said with what was on screen, both time-stamped."""
    speech: list[str] = []
    used = 0
    for row in transcript_rows:
        line = _clean(row.get("text"), 400)
        if len(line) < 4 or used + len(line) > speech_budget:
            continue
        speech.append(f"[{str(row.get('ts') or '')[11:16]}] {line}")
        used += len(line)

    screen: list[str] = []
    used = 0
    # Screens repeat heavily; sample across the meeting rather than take the head.
    for row in _spread([_clean(r.get("text"), 300) for r in ocr_rows], 12):
        if len(row) < 20 or used + len(row) > screen_budget:
            continue
        screen.append(row)
        used += len(row)

    parts: list[str] = []
    if speech:
        parts.append("SPEECH:\n" + "\n".join(speech))
    if screen:
        parts.append("SCREEN TEXT:\n" + "\n".join(screen))
    return "\n\n".join(parts)


def _parse_reply(obj: dict[str, Any]) -> dict[str, Any]:
    def _list(key: str, limit: int, size: int) -> list[str]:
        return [
            _clean(v, size) for v in (obj.get(key) or [])[:limit]
            if isinstance(v, (str, int, float)) and str(v).strip()
        ]

    todos: list[dict[str, str]] = []
    for row in (obj.get("todos") or [])[:8]:
        if isinstance(row, str):
            row = {"text": row}
        if not isinstance(row, dict):
            continue
        text = _clean(row.get("text"), 200)
        if len(text) < 4:
            continue
        todos.append({
            "text": text,
            "owner": _clean(row.get("owner"), 60),
            "due": _clean(row.get("due"), 60),
        })
    return {
        "title": _clean(obj.get("title"), 90),
        "summary": _clean(obj.get("summary"), 1600),
        "key_points": _list("key_points", 6, 160),
        "decisions": _list("decisions", 6, 160),
        "todos": todos,
    }


_EMPTY = {"title": "", "summary": "", "key_points": [], "decisions": [], "todos": []}


def has_enough_evidence(speech_rows: int, screen_rows: int) -> bool:
    """Whether there is enough of a record to summarize without inventing one.

    A meeting is a conversation. Handed one stray screenshot and no speech, the
    model will still happily produce decisions, owners and due dates that were
    never said — so the floor is checked here rather than hoped for in the
    prompt. Screen-only still qualifies, but it takes a real stretch of it.
    """
    return speech_rows >= 3 or screen_rows >= 8


def build_meeting_summary(
    *,
    transcript_rows: list[dict[str, Any]],
    ocr_rows: list[dict[str, Any]],
    started_at: str,
    ended_at: str,
    name: str = "",
    use_llm: bool = True,
) -> dict[str, Any]:
    """Summary + decisions + follow-ups for one meeting's exact span."""
    evidence = build_evidence(transcript_rows, ocr_rows)
    result = dict(_EMPTY)
    note = ""

    if not evidence or not has_enough_evidence(len(transcript_rows), len(ocr_rows)):
        note = "no_evidence"
    elif not use_llm:
        note = "disabled"
    else:
        try:
            from ..engine.llm import chat_ollama, resolve_ollama_settings  # noqa: PLC0415
            from ..learning_memory.topics_llm import _load_json_object  # noqa: PLC0415

            base, model, timeout = resolve_ollama_settings()
            user = (
                f"MEETING: {name or '(untitled)'}\nFROM {started_at} TO {ended_at}\n\n"
                f"{evidence}\n\nReturn the JSON now."
            )
            msg = chat_ollama(
                [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
                base=base, model=model, num_predict=1200, timeout=min(timeout, 600),
            )
            parsed = _parse_reply(_load_json_object(msg.get("content") or ""))
            if parsed["summary"] or parsed["key_points"] or parsed["todos"]:
                result = parsed
            else:
                note = "unavailable"
        except Exception as exc:  # noqa: BLE001 — a summary is never worth crashing on
            logger.debug("meeting summary failed: %s", exc)
            note = "unavailable"

    return {
        **result,
        "started_at": started_at,
        "ended_at": ended_at,
        "note": note,
        "evidence": {
            "transcript_rows": len(transcript_rows),
            "screen_rows": len(ocr_rows),
        },
    }


def dedup_key(meeting_id: int, text: str) -> str:
    """Stable key so regenerating a summary updates todos instead of duplicating."""
    slug = re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", (text or "").lower())[:60]
    return f"meeting:{meeting_id}:{slug}"
