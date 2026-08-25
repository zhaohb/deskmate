"""Reconstruct *how* a study session was spent and what problems arose.

Two independent, session-scoped views that the content recap deliberately does
not produce (it forbids chronological retelling):

* **process** — a deterministic timeline of what the learner was doing
  (watching a lecture / reading / coding / debugging / searching) with time
  allocation, computed from captured frames. No LLM.
* **problems** — errors that surfaced on screen (tracebacks, exceptions, failed
  commands), when they first appeared, whether they were later resolved, and —
  when a local model is available — a short reconstruction of how. Detection is
  deterministic; the "how" is an optional LLM pass that soft-fails to the raw
  evidence trail.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timedelta
from typing import Any

from ..logger import get
from ..workflow.classifier import classify_frame

logger = get("learning.journey")

_BROWSERS = ("chrome", "msedge", "edge", "firefox", "safari", "brave", "vivaldi", "opera", "arc")
_TERMINALS = ("windowsterminal", "powershell", "cmd", "wsl", "bash", "terminal", "conhost")
_EDITORS = ("code", "cursor", "devenv", "pycharm", "webstorm", "intellij", "rider", "clion", "goland", "sublime", "vim", "emacs")
_READERS = ("acrobat", "sumatra", "foxit", "winword", "powerpnt", "wps", "obsidian", "onenote")

# A video lecture rather than generic browsing: host / filename / on-page cues.
_VIDEO_CUE = re.compile(r"(bilibili|youtube|youtu\.be|\.mp4|\bvideo\b|\u5168\u5c4f|\u64ad\u653e|\u8bb2\u5ea7|lecture)", re.I)
# A real search — a search engine's results page or an explicit query, not a
# tracking param like bilibili's "search-card".
_SEARCH_CUE = re.compile(r"(google\.com/search|bing\.com/search|duckduckgo\.com|baidu\.com/s\b|[?&]q=|\u641c\u7d22)", re.I)
# DeskMate's own UI (local dashboard / recap page) is tooling, not courseware.
_TOOL_CUE = re.compile(
    r"(deskmate|127\.0\.0\.1:\d+|localhost:\d+)",
    re.I,
)

# Error signatures. Kept deliberately broad but anchored on real failure words so
# a red button or the word "error" in prose does not, on its own, count.
_ERROR_PATTERNS = re.compile(
    r"(\b[A-Z][A-Za-z]*Error\b"
    r"|\b[A-Z][A-Za-z]*Exception\b"
    r"|ModuleNotFoundError|ImportError|SyntaxError|TypeError|ValueError|NameError"
    r"|KeyError|AttributeError|IndexError|RuntimeError|AssertionError"
    r"|command not found|is not recognized|cannot find|No such file"
    r"|Segmentation fault|error TS\d+|npm ERR!|fatal:"
    r"|\u62a5\u9519|\u5f02\u5e38|\u62a5\u51fa\u9519\u8bef|\u5931\u8d25\uff1a|\u672a\u627e\u5230\u6a21\u5757)",
    re.I,
)

# Evidence that something actually started working again. Used to separate "the
# error was fixed" from "the error scrolled away / the learner gave up", which a
# disappearance alone cannot tell apart.
_SUCCESS_PATTERNS = re.compile(
    r"(Successfully installed|Successfully built|Build succeeded|BUILD SUCCESS"
    r"|\d+ passed\b|all tests passed|\bPASSED\b|\bOK\b\s*$|0 errors?\b|no errors"
    r"|Compiled successfully|Done in \d|\bexit code 0\b|Installation complete"
    r"|\u5b89\u88c5\u6210\u529f|\u6d4b\u8bd5\u901a\u8fc7|\u6784\u5efa\u6210\u529f|\u8fd0\u884c\u6210\u529f|\u6210\u529f\u5b8c\u6210)",
    re.I,
)

# Activity keys → stable display labels (Chinese, matching the UI).
_ACTIVITY_LABELS = {
    "lecture": "\u770b\u8bb2\u5ea7/\u89c6\u9891",
    "read": "\u9605\u8bfb\u8d44\u6599/\u8bfe\u4ef6",
    "search": "\u641c\u7d22\u8d44\u6599",
    "code": "\u7f16\u5199\u4ee3\u7801",
    "debug": "\u7ec8\u7aef/\u8c03\u8bd5",
    "tool": "\u672c\u673a\u5de5\u5177/\u754c\u9762",
    "other": "\u5176\u5b83",
}


def classify_activity(app_name: str, window_name: str = "", browser_url: str = "") -> str:
    """Map one frame to a coarse learning-activity key (see ``_ACTIVITY_LABELS``)."""
    app = (app_name or "").lower()
    blob = f"{window_name or ''} {browser_url or ''}"
    if any(t in app for t in _TERMINALS):
        return "debug"
    if any(e in app for e in _EDITORS):
        return "code"
    if any(b in app for b in _BROWSERS):
        if _TOOL_CUE.search(blob):
            return "tool"
        if _VIDEO_CUE.search(blob):  # a lecture/video host is a stronger signal than a stray query param
            return "lecture"
        if _SEARCH_CUE.search(blob):
            return "search"
        return "read"
    if any(r in app for r in _READERS):
        return "read"
    # Fall back to the shared workflow classifier for anything unmapped.
    return {"coding": "code", "browsing": "read"}.get(classify_frame(app_name, window_name), "other")


def _parse(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(str(ts))
    except (TypeError, ValueError):
        return None


def _hhmm(ts: str) -> str:
    return (str(ts) or "")[11:16]


# Window titles carry the app as a suffix ("… - Google Chrome"); stripping it
# keeps the part that says what was actually open.
_TITLE_SUFFIX = re.compile(
    r"\s*[-—|]\s*(Google Chrome|Microsoft.?Edge|Mozilla Firefox|Firefox|Brave|Vivaldi|Opera"
    r"|Visual Studio Code|Cursor|PyCharm|WebStorm|IntelliJ IDEA|Sublime Text)\s*$",
    re.I,
)


def clean_title(window_name: str) -> str:
    title = " ".join((window_name or "").split())
    for _ in range(2):  # some titles carry two suffixes ("… - VS Code - Cursor")
        title = _TITLE_SUFFIX.sub("", title)
    return title.strip()[:120]


def _dominant_title(titles: Counter[str]) -> str:
    """The title the learner spent most of the segment on, ignoring blanks."""
    for name, _ in titles.most_common():
        if name:
            return name
    return ""


def build_process(frames: list[dict[str, Any]], *, idle_cap_s: float = 150.0, min_segment_s: float = 45.0) -> dict[str, Any]:
    """Deterministic activity timeline + per-activity time allocation.

    ``frames`` are ``{ts, app, window, url}`` in any order. Time between
    consecutive frames is attributed to the earlier frame's activity, capped at
    ``idle_cap_s`` so a long gap (screen locked, stepped away) doesn't inflate a
    single activity.
    """
    rows = sorted(
        ({"dt": _parse(f.get("ts")), "key": classify_activity(f.get("app", ""), f.get("window", ""), f.get("url", "")),
          "ts": f.get("ts"), "app": f.get("app", ""), "window": f.get("window", "")} for f in frames if _parse(f.get("ts"))),
        key=lambda r: r["dt"],
    )
    if not rows:
        return {"segments": [], "allocation": [], "total_min": 0.0}

    alloc: dict[str, float] = {}
    for i, r in enumerate(rows):
        if i + 1 < len(rows):
            gap = (rows[i + 1]["dt"] - r["dt"]).total_seconds()
        else:
            gap = 30.0  # nominal tail for the last frame
        alloc[r["key"]] = alloc.get(r["key"], 0.0) + min(max(gap, 0.0), idle_cap_s)

    segments: list[dict[str, Any]] = []
    for r in rows:
        if segments and segments[-1]["key"] == r["key"]:
            segments[-1]["_last"] = r["dt"]
            segments[-1]["_titles"][clean_title(r["window"])] += 1
        else:
            segments.append({
                "key": r["key"], "_start": r["dt"], "_last": r["dt"], "app": r["app"],
                "start_ts": r["ts"], "_titles": Counter({clean_title(r["window"]): 1}),
            })
    # A segment runs until the next one starts (capped), not until its own last
    # frame — otherwise every segment loses the tail before the switch.
    for i, s in enumerate(segments):
        tail = min((segments[i + 1]["_start"] - s["_last"]).total_seconds(), idle_cap_s) if i + 1 < len(segments) else 30.0
        s["_seconds"] = max(0.0, (s["_last"] - s["_start"]).total_seconds() + max(0.0, tail))

    # Fold away blips (a 5-second alt-tab) so the timeline reads as a narrative;
    # their time is already counted in `allocation`. The final segment is always
    # kept — it only carries a nominal tail, but it is where the session ended.
    kept = [
        s for i, s in enumerate(segments)
        if s["_seconds"] >= min_segment_s or i == len(segments) - 1
    ] or segments
    merged: list[dict[str, Any]] = []
    for s in kept:
        if merged and merged[-1]["key"] == s["key"]:
            merged[-1]["_seconds"] += s["_seconds"]
            merged[-1]["_last"] = s["_last"]
            merged[-1]["_titles"].update(s["_titles"])
        else:
            merged.append(dict(s))

    out_segments = [{
        "start": _hhmm(s["start_ts"]),
        "key": s["key"],
        "label": _ACTIVITY_LABELS.get(s["key"], s["key"]),
        "app": s["app"],
        "title": _dominant_title(s["_titles"]),
        "minutes": round(max(s["_seconds"], 30.0) / 60, 1),
        "start_ts": s["start_ts"],
        "end_ts": s["_last"].isoformat(),
    } for s in merged]
    allocation = sorted(
        ({"key": k, "label": _ACTIVITY_LABELS.get(k, k), "minutes": round(v / 60, 1)} for k, v in alloc.items()),
        key=lambda a: a["minutes"], reverse=True,
    )
    total = round(sum(a["minutes"] for a in allocation), 1)
    return {"segments": out_segments, "allocation": allocation, "total_min": total}


def _error_signature(snippet: str) -> str:
    key = re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", snippet.lower())
    return key[:32]


def _error_source(app_name: str) -> str:
    """Where an error was seen: the learner's own tooling, or course material.

    A lecture about exceptions puts tracebacks on the slides, so a browser/video
    match is very likely *content being taught*, not a failure the learner hit.
    """
    app = (app_name or "").lower()
    if any(t in app for t in _TERMINALS):
        return "terminal"
    if any(e in app for e in _EDITORS):
        return "editor"
    if any(b in app for b in _BROWSERS):
        return "reference"
    return "other"


def detect_errors(
    ocr_rows: list[dict[str, Any]],
    *,
    ended_at: str,
    resolved_grace_min: float = 3.0,
    include_reference: bool = False,
) -> list[dict[str, Any]]:
    """Group on-screen error occurrences into distinct problems, with a verdict.

    ``status`` is three-valued because the evidence genuinely differs:
    ``resolved`` (a success signal followed it in the same app), ``likely_resolved``
    (it simply stopped appearing well before the session ended) and ``unresolved``.
    Collapsing the middle case into "resolved" would claim success for a learner
    who closed the terminal and gave up.
    """
    end_dt = _parse(ended_at)
    groups: dict[str, dict[str, Any]] = {}
    successes: list[tuple[datetime, str]] = []

    for row in ocr_rows:
        text = str(row.get("text") or "")
        ts = row.get("ts")
        dt = _parse(ts)
        if dt is None:
            continue
        source = _error_source(str(row.get("app") or ""))
        if _SUCCESS_PATTERNS.search(text):
            successes.append((dt, source))
        for m in _ERROR_PATTERNS.finditer(text):
            if source == "reference" and not include_reference:
                continue  # a traceback on a lecture slide is not the learner's error
            snippet = text[max(0, m.start() - 24): m.start() + 140].strip()
            # Signature starts at the match so the same error groups regardless
            # of the preceding OCR context ("...call last) ModuleNotFound" vs bare).
            sig = _error_signature(text[m.start(): m.start() + 80])
            if len(sig) < 6:
                continue
            g = groups.get(sig)
            if g is None:
                groups[sig] = {
                    "error": snippet[:180], "first": dt, "last": dt, "first_ts": ts,
                    "count": 1, "source": source, "app": str(row.get("app") or ""),
                }
            else:
                g["count"] += 1
                if dt < g["first"]:
                    g["first"], g["first_ts"] = dt, ts
                if dt > g["last"]:
                    g["last"] = dt
                if g["source"] == "other" and source in ("terminal", "editor"):
                    g["source"], g["app"] = source, str(row.get("app") or "")

    out: list[dict[str, Any]] = []
    for g in sorted(groups.values(), key=lambda x: x["first"]):
        # A success in the same tooling after the last failure is real evidence.
        success_after = next(
            (dt for dt, src in successes if dt > g["last"] and (src == g["source"] or g["source"] == "other")),
            None,
        )
        disappeared = bool(end_dt and (end_dt - g["last"]).total_seconds() > resolved_grace_min * 60)
        if success_after is not None:
            status = "resolved"
        elif disappeared:
            status = "likely_resolved"
        else:
            status = "unresolved"
        out.append({
            "error": g["error"],
            "app": g["app"],
            "source": g["source"],
            "first_seen": _hhmm(g["first_ts"]),
            "last_seen": _hhmm(g["last"].isoformat()),
            "occurrences": g["count"],
            "status": status,
            "resolved": status != "unresolved",
            "success_at": _hhmm(success_after.isoformat()) if success_after else "",
            "_first_dt": g["first"],
            "_last_dt": g["last"],
        })
    return out


_RECON_SYSTEM = (
    "You reconstruct how a coding/study error was handled, from on-screen text (OCR) "
    "and lecture audio in the window around it. Return ONLY JSON: "
    '{"attempts":"what the learner tried, one or two short clauses",'
    '"fix":"the concrete fix if visible, else empty","resolved":true|false}. '
    "Base everything on the evidence. Set resolved=false unless the evidence shows "
    "the command/build/test working afterwards — an error merely scrolling out of "
    "view is not a fix. If nothing shows what was tried, return empty strings. "
    "Prefer the user's language (Chinese if evidence is Chinese)."
)


def _reconstruct(error: dict[str, Any], window_text: str, *, base: str, model: str, timeout: int) -> dict[str, Any]:
    from deskmate.engine.llm import chat_ollama  # noqa: PLC0415
    from .topics_llm import _load_json_object  # noqa: PLC0415

    user = (
        f"ERROR (seen in {error.get('app') or 'unknown app'} at {error.get('first_seen')}):\n{error['error']}\n\n"
        f"EVIDENCE (OCR + audio around it):\n{window_text[:3500]}\n\nReturn the JSON now."
    )
    try:
        msg = chat_ollama(
            [{"role": "system", "content": _RECON_SYSTEM}, {"role": "user", "content": user}],
            base=base, model=model, num_predict=400, timeout=timeout,
        )
        obj = _load_json_object(msg.get("content") or "")
    except Exception as exc:  # noqa: BLE001 — reconstruction is best-effort
        logger.debug("error reconstruction failed: %s", exc)
        obj = {}
    # The model may only *downgrade* the verdict. A confirmed success signal is
    # observed evidence; letting a summarizer promote "stopped showing up" to
    # "fixed" is exactly the false claim this feature must not make.
    status = error["status"]
    if status != "resolved" and "resolved" in obj and not obj.get("resolved"):
        status = "unresolved"
    return {
        "attempts": str(obj.get("attempts") or "").strip()[:300],
        "fix": str(obj.get("fix") or "").strip()[:300],
        "status": status,
    }


def _window_text(
    first: datetime, last: datetime, ocr_rows: list[dict], audio_rows: list[dict],
    *, lead_s: float = 120.0, end_pad_s: float = 600.0,
) -> str:
    """Evidence around an error: a little before (what caused it) and well after
    (the fix often lands minutes later)."""
    lo = first - timedelta(seconds=lead_s)
    hi = last + timedelta(seconds=end_pad_s)
    parts: list[str] = []
    for row in ocr_rows:
        dt = _parse(row.get("ts"))
        if dt and lo <= dt <= hi:
            parts.append(f"[{_hhmm(row.get('ts'))} 屏] {str(row.get('text') or '')[:300]}")
    for row in audio_rows:
        dt = _parse(row.get("ts"))
        if dt and lo <= dt <= hi:
            parts.append(f"[{_hhmm(row.get('ts'))} 音] {str(row.get('text') or '')[:200]}")
    return "\n".join(parts[:60])


_SEGMENT_SYSTEM = (
    "You reconstruct what the learner was actually doing in each stretch of a "
    "study session. For each numbered stretch you get the activity kind, the "
    "window title, and sampled on-screen text and speech from that exact time "
    "range. Return ONLY JSON: "
    '{"segments":[{"i":0,"doing":"what they were doing, one clause",'
    '"detail":"what it was about, one clause","related":true,'
    '"related_why":"why this stretch is or is not part of the study",'
    '"points":["concrete thing covered"]}]}. '
    "`doing` says the concrete action (watching which talk, reading which page, "
    "checking DeskMate, searching which error) — never just the activity kind. "
    "`detail` says what was being learned or inspected. `related` is true only if "
    "this stretch is about the same course/topic as the rest of the session; "
    "DeskMate's own UI, chat, email, or unrelated tabs are related=false. "
    "`related_why` is one short clause. `points` lists techniques/APIs/files/"
    "commands actually visible, 0 for a short stretch and up to 4 for a long one; "
    "each under 40 characters. Use only the evidence — empty is better than invented. "
    "Write in the SAME language as the evidence (Chinese if the evidence is Chinese). "
    "Keep product and API names in their original form."
)

_ARC_SYSTEM = (
    "You write a short wrap-up of one study session from its timeline. Return "
    "ONLY JSON: "
    '{"path":"2-4 sentences: how they studied, in order",'
    '"outcome":"1-2 sentences: what they finished or walked away with",'
    '"related_note":"one clause on any stretch that was not course content, else empty"}. '
    "`path` must walk the stretches in time (first … then … finally …) and name "
    "what was on screen, not just activity kinds. `outcome` answers: after this "
    "session, what was completed (a talk watched, a page read, a recap opened, "
    "a bug left unresolved). Do not invent topics or results absent from the "
    "timeline. Write in the SAME language as the evidence (Chinese if Chinese). "
    "Keep product names in their original form."
)


def _segment_evidence(seg: dict[str, Any], ocr_rows: list[dict], audio_rows: list[dict]) -> str:
    """Sampled speech + screen text from a segment's own time range.

    The sample scales with the stretch: summarizing 25 minutes from the same
    handful of lines as 2 minutes is what makes a long segment read as a vague
    restatement of its title.
    """
    lo, hi = _parse(seg.get("start_ts")), _parse(seg.get("end_ts"))
    if lo is None or hi is None:
        return ""
    minutes = max(1.0, float(seg.get("minutes") or 1))
    speech_n = int(min(18, 6 + minutes / 2))
    budget = int(min(2200, 700 + minutes * 45))
    speech = [str(r.get("text") or "") for r in audio_rows if (d := _parse(r.get("ts"))) and lo <= d <= hi]
    screen = [str(r.get("text") or "") for r in ocr_rows if (d := _parse(r.get("ts"))) and lo <= d <= hi]
    parts: list[str] = []
    used = 0
    # Speech first: during a lecture it says what is being taught, while OCR is
    # mostly player chrome and page furniture.
    for chunk in _spread(speech, speech_n) + _spread(screen, 3):
        piece = " ".join(chunk.split())[:220]
        if len(piece) < 8 or used + len(piece) > budget:
            continue
        parts.append(piece)
        used += len(piece)
    return " / ".join(parts)


def _spread(rows: list[str], n: int) -> list[str]:
    """Evenly sample ``n`` rows so a long stretch isn't summarized from its head."""
    if len(rows) <= n:
        return rows
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)]


def summarize_segments(
    segments: list[dict[str, Any]], ocr_rows: list[dict], audio_rows: list[dict],
    *, base: str, model: str, timeout: int,
) -> dict[int, dict[str, Any]]:
    """One call for the whole timeline — per-segment calls would cost a round trip each."""
    from deskmate.engine.llm import chat_ollama  # noqa: PLC0415
    from .topics_llm import _load_json_object  # noqa: PLC0415

    blocks: list[str] = []
    for i, seg in enumerate(segments):
        evidence = _segment_evidence(seg, ocr_rows, audio_rows)
        if not evidence and not seg.get("title"):
            continue
        blocks.append(
            f"#{i} kind={seg.get('label')} minutes={seg.get('minutes')} title={seg.get('title') or '-'}\nevidence: {evidence or '-'}"
        )
    if not blocks:
        return {}
    try:
        msg = chat_ollama(
            [{"role": "system", "content": _SEGMENT_SYSTEM},
             {"role": "user", "content": "\n\n".join(blocks)[:9000] + "\n\nReturn the JSON now."}],
            base=base, model=model, num_predict=900, timeout=timeout,
        )
        obj = _load_json_object(msg.get("content") or "")
    except Exception as exc:  # noqa: BLE001 — labels are a nicety, never a blocker
        logger.debug("segment summary failed: %s", exc)
        return {}
    out: dict[int, dict[str, Any]] = {}
    for row in obj.get("segments") or []:
        if not isinstance(row, dict):
            continue
        try:
            idx = int(row.get("i"))
        except (TypeError, ValueError):
            continue
        doing = " ".join(str(row.get("doing") or "").split())[:80]
        detail = " ".join(str(row.get("detail") or "").split())[:120]
        related_why = " ".join(str(row.get("related_why") or "").split())[:80]
        related = row.get("related")
        if not isinstance(related, bool):
            related = None
        points = [
            " ".join(str(p).split())[:40]
            for p in (row.get("points") or [])[:4]
            if isinstance(p, (str, int, float)) and str(p).strip()
        ]
        if doing or detail or points or related is not None:
            out[idx] = {
                "doing": doing, "detail": detail, "points": points,
                "related": related, "related_why": related_why,
            }
    return out


def _fallback_related(seg: dict[str, Any]) -> bool:
    """When the model is silent, tooling stretches are off-topic; the rest stay on."""
    return seg.get("key") not in ("tool", "other")


def _fallback_related_why(seg: dict[str, Any]) -> str:
    if seg.get("key") == "tool":
        return "这是 DeskMate 本机界面，不是课件或讲座内容。"
    return ""


def _stamp_fallbacks(seg: dict[str, Any]) -> None:
    seg.setdefault("doing", "")
    seg.setdefault("detail", "")
    seg.setdefault("points", [])
    seg.setdefault("related", _fallback_related(seg))
    seg.setdefault("related_why", _fallback_related_why(seg) if not seg.get("related") else "")


def summarize_arc(
    segments: list[dict[str, Any]],
    ocr_rows: list[dict],
    audio_rows: list[dict],
    *,
    session_title: str = "",
    base: str,
    model: str,
    timeout: int,
) -> dict[str, str]:
    """One wrap-up: how the session unfolded, and what it finished."""
    from deskmate.engine.llm import chat_ollama  # noqa: PLC0415
    from .topics_llm import _load_json_object  # noqa: PLC0415

    if not segments:
        return {}
    lines: list[str] = []
    for i, seg in enumerate(segments):
        evidence = _segment_evidence(seg, ocr_rows, audio_rows)
        related = seg.get("related")
        related_s = "yes" if related is True else "no" if related is False else "unknown"
        lines.append(
            f"#{i} {seg.get('start')} {seg.get('label')} {seg.get('minutes')}m "
            f"title={seg.get('title') or '-'} doing={seg.get('doing') or '-'} "
            f"detail={seg.get('detail') or '-'} related={related_s} "
            f"why={seg.get('related_why') or '-'}\n"
            f"evidence: {evidence or '-'}"
        )
    user = (
        f"SESSION TITLE: {session_title or '-'}\n\n"
        + "\n\n".join(lines)[:8000]
        + "\n\nReturn the JSON now."
    )
    try:
        msg = chat_ollama(
            [{"role": "system", "content": _ARC_SYSTEM}, {"role": "user", "content": user}],
            base=base, model=model, num_predict=500, timeout=timeout,
        )
        obj = _load_json_object(msg.get("content") or "")
    except Exception as exc:  # noqa: BLE001 — wrap-up is a nicety
        logger.debug("session arc failed: %s", exc)
        return {}
    path = " ".join(str(obj.get("path") or "").split())[:400]
    outcome = " ".join(str(obj.get("outcome") or "").split())[:240]
    related_note = " ".join(str(obj.get("related_note") or "").split())[:160]
    if not path and not outcome:
        return {}
    return {"path": path, "outcome": outcome, "related_note": related_note}


def build_journey(
    *,
    frames: list[dict[str, Any]],
    ocr_rows: list[dict[str, Any]],
    audio_rows: list[dict[str, Any]],
    started_at: str,
    ended_at: str,
    use_llm: bool = True,
    max_errors: int = 6,
) -> dict[str, Any]:
    """Assemble the full journey: process timeline + reconstructed problems."""
    process = build_process(frames)
    detected = detect_errors(ocr_rows, ended_at=ended_at)
    # Rank before truncating: an unresolved failure the learner hit repeatedly in
    # their own terminal matters more than the first thing that happened to match.
    ranked = sorted(
        detected,
        key=lambda p: (
            p["status"] == "unresolved",
            p["source"] in ("terminal", "editor"),
            p["occurrences"],
        ),
        reverse=True,
    )
    problems = sorted(ranked[:max_errors], key=lambda p: p["_first_dt"])

    base = model = None
    timeout = 120
    llm_reason = ""
    arc: dict[str, str] = {}
    if use_llm and (problems or process["segments"]):
        try:
            from deskmate.engine.llm import resolve_ollama_settings  # noqa: PLC0415

            base, model, timeout = resolve_ollama_settings()
        except Exception as exc:  # noqa: BLE001
            logger.debug("ollama settings unavailable: %s", exc)
            base = model = None
            llm_reason = "unavailable"

    if not (base and model):
        for seg in process["segments"]:
            _stamp_fallbacks(seg)

    if base and model and process["segments"]:
        details = summarize_segments(
            process["segments"], ocr_rows, audio_rows,
            base=base, model=model, timeout=min(timeout, 180),
        )
        # No details at all means the model never answered (not running, or it
        # returned nothing parseable) — say so rather than show blank rows.
        if not details:
            llm_reason = "unavailable"
        for i, seg in enumerate(process["segments"]):
            got = details.get(i) or {}
            seg["doing"] = got.get("doing", "")
            seg["detail"] = got.get("detail", "")
            seg["points"] = got.get("points", [])
            related = got.get("related")
            seg["related"] = _fallback_related(seg) if related is None else bool(related)
            seg["related_why"] = got.get("related_why", "")
        if details:
            arc = summarize_arc(
                process["segments"], ocr_rows, audio_rows,
                session_title=process["segments"][0].get("title") or "",
                base=base, model=model, timeout=min(timeout, 120),
            )
    elif not use_llm:
        llm_reason = "disabled"

    out_problems: list[dict[str, Any]] = []
    for p in problems:
        recon = {"attempts": "", "fix": "", "status": p["status"]}
        if base and model:
            recon = _reconstruct(
                p, _window_text(p["_first_dt"], p["_last_dt"], ocr_rows, audio_rows),
                base=base, model=model, timeout=min(timeout, 120),
            )
        out_problems.append({
            "error": p["error"],
            "app": p["app"],
            "source": p["source"],
            "first_seen": p["first_seen"],
            "last_seen": p["last_seen"],
            "success_at": p["success_at"],
            "occurrences": p["occurrences"],
            "status": recon["status"],
            "resolved": recon["status"] != "unresolved",
            "attempts": recon["attempts"],
            "fix": recon["fix"],
        })

    start_dt, end_dt = _parse(started_at), _parse(ended_at)
    wall_min = round((end_dt - start_dt).total_seconds() / 60, 1) if start_dt and end_dt else 0.0
    return {
        "started_at": started_at,
        "ended_at": ended_at,
        "process": process,
        "arc": arc,
        "problems": out_problems,
        "llm_note": llm_reason,
        "summary": {
            "total_min": process["total_min"],
            "wall_min": wall_min,
            "problem_count": len(out_problems),
            "resolved_count": sum(1 for p in out_problems if p["status"] == "resolved"),
            "likely_resolved_count": sum(1 for p in out_problems if p["status"] == "likely_resolved"),
            "unresolved_count": sum(1 for p in out_problems if p["status"] == "unresolved"),
        },
    }
