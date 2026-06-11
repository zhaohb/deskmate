"""Todo List Assistant — LLM agent mode.

Builds a single, unified todolist from TWO sources of evidence over the
supplied time range:

* Email — reuses the email-digest per-email-tool prefetch
  (``_do_email_digest_prefetch``): Gmail / Outlook OAuth messages plus local
  screen / UI hits for every other mail tool.
* Meetings — video calls detected in the range (Teams / Zoom / Meet / …) and
  their transcripts, so action items spoken in a call become todos too.

Both evidence blocks are handed to a single-shot extraction that emits a
markdown checklist (`- [ ] ...`) tagging each item's source. The markdown is
written to ``~/.deskmate/apps/todo-list/output/<timestamp>/todo-list.md``
AND parsed into structured rows persisted to the ``todos`` table via
``POST /todos`` so the Todos page can show and check them off.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from deskmate.apps.agent import _http_post, run_agent
from deskmate.apps.common import (
    add_agent_time_args,
    agent_time_kwargs_from_args,
    api_base,
    output_dir,
    run_cli,
    write_markdown,
)
from deskmate.console import echo_stderr
from deskmate.engine.day_recap_context import range_spans_calendar_days

APP_NAME = "todo-list"
PIPE_MD = Path(__file__).with_name("pipe.md")

# Tolerate the formatting variants a small model emits for a checkbox:
#   "- [ ]", "* [x]", "- []", "- [ x ]", "-  [X]  task".
# Capturing an optional x/✓ and ignoring surrounding spaces avoids silently
# dropping (and therefore never storing) otherwise valid todo lines.
_CHECKBOX_RE = re.compile(r"^\s*[-*]\s*\[\s*([xX✓✔])?\s*\]\s*(.+?)\s*$")
# Recognised valid sources (the leading token of `source: <src>:<detail>`).
_VALID_SOURCES = ("email", "meeting", "screen", "manual")
# Field labels we understand inside a `| key: value` segment. Anything else is
# ignored rather than mis-bucketed.
_KNOWN_KEYS = ("from", "due", "source", "priority")
# Legacy em-dash split, kept ONLY as a fallback for output that predates the
# pipe-delimited format. The pipe path is preferred because it never slices a
# task that itself contains an em-dash.
_LEGACY_SPLIT_RE = re.compile(r"\s+[—–]\s+|\s{2,}-\s{2,}")
_PRIORITY_MAP = {
    "high": "H", "medium": "M", "low": "L",
    "h": "H", "m": "M", "l": "L",
    "urgent": "H", "critical": "H", "normal": "M",
    "紧急": "H", "高": "H", "中": "M", "低": "L",
}


def _strip_md(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    return text.strip()


def _field(segments: list[str], key: str) -> str:
    for seg in segments:
        low = seg.strip().lower()
        if low.startswith(key):
            return seg.strip()[len(key):].strip().lstrip(":").strip()
    return ""


def _split_segments(rest: str) -> list[str]:
    """Split a checkbox line into [task, *meta] segments.

    Prefer the pipe `|` delimiter (the current format) — it is unambiguous and
    never cuts a task that contains an em-dash. Fall back to the legacy em-dash
    split only when there is no pipe, so old generated reports still parse."""
    if "|" in rest:
        return [s.strip() for s in rest.split("|")]
    return _LEGACY_SPLIT_RE.split(rest)


def _classify_source(source_detail: str) -> str:
    """Map a `source:` value to one of _VALID_SOURCES.

    Accepts both the labelled form (`email:gmail`, `screen:Code.exe`) and looser
    phrasings; falls back to a substring sniff before giving up. Robust to the
    model omitting the canonical `<src>:` prefix."""
    sd = source_detail.lower()
    head = sd.split(":", 1)[0].strip()
    if head in _VALID_SOURCES:
        return head
    for src in _VALID_SOURCES:
        if src in sd or (src == "meeting" and "meetings" in sd):
            return src
    return ""


def parse_todos(markdown: str) -> list[dict[str, Any]]:
    """Parse the generated checklist into structured todo dicts.

    Each line is `- [ ] <task> | from: … | due: … | source: … | priority: …`.
    Fields are key-anchored, so reordering or a missing field never shifts the
    others, and an em-dash inside the task is preserved (the task is everything
    before the first labelled `|` segment)."""
    todos: list[dict[str, Any]] = []
    for line in markdown.splitlines():
        m = _CHECKBOX_RE.match(line)
        if not m:
            continue
        checked = bool(m.group(1))
        rest = _strip_md(m.group(2))
        if not rest:
            continue
        segments = _split_segments(rest)
        task = _strip_md(segments[0]) if segments else rest
        if not task:
            continue
        meta = segments[1:]
        source_ref = _field(meta, "from")
        due = _field(meta, "due")
        source_detail = _field(meta, "source")
        priority_raw = _field(meta, "priority").lower()

        priority = _PRIORITY_MAP.get(priority_raw, "")
        source = _classify_source(source_detail)
        # Normalize an empty/absent due to "" (the UI hides "no date").
        if due.lower() in ("no date", "none", "n/a"):
            due = ""

        dedup_seed = f"{source_detail}|{source_ref}|{task}".lower()
        dedup_key = "todo-list:" + hashlib.md5(dedup_seed.encode("utf-8")).hexdigest()

        todos.append({
            "text": task,
            "status": "done" if checked else "open",
            "source": source,
            "source_ref": source_ref,
            "source_detail": source_detail,
            "priority": priority,
            "due": due,
            "origin_app": APP_NAME,
            "dedup_key": dedup_key,
        })
    return todos


def _has_custom_range(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "start_time", None) and getattr(args, "end_time", None))


def evidence_window_from_args(args: argparse.Namespace) -> tuple[str, str]:
    """Resolve the activity window used for this extraction run."""
    kwargs = agent_time_kwargs_from_args(args)
    if "start_time" in kwargs:
        return str(kwargs["start_time"]), str(kwargs["end_time"])
    hours = float(kwargs["hours"])
    end = datetime.now().astimezone().replace(microsecond=0)
    start = end - timedelta(hours=hours)
    return start.isoformat(), end.isoformat()


def _persist_todos(
    todos: list[dict[str, Any]],
    *,
    evidence_start: str = "",
    evidence_end: str = "",
    verbose: bool = False,
) -> int:
    if not todos:
        if verbose:
            echo_stderr("  [todo-list] no checkbox lines parsed — nothing to store")
        return 0
    for item in todos:
        item.setdefault("evidence_start", evidence_start)
        item.setdefault("evidence_end", evidence_end)

    # Prefer writing the same SQLite file the API server uses (no HTTP hop).
    try:
        from deskmate.db.manager import DatabaseManager  # noqa: WPS433

        db = DatabaseManager()
        count = 0
        for item in todos:
            db.upsert_todo(
                text=str(item["text"]),
                status=str(item.get("status") or "open"),
                source=str(item.get("source") or ""),
                source_ref=str(item.get("source_ref") or ""),
                source_detail=str(item.get("source_detail") or ""),
                meeting_id=item.get("meeting_id"),
                priority=str(item.get("priority") or ""),
                due=str(item.get("due") or ""),
                origin_app=str(item.get("origin_app") or APP_NAME),
                evidence_start=str(item.get("evidence_start") or ""),
                evidence_end=str(item.get("evidence_end") or ""),
                dedup_key=str(item.get("dedup_key") or ""),
            )
            count += 1
        if verbose:
            echo_stderr(f"  [todo-list] stored {count} todo(s) in {db.path}")
        return count
    except Exception as exc:  # noqa: BLE001
        if verbose:
            echo_stderr(f"  [todo-list] direct DB store failed ({exc}), trying API…")

    try:
        resp = _http_post(f"{api_base()}/todos", {"todos": todos})
    except Exception as exc:  # noqa: BLE001
        echo_stderr(f"  [todo-list] ERROR: could not persist todos: {exc}")
        return 0
    count = int(resp.get("count", 0)) if isinstance(resp, dict) else 0
    if verbose:
        echo_stderr(f"  [todo-list] persisted {count} structured todo(s) via /todos")
    return count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract a unified todolist from email + meeting activity via LLM agent."
    )
    add_agent_time_args(parser, default_hours=24)
    parser.add_argument("--model", type=str, default=None, help="Ollama model override.")
    parser.add_argument(
        "--no-store",
        action="store_true",
        help="Only write markdown; do not persist structured todos to the database.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print agent rounds to stderr.")
    args = parser.parse_args()

    if args.model:
        from deskmate.apps import agent
        agent.OLLAMA_MODEL = args.model

    ev_start, ev_end = evidence_window_from_args(args)
    report = run_agent(PIPE_MD, verbose=args.verbose, **agent_time_kwargs_from_args(args))

    if _has_custom_range(args) or range_spans_calendar_days(ev_start, ev_end):
        report = report.rstrip() + f"\n\n---\n_时间窗：{ev_start} → {ev_end}_\n"

    out = output_dir(APP_NAME)
    write_markdown(out / "todo-list.md", report)

    if not args.no_store:
        todos = parse_todos(report)
        stored = _persist_todos(
            todos,
            evidence_start=ev_start,
            evidence_end=ev_end,
            verbose=args.verbose,
        )
        if not todos and "## Todolist" in report:
            echo_stderr(
                "  [todo-list] WARNING: report generated but no '- [ ]' lines parsed for DB",
            )
        elif todos and stored == 0:
            echo_stderr("  [todo-list] ERROR: parsed todos but none were stored")
            return 1

    print(out / "todo-list.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
