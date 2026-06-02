"""AI Prompt Journal — LLM agent mode.

Reads pipe.md, sends to Ollama via the agent runner, lets the model
extract prompts the user typed into AI tools and emit one ``## HH:MM — Tool — Topic``
block per prompt. This runner then appends only the genuinely new blocks to
today's journal file at::

    %USERPROFILE%\\.deskmate\\apps\\ai-prompt-journal\\journal\\YYYY-MM-DD.md

Deduplication keys on the first 80 characters of each prompt's body, so the
same prompt captured by repeated hourly runs is not re-appended.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))
from agent import _is_prompt_noise, run_agent  # noqa: E402
from common import (  # noqa: E402
    add_agent_time_args,
    agent_time_kwargs_from_args,
    normalize_capture_text,
    output_dir,
    deskmate_home,
    write_markdown,
)

APP_NAME = "ai-prompt-journal"
PIPE_MD = Path(__file__).with_name("pipe.md")

# Match a single prompt block produced by pipe.md. The block ends at a line
# containing only "---" (or at end-of-text).
_BLOCK_RE = re.compile(
    r"^##\s+(?P<header>.+?)\n(?P<body>.*?)(?:\n---\s*$|\Z)",
    re.MULTILINE | re.DOTALL,
)
_BLOCKQUOTE_LINE_RE = re.compile(r"^>\s?", re.MULTILINE)
_HEADER_DATE_RE = re.compile(r"^(?P<date>\d{4}-\d{2}-\d{2})\s+(?P<rest>.+)$")


def _journal_dir() -> Path:
    out = deskmate_home() / "apps" / APP_NAME / "journal"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _today_journal() -> Path:
    return _journal_dir() / f"{datetime.now().date().isoformat()}.md"


def _journal_for_date(day: date) -> Path:
    return _journal_dir() / f"{day.isoformat()}.md"


def _parse_iso_datetime(iso: str) -> datetime | None:
    if not iso:
        return None
    try:
        dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            return dt.astimezone().replace(microsecond=0)
        return dt.replace(microsecond=0)
    except ValueError:
        return None


def _evidence_window_from_args(args: argparse.Namespace) -> tuple[str, str]:
    """Return (start_iso, end_iso) for the run, mirroring todo-list."""
    kwargs: dict[str, Any] = agent_time_kwargs_from_args(args)
    if "start_time" in kwargs:
        return str(kwargs["start_time"]), str(kwargs["end_time"])
    hours = float(kwargs["hours"])
    end = datetime.now().astimezone().replace(microsecond=0)
    start = end - timedelta(hours=hours)
    return start.isoformat(), end.isoformat()


def _has_custom_range(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "start_time", None) and getattr(args, "end_time", None))


def _window_calendar_days(start_iso: str, end_iso: str) -> list[date]:
    start_dt = _parse_iso_datetime(start_iso)
    end_dt = _parse_iso_datetime(end_iso)
    if not start_dt or not end_dt:
        return [datetime.now().date()]
    start_d, end_d = start_dt.date(), end_dt.date()
    if end_d < start_d:
        start_d, end_d = end_d, start_d
    days: list[date] = []
    cur = start_d
    while cur <= end_d:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def _range_spans_multiple_days(start_iso: str, end_iso: str) -> bool:
    days = _window_calendar_days(start_iso, end_iso)
    return len(days) > 1


def _ensure_journal_header(path: Path, day: date | None = None) -> None:
    if path.exists():
        return
    journal_date = (day or datetime.now().date()).isoformat()
    header = (
        f"---\ndate: {journal_date}\n"
        f"tags: [ai-prompts, deskmate]\n---\n\n"
        f"# AI Prompts — {journal_date}\n\n"
    )
    path.write_text(header, encoding="utf-8")


def _header_journal_date(header: str) -> date | None:
    m = _HEADER_DATE_RE.match(header.strip())
    if not m:
        return None
    try:
        return date.fromisoformat(m.group("date"))
    except ValueError:
        return None


def _prompt_key(body: str) -> str:
    """First 80 chars of the unquoted prompt body, normalized for dedup."""
    text = normalize_capture_text(_BLOCKQUOTE_LINE_RE.sub("", body))
    text = " ".join(text.split())
    return text[:80].lower()


def _existing_keys(journal_path: Path) -> set[str]:
    if not journal_path.exists():
        return set()
    content = journal_path.read_text(encoding="utf-8")
    keys: set[str] = set()
    for match in _BLOCK_RE.finditer(content):
        body = match.group("body").strip()
        # Skip the "**Category**: ... | **Length**: ..." metadata line.
        lines = [ln for ln in body.splitlines() if not ln.startswith("**Category**")]
        keys.add(_prompt_key("\n".join(lines)))
    return keys


def _split_blocks(report: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    for match in _BLOCK_RE.finditer(report):
        header = match.group("header").strip()
        body = match.group("body").strip()
        if not body:
            continue
        blocks.append((header, body))
    return blocks


def _upgrade_block_if_longer(journal_path: Path, key: str, header: str, body: str) -> bool:
    """Replace an existing journal block when we capture a longer version of the same prompt."""
    if not journal_path.exists():
        return False
    content = journal_path.read_text(encoding="utf-8")
    header_end = content.find("\n## ")
    if header_end < 0:
        return False
    header_part = content[: header_end + 1]
    body_region = content[header_end + 1 :]
    new_text = _block_body_text(body)
    if not new_text:
        return False

    kept: list[str] = []
    replaced = False
    for match in _BLOCK_RE.finditer(body_region):
        block_header = match.group("header").strip()
        block_body = match.group("body").strip()
        block_key = _prompt_key(block_body)
        old_text = _block_body_text(block_body)
        if block_key == key and len(new_text) > len(old_text):
            kept.append(f"## {header}\n{body}\n\n---\n\n")
            replaced = True
        else:
            kept.append(f"## {block_header}\n{block_body}\n\n---\n\n")
    if not replaced:
        return False
    journal_path.write_text(header_part.rstrip() + "\n\n" + "".join(kept), encoding="utf-8")
    return True


def _append_new_blocks(journal_path: Path, blocks: list[tuple[str, str]]) -> int:
    if not blocks:
        return 0
    _ensure_journal_header(journal_path)
    existing = _existing_keys(journal_path)
    appended = 0
    pending: list[tuple[str, str]] = []
    for header, body in blocks:
        metadata_stripped = "\n".join(
            ln for ln in body.splitlines() if not ln.startswith("**Category**")
        )
        key = _prompt_key(metadata_stripped)
        if not key:
            continue
        if key in existing:
            if _upgrade_block_if_longer(journal_path, key, header, body):
                appended += 1
            continue
        existing.add(key)
        pending.append((header, body))
    if not pending:
        return appended
    with journal_path.open("a", encoding="utf-8") as fh:
        for header, body in pending:
            fh.write(f"## {header}\n{body}\n\n---\n\n")
            appended += 1
    return appended


def _append_blocks_by_date(
    blocks: list[tuple[str, str]],
    *,
    fallback_day: date | None = None,
) -> int:
    """Append blocks to per-day journal files (header may include ``YYYY-MM-DD``)."""
    if not blocks:
        return 0
    default_day = fallback_day or datetime.now().date()
    by_day: dict[date, list[tuple[str, str]]] = {}
    for header, body in blocks:
        day = _header_journal_date(header) or default_day
        by_day.setdefault(day, []).append((header, body))
    total = 0
    for day, day_blocks in by_day.items():
        total += _append_new_blocks(_journal_for_date(day), day_blocks)
    return total


def _wrap_range_display(start_iso: str, end_iso: str, body: str) -> str:
    days = _window_calendar_days(start_iso, end_iso)
    start_d, end_d = days[0], days[-1]
    if start_d == end_d:
        title = f"# AI Prompts — {start_d.isoformat()}"
        fm_date = start_d.isoformat()
    else:
        title = f"# AI Prompts — {start_d.isoformat()} … {end_d.isoformat()}"
        fm_date = f"{start_d.isoformat()} … {end_d.isoformat()}"
    header = (
        f"---\ndate: {fm_date}\n"
        f"tags: [ai-prompts, deskmate]\n"
        f"window_start: {start_iso}\n"
        f"window_end: {end_iso}\n---\n\n"
        f"{title}\n\n"
    )
    return header + body.strip() + "\n"


def _merge_journals_in_range(start_iso: str, end_iso: str) -> str:
    """Collect blocks from daily journal files inside the selected window."""
    merged: dict[str, tuple[str, str]] = {}
    for day in _window_calendar_days(start_iso, end_iso):
        path = _journal_for_date(day)
        if not path.exists():
            continue
        for header, body in _split_blocks(path.read_text(encoding="utf-8")):
            metadata_stripped = "\n".join(
                ln for ln in body.splitlines() if not ln.startswith("**Category**")
            )
            key = _prompt_key(metadata_stripped)
            if key:
                merged[key] = (header, body)
    if not merged:
        return "_该时间范围内未捕获到 AI 提示词。_\n"
    parts: list[str] = []
    for header, body in merged.values():
        parts.append(f"## {header}\n{body}\n\n---\n\n")
    return "".join(parts)


def _build_range_display(start_iso: str, end_iso: str, report: str) -> str:
    """Show prompts for the user-selected window, not only today's journal file."""
    if report.strip() != "NO_NEW_PROMPTS":
        body = report.strip() + "\n"
    else:
        body = _merge_journals_in_range(start_iso, end_iso)
    return _wrap_range_display(start_iso, end_iso, body)


def _block_body_text(body: str) -> str:
    """Return the prompt body text with blockquote markers and metadata stripped."""
    lines = [ln for ln in body.splitlines() if not ln.startswith("**Category**")]
    return normalize_capture_text(_BLOCKQUOTE_LINE_RE.sub("", "\n".join(lines)))


def _prune_noise_from_journal(journal_path: Path) -> int:
    """Drop any existing blocks whose body matches the noise filter.

    Self-healing pass: removes accessibility/UI-chrome entries that may have
    been written before the agent gained its noise filter. Returns the number
    of pruned blocks. Safe to call every run; rewrites file only if changes.
    """
    if not journal_path.exists():
        return 0
    content = journal_path.read_text(encoding="utf-8")
    # Split off the YAML frontmatter + H1 header (everything before the first ``## `` block).
    header_end = content.find("\n## ")
    if header_end < 0:
        return 0
    header = content[: header_end + 1]
    body_region = content[header_end + 1 :]

    kept: list[str] = []
    pruned = 0
    for match in _BLOCK_RE.finditer(body_region):
        block_header = match.group("header").strip()
        block_body = match.group("body").strip()
        text = _block_body_text(block_body)
        if _is_prompt_noise(text):
            pruned += 1
            continue
        kept.append(f"## {block_header}\n{block_body}\n\n---\n\n")

    if pruned == 0:
        return 0
    journal_path.write_text(header.rstrip() + "\n\n" + "".join(kept), encoding="utf-8")
    return pruned


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture prompts sent to AI tools and append them to a daily journal."
    )
    add_agent_time_args(parser, default_hours=1)
    parser.add_argument("--model", type=str, default=None, help="Ollama model override.")
    parser.add_argument("--verbose", action="store_true", help="Print agent rounds to stderr.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete today's journal before running, so the LLM rewrites it from scratch.",
    )
    args = parser.parse_args()

    if args.model:
        import agent
        agent.OLLAMA_MODEL = args.model

    start_iso, end_iso = _evidence_window_from_args(args)
    custom_range = _has_custom_range(args)
    multi_day = _range_spans_multiple_days(start_iso, end_iso)

    journal_path = _today_journal()
    if args.rebuild:
        if custom_range or multi_day:
            for day in _window_calendar_days(start_iso, end_iso):
                p = _journal_for_date(day)
                if p.exists():
                    p.unlink()
        elif journal_path.exists():
            journal_path.unlink()

    for day in _window_calendar_days(start_iso, end_iso):
        pruned = _prune_noise_from_journal(_journal_for_date(day))
        if pruned and args.verbose:
            print(
                f"  [prompt-journal] pruned {pruned} stale noise block(s) from {day}",
                file=sys.stderr,
            )

    report = run_agent(PIPE_MD, verbose=args.verbose, **agent_time_kwargs_from_args(args))

    appended = 0
    if report.strip() != "NO_NEW_PROMPTS":
        blocks = _split_blocks(report)
        if custom_range or multi_day:
            window_days = _window_calendar_days(start_iso, end_iso)
            fallback = window_days[0] if len(window_days) == 1 else None
            appended = _append_blocks_by_date(blocks, fallback_day=fallback)
        else:
            appended = _append_new_blocks(journal_path, blocks)

    # Default (rolling hours, same-day): show today's cumulative journal.
    # Custom --start/--end or multi-day window: show this run's window report.
    if custom_range or multi_day:
        window_note = (
            f"\n\n---\n_时间窗：{start_iso} → {end_iso}；"
            f"本 run 追加 **{appended}** 条新提示。_\n"
        )
        display = _build_range_display(start_iso, end_iso, report).rstrip() + window_note
    else:
        if journal_path.exists():
            journal_text = journal_path.read_text(encoding="utf-8")
        else:
            journal_text = "_no prompts captured yet today._\n"
        window_note = f"\n\n---\n_Appended **{appended}** new prompt(s) this run._\n"
        display = journal_text.rstrip() + window_note

    out = output_dir(APP_NAME)
    write_markdown(out / "ai-prompt-journal.md", display)

    print(out / "ai-prompt-journal.md")
    if custom_range or multi_day:
        print(
            f"journal: {_journal_dir()} ({start_iso} → {end_iso}, "
            f"appended {appended} new prompt block(s))",
            file=sys.stderr,
        )
    else:
        print(
            f"journal: {journal_path} (appended {appended} new prompt block(s))",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
