"""AI Prompt Journal — LLM agent mode.

Reads pipe.md, sends to Ollama via the agent runner, lets the model
extract prompts the user typed into AI tools and emit one ``## HH:MM — Tool — Topic``
block per prompt. This runner then appends only the genuinely new blocks to
today's journal file at::

    %USERPROFILE%\\.pc_assistant\\apps\\ai-prompt-journal\\journal\\YYYY-MM-DD.md

Deduplication keys on the first 80 characters of each prompt's body, so the
same prompt captured by repeated hourly runs is not re-appended.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from agent import _is_prompt_noise, run_agent  # noqa: E402
from common import output_dir, pc_assistant_home, write_markdown  # noqa: E402

APP_NAME = "ai-prompt-journal"
PIPE_MD = Path(__file__).with_name("pipe.md")

# Match a single prompt block produced by pipe.md. The block ends at a line
# containing only "---" (or at end-of-text).
_BLOCK_RE = re.compile(
    r"^##\s+(?P<header>.+?)\n(?P<body>.*?)(?:\n---\s*$|\Z)",
    re.MULTILINE | re.DOTALL,
)
_BLOCKQUOTE_LINE_RE = re.compile(r"^>\s?", re.MULTILINE)


def _journal_dir() -> Path:
    out = pc_assistant_home() / "apps" / APP_NAME / "journal"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _today_journal() -> Path:
    return _journal_dir() / f"{datetime.now().date().isoformat()}.md"


def _ensure_journal_header(path: Path) -> None:
    if path.exists():
        return
    header = (
        f"---\ndate: {datetime.now().date().isoformat()}\n"
        f"tags: [ai-prompts, pc_assistant]\n---\n\n"
        f"# AI Prompts — {datetime.now().date().isoformat()}\n\n"
    )
    path.write_text(header, encoding="utf-8")


def _prompt_key(body: str) -> str:
    """First 80 chars of the unquoted prompt body, normalized for dedup."""
    text = _BLOCKQUOTE_LINE_RE.sub("", body).strip()
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


def _append_new_blocks(journal_path: Path, blocks: list[tuple[str, str]]) -> int:
    if not blocks:
        return 0
    _ensure_journal_header(journal_path)
    existing = _existing_keys(journal_path)
    appended = 0
    with journal_path.open("a", encoding="utf-8") as fh:
        for header, body in blocks:
            metadata_stripped = "\n".join(
                ln for ln in body.splitlines() if not ln.startswith("**Category**")
            )
            key = _prompt_key(metadata_stripped)
            if not key or key in existing:
                continue
            existing.add(key)
            fh.write(f"## {header}\n{body}\n\n---\n\n")
            appended += 1
    return appended


def _block_body_text(body: str) -> str:
    """Return the prompt body text with blockquote markers and metadata stripped."""
    lines = [ln for ln in body.splitlines() if not ln.startswith("**Category**")]
    text = _BLOCKQUOTE_LINE_RE.sub("", "\n".join(lines)).strip()
    return text


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
    parser.add_argument("--hours", type=float, default=1, help="Look back this many hours (default: 1).")
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

    journal_path = _today_journal()
    if args.rebuild and journal_path.exists():
        journal_path.unlink()
    pruned = _prune_noise_from_journal(journal_path)
    if pruned and args.verbose:
        print(f"  [prompt-journal] pruned {pruned} stale noise block(s) from journal", file=sys.stderr)

    report = run_agent(PIPE_MD, hours=args.hours, verbose=args.verbose)

    appended = 0
    if report.strip() != "NO_NEW_PROMPTS":
        blocks = _split_blocks(report)
        appended = _append_new_blocks(journal_path, blocks)

    # The report the UI displays is the *cumulative* journal for today plus a
    # short status footer, so the user always sees their captured prompts —
    # even when the current hourly run had nothing new to add.
    if journal_path.exists():
        journal_text = journal_path.read_text(encoding="utf-8")
    else:
        journal_text = "_no prompts captured yet today._\n"
    footer = (
        f"\n\n---\n_Run summary: pruned **{pruned}** stale noise block(s) and "
        f"appended **{appended}** new prompt block(s) this run; full journal at "
        f"`{journal_path}`._\n"
    )
    display = journal_text.rstrip() + footer

    out = output_dir(APP_NAME)
    write_markdown(out / "ai-prompt-journal.md", display)

    print(out / "ai-prompt-journal.md")
    print(
        f"journal: {journal_path} (appended {appended} new prompt block(s))",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
