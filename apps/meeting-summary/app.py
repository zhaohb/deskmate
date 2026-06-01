"""Meeting Summary — LLM agent mode.

Reads pipe.md, finds the meeting that just ended, summarizes its transcript via
Ollama, and patches the summary back onto the meeting record (note + title).
Runs the pipe agent against local pc_assistant API data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from agent import run_agent  # noqa: E402
from common import output_dir, write_markdown  # noqa: E402

APP_NAME = "meeting-summary"
PIPE_MD = Path(__file__).with_name("pipe.md")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize the meeting that just ended via LLM agent and patch it back."
    )
    parser.add_argument("--model", type=str, default=None, help="Ollama model override.")
    parser.add_argument("--verbose", action="store_true", help="Print agent steps to stderr.")
    parser.add_argument(
        "--hours",
        type=float,
        default=None,
        help="Ignored (My Apps / run API may pass this). Meeting scope comes from the meeting record.",
    )
    parser.add_argument(
        "--meeting-id",
        dest="meeting_id",
        type=int,
        default=None,
        help="Summarize this specific meeting (default: the most recent meeting).",
    )
    args = parser.parse_args()

    if args.model:
        import agent
        agent.OLLAMA_MODEL = args.model

    # The meeting-summary agent ignores the time window (it scopes to the
    # meeting record itself), but run_agent requires a range — pass 24h.
    report = run_agent(PIPE_MD, hours=24, meeting_id=args.meeting_id, verbose=args.verbose)

    out = output_dir(APP_NAME)
    write_markdown(out / "meeting-summary.md", report)
    print(out / "meeting-summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
