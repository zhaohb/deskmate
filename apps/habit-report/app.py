"""Habit Report — LLM agent mode.

Summarizes the user's repeatable behavioral habits — daily rhythm, focus vs.
context-switching, and tool routines — from the mined habit profiles (作息规律)
plus activity data. Reuses the shared agent runner; the habit-report branch in
agent.py does the prefetch.

Like the profile, habits are a multi-day signal, so it defaults to 7 days; an
explicit --hours / --start/--end (or a UI range) overrides.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from agent import run_agent  # noqa: E402
from common import add_agent_time_args, agent_time_kwargs_from_args, output_dir, run_cli, write_markdown  # noqa: E402

APP_NAME = "habit-report"
PIPE_MD = Path(__file__).with_name("pipe.md")

DEFAULT_HOURS = 24 * 7


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize behavioral habits via LLM agent.")
    add_agent_time_args(parser, default_hours=DEFAULT_HOURS)
    parser.add_argument("--model", type=str, default=None, help="Ollama model override.")
    parser.add_argument("--verbose", action="store_true", help="Print agent rounds to stderr.")
    args = parser.parse_args()

    if args.model:
        import agent
        agent.OLLAMA_MODEL = args.model

    kwargs = agent_time_kwargs_from_args(args)
    report = run_agent(PIPE_MD, verbose=args.verbose, **kwargs)

    out = output_dir(APP_NAME)
    write_markdown(out / "habit-report.md", report)
    print(out / "habit-report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
