"""Standup Update — LLM agent mode.

Reads pipe.md, sends to Ollama via the agent runner, lets the model
query DeskMate /activity-summary and /search and generate a
short standup update covering yesterday / today / blockers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from deskmate.apps.agent import run_agent
from deskmate.apps.common import output_dir, run_cli, write_markdown

APP_NAME = "standup-update"
PIPE_MD = Path(__file__).with_name("pipe.md")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a standup update via LLM agent.")
    parser.add_argument("--hours", type=float, default=24, help="Look back this many hours (default: 24).")
    parser.add_argument("--model", type=str, default=None, help="Ollama model override.")
    parser.add_argument("--verbose", action="store_true", help="Print agent rounds to stderr.")
    args = parser.parse_args()

    if args.model:
        from deskmate.apps import agent
        agent.OLLAMA_MODEL = args.model

    report = run_agent(PIPE_MD, hours=args.hours, verbose=args.verbose)

    out = output_dir(APP_NAME)
    write_markdown(out / "standup-update.md", report)
    print(out / "standup-update.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
