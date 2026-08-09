"""User Learning — detect learning phases, slice evidence, summarize + next plan.

Unlike day-recap (everything that happened) or user-profile (who you are), this
app asks: was the user studying? If yes, keep only courseware / material-query /
code-practice / problem evidence, then ask the LLM for a study summary and a
concrete next-step learning plan.

Prefetch + session detection live in agent.py / learning_slice.py; this file is
the CLI entry point discovered by My Apps.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from deskmate.apps.agent import run_agent
from deskmate.apps.common import add_agent_time_args, agent_time_kwargs_from_args, output_dir, run_cli, write_markdown

APP_NAME = "user-learning"
PIPE_MD = Path(__file__).with_name("pipe.md")

# A study recap is usually same-day; default 8h covers a typical study block.
DEFAULT_HOURS = 8


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect learning sessions and generate a study summary + next plan.",
    )
    add_agent_time_args(parser, default_hours=DEFAULT_HOURS)
    parser.add_argument("--model", type=str, default=None, help="Ollama model override.")
    parser.add_argument("--verbose", action="store_true", help="Print agent rounds to stderr.")
    args = parser.parse_args()

    if args.model:
        from deskmate.apps import agent
        agent.OLLAMA_MODEL = args.model

    kwargs = agent_time_kwargs_from_args(args)
    report = run_agent(PIPE_MD, verbose=args.verbose, **kwargs)

    out = output_dir(APP_NAME)
    write_markdown(out / "user-learning.md", report)
    print(out / "user-learning.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
