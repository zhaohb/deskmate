"""AI Habits — LLM agent mode.

Reads pipe.md, sends to Ollama via the agent runner, lets the model
autonomously query DeskMate /search API per AI tool and generate
an AI usage report via the local pipe agent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from deskmate.apps.agent import run_agent
from deskmate.apps.common import add_agent_time_args, agent_time_kwargs_from_args, output_dir, run_cli, write_markdown

APP_NAME = "ai-habits"
PIPE_MD = Path(__file__).with_name("pipe.md")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze AI tool usage via LLM agent.")
    add_agent_time_args(parser, default_hours=24)
    parser.add_argument("--model", type=str, default=None, help="Ollama model override.")
    parser.add_argument("--verbose", action="store_true", help="Print agent rounds to stderr.")
    args = parser.parse_args()

    if args.model:
        from deskmate.apps import agent
        agent.OLLAMA_MODEL = args.model

    report = run_agent(PIPE_MD, verbose=args.verbose, **agent_time_kwargs_from_args(args))

    out = output_dir(APP_NAME)
    write_markdown(out / "ai-habits.md", report)
    print(out / "ai-habits.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
