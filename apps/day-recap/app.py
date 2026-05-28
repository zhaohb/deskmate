"""Day Recap — LLM agent mode.

Reads pipe.md, sends to Ollama via the agent runner, lets the model
autonomously query pc_assistant /search API and generate a day recap.
Runs the pipe agent against local pc_assistant API data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from agent import run_agent  # noqa: E402
from common import add_agent_time_args, agent_time_kwargs_from_args, output_dir, write_markdown  # noqa: E402

APP_NAME = "day-recap"
PIPE_MD = Path(__file__).with_name("pipe.md")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate today's pc_assistant recap via LLM agent.")
    add_agent_time_args(parser, default_hours=16)
    parser.add_argument("--model", type=str, default=None, help="Ollama model override.")
    parser.add_argument("--verbose", action="store_true", help="Print agent rounds to stderr.")
    args = parser.parse_args()

    if args.model:
        import agent
        agent.OLLAMA_MODEL = args.model

    report = run_agent(PIPE_MD, verbose=args.verbose, **agent_time_kwargs_from_args(args))

    out = output_dir(APP_NAME)
    write_markdown(out / "day-recap.md", report)
    print(out / "day-recap.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
