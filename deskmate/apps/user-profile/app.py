"""User Profile — LLM agent mode.

Synthesizes a multi-day portrait of WHO the user is and HOW they work, from
behavioral activity + mined habits + meetings + (best-effort) email. Reuses the
shared agent runner; the user-profile branch in agent.py does the prefetch.

Unlike day-recap (which looks at the last ~16h), a profile needs a wider window
to capture stable traits, so it defaults to 7 days. An explicit --hours / --start
/--end (or the UI passing a range) still overrides.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from deskmate.apps.agent import run_agent
from deskmate.apps.common import add_agent_time_args, agent_time_kwargs_from_args, output_dir, run_cli, write_markdown

APP_NAME = "user-profile"
PIPE_MD = Path(__file__).with_name("pipe.md")

# A profile is a multi-day portrait; default to a week unless overridden.
DEFAULT_HOURS = 24 * 7


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a user profile via LLM agent.")
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
    write_markdown(out / "user-profile.md", report)
    print(out / "user-profile.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
