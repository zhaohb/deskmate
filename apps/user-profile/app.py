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
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from agent import run_agent  # noqa: E402
from common import add_agent_time_args, agent_time_kwargs_from_args, output_dir, run_cli, write_markdown  # noqa: E402

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
        import agent
        agent.OLLAMA_MODEL = args.model

    kwargs = agent_time_kwargs_from_args(args)
    # The UI's generic run passes hours=16; widen it for a meaningful profile
    # unless the user gave an explicit start/end range.
    if "start_time" not in kwargs and float(kwargs.get("hours", 0) or 0) < DEFAULT_HOURS:
        kwargs["hours"] = DEFAULT_HOURS

    report = run_agent(PIPE_MD, verbose=args.verbose, **kwargs)

    out = output_dir(APP_NAME)
    write_markdown(out / "user-profile.md", report)
    print(out / "user-profile.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
