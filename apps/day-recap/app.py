"""Day Recap — LLM agent mode.

Reads pipe.md, sends to Ollama via the agent runner, lets the model
autonomously query DeskMate /search API and generate a day recap.
Runs the pipe agent against local DeskMate API data.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))
from agent import run_agent  # noqa: E402
from common import add_agent_time_args, agent_time_kwargs_from_args, output_dir, write_markdown  # noqa: E402
from deskmate.engine.day_recap_context import range_spans_calendar_days  # noqa: E402

APP_NAME = "day-recap"
PIPE_MD = Path(__file__).with_name("pipe.md")


def _evidence_window_from_args(args: argparse.Namespace) -> tuple[str, str]:
    kwargs: dict[str, Any] = agent_time_kwargs_from_args(args)
    if "start_time" in kwargs:
        return str(kwargs["start_time"]), str(kwargs["end_time"])
    hours = float(kwargs["hours"])
    end = datetime.now().astimezone().replace(microsecond=0)
    start = end - timedelta(hours=hours)
    return start.isoformat(), end.isoformat()


def _has_custom_range(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "start_time", None) and getattr(args, "end_time", None))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate today's DeskMate recap via LLM agent.")
    add_agent_time_args(parser, default_hours=16)
    parser.add_argument("--model", type=str, default=None, help="Ollama model override.")
    parser.add_argument("--verbose", action="store_true", help="Print agent rounds to stderr.")
    args = parser.parse_args()

    if args.model:
        import agent
        agent.OLLAMA_MODEL = args.model

    start_iso, end_iso = _evidence_window_from_args(args)
    report = run_agent(PIPE_MD, verbose=args.verbose, **agent_time_kwargs_from_args(args))

    if _has_custom_range(args) or range_spans_calendar_days(start_iso, end_iso):
        report = (
            report.rstrip()
            + f"\n\n---\n_时间窗：{start_iso} → {end_iso}_\n"
        )

    out = output_dir(APP_NAME)
    write_markdown(out / "day-recap.md", report)
    print(out / "day-recap.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
