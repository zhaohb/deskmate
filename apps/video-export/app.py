"""Video Export — LLM agent mode.

Reads pipe.md, sends to Ollama via the agent runner, lets the model
autonomously call POST /frames/export and report the exported file path.
Runs the pipe agent against local DeskMate API data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from agent import run_agent  # noqa: E402
from common import output_dir, write_markdown  # noqa: E402

APP_NAME = "video-export"
PIPE_MD = Path(__file__).with_name("pipe.md")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export screen activity video via LLM agent.")
    parser.add_argument(
        "--start",
        dest="start_time",
        type=str,
        default=None,
        help="Export range start (ISO 8601). Use with --end; overrides --minutes.",
    )
    parser.add_argument(
        "--end",
        dest="end_time",
        type=str,
        default=None,
        help="Export range end (ISO 8601). Use with --start; overrides --minutes.",
    )
    parser.add_argument("--minutes", type=float, default=5, help="Look back this many minutes (default: 5).")
    parser.add_argument("--model", type=str, default=None, help="Ollama model override.")
    parser.add_argument("--verbose", action="store_true", help="Print agent rounds to stderr.")
    args = parser.parse_args()

    if bool(args.start_time) ^ bool(args.end_time):
        parser.error("--start and --end must be used together")
    if args.start_time and args.end_time and args.start_time >= args.end_time:
        parser.error("--start must be before --end")

    if args.model:
        import agent
        agent.OLLAMA_MODEL = args.model

    if args.start_time and args.end_time:
        report = run_agent(
            PIPE_MD,
            start_time=args.start_time,
            end_time=args.end_time,
            verbose=args.verbose,
        )
    else:
        report = run_agent(PIPE_MD, hours=args.minutes / 60.0, verbose=args.verbose)

    out = output_dir(APP_NAME)
    write_markdown(out / "export-report.md", report)
    print(out / "export-report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
