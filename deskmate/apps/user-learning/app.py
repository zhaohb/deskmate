"""User Learning — detect learning phases, slice evidence, summarize + next plan.

Unlike day-recap (everything that happened) or user-profile (who you are), this
app asks: was the user studying? If yes, keep only courseware / material-query /
code-practice / problem evidence, then ask the LLM for a study summary and a
concrete next-step learning plan.

Prefetch also runs deterministic concept extraction, lecture structure
(definition/step/relation), and SM-2 review seeding into the local DB.

Prefetch + session detection live in agent.py / learning_slice.py /
learning_memory/; this file is the CLI entry point discovered by My Apps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deskmate.apps.agent import G_LEARNING_ENRICHMENT, run_agent
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

    # The complete lecture transcript, kept next to the report. The report is
    # written from as much of it as a local model can read at once; this file is
    # the whole session, so nothing said during it is lost to that budget.
    full = (G_LEARNING_ENRICHMENT or {}).get("full_transcript") or {}
    if full.get("text"):
        write_markdown(
            out / "transcript.md",
            f"# Full transcript\n\n"
            f"_{full.get('range_start')} → {full.get('range_end')}_\n\n"
            f"_{full.get('rows')} transcript rows; "
            f"{full.get('rows_in_prompt')} of them reached the report prompt._\n\n"
            f"{full['text']}",
        )

    if G_LEARNING_ENRICHMENT:
        side = {
            "extraction": G_LEARNING_ENRICHMENT.get("extraction"),
            "topics": G_LEARNING_ENRICHMENT.get("topics"),
            "due_reviews": G_LEARNING_ENRICHMENT.get("due_reviews"),
            "events": G_LEARNING_ENRICHMENT.get("events"),
            "edges": G_LEARNING_ENRICHMENT.get("edges"),
            "must_cover": G_LEARNING_ENRICHMENT.get("must_cover"),
            "session_ids": G_LEARNING_ENRICHMENT.get("session_ids"),
            "persisted": G_LEARNING_ENRICHMENT.get("persisted"),
        }
        (out / "learning-enrichment.json").write_text(
            json.dumps(side, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(out / "user-learning.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))
