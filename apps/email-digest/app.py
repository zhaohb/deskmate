"""Email Digest — LLM agent mode.

Reads pipe.md, lets the agent runner do the per-email-tool prefetch
(``_do_email_digest_prefetch``), then asks Ollama to summarize the inbox
activity captured locally by pc_assistant — apps used, top senders / threads,
drafts in progress, action items, and patterns.

Gmail and Outlook can be read via OAuth-backed mail APIs
(``/connections/gmail/...``, ``/connections/outlook/...``). For all other tools
pc_assistant reconstructs email activity from local screen / UI recordings.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from agent import run_agent  # noqa: E402
from common import add_agent_time_args, agent_time_kwargs_from_args, output_dir, write_markdown  # noqa: E402

APP_NAME = "email-digest"
PIPE_MD = Path(__file__).with_name("pipe.md")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize local email-client and webmail activity via LLM agent."
    )
    add_agent_time_args(parser, default_hours=24)
    parser.add_argument("--model", type=str, default=None, help="Ollama model override.")
    parser.add_argument("--verbose", action="store_true", help="Print agent rounds to stderr.")
    args = parser.parse_args()

    if args.model:
        import agent
        agent.OLLAMA_MODEL = args.model

    report = run_agent(PIPE_MD, verbose=args.verbose, **agent_time_kwargs_from_args(args))

    out = output_dir(APP_NAME)
    write_markdown(out / "email-digest.md", report)
    print(out / "email-digest.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
