"""Todo List Assistant — LLM agent mode.

Builds a single, unified todolist from TWO sources of evidence over the
supplied time range:

* Email — reuses the email-digest per-email-tool prefetch
  (``_do_email_digest_prefetch``): Gmail / Outlook OAuth messages plus local
  screen / UI hits for every other mail tool.
* Meetings — video calls detected in the range (Teams / Zoom / Meet / …) and
  their transcripts, so action items spoken in a call become todos too.

Both evidence blocks are handed to a single-shot extraction that emits a
markdown checklist (`- [ ] ...`) tagging each item's source, written to
``~/.pc_assistant/apps/todo-list/output/<timestamp>/todo-list.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from agent import run_agent  # noqa: E402
from common import add_agent_time_args, agent_time_kwargs_from_args, output_dir, write_markdown  # noqa: E402

APP_NAME = "todo-list"
PIPE_MD = Path(__file__).with_name("pipe.md")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract a unified todolist from email + meeting activity via LLM agent."
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
    write_markdown(out / "todo-list.md", report)
    print(out / "todo-list.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
