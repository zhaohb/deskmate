"""Pipes — markdown files with YAML frontmatter that describe scheduled
tasks.

This is a lightweight pipes engine. We implement:

- discovery of `.md` pipes under a configurable folder,
- frontmatter parsing (`---\nkey: value\n---\n...`),
- a scheduler that executes `runtime: python | js | none` pipes on the cron /
  interval declared by each pipe,
- a permissions check based on the same `PipePermissions` shape (read_db /
  trigger_capture / etc.).

Pipes can call the local API or DB according to the context passed in
environment variables.

The on-disk format is intentionally simple so users can write pipes by hand.
"""

from .loader import Pipe, PipeFrontmatter, load_pipes
from .runtime import PipeRuntime
from .scheduler import PipeScheduler

__all__ = ["Pipe", "PipeFrontmatter", "PipeRuntime", "PipeScheduler", "load_pipes"]
