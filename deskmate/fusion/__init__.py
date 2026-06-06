"""Additive context-fusion + capture-control module.

Two product features, one self-contained package (touches no existing capture
behavior beyond tiny fail-open gate checks):

* **Unified timeline** — :class:`ContextFusionBus` subscribes to the in-process
  event bus and projects every signal (screen / audio / input / clipboard /
  window) into the ``context_events`` table via :class:`ContextStore`.
* **Pause / forget / per-source switches** — :class:`CaptureControl` is the
  DB-backed runtime control surface; :func:`capture_allowed` is the fail-open
  cached gate the daemon's capture chokepoints consult.
"""

from __future__ import annotations

from .bus import ContextFusionBus
from .control import CaptureControl, capture_allowed, invalidate_cache, shutdown_gate
from .store import ContextStore

__all__ = [
    "ContextFusionBus",
    "ContextStore",
    "CaptureControl",
    "capture_allowed",
    "invalidate_cache",
    "shutdown_gate",
]
