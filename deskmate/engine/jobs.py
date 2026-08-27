"""Registry of long jobs the UI started and is still waiting on.

The browser used to be the only place that knew "a recap is generating". That
state lived in a JS Map, so a reload — or a second tab — saw nothing while the
work was still running, and the user assumed it had finished.

Those endpoints hold the HTTP connection open for the whole job (the work runs
in a worker thread or a child process). A reload drops that connection, but the
job keeps going and finishes normally; only the answer is lost. Recording the
job here lets any client ask what is still in flight.

Deliberately in-memory: the threads and child processes doing the work belong
to this process, so if it exits they die with it. Persisting the registry would
resurrect jobs that are no longer running.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_lock = threading.Lock()
_jobs: dict[str, dict[str, Any]] = {}


def start(key: str, *, label: str = "", meta: dict[str, Any] | None = None) -> None:
    """Record that `key` is running. A repeated key refreshes the entry."""
    if not key:
        return
    with _lock:
        existing = _jobs.get(key)
        started = existing["started_at"] if existing else time.time()
        _jobs[key] = {
            "key": key,
            "label": label,
            "meta": dict(meta or {}),
            "started_at": started,
        }


def finish(key: str) -> None:
    with _lock:
        _jobs.pop(key, None)


def running() -> list[dict[str, Any]]:
    """Jobs still in flight, oldest first."""
    now = time.time()
    with _lock:
        rows = list(_jobs.values())
    rows.sort(key=lambda r: r["started_at"])
    return [
        {
            "key": r["key"],
            "label": r["label"],
            "meta": r["meta"],
            "running_ms": int((now - r["started_at"]) * 1000),
        }
        for r in rows
    ]


def is_running(key: str) -> bool:
    with _lock:
        return key in _jobs


@contextmanager
def track(key: str, *, label: str = "", meta: dict[str, Any] | None = None) -> Iterator[None]:
    """Mark `key` running for the duration of the block.

    The cleanup also covers the request being cancelled, so a client that walks
    away cannot strand the entry and leave the UI showing a job forever.
    """
    start(key, label=label, meta=meta)
    try:
        yield
    finally:
        finish(key)


def clear() -> None:
    """Drop every entry — for tests."""
    with _lock:
        _jobs.clear()
