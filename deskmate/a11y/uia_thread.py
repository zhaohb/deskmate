"""COM / UI Automation initialization for worker threads on Windows.

``uiautomation`` requires ``UIAutomationInitializerInThread`` in any thread that
calls UIA APIs. Deskmate runs capture and UI-event handlers on background
threads, so public entry points wrap work in :func:`uia_com_session`.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import TypeVar

T = TypeVar("T")


@contextmanager
def uia_com_session() -> Iterator[None]:
    """Initialize COM/UIA for the current thread (no-op off Windows)."""
    if os.name != "nt":
        yield
        return
    try:
        import uiautomation as auto  # noqa: PLC0415
    except ImportError:
        yield
        return
    with auto.UIAutomationInitializerInThread():
        yield


def run_in_uia_thread(fn: Callable[[], T], *, default: T) -> T:
    """Run ``fn`` inside :func:`uia_com_session`, returning ``default`` on failure."""
    try:
        with uia_com_session():
            return fn()
    except Exception:
        return default
