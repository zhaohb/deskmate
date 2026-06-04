"""Safe console output when stderr/stdout handles are invalid (Windows terminals).

Some Windows terminals (including Cursor's integrated terminal) expose stderr
handles that fail with ``OSError: [WinError 6]`` when Click/Colorama wrap them.
Use these helpers instead of ``typer.echo(..., err=True)`` or bare ``print(...,
file=sys.stderr)``.
"""

from __future__ import annotations

import logging
import sys
from typing import TextIO

_FALLBACK_LOGGER = logging.getLogger("deskmate.console")


def write_line(message: str, stream: TextIO | None = None) -> None:
    """Write one line, trying *stream* then stdout/stderr, then logging."""
    candidates: list[TextIO] = []
    if stream is not None:
        candidates.append(stream)
    for fallback in (sys.stdout, sys.stderr):
        if fallback not in candidates:
            candidates.append(fallback)

    for target in candidates:
        try:
            print(message, file=target, flush=True)
            return
        except OSError:
            continue

    _FALLBACK_LOGGER.warning("%s", message)


def echo_stderr(message: str) -> None:
    """Print to stderr without raising if the handle is invalid."""
    write_line(message, sys.stderr)


def echo_stdout(message: str) -> None:
    """Print to stdout without raising if the handle is invalid."""
    write_line(message, sys.stdout)


def echo(message: str, *, err: bool = False) -> None:
    """Safe replacement for ``typer.echo`` (avoids Colorama WinConsole crashes)."""
    if err:
        echo_stderr(message)
    else:
        echo_stdout(message)


class SafeStreamHandler(logging.StreamHandler):
    """``StreamHandler`` that falls back when the primary stream handle is invalid."""

    def __init__(self, stream: TextIO | None = None) -> None:
        super().__init__(stream or sys.stderr)
        primary = stream or sys.stderr
        self._fallback = sys.stdout if primary is sys.stderr else sys.stderr

    def emit(self, record: logging.LogRecord) -> None:
        try:
            super().emit(record)
        except OSError:
            old = self.stream
            try:
                self.stream = self._fallback
                super().emit(record)
            except OSError:
                self.handleError(record)
            finally:
                self.stream = old
