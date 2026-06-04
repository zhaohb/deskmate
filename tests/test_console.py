"""Tests for safe console output helpers."""

from __future__ import annotations

import logging
import sys

import pytest

from deskmate.console import SafeStreamHandler, echo, echo_stderr, write_line


class _BrokenStream:
    def write(self, text: str) -> None:
        raise OSError(6, "Invalid handle")

    def flush(self) -> None:
        raise OSError(6, "Invalid handle")


def test_write_line_falls_back_from_broken_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    write_line("hello", _BrokenStream())
    assert capsys.readouterr().out.strip() == "hello"


def test_echo_stderr_does_not_raise_on_invalid_handle(capsys: pytest.CaptureFixture[str]) -> None:
    old_stderr = sys.stderr
    sys.stderr = _BrokenStream()  # type: ignore[assignment]
    try:
        echo_stderr("warning message")
    finally:
        sys.stderr = old_stderr
    assert capsys.readouterr().out.strip() == "warning message"


def test_echo_routes_to_stderr_when_err_true(capsys: pytest.CaptureFixture[str]) -> None:
    echo("stderr line", err=True)
    captured = capsys.readouterr()
    assert captured.err.strip() == "stderr line"
    assert captured.out == ""


def test_safe_stream_handler_falls_back_on_invalid_handle(capsys: pytest.CaptureFixture[str]) -> None:
    handler = SafeStreamHandler(_BrokenStream())
    handler.setFormatter(logging.Formatter("%(message)s"))
    record = logging.LogRecord("test", logging.WARNING, __file__, 1, "logged", (), None)
    handler.emit(record)
    assert capsys.readouterr().out.strip() == "logged"
