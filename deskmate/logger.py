"""Tiny logging helper: one logger per module."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from . import paths
from .console import SafeStreamHandler

_CONFIGURED = False


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    paths.ensure_dirs()
    root = logging.getLogger("deskmate")
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

    stream = SafeStreamHandler()
    stream.setFormatter(fmt)
    root.addHandler(stream)

    fh = RotatingFileHandler(paths.logs_dir() / "deskmate.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    _CONFIGURED = True


def get(name: str) -> logging.Logger:
    _configure()
    return logging.getLogger(f"deskmate.{name}")
