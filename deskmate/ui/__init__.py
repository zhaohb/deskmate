"""Browser UI for DeskMate.

The UI is intentionally shipped as static files served by FastAPI. This keeps
the project installable without a Node build step while still exposing health,
search, timeline, frame preview, events and configuration.
"""

from __future__ import annotations

from pathlib import Path


def static_dir() -> Path:
    return Path(__file__).resolve().parent / "static"


def index_file() -> Path:
    return static_dir() / "index.html"


__all__ = ["index_file", "static_dir"]
