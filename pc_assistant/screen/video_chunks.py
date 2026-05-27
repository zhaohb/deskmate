"""Video chunk storage helpers."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

from .. import paths

_SAFE_DEVICE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def video_chunk_path(
    *,
    device_name: str,
    captured_at: datetime | None = None,
    extension: str = "mp4",
) -> Path:
    """Return the canonical path for a recorded/imported video chunk."""
    ts = captured_at or datetime.now(timezone.utc).astimezone()
    safe_device = _SAFE_DEVICE_RE.sub("_", (device_name or "screen").strip()).strip("_") or "screen"
    ext = extension.lstrip(".") or "mp4"
    day_dir = paths.videos_dir() / ts.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir / f"{safe_device}_{ts.strftime('%Y%m%dT%H%M%S%f')}.{ext}"
