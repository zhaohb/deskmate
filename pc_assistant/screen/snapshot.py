"""Writes JPEG frames to disk in a date-sharded layout."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from PIL import Image

from .. import paths
from .capture import encode_jpeg


class SnapshotWriter:
    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root else paths.frames_dir()
        self.root.mkdir(parents=True, exist_ok=True)

    def write(self, image: Image.Image, *, monitor_id: int, captured_at: datetime, quality: int = 80) -> Path:
        day_dir = self.root / captured_at.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        stamp = captured_at.strftime("%Y%m%dT%H%M%S%f")
        path = day_dir / f"{stamp}_m{monitor_id}.jpg"
        path.write_bytes(encode_jpeg(image, quality=quality))
        return path
