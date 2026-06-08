"""Filesystem layout. One canonical place so every module agrees."""

from __future__ import annotations

import os
from pathlib import Path


def root() -> Path:
    """DeskMate data directory (``~/.deskmate`` by default)."""
    override = os.environ.get("DESKMATE_HOME")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".deskmate"


def db_path() -> Path:
    return root() / "data.db"


def frames_dir() -> Path:
    return root() / "frames"


def videos_dir() -> Path:
    return root() / "videos"


def audio_dir() -> Path:
    return root() / "audio"


def logs_dir() -> Path:
    return root() / "logs"


def config_path() -> Path:
    return root() / "config.toml"


def config_dir() -> Path:
    return root()


def pipes_dir() -> Path:
    return root() / "pipes"


def ov_cache_dir() -> Path:
    """OpenVINO compiled-model cache (CACHE_DIR for the openvino_genai backend)."""
    return root() / "ov_cache"


def paused_flag() -> Path:
    return root() / ".paused"


def restart_marker_path() -> Path:
    """Marker the API writes to request a process restart (see /restart).

    A supervising launcher can watch for this file to know a relaunch was asked
    for; it is advisory and safe to ignore when running unsupervised."""
    return root() / ".restart-requested"


def ensure_dirs() -> None:
    for p in (root(), frames_dir(), videos_dir(), audio_dir(), logs_dir(), pipes_dir()):
        p.mkdir(parents=True, exist_ok=True)
