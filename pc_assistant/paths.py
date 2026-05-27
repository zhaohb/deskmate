"""Filesystem layout. One canonical place so every module agrees."""

from __future__ import annotations

import os
from pathlib import Path


def root() -> Path:
    override = os.environ.get("PC_ASSISTANT_HOME")
    return Path(override).expanduser() if override else Path.home() / ".pc_assistant"


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


def paused_flag() -> Path:
    return root() / ".paused"


def ensure_dirs() -> None:
    for p in (root(), frames_dir(), videos_dir(), audio_dir(), logs_dir(), pipes_dir()):
        p.mkdir(parents=True, exist_ok=True)
