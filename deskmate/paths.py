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


# ─── apps (pipe apps / plugins) ──────────────────────────────────────────────
def builtin_apps_dir() -> Path:
    """Directory holding the apps that ship with DeskMate (``deskmate/apps``).

    Apps live INSIDE the package (``deskmate/apps``) so they ship in the wheel
    as package data; ``paths.py`` is ``deskmate/paths.py``, so they sit beside
    it. Resolved from the package — not a caller's CWD — so it's stable no
    matter where DeskMate is launched from, including a non-editable install.
    """
    return Path(__file__).resolve().parent / "apps"


def user_apps_dir() -> Path:
    """Directory where users drop their OWN apps (plugins).

    Lives under the data dir (``~/.deskmate/apps/plugins``) so it survives
    upgrades and sits outside the install tree. ``apps/<name>/output`` and
    ``apps/schedules.json`` already live under ``~/.deskmate/apps``; plugins get
    their own ``plugins/`` subdir so user app *sources* never collide with app
    *outputs*.
    """
    return root() / "apps" / "plugins"


def app_source_roots() -> list[Path]:
    """All directories scanned for apps, in precedence order.

    User plugins come FIRST so a user can shadow a built-in app by giving their
    folder the same name. Only existing directories are returned.
    """
    roots = [user_apps_dir(), builtin_apps_dir()]
    return [r for r in roots if r.is_dir()]


def find_app_dir(name: str) -> Path | None:
    """Resolve an app NAME to its source directory across all roots.

    Returns the first match in :func:`app_source_roots` precedence (user
    plugins shadow built-ins), or ``None`` if no such app exists. ``name`` is
    sanitized so it can't escape an app root via ``..`` or path separators.
    """
    safe = (name or "").strip()
    if not safe or safe in (".", "..") or "/" in safe or "\\" in safe:
        return None
    for r in app_source_roots():
        candidate = r / safe
        if (candidate / "pipe.md").is_file():
            return candidate
    return None


def discover_app_dirs() -> list[Path]:
    """Every valid app directory (has ``pipe.md``) across all roots.

    De-duplicated by app name with user-plugin precedence, so a shadowing user
    app appears once (its own dir), not twice.
    """
    seen: set[str] = set()
    out: list[Path] = []
    for r in app_source_roots():
        for pipe_md in sorted(r.glob("*/pipe.md")):
            name = pipe_md.parent.name
            if name in seen:
                continue
            seen.add(name)
            out.append(pipe_md.parent)
    return out


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
    for p in (root(), frames_dir(), videos_dir(), audio_dir(), logs_dir(), pipes_dir(), user_apps_dir()):
        p.mkdir(parents=True, exist_ok=True)
