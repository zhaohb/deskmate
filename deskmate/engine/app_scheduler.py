"""Scheduler for ``apps/<name>/`` CLI apps.

Reads the ``schedule:`` field in each app's ``pipe.md`` frontmatter and, if it
matches a recognized natural-language interval (``every Nh``, ``every Nm``,
``every Ns``), fires the app's ``app.py`` as a subprocess at that cadence.

The ``apps/`` directory is shipped with the source tree (sibling of the
``deskmate/`` package). This scheduler is independent from
``deskmate.pipes.PipeScheduler`` which runs python/js pipes from
``%USERPROFILE%\\.deskmate\\pipes\\``.
"""

from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from ..logger import get

logger = get("engine.app_scheduler")

_APPS_SRC = Path(__file__).resolve().parents[2] / "apps"
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SCHEDULE_RE = re.compile(r"^\s*every\s+(\d+)\s*([hms])\s*$", re.IGNORECASE)


def _parse_schedule(value: str | None) -> int | None:
    """Convert ``every 1h`` / ``every 30m`` / ``every 90s`` to seconds."""
    if not value:
        return None
    m = _SCHEDULE_RE.match(value)
    if not m:
        return None
    n = int(m.group(1))
    unit = m.group(2).lower()
    if n <= 0:
        return None
    if unit == "h":
        return n * 3600
    if unit == "m":
        return n * 60
    return n


def _read_frontmatter(pipe_md: Path) -> dict[str, str]:
    try:
        text = pipe_md.read_text(encoding="utf-8")
    except OSError:
        return {}
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}
    fm: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        fm[k.strip()] = v.strip().strip('"').strip("'")
    return fm


def _discover_scheduled_apps() -> list[tuple[str, int]]:
    """Return ``[(app_name, interval_seconds), ...]`` for apps with a parseable
    ``schedule:`` and an ``app.py``. Only apps with ``enabled: true`` (or no
    explicit ``enabled`` key) are scheduled.
    """
    out: list[tuple[str, int]] = []
    if not _APPS_SRC.is_dir():
        return out
    for pipe_md in sorted(_APPS_SRC.glob("*/pipe.md")):
        fm = _read_frontmatter(pipe_md)
        if fm.get("enabled", "true").lower() == "false":
            continue
        interval = _parse_schedule(fm.get("schedule"))
        if interval is None:
            continue
        if not (pipe_md.parent / "app.py").is_file():
            continue
        out.append((pipe_md.parent.name, interval))
    return out


class AppScheduler:
    """Background thread that fires ``apps/<name>/app.py`` on a fixed cadence."""

    def __init__(self, *, tick_seconds: float = 5.0, run_timeout_seconds: int = 900) -> None:
        self.tick_seconds = tick_seconds
        self.run_timeout_seconds = run_timeout_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._next_fire: dict[str, float] = {}
        self._apps: list[tuple[str, int]] = []

    def _init_schedule(self) -> None:
        now = time.time()
        self._apps = _discover_scheduled_apps()
        # Stagger first fire by interval so we don't slam Ollama at startup.
        for name, interval in self._apps:
            self._next_fire[name] = now + interval

    def start(self) -> None:
        if self._thread:
            return
        self._init_schedule()
        if not self._apps:
            logger.info("app scheduler: no scheduled apps found, not starting")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="AppScheduler", daemon=True)
        self._thread.start()
        logger.info(
            "app scheduler started: %s",
            ", ".join(f"{n}=every{i}s" for n, i in self._apps),
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            for name, interval in self._apps:
                if now >= self._next_fire.get(name, 0):
                    self._fire(name)
                    self._next_fire[name] = now + interval
            self._stop.wait(self.tick_seconds)

    def _fire(self, app_name: str) -> None:
        app_py = _APPS_SRC / app_name / "app.py"
        if not app_py.is_file():
            return
        try:
            logger.info("app scheduler firing %s", app_name)
            proc = subprocess.run(  # noqa: S603
                [sys.executable, str(app_py)],
                capture_output=True,
                text=True,
                timeout=self.run_timeout_seconds,
                check=False,
            )
            if proc.returncode == 0:
                logger.info("app scheduler %s ok", app_name)
            else:
                logger.warning(
                    "app scheduler %s exit=%d stderr=%s",
                    app_name, proc.returncode, (proc.stderr or "")[-500:],
                )
        except subprocess.TimeoutExpired:
            logger.warning("app scheduler %s timed out after %ds", app_name, self.run_timeout_seconds)
        except Exception as exc:  # noqa: BLE001
            logger.warning("app scheduler %s failed: %s", app_name, exc)
