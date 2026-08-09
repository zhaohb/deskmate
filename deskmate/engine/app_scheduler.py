"""Scheduler for ``apps/<name>/`` CLI apps.

Schedules come from user overrides in ``~/.deskmate/apps/schedules.json``,
falling back to each app's ``pipe.md`` ``schedule:`` field. Supported modes:

- **interval** — ``every 30m``, ``every 1h``, …
- **daily** — run once per day at a local ``HH:MM`` time
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

from .app_schedules import EffectiveSchedule, daily_next_fire, discover_scheduled_apps
from .. import paths
from ..logger import get

logger = get("engine.app_scheduler")

# Apps whose CLI needs special args — scheduled runs only pass ``--hours`` where safe.
_HOURS_ARG_APPS = frozenset({
    "ai-prompt-journal",
    "ai-habits",
    "day-recap",
    "email-digest",
    "habit-report",
    "standup-update",
    "time-breakdown",
    "todo-list",
    "user-profile",
    "user-learning",
})


@dataclass
class _TrackedApp:
    name: str
    spec: EffectiveSchedule
    next_fire: float = 0.0


class AppScheduler:
    """Background thread that fires ``apps/<name>/app.py`` on a fixed cadence."""

    def __init__(
        self,
        *,
        tick_seconds: float = 5.0,
        run_timeout_seconds: int = 900,
        api_base: str = "http://127.0.0.1:3030",
    ) -> None:
        self.tick_seconds = tick_seconds
        self.run_timeout_seconds = run_timeout_seconds
        self.api_base = api_base.rstrip("/")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._apps: list[_TrackedApp] = []

    def _init_schedule(self) -> None:
        now = time.time()
        discovered = discover_scheduled_apps()
        tracked: list[_TrackedApp] = []
        for name, spec in discovered:
            if spec.mode == "interval" and spec.interval_seconds:
                # Stagger first fire so we don't slam Ollama at startup.
                tracked.append(_TrackedApp(name=name, spec=spec, next_fire=now + spec.interval_seconds))
            elif spec.mode == "daily" and spec.daily_hour is not None and spec.daily_minute is not None:
                tracked.append(_TrackedApp(
                    name=name,
                    spec=spec,
                    next_fire=daily_next_fire(spec.daily_hour, spec.daily_minute, now=now),
                ))
        self._apps = tracked

    def reload(self) -> None:
        """Re-read schedules (e.g. after a UI save) and start the thread if needed."""
        self._init_schedule()
        if self._apps and not self._thread:
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="AppScheduler", daemon=True)
            self._thread.start()
            logger.info(
                "app scheduler started: %s",
                ", ".join(f"{a.name}={a.spec.display}" for a in self._apps),
            )
        elif self._apps and self._thread:
            logger.info(
                "app scheduler reloaded: %s",
                ", ".join(f"{a.name}={a.spec.display}" for a in self._apps),
            )
        elif not self._apps:
            logger.info("app scheduler: no scheduled apps")

    def start(self) -> None:
        if self._thread:
            return
        self.reload()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            for tracked in list(self._apps):
                if now >= tracked.next_fire:
                    self._fire(tracked.name, tracked.spec)
                    if tracked.spec.mode == "interval" and tracked.spec.interval_seconds:
                        tracked.next_fire = now + tracked.spec.interval_seconds
                    elif (
                        tracked.spec.mode == "daily"
                        and tracked.spec.daily_hour is not None
                        and tracked.spec.daily_minute is not None
                    ):
                        tracked.next_fire = daily_next_fire(
                            tracked.spec.daily_hour,
                            tracked.spec.daily_minute,
                            now=now,
                        )
            self._stop.wait(self.tick_seconds)

    def _build_cmd_args(self, app_name: str, spec: EffectiveSchedule) -> list[str]:
        args = ["--verbose"]
        if app_name in _HOURS_ARG_APPS and spec.hours:
            args += ["--hours", str(spec.hours)]
        return args

    def _fire(self, app_name: str, spec: EffectiveSchedule) -> None:
        app_dir = paths.find_app_dir(app_name)
        app_py = app_dir / "app.py" if app_dir else None
        if app_py is None or not app_py.is_file():
            return
        cmd_args = self._build_cmd_args(app_name, spec)
        try:
            logger.info("app scheduler firing %s (%s)", app_name, spec.display)
            proc = subprocess.run(  # noqa: S603
                [sys.executable, str(app_py), *cmd_args],
                capture_output=True,
                text=True,
                timeout=self.run_timeout_seconds,
                check=False,
                env={
                    **dict(os.environ),
                    "DESKMATE_API": self.api_base,
                },
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
