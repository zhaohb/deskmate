"""User-configurable schedules for ``apps/<name>/app.py``.

Overrides are stored in ``~/.deskmate/apps/schedules.json``. When a user has not
configured an app, the ``schedule:`` field in the app's ``pipe.md`` is used as
the default (usually ``manual``).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from .. import paths
from ..logger import get

logger = get("engine.app_schedules")

_SCHEDULE_RE = re.compile(r"^\s*every\s+(\d+)\s*([hms])\s*$", re.IGNORECASE)
_DAILY_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")
_APPS_SRC = Path(__file__).resolve().parents[2] / "apps"

ScheduleMode = Literal["manual", "interval", "daily"]
ScheduleSource = Literal["user", "default", "manual"]


@dataclass(frozen=True)
class EffectiveSchedule:
    """Resolved schedule for one app (display + runtime)."""

    display: str
    source: ScheduleSource
    enabled: bool
    mode: ScheduleMode
    interval_seconds: int | None = None
    daily_hour: int | None = None
    daily_minute: int | None = None
    hours: str | None = None


def schedules_path() -> Path:
    return paths.root() / "apps" / "schedules.json"


def _ensure_parent() -> None:
    schedules_path().parent.mkdir(parents=True, exist_ok=True)


def load_all() -> dict[str, dict[str, Any]]:
    path = schedules_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not read %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, value in data.items():
        if isinstance(key, str) and isinstance(value, dict):
            out[key] = value
    return out


def save_entry(app_name: str, entry: dict[str, Any] | None) -> dict[str, Any]:
    """Persist one app's schedule. ``entry=None`` removes the override."""
    all_entries = load_all()
    if entry is None:
        all_entries.pop(app_name, None)
    else:
        all_entries[app_name] = entry
    _ensure_parent()
    schedules_path().write_text(
        json.dumps(all_entries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return all_entries.get(app_name, {})


def parse_interval_seconds(value: str | None) -> int | None:
    """Convert ``every 1h`` / ``every 30m`` / ``every 90s`` to seconds."""
    if not value:
        return None
    m = _SCHEDULE_RE.match(value.strip())
    if not m:
        return None
    n = int(m.group(1))
    if n <= 0:
        return None
    unit = m.group(2).lower()
    if unit == "h":
        if n > 168:
            return None
        return n * 3600
    if unit == "m":
        if n > 10_080:
            return None
        return n * 60
    if n > 86_400:
        return None
    return n


def parse_daily_time(value: str | None) -> tuple[int, int] | None:
    if not value:
        return None
    m = _DAILY_TIME_RE.match(value.strip())
    if not m:
        return None
    hour, minute = int(m.group(1)), int(m.group(2))
    if hour > 23 or minute > 59:
        return None
    return hour, minute


def format_interval(seconds: int) -> str:
    if seconds % 3600 == 0:
        return f"every {seconds // 3600}h"
    if seconds % 60 == 0:
        return f"every {seconds // 60}m"
    return f"every {seconds}s"


def schedule_display(
    *,
    enabled: bool,
    mode: ScheduleMode,
    interval: str | None = None,
    time: str | None = None,
) -> str:
    if not enabled or mode == "manual":
        return "manual"
    if mode == "interval" and interval:
        return interval.strip()
    if mode == "daily" and time:
        return f"daily {time.strip()}"
    return "manual"


def _user_entry_to_effective(entry: dict[str, Any]) -> EffectiveSchedule | None:
    if not entry.get("enabled"):
        return None
    mode = str(entry.get("mode") or "manual").strip().lower()
    hours = entry.get("hours")
    hours_str = str(hours).strip() if hours is not None and str(hours).strip() else None

    if mode == "interval":
        interval = str(entry.get("interval") or "").strip()
        seconds = parse_interval_seconds(interval)
        if seconds is None:
            return None
        return EffectiveSchedule(
            display=interval,
            source="user",
            enabled=True,
            mode="interval",
            interval_seconds=seconds,
            hours=hours_str,
        )

    if mode == "daily":
        daily = str(entry.get("time") or "").strip()
        parsed = parse_daily_time(daily)
        if parsed is None:
            return None
        hour, minute = parsed
        return EffectiveSchedule(
            display=f"daily {daily}",
            source="user",
            enabled=True,
            mode="daily",
            daily_hour=hour,
            daily_minute=minute,
            hours=hours_str,
        )
    return None


def _pipe_default_to_effective(pipe_schedule: str | None) -> EffectiveSchedule | None:
    raw = (pipe_schedule or "manual").strip()
    if not raw or raw.lower() == "manual":
        return None
    seconds = parse_interval_seconds(raw)
    if seconds is None:
        return None
    return EffectiveSchedule(
        display=raw,
        source="default",
        enabled=True,
        mode="interval",
        interval_seconds=seconds,
        hours=None,
    )


def effective_schedule(app_name: str, pipe_schedule: str | None = None) -> EffectiveSchedule:
    """Resolve the schedule for ``app_name`` (user override beats pipe default)."""
    user = load_all().get(app_name)
    if user is not None:
        if not user.get("enabled"):
            return EffectiveSchedule(
                display="manual",
                source="user",
                enabled=False,
                mode="manual",
            )
        resolved = _user_entry_to_effective(user)
        if resolved:
            return resolved

    fallback = _pipe_default_to_effective(pipe_schedule)
    if fallback:
        return fallback

    return EffectiveSchedule(
        display="manual",
        source="manual",
        enabled=False,
        mode="manual",
    )


def validate_schedule_payload(body: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a schedule update payload."""
    enabled = bool(body.get("enabled"))
    if not enabled:
        return {"enabled": False, "mode": "manual"}

    mode = str(body.get("mode") or "").strip().lower()
    if mode not in ("interval", "daily"):
        raise ValueError("mode must be 'interval' or 'daily' when enabled")

    hours_raw = body.get("hours")
    hours: str | None = None
    if hours_raw is not None and str(hours_raw).strip() != "":
        try:
            hval = float(str(hours_raw).strip())
        except ValueError as exc:
            raise ValueError("hours must be a number") from exc
        if hval <= 0 or hval > 24 * 365:
            raise ValueError("hours must be between 0 and 8760")
        hours = str(hours_raw).strip()

    if mode == "interval":
        interval = str(body.get("interval") or "").strip()
        if parse_interval_seconds(interval) is None:
            raise ValueError("interval must look like 'every 30m' or 'every 1h'")
        out: dict[str, Any] = {"enabled": True, "mode": "interval", "interval": interval}
        if hours is not None:
            out["hours"] = hours
        return out

    daily = str(body.get("time") or body.get("daily_time") or "").strip()
    if parse_daily_time(daily) is None:
        raise ValueError("time must be HH:MM (24-hour clock)")
    out = {"enabled": True, "mode": "daily", "time": daily}
    if hours is not None:
        out["hours"] = hours
    return out


def entry_for_api(app_name: str, pipe_schedule: str | None = None) -> dict[str, Any]:
    """Serialize schedule state for the UI."""
    eff = effective_schedule(app_name, pipe_schedule)
    user = load_all().get(app_name) or {}
    return {
        "enabled": eff.enabled,
        "mode": eff.mode,
        "interval": user.get("interval") or (
            eff.display if eff.mode == "interval" and eff.source == "default" else None
        ),
        "time": user.get("time"),
        "hours": user.get("hours"),
        "display": eff.display,
        "source": eff.source,
    }


def discover_scheduled_apps(apps_src: Path | None = None) -> list[tuple[str, EffectiveSchedule]]:
    """Return apps that should be fired by :class:`AppScheduler`."""
    root = apps_src or _APPS_SRC
    if not root.is_dir():
        return []

    out: list[tuple[str, EffectiveSchedule]] = []
    for pipe_md in sorted(root.glob("*/pipe.md")):
        if not (pipe_md.parent / "app.py").is_file():
            continue
        name = pipe_md.parent.name
        fm = _read_frontmatter(pipe_md)
        eff = effective_schedule(name, fm.get("schedule"))
        if eff.enabled and eff.mode in ("interval", "daily"):
            out.append((name, eff))
    return out


def daily_next_fire(hour: int, minute: int, *, now: float | None = None) -> float:
    """Epoch seconds for the next daily run (local timezone)."""
    ts = now if now is not None else datetime.now().astimezone().timestamp()
    dt = datetime.fromtimestamp(ts).astimezone()
    candidate = dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if candidate.timestamp() <= ts:
        candidate += timedelta(days=1)
    return candidate.timestamp()


def _read_frontmatter(pipe_md: Path) -> dict[str, str]:
    try:
        text = pipe_md.read_text(encoding="utf-8")
    except OSError:
        return {}
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not m:
        return {}
    fm: dict[str, str] = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        fm[k.strip()] = v.strip().strip('"').strip("'")
    return fm
