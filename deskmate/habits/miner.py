"""HabitMiner — aggregate frames into learned routines (data → patterns).

Pulls gap-attributed frame durations from the last ``lookback_days``, buckets
them by ``(day_type, half-hour slot, category)``, and writes the resulting
``habit_profiles``. ``frequency`` is the share of active days a behavior
occurred — the core "habit strength" signal used by the rules engine.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any

from ..logger import get
from ..workflow.classifier import classify_frame
from .store import HabitStore

logger = get("habits.miner")


def day_type_of(d: datetime) -> str:
    """'weekday' (Mon-Fri) or 'weekend' (Sat/Sun)."""
    return "weekend" if d.weekday() >= 5 else "weekday"


def slot_of(d: datetime) -> int:
    """Half-hour slot index 0..47 from local wall-clock time."""
    return d.hour * 2 + (1 if d.minute >= 30 else 0)


def _parse_ts(ts: str) -> datetime | None:
    """Parse a stored ISO timestamp, preserving wall-clock hour/minute."""
    if not ts:
        return None
    text = str(ts).strip().replace(" ", "T", 1) if " " in str(ts)[:11] else str(ts)
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        try:
            return datetime.strptime(str(ts)[:19], "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return None


class HabitMiner:
    def __init__(
        self,
        store: HabitStore,
        *,
        lookback_days: int = 30,
        min_frequency: float = 0.5,
        min_sample_days: int = 3,
    ) -> None:
        self._store = store
        self._lookback_days = lookback_days
        self._min_frequency = min_frequency
        self._min_sample_days = min_sample_days

    def mine(self) -> dict[str, Any]:
        """Recompute habit_profiles. Returns a summary dict."""
        start = (datetime.now().astimezone() - timedelta(days=self._lookback_days))
        start_iso = start.replace(microsecond=0).isoformat()
        rows = self._store.frame_durations_since(start_iso)

        # (day_type, slot, category) -> {day -> minutes}, plus per-app minutes.
        bucket_day_minutes: dict[tuple[str, int, str], dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        bucket_app_minutes: dict[tuple[str, int, str], dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        # day_type -> set of active days (any activity) for frequency denominator.
        active_days: dict[str, set[str]] = defaultdict(set)

        for row in rows:
            dt = _parse_ts(row.get("timestamp", ""))
            if dt is None:
                continue
            minutes = float(row.get("dur_sec") or 0.0) / 60.0
            if minutes <= 0:
                continue
            app = row.get("app_name") or ""
            window = row.get("window_name") or ""
            dtype = day_type_of(dt)
            slot = slot_of(dt)
            category = classify_frame(app, window)
            day = dt.date().isoformat()
            key = (dtype, slot, category)

            bucket_day_minutes[key][day] += minutes
            bucket_app_minutes[key][app] += minutes
            active_days[dtype].add(day)

        profiles: list[dict[str, Any]] = []
        for key, day_minutes in bucket_day_minutes.items():
            dtype, slot, category = key
            denom = len(active_days.get(dtype) or set()) or 1
            sample_days = len(day_minutes)
            frequency = sample_days / denom
            if frequency < self._min_frequency or sample_days < self._min_sample_days:
                continue
            avg_minutes = sum(day_minutes.values()) / sample_days
            top_app = ""
            apps = bucket_app_minutes.get(key) or {}
            if apps:
                top_app = max(apps.items(), key=lambda kv: kv[1])[0]
            profiles.append(
                {
                    "day_type": dtype,
                    "slot": slot,
                    "category": category,
                    "top_app": top_app,
                    "avg_minutes": round(avg_minutes, 1),
                    "frequency": round(frequency, 3),
                    "sample_days": sample_days,
                }
            )

        written = self._store.replace_profiles(profiles)
        summary = {
            "profiles": written,
            "frames_scanned": len(rows),
            "lookback_days": self._lookback_days,
        }
        logger.info("habit mining done: %s", summary)
        return summary
