"""Habit rules — evaluate "learned routine + present moment → should we nudge".

Each rule is a pure function ``(rule, state, profiles) -> (hit, message, ctx)``
so it is trivially testable and easy to extend. ``DEFAULT_RULES`` seeds the
``habit_rules`` table on first run.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from ..logger import get
from ..workflow.classifier import classify_frame
from .miner import _parse_ts, day_type_of, slot_of
from .store import HabitStore

logger = get("habits.rules")


# Default rule set seeded into habit_rules (idempotent).
DEFAULT_RULES: list[dict[str, Any]] = [
    {
        "name": "distraction_peak",
        "rule_type": "deviation",
        "enabled": True,
        "priority": "M",
        "cooldown_min": 120,
        "quiet_hours": "22-8",
        "params": {
            # This slot is usually a focused category, but right now the user is
            # in `off_category` and has been for >= min_minutes.
            "focus_categories": ["coding", "writing"],
            "off_category": "browsing",
            "min_habit_frequency": 0.6,
            "min_minutes": 20,
        },
    },
    {
        "name": "overwork",
        "rule_type": "threshold",
        "enabled": True,
        "priority": "M",
        "cooldown_min": 90,
        "quiet_hours": "22-8",
        "params": {
            "categories": ["coding", "meeting"],
            "max_minutes": 120,
        },
    },
]


class CurrentState:
    """A lightweight snapshot of "what the user is doing right now"."""

    def __init__(
        self,
        *,
        category: str,
        app_name: str,
        window_name: str,
        continuous_minutes: float,
    ) -> None:
        self.category = category
        self.app_name = app_name
        self.window_name = window_name
        self.continuous_minutes = continuous_minutes

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "app_name": self.app_name,
            "window_name": self.window_name,
            "continuous_minutes": round(self.continuous_minutes, 1),
        }


def read_current_state(store: HabitStore, now: datetime, *, window_minutes: int = 5) -> CurrentState | None:
    """Determine the dominant current activity over the last ``window_minutes``."""
    start = (now - timedelta(minutes=window_minutes)).replace(microsecond=0).isoformat()
    end = now.replace(microsecond=0).isoformat()
    frames = store.recent_frames_between(start, end)
    if not frames:
        return None

    # Dominant app by frame count in the window.
    counts: dict[tuple[str, str], int] = {}
    for f in frames:
        key = (f.get("app_name") or "", f.get("window_name") or "")
        counts[key] = counts.get(key, 0) + 1
    (app_name, window_name), _ = max(counts.items(), key=lambda kv: kv[1])
    category = classify_frame(app_name, window_name)

    # How long has the user been continuously in this app? Prefer an app_switch
    # event; fall back to the earliest frame of this app in the window.
    continuous_minutes = float(window_minutes)
    switch_ts = store.last_app_switch_ts(app_name, end)
    anchor = _parse_ts(switch_ts) if switch_ts else None
    if anchor is not None:
        continuous_minutes = max(0.0, (now - anchor).total_seconds() / 60.0)

    return CurrentState(
        category=category,
        app_name=app_name,
        window_name=window_name,
        continuous_minutes=continuous_minutes,
    )


def _slot_habit_frequency(profiles: list[dict[str, Any]], categories: list[str]) -> float:
    """Max frequency among the given categories for the current slot."""
    best = 0.0
    for p in profiles:
        if p.get("category") in categories:
            best = max(best, float(p.get("frequency") or 0.0))
    return best


def evaluate_distraction(
    rule: dict[str, Any], state: CurrentState, profiles: list[dict[str, Any]]
) -> tuple[bool, str, dict[str, Any]]:
    params = rule.get("params", {})
    focus_cats = params.get("focus_categories", ["coding", "writing"])
    off_cat = params.get("off_category", "browsing")
    min_freq = float(params.get("min_habit_frequency", 0.6))
    min_minutes = float(params.get("min_minutes", 20))

    habit_freq = _slot_habit_frequency(profiles, focus_cats)
    hit = (
        habit_freq >= min_freq
        and state.category == off_cat
        and state.continuous_minutes >= min_minutes
    )
    ctx = {
        "habit_frequency": round(habit_freq, 3),
        "state": state.as_dict(),
        "focus_categories": focus_cats,
    }
    msg = (
        f"现在通常是你的专注时段（{int(round(habit_freq * 100))}% 的同类日子在做 "
        f"{'/'.join(focus_cats)}），但你已经在 {state.app_name} 上停留了 "
        f"{int(state.continuous_minutes)} 分钟。要回到专注状态吗？"
    )
    return hit, msg, ctx


def evaluate_overwork(
    rule: dict[str, Any], state: CurrentState, profiles: list[dict[str, Any]]
) -> tuple[bool, str, dict[str, Any]]:
    params = rule.get("params", {})
    cats = params.get("categories", ["coding", "meeting"])
    max_minutes = float(params.get("max_minutes", 120))

    hit = state.category in cats and state.continuous_minutes >= max_minutes
    ctx = {"state": state.as_dict(), "max_minutes": max_minutes}
    msg = (
        f"你已经连续 {int(state.continuous_minutes)} 分钟处于 {state.category} 状态，"
        "起来活动一下、喝口水？"
    )
    return hit, msg, ctx


_EVALUATORS = {
    "deviation": evaluate_distraction,
    "threshold": evaluate_overwork,
}


def evaluate_rule(
    rule: dict[str, Any], state: CurrentState, profiles: list[dict[str, Any]]
) -> tuple[bool, str, dict[str, Any]]:
    """Dispatch a rule to its evaluator. Unknown types never fire."""
    evaluator = _EVALUATORS.get(rule.get("rule_type", ""))
    if evaluator is None:
        return False, "", {}
    try:
        return evaluator(rule, state, profiles)
    except Exception as exc:  # noqa: BLE001 — a bad rule must never crash the watcher
        logger.warning("rule %s failed: %s", rule.get("name"), exc)
        return False, "", {}


def current_slot_key(now: datetime) -> tuple[str, int]:
    return day_type_of(now), slot_of(now)
