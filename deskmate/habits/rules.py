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
    # ── Fixed strategies (no learned history needed; fire from day one) ────────
    {
        # Continuous screen use in ANY category for too long → stand up / rest
        # the eyes. Empty `categories` means "match any category".
        "name": "break_reminder",
        "rule_type": "threshold",
        "enabled": True,
        "priority": "M",
        "cooldown_min": 60,
        "quiet_hours": "22-8",
        "params": {
            "categories": [],
            "max_minutes": 50,
            "message": "你已经连续用屏 {minutes} 分钟了，起身活动一下、远眺放松眼睛吧。",
        },
    },
    {
        # Still active late at night (23:00–04:00) → wind down. quiet_hours is
        # disabled ("0-0") so this rule can actually fire during the night.
        "name": "late_night",
        "rule_type": "schedule",
        "enabled": True,
        "priority": "M",
        "cooldown_min": 120,
        "quiet_hours": "0-0",
        "params": {
            "between": [23, 4],
            "message": "已经深夜了，准备收尾、早点休息吧。",
        },
    },
    {
        # Still working through the lunch hour (12:00–13:00) → take a break.
        "name": "lunch_break",
        "rule_type": "schedule",
        "enabled": True,
        "priority": "L",
        "cooldown_min": 180,
        "quiet_hours": "22-8",
        "params": {
            "between": [12, 13],
            "categories": ["coding", "meeting", "writing"],
            "min_minutes": 40,
            "message": "午饭时间到了，已连续工作 {minutes} 分钟，去吃点东西、歇一会吧。",
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
        app_minutes: float | None = None,
    ) -> None:
        self.category = category
        self.app_name = app_name
        self.window_name = window_name
        # Continuous on-screen time across app switches (resets only on idle).
        self.continuous_minutes = continuous_minutes
        # Continuous time in the *current* app (resets on every app switch).
        self.app_minutes = continuous_minutes if app_minutes is None else app_minutes

    def as_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "app_name": self.app_name,
            "window_name": self.window_name,
            "continuous_minutes": round(self.continuous_minutes, 1),
            "app_minutes": round(self.app_minutes, 1),
        }


def _continuous_screen_minutes(
    store: HabitStore, now: datetime, *, idle_gap_sec: float = 300.0,
    lookback_hours: float = 16.0,
) -> float:
    """Minutes the user has been continuously present at the screen, spanning
    app switches. A gap larger than ``idle_gap_sec`` between activity signals
    (frames + UI events) counts as a break and resets the run.

    Returns 0.0 when the most recent activity is already older than the idle
    gap (i.e. the user stepped away).
    """
    start = (now - timedelta(hours=lookback_hours)).replace(microsecond=0).isoformat()
    raw = store.activity_timestamps_since(start)
    stamps = [ts for ts in (_parse_ts(t) for t in raw) if ts is not None]
    if not stamps:
        return 0.0
    # Already idle right now? Then there is no active run.
    if (now - stamps[-1]).total_seconds() > idle_gap_sec:
        return 0.0
    run_start = stamps[-1]
    for i in range(len(stamps) - 1, 0, -1):
        if (stamps[i] - stamps[i - 1]).total_seconds() <= idle_gap_sec:
            run_start = stamps[i - 1]
        else:
            break
    return max(0.0, (now - run_start).total_seconds() / 60.0)


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

    # Per-app continuity: how long in THIS app (resets on app switch). Prefer an
    # app_switch event; fall back to the window length.
    app_minutes = float(window_minutes)
    switch_ts = store.last_app_switch_ts(app_name, end)
    anchor = _parse_ts(switch_ts) if switch_ts else None
    if anchor is not None:
        app_minutes = max(0.0, (now - anchor).total_seconds() / 60.0)

    # Cross-app continuity: how long continuously at the screen (resets on idle).
    continuous_minutes = _continuous_screen_minutes(store, now)
    # Never report less than the per-app figure (data sparsity guard).
    continuous_minutes = max(continuous_minutes, app_minutes)

    return CurrentState(
        category=category,
        app_name=app_name,
        window_name=window_name,
        continuous_minutes=continuous_minutes,
        app_minutes=app_minutes,
    )


def _slot_habit_frequency(profiles: list[dict[str, Any]], categories: list[str]) -> float:
    """Max frequency among the given categories for the current slot."""
    best = 0.0
    for p in profiles:
        if p.get("category") in categories:
            best = max(best, float(p.get("frequency") or 0.0))
    return best


def evaluate_distraction(
    rule: dict[str, Any], state: CurrentState, profiles: list[dict[str, Any]],
    now: datetime | None = None,
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
        and state.app_minutes >= min_minutes
    )
    ctx = {
        "habit_frequency": round(habit_freq, 3),
        "state": state.as_dict(),
        "focus_categories": focus_cats,
    }
    msg = (
        f"现在通常是你的专注时段（{int(round(habit_freq * 100))}% 的同类日子在做 "
        f"{'/'.join(focus_cats)}），但你已经在 {state.app_name} 上停留了 "
        f"{int(state.app_minutes)} 分钟。要回到专注状态吗？"
    )
    return hit, msg, ctx


def evaluate_overwork(
    rule: dict[str, Any], state: CurrentState, profiles: list[dict[str, Any]],
    now: datetime | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    params = rule.get("params", {})
    cats = params.get("categories", ["coding", "meeting"])
    max_minutes = float(params.get("max_minutes", 120))

    cat_ok = (not cats) or (state.category in cats)
    hit = cat_ok and state.continuous_minutes >= max_minutes
    ctx = {"state": state.as_dict(), "max_minutes": max_minutes}
    template = params.get("message")
    if template:
        msg = template.format(
            minutes=int(state.continuous_minutes),
            category=state.category,
            app=state.app_name,
        )
    else:
        msg = (
            f"你已经连续 {int(state.continuous_minutes)} 分钟处于 {state.category} 状态，"
            "起来活动一下、喝口水？"
        )
    return hit, msg, ctx


def evaluate_schedule(
    rule: dict[str, Any], state: CurrentState, profiles: list[dict[str, Any]],
    now: datetime | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    """Time-of-day fixed rule — fires on wall-clock + optional duration, no
    learned history required.

    ``params``:
      * ``between``   — ``[start_hour, end_hour]`` window (wraps midnight when
        ``start > end``, e.g. ``[23, 4]``).
      * ``after_hour``— fire only when ``now.hour >= after_hour``.
      * ``day_types`` — optional subset of ``{"weekday", "weekend"}``.
      * ``categories``— optional category filter (``[]`` = any).
      * ``min_minutes``— minimum continuous minutes in the current app.
      * ``message``   — nudge text (supports ``{minutes}``/``{category}``/``{app}``).
    """
    if now is None:
        return False, "", {}
    params = rule.get("params", {})
    h = now.hour

    in_window = True
    between = params.get("between")
    if between and len(between) == 2:
        s, e = int(between[0]), int(between[1])
        in_window = (s <= h < e) if s < e else (h >= s or h < e)
    after_hour = params.get("after_hour")
    if after_hour is not None:
        in_window = in_window and (h >= int(after_hour))
    day_types = params.get("day_types") or []
    if day_types:
        in_window = in_window and (day_type_of(now) in day_types)

    cats = params.get("categories") or []
    cat_ok = (not cats) or (state.category in cats)
    min_minutes = float(params.get("min_minutes", 0))
    dur_ok = state.continuous_minutes >= min_minutes

    hit = in_window and cat_ok and dur_ok
    template = params.get("message") or "现在该休息一下了。"
    msg = template.format(
        minutes=int(state.continuous_minutes),
        category=state.category,
        app=state.app_name,
    )
    ctx = {"state": state.as_dict(), "hour": h}
    return hit, msg, ctx


_EVALUATORS = {
    "deviation": evaluate_distraction,
    "threshold": evaluate_overwork,
    "schedule": evaluate_schedule,
}


def evaluate_rule(
    rule: dict[str, Any], state: CurrentState, profiles: list[dict[str, Any]],
    now: datetime | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    """Dispatch a rule to its evaluator. Unknown types never fire."""
    evaluator = _EVALUATORS.get(rule.get("rule_type", ""))
    if evaluator is None:
        return False, "", {}
    try:
        return evaluator(rule, state, profiles, now)
    except Exception as exc:  # noqa: BLE001 — a bad rule must never crash the watcher
        logger.warning("rule %s failed: %s", rule.get("name"), exc)
        return False, "", {}


def current_slot_key(now: datetime) -> tuple[str, int]:
    return day_type_of(now), slot_of(now)
