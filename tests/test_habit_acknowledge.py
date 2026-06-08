"""Tests for nudge acknowledgement: clicking '知道了' restarts cooldown and
resets continuous-screen-time, so an acted-on reminder doesn't immediately refire.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from deskmate.db import DatabaseManager
from deskmate.habits.notifier import Notifier
from deskmate.habits.store import HabitStore


def _store(tmp_path: Path) -> HabitStore:
    db = tmp_path / "t.db"
    DatabaseManager(db)  # creates the schema (incl. acknowledged_at column)
    return HabitStore(db)


def test_acknowledged_at_stamped_on_dismiss_and_feedback(tmp_path: Path) -> None:
    store = _store(tmp_path)
    sid = store.insert_suggestion(
        rule_name="break_reminder", message="歇会儿", context={}, channel="ui", status="sent",
    )
    assert store.last_acknowledged_ts("break_reminder") is None  # not acked yet
    store.set_suggestion_status(sid, "dismissed")
    assert store.last_acknowledged_ts("break_reminder") is not None  # dismiss stamps it

    sid2 = store.insert_suggestion(
        rule_name="overwork", message="累了", context={}, channel="ui", status="sent",
    )
    store.set_suggestion_feedback(sid2, 1)
    assert store.last_acknowledged_ts("overwork") is not None  # rating stamps it too
    store.close()


def test_cooldown_restarts_from_acknowledgement(tmp_path: Path) -> None:
    store = _store(tmp_path)
    rule = {"name": "overwork", "cooldown_min": 90, "quiet_hours": "0-0"}
    notifier = Notifier(store, daily_quota=99)
    now = datetime.now().astimezone()

    # A nudge was sent 2h ago — past the 90-min cooldown, so normally re-fireable.
    sid = store.insert_suggestion(
        rule_name="overwork", message="x", context={}, channel="ui", status="sent",
    )
    store._conn.execute(  # backdate the send
        "UPDATE habit_suggestions SET created_at = ? WHERE id = ?",
        ((now - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"), sid),
    )
    assert not notifier._in_cooldown(rule, now)  # send alone: cooldown expired

    # But the user acknowledged it just now → cooldown restarts from the click.
    store.set_suggestion_status(sid, "dismissed")
    assert notifier._in_cooldown(rule, now)  # acked recently → still in cooldown
    store.close()


def test_continuous_time_capped_by_acknowledgement(tmp_path: Path) -> None:
    from deskmate.habits import rules as rules_mod

    store = _store(tmp_path)
    now = datetime.now().astimezone()
    # Simulate 60 min of continuous frames ending "now". Frames are stored as
    # isoformat (T separator + tz), matching how the recorder writes them.
    base = now - timedelta(minutes=60)
    with store._lock:
        for i in range(61):
            ts = (base + timedelta(minutes=i)).replace(microsecond=0).isoformat()
            store._conn.execute(
                "INSERT INTO frames(timestamp, app_name, window_name) VALUES (?,?,?)",
                (ts, "Code.exe", "main.py"),
            )

    # Without any ack: continuous run ≈ 60 min.
    mins_before = rules_mod._continuous_screen_minutes(store, now)
    assert mins_before > 45

    # User acknowledged a break 5 min ago → run is clamped to ≈5 min.
    sid = store.insert_suggestion(
        rule_name="break_reminder", message="歇会儿", context={}, channel="ui", status="sent",
    )
    store.set_suggestion_status(sid, "dismissed")
    store._conn.execute(
        "UPDATE habit_suggestions SET acknowledged_at = ? WHERE id = ?",
        ((now - timedelta(minutes=5)).replace(microsecond=0).isoformat(), sid),
    )
    mins_after = rules_mod._continuous_screen_minutes(store, now)
    assert mins_after <= 6, f"expected ~5 min after ack, got {mins_after}"
    store.close()
