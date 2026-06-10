"""Tests for the 3-tier reminder redesign:

* eye_break / standup fire on CONTINUOUS screen time (the honest fix), with
  smart deferral around a natural break.
* bilingual (zh/en) message rendering.
* global master switch, per-rule snooze, and presence gating in the notifier.
* break_reminder is retired and superseded by eye_break/standup.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from deskmate.db import DatabaseManager
from deskmate.habits import rules as rules_mod
from deskmate.habits.notifier import Notifier
from deskmate.habits.rules import CurrentState, evaluate_rule, render_message
from deskmate.habits.store import HabitStore


def _store(tmp_path: Path) -> HabitStore:
    db = tmp_path / "t.db"
    DatabaseManager(db)
    return HabitStore(db)


def _state(continuous=80.0, since_switch=None, cat="coding"):
    return CurrentState(
        category=cat, app_name="Code.exe", window_name="main.py",
        continuous_minutes=continuous, app_minutes=continuous,
        today_category_minutes={cat: continuous},
        seconds_since_app_switch=since_switch,
    )


# ─── continuous tier + smart deferral ────────────────────────────────────────

def test_continuous_below_threshold_does_not_fire():
    rule = {"rule_type": "continuous", "params": {"max_minutes": 70, "defer_minutes": 10}}
    hit, _, _ = evaluate_rule(rule, _state(continuous=40, since_switch=10), [])
    assert not hit


def test_continuous_defers_mid_flow_then_fires_at_break():
    rule = {"rule_type": "continuous", "params": {"max_minutes": 70, "defer_minutes": 10}}
    # In the deferral band (70..80), deep in one app (no recent switch) → defer.
    hit_busy, _, ctx_busy = evaluate_rule(rule, _state(continuous=72, since_switch=600), [])
    assert not hit_busy and ctx_busy["deferred"] is True
    # Same band, but a switch 30s ago = natural break → fire.
    hit_break, _, _ = evaluate_rule(rule, _state(continuous=72, since_switch=30), [])
    assert hit_break


def test_continuous_fires_anyway_past_grace():
    rule = {"rule_type": "continuous", "params": {"max_minutes": 70, "defer_minutes": 10}}
    # Past max+defer (>80), even mid-flow → health wins.
    hit, _, _ = evaluate_rule(rule, _state(continuous=95, since_switch=9999), [])
    assert hit


def test_continuous_respects_category_filter():
    rule = {"rule_type": "continuous", "params": {"max_minutes": 30, "categories": ["coding"]}}
    hit, _, _ = evaluate_rule(rule, _state(continuous=60, cat="browsing"), [])
    assert not hit


# ─── bilingual rendering ─────────────────────────────────────────────────────

def test_render_message_picks_language_and_falls_back():
    params = {"messages": {"zh": "连续 {minutes} 分钟", "en": "{minutes} min straight"}}
    assert render_message(params, "en", {"minutes": 50}, "d") == "50 min straight"
    assert render_message(params, "zh", {"minutes": 50}, "d") == "连续 50 分钟"
    # Unknown lang → zh fallback.
    assert render_message(params, "fr", {"minutes": 50}, "d") == "连续 50 分钟"
    # Legacy single message string still honored.
    assert render_message({"message": "hi {minutes}"}, "en", {"minutes": 3}, "d") == "hi 3"


def test_overwork_renders_english():
    rule = next(r for r in rules_mod.DEFAULT_RULES if r["name"] == "overwork")
    st = _state(continuous=300, cat="coding")
    _, msg_en, _ = evaluate_rule(rule, st, [], None, "en")
    _, msg_zh, _ = evaluate_rule(rule, st, [], None, "zh")
    assert "worked" in msg_en and "今天" in msg_zh


# ─── global switch ───────────────────────────────────────────────────────────

def test_global_switch_suppresses_everything(tmp_path):
    store = _store(tmp_path)
    notifier = Notifier(store, daily_quota=99)
    rule = {"name": "standup", "cooldown_min": 60, "quiet_hours": "0-0"}
    now = datetime.now().astimezone()
    store.set_notifications_enabled(False)
    res = notifier.deliver(rule, "msg", {}, now)
    assert res["status"] == "suppressed" and res["reason"] == "globally_disabled"
    store.set_notifications_enabled(True)
    res2 = notifier.deliver(rule, "msg", {}, now)
    assert res2["status"] == "sent"
    store.close()


# ─── per-rule snooze ─────────────────────────────────────────────────────────

def test_snooze_mutes_rule_until_expiry(tmp_path):
    store = _store(tmp_path)
    store.ensure_rules(rules_mod.DEFAULT_RULES)
    notifier = Notifier(store, daily_quota=99)
    now = datetime.now().astimezone()

    future = (now + timedelta(minutes=30)).replace(microsecond=0).isoformat()
    store.snooze_rule("standup", future)
    rule = next(r for r in store.enabled_rules() if r["name"] == "standup")
    assert notifier._snoozed(rule, now)
    res = notifier.deliver(rule, "msg", {}, now)
    assert res["status"] == "suppressed" and res["reason"] == "snoozed"

    # Past expiry → no longer snoozed.
    past = (now - timedelta(minutes=1)).replace(microsecond=0).isoformat()
    store.snooze_rule("standup", past)
    rule2 = next(r for r in store.enabled_rules() if r["name"] == "standup")
    assert not notifier._snoozed(rule2, now)
    store.close()


# ─── presence gating ─────────────────────────────────────────────────────────

def test_busy_reason_holds_nudge(tmp_path):
    store = _store(tmp_path)
    notifier = Notifier(store, daily_quota=99)
    rule = {"name": "standup", "cooldown_min": 60, "quiet_hours": "0-0"}
    now = datetime.now().astimezone()
    res = notifier.deliver(rule, "msg", {}, now, busy_reason="in_meeting")
    assert res["status"] == "suppressed" and res["reason"] == "in_meeting"
    store.close()


# ─── compliance backoff ──────────────────────────────────────────────────────

def test_cooldown_backs_off_when_ignored(tmp_path):
    store = _store(tmp_path)
    notifier = Notifier(store, daily_quota=99)
    rule = {"name": "standup", "cooldown_min": 60}
    now = datetime.now().astimezone()
    base = notifier._cooldown_minutes(rule, now)
    assert base == 60  # nothing sent yet

    # 4 sent, none acknowledged, all within the window → ×3 backoff.
    for _ in range(4):
        store.insert_suggestion(
            rule_name="standup", message="m", context={}, channel="toast", status="sent",
        )
    assert notifier._cooldown_minutes(rule, now) == 180
    store.close()


# ─── migration: break_reminder retired ───────────────────────────────────────

def test_break_reminder_retired_and_split(tmp_path):
    store = _store(tmp_path)
    # Seed a legacy break_reminder, then run the watcher's migration sequence.
    store.ensure_rules([{"name": "break_reminder", "rule_type": "threshold",
                         "enabled": True, "params": {}}])
    store.delete_rules(rules_mod.RETIRED_RULE_NAMES)
    store.ensure_rules(rules_mod.DEFAULT_RULES)
    names = {r["name"] for r in store.enabled_rules()}
    assert "break_reminder" not in names
    assert {"eye_break", "standup", "overwork"} <= names
    store.close()
