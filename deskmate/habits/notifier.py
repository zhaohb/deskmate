"""Notifier — the single, guarded exit for every proactive suggestion.

Responsibilities:
* Quiet hours, per-rule cooldown, and a global daily quota (anti-nag).
* Feedback decay — a rule repeatedly marked "not useful" is auto-disabled.
* Delivery to channels: a best-effort native Windows toast, and always a row
  in ``habit_suggestions`` that the UI inbox polls.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from ..logger import get
from .store import HabitStore

logger = get("habits.notifier")


def parse_quiet_hours(spec: str) -> tuple[int, int] | None:
    """Parse a ``"start-end"`` hour spec (e.g. ``"22-8"``) into ``(start, end)``."""
    try:
        a, b = spec.split("-", 1)
        return int(a) % 24, int(b) % 24
    except (ValueError, AttributeError):
        return None


def in_quiet_hours(now: datetime, spec: str) -> bool:
    parsed = parse_quiet_hours(spec)
    if not parsed:
        return False
    start, end = parsed
    h = now.hour
    if start == end:
        return False
    if start < end:
        return start <= h < end
    # Wraps midnight (e.g. 22-8).
    return h >= start or h < end


def _try_windows_toast(title: str, message: str) -> bool:
    """Attempt a native Windows toast. Returns True on success, never raises."""
    try:
        from winrt.windows.ui.notifications import (  # type: ignore[import-not-found]
            ToastNotification,
            ToastNotificationManager,
        )
        from winrt.windows.data.xml.dom import XmlDocument  # type: ignore[import-not-found]

        xml = (
            "<toast><visual><binding template='ToastGeneric'>"
            f"<text>{title}</text><text>{message}</text>"
            "</binding></visual></toast>"
        )
        doc = XmlDocument()
        doc.load_xml(xml)
        notifier = ToastNotificationManager.create_toast_notifier("DeskMate")
        notifier.show(ToastNotification(doc))
        return True
    except Exception as exc:  # noqa: BLE001 — toasts are best-effort only
        logger.debug("windows toast unavailable: %s", exc)
        return False


class Notifier:
    def __init__(
        self,
        store: HabitStore,
        *,
        daily_quota: int = 5,
        toast_enabled: bool = True,
        feedback_decay_strikes: int = 3,
    ) -> None:
        self._store = store
        self._daily_quota = daily_quota
        self._toast_enabled = toast_enabled
        self._decay_strikes = feedback_decay_strikes

    def _quota_exceeded(self, now: datetime) -> bool:
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        return self._store.count_sent_since(day_start) >= self._daily_quota

    def _in_cooldown(self, rule: dict[str, Any], now: datetime) -> bool:
        last = self._store.last_suggestion_ts(rule["name"])
        if not last:
            return False
        try:
            last_dt = datetime.fromisoformat(str(last).replace(" ", "T"))
        except ValueError:
            return False
        if last_dt.tzinfo is None and now.tzinfo is not None:
            last_dt = last_dt.replace(tzinfo=now.tzinfo)
        cooldown = timedelta(minutes=int(rule.get("cooldown_min", 120)))
        return (now - last_dt) < cooldown

    def _decayed(self, rule: dict[str, Any]) -> bool:
        """True if the rule has been marked unhelpful too many times in a row."""
        recent = self._store.recent_feedback(rule["name"], limit=self._decay_strikes)
        return len(recent) >= self._decay_strikes and all(v < 0 for v in recent)

    def deliver(self, rule: dict[str, Any], message: str, context: dict[str, Any], now: datetime) -> dict[str, Any]:
        """Run all gates, then deliver. Returns a result dict for logging/tests."""
        name = rule["name"]

        if self._decayed(rule):
            self._store.set_rule_enabled(name, False)
            logger.info("rule %s auto-disabled after repeated negative feedback", name)
            return {"status": "disabled", "rule": name}

        if in_quiet_hours(now, rule.get("quiet_hours", "22-8")):
            self._store.insert_suggestion(
                rule_name=name, message=message, context=context,
                channel="ui", status="suppressed",
            )
            return {"status": "suppressed", "reason": "quiet_hours", "rule": name}

        if self._in_cooldown(rule, now):
            return {"status": "suppressed", "reason": "cooldown", "rule": name}

        if self._quota_exceeded(now):
            self._store.insert_suggestion(
                rule_name=name, message=message, context=context,
                channel="ui", status="suppressed",
            )
            return {"status": "suppressed", "reason": "daily_quota", "rule": name}

        channel = "ui"
        if self._toast_enabled and _try_windows_toast("DeskMate", message):
            channel = "toast"

        sid = self._store.insert_suggestion(
            rule_name=name, message=message, context=context,
            channel=channel, status="sent",
        )
        logger.info("suggestion sent id=%s rule=%s channel=%s", sid, name, channel)
        return {"status": "sent", "id": sid, "rule": name, "channel": channel}
