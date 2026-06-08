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
from xml.sax.saxutils import escape

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


# AppUserModelID the toast is shown under. Pointing at the stock Windows
# PowerShell app id lets the toast display on any Win10/11 box without having
# to register a custom Start-menu shortcut for DeskMate.
_TOAST_APP_ID = (
    r"{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe"
)


def _try_windows_toast(title: str, message: str) -> bool:
    """Show a native Windows toast in-process via the ``winrt`` projection.

    Single, deterministic delivery method — no subprocess, no backend probing.
    User text is XML-escaped (injection-safe). Never raises; returns ``True``
    only when the toast was actually shown.
    """
    try:
        from winrt.windows.data.xml.dom import XmlDocument
        from winrt.windows.ui.notifications import (
            ToastNotification,
            ToastNotificationManager,
        )

        # scenario="reminder" keeps the toast on screen until the user acts on
        # it (instead of auto-dismissing after ~5s), so an away-from-keyboard
        # user still sees it. A reminder needs at least one action, so a single
        # system "知道了" dismiss button is provided. The audio element adds a
        # reminder chime to catch attention even when the user isn't looking.
        xml = (
            "<toast scenario='reminder'>"
            "<visual><binding template='ToastGeneric'>"
            f"<text>{escape(title)}</text><text>{escape(message)}</text>"
            "</binding></visual>"
            "<audio src='ms-winsoundevent:Notification.Reminder' loop='false'/>"
            "<actions>"
            "<action content='知道了' arguments='dismiss' activationType='system'/>"
            "</actions>"
            "</toast>"
        )
        doc = XmlDocument()
        doc.load_xml(xml)
        notifier = ToastNotificationManager.create_toast_notifier_with_id(_TOAST_APP_ID)
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
        # Anchor the cooldown on the LATER of "last sent" and "last acknowledged":
        # if the user actively clicked/dismissed/rated the nudge, the quiet window
        # restarts from that click — a manual ack means "I dealt with it, leave me
        # alone for another cooldown" rather than re-nagging on the send schedule.
        candidates = [
            self._store.last_suggestion_ts(rule["name"]),
            self._store.last_acknowledged_ts(rule["name"]),
        ]
        anchors: list[datetime] = []
        for ts in candidates:
            if not ts:
                continue
            try:
                dt = datetime.fromisoformat(str(ts).replace(" ", "T"))
            except ValueError:
                continue
            if dt.tzinfo is None and now.tzinfo is not None:
                dt = dt.replace(tzinfo=now.tzinfo)
            anchors.append(dt)
        if not anchors:
            return False
        cooldown = timedelta(minutes=int(rule.get("cooldown_min", 120)))
        return (now - max(anchors)) < cooldown

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
