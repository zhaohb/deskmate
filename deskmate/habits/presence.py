"""Presence / interruptibility detection — "is it rude to nudge right now?".

The notifier consults :func:`busy_reason` before showing a proactive reminder.
Every probe is best-effort and FAIL-OPEN: if a check can't run (non-Windows, API
missing, error), it returns ``None`` (= "not busy") so a detection gap never
silences reminders entirely. The meeting signal is read from the DB by the
caller (the daemon's MeetingDetector owns it); this module covers the two
machine-local signals — a full-screen app (presentation / video / game) and the
Windows Focus Assist "do not disturb" mode.
"""

from __future__ import annotations

import sys

from ..logger import get

logger = get("habits.presence")


def is_fullscreen_foreground() -> bool | None:
    """True if the foreground window covers the whole primary monitor.

    Heuristic for "presenting / watching video / gaming, don't interrupt". Uses
    Win32 via pywin32. Returns ``None`` when it can't be determined (so the
    caller treats it as not-busy). The desktop / shell itself is excluded so an
    empty desktop doesn't read as fullscreen.
    """
    if sys.platform != "win32":
        return None
    try:
        import win32api  # noqa: PLC0415
        import win32con  # noqa: PLC0415
        import win32gui  # noqa: PLC0415

        hwnd = win32gui.GetForegroundWindow()
        if not hwnd:
            return None
        cls = win32gui.GetClassName(hwnd)
        # Shell / desktop windows are never "fullscreen apps".
        if cls in ("Progman", "WorkerW", "Shell_TrayWnd", ""):
            return False
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        win_w, win_h = right - left, bottom - top
        # Monitor the window is on (handles multi-monitor + scaling).
        monitor = win32api.MonitorFromWindow(hwnd, win32con.MONITOR_DEFAULTTOPRIMARY)
        info = win32api.GetMonitorInfo(monitor)
        m_left, m_top, m_right, m_bottom = info["Monitor"]
        mon_w, mon_h = m_right - m_left, m_bottom - m_top
        # Allow a few px slack for borderless-fullscreen rounding.
        return win_w >= mon_w - 2 and win_h >= mon_h - 2
    except Exception as exc:  # noqa: BLE001 — fail-open
        logger.debug("fullscreen probe failed: %s", exc)
        return None


# Windows Focus Assist (Quiet Hours) state, read from the per-user registry key
# Microsoft uses for the notification-center setting. 0 = Off, 1 = Priority
# only, 2 = Alarms only. Anything > 0 means the user asked not to be disturbed.
_FOCUS_ASSIST_VALUE = (
    r"SOFTWARE\Microsoft\Windows\CurrentVersion\Notifications\Settings"
    r"\Windows.SystemToast.QuietHours"
)


def focus_assist_on() -> bool | None:
    """True if Windows Focus Assist / Do-Not-Disturb is active. ``None`` if unknown.

    There is no stable public API; the cloud-store registry value Windows writes
    for the toast setting is the pragmatic signal. Best-effort and fail-open."""
    if sys.platform != "win32":
        return None
    try:
        import winreg  # noqa: PLC0415

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, _FOCUS_ASSIST_VALUE) as key:
            val, _ = winreg.QueryValueEx(key, "Enabled")
            return int(val) > 0
    except FileNotFoundError:
        return False  # key absent → feature simply off
    except Exception as exc:  # noqa: BLE001 — fail-open
        logger.debug("focus-assist probe failed: %s", exc)
        return None


def busy_reason(*, in_meeting: bool = False) -> str | None:
    """Return a short reason string if NOW is a bad time to interrupt, else None.

    ``in_meeting`` is supplied by the caller (read from the meetings table). The
    full-screen and focus-assist checks run here. First positive signal wins;
    order is meeting → focus-assist → fullscreen (most→least certain intent)."""
    if in_meeting:
        return "in_meeting"
    if focus_assist_on() is True:
        return "focus_assist"
    if is_fullscreen_foreground() is True:
        return "fullscreen"
    return None
