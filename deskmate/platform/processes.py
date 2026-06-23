"""User-facing app enumeration + user-driven EcoQoS control.

Lists the apps the user actually sees (processes owning a visible, titled
top-level window) — not the ~200 background services — so a UI can show a
pick-list. The user chooses which apps to push onto E/LPE-cores; we never
auto-pick, which is what keeps this safe (no "did DeskMate just slow down the
app I'm using?" risk).

Capabilities, all verified to work same-user without admin:
  * enumerate apps (EnumWindows + visible + has-title) with exe name, PID,
    foreground flag, and whether we can throttle it.
  * apply / clear process-level EcoQoS for a PID the user selected.

Everything degrades gracefully off-Windows: enumeration returns [] and control
calls return False (never raise).
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from dataclasses import dataclass

from ..logger import get
from .qos import (
    can_throttle_pid,
    clear_process,
    qos_available,
    set_process_eco,
)

logger = get("platform.processes")

_IS_WINDOWS = sys.platform == "win32"

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


@dataclass(frozen=True)
class AppInfo:
    pid: int
    name: str             # exe file name, e.g. "chrome.exe"
    title: str            # a representative window title
    is_foreground: bool   # the app the user is currently looking at
    can_throttle: bool    # can we eco-throttle it without admin?


if _IS_WINDOWS:
    _u32 = ctypes.WinDLL("user32", use_last_error=True)
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    _WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    _u32.EnumWindows.argtypes = [_WNDENUMPROC, wintypes.LPARAM]
    _u32.IsWindowVisible.argtypes = [wintypes.HWND]
    _u32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    _u32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    _u32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    _u32.GetForegroundWindow.restype = wintypes.HWND
    _k32.OpenProcess.restype = wintypes.HANDLE
    _k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    _k32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD),
    ]
    _k32.CloseHandle.argtypes = [wintypes.HANDLE]


def _exe_name(pid: int) -> str:
    handle = _k32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(1024)
        if _k32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return buf.value.replace("\\", "/").split("/")[-1]
    finally:
        _k32.CloseHandle(handle)
    return ""


# DeskMate's own process should never appear as a throttle target.
_SELF_PID = None


def _self_pid() -> int:
    global _SELF_PID
    if _SELF_PID is None:
        import os  # noqa: PLC0415

        _SELF_PID = os.getpid()
    return _SELF_PID


def list_apps() -> list[AppInfo]:
    """Enumerate user-facing apps (visible, titled top-level windows). [] off-Windows."""
    if not _IS_WINDOWS:
        return []

    fg_hwnd = _u32.GetForegroundWindow()
    fg_pid = wintypes.DWORD()
    _u32.GetWindowThreadProcessId(fg_hwnd, ctypes.byref(fg_pid))

    seen: dict[int, str] = {}  # pid -> first window title

    def _cb(hwnd, _lparam):  # noqa: ANN001, ANN202
        if not _u32.IsWindowVisible(hwnd):
            return True
        n = _u32.GetWindowTextLengthW(hwnd)
        if n == 0:
            return True
        pid = wintypes.DWORD()
        _u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value and pid.value not in seen:
            buf = ctypes.create_unicode_buffer(n + 1)
            _u32.GetWindowTextW(hwnd, buf, n + 1)
            seen[pid.value] = buf.value
        return True

    try:
        _u32.EnumWindows(_WNDENUMPROC(_cb), 0)
    except Exception as exc:  # noqa: BLE001
        logger.debug("EnumWindows failed: %s", exc)
        return []

    self_pid = _self_pid()
    apps: list[AppInfo] = []
    for pid, title in seen.items():
        if pid == self_pid:
            continue  # never list ourselves
        name = _exe_name(pid)
        if not name:
            continue  # couldn't resolve → skip rather than show a blank row
        apps.append(AppInfo(
            pid=pid,
            name=name,
            title=title,
            is_foreground=(pid == fg_pid.value),
            can_throttle=can_throttle_pid(pid),
        ))
    apps.sort(key=lambda a: (not a.can_throttle, a.name.lower()))
    return apps


class AppPowerController:
    """Tracks which user-selected PIDs are currently eco-throttled.

    Stateful so the UI can show what's active and clear everything on demand.
    Note: PIDs are not persisted across reboots — a throttled app that exits
    simply drops out of the set on the next reconcile.
    """

    def __init__(self) -> None:
        self._eco_pids: set[int] = set()

    def available(self) -> bool:
        return qos_available()

    def eco(self, pid: int) -> bool:
        """Push a user-selected app onto E-cores."""
        if set_process_eco(pid):
            self._eco_pids.add(pid)
            logger.info("eco-throttled app pid=%d", pid)
            return True
        return False

    def restore(self, pid: int) -> bool:
        """Restore a previously eco'd app to the system default."""
        ok = clear_process(pid)
        self._eco_pids.discard(pid)
        if ok:
            logger.info("restored app pid=%d", pid)
        return ok

    def restore_all(self) -> int:
        n = 0
        for pid in list(self._eco_pids):
            if clear_process(pid):
                n += 1
        self._eco_pids.clear()
        return n

    def active_pids(self) -> list[int]:
        return sorted(self._eco_pids)

    def list_with_state(self) -> list[dict]:
        """App pick-list annotated with whether each is currently eco'd by us."""
        out = []
        for a in list_apps():
            out.append({
                "pid": a.pid,
                "name": a.name,
                "title": a.title,
                "is_foreground": a.is_foreground,
                "can_throttle": a.can_throttle,
                "eco": a.pid in self._eco_pids,
            })
        return out
