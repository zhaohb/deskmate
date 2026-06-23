"""Thread-level EcoQoS (Windows power throttling) via Win32 SetThreadInformation.

Why per-thread, not per-process: the daemon runs Ask, capture, OCR, redaction
and indexing in ONE process. Process-level throttling would slow Ask too. The
thread API lets us push background workers onto E/LPE-cores while leaving the
Ask path on P-cores — all inside the same process.

We tag threads we do NOT own by opening them by native thread id (TID), so the
existing workers need zero code changes — a central manager finds them by name.

EcoQoS semantics (PROCESS/THREAD_POWER_THROTTLING_EXECUTION_SPEED):
  ControlMask=SPEED, StateMask=SPEED  → "throttle me": scheduler prefers E-cores,
                                         caps frequency  → eco / battery saver.
  ControlMask=SPEED, StateMask=0      → "do NOT throttle me": HighQoS, keep on
                                         P-cores even under a global power-save.
  ControlMask=0,     StateMask=0      → clear our override, follow system default.

All functions are best-effort and never raise: on non-Windows, missing API, or
an access-denied TID they log at debug and return False.
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

from ..logger import get

logger = get("platform.qos")

_IS_WINDOWS = sys.platform == "win32"

# THREAD_INFORMATION_CLASS.ThreadPowerThrottling
_ThreadPowerThrottling = 3
_THREAD_POWER_THROTTLING_CURRENT_VERSION = 1
_THREAD_POWER_THROTTLING_EXECUTION_SPEED = 0x1
# OpenThread access rights
_THREAD_SET_INFORMATION = 0x0020
_THREAD_QUERY_LIMITED_INFORMATION = 0x0800

# PROCESS_INFORMATION_CLASS.ProcessPowerThrottling + access rights (for tagging
# OTHER user-facing apps the user picks — same-user/same-integrity, no admin).
_ProcessPowerThrottling = 4
_PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
_PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
_PROCESS_SET_INFORMATION = 0x0200
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000


if _IS_WINDOWS:

    class _TPTS(ctypes.Structure):
        _fields_ = [
            ("Version", wintypes.ULONG),
            ("ControlMask", wintypes.ULONG),
            ("StateMask", wintypes.ULONG),
        ]

    try:
        _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        _k32.GetCurrentThread.restype = wintypes.HANDLE
        _k32.OpenThread.restype = wintypes.HANDLE
        _k32.OpenThread.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        _k32.SetThreadInformation.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
        ]
        _k32.SetThreadInformation.restype = wintypes.BOOL
        _k32.OpenProcess.restype = wintypes.HANDLE
        _k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        _k32.SetProcessInformation.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD,
        ]
        _k32.SetProcessInformation.restype = wintypes.BOOL
        _k32.CloseHandle.argtypes = [wintypes.HANDLE]
        _AVAILABLE = True
    except Exception as exc:  # noqa: BLE001
        logger.debug("kernel32 QoS init failed: %s", exc)
        _k32 = None
        _AVAILABLE = False
else:
    _k32 = None
    _AVAILABLE = False


def qos_available() -> bool:
    """True if thread-level EcoQoS can be applied on this platform."""
    return _AVAILABLE


def _state(*, control: int, state: int) -> "_TPTS":
    s = _TPTS()
    s.Version = _THREAD_POWER_THROTTLING_CURRENT_VERSION
    s.ControlMask = control
    s.StateMask = state
    return s


def _apply_to_handle(handle, control: int, state: int) -> bool:
    s = _state(control=control, state=state)
    ok = _k32.SetThreadInformation(
        handle, _ThreadPowerThrottling, ctypes.byref(s), ctypes.sizeof(s),
    )
    if not ok:
        logger.debug("SetThreadInformation failed err=%d", ctypes.get_last_error())
    return bool(ok)


def _apply_to_tid(tid: int, control: int, state: int) -> bool:
    """Open another thread by TID and apply the throttling state."""
    handle = _k32.OpenThread(
        _THREAD_SET_INFORMATION | _THREAD_QUERY_LIMITED_INFORMATION, False, tid,
    )
    if not handle:
        logger.debug("OpenThread(%d) failed err=%d", tid, ctypes.get_last_error())
        return False
    try:
        return _apply_to_handle(handle, control, state)
    finally:
        _k32.CloseHandle(handle)


def set_thread_eco(tid: int) -> bool:
    """Throttle thread `tid` onto E/LPE-cores (battery saver). No-op off-Windows."""
    if not _AVAILABLE:
        return False
    return _apply_to_tid(
        tid,
        control=_THREAD_POWER_THROTTLING_EXECUTION_SPEED,
        state=_THREAD_POWER_THROTTLING_EXECUTION_SPEED,
    )


def set_thread_high(tid: int) -> bool:
    """Pin thread `tid` to high performance (P-core), exempt from power-save."""
    if not _AVAILABLE:
        return False
    return _apply_to_tid(
        tid,
        control=_THREAD_POWER_THROTTLING_EXECUTION_SPEED,
        state=0,
    )


def clear_thread(tid: int) -> bool:
    """Drop our override on thread `tid`; it follows the system default again."""
    if not _AVAILABLE:
        return False
    return _apply_to_tid(tid, control=0, state=0)


# ── process-level EcoQoS (for user-selected third-party apps) ────────────────
# Same EXECUTION_SPEED knob, applied to a whole process the user picks. Works
# for same-user/same-integrity processes without admin; OpenProcess fails (and
# we return False, never raise) for elevated/protected processes.
def _apply_to_pid(pid: int, control: int, state: int) -> bool:
    handle = _k32.OpenProcess(
        _PROCESS_SET_INFORMATION | _PROCESS_QUERY_LIMITED_INFORMATION, False, pid,
    )
    if not handle:
        logger.debug("OpenProcess(%d) failed err=%d", pid, ctypes.get_last_error())
        return False
    try:
        s = _TPTS()  # PROCESS_POWER_THROTTLING_STATE has the same 3-DWORD layout
        s.Version = _PROCESS_POWER_THROTTLING_CURRENT_VERSION
        s.ControlMask = control
        s.StateMask = state
        ok = _k32.SetProcessInformation(
            handle, _ProcessPowerThrottling, ctypes.byref(s), ctypes.sizeof(s),
        )
        if not ok:
            logger.debug("SetProcessInformation(%d) failed err=%d", pid, ctypes.get_last_error())
        return bool(ok)
    finally:
        _k32.CloseHandle(handle)


def set_process_eco(pid: int) -> bool:
    """Throttle an entire process onto E/LPE-cores. No-op off-Windows."""
    if not _AVAILABLE:
        return False
    return _apply_to_pid(
        pid,
        control=_PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
        state=_PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
    )


def clear_process(pid: int) -> bool:
    """Drop our override on a process; it follows the system default again."""
    if not _AVAILABLE:
        return False
    return _apply_to_pid(pid, control=0, state=0)


def can_throttle_pid(pid: int) -> bool:
    """True if we can open `pid` for SET_INFORMATION (i.e. throttle it) w/o admin."""
    if not _AVAILABLE:
        return False
    handle = _k32.OpenProcess(_PROCESS_SET_INFORMATION, False, pid)
    if handle:
        _k32.CloseHandle(handle)
        return True
    return False
