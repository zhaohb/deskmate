"""Power-source detection via Win32 GetSystemPowerStatus.

Reports whether the machine is on AC (plugged in) or on battery, plus the
charge percentage. Best-effort: on non-Windows or API failure, returns
PowerSource.UNKNOWN so callers can fail-open (treat as AC → no throttling).
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes
from dataclasses import dataclass
from enum import Enum

from ..logger import get

logger = get("platform.battery")

_IS_WINDOWS = sys.platform == "win32"

# SYSTEM_POWER_STATUS.ACLineStatus
_AC_OFFLINE = 0   # on battery
_AC_ONLINE = 1    # plugged in
_AC_UNKNOWN = 255

# BatteryFlag bit: no system battery present (desktop)
_BATTERY_FLAG_NO_BATTERY = 128
# BatteryLifePercent sentinel for "unknown"
_PERCENT_UNKNOWN = 255
# BatteryLifeTime sentinel for "unknown" (also returned while on AC)
_LIFETIME_UNKNOWN = 0xFFFFFFFF


class PowerSource(str, Enum):
    AC = "ac"            # plugged in
    BATTERY = "battery"  # running on battery
    UNKNOWN = "unknown"  # could not determine (fail-open as AC)


@dataclass(frozen=True)
class PowerStatus:
    source: PowerSource
    percent: int | None        # 0-100, or None if unknown
    has_battery: bool          # False on a desktop with no battery
    runtime_seconds: int | None  # OS estimate of battery runtime left; None on AC/unknown


if _IS_WINDOWS:

    class _SYSTEM_POWER_STATUS(ctypes.Structure):
        _fields_ = [
            ("ACLineStatus", wintypes.BYTE),
            ("BatteryFlag", wintypes.BYTE),
            ("BatteryLifePercent", wintypes.BYTE),
            ("SystemStatusFlag", wintypes.BYTE),
            ("BatteryLifeTime", wintypes.DWORD),
            ("BatteryFullLifeTime", wintypes.DWORD),
        ]

    try:
        _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        _k32.GetSystemPowerStatus.argtypes = [ctypes.POINTER(_SYSTEM_POWER_STATUS)]
        _k32.GetSystemPowerStatus.restype = wintypes.BOOL
        _AVAILABLE = True
    except Exception as exc:  # noqa: BLE001
        logger.debug("GetSystemPowerStatus init failed: %s", exc)
        _k32 = None
        _AVAILABLE = False
else:
    _k32 = None
    _AVAILABLE = False


def power_status() -> PowerStatus:
    """Read the current power status. Never raises."""
    if not _AVAILABLE:
        return PowerStatus(PowerSource.UNKNOWN, None, has_battery=False, runtime_seconds=None)
    sps = _SYSTEM_POWER_STATUS()
    if not _k32.GetSystemPowerStatus(ctypes.byref(sps)):
        logger.debug("GetSystemPowerStatus failed err=%d", ctypes.get_last_error())
        return PowerStatus(PowerSource.UNKNOWN, None, has_battery=False, runtime_seconds=None)

    # ctypes BYTE is signed; mask to 0-255.
    ac = sps.ACLineStatus & 0xFF
    flag = sps.BatteryFlag & 0xFF
    pct_raw = sps.BatteryLifePercent & 0xFF
    life_raw = sps.BatteryLifeTime & 0xFFFFFFFF

    has_battery = not (flag & _BATTERY_FLAG_NO_BATTERY)
    percent = None if pct_raw == _PERCENT_UNKNOWN else int(pct_raw)
    runtime_seconds = None if life_raw == _LIFETIME_UNKNOWN else int(life_raw)

    if ac == _AC_ONLINE:
        source = PowerSource.AC
    elif ac == _AC_OFFLINE:
        source = PowerSource.BATTERY
    else:
        source = PowerSource.UNKNOWN
    return PowerStatus(source, percent, has_battery, runtime_seconds)


def power_source() -> PowerSource:
    """Convenience: just the AC/battery/unknown verdict."""
    return power_status().source
