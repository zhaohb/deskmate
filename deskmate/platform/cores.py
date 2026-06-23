"""Per-core CPU load grouped by P-core vs E-core (Intel hybrid topology).

Two Windows facts make this work without any third-party package:
  * GetSystemCpuSetInformation gives each logical processor an EfficiencyClass
    (higher = performance / P-core, 0 = efficient / E/LPE-core).
  * NtQuerySystemInformation(SystemProcessorPerformanceInformation) gives each
    logical processor's cumulative idle/kernel/user times; diffing two samples
    yields per-core busy %.

We sample twice ~0.4s apart, compute per-core utilization, then average within
each efficiency class so the 续航管家 UI can show "P-core load vs E-core load" —
which visualizes that eco-throttled background work really does ride the E-cores.

Best-effort: on non-Windows / API failure, returns an unavailable snapshot
(never raises).
"""

from __future__ import annotations

import ctypes
import sys
import time
from ctypes import wintypes
from dataclasses import dataclass

from ..logger import get

logger = get("platform.cores")

_IS_WINDOWS = sys.platform == "win32"
_SystemProcessorPerformanceInformation = 8


@dataclass(frozen=True)
class CoreLoad:
    available: bool
    p_core_count: int
    e_core_count: int
    p_core_load: float | None   # 0-100 average busy % across P-cores
    e_core_load: float | None   # 0-100 average busy % across E/LPE-cores
    per_core: list[dict]        # [{index, klass: 'P'|'E', load}], for a bar strip


if _IS_WINDOWS:
    _ntdll = ctypes.WinDLL("ntdll")
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _SPPI(ctypes.Structure):
        # SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION
        _fields_ = [
            ("IdleTime", ctypes.c_longlong),
            ("KernelTime", ctypes.c_longlong),  # includes idle
            ("UserTime", ctypes.c_longlong),
            ("DpcTime", ctypes.c_longlong),
            ("InterruptTime", ctypes.c_longlong),
            ("InterruptCount", wintypes.ULONG),
        ]

    _ntdll.NtQuerySystemInformation.argtypes = [
        ctypes.c_int, ctypes.c_void_p, wintypes.ULONG, ctypes.POINTER(wintypes.ULONG),
    ]
    _k32.GetSystemCpuSetInformation.argtypes = [
        ctypes.c_void_p, wintypes.ULONG, ctypes.POINTER(wintypes.ULONG),
        wintypes.HANDLE, wintypes.ULONG,
    ]
    _k32.GetSystemCpuSetInformation.restype = wintypes.BOOL


# Cache the efficiency-class map (topology doesn't change at runtime).
_EFFICIENCY_BY_INDEX: dict[int, int] | None = None


def _efficiency_classes() -> dict[int, int]:
    """Map logical-processor index -> EfficiencyClass byte. {} on failure."""
    global _EFFICIENCY_BY_INDEX
    if _EFFICIENCY_BY_INDEX is not None:
        return _EFFICIENCY_BY_INDEX
    out: dict[int, int] = {}
    if _IS_WINDOWS:
        try:
            length = wintypes.ULONG(0)
            _k32.GetSystemCpuSetInformation(None, 0, ctypes.byref(length), None, 0)
            buf = (ctypes.c_ubyte * length.value)()
            if _k32.GetSystemCpuSetInformation(buf, length.value, ctypes.byref(length), None, 0):
                addr = ctypes.addressof(buf)
                end = addr + length.value
                p = addr
                while p < end:
                    size = ctypes.c_uint32.from_address(p).value
                    if size == 0:
                        break
                    # SYSTEM_CPU_SET_INFORMATION: Size(4) Type(4) then CpuSet union
                    # starting at +8: Id(4) Group(2) LogicalProcessorIndex(1)[+14]
                    # CoreIndex(1) LastLevelCacheIndex(1) NumaNodeIndex(1)
                    # EfficiencyClass(1)[+18]. Offsets verified by byte-dump.
                    lpi = ctypes.c_ubyte.from_address(p + 14).value
                    eff = ctypes.c_ubyte.from_address(p + 18).value
                    out[lpi] = eff
                    p += size
        except Exception as exc:  # noqa: BLE001
            logger.debug("GetSystemCpuSetInformation failed: %s", exc)
    _EFFICIENCY_BY_INDEX = out
    return out


def _sample_times() -> list[tuple[int, int]] | None:
    """Return [(idle, total)] per logical processor, or None on failure."""
    import os  # noqa: PLC0415

    n = os.cpu_count() or 1
    buf = (_SPPI * n)()
    ret = wintypes.ULONG(0)
    status = _ntdll.NtQuerySystemInformation(
        _SystemProcessorPerformanceInformation, buf, ctypes.sizeof(buf), ctypes.byref(ret),
    )
    if status != 0:
        logger.debug("NtQuerySystemInformation status=0x%x", status & 0xFFFFFFFF)
        return None
    out = []
    for c in buf:
        # KernelTime includes IdleTime; total busy = (kernel - idle) + user.
        total = c.KernelTime + c.UserTime
        out.append((c.IdleTime, total))
    return out


def core_load(sample_ms: int = 400) -> CoreLoad:
    """Sample per-core load twice and group by P/E efficiency class. Never raises."""
    if not _IS_WINDOWS:
        return CoreLoad(False, 0, 0, None, None, [])
    try:
        eff = _efficiency_classes()
        s1 = _sample_times()
        if s1 is None:
            return CoreLoad(False, 0, 0, None, None, [])
        time.sleep(max(0.05, sample_ms / 1000.0))
        s2 = _sample_times()
        if s2 is None:
            return CoreLoad(False, 0, 0, None, None, [])

        # Highest efficiency class present == P-cores; class 0 == E/LPE.
        classes = set(eff.values())
        p_class = max(classes) if classes else 1

        per_core: list[dict] = []
        p_loads: list[float] = []
        e_loads: list[float] = []
        for i in range(min(len(s1), len(s2))):
            idle_d = s2[i][0] - s1[i][0]
            total_d = s2[i][1] - s1[i][1]
            busy = 0.0 if total_d <= 0 else max(0.0, min(100.0, 100.0 * (1.0 - idle_d / total_d)))
            is_p = eff.get(i, 0) == p_class and len(classes) > 1
            per_core.append({"index": i, "klass": "P" if is_p else "E", "load": round(busy, 1)})
            (p_loads if is_p else e_loads).append(busy)

        p_avg = round(sum(p_loads) / len(p_loads), 1) if p_loads else None
        e_avg = round(sum(e_loads) / len(e_loads), 1) if e_loads else None
        return CoreLoad(True, len(p_loads), len(e_loads), p_avg, e_avg, per_core)
    except Exception as exc:  # noqa: BLE001
        logger.debug("core_load failed: %s", exc)
        return CoreLoad(False, 0, 0, None, None, [])
