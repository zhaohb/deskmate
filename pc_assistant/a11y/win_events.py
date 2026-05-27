"""SetWinEventHook-based foreground / focus / title / value watcher.

Tracks system UI events that trigger paired captures:
  EVENT_SYSTEM_FOREGROUND    → window_focus
  EVENT_OBJECT_FOCUS         → window_focus
  EVENT_OBJECT_NAMECHANGE    → title_change
  EVENT_OBJECT_VALUECHANGE   → value_change

Implementation pattern: dedicated thread owns the message pump (WinEvent hooks
are OUTOFCONTEXT but Windows still routes them to the installing thread's
message loop). Callbacks dispatch into the `events` bus.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import os
import threading
from collections.abc import Callable
from typing import Any

from .. import events as bus
from ..logger import get

logger = get("a11y.win_events")

EVENT_SYSTEM_FOREGROUND = 0x0003
EVENT_OBJECT_FOCUS = 0x8005
EVENT_OBJECT_NAMECHANGE = 0x800C
EVENT_OBJECT_VALUECHANGE = 0x800E

WINEVENT_OUTOFCONTEXT = 0x0000
WINEVENT_SKIPOWNPROCESS = 0x0002
OBJID_WINDOW = 0

WinEventProc = ctypes.WINFUNCTYPE(  # type: ignore[attr-defined]
    None, wt.HANDLE, wt.DWORD, wt.HWND, wt.LONG, wt.LONG, wt.DWORD, wt.DWORD
) if os.name == "nt" else None


def _foreground_hwnd_pid() -> tuple[int, int, str]:
    if os.name != "nt":
        return (0, 0, "")
    user32 = ctypes.windll.user32  # type: ignore[attr-defined]
    user32.GetForegroundWindow.argtypes = []
    user32.GetForegroundWindow.restype = wt.HWND
    user32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
    user32.GetWindowThreadProcessId.restype = wt.DWORD
    hwnd = user32.GetForegroundWindow() or 0
    if not hwnd:
        return (0, 0, "")
    pid = wt.DWORD(0)
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    buf = ctypes.create_unicode_buffer(1024)
    user32.GetWindowTextW(hwnd, buf, 1024)
    return int(hwnd), int(pid.value), buf.value or ""


def foreground_app_name(pid: int) -> str:
    if os.name != "nt" or not pid:
        return ""
    try:
        import psutil  # noqa: PLC0415
        return psutil.Process(pid).name() or ""
    except Exception:  # noqa: BLE001
        return ""


class WinEventWatcher:
    """Owns a thread + message pump that installs WinEvent hooks."""

    EVENTS: dict[int, str] = {
        EVENT_SYSTEM_FOREGROUND: "window_focus",
        EVENT_OBJECT_FOCUS: "window_focus",
        EVENT_OBJECT_NAMECHANGE: "title_change",
        EVENT_OBJECT_VALUECHANGE: "value_change",
    }

    def __init__(self, on_event: Callable[[dict[str, Any]], None] | None = None) -> None:
        self._on_event = on_event
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._hooks: list[wt.HANDLE] = []

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def available(self) -> bool:
        return os.name == "nt"

    def start(self) -> None:
        if not self.available:
            logger.info("WinEventWatcher not available off Windows")
            return
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="WinEventWatcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        SetWinEventHook = user32.SetWinEventHook
        SetWinEventHook.restype = wt.HANDLE
        UnhookWinEvent = user32.UnhookWinEvent
        GetMessageW = user32.GetMessageW
        TranslateMessage = user32.TranslateMessage
        DispatchMessageW = user32.DispatchMessageW

        @WinEventProc  # type: ignore[misc]
        def _proc(hook, event, hwnd, id_object, id_child, thread, time):  # noqa: ANN001
            if id_object != OBJID_WINDOW:
                return
            try:
                ev_name = self.EVENTS.get(int(event))
                if not ev_name:
                    return
                hwnd_i, pid_i, title = _foreground_hwnd_pid()
                app = foreground_app_name(pid_i)
                payload = {
                    "hwnd": hwnd_i,
                    "pid": pid_i,
                    "app_name": app,
                    "window_title": title,
                }
                bus.send(bus.EventType.WINDOW_FOCUS if ev_name == "window_focus"
                         else bus.EventType.TITLE_CHANGE if ev_name == "title_change"
                         else bus.EventType.VALUE_CHANGE, **payload)
                if self._on_event:
                    self._on_event({"event_type": ev_name, **payload})
            except Exception as exc:  # noqa: BLE001
                logger.warning("WinEvent callback error: %s", exc)

        # keep proc reachable so it's not GC'd
        self._proc = _proc

        for ev in self.EVENTS:
            h = SetWinEventHook(ev, ev, 0, _proc, 0, 0, WINEVENT_OUTOFCONTEXT | WINEVENT_SKIPOWNPROCESS)
            if h:
                self._hooks.append(h)

        logger.info("WinEventWatcher installed %d hooks", len(self._hooks))
        msg = wt.MSG()
        while not self._stop.is_set():
            r = user32.PeekMessageW(ctypes.byref(msg), 0, 0, 0, 0x0001)  # PM_REMOVE
            if r:
                TranslateMessage(ctypes.byref(msg))
                DispatchMessageW(ctypes.byref(msg))
            else:
                self._stop.wait(0.02)

        for h in self._hooks:
            UnhookWinEvent(h)
        self._hooks.clear()
        logger.info("WinEventWatcher stopped")
