"""Low-level mouse + keyboard hooks.

We aggregate keystrokes into `key_text` events with a debounce so we don't
emit one event per character. Mouse clicks emit immediately."""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import os
import threading
import time
from collections.abc import Callable
from typing import Any

from .. import events as bus
from ..logger import get
from .activity_feed import ActivityKind
from .activity_feed import default as activity_default
from .win_events import _foreground_hwnd_pid, foreground_app_name

logger = get("a11y.input")

WH_KEYBOARD_LL = 13
WH_MOUSE_LL = 14
WM_KEYDOWN = 0x0100
WM_SYSKEYDOWN = 0x0104
WM_LBUTTONDOWN = 0x0201
WM_RBUTTONDOWN = 0x0204
WM_MBUTTONDOWN = 0x0207
WM_XBUTTONDOWN = 0x020B
WM_MOUSEWHEEL = 0x020A
HC_ACTION = 0

_MOVE_THROTTLE_S = 0.25

VK_BACK = 0x08
VK_TAB = 0x09
VK_RETURN = 0x0D
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12
VK_CAPITAL = 0x14
VK_ESCAPE = 0x1B
VK_SPACE = 0x20
VK_PRIOR = 0x21
VK_DOWN = 0x28
VK_RIGHT = 0x27
VK_UP = 0x26
VK_LEFT = 0x25
VK_DELETE = 0x2E
VK_LWIN = 0x5B
VK_RWIN = 0x5C
VK_F1 = 0x70
VK_F24 = 0x87

_NAV_KEYS = set(range(VK_PRIOR, VK_DELETE + 1)) | {VK_ESCAPE, VK_CAPITAL, VK_TAB} | set(range(VK_F1, VK_F24 + 1)) | {VK_LWIN, VK_RWIN}
_MODIFIERS = {VK_SHIFT, VK_CONTROL, VK_MENU}


class _KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("vkCode", wt.DWORD),
        ("scanCode", wt.DWORD),
        ("flags", wt.DWORD),
        ("time", wt.DWORD),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


class _POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class _MSLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [
        ("pt", _POINT),
        ("mouseData", wt.DWORD),
        ("flags", wt.DWORD),
        ("time", wt.DWORD),
        ("dwExtraInfo", ctypes.c_void_p),
    ]


HOOKPROC = ctypes.WINFUNCTYPE(  # type: ignore[attr-defined]
    ctypes.c_long, ctypes.c_int, wt.WPARAM, wt.LPARAM
) if os.name == "nt" else None


class InputHooks:
    """Manages WH_KEYBOARD_LL + WH_MOUSE_LL on a dedicated message-pump thread."""

    def __init__(
        self,
        *,
        capture_clicks: bool = True,
        capture_keystrokes: bool = True,
        capture_mouse_move: bool = False,
        debounce_seconds: float = 0.3,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.capture_clicks = capture_clicks
        self.capture_keystrokes = capture_keystrokes
        self.capture_mouse_move = capture_mouse_move
        self.debounce_seconds = debounce_seconds
        self._on_event = on_event
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._buf_lock = threading.Lock()
        self._buf: list[str] = []
        self._buf_started_at: float = 0.0
        self._flush_thread: threading.Thread | None = None
        self._last_move_emit = 0.0

    @property
    def available(self) -> bool:
        return os.name == "nt"

    def start(self) -> None:
        if not self.available or self._thread:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="InputHooks", daemon=True)
        self._thread.start()
        self._flush_thread = threading.Thread(target=self._flush_loop, name="InputHooks-flush", daemon=True)
        self._flush_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0); self._thread = None
        if self._flush_thread:
            self._flush_thread.join(timeout=1.0); self._flush_thread = None
        self._flush(reason="stop")

    # ─── pump ──────────────────────────────────────────────────────────────
    def _run(self) -> None:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

        mod = kernel32.GetModuleHandleW(None)
        kproc = HOOKPROC(self._kb_callback) if self.capture_keystrokes else None
        need_mouse = self.capture_clicks or self.capture_mouse_move or self._on_event is not None
        mproc = HOOKPROC(self._mouse_callback) if need_mouse else None
        self._kproc, self._mproc = kproc, mproc

        khook = user32.SetWindowsHookExW(WH_KEYBOARD_LL, kproc, mod, 0) if kproc else None
        mhook = user32.SetWindowsHookExW(WH_MOUSE_LL, mproc, mod, 0) if mproc else None

        msg = wt.MSG()
        while not self._stop.is_set():
            r = user32.PeekMessageW(ctypes.byref(msg), 0, 0, 0, 0x0001)
            if r:
                user32.TranslateMessage(ctypes.byref(msg))
                user32.DispatchMessageW(ctypes.byref(msg))
            else:
                self._stop.wait(0.02)

        if khook:
            user32.UnhookWindowsHookEx(khook)
        if mhook:
            user32.UnhookWindowsHookEx(mhook)

    def _kb_callback(self, ncode: int, wparam: int, lparam: int) -> int:
        try:
            if ncode == HC_ACTION and wparam in (WM_KEYDOWN, WM_SYSKEYDOWN):
                activity_default().record(ActivityKind.KEY_PRESS)
                kbd = ctypes.cast(lparam, ctypes.POINTER(_KBDLLHOOKSTRUCT)).contents
                vk = int(kbd.vkCode)
                hwnd, pid, title = _foreground_hwnd_pid()
                app = foreground_app_name(pid)
                if vk in _NAV_KEYS and self._on_event:
                    self._on_event({
                        "event_type": "key",
                        "key_code": vk,
                        "app_name": app,
                        "window_title": title,
                        "hwnd": hwnd,
                        "pid": pid,
                    })
                if vk in _MODIFIERS:
                    return ctypes.windll.user32.CallNextHookEx(0, ncode, wparam, lparam)  # type: ignore[attr-defined]
                char = self._vk_to_char(vk)
                if char:
                    with self._buf_lock:
                        if not self._buf:
                            self._buf_started_at = time.time()
                        self._buf.append(char)
        except Exception as exc:  # noqa: BLE001
            logger.debug("kb cb err: %s", exc)
        return ctypes.windll.user32.CallNextHookEx(0, ncode, wparam, lparam)  # type: ignore[attr-defined]

    def _mouse_callback(self, ncode: int, wparam: int, lparam: int) -> int:
        try:
            if ncode == HC_ACTION:
                ms = ctypes.cast(lparam, ctypes.POINTER(_MSLLHOOKSTRUCT)).contents
                hwnd, pid, title = _foreground_hwnd_pid()
                app = foreground_app_name(pid)
                button = self._wparam_to_button(int(wparam))
                if button:
                    activity_default().record(ActivityKind.MOUSE_CLICK)
                    self._flush(reason="click")
                    payload = {"button": button, "x": int(ms.pt.x), "y": int(ms.pt.y),
                               "app_name": app, "window_title": title, "hwnd": hwnd, "pid": pid}
                    bus.send(bus.EventType.CLICK, **payload)
                    if self._on_event:
                        self._on_event({"event_type": "click", **payload})
                elif int(wparam) == WM_MOUSEWHEEL:
                    activity_default().record(ActivityKind.SCROLL)
                    delta = ctypes.c_int16(ms.mouseData >> 16).value
                    payload = {
                        "event_type": "scroll",
                        "x": int(ms.pt.x),
                        "y": int(ms.pt.y),
                        "delta_x": 0,
                        "delta_y": int(delta),
                        "app_name": app,
                        "window_title": title,
                        "hwnd": hwnd,
                        "pid": pid,
                    }
                    if self._on_event:
                        self._on_event(payload)
                elif self.capture_mouse_move:
                    now = time.time()
                    if now - self._last_move_emit >= _MOVE_THROTTLE_S:
                        self._last_move_emit = now
                        activity_default().record(ActivityKind.MOUSE_MOVE)
                        if self._on_event:
                            self._on_event({
                                "event_type": "move",
                                "x": int(ms.pt.x),
                                "y": int(ms.pt.y),
                                "app_name": app,
                                "window_title": title,
                                "hwnd": hwnd,
                                "pid": pid,
                            })
                else:
                    activity_default().record(ActivityKind.MOUSE_MOVE)
        except Exception as exc:  # noqa: BLE001
            logger.debug("mouse cb err: %s", exc)
        return ctypes.windll.user32.CallNextHookEx(0, ncode, wparam, lparam)  # type: ignore[attr-defined]

    @staticmethod
    def _wparam_to_button(wparam: int) -> str:
        return {
            WM_LBUTTONDOWN: "left",
            WM_RBUTTONDOWN: "right",
            WM_MBUTTONDOWN: "other",
            WM_XBUTTONDOWN: "other",
        }.get(wparam, "")

    @staticmethod
    def _vk_to_char(vk: int) -> str:
        if vk == VK_RETURN:
            return "\n"
        if vk == VK_BACK:
            return "\b"
        if vk == VK_SPACE:
            return " "
        try:
            user32 = ctypes.windll.user32  # type: ignore[attr-defined]
            state = (ctypes.c_byte * 256)()
            user32.GetKeyboardState(ctypes.byref(state))
            buf = ctypes.create_unicode_buffer(8)
            n = user32.ToUnicode(vk, 0, ctypes.byref(state), buf, 8, 0)
            if n > 0:
                return buf.value
        except Exception:  # noqa: BLE001
            pass
        if 0x30 <= vk <= 0x5A:
            return chr(vk).lower()
        return ""

    # ─── flush ─────────────────────────────────────────────────────────────
    def _flush_loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(0.5)
            with self._buf_lock:
                age = time.time() - self._buf_started_at if self._buf else 0
            if age and age >= self.debounce_seconds:
                self._flush(reason="debounce")

    def _flush(self, *, reason: str) -> None:
        with self._buf_lock:
            if not self._buf:
                return
            text = "".join(self._buf)
            self._buf.clear()
            self._buf_started_at = 0.0
        hwnd, pid, title = _foreground_hwnd_pid()
        app = foreground_app_name(pid)
        payload = {"text": text, "reason": reason, "app_name": app, "window_title": title, "hwnd": hwnd, "pid": pid}
        bus.send(bus.EventType.KEY_TEXT, **payload)
        if self._on_event:
            self._on_event({"event_type": "text", **payload})
