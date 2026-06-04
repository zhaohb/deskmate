"""Low-level mouse + keyboard hooks.

We do **not** record individual keystrokes. Instead, on Enter / Ctrl+Enter
(the common AI-chat "send" shortcuts) we take a single snapshot of the focused
input box via UIA and emit it as one `text` event with ``source="send"``. This
captures the complete, already-composed prompt (including IME / pasted / voice
input) without sniffing every character, and trips a one-shot screenshot
capture at the send moment. Mouse clicks emit immediately.

Shift+Enter inserts a newline and is therefore ignored (not a send).
"""

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

VK_TAB = 0x09
VK_RETURN = 0x0D
VK_SHIFT = 0x10
VK_CONTROL = 0x11
VK_MENU = 0x12
VK_CAPITAL = 0x14
VK_ESCAPE = 0x1B
VK_PRIOR = 0x21
VK_DELETE = 0x2E
VK_LWIN = 0x5B
VK_RWIN = 0x5C
VK_F1 = 0x70
VK_F24 = 0x87

_NAV_KEYS = set(range(VK_PRIOR, VK_DELETE + 1)) | {VK_ESCAPE, VK_CAPITAL, VK_TAB} | set(range(VK_F1, VK_F24 + 1)) | {VK_LWIN, VK_RWIN}
_MODIFIERS = {VK_SHIFT, VK_CONTROL, VK_MENU}


def return_flush_reason(*, ctrl_down: bool, shift_down: bool) -> str | None:
    """Flush reason for Return, or None when only inserting a newline (Shift+Enter)."""
    if shift_down:
        return None
    if ctrl_down:
        return "ctrl_enter"
    return "enter"


def return_inserts_newline(*, shift_down: bool) -> bool:
    return shift_down


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


def _modifier_down(vk: int) -> bool:
    if os.name != "nt":
        return False
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return False


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
        self.debounce_seconds = debounce_seconds  # retained for API compat; unused
        self._on_event = on_event
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_move_emit = 0.0
        # Serializes the off-thread UIA focused-value reads triggered on Enter so
        # rapid sends don't pile up overlapping reads. Set while a read is queued.
        self._send_busy = threading.Event()

    @property
    def available(self) -> bool:
        return os.name == "nt"

    def start(self) -> None:
        if not self.available or self._thread:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="InputHooks", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ─── pump ──────────────────────────────────────────────────────────────
    def _run(self) -> None:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]

        # Pin correct 64-bit signatures. Without these, ctypes defaults the
        # return type to a 32-bit C int, truncating HMODULE/HHOOK handles on
        # 64-bit Windows: GetModuleHandleW's handle is mangled and
        # SetWindowsHookExW then fails with ERROR_MOD_NOT_FOUND (126), so the
        # keyboard/mouse hooks never install and no text/key/click events are
        # ever recorded.
        kernel32.GetModuleHandleW.restype = wt.HMODULE
        kernel32.GetModuleHandleW.argtypes = [wt.LPCWSTR]
        user32.SetWindowsHookExW.restype = ctypes.c_void_p
        user32.SetWindowsHookExW.argtypes = [ctypes.c_int, ctypes.c_void_p, wt.HMODULE, wt.DWORD]
        user32.UnhookWindowsHookEx.restype = wt.BOOL
        user32.UnhookWindowsHookEx.argtypes = [ctypes.c_void_p]
        # CallNextHookEx's wParam/lParam are pointer-sized (LRESULT/WPARAM/LPARAM).
        # Without pinned argtypes ctypes defaults each argument to a 32-bit C int,
        # so on 64-bit Windows the lParam (arg 4) — the address of the hook struct —
        # overflows with "argument 4: OverflowError: int too long to convert".
        user32.CallNextHookEx.restype = wt.LPARAM
        user32.CallNextHookEx.argtypes = [ctypes.c_void_p, ctypes.c_int, wt.WPARAM, wt.LPARAM]

        mod = kernel32.GetModuleHandleW(None)
        kproc = HOOKPROC(self._kb_callback) if self.capture_keystrokes else None
        need_mouse = self.capture_clicks or self.capture_mouse_move or self._on_event is not None
        mproc = HOOKPROC(self._mouse_callback) if need_mouse else None
        self._kproc, self._mproc = kproc, mproc

        khook = user32.SetWindowsHookExW(WH_KEYBOARD_LL, kproc, mod, 0) if kproc else None
        mhook = user32.SetWindowsHookExW(WH_MOUSE_LL, mproc, mod, 0) if mproc else None
        if kproc and not khook:
            logger.error(
                "keyboard hook install failed (GetLastError=%s); no key/text events will be captured",
                kernel32.GetLastError(),
            )
        if mproc and not mhook:
            logger.error(
                "mouse hook install failed (GetLastError=%s); no click events will be captured",
                kernel32.GetLastError(),
            )
        if khook or mhook:
            logger.info("InputHooks installed (keyboard=%s mouse=%s)", bool(khook), bool(mhook))

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

                # We do not record individual characters. Only a "send" Enter
                # (Enter / Ctrl+Enter, but not Shift+Enter) triggers a one-shot
                # snapshot of the focused input box. Shift+Enter inserts a
                # newline and is intentionally ignored.
                if vk == VK_RETURN and self.capture_keystrokes:
                    ctrl_down = _modifier_down(VK_CONTROL)
                    shift_down = _modifier_down(VK_SHIFT)
                    reason = return_flush_reason(ctrl_down=ctrl_down, shift_down=shift_down)
                    if reason:
                        self._schedule_send_capture(reason=reason)
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
                    payload = {
                        "button": button,
                        "x": int(ms.pt.x),
                        "y": int(ms.pt.y),
                        "app_name": app,
                        "window_title": title,
                        "hwnd": hwnd,
                        "pid": pid,
                    }
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

    # ─── send capture ──────────────────────────────────────────────────────
    def _schedule_send_capture(self, *, reason: str) -> None:
        """Read the focused input box off-thread after a send keypress.

        UIA reads can block, so they must never run on the low-level hook
        callback. A single in-flight read is allowed at a time; rapid repeated
        sends while one is pending are dropped (the pending read already covers
        the latest input-box contents).
        """
        if not self._on_event or self._send_busy.is_set():
            return
        self._send_busy.set()
        hwnd, pid, title = _foreground_hwnd_pid()
        app = foreground_app_name(pid)
        threading.Thread(
            target=self._emit_focused_value,
            kwargs={"app": app, "title": title, "hwnd": hwnd, "pid": pid, "reason": reason},
            name="InputHooks-send",
            daemon=True,
        ).start()

    def _emit_focused_value(
        self, *, app: str, title: str, hwnd: int, pid: int, reason: str
    ) -> None:
        """Read the full text of the focused input box via UIA and emit it as a
        single ``text`` event with ``source="send"``. Runs on a short-lived
        worker thread — never the low-level hook callback — because UIA calls can
        block. Captures the complete, already-composed prompt (IME, pasted, or
        voice input) at the moment the user pressed Enter, and trips a one-shot
        screenshot capture via the SEND trigger.
        """
        try:
            if not self._on_event:
                return
            try:
                from .uia_tree import read_focused_value  # noqa: PLC0415

                role, value = read_focused_value()
            except Exception as exc:  # noqa: BLE001
                logger.debug("focused-value read failed: %s", exc)
                return
            value = (value or "").strip()
            if len(value) < 5:
                return
            payload = {
                "text": value,
                "reason": reason,
                "app_name": app,
                "window_title": title,
                "hwnd": hwnd,
                "pid": pid,
            }
            bus.send(bus.EventType.KEY_TEXT, **payload)
            self._on_event({
                "event_type": "text",
                "source": "send",
                "focused_role": role,
                **payload,
            })
        finally:
            self._send_busy.clear()
