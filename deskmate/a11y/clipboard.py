"""Clipboard watcher for text capture.

Implementation: poll `GetClipboardSequenceNumber` cheaply; only opens the
clipboard when the sequence number changes. Skips bursts we made ourselves
by ignoring identical text content."""

from __future__ import annotations

import ctypes
import os
import threading
import time
from collections.abc import Callable
from typing import Any

from .. import events as bus
from ..logger import get
from .win_events import _foreground_hwnd_pid, foreground_app_name

logger = get("a11y.clipboard")


class ClipboardWatcher:
    def __init__(
        self,
        *,
        poll_seconds: float = 1.0,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.poll = poll_seconds
        self._on_event = on_event
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_seq = 0
        self._last_text: str | None = None

    @property
    def available(self) -> bool:
        return os.name == "nt"

    def start(self) -> None:
        if not self.available or self._thread:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="ClipboardWatcher", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]
        try:
            import win32clipboard  # noqa: PLC0415
        except ImportError:
            logger.warning("pywin32 not installed; clipboard watcher disabled")
            return

        while not self._stop.is_set():
            try:
                seq = int(user32.GetClipboardSequenceNumber())
                if seq != self._last_seq:
                    self._last_seq = seq
                    text = ""
                    try:
                        win32clipboard.OpenClipboard()
                        if win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_UNICODETEXT):
                            text = win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT) or ""
                    except Exception:  # noqa: BLE001
                        text = ""
                    finally:
                        try:
                            win32clipboard.CloseClipboard()
                        except Exception:  # noqa: BLE001
                            pass
                    text = (text or "").strip()
                    if text and text != self._last_text:
                        self._last_text = text
                        hwnd, pid, title = _foreground_hwnd_pid()
                        app = foreground_app_name(pid)
                        bus.send(
                            bus.EventType.CLIPBOARD,
                            text=text[:4000],
                            length=len(text),
                            app_name=app, window_title=title, hwnd=hwnd, pid=pid,
                        )
                        if self._on_event:
                            self._on_event({
                                "event_type": "clipboard",
                                "text": text[:4000],
                                "length": len(text),
                                "app_name": app,
                                "window_title": title,
                                "hwnd": hwnd,
                                "pid": pid,
                            })
            except Exception as exc:  # noqa: BLE001
                logger.debug("clipboard poll err: %s", exc)
            time.sleep(self.poll)
