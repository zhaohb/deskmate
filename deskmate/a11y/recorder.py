"""Orchestrates the three watchers (WinEvent + Input + Clipboard) under one
handle."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..config import A11yConfig
from .clipboard import ClipboardWatcher
from .input_hooks import InputHooks
from .win_events import WinEventWatcher


class UiRecorder:
    def __init__(self, cfg: A11yConfig, on_event: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.cfg = cfg
        self._on_event = on_event
        self.win_events = WinEventWatcher(on_event=on_event)
        self.input = InputHooks(
            capture_clicks=cfg.capture_clicks,
            capture_keystrokes=cfg.capture_keystrokes,
            capture_mouse_move=cfg.capture_mouse_move,
            debounce_seconds=cfg.text_input_debounce_seconds,
            on_event=on_event,
        )
        self.clipboard = ClipboardWatcher(on_event=on_event) if cfg.capture_clipboard else None

    def start(self) -> None:
        if not self.cfg.enabled:
            return
        self.win_events.start()
        self.input.start()
        if self.clipboard:
            self.clipboard.start()

    def stop(self) -> None:
        self.win_events.stop()
        self.input.stop()
        if self.clipboard:
            self.clipboard.stop()
