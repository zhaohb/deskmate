"""UI event types and capture-trigger mapping."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

CorrelationId = int


class UiEventType(str, Enum):
    CLICK = "click"
    MOVE = "move"
    SCROLL = "scroll"
    KEY = "key"
    TEXT = "text"
    APP_SWITCH = "app_switch"
    WINDOW_FOCUS = "window_focus"
    CLIPBOARD = "clipboard"
    # Legacy alias from older deskmate builds
    TITLE_CHANGE = "title_change"
    VALUE_CHANGE = "value_change"


class CaptureTrigger(str, Enum):
    APP_SWITCH = "app_switch"
    WINDOW_FOCUS = "window_focus"
    CLICK = "click"
    TYPING_PAUSE = "typing_pause"
    SCROLL_STOP = "scroll_stop"
    KEY_PRESS = "key_press"
    CLIPBOARD = "clipboard"
    IDLE = "idle"
    MANUAL = "manual"

    def as_capture_trigger(self) -> str:
        return self.value


@dataclass
class TriggerGates:
    capture_on_keystroke: bool = False
    capture_on_clipboard: bool = False


@dataclass
class UiEventInsert:
    """Row payload for `ui_events` — maps to data_json + top-level columns."""

    event_type: UiEventType
    app_name: str | None = None
    window_title: str | None = None
    browser_url: str | None = None
    app_pid: int | None = None
    hwnd: int | None = None
    x: int | None = None
    y: int | None = None
    delta_x: int | None = None
    delta_y: int | None = None
    button: str | None = None
    click_count: int = 1
    key_code: int | None = None
    text_content: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_db_row(self) -> tuple[str, str | None, str | None, str | None, str]:
        data: dict[str, Any] = dict(self.extra)
        if self.app_pid is not None:
            data["pid"] = self.app_pid
        if self.hwnd is not None:
            data["hwnd"] = self.hwnd
        if self.x is not None:
            data["x"] = self.x
        if self.y is not None:
            data["y"] = self.y
        if self.delta_x is not None:
            data["delta_x"] = self.delta_x
        if self.delta_y is not None:
            data["delta_y"] = self.delta_y
        if self.button is not None:
            data["button"] = self.button
            data["click_count"] = self.click_count
        if self.key_code is not None:
            data["key_code"] = self.key_code
        if self.text_content is not None:
            data["content"] = self.text_content
            data["char_count"] = len(self.text_content)
        return (
            self.event_type.value,
            self.app_name,
            self.window_title,
            self.browser_url,
            __import__("json").dumps(data, ensure_ascii=False),
        )


def _is_ignored(app_name: str | None, window_title: str | None, ignored_patterns: list) -> bool:
    from ..core.filter import WindowFilter  # noqa: PLC0415

    app = (app_name or "").lower()
    title = (window_title or "").lower()
    if not app and not title:
        return False
    wf = WindowFilter(ignored_windows=ignored_patterns)
    return not wf.passes(app_name or "unknown", window_title or "")


def capture_trigger_kind(
    event: UiEventInsert,
    *,
    ignored_patterns: list[str],
    gates: TriggerGates,
) -> CaptureTrigger | None:
    """Map a UI event to a capture trigger kind, if any."""
    if _is_ignored(event.app_name, event.window_title, ignored_patterns):
        if event.event_type in (UiEventType.APP_SWITCH, UiEventType.WINDOW_FOCUS):
            return None
    app = event.app_name or ""
    title = event.window_title or ""
    match event.event_type:
        case UiEventType.APP_SWITCH:
            return CaptureTrigger.APP_SWITCH
        case UiEventType.WINDOW_FOCUS:
            return CaptureTrigger.WINDOW_FOCUS
        case UiEventType.CLICK:
            return CaptureTrigger.CLICK
        case UiEventType.CLIPBOARD if gates.capture_on_clipboard:
            return CaptureTrigger.CLIPBOARD
        case UiEventType.CLIPBOARD:
            return None
        case UiEventType.TEXT:
            return CaptureTrigger.TYPING_PAUSE
        case UiEventType.SCROLL:
            return None
        case UiEventType.KEY if gates.capture_on_keystroke:
            return CaptureTrigger.KEY_PRESS
        case UiEventType.KEY:
            return None
        case UiEventType.TITLE_CHANGE:
            return CaptureTrigger.WINDOW_FOCUS
        case UiEventType.VALUE_CHANGE:
            return None
        case _:
            return None


class ScrollBurstTracker:
    """Link last scroll row to ScrollStop frame after burst ends."""

    def __init__(self, delay_s: float = 0.3) -> None:
        self.delay_s = delay_s
        self._last_at: float | None = None
        self._last_corr: CorrelationId | None = None

    def record(self, corr_id: CorrelationId) -> None:
        self._last_at = time.monotonic()
        self._last_corr = corr_id

    def poll_burst_end(self) -> CorrelationId | None:
        if self._last_at is None or self._last_corr is None:
            return None
        if time.monotonic() - self._last_at < self.delay_s:
            return None
        corr = self._last_corr
        self._last_at = None
        self._last_corr = None
        return corr


@dataclass
class CaptureTriggerMsg:
    trigger: CaptureTrigger
    correlation_id: CorrelationId | None = None

    @classmethod
    def with_correlation(cls, trigger: CaptureTrigger, corr_id: CorrelationId) -> CaptureTriggerMsg:
        return cls(trigger=trigger, correlation_id=corr_id)


def raw_dict_to_insert(ev: dict[str, Any], *, browser_url: str | None = None) -> UiEventInsert:
    """Normalize legacy a11y callback payloads to `UiEventInsert`."""
    raw_type = str(ev.get("event_type") or "event")
    if raw_type == "key_text":
        et = UiEventType.TEXT
    elif raw_type == "title_change":
        et = UiEventType.TITLE_CHANGE
    elif raw_type == "value_change":
        et = UiEventType.VALUE_CHANGE
    else:
        try:
            et = UiEventType(raw_type)
        except ValueError:
            et = UiEventType.CLICK
    text = ev.get("text") or ev.get("content")
    if et == UiEventType.TEXT and text:
        return UiEventInsert(
            event_type=et,
            app_name=ev.get("app_name"),
            window_title=ev.get("window_title"),
            browser_url=browser_url,
            app_pid=int(ev["pid"]) if ev.get("pid") else None,
            hwnd=int(ev["hwnd"]) if ev.get("hwnd") else None,
            text_content=str(text),
            extra={k: v for k, v in ev.items() if k not in {"event_type", "text", "content"}},
        )
    if et == UiEventType.CLIPBOARD:
        return UiEventInsert(
            event_type=et,
            app_name=ev.get("app_name"),
            window_title=ev.get("window_title"),
            browser_url=browser_url,
            app_pid=int(ev["pid"]) if ev.get("pid") else None,
            hwnd=int(ev["hwnd"]) if ev.get("hwnd") else None,
            text_content=str(text or ""),
            extra={"operation": ev.get("operation", "c"), "length": ev.get("length", len(text or ""))},
        )
    if et == UiEventType.CLICK:
        return UiEventInsert(
            event_type=et,
            app_name=ev.get("app_name"),
            window_title=ev.get("window_title"),
            browser_url=browser_url,
            app_pid=int(ev["pid"]) if ev.get("pid") else None,
            hwnd=int(ev["hwnd"]) if ev.get("hwnd") else None,
            x=int(ev["x"]) if ev.get("x") is not None else None,
            y=int(ev["y"]) if ev.get("y") is not None else None,
            button=str(ev.get("button") or "left"),
            click_count=int(ev.get("click_count") or 1),
        )
    if et == UiEventType.SCROLL:
        return UiEventInsert(
            event_type=et,
            app_name=ev.get("app_name"),
            window_title=ev.get("window_title"),
            browser_url=browser_url,
            x=int(ev.get("x") or 0),
            y=int(ev.get("y") or 0),
            delta_x=int(ev.get("delta_x") or 0),
            delta_y=int(ev.get("delta_y") or 0),
        )
    if et == UiEventType.KEY:
        return UiEventInsert(
            event_type=et,
            app_name=ev.get("app_name"),
            window_title=ev.get("window_title"),
            browser_url=browser_url,
            key_code=int(ev.get("key_code") or 0),
        )
    if et == UiEventType.MOVE:
        return UiEventInsert(
            event_type=et,
            app_name=ev.get("app_name"),
            window_title=ev.get("window_title"),
            browser_url=browser_url,
            x=int(ev.get("x") or 0),
            y=int(ev.get("y") or 0),
        )
    return UiEventInsert(
        event_type=et,
        app_name=ev.get("app_name"),
        window_title=ev.get("window_title"),
        browser_url=browser_url,
        app_pid=int(ev["pid"]) if ev.get("pid") else None,
        hwnd=int(ev["hwnd"]) if ev.get("hwnd") else None,
        extra={k: v for k, v in ev.items() if k not in {"event_type", "app_name", "window_title"}},
    )
