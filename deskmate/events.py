"""In-process event bus.

Used by capture / a11y / audio threads to publish typed events that the engine
or external subscribers (HTTP /events stream) consume. Pure stdlib, thread-safe.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from queue import Empty, Queue
from typing import Any


class EventType(str, Enum):
    PERMISSION = "permission"
    WINDOW_FOCUS = "window_focus"
    TITLE_CHANGE = "title_change"
    VALUE_CHANGE = "value_change"
    CLICK = "click"
    KEY_TEXT = "key_text"
    CLIPBOARD = "clipboard"
    FRAME_WRITTEN = "frame_written"
    A11Y_WRITTEN = "a11y_written"
    AUDIO_TRANSCRIBED = "audio_transcribed"
    TRANSCRIPT_TRANSLATED = "transcript_translated"
    WORKFLOW = "workflow"
    MEETING_STARTED = "meeting_started"
    MEETING_ENDED = "meeting_ended"


class PermissionKind(str, Enum):
    ACCESSIBILITY = "accessibility"
    INPUT_MONITORING = "input_monitoring"
    SCREEN_RECORDING = "screen_recording"
    AUDIO = "audio"


@dataclass
class Event:
    type: EventType
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)


_lock = threading.RLock()
_subscribers: list[Callable[[Event], None]] = []
_history_lock = threading.Lock()
_history: list[Event] = []
_HISTORY_MAX = 2048


def subscribe(callback: Callable[[Event], None]) -> Callable[[], None]:
    with _lock:
        _subscribers.append(callback)

    def _unsub() -> None:
        with _lock:
            try:
                _subscribers.remove(callback)
            except ValueError:
                pass

    return _unsub


def emit(event: Event) -> None:
    with _history_lock:
        _history.append(event)
        if len(_history) > _HISTORY_MAX:
            del _history[: len(_history) - _HISTORY_MAX]
    with _lock:
        listeners = list(_subscribers)
    for cb in listeners:
        try:
            cb(event)
        except Exception:  # noqa: BLE001
            pass


def send(event_type: EventType, **data: Any) -> Event:
    e = Event(type=event_type, data=data)
    emit(e)
    return e


def recent(limit: int = 100) -> list[Event]:
    with _history_lock:
        return list(_history[-limit:])


def stream(timeout: float | None = None) -> EventStream:
    return EventStream(timeout=timeout)


class EventStream:
    """Blocking iterator over future events. Used by HTTP /events SSE."""

    def __init__(self, timeout: float | None = None) -> None:
        self._q: Queue[Event] = Queue()
        self._timeout = timeout
        self._unsub = subscribe(self._q.put)

    def __iter__(self) -> EventStream:
        return self

    def __next__(self) -> Event:
        try:
            return self._q.get(timeout=self._timeout)
        except Empty as exc:
            raise StopIteration from exc

    def close(self) -> None:
        self._unsub()
