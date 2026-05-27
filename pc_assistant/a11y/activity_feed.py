"""Lightweight activity feed used by adaptive-FPS consumers. Tracks
time-since-last-input and exposes `get_capture_params()` returning the
recommended capture interval and frame-skip threshold.
"""

from __future__ import annotations

import enum
import threading
import time
from dataclasses import dataclass


def _now_ms() -> int:
    return int(time.time() * 1000)


class ActivityKind(str, enum.Enum):
    KEY_PRESS = "key_press"
    KEY_RELEASE = "key_release"
    MOUSE_CLICK = "mouse_click"
    MOUSE_MOVE = "mouse_move"
    SCROLL = "scroll"


@dataclass
class CaptureParams:
    """Recommended capture parameters based on activity level."""

    interval_ms: int = 1000
    skip_threshold: float = 0.02  # 0.0–1.0


class ActivityFeed:
    """Thread-safe activity feed.

    Atomic-ish reads/writes via a single lock — we are nowhere near the
    contention level where the lock would matter, and a single lock keeps the
    semantics dead-simple compared to Rust's AtomicU64.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        now = _now_ms()
        self._last_activity_ms = now
        self._last_keyboard_ms = 0
        self._keyboard_count = 0
        self._last_count_reset_ms = now

    def record(self, kind: ActivityKind) -> None:
        now = _now_ms()
        with self._lock:
            self._last_activity_ms = now
            if now - self._last_count_reset_ms > 500:
                self._keyboard_count = 0
                self._last_count_reset_ms = now
            if kind == ActivityKind.KEY_PRESS:
                self._last_keyboard_ms = now
                self._keyboard_count += 1

    def idle_ms(self) -> int:
        with self._lock:
            return max(0, _now_ms() - self._last_activity_ms)

    def keyboard_idle_ms(self) -> int:
        with self._lock:
            if self._last_keyboard_ms == 0:
                return 2**63 - 1  # effectively "no keyboard activity yet"
            return max(0, _now_ms() - self._last_keyboard_ms)

    def is_typing(self) -> bool:
        return self.keyboard_idle_ms() < 300

    def is_keyboard_burst(self) -> bool:
        with self._lock:
            kb_idle = (
                2**63 - 1
                if self._last_keyboard_ms == 0
                else max(0, _now_ms() - self._last_keyboard_ms)
            )
            return kb_idle < 500 and self._keyboard_count >= 3

    def is_active(self, threshold_ms: int) -> bool:
        return self.idle_ms() < threshold_ms

    def get_capture_params(self) -> CaptureParams:
        """Adaptive FPS curve for keyboard/mouse activity."""
        idle = self.idle_ms()
        kb_idle = self.keyboard_idle_ms()

        if self.is_keyboard_burst():
            return CaptureParams(interval_ms=200, skip_threshold=0.02)  # 5 FPS, 2%
        if kb_idle < 300:
            return CaptureParams(interval_ms=150, skip_threshold=0.01)  # ~7 FPS, 1%
        if idle < 500:
            return CaptureParams(interval_ms=200, skip_threshold=0.02)  # 5 FPS
        if idle < 2000:
            return CaptureParams(interval_ms=500, skip_threshold=0.02)  # 2 FPS
        if idle < 5000:
            return CaptureParams(interval_ms=1000, skip_threshold=0.02)  # 1 FPS
        return CaptureParams(interval_ms=2000, skip_threshold=0.02)  # 0.5 FPS


# Module-level singleton — input hooks record into it; capture loop reads it.
_default = ActivityFeed()


def default() -> ActivityFeed:
    return _default
