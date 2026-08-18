"""Detect on-screen change so capture can fire when only pixels move.

Idle/heartbeat capture dedups on (app, title): a fullscreen video keeps both
constant while the picture changes every frame, so those captures are skipped
and the playback never reaches the timeline. This probes a tiny grayscale
thumbnail on a fast timer and reports when it differs enough to be worth a full
capture — cheap enough to run every second, and it stays quiet on a static
screen.
"""

from __future__ import annotations

from ..logger import get
from ..screen.capture import grab_monitor, list_monitors

logger = get("capture.visual_change")

_THUMB = 16


def diff_ratio(prev: bytes, sig: bytes) -> float:
    """Mean per-pixel intensity change in 0..1, or 0 when not comparable."""
    if not prev or not sig or len(prev) != len(sig):
        return 0.0
    total = sum(abs(a - b) for a, b in zip(prev, sig, strict=False))
    return total / (len(sig) * 255)


class VisualChangeProbe:
    """Remembers the last thumbnail and flags a meaningful change from it."""

    def __init__(self, *, threshold: float = 0.06, thumb: int = _THUMB) -> None:
        self.threshold = threshold
        self.thumb = thumb
        self._last: bytes | None = None
        self._monitors = list_monitors()

    def _capture_signature(self) -> bytes | None:
        mons = self._monitors or list_monitors()
        if not mons:
            return None
        img = grab_monitor(mons[0])
        if img is None:
            return None
        return img.convert("L").resize((self.thumb, self.thumb)).tobytes()

    def evaluate(self, sig: bytes | None) -> bool:
        """Update state and return whether ``sig`` changed past the threshold.

        The first signature only seeds the baseline (never fires) so entering a
        static screen doesn't trigger a redundant capture.
        """
        if sig is None:
            return False
        prev = self._last
        self._last = sig
        if prev is None:
            return False
        return diff_ratio(prev, sig) >= self.threshold

    def changed(self) -> bool:
        try:
            return self.evaluate(self._capture_signature())
        except Exception as exc:  # noqa: BLE001 — probing must never break capture
            logger.debug("visual probe failed: %s", exc)
            return False
