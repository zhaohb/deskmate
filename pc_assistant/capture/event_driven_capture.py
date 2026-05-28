"""Event-driven capture loop (single monitor)."""

from __future__ import annotations

import queue
import threading
import time
from typing import TYPE_CHECKING, Any

from ..a11y.activity_feed import default as activity_default
from ..a11y.ui_event_types import CaptureTrigger, CaptureTriggerMsg
from ..capture.frame_linker import DropReason, FrameCaptured, LinkerMessage
from ..logger import get

if TYPE_CHECKING:
    from ..config import Config
    from ..capture.paired import PairedCapture
    from ..capture.frame_linker import FrameLinkerActor
    from ..db import DatabaseManager

logger = get("capture.event_driven")


class EventDrivenCapture:
    """Rate limiting and idle-capture state for event-driven mode."""

    def __init__(self, cfg: Config) -> None:
        self.min_interval_s = cfg.capture.min_capture_interval_ms / 1000.0
        self.idle_interval_s = cfg.capture.idle_capture_interval_ms / 1000.0
        self.capture_on_keystroke = cfg.capture.capture_on_keystroke
        self.capture_on_clipboard = cfg.capture.capture_on_clipboard
        self._last_capture = time.monotonic()

    def can_capture(self) -> bool:
        return time.monotonic() - self._last_capture >= self.min_interval_s

    def mark_captured(self) -> None:
        self._last_capture = time.monotonic()

    def needs_idle_capture(self) -> bool:
        return time.monotonic() - self._last_capture >= self.idle_interval_s

    def poll_activity(self) -> CaptureTrigger | None:
        if self.needs_idle_capture():
            return CaptureTrigger.IDLE
        return None


def _trigger_allowed(trigger: CaptureTrigger, state: EventDrivenCapture) -> bool:
    if trigger == CaptureTrigger.CLIPBOARD and not state.capture_on_clipboard:
        return False
    if trigger == CaptureTrigger.KEY_PRESS and not state.capture_on_keystroke:
        return False
    return True


def _reduce_drained(drained: list[CaptureTriggerMsg], state: EventDrivenCapture) -> tuple[CaptureTrigger | None, list[int]]:
    """Last trigger kind wins; collect all correlation ids that pass gates."""
    if not drained:
        return None, []
    corr_ids: list[int] = []
    last_trigger: CaptureTrigger | None = None
    for msg in drained:
        if msg.correlation_id is not None and _trigger_allowed(msg.trigger, state):
            corr_ids.append(msg.correlation_id)
        if _trigger_allowed(msg.trigger, state):
            last_trigger = msg.trigger
    return last_trigger, corr_ids


def run_event_driven_capture_loop(
    *,
    cfg: Config,
    db: DatabaseManager,
    paired: PairedCapture,
    trigger_rx: queue.Queue[CaptureTriggerMsg],
    linker: FrameLinkerActor,
    stop: threading.Event,
    meeting_observe: Any | None = None,
) -> None:
    """Blocking loop — run on daemon thread."""
    state = EventDrivenCapture(cfg)
    poll_interval_s = 0.25

    # Startup capture seeds the timeline immediately.
    if state.can_capture():
        frame_ids = paired.capture_once(trigger=CaptureTrigger.MANUAL.value)
        state.mark_captured()
        if frame_ids:
            linker.try_send(
                LinkerMessage(frame_captured=FrameCaptured(frame_id=frame_ids[0], correlation_ids=[])),
            )
        if meeting_observe and frame_ids:
            _observe_frames(db, meeting_observe, frame_ids[0])

    while not stop.is_set():
        drained: list[CaptureTriggerMsg] = []
        try:
            first = trigger_rx.get(timeout=poll_interval_s)
            drained.append(first)
        except queue.Empty:
            pass

        while True:
            try:
                drained.append(trigger_rx.get_nowait())
            except queue.Empty:
                break

        trigger: CaptureTrigger | None
        corr_ids: list[int]
        if drained:
            trigger, corr_ids = _reduce_drained(drained, state)
        else:
            trigger = state.poll_activity()
            corr_ids = []

        if trigger is None:
            continue
        if not state.can_capture():
            if corr_ids:
                linker.try_send(
                    LinkerMessage(trigger_dropped=(corr_ids, DropReason.OTHER)),
                )
            continue

        frame_ids = paired.capture_once(trigger=trigger.value)
        state.mark_captured()
        if frame_ids:
            linker.try_send(
                LinkerMessage(
                    frame_captured=FrameCaptured(frame_id=frame_ids[0], correlation_ids=corr_ids),
                ),
            )
            if meeting_observe:
                _observe_frames(db, meeting_observe, frame_ids[0])
        elif corr_ids:
            linker.try_send(
                LinkerMessage(trigger_dropped=(corr_ids, DropReason.CAPTURE_ERROR)),
            )

        # Activity feed still drives adaptive consumers; idle handled above.
        activity_default()


def _observe_frames(db: DatabaseManager, observe_fn: Any, frame_id: int) -> None:
    try:
        row = db.frame_by_id(frame_id)
        if not row:
            return
        observe_fn(
            app_name=row.get("app_name") or "",
            window_title=row.get("window_name") or "",
            browser_url=row.get("browser_url"),
            text="\n".join(filter(None, [row.get("accessibility_text"), row.get("ocr_text")])),
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("meeting observe after capture failed: %s", exc)
