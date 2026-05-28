"""Frame linker — pairs ui_events rows with frames they triggered."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from ..a11y.ui_event_types import CorrelationId
from ..logger import get

if TYPE_CHECKING:
    from ..db import DatabaseManager

logger = get("capture.frame_linker")


@dataclass(frozen=True)
class EventPersisted:
    correlation_id: CorrelationId
    row_id: int


@dataclass(frozen=True)
class FrameCaptured:
    frame_id: int
    correlation_ids: list[CorrelationId]


@dataclass(frozen=True)
class LinkUpdate:
    row_id: int
    frame_id: int


class DropReason(Enum):
    DRM = "drm"
    PAUSED = "paused"
    LAGGED = "lagged"
    CAPTURE_ERROR = "capture_error"
    OTHER = "other"


@dataclass
class FrameLinkerConfig:
    ttl_s: float = 60.0
    capacity: int = 4096


@dataclass
class _PendingEvent:
    row_id: int
    inserted_at: float


@dataclass
class _PendingFrame:
    frame_id: int
    unmatched: list[CorrelationId]
    inserted_at: float


class FrameLinker:
    """Pure state machine — no I/O."""

    def __init__(self, config: FrameLinkerConfig | None = None) -> None:
        self.config = config or FrameLinkerConfig()
        self._pending_events: dict[CorrelationId, _PendingEvent] = {}
        self._pending_frames: list[_PendingFrame] = []

    def on_event_persisted(self, e: EventPersisted, now: float | None = None) -> LinkUpdate | None:
        now = now if now is not None else time.monotonic()
        for pf in self._pending_frames:
            if e.correlation_id in pf.unmatched:
                pf.unmatched = [c for c in pf.unmatched if c != e.correlation_id]
                frame_id = pf.frame_id
                self._compact_pending_frames()
                return LinkUpdate(row_id=e.row_id, frame_id=frame_id)
        self._evict_if_full_events(now)
        self._pending_events[e.correlation_id] = _PendingEvent(row_id=e.row_id, inserted_at=now)
        return None

    def on_frame_captured(self, c: FrameCaptured, now: float | None = None) -> list[LinkUpdate]:
        now = now if now is not None else time.monotonic()
        updates: list[LinkUpdate] = []
        unmatched: list[CorrelationId] = []
        for corr_id in c.correlation_ids:
            pe = self._pending_events.pop(corr_id, None)
            if pe is not None:
                updates.append(LinkUpdate(row_id=pe.row_id, frame_id=c.frame_id))
            else:
                unmatched.append(corr_id)
        if unmatched:
            self._evict_if_full_frames(now)
            self._pending_frames.append(
                _PendingFrame(frame_id=c.frame_id, unmatched=unmatched, inserted_at=now)
            )
        return updates

    def on_trigger_dropped(self, correlation_ids: list[CorrelationId]) -> None:
        for corr_id in correlation_ids:
            self._pending_events.pop(corr_id, None)

    def tick(self, now: float | None = None) -> int:
        now = now if now is not None else time.monotonic()
        cutoff = now - self.config.ttl_s
        before = len(self._pending_events) + len(self._pending_frames)
        self._pending_events = {
            k: v for k, v in self._pending_events.items() if v.inserted_at >= cutoff
        }
        self._pending_frames = [pf for pf in self._pending_frames if pf.inserted_at >= cutoff]
        return before - len(self._pending_events) - len(self._pending_frames)

    def _compact_pending_frames(self) -> None:
        self._pending_frames = [pf for pf in self._pending_frames if pf.unmatched]

    def _evict_if_full_events(self, now: float) -> None:
        if len(self._pending_events) < self.config.capacity:
            return
        oldest = min(self._pending_events.items(), key=lambda kv: kv[1].inserted_at)
        del self._pending_events[oldest[0]]

    def _evict_if_full_frames(self, now: float) -> None:
        if len(self._pending_frames) < self.config.capacity:
            return
        self._pending_frames.sort(key=lambda pf: pf.inserted_at)
        self._pending_frames.pop(0)


class LinkerMessage:
    def __init__(
        self,
        *,
        event_persisted: EventPersisted | None = None,
        frame_captured: FrameCaptured | None = None,
        trigger_dropped: tuple[list[CorrelationId], DropReason] | None = None,
        tick: bool = False,
    ) -> None:
        self.event_persisted = event_persisted
        self.frame_captured = frame_captured
        self.trigger_dropped = trigger_dropped
        self.tick = tick


class FrameLinkerActor:
    """Background thread applying linker updates to SQLite."""

    def __init__(self, db: DatabaseManager) -> None:
        self._db = db
        self._linker = FrameLinker()
        self._q: queue.Queue[LinkerMessage | None] = queue.Queue(maxsize=8192)
        self._thread = threading.Thread(target=self._run, name="FrameLinkerActor", daemon=True)
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()

    def stop(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=2.0)
        self._started = False

    def try_send(self, msg: LinkerMessage) -> bool:
        try:
            self._q.put_nowait(msg)
            return True
        except queue.Full:
            return False

    def _run(self) -> None:
        while True:
            try:
                msg = self._q.get(timeout=1.0)
            except queue.Empty:
                self._apply_tick()
                continue
            if msg is None:
                break
            self._handle(msg)

    def _apply_tick(self) -> None:
        evicted = self._linker.tick()
        if evicted:
            logger.debug("frame linker TTL evicted %d half-paired entries", evicted)

    def _handle(self, msg: LinkerMessage) -> None:
        if msg.tick:
            self._apply_tick()
            return
        if msg.event_persisted is not None:
            upd = self._linker.on_event_persisted(msg.event_persisted)
            if upd:
                self._db.update_ui_event_frame_id(upd.row_id, upd.frame_id)
        if msg.frame_captured is not None:
            for upd in self._linker.on_frame_captured(msg.frame_captured):
                self._db.update_ui_event_frame_id(upd.row_id, upd.frame_id)
        if msg.trigger_dropped is not None:
            ids, _reason = msg.trigger_dropped
            self._linker.on_trigger_dropped(ids)
