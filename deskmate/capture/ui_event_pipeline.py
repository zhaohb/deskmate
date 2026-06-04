"""UI event recorder pipeline — batch DB writes + capture triggers + linker."""

from __future__ import annotations

import itertools
import queue
import threading
from typing import TYPE_CHECKING, Any

from ..a11y.browser_url import resolve_browser_url
from ..a11y.ui_event_types import (
    CaptureTrigger,
    CaptureTriggerMsg,
    ScrollBurstTracker,
    TriggerGates,
    UiEventInsert,
    capture_trigger_kind,
    raw_dict_to_insert,
)
from ..capture.frame_linker import EventPersisted, FrameLinkerActor, LinkerMessage
from ..logger import get

if TYPE_CHECKING:
    from ..config import Config
    from ..db import DatabaseManager

logger = get("capture.ui_event_pipeline")

_next_correlation_id = itertools.count(1)


class UiEventPipeline:
    def __init__(
        self,
        cfg: Config,
        db: DatabaseManager,
        *,
        trigger_queue: queue.Queue[CaptureTriggerMsg],
        linker: FrameLinkerActor,
        on_meeting_observe: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.db = db
        self.trigger_queue = trigger_queue
        self.linker = linker
        self._on_meeting_observe = on_meeting_observe
        self._batch: list[UiEventInsert] = []
        self._corr_ids: list[int | None] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._flush_thread: threading.Thread | None = None
        self._scroll_burst = ScrollBurstTracker(
            delay_s=cfg.capture.scroll_stop_delay_ms / 1000.0,
        )
        self._gates = TriggerGates(
            capture_on_keystroke=cfg.capture.capture_on_keystroke,
            capture_on_clipboard=cfg.capture.capture_on_clipboard,
        )
        self._ignored = list(cfg.filters.ignored_windows)

    def start(self) -> None:
        self._stop.clear()
        self._flush_thread = threading.Thread(target=self._flush_loop, name="ui-event-flush", daemon=True)
        self._flush_thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._flush_thread:
            self._flush_thread.join(timeout=2.0)
        self._flush_batch()

    def handle_event(self, ev: dict[str, Any]) -> None:
        """Entry from UiRecorder — raw WinEvent dict to persisted row + optional capture."""
        browser_url = resolve_browser_url(
            ev.get("app_name") or "",
            pid=int(ev.get("pid") or 0),
            hwnd=int(ev.get("hwnd") or 0),
        )
        insert = raw_dict_to_insert(ev, browser_url=browser_url)

        if self._on_meeting_observe:
            self._on_meeting_observe(insert)

        trigger_kind = capture_trigger_kind(
            insert, ignored_patterns=self._ignored, gates=self._gates,
        )
        is_scroll = insert.event_type.value == "scroll"
        has_receivers = self.cfg.capture.event_driven
        want_corr = (trigger_kind is not None or is_scroll) and has_receivers
        correlation_id: int | None = next(_next_correlation_id) if want_corr else None

        if is_scroll and correlation_id is not None:
            self._scroll_burst.record(correlation_id)
        elif trigger_kind is not None and correlation_id is not None:
            try:
                self.trigger_queue.put_nowait(
                    CaptureTriggerMsg.with_correlation(trigger_kind, correlation_id),
                )
            except queue.Full:
                correlation_id = None

        if not self.cfg.capture.record_input_events:
            return
        from ..core.filter import WindowFilter  # noqa: PLC0415

        if not WindowFilter(ignored_windows=self._ignored).passes(
            insert.app_name or "unknown", insert.window_title or "",
        ):
            return

        with self._lock:
            self._batch.append(insert)
            self._corr_ids.append(correlation_id)

    def poll_scroll_burst(self) -> None:
        corr = self._scroll_burst.poll_burst_end()
        if corr is None:
            return
        try:
            self.trigger_queue.put_nowait(
                CaptureTriggerMsg.with_correlation(CaptureTrigger.SCROLL_STOP, corr),
            )
        except queue.Full:
            self.linker.try_send(LinkerMessage(trigger_dropped=([corr], None)))

    def _flush_loop(self) -> None:
        timeout_s = self.cfg.capture.ui_event_batch_timeout_ms / 1000.0
        while not self._stop.is_set():
            self._stop.wait(timeout_s)
            self.poll_scroll_burst()
            self._flush_batch()

    def _flush_batch(self) -> None:
        with self._lock:
            if not self._batch:
                return
            batch = self._batch
            corr_ids = self._corr_ids
            self._batch = []
            self._corr_ids = []
        try:
            row_ids = self.db.insert_ui_events_batch(batch)
            for row_id, corr in zip(row_ids, corr_ids, strict=True):
                if corr is not None:
                    self.linker.try_send(
                        LinkerMessage(event_persisted=EventPersisted(correlation_id=corr, row_id=row_id)),
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("ui event batch flush failed: %s", exc)
