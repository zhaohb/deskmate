"""Context fusion bus: project every signal into one unified timeline.

Subscribes to the existing in-process :mod:`deskmate.events` bus and writes a
normalized row into ``context_events`` for each signal, with ``source`` /
``kind`` / ``confidence`` provenance. **Producers are never modified** — this is
a pure, additive consumer.

Design:
* The event-bus callback only enqueues (never touches SQLite), so capture
  threads are never blocked or slowed by fusion.
* A single background writer thread drains the queue and persists rows.
* Per-source recording + global pause are honored via :class:`CaptureControl`,
  so toggling a source off (or pausing) also stops it appearing in the timeline.
"""

from __future__ import annotations

import queue
import threading
from typing import TYPE_CHECKING, Any

from .. import events as bus
from ..logger import get
from .control import CaptureControl
from .store import ContextStore, now_iso

if TYPE_CHECKING:
    from ..config import Config

logger = get("fusion.bus")

# event-bus EventType -> (source, kind, control-source). control-source is the
# switch consulted in CaptureControl ("window" => global-pause only).
_MAPPING: dict[str, tuple[str, str, str]] = {
    "frame_written": ("screen", "frame", "screen"),
    "audio_transcribed": ("audio", "transcript", "audio"),
    "clipboard": ("clipboard", "clipboard", "clipboard"),
    "key_text": ("input", "text", "input"),
    "click": ("input", "click", "input"),
    "window_focus": ("window", "focus", "window"),
    "title_change": ("window", "title", "window"),
    "value_change": ("window", "value", "window"),
}

# Confidence per source: UIA/window signals are exact; ASR is probabilistic.
_CONFIDENCE = {"audio": 0.8}


class ContextFusionBus:
    """Background subscriber that fuses all sources into ``context_events``."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._summary_max = max(0, int(getattr(cfg.fusion, "summary_max_chars", 200)))
        self._record_window = bool(getattr(cfg.fusion, "record_window_events", True))
        self._queue: queue.Queue[tuple[str, dict[str, Any], float]] = queue.Queue(maxsize=8192)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._unsub: Any = None
        self._store: ContextStore | None = None
        self._control: CaptureControl | None = None

    def start(self) -> None:
        self._store = ContextStore()
        self._control = CaptureControl()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="daemon-fusion", daemon=True)
        self._thread.start()
        self._unsub = bus.subscribe(self._on_event)
        logger.info("context fusion bus started")

    def stop(self) -> None:
        if self._unsub:
            try:
                self._unsub()
            except Exception:  # noqa: BLE001
                pass
            self._unsub = None
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._store:
            self._store.close()
            self._store = None
        if self._control:
            self._control.close()
            self._control = None
        logger.info("context fusion bus stopped")

    # ─── event-bus callback (must be cheap, never block producers) ───────────

    def _on_event(self, event: bus.Event) -> None:
        et = event.type.value if hasattr(event.type, "value") else str(event.type)
        if et not in _MAPPING:
            return
        try:
            self._queue.put_nowait((et, dict(event.data or {}), float(event.timestamp)))
        except queue.Full:
            pass  # drop under backpressure; the timeline is best-effort context

    # ─── background writer ───────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                et, data, _ts = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._persist(et, data)
            except Exception as exc:  # noqa: BLE001
                logger.debug("fusion persist failed (%s): %s", et, exc)

    def _persist(self, et: str, data: dict[str, Any]) -> None:
        source, kind, control_source = _MAPPING[et]
        if source == "window" and not self._record_window:
            return
        if self._control is not None and not self._control.allows(control_source):
            return
        if self._store is None:
            return

        app = str(data.get("app_name") or "")
        window = str(data.get("window_title") or "")
        frame_id = data.get("frame_id")
        confidence = _CONFIDENCE.get(source, 1.0)
        summary, payload = self._shape(et, data)

        self._store.insert_event(
            ts=now_iso(),
            source=source,
            kind=kind,
            app_name=app,
            window_title=window,
            summary=summary,
            payload=payload,
            confidence=confidence,
            frame_id=int(frame_id) if isinstance(frame_id, int) else None,
        )

    def _shape(self, et: str, data: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Derive a short summary + compact payload from the raw event data."""
        cap = self._summary_max

        def clip(text: str) -> str:
            text = (text or "").strip().replace("\n", " ")
            return text[:cap] if cap else text

        if et == "frame_written":
            return clip(data.get("trigger") or "frame"), {
                "monitor_id": data.get("monitor_id"),
                "trigger": data.get("trigger"),
            }
        if et == "audio_transcribed":
            return clip(data.get("text") or ""), {
                "transcript_id": data.get("transcript_id"),
                "device": data.get("device"),
                "speaker_id": data.get("speaker_id"),
            }
        if et == "clipboard":
            return clip(data.get("text") or ""), {"length": data.get("length")}
        if et == "key_text":
            return clip(data.get("text") or ""), {"reason": data.get("reason")}
        if et == "click":
            btn = data.get("button") or "click"
            x, y = data.get("x"), data.get("y")
            return f"{btn} @({x},{y})", {"button": btn, "x": x, "y": y}
        # window_focus / title_change / value_change
        return clip(data.get("window_title") or data.get("app_name") or ""), {}
