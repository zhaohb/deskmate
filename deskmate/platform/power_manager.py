"""Central power manager: tag background worker threads with EcoQoS by name.

Zero-invasive battery saver. Existing workers are NOT modified — this manager
discovers them via threading.enumerate() by thread name and applies thread-level
EcoQoS from the outside:

  * On BATTERY  → eco-throttle the configured background workers onto E/LPE-cores.
  * On AC       → clear the override so they run at the system default again.

It re-applies on a poll because (a) the AC/battery state changes, and (b) some
workers (e.g. capture) may be recreated, getting a fresh TID that needs re-tagging.

Default eco targets (by thread name, see daemon.py / reconciler.py / *_loop):
  - daemon-semantic-index   (embedding index, pure background)
  - RedactReconciler        (ONNX redaction, post-processing)
  - event-driven-capture    (screen capture + OCR; OCR is CPU-bound RapidOCR)
  - daemon-heartbeat        (legacy capture loop, same role as above)
  - daemon-retention        (periodic DB cleanup)

Ask is request-driven (no single long-lived thread) and its heavy lifting runs on
the GPU/Ollama, so it is intentionally NOT in the eco list — it keeps P-cores.

Everything is best-effort; a missing capability turns the whole manager into a
no-op without raising.
"""

from __future__ import annotations

import threading

from ..logger import get
from .battery import PowerSource, power_status
from .qos import clear_thread, qos_available, set_thread_eco

logger = get("platform.power_manager")

# Background worker thread names that are safe to push onto E/LPE-cores while on
# battery. All are either pure background (no user waiting) or already accepted
# by the user as throttle-on-battery (OCR/capture).
DEFAULT_ECO_THREAD_NAMES = (
    "daemon-semantic-index",
    "RedactReconciler",
    "event-driven-capture",
    "daemon-heartbeat",
    "daemon-retention",
)


class PowerManager:
    """Polls power source and tags worker threads with EcoQoS accordingly."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        poll_seconds: float = 15.0,
        eco_thread_names: tuple[str, ...] = DEFAULT_ECO_THREAD_NAMES,
    ) -> None:
        self.enabled = enabled
        self.poll_seconds = max(2.0, poll_seconds)
        self.eco_thread_names = set(eco_thread_names)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        # Last source we acted on, so we only re-tag on transitions or new TIDs.
        self._last_source: PowerSource | None = None
        # TIDs we have currently eco-tagged, to avoid redundant syscalls.
        self._eco_tids: set[int] = set()

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self) -> None:
        if self._thread:
            return
        if not self.enabled:
            logger.info("power manager disabled by config")
            return
        if not qos_available():
            logger.info("thread QoS unavailable on this platform — power manager idle")
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="daemon-power-manager", daemon=True,
        )
        self._thread.start()
        logger.info(
            "power manager started (poll=%.0fs, eco targets=%d)",
            self.poll_seconds, len(self.eco_thread_names),
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        # Leave threads un-throttled on shutdown.
        self._clear_all()

    # ── status (for UI / API) ────────────────────────────────────────────────
    def status(self) -> dict:
        """Snapshot for the battery-saver UI. Never raises."""
        st = power_status()
        return {
            "available": qos_available(),
            "enabled": self.enabled,
            "source": st.source.value,
            "percent": st.percent,
            "has_battery": st.has_battery,
            "runtime_seconds": st.runtime_seconds,
            "eco_active": bool(self._eco_tids),
            "eco_thread_count": len(self._eco_tids),
            "eco_targets": sorted(self.eco_thread_names),
        }

    # ── core loop ──────────────────────────────────────────────────────────
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as exc:  # noqa: BLE001
                logger.debug("power manager tick failed: %s", exc)
            self._stop.wait(self.poll_seconds)

    def _tick(self) -> None:
        status = power_status()
        # Fail-open: UNKNOWN (e.g. API failed) is treated as AC → no throttling.
        on_battery = status.source == PowerSource.BATTERY

        targets = self._current_target_tids()
        if on_battery:
            self._apply_eco(targets)
        else:
            self._clear_all()

        if status.source != self._last_source:
            logger.info(
                "power source -> %s (battery=%s%%) ; eco-tagged %d worker thread(s)",
                status.source.value,
                status.percent if status.percent is not None else "?",
                len(self._eco_tids),
            )
            self._last_source = status.source

    # ── thread discovery + tagging ───────────────────────────────────────────
    def _current_target_tids(self) -> dict[int, str]:
        """Map native TID -> thread name for currently-alive eco-target threads."""
        out: dict[int, str] = {}
        for t in threading.enumerate():
            if t.name in self.eco_thread_names:
                tid = getattr(t, "native_id", None)
                if tid:
                    out[tid] = t.name
        return out

    def _apply_eco(self, targets: dict[int, str]) -> None:
        for tid, name in targets.items():
            if tid in self._eco_tids:
                continue  # already throttled
            if set_thread_eco(tid):
                self._eco_tids.add(tid)
                logger.debug("eco-throttled %s (tid=%d)", name, tid)
        # Forget TIDs that no longer exist (worker restarted/exited).
        self._eco_tids &= set(targets)

    def _clear_all(self) -> None:
        if not self._eco_tids:
            return
        for tid in list(self._eco_tids):
            clear_thread(tid)
        logger.debug("cleared eco throttling on %d thread(s)", len(self._eco_tids))
        self._eco_tids.clear()
