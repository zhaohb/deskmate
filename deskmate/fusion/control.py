"""Runtime capture control: pause / resume / per-source switches.

Two layers:

* :class:`CaptureControl` — DB-backed read/write of the single ``capture_control``
  row. Used by the API to toggle state and by the fusion bus to decide what to
  record. DB-mediated (not a process-local singleton) so the API and daemon
  stay decoupled, mirroring the habits module.
* :func:`capture_allowed` — a *fail-open*, ~1s-cached gate for the daemon's hot
  capture chokepoints (screen loop, audio loop, UI-event pipeline). It never
  raises and defaults to "allowed" on any error, so capture can never be broken
  by this additive layer.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..logger import get
from .store import open_connection

logger = get("fusion.control")

# Sources that have an explicit on/off switch. ``window`` (focus/title/value)
# is low-sensitivity context and is governed by global pause only.
TOGGLEABLE = ("screen", "audio", "input", "clipboard")


def _now() -> datetime:
    return datetime.now(timezone.utc).astimezone()


def _now_iso() -> str:
    return _now().replace(microsecond=0).isoformat()


class CaptureControl:
    """Read/write the single-row ``capture_control`` surface."""

    def __init__(self, db_file: Path | None = None) -> None:
        self._lock = threading.RLock()
        self._conn = open_connection(Path(db_file) if db_file else None)
        # Defensive: the seed row is created by the schema, but ensure it exists.
        with self._lock:
            self._conn.execute("INSERT OR IGNORE INTO capture_control(id) VALUES (1)")

    def _row(self) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT paused, pause_until, screen, audio, input, clipboard, "
                "updated_at FROM capture_control WHERE id = 1"
            ).fetchone()
        return row or {
            "paused": 0, "pause_until": None,
            "screen": 1, "audio": 1, "input": 1, "clipboard": 1, "updated_at": None,
        }

    def state(self) -> dict[str, Any]:
        """Current control state with auto-resume (``pause_until``) applied."""
        row = self._row()
        paused = bool(row["paused"])
        pause_until = row.get("pause_until")
        if paused and pause_until:
            try:
                if _now() >= datetime.fromisoformat(pause_until):
                    self.resume()
                    paused = False
                    pause_until = None
            except ValueError:
                pass
        return {
            "paused": paused,
            "pause_until": pause_until,
            "screen": bool(row["screen"]),
            "audio": bool(row["audio"]),
            "input": bool(row["input"]),
            "clipboard": bool(row["clipboard"]),
            "updated_at": row.get("updated_at"),
        }

    def allows(self, source: str) -> bool:
        """Whether ``source`` may be captured/recorded right now."""
        st = self.state()
        if st["paused"]:
            return False
        if source in TOGGLEABLE:
            return bool(st[source])
        return True  # window / unknown low-sensitivity context

    # ─── mutations ───────────────────────────────────────────────────────────

    def set_paused(self, paused: bool, *, minutes: int | None = None) -> dict[str, Any]:
        until = None
        if paused and minutes and minutes > 0:
            until = (_now() + _timedelta(minutes)).replace(microsecond=0).isoformat()
        with self._lock:
            self._conn.execute(
                "UPDATE capture_control SET paused = ?, pause_until = ?, updated_at = ? WHERE id = 1",
                (1 if paused else 0, until, _now_iso()),
            )
        invalidate_cache()
        return self.state()

    def resume(self) -> dict[str, Any]:
        with self._lock:
            self._conn.execute(
                "UPDATE capture_control SET paused = 0, pause_until = NULL, updated_at = ? WHERE id = 1",
                (_now_iso(),),
            )
        invalidate_cache()
        return self.state()

    def set_source(self, source: str, enabled: bool) -> dict[str, Any]:
        if source not in TOGGLEABLE:
            raise ValueError(f"unknown source: {source!r}")
        with self._lock:
            self._conn.execute(
                f"UPDATE capture_control SET {source} = ?, updated_at = ? WHERE id = 1",
                (1 if enabled else 0, _now_iso()),
            )
        invalidate_cache()
        return self.state()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass


def _timedelta(minutes: int):  # noqa: ANN202 - tiny local import shim
    from datetime import timedelta  # noqa: PLC0415

    return timedelta(minutes=minutes)


# ─── fail-open cached hot-path gate (daemon process) ─────────────────────────

_GATE_TTL_S = 1.0
_gate_lock = threading.Lock()
_gate_control: CaptureControl | None = None
_gate_state: dict[str, Any] | None = None
_gate_ts: float = 0.0


def invalidate_cache() -> None:
    """Force the next :func:`capture_allowed` call to re-read the DB."""
    global _gate_ts
    with _gate_lock:
        _gate_ts = 0.0


def capture_allowed(source: str) -> bool:
    """Fail-open gate for hot capture chokepoints.

    Returns ``True`` (allow capture) on any error so this additive control layer
    can never break the recorder. Caches the control row for ~1s.
    """
    global _gate_control, _gate_state, _gate_ts
    try:
        now = time.monotonic()
        with _gate_lock:
            fresh = _gate_state is not None and (now - _gate_ts) < _GATE_TTL_S
            if not fresh:
                if _gate_control is None:
                    _gate_control = CaptureControl()
                _gate_state = _gate_control.state()
                _gate_ts = now
            st = _gate_state
        if st["paused"]:
            return False
        if source in TOGGLEABLE:
            return bool(st[source])
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("capture_allowed fail-open (%s): %s", source, exc)
        return True


def shutdown_gate() -> None:
    """Close the module-level gate connection (best effort)."""
    global _gate_control, _gate_state
    with _gate_lock:
        if _gate_control is not None:
            _gate_control.close()
            _gate_control = None
        _gate_state = None
