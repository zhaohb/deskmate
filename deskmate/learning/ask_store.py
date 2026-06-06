"""Persistence for answered Ask queries (``ask_history``).

Additive and write-mostly: the ``/ask`` API route records each answered query
here so the optional LoRA training miner can turn the user's own questions and
the assistant's evidence-based answers into high-quality SFT pairs.

Owns its own connection to the same database file (WAL + ``busy_timeout`` make
this safe alongside the main ``DatabaseManager``), mirroring
:class:`~deskmate.fusion.store.ContextStore` and the habits store.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import paths
from ..logger import get

logger = get("learning.ask_store")


def _dict_factory(cursor: sqlite3.Cursor, row: tuple) -> dict[str, Any]:
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().replace(microsecond=0).isoformat()


class AskHistoryStore:
    """Serialized write access to the ``ask_history`` table."""

    def __init__(self, db_file: Path | None = None) -> None:
        self.path = Path(db_file) if db_file else paths.db_path()
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = _dict_factory
        with self._lock:
            self._conn.execute("PRAGMA busy_timeout = 5000")

    def record(self, *, question: str, answer: str, tool_count: int = 0) -> int | None:
        """Persist one (question, answer) pair. No-op for empty inputs."""
        q = (question or "").strip()
        a = (answer or "").strip()
        if not q or not a:
            return None
        try:
            with self._lock:
                cur = self._conn.execute(
                    """INSERT INTO ask_history (question, answer, tool_count, created_at)
                       VALUES (?,?,?,?)""",
                    (q, a, int(tool_count), _now_iso()),
                )
                return int(cur.lastrowid)
        except sqlite3.Error as exc:
            logger.debug("ask_history insert failed: %s", exc)
            return None

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass


__all__ = ["AskHistoryStore"]
