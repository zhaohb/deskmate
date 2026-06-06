"""Data access for the unified context timeline (``context_events``).

Owns its own read/write SQLite connection to the same database file, exactly
like :class:`~deskmate.habits.store.HabitStore`. The ``context_events`` and
``capture_control`` tables are created by the main schema at startup, so this
module only reads/writes rows. WAL mode + ``busy_timeout`` make a second
connection safe alongside the main :class:`~deskmate.db.manager.DatabaseManager`.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import paths
from ..logger import get

logger = get("fusion.store")


def _dict_factory(cursor: sqlite3.Cursor, row: tuple) -> dict[str, Any]:
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


def open_connection(db_file: Path | None = None) -> sqlite3.Connection:
    """Open a second connection to the DB tuned for concurrent read/write."""
    path = Path(db_file) if db_file else paths.db_path()
    conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    conn.row_factory = _dict_factory
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().replace(microsecond=0).isoformat()


class ContextStore:
    """Serialized read/write access to the ``context_events`` table."""

    def __init__(self, db_file: Path | None = None) -> None:
        self.path = Path(db_file) if db_file else paths.db_path()
        self._lock = threading.RLock()
        self._conn = open_connection(self.path)

    # ─── writes ──────────────────────────────────────────────────────────────

    def insert_event(
        self,
        *,
        ts: str,
        source: str,
        kind: str,
        app_name: str = "",
        window_title: str = "",
        summary: str = "",
        payload: dict[str, Any] | None = None,
        confidence: float = 1.0,
        frame_id: int | None = None,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO context_events
                   (ts, source, kind, app_name, window_title, summary,
                    payload_json, confidence, frame_id)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    ts, source, kind, app_name or "", window_title or "",
                    summary or "", json.dumps(payload or {}, ensure_ascii=False),
                    float(confidence), frame_id,
                ),
            )
            return int(cur.lastrowid)

    def forget_since(self, iso_cutoff: str) -> int:
        """Delete unified events at/after ``iso_cutoff`` (privacy "forget")."""
        with self._lock:
            return int(
                self._conn.execute(
                    "DELETE FROM context_events WHERE ts >= ?", (iso_cutoff,),
                ).rowcount or 0
            )

    # ─── reads ───────────────────────────────────────────────────────────────

    def list_events(
        self,
        *,
        since: str | None = None,
        until: str | None = None,
        sources: list[str] | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if since:
            clauses.append("ts >= ?")
            params.append(since)
        if until:
            clauses.append("ts <= ?")
            params.append(until)
        if sources:
            placeholders = ",".join("?" for _ in sources)
            clauses.append(f"source IN ({placeholders})")
            params.extend(sources)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = (
            "SELECT id, ts, source, kind, app_name, window_title, summary, "
            "payload_json, confidence, frame_id "
            f"FROM context_events{where} ORDER BY ts DESC, id DESC LIMIT ?"
        )
        params.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        for r in rows:
            try:
                r["payload"] = json.loads(r.pop("payload_json") or "{}")
            except (ValueError, TypeError):
                r["payload"] = {}
        return rows

    def count_since(self, iso_cutoff: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM context_events WHERE ts >= ?", (iso_cutoff,),
            ).fetchone()
        return int(row["n"] if row else 0)

    def source_breakdown(self, *, since: str | None = None) -> list[dict[str, Any]]:
        where, params = ("", [])
        if since:
            where, params = (" WHERE ts >= ?", [since])
        with self._lock:
            return self._conn.execute(
                f"SELECT source, COUNT(*) AS n FROM context_events{where} "
                "GROUP BY source ORDER BY n DESC", params,
            ).fetchall()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
