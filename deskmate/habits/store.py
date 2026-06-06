"""Data access for the habits module.

Owns its own read/write SQLite connection to the same database file. The
``habit_*`` tables are created by the main schema at startup, so this store
only reads/writes rows. Reads from ``frames`` / ``ui_events`` power mining and
"current state" detection. WAL mode + ``busy_timeout`` make a second connection
safe alongside the main :class:`~deskmate.db.manager.DatabaseManager`.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

from .. import paths
from ..logger import get

logger = get("habits.store")


def _dict_factory(cursor: sqlite3.Cursor, row: tuple) -> dict[str, Any]:
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


class HabitStore:
    """Serialized read/write access to the ``habit_*`` tables."""

    def __init__(self, db_file: Path | None = None) -> None:
        self.path = Path(db_file) if db_file else paths.db_path()
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = _dict_factory
        with self._lock:
            self._conn.execute("PRAGMA busy_timeout = 5000")
            self._conn.execute("PRAGMA foreign_keys = ON")

    # ─── raw frame reads (for mining + current state) ────────────────────────

    def frame_durations_since(self, start_iso: str, *, gap_cap_sec: float = 300.0) -> list[dict[str, Any]]:
        """Return per-frame attributed durations from ``start_iso`` onward.

        Each row: ``timestamp``, ``app_name``, ``window_name``, ``dur_sec``
        (seconds attributed to the frame, capped at ``gap_cap_sec``). Duration
        is the gap to the next chronological frame, mirroring the activity
        summary's gap-based time accounting.
        """
        sql = """
            SELECT timestamp, app_name, window_name, dur_sec FROM (
                SELECT timestamp,
                       app_name,
                       COALESCE(window_name, '') AS window_name,
                       MIN(
                         (JULIANDAY(LEAD(timestamp) OVER (ORDER BY timestamp))
                          - JULIANDAY(timestamp)) * 86400.0,
                         ?
                       ) AS dur_sec
                  FROM frames
                 WHERE timestamp >= ?
                   AND app_name IS NOT NULL AND app_name != ''
            )
            WHERE dur_sec IS NOT NULL AND dur_sec > 0
        """
        with self._lock:
            return self._conn.execute(sql, (gap_cap_sec, start_iso)).fetchall()

    def recent_frames_between(self, start_iso: str, end_iso: str) -> list[dict[str, Any]]:
        """Frames in ``[start_iso, end_iso]`` ordered by time (for current state)."""
        sql = """
            SELECT timestamp, app_name, COALESCE(window_name, '') AS window_name
              FROM frames
             WHERE timestamp BETWEEN ? AND ?
               AND app_name IS NOT NULL AND app_name != ''
             ORDER BY timestamp
        """
        with self._lock:
            return self._conn.execute(sql, (start_iso, end_iso)).fetchall()

    def last_app_switch_ts(self, app_name: str, before_iso: str) -> str | None:
        """Timestamp of the most recent switch INTO ``app_name`` before ``before_iso``.

        Uses ``ui_events`` of type ``app_switch`` / ``window_focus``. Returns
        ``None`` when no such event exists.
        """
        sql = """
            SELECT MAX(timestamp) AS ts
              FROM ui_events
             WHERE timestamp <= ?
               AND event_type IN ('app_switch', 'window_focus')
               AND app_name = ?
        """
        with self._lock:
            row = self._conn.execute(sql, (before_iso, app_name)).fetchone()
        return row["ts"] if row and row.get("ts") else None

    # ─── habit_profiles ──────────────────────────────────────────────────────

    def replace_profiles(self, rows: list[dict[str, Any]]) -> int:
        """Atomically replace all habit profiles with ``rows``.

        Each row needs: ``day_type``, ``slot``, ``category``, ``top_app``,
        ``avg_minutes``, ``frequency``, ``sample_days``.
        """
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                self._conn.execute("DELETE FROM habit_profiles")
                self._conn.executemany(
                    """INSERT INTO habit_profiles
                       (day_type, slot, category, top_app, avg_minutes,
                        frequency, sample_days, updated_at)
                       VALUES (:day_type, :slot, :category, :top_app, :avg_minutes,
                               :frequency, :sample_days, datetime('now'))""",
                    rows,
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise
        return len(rows)

    def profiles_for_slot(self, day_type: str, slot: int) -> list[dict[str, Any]]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM habit_profiles WHERE day_type = ? AND slot = ? "
                "ORDER BY frequency DESC",
                (day_type, slot),
            ).fetchall()

    def all_profiles(self) -> list[dict[str, Any]]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM habit_profiles ORDER BY day_type, slot, frequency DESC"
            ).fetchall()

    # ─── habit_rules ─────────────────────────────────────────────────────────

    def ensure_rules(self, defaults: list[dict[str, Any]]) -> None:
        """Insert any missing default rules (idempotent; never overwrites)."""
        with self._lock:
            for r in defaults:
                self._conn.execute(
                    """INSERT OR IGNORE INTO habit_rules
                       (name, rule_type, enabled, params_json, cooldown_min,
                        quiet_hours, priority)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        r["name"], r["rule_type"], 1 if r.get("enabled", True) else 0,
                        json.dumps(r.get("params", {})), int(r.get("cooldown_min", 120)),
                        r.get("quiet_hours", "22-8"), r.get("priority", "M"),
                    ),
                )

    def enabled_rules(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM habit_rules WHERE enabled = 1 "
                "ORDER BY CASE priority WHEN 'H' THEN 0 WHEN 'M' THEN 1 ELSE 2 END"
            ).fetchall()
        for r in rows:
            try:
                r["params"] = json.loads(r.get("params_json") or "{}")
            except (TypeError, ValueError):
                r["params"] = {}
        return rows

    def set_rule_enabled(self, name: str, enabled: bool) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE habit_rules SET enabled = ? WHERE name = ?",
                (1 if enabled else 0, name),
            )
            return cur.rowcount > 0

    # ─── habit_suggestions ───────────────────────────────────────────────────

    def last_suggestion_ts(self, rule_name: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT MAX(created_at) AS ts FROM habit_suggestions "
                "WHERE rule_name = ? AND status != 'suppressed'",
                (rule_name,),
            ).fetchone()
        return row["ts"] if row and row.get("ts") else None

    def count_sent_since(self, since_iso: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM habit_suggestions "
                "WHERE status = 'sent' AND created_at >= ?",
                (since_iso,),
            ).fetchone()
        return int(row["n"]) if row else 0

    def recent_feedback(self, rule_name: str, limit: int = 5) -> list[int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT feedback FROM habit_suggestions "
                "WHERE rule_name = ? AND feedback IS NOT NULL "
                "ORDER BY created_at DESC LIMIT ?",
                (rule_name, limit),
            ).fetchall()
        return [int(r["feedback"]) for r in rows if r.get("feedback") is not None]

    def insert_suggestion(
        self,
        *,
        rule_name: str,
        message: str,
        context: dict[str, Any],
        channel: str,
        status: str,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO habit_suggestions
                   (rule_name, message, context_json, channel, status)
                   VALUES (?, ?, ?, ?, ?)""",
                (rule_name, message, json.dumps(context), channel, status),
            )
            return int(cur.lastrowid)

    def list_suggestions(self, *, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        sql = "SELECT * FROM habit_suggestions"
        params: list[Any] = []
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    def set_suggestion_feedback(self, suggestion_id: int, feedback: int) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE habit_suggestions SET feedback = ? WHERE id = ?",
                (feedback, suggestion_id),
            )
            return cur.rowcount > 0

    def set_suggestion_status(self, suggestion_id: int, status: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE habit_suggestions SET status = ? WHERE id = ?",
                (status, suggestion_id),
            )
            return cur.rowcount > 0

    def close(self) -> None:
        with self._lock:
            self._conn.close()
