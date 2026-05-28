"""Thread-safe SQLite wrapper for capture, search and retention.

Note: writes are serialized through a single RLock. SQLite is fine with this
and it keeps concurrency behavior simple.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import paths
from ..logger import get
from .schema import SCHEMA, SCHEMA_VERSION

logger = get("db")


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().replace(microsecond=0).isoformat()


def _dict_factory(cursor: sqlite3.Cursor, row: tuple) -> dict[str, Any]:
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


class DatabaseManager:
    """One connection, serialized writes."""

    def __init__(self, db_file: Path | None = None) -> None:
        paths.ensure_dirs()
        self.path = Path(db_file) if db_file else paths.db_path()
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = _dict_factory
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._record_migration(SCHEMA_VERSION)

    # ─── migrations ──────────────────────────────────────────────────────────
    def _record_migration(self, version: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO _pca_migrations(version, applied_at) VALUES (?, ?)",
            (version, _now_iso()),
        )

    def schema_version(self) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT version FROM _pca_migrations ORDER BY version DESC LIMIT 1"
            ).fetchone()
        return row["version"] if row else None

    # ─── video chunks ────────────────────────────────────────────────────────
    def insert_video_chunk(self, *, file_path: str, device_name: str = "", fps: float = 1.0) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO video_chunks(file_path, device_name, fps) VALUES (?, ?, ?)",
                (file_path, device_name, fps),
            )
            return int(cur.lastrowid)

    def list_video_chunks(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            return self._conn.execute(
                """SELECT vc.*,
                          COUNT(f.id) AS frame_count,
                          MIN(f.timestamp) AS first_frame_timestamp,
                          MAX(f.timestamp) AS last_frame_timestamp
                     FROM video_chunks vc
                     LEFT JOIN frames f ON f.video_chunk_id = vc.id
                    GROUP BY vc.id
                    ORDER BY vc.id DESC
                    LIMIT ?""",
                (limit,),
            ).fetchall()

    def video_chunk_by_id(self, chunk_id: int) -> dict[str, Any] | None:
        with self._lock:
            return self._conn.execute(
                """SELECT vc.*,
                          COUNT(f.id) AS frame_count,
                          MIN(f.timestamp) AS first_frame_timestamp,
                          MAX(f.timestamp) AS last_frame_timestamp
                     FROM video_chunks vc
                     LEFT JOIN frames f ON f.video_chunk_id = vc.id
                    WHERE vc.id = ?
                    GROUP BY vc.id""",
                (chunk_id,),
            ).fetchone()

    # ─── frames ──────────────────────────────────────────────────────────────
    def insert_frame(
        self,
        *,
        monitor_id: int,
        device_name: str,
        app_name: str,
        window_name: str,
        browser_url: str | None,
        focused: bool,
        snapshot_path: str | None,
        width: int,
        height: int,
        capture_trigger: str,
        video_chunk_id: int | None = None,
        offset_index: int = 0,
        name: str | None = None,
        document_path: str | None = None,
        timestamp: str | None = None,
    ) -> int:
        ts = timestamp or _now_iso()
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO frames
                   (video_chunk_id, offset_index, timestamp, name,
                    app_name, window_name, focused, browser_url, document_path,
                    snapshot_path, monitor_id, device_name, width, height, capture_trigger)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    video_chunk_id, offset_index, ts, name,
                    app_name, window_name, 1 if focused else 0, browser_url, document_path,
                    snapshot_path, monitor_id, device_name, width, height, capture_trigger,
                ),
            )
            return int(cur.lastrowid)

    def attach_ocr(
        self,
        frame_id: int,
        *,
        text: str,
        text_json: str | None,
        engine: str,
        confidence: float | None,
        redacted_text: str | None = None,
        redacted_text_json: str | None = None,
    ) -> None:
        text_length = len(text or "")
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO ocr_text
                   (frame_id, text, text_json, ocr_engine, text_length,
                    redacted_text, redacted_text_json)
                   VALUES (?,?,?,?,?,?,?)""",
                (frame_id, text or "", text_json or "[]", engine,
                 text_length, redacted_text, redacted_text_json),
            )
            self._conn.execute(
                "UPDATE frames SET text_length = ? WHERE id = ?",
                (text_length, frame_id),
            )
            self._reindex_frame_fts(frame_id)

    def attach_accessibility(
        self,
        frame_id: int,
        *,
        text: str,
        focused_role: str | None,
        focused_name: str | None,
        focused_value: str | None,
        tree_json: str | None,
        on_screen: int | None = None,
        elements_ref_frame_id: int | None = None,
        redacted_text: str | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO frame_accessibility
                   (frame_id, text, tree_json,
                    focused_role, focused_name, focused_value,
                    on_screen, elements_ref_frame_id, redacted_text)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (frame_id, text or "", tree_json or "{}",
                 focused_role, focused_name, focused_value,
                 on_screen, elements_ref_frame_id, redacted_text),
            )
            self._reindex_frame_fts(frame_id)

    def _reindex_frame_fts(self, frame_id: int) -> None:
        row = self._conn.execute(
            """SELECT f.timestamp, f.app_name, f.window_name, f.browser_url,
                      f.document_path,
                      COALESCE(o.text,'')        AS ocr,
                      COALESCE(a.text,'')        AS ax
                 FROM frames f
                 LEFT JOIN ocr_text             o ON o.frame_id = f.id
                 LEFT JOIN frame_accessibility  a ON a.frame_id = f.id
                WHERE f.id = ?""",
            (frame_id,),
        ).fetchone()
        if not row:
            return
        self._conn.execute("DELETE FROM frames_full_text WHERE frame_id = ?", (frame_id,))
        self._conn.execute(
            """INSERT INTO frames_full_text
               (frame_id, timestamp, app_name, window_name, browser_url,
                document_path, ocr_text, accessibility_text)
               VALUES (?,?,?,?,?,?,?,?)""",
            (
                frame_id, row["timestamp"], row["app_name"] or "",
                row["window_name"] or "", row["browser_url"] or "",
                row["document_path"] or "", row["ocr"] or "", row["ax"] or "",
            ),
        )

    # ─── ui_events ───────────────────────────────────────────────────────────
    def insert_ui_events_batch(self, events: list[Any]) -> list[int]:
        """Batch insert UI events."""
        from ..a11y.ui_event_types import UiEventInsert  # noqa: PLC0415

        if not events:
            return []
        row_ids: list[int] = []
        with self._lock:
            for ev in events:
                if isinstance(ev, UiEventInsert):
                    et, app, title, url, data_json = ev.to_db_row()
                else:
                    et = str(ev.get("event_type", "event"))
                    app = ev.get("app_name")
                    title = ev.get("window_title")
                    url = ev.get("browser_url")
                    data_json = json.dumps(ev.get("data") or {}, ensure_ascii=False)
                cur = self._conn.execute(
                    """INSERT INTO ui_events
                       (timestamp, relative_ms, event_type,
                        app_name, window_title, browser_url, frame_id,
                        data_json, element_json)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        _now_iso(), 0, et, app, title, url, None,
                        data_json, None,
                    ),
                )
                eid = int(cur.lastrowid)
                row_ids.append(eid)
                text_content = ""
                if et == "text" or et == "clipboard":
                    try:
                        payload = json.loads(data_json)
                        text_content = payload.get("content", "") or ""
                    except Exception:  # noqa: BLE001
                        pass
                self._conn.execute(
                    """INSERT INTO ui_events_fts
                       (event_id, timestamp, event_type,
                        app_name, window_title, text_content)
                       VALUES (?,?,?,?,?,?)""",
                    (eid, _now_iso(), et, app or "", title or "", text_content),
                )
        return row_ids

    def update_ui_event_frame_id(self, row_id: int, frame_id: int) -> None:
        """Set frame_id on a UI event row if not already linked."""
        with self._lock:
            self._conn.execute(
                "UPDATE ui_events SET frame_id = ? WHERE id = ? AND frame_id IS NULL",
                (frame_id, row_id),
            )

    def insert_ui_event(
        self,
        *,
        event_type: str,
        app_name: str | None,
        window_title: str | None,
        data: dict[str, Any],
        browser_url: str | None = None,
        frame_id: int | None = None,
        element: dict[str, Any] | None = None,
        relative_ms: int = 0,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO ui_events
                   (timestamp, relative_ms, event_type,
                    app_name, window_title, browser_url, frame_id,
                    data_json, element_json)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    _now_iso(), relative_ms, event_type,
                    app_name, window_title, browser_url, frame_id,
                    json.dumps(data, ensure_ascii=False),
                    json.dumps(element, ensure_ascii=False) if element else None,
                ),
            )
            eid = int(cur.lastrowid)
            text_content = ""
            if event_type == "text":
                text_content = data.get("content", "") or ""
            elif event_type == "clipboard":
                text_content = data.get("content", "") or ""
            self._conn.execute(
                """INSERT INTO ui_events_fts
                   (event_id, timestamp, event_type,
                    app_name, window_title, text_content)
                   VALUES (?,?,?,?,?,?)""",
                (eid, _now_iso(), event_type, app_name or "", window_title or "", text_content),
            )
            return eid

    # ─── audio ───────────────────────────────────────────────────────────────
    def insert_audio_chunk(self, *, file_path: str, device_name: str, duration_ms: int) -> int:
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO audio_chunks(file_path, device_name, timestamp, duration_ms)
                   VALUES (?, ?, ?, ?)""",
                (file_path, device_name, _now_iso(), duration_ms),
            )
            return int(cur.lastrowid)

    def mark_audio_chunk_status(self, chunk_id: int, status: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE audio_chunks SET processing_status = ? WHERE id = ?", (status, chunk_id)
            )

    def insert_transcript(
        self,
        *,
        device: str,
        text: str,
        language: str | None,
        audio_chunk_id: int | None = None,
        offset_index: int = 0,
        start_time: float | None = None,
        end_time: float | None = None,
        speaker_id: int | None = None,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO audio_transcriptions
                   (audio_chunk_id, offset_index, timestamp, transcription,
                    device, language, speaker_id, start_time, end_time, text_length)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (audio_chunk_id, offset_index, _now_iso(), text or "",
                 device, language, speaker_id, start_time, end_time, len(text or "")),
            )
            tid = int(cur.lastrowid)
            self._conn.execute(
                """INSERT INTO audio_transcriptions_fts
                   (transcription_id, timestamp, speaker_id, transcription)
                   VALUES (?, ?, ?, ?)""",
                (tid, _now_iso(), speaker_id or 0, text or ""),
            )
            return tid

    def insert_speaker_embedding(
        self,
        *,
        speaker_id: int,
        embedding: list[float],
        audio_chunk_id: int | None = None,
        transcription_id: int | None = None,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO speaker_embeddings
                   (speaker_id, embedding_json, audio_chunk_id, transcription_id)
                   VALUES (?, ?, ?, ?)""",
                (speaker_id, json.dumps(embedding), audio_chunk_id, transcription_id),
            )
            return int(cur.lastrowid)

    # ─── meetings ────────────────────────────────────────────────────────────
    def insert_meeting(
        self,
        *,
        name: str,
        started_at: str | None = None,
        note: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO meetings(name, note, started_at, metadata)
                   VALUES (?, ?, ?, ?)""",
                (name, note, started_at or _now_iso(), json.dumps(metadata or {}, ensure_ascii=False)),
            )
            return int(cur.lastrowid)

    def end_meeting(self, meeting_id: int, *, ended_at: str | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE meetings SET ended_at = COALESCE(ended_at, ?) WHERE id = ?",
                (ended_at or _now_iso(), meeting_id),
            )

    def update_meeting(
        self,
        meeting_id: int,
        *,
        name: str | None = None,
        note: str | None = None,
    ) -> bool:
        """Patch a meeting's name and/or note. Returns True if a row was updated."""
        sets: list[str] = []
        params: list[Any] = []
        if name is not None:
            sets.append("name = ?")
            params.append(name)
        if note is not None:
            sets.append("note = ?")
            params.append(note)
        if not sets:
            return False
        params.append(meeting_id)
        with self._lock:
            cur = self._conn.execute(
                f"UPDATE meetings SET {', '.join(sets)} WHERE id = ?",
                params,
            )
            return cur.rowcount > 0

    def update_meeting_metadata(self, meeting_id: int, metadata: dict[str, Any]) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT metadata FROM meetings WHERE id = ?", (meeting_id,)
            ).fetchone()
            current: dict[str, Any] = {}
            if row and row.get("metadata"):
                try:
                    current = json.loads(row["metadata"])
                except (TypeError, ValueError):
                    current = {}
            current.update(metadata)
            self._conn.execute(
                "UPDATE meetings SET metadata = ? WHERE id = ?",
                (json.dumps(current, ensure_ascii=False), meeting_id),
            )

    def insert_meeting_segment(
        self,
        *,
        meeting_id: int,
        transcription_id: int | None,
        speaker_id: int | None,
        text: str,
        start_time: float,
        end_time: float,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO meeting_transcript_segments
                   (meeting_id, transcription_id, speaker_id, text, start_time, end_time)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (meeting_id, transcription_id, speaker_id, text or "", start_time, end_time),
            )
            return int(cur.lastrowid)

    def list_meetings(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            return self._conn.execute(
                """SELECT m.*,
                          COUNT(s.id) AS segment_count,
                          COALESCE(SUM(LENGTH(s.text)), 0) AS transcript_length
                     FROM meetings m
                     LEFT JOIN meeting_transcript_segments s ON s.meeting_id = m.id
                    GROUP BY m.id
                    ORDER BY m.started_at DESC
                    LIMIT ?""",
                (limit,),
            ).fetchall()

    def meeting_by_id(self, meeting_id: int) -> dict[str, Any] | None:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM meetings WHERE id = ?", (meeting_id,)
            ).fetchone()

    def active_meeting(self) -> dict[str, Any] | None:
        with self._lock:
            return self._conn.execute(
                """SELECT * FROM meetings
                    WHERE ended_at IS NULL
                    ORDER BY started_at DESC LIMIT 1"""
            ).fetchone()

    def list_meeting_segments(self, meeting_id: int) -> list[dict[str, Any]]:
        with self._lock:
            return self._conn.execute(
                """SELECT s.*, sp.name AS speaker_name
                     FROM meeting_transcript_segments s
                     LEFT JOIN speakers sp ON sp.id = s.speaker_id
                    WHERE s.meeting_id = ?
                    ORDER BY s.start_time ASC, s.id ASC""",
                (meeting_id,),
            ).fetchall()

    # ─── search ─────────────────────────────────────────────────────────────
    def search(
        self,
        query: str | None,
        content_type: str = "all",
        *,
        limit: int = 50,
        offset: int = 0,
        start_time: str | None = None,
        end_time: str | None = None,
        app_name: str | None = None,
        window_name: str | None = None,
        frame_name: str | None = None,
        browser_url: str | None = None,
        focused: bool | None = None,
        min_length: int | None = None,
        max_length: int | None = None,
        speaker_ids: list[int] | None = None,
    ) -> list[Any]:
        from .search_engine import SearchEngine

        return SearchEngine(self._conn, self._lock).search(
            query or "",
            content_type,
            limit=limit,
            offset=offset,
            start_time=start_time,
            end_time=end_time,
            app_name=app_name,
            window_name=window_name,
            frame_name=frame_name,
            browser_url=browser_url,
            focused=focused,
            min_length=min_length,
            max_length=max_length,
            speaker_ids=speaker_ids,
        )

    def search_frames(
        self,
        query: str | None,
        *,
        app_name: str | None = None,
        window_name: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        from .search_engine import SearchEngine

        rows = SearchEngine(self._conn, self._lock)._search_ocr(
            query or "",
            limit=limit,
            offset=offset,
            start=start,
            end=end,
            app_name=app_name,
            window_name=window_name,
            frame_name=None,
            browser_url=None,
            focused=None,
            min_length=None,
            max_length=None,
        )
        return rows

    def recent_frames(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            return self._conn.execute(
                """SELECT f.*, o.text AS ocr_text, a.text AS accessibility_text
                     FROM frames f
                     LEFT JOIN ocr_text             o ON o.frame_id = f.id
                     LEFT JOIN frame_accessibility  a ON a.frame_id = f.id
                    ORDER BY f.timestamp DESC LIMIT ? OFFSET ?""",
                (limit, offset),
            ).fetchall()

    def frame_by_id(self, frame_id: int) -> dict[str, Any] | None:
        with self._lock:
            return self._conn.execute(
                """SELECT f.*, o.text AS ocr_text, o.text_json AS ocr_text_json,
                          o.ocr_engine, a.text AS accessibility_text, a.tree_json,
                          vc.file_path AS video_file_path, vc.fps AS video_fps
                     FROM frames f
                     LEFT JOIN ocr_text             o ON o.frame_id = f.id
                     LEFT JOIN frame_accessibility  a ON a.frame_id = f.id
                     LEFT JOIN video_chunks         vc ON vc.id = f.video_chunk_id
                    WHERE f.id = ?""",
                (frame_id,),
            ).fetchone()

    def mark_frame_image_redacted(self, frame_id: int) -> None:
        with self._lock:
            self._conn.execute("UPDATE frames SET image_redacted = 1 WHERE id = ?", (frame_id,))

    def recent_events(self, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM ui_events ORDER BY timestamp DESC LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()

    def recent_transcripts(self, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM audio_transcriptions ORDER BY timestamp DESC LIMIT ? OFFSET ?", (limit, offset)
            ).fetchall()

    # ─── tags / memories ─────────────────────────────────────────────────────
    def set_frame_tags(self, frame_id: int, tags: list[str]) -> list[str]:
        clean = [t.strip() for t in tags if t and t.strip()]
        with self._lock:
            for tag in clean:
                self._conn.execute("INSERT OR IGNORE INTO tags(name) VALUES (?)", (tag,))
                row = self._conn.execute("SELECT id FROM tags WHERE name = ?", (tag,)).fetchone()
                if row:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO frame_tags(frame_id, tag_id) VALUES (?, ?)",
                        (frame_id, row["id"]),
                    )
        return self.frame_tags(frame_id)

    def remove_frame_tags(self, frame_id: int, tags: list[str]) -> list[str]:
        with self._lock:
            for tag in tags:
                self._conn.execute(
                    """DELETE FROM frame_tags
                         WHERE frame_id = ?
                           AND tag_id IN (SELECT id FROM tags WHERE name = ?)""",
                    (frame_id, tag),
                )
        return self.frame_tags(frame_id)

    def frame_tags(self, frame_id: int) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT t.name
                     FROM tags t
                     JOIN frame_tags ft ON ft.tag_id = t.id
                    WHERE ft.frame_id = ?
                    ORDER BY t.name""",
                (frame_id,),
            ).fetchall()
        return [r["name"] for r in rows]

    def tag_batch(self, frame_ids: list[int]) -> dict[int, list[str]]:
        return {frame_id: self.frame_tags(frame_id) for frame_id in frame_ids}

    def create_memory(self, content: str, *, frame_id: int | None = None, sync_id: str | None = None) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO memories(content, frame_id, sync_id) VALUES (?, ?, ?)",
                (content, frame_id, sync_id),
            )
            return int(cur.lastrowid)

    def list_memories(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM memories ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()

    def memory_by_id(self, memory_id: int) -> dict[str, Any] | None:
        with self._lock:
            return self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()

    def update_memory(self, memory_id: int, content: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE memories SET content = ? WHERE id = ?", (content, memory_id))

    def delete_memory(self, memory_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

    def pending_audio_chunks(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            return self._conn.execute(
                """SELECT * FROM audio_chunks
                    WHERE processing_status = 'pending'
                    ORDER BY timestamp ASC LIMIT ?""",
                (limit,),
            ).fetchall()

    # ─── pipe executions ─────────────────────────────────────────────────────
    def insert_pipe_execution(
        self,
        *,
        pipe_name: str,
        status: str,
        output: str = "",
        session_path: str | None = None,
        started_at: str | None = None,
        ended_at: str | None = None,
    ) -> int:
        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO pipe_executions
                   (pipe_name, session_path, status, output, started_at, ended_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (pipe_name, session_path, status, output, started_at or _now_iso(), ended_at),
            )
            return int(cur.lastrowid)

    def finish_pipe_execution(
        self,
        execution_id: int,
        *,
        status: str,
        output: str,
        session_path: str | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """UPDATE pipe_executions
                      SET status = ?, output = ?, session_path = COALESCE(?, session_path),
                          ended_at = ?
                    WHERE id = ?""",
                (status, output, session_path, _now_iso(), execution_id),
            )

    def list_pipe_executions(self, pipe_name: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            if pipe_name:
                return self._conn.execute(
                    """SELECT * FROM pipe_executions
                        WHERE pipe_name = ?
                        ORDER BY id DESC LIMIT ?""",
                    (pipe_name, limit),
                ).fetchall()
            return self._conn.execute(
                "SELECT * FROM pipe_executions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()

    def health(self) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                """SELECT (SELECT COUNT(*) FROM frames) AS frames,
                          (SELECT COUNT(*) FROM ui_events) AS events,
                          (SELECT COUNT(*) FROM audio_transcriptions) AS transcripts,
                          (SELECT timestamp FROM frames               ORDER BY timestamp DESC LIMIT 1) AS last_frame_timestamp,
                          (SELECT timestamp FROM audio_transcriptions ORDER BY timestamp DESC LIMIT 1) AS last_audio_timestamp"""
            ).fetchone()
        return row or {}

    # ─── retention ───────────────────────────────────────────────────────────
    def cleanup(self, *, frame_iso_cutoff: str, audio_iso_cutoff: str) -> dict[str, int]:
        with self._lock:
            n1 = self._conn.execute("DELETE FROM frames                WHERE timestamp < ?", (frame_iso_cutoff,)).rowcount
            n2 = self._conn.execute("DELETE FROM audio_transcriptions  WHERE timestamp < ?", (audio_iso_cutoff,)).rowcount
            self._conn.execute("DELETE FROM frames_full_text          WHERE timestamp < ?", (frame_iso_cutoff,))
            self._conn.execute("DELETE FROM audio_transcriptions_fts  WHERE timestamp < ?", (audio_iso_cutoff,))
        return {"frames": int(n1 or 0), "transcripts": int(n2 or 0)}

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass

    def __enter__(self) -> "DatabaseManager":
        return self

    def __exit__(self, *exc: Iterable[Any]) -> None:
        self.close()
