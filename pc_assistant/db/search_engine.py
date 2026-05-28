"""Full-text search over frames, audio, UI events and memories."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from .search_types import ContentType, normalize_content_type
from .text_normalizer import sanitize_fts5_query, value_to_fts5_column_query


ResultKind = Literal["ocr", "audio", "ui", "input", "memory"]


class SearchResultKind(str, Enum):
    OCR = "ocr"
    AUDIO = "audio"
    UI = "ui"
    INPUT = "input"
    MEMORY = "memory"


@dataclass
class SearchResult:
    kind: SearchResultKind
    timestamp: str
    payload: dict[str, Any]


def _parse_ts(value: str | None) -> datetime:
    if not value:
        return datetime.min
    compact = str(value)
    m = __import__("re").match(r"^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})", compact)
    if m:
        compact = (
            f"{m.group(1)}-{m.group(2)}-{m.group(3)}T"
            f"{m.group(4)}:{m.group(5)}:{m.group(6)}"
        )
    try:
        return datetime.fromisoformat(compact.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min


def _build_frame_fts_query(
    query: str,
    *,
    app_name: str | None,
    window_name: str | None,
    browser_url: str | None = None,
) -> tuple[str, bool]:
    """Assemble FTS MATCH clauses for OCR and accessibility text search."""
    parts: list[str] = []
    if app_name:
        parts.append(value_to_fts5_column_query("app_name", app_name))
    if window_name:
        parts.append(value_to_fts5_column_query("window_name", window_name))
    if browser_url:
        parts.append(value_to_fts5_column_query("browser_url", browser_url))
    if query.strip():
        sanitized = sanitize_fts5_query(query)
        if sanitized:
            parts.append(sanitized)
    fts_query = " ".join(parts)
    return fts_query, bool(fts_query.strip())


class SearchEngine:
    def __init__(self, conn: Any, lock: Any) -> None:
        self._conn = conn
        self._lock = lock

    def search(
        self,
        query: str,
        content_type: ContentType | str,
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
    ) -> list[SearchResult]:
        ct = content_type if isinstance(content_type, ContentType) else normalize_content_type(str(content_type))
        q = query or ""
        results: list[SearchResult] = []

        # browser_url / focused filters restrict to OCR-only.
        if browser_url or focused is not None:
            ct = ContentType.OCR

        if ct == ContentType.ALL:
            fetch_limit = limit + offset
            has_frame_filters = bool(app_name or window_name or frame_name)
            ocr_rows = self._search_ocr(
                q,
                limit=fetch_limit,
                offset=0,
                start=start_time,
                end=end_time,
                app_name=app_name,
                window_name=window_name,
                frame_name=frame_name,
                browser_url=browser_url,
                focused=focused,
                min_length=min_length,
                max_length=max_length,
            )
            for row in ocr_rows:
                results.append(self._ocr_result(row))

            if not has_frame_filters:
                for row in self._search_audio(
                    q,
                    limit=fetch_limit,
                    offset=0,
                    start=start_time,
                    end=end_time,
                    min_length=min_length,
                    max_length=max_length,
                    speaker_ids=speaker_ids,
                ):
                    results.append(self._audio_result(row))

            for row in self._search_accessibility(
                q,
                limit=fetch_limit,
                offset=0,
                start=start_time,
                end=end_time,
                app_name=app_name,
                window_name=window_name,
            ):
                results.append(self._accessibility_result(row))

        elif ct == ContentType.OCR:
            for row in self._search_ocr(
                q,
                limit=limit,
                offset=offset,
                start=start_time,
                end=end_time,
                app_name=app_name,
                window_name=window_name,
                frame_name=frame_name,
                browser_url=browser_url,
                focused=focused,
                min_length=min_length,
                max_length=max_length,
            ):
                results.append(self._ocr_result(row))

        elif ct == ContentType.AUDIO:
            if not (app_name or window_name):
                for row in self._search_audio(
                    q,
                    limit=limit,
                    offset=offset,
                    start=start_time,
                    end=end_time,
                    min_length=min_length,
                    max_length=max_length,
                    speaker_ids=speaker_ids,
                ):
                    results.append(self._audio_result(row))

        elif ct == ContentType.ACCESSIBILITY:
            for row in self._search_accessibility(
                q,
                limit=limit,
                offset=offset,
                start=start_time,
                end=end_time,
                app_name=app_name,
                window_name=window_name,
            ):
                results.append(self._accessibility_result(row))

        elif ct == ContentType.INPUT:
            for row in self._search_input(
                q,
                limit=limit,
                offset=offset,
                start=start_time,
                end=end_time,
                app_name=app_name,
                window_name=window_name,
            ):
                results.append(self._input_result(row))

        elif ct == ContentType.MEMORY:
            for row in self._search_memory(
                q,
                limit=limit,
                offset=offset,
                start=start_time,
                end=end_time,
            ):
                results.append(self._memory_result(row))

        results.sort(key=lambda r: _parse_ts(r.timestamp), reverse=True)
        if ct == ContentType.ALL:
            results = results[offset : offset + limit]
        return results

    def _search_ocr(
        self,
        query: str,
        *,
        limit: int,
        offset: int,
        start: str | None,
        end: str | None,
        app_name: str | None,
        window_name: str | None,
        frame_name: str | None,
        browser_url: str | None,
        focused: bool | None,
        min_length: int | None,
        max_length: int | None,
    ) -> list[dict[str, Any]]:
        fts_query, has_fts = _build_frame_fts_query(
            query, app_name=app_name, window_name=window_name, browser_url=browser_url
        )
        wheres = ["1=1"]
        args: list[Any] = []
        if has_fts:
            wheres.append("frames_full_text MATCH ?")
            args.append(fts_query)
        if start:
            wheres.append("fft.timestamp >= ?")
            args.append(start)
        if end:
            wheres.append("fft.timestamp <= ?")
            args.append(end)
        if focused is not None:
            wheres.append("f.focused = ?")
            args.append(1 if focused else 0)
        if frame_name:
            wheres.append("f.name LIKE '%' || ? || '%'")
            args.append(frame_name)
        if min_length is not None:
            wheres.append(
                "LENGTH(COALESCE(fft.ocr_text, '') || COALESCE(fft.accessibility_text, '')) >= ?"
            )
            args.append(min_length)
        if max_length is not None:
            wheres.append(
                "LENGTH(COALESCE(fft.ocr_text, '') || COALESCE(fft.accessibility_text, '')) <= ?"
            )
            args.append(max_length)
        sql = f"""
            SELECT fft.frame_id AS frame_id,
                   fft.timestamp AS timestamp,
                   fft.app_name AS app_name,
                   fft.window_name AS window_name,
                   fft.browser_url AS browser_url,
                   fft.ocr_text AS ocr_text,
                   fft.accessibility_text AS accessibility_text,
                   f.snapshot_path AS snapshot_path,
                   f.focused AS focused,
                   COALESCE(NULLIF(fft.accessibility_text, ''), fft.ocr_text, '') AS full_text
              FROM frames_full_text fft
              JOIN frames f ON f.id = fft.frame_id
             WHERE {' AND '.join(wheres)}
             ORDER BY fft.timestamp DESC
             LIMIT ? OFFSET ?
        """
        args.extend([limit, offset])
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    def _search_accessibility(
        self,
        query: str,
        *,
        limit: int,
        offset: int,
        start: str | None,
        end: str | None,
        app_name: str | None,
        window_name: str | None,
    ) -> list[dict[str, Any]]:
        fts_query, has_fts = _build_frame_fts_query(query, app_name=app_name, window_name=window_name)
        wheres = [
            "COALESCE(fft.accessibility_text, '') != ''",
        ]
        args: list[Any] = []
        if has_fts:
            wheres.append("frames_full_text MATCH ?")
            args.append(fts_query)
        if start:
            wheres.append("fft.timestamp >= ?")
            args.append(start)
        if end:
            wheres.append("fft.timestamp <= ?")
            args.append(end)
        sql = f"""
            SELECT fft.frame_id AS frame_id,
                   fft.timestamp AS timestamp,
                   fft.app_name AS app_name,
                   fft.window_name AS window_name,
                   fft.browser_url AS browser_url,
                   fft.accessibility_text AS text_output,
                   f.snapshot_path AS file_path,
                   fft.accessibility_text AS full_text
              FROM frames_full_text fft
              JOIN frames f ON f.id = fft.frame_id
             WHERE {' AND '.join(wheres)}
             ORDER BY fft.timestamp DESC
             LIMIT ? OFFSET ?
        """
        args.extend([limit, offset])
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    def _search_audio(
        self,
        query: str,
        *,
        limit: int,
        offset: int,
        start: str | None,
        end: str | None,
        min_length: int | None,
        max_length: int | None,
        speaker_ids: list[int] | None,
    ) -> list[dict[str, Any]]:
        wheres: list[str] = []
        args: list[Any] = []
        if query.strip():
            wheres.append(
                """t.audio_chunk_id IN (
                    SELECT at_inner.audio_chunk_id
                      FROM audio_transcriptions_fts
                      JOIN audio_transcriptions at_inner
                        ON at_inner.id = audio_transcriptions_fts.transcription_id
                     WHERE audio_transcriptions_fts MATCH ?
                     ORDER BY bm25(audio_transcriptions_fts)
                     LIMIT 5000
                )"""
            )
            args.append(sanitize_fts5_query(query))
        if start:
            wheres.append("t.timestamp >= ?")
            args.append(start)
        if end:
            wheres.append("t.timestamp <= ?")
            args.append(end)
        if min_length is not None:
            wheres.append("COALESCE(t.text_length, LENGTH(t.transcription)) >= ?")
            args.append(min_length)
        if max_length is not None:
            wheres.append("COALESCE(t.text_length, LENGTH(t.transcription)) <= ?")
            args.append(max_length)
        if speaker_ids:
            placeholders = ",".join("?" * len(speaker_ids))
            wheres.append(f"t.speaker_id IN ({placeholders})")
            args.extend(speaker_ids)
        where_clause = f"WHERE {' AND '.join(wheres)}" if wheres else ""
        sql = f"""
            SELECT t.id AS transcription_id,
                   t.audio_chunk_id,
                   t.offset_index,
                   t.timestamp,
                   t.transcription,
                   t.device,
                   t.language,
                   t.speaker_id,
                   t.start_time,
                   t.end_time,
                   t.text_length,
                   t.redacted_transcription,
                   '' AS snippet
              FROM audio_transcriptions t
              LEFT JOIN audio_chunks c ON c.id = t.audio_chunk_id
              {where_clause}
             GROUP BY t.audio_chunk_id, t.offset_index
             ORDER BY t.timestamp DESC
             LIMIT ? OFFSET ?
        """
        args.extend([limit, offset])
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    def _search_input(
        self,
        query: str,
        *,
        limit: int,
        offset: int,
        start: str | None,
        end: str | None,
        app_name: str | None,
        window_name: str | None,
    ) -> list[dict[str, Any]]:
        """Search UI events with LIKE on text, app, window (not FTS)."""
        if not query.strip():
            return []
        wheres = ["1=1"]
        args: list[Any] = []
        wheres.append(
            "(COALESCE(json_extract(data_json, '$.content'), '') LIKE '%' || ? || '%' "
            "OR COALESCE(app_name, '') LIKE '%' || ? || '%' "
            "OR COALESCE(window_title, '') LIKE '%' || ? || '%')"
        )
        args.extend([query, query, query])
        if app_name:
            wheres.append("app_name LIKE '%' || ? || '%'")
            args.append(app_name)
        if window_name:
            wheres.append("window_title LIKE '%' || ? || '%'")
            args.append(window_name)
        if start:
            wheres.append("timestamp >= ?")
            args.append(start)
        if end:
            wheres.append("timestamp <= ?")
            args.append(end)
        sql = f"""
            SELECT id, timestamp, event_type, app_name, window_title,
                   browser_url, frame_id, data_json, element_json
              FROM ui_events
             WHERE {' AND '.join(wheres)}
             ORDER BY timestamp DESC
             LIMIT ? OFFSET ?
        """
        args.extend([limit, offset])
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    def _search_memory(
        self,
        query: str,
        *,
        limit: int,
        offset: int,
        start: str | None,
        end: str | None,
    ) -> list[dict[str, Any]]:
        wheres = ["1=1"]
        args: list[Any] = []
        if query.strip():
            wheres.append("content LIKE '%' || ? || '%'")
            args.append(query)
        if start:
            wheres.append("created_at >= ?")
            args.append(start)
        if end:
            wheres.append("created_at <= ?")
            args.append(end)
        sql = f"""
            SELECT * FROM memories
             WHERE {' AND '.join(wheres)}
             ORDER BY id DESC
             LIMIT ? OFFSET ?
        """
        args.extend([limit, offset])
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    @staticmethod
    def _ocr_result(row: dict[str, Any]) -> SearchResult:
        text = row.get("full_text") or row.get("ocr_text") or row.get("accessibility_text") or ""
        return SearchResult(
            kind=SearchResultKind.OCR,
            timestamp=row["timestamp"],
            payload={
                "frame_id": row["frame_id"],
                "text": text,
                "ocr_text": row.get("ocr_text") or "",
                "accessibility_text": row.get("accessibility_text") or "",
                "timestamp": row["timestamp"],
                "file_path": row.get("snapshot_path") or "",
                "app_name": row.get("app_name") or "",
                "window_name": row.get("window_name") or "",
                "browser_url": row.get("browser_url"),
                "focused": row.get("focused"),
            },
        )

    @staticmethod
    def _accessibility_result(row: dict[str, Any]) -> SearchResult:
        text = row.get("text_output") or row.get("full_text") or ""
        return SearchResult(
            kind=SearchResultKind.UI,
            timestamp=row["timestamp"],
            payload={
                "id": row["frame_id"],
                "frame_id": row["frame_id"],
                "text": text,
                "timestamp": row["timestamp"],
                "app_name": row.get("app_name") or "",
                "window_name": row.get("window_name") or "",
                "browser_url": row.get("browser_url"),
                "file_path": row.get("file_path") or "",
            },
        )

    @staticmethod
    def _audio_result(row: dict[str, Any]) -> SearchResult:
        transcription = row.get("transcription") or ""
        return SearchResult(
            kind=SearchResultKind.AUDIO,
            timestamp=row["timestamp"],
            payload={
                "id": row.get("transcription_id"),
                "audio_chunk_id": row.get("audio_chunk_id"),
                "transcription": transcription,
                "timestamp": row["timestamp"],
                "device": row.get("device") or "",
                "device_name": row.get("device") or "",
                "language": row.get("language"),
                "speaker_id": row.get("speaker_id"),
                "start_time": row.get("start_time"),
                "end_time": row.get("end_time"),
                "text_length": row.get("text_length"),
                "redacted_transcription": row.get("redacted_transcription"),
                "offset_index": row.get("offset_index", 0),
            },
        )

    @staticmethod
    def _input_result(row: dict[str, Any]) -> SearchResult:
        data = row.get("data_json")
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                data = {}
        text_content = (data or {}).get("content") if isinstance(data, dict) else None
        return SearchResult(
            kind=SearchResultKind.INPUT,
            timestamp=row["timestamp"],
            payload={
                "id": row["id"],
                "timestamp": row["timestamp"],
                "event_type": row.get("event_type"),
                "app_name": row.get("app_name"),
                "window_title": row.get("window_title"),
                "browser_url": row.get("browser_url"),
                "text_content": text_content,
                "frame_id": row.get("frame_id"),
                "data": data,
                "element": row.get("element_json"),
            },
        )

    @staticmethod
    def _memory_result(row: dict[str, Any]) -> SearchResult:
        return SearchResult(
            kind=SearchResultKind.MEMORY,
            timestamp=row.get("created_at") or "",
            payload={
                "id": row["id"],
                "content": row["content"],
                "created_at": row.get("created_at"),
                "frame_id": row.get("frame_id"),
            },
        )
