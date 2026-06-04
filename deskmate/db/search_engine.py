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
    ELEMENT = "element"
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
        role: str | None = None,
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

        elif ct == ContentType.ELEMENT:
            for row in self._search_elements(
                q,
                limit=limit,
                offset=offset,
                start=start_time,
                end=end_time,
                app_name=app_name,
                window_name=window_name,
                role=role,
            ):
                results.append(self._element_result(row))

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

    def _search_elements(
        self,
        query: str,
        *,
        limit: int,
        offset: int,
        start: str | None,
        end: str | None,
        app_name: str | None,
        window_name: str | None,
        role: str | None,
    ) -> list[dict[str, Any]]:
        """Search normalized accessibility elements by role and/or text (P3).

        When ``query`` is given it runs against ``elements_fts`` (name/value);
        ``role`` filters on the structured column. Either may be omitted.
        """
        wheres = ["1=1"]
        args: list[Any] = []
        if query.strip():
            sanitized = sanitize_fts5_query(query)
            if sanitized:
                wheres.append(
                    "e.id IN (SELECT element_id FROM elements_fts "
                    "WHERE elements_fts MATCH ?)"
                )
                args.append(sanitized)
        if role:
            wheres.append("e.role = ?")
            args.append(role)
        if app_name:
            wheres.append("f.app_name LIKE '%' || ? || '%'")
            args.append(app_name)
        if window_name:
            wheres.append("f.window_name LIKE '%' || ? || '%'")
            args.append(window_name)
        if start:
            wheres.append("f.timestamp >= ?")
            args.append(start)
        if end:
            wheres.append("f.timestamp <= ?")
            args.append(end)
        sql = f"""
            SELECT e.id AS element_id, e.frame_id AS frame_id,
                   e.role AS role, e.name AS name, e.value AS value,
                   e.automation_id AS automation_id, e.is_focused AS is_focused,
                   e.bounds AS bounds,
                   f.timestamp AS timestamp,
                   f.app_name AS app_name, f.window_name AS window_name,
                   f.browser_url AS browser_url, f.snapshot_path AS file_path
              FROM elements e
              JOIN frames f ON f.id = e.frame_id
             WHERE {' AND '.join(wheres)}
             ORDER BY f.timestamp DESC
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
    def _element_result(row: dict[str, Any]) -> SearchResult:
        return SearchResult(
            kind=SearchResultKind.ELEMENT,
            timestamp=row.get("timestamp") or "",
            payload={
                "id": row.get("element_id"),
                "element_id": row.get("element_id"),
                "frame_id": row.get("frame_id"),
                "role": row.get("role") or "",
                "name": row.get("name") or "",
                "value": row.get("value") or "",
                "text": row.get("value") or row.get("name") or "",
                "automation_id": row.get("automation_id"),
                "is_focused": bool(row.get("is_focused")),
                "bounds": row.get("bounds"),
                "timestamp": row.get("timestamp") or "",
                "app_name": row.get("app_name") or "",
                "window_name": row.get("window_name") or "",
                "browser_url": row.get("browser_url"),
                "file_path": row.get("file_path") or "",
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

    # ─── semantic / hybrid search ────────────────────────────────────────────
    @staticmethod
    def _result_key(result: SearchResult) -> tuple[str, Any]:
        """Stable identity used to merge keyword and semantic hits."""
        p = result.payload
        if result.kind == SearchResultKind.OCR:
            return ("ocr", p.get("frame_id"))
        if result.kind == SearchResultKind.AUDIO:
            return ("audio", p.get("id") or p.get("audio_chunk_id"))
        if result.kind == SearchResultKind.UI:
            return ("ax", p.get("frame_id"))
        if result.kind == SearchResultKind.INPUT:
            return ("ui", p.get("id"))
        if result.kind == SearchResultKind.ELEMENT:
            return ("element", p.get("element_id"))
        return ("memory", p.get("id"))

    @staticmethod
    def _emb_content_types(ct: ContentType) -> list[str]:
        """Map a :class:`ContentType` onto embedding content-type tags."""
        if ct == ContentType.ALL:
            return ["ocr", "audio", "ui"]
        if ct in (ContentType.OCR, ContentType.ACCESSIBILITY):
            return ["ocr"]
        if ct == ContentType.AUDIO:
            return ["audio"]
        if ct == ContentType.INPUT:
            return ["ui"]
        return []

    def _fetch_ocr_by_ids(
        self, ids: list[int], *, app_name: str | None, window_name: str | None
    ) -> dict[int, dict[str, Any]]:
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        wheres = [f"fft.frame_id IN ({placeholders})"]
        args: list[Any] = list(ids)
        if app_name:
            wheres.append("fft.app_name LIKE '%' || ? || '%'")
            args.append(app_name)
        if window_name:
            wheres.append("fft.window_name LIKE '%' || ? || '%'")
            args.append(window_name)
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
        """
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return {int(r["frame_id"]): r for r in rows}

    def _fetch_audio_by_ids(
        self, ids: list[int], *, speaker_ids: list[int] | None
    ) -> dict[int, dict[str, Any]]:
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        wheres = [f"t.id IN ({placeholders})"]
        args: list[Any] = list(ids)
        if speaker_ids:
            sp = ",".join("?" * len(speaker_ids))
            wheres.append(f"t.speaker_id IN ({sp})")
            args.extend(speaker_ids)
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
                   t.redacted_transcription
              FROM audio_transcriptions t
             WHERE {' AND '.join(wheres)}
        """
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return {int(r["transcription_id"]): r for r in rows}

    def _fetch_ui_by_ids(
        self, ids: list[int], *, app_name: str | None, window_name: str | None
    ) -> dict[int, dict[str, Any]]:
        if not ids:
            return {}
        placeholders = ",".join("?" * len(ids))
        wheres = [f"id IN ({placeholders})"]
        args: list[Any] = list(ids)
        if app_name:
            wheres.append("app_name LIKE '%' || ? || '%'")
            args.append(app_name)
        if window_name:
            wheres.append("window_title LIKE '%' || ? || '%'")
            args.append(window_name)
        sql = f"""
            SELECT id, timestamp, event_type, app_name, window_title,
                   browser_url, frame_id, data_json, element_json
              FROM ui_events
             WHERE {' AND '.join(wheres)}
        """
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return {int(r["id"]): r for r in rows}

    def _load_candidates(
        self,
        emb_type: str,
        *,
        model_name: str,
        start_time: str | None,
        end_time: str | None,
        candidate_pool: int,
    ) -> list[dict[str, Any]]:
        wheres = ["content_type = ?", "model = ?"]
        args: list[Any] = [emb_type, model_name]
        if start_time:
            wheres.append("timestamp >= ?")
            args.append(start_time)
        if end_time:
            wheres.append("timestamp <= ?")
            args.append(end_time)
        sql = f"""
            SELECT content_id, timestamp, embedding
              FROM content_embeddings
             WHERE {' AND '.join(wheres)}
             ORDER BY timestamp DESC
             LIMIT ?
        """
        args.append(candidate_pool)
        with self._lock:
            return self._conn.execute(sql, args).fetchall()

    def semantic_search(
        self,
        query: str,
        content_type: ContentType | str,
        *,
        model_name: str,
        limit: int = 50,
        start_time: str | None = None,
        end_time: str | None = None,
        app_name: str | None = None,
        window_name: str | None = None,
        speaker_ids: list[int] | None = None,
        candidate_pool: int = 5000,
    ) -> list[tuple[SearchResult, float]]:
        """Rank content by embedding cosine similarity to ``query``.

        Returns ``(result, score)`` pairs ordered by descending similarity.
        Returns an empty list if the embedder or numpy is unavailable (the
        caller then falls back to keyword search).
        """
        q = (query or "").strip()
        if not q:
            return []
        ct = content_type if isinstance(content_type, ContentType) else normalize_content_type(str(content_type))
        emb_types = self._emb_content_types(ct)
        if not emb_types:
            return []
        try:
            import numpy as np  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            return []
        from .embeddings import blob_to_vector, get_embedder  # noqa: PLC0415

        embedder = get_embedder(model_name)
        if embedder is None:
            return []
        qvec = embedder.embed_one(q)
        if not qvec:
            return []
        qarr = np.asarray(qvec, dtype=np.float32)
        qnorm = float(np.linalg.norm(qarr)) or 1.0
        qarr = qarr / qnorm

        scored: list[tuple[str, int, str, float]] = []  # (emb_type, id, ts, score)
        for emb_type in emb_types:
            rows = self._load_candidates(
                emb_type,
                model_name=model_name,
                start_time=start_time,
                end_time=end_time,
                candidate_pool=candidate_pool,
            )
            if not rows:
                continue
            mat = np.array([blob_to_vector(r["embedding"]) for r in rows], dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1)
            norms[norms == 0] = 1.0
            sims = (mat @ qarr) / norms
            for r, sim in zip(rows, sims):
                scored.append((emb_type, int(r["content_id"]), r["timestamp"], float(sim)))

        if not scored:
            return []
        scored.sort(key=lambda x: x[3], reverse=True)
        top = scored[: max(limit, 1)]

        # Group ids by type, fetch full rows, then rebuild in score order.
        ids_by_type: dict[str, list[int]] = {"ocr": [], "audio": [], "ui": []}
        for emb_type, cid, _ts, _score in top:
            ids_by_type[emb_type].append(cid)
        ocr_rows = self._fetch_ocr_by_ids(ids_by_type["ocr"], app_name=app_name, window_name=window_name)
        audio_rows = self._fetch_audio_by_ids(ids_by_type["audio"], speaker_ids=speaker_ids)
        ui_rows = self._fetch_ui_by_ids(ids_by_type["ui"], app_name=app_name, window_name=window_name)

        out: list[tuple[SearchResult, float]] = []
        for emb_type, cid, _ts, score in top:
            if emb_type == "ocr":
                row = ocr_rows.get(cid)
                if row is not None:
                    out.append((self._ocr_result(row), score))
            elif emb_type == "audio":
                row = audio_rows.get(cid)
                if row is not None:
                    out.append((self._audio_result(row), score))
            else:
                row = ui_rows.get(cid)
                if row is not None:
                    out.append((self._input_result(row), score))
        return out

    def hybrid_search(
        self,
        query: str,
        content_type: ContentType | str,
        *,
        model_name: str,
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
        rrf_k: int = 60,
        candidate_pool: int = 5000,
    ) -> list[SearchResult]:
        """Combine keyword (FTS5/BM25) and semantic hits via Reciprocal Rank
        Fusion.

        Neither leg is sufficient on its own: keyword search nails exact terms
        and identifiers but misses paraphrases, while embeddings capture meaning
        but can drift on rare tokens. RRF fuses the two ranked lists using only
        rank position (``score = Σ 1/(k + rank)``), which sidesteps the fact
        that BM25 and cosine scores live on incomparable scales.

        Falls back to the keyword list when semantic search is unavailable or
        the query is empty.
        """
        pool = max((limit + offset) * 4, limit, 20)
        fts_results = self.search(
            query,
            content_type,
            limit=pool,
            offset=0,
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
        semantic_results = self.semantic_search(
            query,
            content_type,
            model_name=model_name,
            limit=pool,
            start_time=start_time,
            end_time=end_time,
            app_name=app_name,
            window_name=window_name,
            speaker_ids=speaker_ids,
            candidate_pool=candidate_pool,
        )
        if not semantic_results:
            return fts_results[offset : offset + limit]

        scores: dict[tuple[str, Any], float] = {}
        holder: dict[tuple[str, Any], SearchResult] = {}
        for rank, result in enumerate(fts_results):
            key = self._result_key(result)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            holder.setdefault(key, result)
        for rank, (result, _sim) in enumerate(semantic_results):
            key = self._result_key(result)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            holder.setdefault(key, result)

        ordered = sorted(holder.values(), key=lambda r: scores[self._result_key(r)], reverse=True)
        return ordered[offset : offset + limit]
