"""Build and maintain the semantic (vector) index.

Keyword search (FTS5) and semantic search use different indexes. FTS5 is kept
up to date inline as content is written; the vector index is built here, either
as a one-time *backfill* of existing rows or *incrementally* as new content
arrives. Each indexed row stores one normalized embedding in
``content_embeddings`` keyed by ``(content_type, content_id, model)`` so
re-runs are idempotent and switching models re-indexes cleanly.

If the embedding model can't load (missing optional dependency), indexing is a
no-op and the system continues with keyword-only search.
"""

from __future__ import annotations

import sqlite3
import threading
from typing import Any, Callable

from ..logger import get
from .embeddings import EmbeddingModel, get_embedder, vector_to_blob

logger = get("semantic_index")

# (content_type, SQL selecting pending rows as (content_id, timestamp, text))
# `:model` and `:min_chars` and `:limit` are bound by name.
_PENDING_SQL: dict[str, str] = {
    "ocr": """
        SELECT fft.frame_id AS content_id,
               fft.timestamp AS timestamp,
               COALESCE(NULLIF(TRIM(fft.accessibility_text), ''), fft.ocr_text) AS text
          FROM frames_full_text fft
          LEFT JOIN content_embeddings ce
            ON ce.content_type = 'ocr'
           AND ce.content_id = fft.frame_id
           AND ce.model = :model
         WHERE ce.id IS NULL
           AND LENGTH(COALESCE(NULLIF(TRIM(fft.accessibility_text), ''), fft.ocr_text)) >= :min_chars
         ORDER BY fft.timestamp DESC
         LIMIT :limit
    """,
    "audio": """
        SELECT t.id AS content_id,
               t.timestamp AS timestamp,
               t.transcription AS text
          FROM audio_transcriptions t
          LEFT JOIN content_embeddings ce
            ON ce.content_type = 'audio'
           AND ce.content_id = t.id
           AND ce.model = :model
         WHERE ce.id IS NULL
           AND LENGTH(COALESCE(t.transcription, '')) >= :min_chars
         ORDER BY t.timestamp DESC
         LIMIT :limit
    """,
    "ui": """
        SELECT e.id AS content_id,
               e.timestamp AS timestamp,
               json_extract(e.data_json, '$.content') AS text
          FROM ui_events e
          LEFT JOIN content_embeddings ce
            ON ce.content_type = 'ui'
           AND ce.content_id = e.id
           AND ce.model = :model
         WHERE ce.id IS NULL
           AND e.event_type IN ('text', 'clipboard')
           AND LENGTH(COALESCE(json_extract(e.data_json, '$.content'), '')) >= :min_chars
         ORDER BY e.timestamp DESC
         LIMIT :limit
    """,
}

CONTENT_TYPES = tuple(_PENDING_SQL.keys())


class SemanticIndexer:
    """Embeds unindexed text content into ``content_embeddings``."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        lock: threading.RLock,
        *,
        model_name: str,
        batch_size: int = 64,
        min_chars: int = 12,
        embedder: EmbeddingModel | None = None,
    ) -> None:
        self._conn = conn
        self._lock = lock
        self.model_name = model_name
        self.batch_size = max(1, batch_size)
        self.min_chars = max(1, min_chars)
        self._embedder = embedder

    def _get_embedder(self) -> EmbeddingModel | None:
        if self._embedder is None:
            self._embedder = get_embedder(self.model_name)
        return self._embedder

    def pending_count(self, content_type: str | None = None) -> int:
        """Number of rows that still need an embedding."""
        types = (content_type,) if content_type else CONTENT_TYPES
        total = 0
        for ct in types:
            sql = _PENDING_SQL[ct].replace("LIMIT :limit", "")
            wrapped = f"SELECT COUNT(*) AS n FROM ({sql})"
            with self._lock:
                row = self._conn.execute(
                    wrapped, {"model": self.model_name, "min_chars": self.min_chars}
                ).fetchone()
            total += int(row["n"] if isinstance(row, dict) else row[0])
        return total

    def _fetch_pending(self, content_type: str, limit: int) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                _PENDING_SQL[content_type],
                {"model": self.model_name, "min_chars": self.min_chars, "limit": limit},
            ).fetchall()
        return [r for r in rows if (r.get("text") or "").strip()]

    def _store(self, content_type: str, rows: list[dict[str, Any]], vectors: list[list[float]]) -> None:
        dim = len(vectors[0]) if vectors else 0
        params = [
            (
                content_type,
                int(row["content_id"]),
                self.model_name,
                dim,
                vector_to_blob(vec),
                row["timestamp"],
            )
            for row, vec in zip(rows, vectors)
        ]
        with self._lock:
            self._conn.executemany(
                """INSERT OR REPLACE INTO content_embeddings
                   (content_type, content_id, model, dim, embedding, timestamp)
                   VALUES (?,?,?,?,?,?)""",
                params,
            )

    def index_pending(
        self,
        content_type: str | None = None,
        *,
        max_rows: int | None = None,
        progress: Callable[[str, int], None] | None = None,
    ) -> int:
        """Embed pending rows. Returns the number of rows indexed.

        ``max_rows`` caps the total embedded this call (``None`` = until drained).
        """
        embedder = self._get_embedder()
        if embedder is None:
            logger.info("semantic indexing skipped: embedder unavailable")
            return 0

        types = (content_type,) if content_type else CONTENT_TYPES
        indexed = 0
        for ct in types:
            while True:
                remaining = None if max_rows is None else max_rows - indexed
                if remaining is not None and remaining <= 0:
                    return indexed
                limit = self.batch_size if remaining is None else min(self.batch_size, remaining)
                rows = self._fetch_pending(ct, limit)
                if not rows:
                    break
                vectors = embedder.embed([r["text"] for r in rows])
                if vectors is None:
                    logger.warning("embedding batch failed; aborting indexing for %s", ct)
                    break
                self._store(ct, rows, vectors)
                indexed += len(rows)
                if progress is not None:
                    progress(ct, indexed)
                if len(rows) < limit:
                    break
        return indexed

    def backfill(self, progress: Callable[[str, int], None] | None = None) -> int:
        """Index every pending row across all content types."""
        return self.index_pending(progress=progress)
