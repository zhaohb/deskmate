"""Async redaction reconciler.

Re-scans recently-stored OCR / accessibility / transcription text with the
heavier ONNX detector and writes results into the `redacted_*` columns.

Design:
- One worker thread, sleeps between batches.
- Per pass: pull up to `batch_size` frames whose `redacted_text` is NULL
  but whose `text` is non-empty, redact them, UPDATE in one go.
- Same logic for `audio_transcriptions.redacted_transcription`.

Idempotent: re-running the worker on a populated DB is a no-op.
"""

from __future__ import annotations

import threading
import time

from ..logger import get
from .onnx import OnnxRedactor

logger = get("redact.reconciler")


class RedactReconciler:
    def __init__(
        self,
        db,
        redactor: OnnxRedactor,
        *,
        interval_seconds: float = 30.0,
        batch_size: int = 100,
    ) -> None:
        self.db = db
        self.redactor = redactor
        self.interval_seconds = interval_seconds
        self.batch_size = batch_size
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread:
            return
        if not self.redactor.available:
            logger.info("ONNX redactor unavailable — reconciler disabled")
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="RedactReconciler", daemon=True)
        self._thread.start()
        logger.info("redact reconciler started (interval=%.1fs)", self.interval_seconds)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                processed = self._step()
                if processed == 0:
                    self._stop.wait(self.interval_seconds)
                else:
                    self._stop.wait(0.5)
            except Exception as exc:  # noqa: BLE001
                logger.warning("reconciler step failed: %s", exc)
                self._stop.wait(self.interval_seconds)

    def _step(self) -> int:
        processed = 0
        processed += self._reconcile_ocr()
        processed += self._reconcile_accessibility()
        processed += self._reconcile_transcripts()
        return processed

    def _reconcile_ocr(self) -> int:
        rows = self.db._conn.execute(  # noqa: SLF001
            """SELECT frame_id, text FROM ocr_text
                WHERE text <> '' AND redacted_text IS NULL
                LIMIT ?""",
            (self.batch_size,),
        ).fetchall()
        for row in rows:
            new = self.redactor.redact(row["text"])
            self.db._conn.execute(  # noqa: SLF001
                "UPDATE ocr_text SET redacted_text = ? WHERE frame_id = ?",
                (new, row["frame_id"]),
            )
        if rows:
            logger.debug("redact ocr: %d rows", len(rows))
        return len(rows)

    def _reconcile_accessibility(self) -> int:
        rows = self.db._conn.execute(  # noqa: SLF001
            """SELECT frame_id, text FROM frame_accessibility
                WHERE text <> '' AND redacted_text IS NULL
                LIMIT ?""",
            (self.batch_size,),
        ).fetchall()
        for row in rows:
            new = self.redactor.redact(row["text"])
            self.db._conn.execute(  # noqa: SLF001
                "UPDATE frame_accessibility SET redacted_text = ? WHERE frame_id = ?",
                (new, row["frame_id"]),
            )
        if rows:
            logger.debug("redact a11y: %d rows", len(rows))
        return len(rows)

    def _reconcile_transcripts(self) -> int:
        rows = self.db._conn.execute(  # noqa: SLF001
            """SELECT id, transcription FROM audio_transcriptions
                WHERE transcription <> '' AND redacted_transcription IS NULL
                LIMIT ?""",
            (self.batch_size,),
        ).fetchall()
        for row in rows:
            new = self.redactor.redact(row["transcription"])
            self.db._conn.execute(  # noqa: SLF001
                "UPDATE audio_transcriptions SET redacted_transcription = ? WHERE id = ?",
                (new, row["id"]),
            )
        if rows:
            logger.debug("redact transcripts: %d rows", len(rows))
        return len(rows)
