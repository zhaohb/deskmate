"""Long-running orchestrator for capture, audio, retention and automations."""

from __future__ import annotations

import queue
import threading
from datetime import datetime, timedelta, timezone
from typing import Any

from .. import paths
from ..a11y import UiRecorder
from ..a11y.ui_event_types import UiEventInsert
from ..audio import AudioRecorder, SpeakerIdentifier, WhisperTranscriber
from ..capture import PairedCapture
from ..capture.event_driven_capture import run_event_driven_capture_loop
from ..capture.frame_linker import FrameLinkerActor
from ..capture.ui_event_pipeline import UiEventPipeline
from ..config import Config, load as load_config
from ..db import DatabaseManager
from ..logger import get
from ..meeting import MeetingDetector
from ..pipes import PipeRuntime, PipeScheduler, load_pipes
from ..redact import OnnxRedactor, RedactReconciler
from .app_scheduler import AppScheduler

logger = get("engine.daemon")


class Daemon:
    def __init__(self, cfg: Config | None = None, db: DatabaseManager | None = None) -> None:
        self.cfg = cfg or load_config()
        self.db = db or DatabaseManager()
        self.paired = PairedCapture(self.cfg, self.db)
        self.meeting_detector = MeetingDetector(self.db)

        self._trigger_queue: queue.Queue = queue.Queue(maxsize=4096)
        self._linker = FrameLinkerActor(self.db)
        self._ui_pipeline = UiEventPipeline(
            self.cfg,
            self.db,
            trigger_queue=self._trigger_queue,
            linker=self._linker,
            on_meeting_observe=self._meeting_observe_insert,
        )
        self.ui = UiRecorder(self.cfg.a11y, on_event=self._ui_pipeline.handle_event)

        self.audio = AudioRecorder(
            sample_rate=self.cfg.audio.sample_rate,
            chunk_seconds=self.cfg.audio.chunk_seconds,
            microphone=self.cfg.audio.microphone,
            loopback=self.cfg.audio.loopback,
        ) if self.cfg.audio.enabled else None
        self.transcriber = WhisperTranscriber(
            self.cfg.audio.whisper_model,
            self.cfg.audio.device,
            vad_threshold=self.cfg.audio.vad_threshold,
            vad_min_segment_ms=self.cfg.audio.vad_min_segment_ms,
            vad_padding_ms=self.cfg.audio.vad_padding_ms,
            compute_type=self.cfg.audio.compute_type,
            languages=self.cfg.audio.languages,
        ) if self.cfg.audio.enabled else None
        self.speaker = SpeakerIdentifier() if self.cfg.audio.enabled and self.cfg.audio.speaker_recognition else None

        onnx_path = getattr(self.cfg.redact, "onnx_model_path", None)
        tok_path = getattr(self.cfg.redact, "onnx_tokenizer_path", None)
        self.redactor = OnnxRedactor(onnx_path, tokenizer_path=tok_path) if onnx_path else None
        self.reconciler = RedactReconciler(self.db, self.redactor) if self.redactor else None

        pipes_dir = paths.config_dir() / "pipes"
        self.pipes = load_pipes(pipes_dir)
        self.pipe_runtime = PipeRuntime(self.db, self.cfg)
        self.pipe_scheduler = PipeScheduler(self.db, self.pipes, runtime=self.pipe_runtime) if self.pipes else None
        self.app_scheduler = AppScheduler()

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def _meeting_observe_insert(self, insert: UiEventInsert) -> None:
        self.meeting_detector.observe(
            app_name=insert.app_name or "",
            window_title=insert.window_title or "",
            browser_url=insert.browser_url,
            text=insert.text_content or "",
        )

    def start(self) -> None:
        paths.ensure_dirs()
        logger.info("daemon starting cfg=%s", paths.config_path())
        self._linker.start()
        self._ui_pipeline.start()
        self.ui.start()
        if self.audio:
            self.audio.start()
        if self.reconciler:
            self.reconciler.start()
        if self.pipe_scheduler:
            self.pipe_scheduler.start()
        self.app_scheduler.start()

        self._threads = [
            threading.Thread(target=self._audio_loop, name="daemon-audio", daemon=True),
            threading.Thread(target=self._retention_loop, name="daemon-retention", daemon=True),
        ]
        if self.cfg.search.semantic_enabled and self.cfg.search.auto_index:
            self._threads.append(
                threading.Thread(target=self._semantic_index_loop, name="daemon-semantic-index", daemon=True),
            )
        if self.cfg.capture.event_driven and self.cfg.capture.enabled:
            self._threads.append(
                threading.Thread(
                    target=lambda: run_event_driven_capture_loop(
                        cfg=self.cfg,
                        db=self.db,
                        paired=self.paired,
                        trigger_rx=self._trigger_queue,
                        linker=self._linker,
                        stop=self._stop,
                        meeting_observe=self._meeting_observe_frame,
                    ),
                    name="event-driven-capture",
                    daemon=True,
                ),
            )
        else:
            self._threads.append(
                threading.Thread(target=self._heartbeat_loop, name="daemon-heartbeat", daemon=True),
            )
        for t in self._threads:
            t.start()
        logger.info("daemon started (event_driven=%s)", self.cfg.capture.event_driven)

    def stop(self) -> None:
        logger.info("daemon stopping")
        self._stop.set()
        self._ui_pipeline.stop()
        self.ui.stop()
        self._linker.stop()
        if self.audio:
            self.audio.stop()
        if self.reconciler:
            self.reconciler.stop()
        if self.pipe_scheduler:
            self.pipe_scheduler.stop()
        self.app_scheduler.stop()
        for t in self._threads:
            t.join(timeout=3.0)
        self.db.close()
        logger.info("daemon stopped")

    def run_forever(self) -> None:
        self.start()
        try:
            while not self._stop.is_set():
                self._stop.wait(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def _meeting_observe_frame(
        self,
        *,
        app_name: str,
        window_title: str,
        browser_url: str | None,
        text: str,
    ) -> None:
        self.meeting_detector.observe(
            app_name=app_name,
            window_title=window_title,
            browser_url=browser_url,
            text=text,
        )

    def _heartbeat_loop(self) -> None:
        """Legacy polling path when `capture.event_driven=false`."""
        from ..a11y.activity_feed import default as activity_default  # noqa: PLC0415

        floor_ms = max(5, int(self.cfg.capture.heartbeat_seconds)) * 1000
        feed = activity_default()
        while not self._stop.is_set():
            try:
                frame_ids = self.paired.capture_once(trigger="idle")
                if frame_ids:
                    self._meeting_observe_frame_from_id(frame_ids[0])
            except Exception as exc:  # noqa: BLE001
                logger.warning("heartbeat capture failed: %s", exc)
            params = feed.get_capture_params()
            interval_ms = max(params.interval_ms, floor_ms) if self.cfg.capture.adaptive_fps_floor else params.interval_ms
            self._stop.wait(interval_ms / 1000.0)
            self.meeting_detector.expire_if_idle()

    def _meeting_observe_frame_from_id(self, frame_id: int) -> None:
        try:
            row = self.db.frame_by_id(frame_id)
            if not row:
                return
            self._meeting_observe_frame(
                app_name=row.get("app_name") or "",
                window_title=row.get("window_name") or "",
                browser_url=row.get("browser_url"),
                text="\n".join(filter(None, [row.get("accessibility_text"), row.get("ocr_text")])),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("meeting observation failed: %s", exc)

    def _audio_loop(self) -> None:
        if not self.audio or not self.transcriber:
            return
        from .. import events as bus  # noqa: PLC0415

        while not self._stop.is_set():
            chunk = self.audio.take_next_chunk(timeout=1.0)
            if chunk is None:
                continue
            label, path, duration_ms = chunk
            try:
                chunk_id = self.db.insert_audio_chunk(
                    file_path=str(path), device_name=label, duration_ms=duration_ms,
                )
                segments = self.transcriber.transcribe_segments(path)
                if not segments:
                    self.db.mark_audio_chunk_status(chunk_id, "processed")
                    continue
                for idx, seg in enumerate(segments):
                    speaker_id: int | None = None
                    emb: list[float] = []
                    if self.speaker:
                        try:
                            emb = self.speaker.embed(path, seg.start_time, seg.end_time)
                            if emb:
                                speaker_id = self.speaker.match_or_create(self.db, emb)
                        except Exception as exc:  # noqa: BLE001
                            logger.debug("speaker embed failed: %s", exc)
                    tid = self.db.insert_transcript(
                        device=label, text=seg.text, language=seg.language,
                        audio_chunk_id=chunk_id, offset_index=idx,
                        start_time=seg.start_time, end_time=seg.end_time,
                        speaker_id=speaker_id,
                    )
                    if speaker_id is not None and emb:
                        self.db.insert_speaker_embedding(
                            speaker_id=speaker_id,
                            embedding=emb,
                            audio_chunk_id=chunk_id,
                            transcription_id=tid,
                        )
                    self.meeting_detector.link_transcript(
                        transcription_id=tid,
                        speaker_id=speaker_id,
                        text=seg.text,
                        start_time=seg.start_time,
                        end_time=seg.end_time,
                    )
                    bus.send(
                        bus.EventType.AUDIO_TRANSCRIBED,
                        transcript_id=tid, device=label, text=seg.text[:200],
                        speaker_id=speaker_id,
                    )
                self.db.mark_audio_chunk_status(chunk_id, "processed")
            except Exception as exc:  # noqa: BLE001
                logger.warning("transcribe loop err: %s", exc)

    def _retention_loop(self) -> None:
        while not self._stop.is_set():
            self._stop.wait(60 * 60)
            if self._stop.is_set():
                break
            try:
                now = datetime.now(timezone.utc).astimezone()
                frame_cut = (now - timedelta(days=self.cfg.retention.frame_days)).replace(microsecond=0).isoformat()
                audio_cut = (now - timedelta(days=self.cfg.retention.audio_days)).replace(microsecond=0).isoformat()
                removed = self.db.cleanup(frame_iso_cutoff=frame_cut, audio_iso_cutoff=audio_cut)
                logger.info("retention sweep: %s", removed)
            except Exception as exc:  # noqa: BLE001
                logger.warning("retention err: %s", exc)

    def _semantic_index_loop(self) -> None:
        """Incrementally embed new text content for semantic search."""
        # Initial delay so startup isn't competing with the embedding download.
        self._stop.wait(30)
        model = self.cfg.search.embedding_model
        while not self._stop.is_set():
            try:
                indexed = self.db.build_semantic_index(
                    model_name=model,
                    batch_size=self.cfg.search.index_batch,
                    min_chars=self.cfg.search.min_chars,
                    max_rows=self.cfg.search.candidate_pool,
                )
                if indexed:
                    logger.info("semantic index: embedded %d new item(s)", indexed)
            except Exception as exc:  # noqa: BLE001
                logger.warning("semantic index err: %s", exc)
            self._stop.wait(5 * 60)
