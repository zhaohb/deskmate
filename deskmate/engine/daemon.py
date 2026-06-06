"""Long-running orchestrator for capture, audio, retention and automations."""

from __future__ import annotations

import queue
import threading
from datetime import datetime, timedelta, timezone

from .. import events as bus
from .. import paths
from ..a11y import UiRecorder
from ..a11y.ui_event_types import UiEventInsert
from ..audio import (
    AudioRecorder,
    SpeakerIdentifier,
    TranscriptTranslator,
    WhisperTranscriber,
)
from ..capture import PairedCapture
from ..capture.event_driven_capture import run_event_driven_capture_loop
from ..capture.frame_linker import FrameLinkerActor
from ..capture.ui_event_pipeline import UiEventPipeline
from ..config import Config, load as load_config
from ..db import DatabaseManager
from ..habits import HabitWatcher
from ..fusion import ContextFusionBus, capture_allowed, shutdown_gate
from ..logger import get
from ..meeting import MeetingDetector
from ..pipes import PipeRuntime, PipeScheduler, load_pipes
from ..redact import OnnxRedactor, RedactReconciler
from .app_scheduler import AppScheduler

logger = get("engine.daemon")

# Maps the user-facing latency/quality preset onto the endpoint silence gap (ms).
# Larger gap => waits for a fuller sentence => higher latency, better translation.
_LATENCY_SILENCE_MS = {"fast": 400, "balanced": 700, "quality": 1000}


def _endpoint_silence_ms(audio) -> int:
    """Resolve the endpoint pause threshold from the translate latency preset,
    falling back to the explicit ``endpoint_silence_ms`` for unknown presets."""
    return _LATENCY_SILENCE_MS.get(
        getattr(audio, "translate_latency_mode", "balanced"),
        audio.endpoint_silence_ms,
    )


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
            chunk_mode=self.cfg.audio.chunk_mode,
            endpoint_silence_ms=_endpoint_silence_ms(self.cfg.audio),
            endpoint_max_chunk_s=self.cfg.audio.endpoint_max_chunk_s,
            endpoint_min_chunk_s=self.cfg.audio.endpoint_min_chunk_s,
        ) if self.cfg.audio.enabled else None
        self.transcriber = WhisperTranscriber(
            self.cfg.audio.whisper_model,
            self.cfg.audio.device,
            vad_threshold=self.cfg.audio.vad_threshold,
            vad_min_segment_ms=self.cfg.audio.vad_min_segment_ms,
            vad_padding_ms=self.cfg.audio.vad_padding_ms,
            compute_type=self.cfg.audio.compute_type,
            languages=self.cfg.audio.languages,
            backend=self.cfg.audio.whisper_backend,
            openvino_genai_model=self.cfg.audio.openvino_genai_model,
            openvino_device=self.cfg.audio.openvino_device,
            openvino_cache_dir=self.cfg.audio.openvino_cache_dir or str(paths.ov_cache_dir()),
        ) if self.cfg.audio.enabled else None
        self.speaker = SpeakerIdentifier() if self.cfg.audio.enabled and self.cfg.audio.speaker_recognition else None

        # Live translation (opt-in): translate each transcript utterance via the
        # local LLM on a background worker so it never blocks the audio loop.
        self.translator = (
            TranscriptTranslator(
                target_lang=self.cfg.audio.translate_target_lang,
                model=self.cfg.audio.translate_model,
                skip_if_same=self.cfg.audio.translate_skip_if_same,
                context_window=self.cfg.audio.translate_context_window,
            )
            if self.cfg.audio.enabled and self.cfg.audio.translate_enabled
            else None
        )
        # (transcript_id, text, source_lang, device) jobs for the translate worker.
        self._translate_queue: "queue.Queue[tuple[int, str, str | None, str]]" = queue.Queue(maxsize=256)

        onnx_path = getattr(self.cfg.redact, "onnx_model_path", None)
        tok_path = getattr(self.cfg.redact, "onnx_tokenizer_path", None)
        self.redactor = OnnxRedactor(
            onnx_path,
            tokenizer_path=tok_path,
        ) if onnx_path else None
        self.reconciler = RedactReconciler(self.db, self.redactor) if self.redactor else None

        pipes_dir = paths.config_dir() / "pipes"
        self.pipes = load_pipes(pipes_dir)
        self.pipe_runtime = PipeRuntime(self.db, self.cfg)
        self.pipe_scheduler = PipeScheduler(self.db, self.pipes, runtime=self.pipe_runtime) if self.pipes else None
        self.app_scheduler = AppScheduler()

        # Additive: learn routines + proactive suggestions. Owns its own thread
        # and DB connection; only constructed when explicitly enabled.
        self.habit_watcher = (
            HabitWatcher(self.cfg)
            if getattr(self.cfg, "habits", None) and self.cfg.habits.enabled
            else None
        )

        # Additive: fuse all sources into the unified context timeline. Owns its
        # own thread + DB connection and subscribes to the in-process event bus,
        # so no producer is modified.
        self.fusion_bus = (
            ContextFusionBus(self.cfg)
            if getattr(self.cfg, "fusion", None) and self.cfg.fusion.enabled
            else None
        )

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._translate_thread: threading.Thread | None = None

    def set_translation(
        self,
        *,
        enabled: bool | None = None,
        target_lang: str | None = None,
        latency_mode: str | None = None,
    ) -> dict[str, object]:
        """Hot-toggle / reconfigure live translation without a restart.

        Builds (or tears down) the ``TranscriptTranslator`` and ensures the
        background translate worker is running when enabled. Safe to call from
        the API thread. Returns the resulting effective settings."""
        if target_lang is not None:
            self.cfg.audio.translate_target_lang = target_lang
        if latency_mode is not None:
            self.cfg.audio.translate_latency_mode = latency_mode

        want = self.cfg.audio.translate_enabled if enabled is None else enabled
        self.cfg.audio.translate_enabled = want

        if not want:
            # Leave the worker thread alive but idle; dropping the translator
            # makes both the enqueue check and the worker no-op.
            self.translator = None
        else:
            self.translator = TranscriptTranslator(
                target_lang=self.cfg.audio.translate_target_lang,
                model=self.cfg.audio.translate_model,
                skip_if_same=self.cfg.audio.translate_skip_if_same,
                context_window=self.cfg.audio.translate_context_window,
            )
            self._ensure_translate_worker()
        return {
            "translate_enabled": want,
            "translate_target_lang": self.cfg.audio.translate_target_lang,
            "translate_latency_mode": self.cfg.audio.translate_latency_mode,
        }

    def _ensure_translate_worker(self) -> None:
        """Start the translate worker thread if it isn't already running."""
        t = self._translate_thread
        if t is not None and t.is_alive():
            return
        if self._stop.is_set():
            return
        t = threading.Thread(target=self._translate_loop, name="daemon-translate", daemon=True)
        self._translate_thread = t
        t.start()

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

        if self.habit_watcher:
            self.habit_watcher.start()

        if self.fusion_bus:
            self.fusion_bus.start()

        # Auto-summarize a meeting the moment it ends: the detector emits
        # MEETING_ENDED, and we fire the meeting-summary app in the background.
        self._meeting_unsub = bus.subscribe(self._on_bus_event)

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
        # Live translation worker (its own tracked thread so it can be toggled at
        # runtime via set_translation without restarting the daemon).
        if self.translator is not None:
            self._ensure_translate_worker()
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
        unsub = getattr(self, "_meeting_unsub", None)
        if callable(unsub):
            unsub()
        if self.habit_watcher:
            self.habit_watcher.stop()
        if self.fusion_bus:
            self.fusion_bus.stop()
        shutdown_gate()
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

    def _on_bus_event(self, event: "bus.Event") -> None:
        """React to in-process events. Currently: auto-summarize ended meetings."""
        try:
            if event.type == bus.EventType.MEETING_ENDED:
                meeting_id = event.data.get("meeting_id")
                if meeting_id is not None:
                    self._fire_meeting_summary(int(meeting_id))
                # Drop the translator's rolling context so the next conversation
                # doesn't inherit stale context from the meeting that just ended.
                if self.translator is not None:
                    self.translator.reset_context()
        except Exception as exc:  # noqa: BLE001
            logger.debug("bus event handler error: %s", exc)

    def _fire_meeting_summary(self, meeting_id: int) -> None:
        """Launch the meeting-summary app for a just-ended meeting, detached.

        Runs in a daemon thread so the (Ollama-backed) summary never blocks the
        capture pipeline. The app scopes itself to ``--meeting-id`` and writes
        the summary + extracted todos back to the DB."""
        import subprocess  # noqa: PLC0415
        import sys  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        app_py = Path(__file__).resolve().parents[2] / "apps" / "meeting-summary" / "app.py"
        if not app_py.is_file():
            logger.debug("meeting-summary app not found at %s", app_py)
            return

        def _run() -> None:
            try:
                logger.info("auto meeting-summary firing for meeting %s", meeting_id)
                proc = subprocess.run(  # noqa: S603
                    [sys.executable, str(app_py), "--meeting-id", str(meeting_id)],
                    capture_output=True, text=True, timeout=900, check=False,
                )
                if proc.returncode == 0:
                    logger.info("auto meeting-summary ok for meeting %s", meeting_id)
                else:
                    logger.warning(
                        "auto meeting-summary meeting %s exit=%d stderr=%s",
                        meeting_id, proc.returncode, (proc.stderr or "")[-400:],
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("auto meeting-summary failed for %s: %s", meeting_id, exc)

        threading.Thread(target=_run, name=f"meeting-summary-{meeting_id}", daemon=True).start()

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
            # Additive capture control: skip transcription while audio capture is
            # paused or its source switch is off (fail-open if control unavailable).
            if not capture_allowed("audio"):
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
                    # Live translation: hand off to the background worker (never
                    # block the transcribe loop on an LLM call). Drop silently if
                    # the queue is saturated — translation is best-effort.
                    if self.translator is not None and seg.text.strip():
                        try:
                            self._translate_queue.put_nowait(
                                (tid, seg.text, seg.language, label)
                            )
                        except queue.Full:
                            logger.debug("translate queue full; skipping transcript %s", tid)
                self.db.mark_audio_chunk_status(chunk_id, "processed")
            except Exception as exc:  # noqa: BLE001
                logger.warning("transcribe loop err: %s", exc)

    def _translate_loop(self) -> None:
        """Background worker: translate queued transcript utterances via the LLM.

        Runs while live translation is enabled. Each job is translated with the
        per-device context window, the result is back-filled into the
        ``audio_transcriptions`` row, and a ``TRANSCRIPT_TRANSLATED`` event is
        emitted so the UI updates the matching line in real time. The translator
        reference is read per-iteration so a runtime enable/disable toggle takes
        effect immediately (jobs are dropped while disabled)."""
        from .. import events as bus  # noqa: PLC0415

        while not self._stop.is_set():
            try:
                job = self._translate_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            translator = self.translator
            if translator is None:
                continue  # disabled at runtime; drop the job
            tid, text, source_lang, device = job
            try:
                translation = translator.translate(
                    text, source_lang=source_lang, device=device
                )
                if not translation:
                    continue
                self.db.set_transcript_translation(
                    tid, translation, self.cfg.audio.translate_target_lang
                )
                bus.send(
                    bus.EventType.TRANSCRIPT_TRANSLATED,
                    transcript_id=tid,
                    device=device,
                    translation=translation[:500],
                    lang=self.cfg.audio.translate_target_lang,
                    final=True,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("translate worker err for transcript %s: %s", tid, exc)

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
