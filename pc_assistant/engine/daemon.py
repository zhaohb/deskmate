"""Long-running orchestrator for capture, audio, retention and automations."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from .. import events as bus
from .. import paths
from ..a11y import UiRecorder
from ..a11y.activity_feed import default as activity_default
from ..a11y.browser_url import resolve_browser_url
from ..audio import AudioRecorder, SpeakerIdentifier, WhisperTranscriber
from ..capture import PairedCapture
from ..config import Config, load as load_config
from ..db import DatabaseManager
from ..logger import get
from ..meeting import MeetingDetector
from ..pipes import PipeRuntime, PipeScheduler, load_pipes
from ..redact import OnnxRedactor, RedactReconciler

logger = get("engine.daemon")


class Daemon:
    def __init__(self, cfg: Config | None = None, db: DatabaseManager | None = None) -> None:
        self.cfg = cfg or load_config()
        self.db = db or DatabaseManager()
        self.paired = PairedCapture(self.cfg, self.db)
        self.ui = UiRecorder(self.cfg.a11y, on_event=self._on_a11y_event)
        self.meeting_detector = MeetingDetector(self.db)
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
        ) if self.cfg.audio.enabled else None
        self.speaker = SpeakerIdentifier() if self.cfg.audio.enabled and self.cfg.audio.speaker_recognition else None

        # ONNX-backed redact reconciler (no-op if no model configured).
        onnx_path = getattr(self.cfg.redact, "onnx_model_path", None)
        tok_path = getattr(self.cfg.redact, "onnx_tokenizer_path", None)
        self.redactor = OnnxRedactor(onnx_path, tokenizer_path=tok_path) if onnx_path else None
        self.reconciler = RedactReconciler(self.db, self.redactor) if self.redactor else None

        # Pipes (optional; folder may not exist).
        pipes_dir = paths.config_dir() / "pipes"
        self.pipes = load_pipes(pipes_dir)
        self.pipe_runtime = PipeRuntime(self.db, self.cfg)
        self.pipe_scheduler = PipeScheduler(self.db, self.pipes, runtime=self.pipe_runtime) if self.pipes else None

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._last_event_capture: float = 0.0

    # ─── lifecycle ─────────────────────────────────────────────────────────
    def start(self) -> None:
        paths.ensure_dirs()
        logger.info("daemon starting cfg=%s", paths.config_path())
        self.ui.start()
        if self.audio:
            self.audio.start()
        if self.reconciler:
            self.reconciler.start()
        if self.pipe_scheduler:
            self.pipe_scheduler.start()
        self._threads = [
            threading.Thread(target=self._heartbeat_loop, name="daemon-heartbeat", daemon=True),
            threading.Thread(target=self._audio_loop, name="daemon-audio", daemon=True),
            threading.Thread(target=self._retention_loop, name="daemon-retention", daemon=True),
        ]
        for t in self._threads:
            t.start()
        logger.info("daemon started")

    def stop(self) -> None:
        logger.info("daemon stopping")
        self._stop.set()
        self.ui.stop()
        if self.audio:
            self.audio.stop()
        if self.reconciler:
            self.reconciler.stop()
        if self.pipe_scheduler:
            self.pipe_scheduler.stop()
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

    # ─── event-driven capture ──────────────────────────────────────────────
    def _on_a11y_event(self, ev: dict[str, Any]) -> None:
        """Callback installed on UiRecorder. Triggers a paired capture on
        focus/title/value changes, throttled by `min_capture_gap_seconds`."""
        self._persist_ui_event(ev)
        now = time.time()
        if now - self._last_event_capture < self.cfg.capture.debounce_seconds:
            return
        self._last_event_capture = now
        try:
            frame_ids = self.paired.capture_once(trigger=ev.get("event_type") or "event", trigger_data=ev)
            self._observe_frame_ids(frame_ids)
        except Exception as exc:  # noqa: BLE001
            logger.warning("event capture failed: %s", exc)

    def _persist_ui_event(self, ev: dict[str, Any]) -> None:
        """Persist raw UI/input/clipboard events even when capture is throttled."""
        raw_type = str(ev.get("event_type") or "event")
        app_name = ev.get("app_name")
        window_title = ev.get("window_title")
        browser_url = resolve_browser_url(
            app_name or "",
            pid=int(ev.get("pid") or 0),
            hwnd=int(ev.get("hwnd") or 0),
        )
        data: dict[str, Any]
        event_type = raw_type

        if raw_type == "key_text":
            event_type = "text"
            text = ev.get("text") or ""
            data = {"content": text, "char_count": len(text)}
        elif raw_type == "clipboard":
            event_type = "clipboard"
            text = ev.get("text") or ""
            data = {"operation": "c", "content": text, "length": ev.get("length", len(text))}
        elif raw_type == "click":
            data = {
                "x": ev.get("x"),
                "y": ev.get("y"),
                "button": ev.get("button"),
                "click_count": ev.get("click_count", 1),
            }
        else:
            data = {k: v for k, v in ev.items() if k not in {"event_type", "app_name", "window_title"}}

        self.meeting_detector.observe(
            app_name=app_name or "",
            window_title=window_title or "",
            browser_url=browser_url,
            text=str(data.get("content") or ""),
        )

        try:
            self.db.insert_ui_event(
                event_type=event_type,
                app_name=app_name,
                window_title=window_title,
                browser_url=browser_url,
                data=data,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("ui event persist failed: %s", exc)

    def _observe_frame_ids(self, frame_ids: list[int]) -> None:
        if not frame_ids:
            self.meeting_detector.expire_if_idle()
            return
        try:
            row = self.db.frame_by_id(frame_ids[0])
            if not row:
                return
            self.meeting_detector.observe(
                app_name=row.get("app_name") or "",
                window_title=row.get("window_name") or "",
                browser_url=row.get("browser_url"),
                text="\n".join((
                    row.get("accessibility_text") or "",
                    row.get("ocr_text") or "",
                )),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("meeting observation failed: %s", exc)

    def _heartbeat_loop(self) -> None:
        """Adaptive heartbeat. Asks the ActivityFeed each iteration what the
        recommended interval is (busy typing → fast, idle → slow), capped by
        `cfg.capture.heartbeat_seconds` as a hard floor."""
        floor_ms = max(5, int(self.cfg.capture.heartbeat_seconds)) * 1000
        feed = activity_default()
        while not self._stop.is_set():
            try:
                frame_ids = self.paired.capture_once(trigger="heartbeat")
                self._observe_frame_ids(frame_ids)
            except Exception as exc:  # noqa: BLE001
                logger.warning("heartbeat capture failed: %s", exc)
            params = feed.get_capture_params()
            # honor user-configured floor (don't shoot below it)
            interval_ms = max(params.interval_ms, floor_ms) if self.cfg.capture.adaptive_fps_floor \
                else params.interval_ms
            self._stop.wait(interval_ms / 1000.0)
            self.meeting_detector.expire_if_idle()

    def _audio_loop(self) -> None:
        """Pull audio chunks → register them in `audio_chunks` → run
        VAD+Whisper segmentation → one `audio_transcriptions` row per
        segment, with optional speaker identification."""
        if not self.audio or not self.transcriber:
            return
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
            self._stop.wait(60 * 60)  # hourly
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
