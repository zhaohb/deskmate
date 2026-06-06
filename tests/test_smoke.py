"""Cross-platform smoke tests. They only exercise pure-Python paths so they
pass on Linux/CI even though deskmate is primarily a Windows tool."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


def test_imports() -> None:
    import deskmate  # noqa: F401
    from deskmate import (  # noqa: F401
        config, core, db, events, paths, redact, screen,
    )


def test_pii_redact() -> None:
    from deskmate.core import find_pii_spans, remove_pii

    masked = remove_pii("contact me at alice@example.com or +1-555-123-4567")
    assert "alice@example.com" not in masked
    assert "[REDACTED]" in masked
    spans = find_pii_spans("alice@example.com")
    assert spans and spans[0].rule == "email"


def test_window_filter() -> None:
    from deskmate.core.filter import WindowFilter, is_app_excluded

    assert is_app_excluded("1Password", ["1password"])
    f = WindowFilter(ignored_windows=["Cursor::Private"], included_windows=["Chrome::"])
    assert f.passes("chrome", "Some tab")
    assert not f.passes("cursor", "Private dialog")


def test_incognito() -> None:
    from deskmate.core import is_title_private

    assert is_title_private("Google — Private Browsing")
    assert is_title_private("无痕窗口")
    assert not is_title_private("Normal window")


def test_browser_url_helpers() -> None:
    from deskmate.a11y.browser_url import is_browser_app, normalize_url_text

    assert is_browser_app("chrome.exe")
    assert is_browser_app("msedge.exe")
    assert not is_browser_app("notepad.exe")
    assert normalize_url_text("example.com/path") == "https://example.com/path"
    assert normalize_url_text("https://example.com/path") == "https://example.com/path"
    assert normalize_url_text("search terms") is None
    assert normalize_url_text("chrome://settings") is None


def test_event_bus() -> None:
    from deskmate import events as bus

    captured: list = []
    unsub = bus.subscribe(captured.append)
    bus.send(bus.EventType.CLICK, x=1, y=2)
    unsub()
    assert captured and captured[0].type is bus.EventType.CLICK
    assert captured[0].data["x"] == 1


def test_win_event_payload_shape() -> None:
    # WinEvent callbacks should send `event_type` only to UiRecorder callback,
    # not through bus.send kwargs (avoids duplicate argument collision).
    bus_payload = {
        "hwnd": 1,
        "pid": 2,
        "app_name": "chrome.exe",
        "window_title": "Demo",
    }
    callback_payload = {"event_type": "window_focus", **bus_payload}
    assert "event_type" not in bus_payload
    assert callback_payload["event_type"] == "window_focus"


def test_db_roundtrip() -> None:
    from deskmate.db import DatabaseManager

    with tempfile.TemporaryDirectory() as tmp:
        db = DatabaseManager(Path(tmp) / "test.db")
        video_id = db.insert_video_chunk(file_path=str(Path(tmp) / "screen.mp4"), device_name="screen", fps=1.0)
        assert db.video_chunk_by_id(video_id)["file_path"].endswith("screen.mp4")
        assert db.list_video_chunks()[0]["id"] == video_id
        fid = db.insert_frame(
            monitor_id=1, device_name="m1", app_name="Cursor", window_name="paired.py",
            browser_url="https://example.com/docs", focused=True, snapshot_path=None, width=0, height=0,
            capture_trigger="manual",
        )
        db.attach_ocr(fid, text="hello kubernetes world", text_json="[]", engine="tesseract", confidence=0.9)
        db.attach_accessibility(fid, text="visible body", focused_role="Edit", focused_name="msg",
                                focused_value="draft", tree_json=json.dumps({"x": 1}))
        out = db.search_frames("kubernetes")
        assert out and out[0]["frame_id"] == fid
        assert out[0]["browser_url"] == "https://example.com/docs"
        assert db.set_frame_tags(fid, ["work", "docs"]) == ["docs", "work"]
        assert db.remove_frame_tags(fid, ["docs"]) == ["work"]
        memory_id = db.create_memory("remember this", frame_id=fid)
        assert db.memory_by_id(memory_id)["content"] == "remember this"
        db.update_memory(memory_id, "updated")
        assert db.list_memories()[0]["content"] == "updated"
        h = db.health()
        assert h["frames"] == 1
        assert db.schema_version() is not None
        db.close()


def test_config_load_writes_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    from deskmate.config import load

    cfg = load()
    assert (tmp_path / "config.toml").exists()
    assert cfg.server.port == 3030
    assert cfg.ollama.model == "qwen3_8b_ov:v1"


def test_ollama_resolve_from_config_and_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    (tmp_path / "config.toml").write_text(
        '[ollama]\nmodel = "from-toml:tag"\nbase = "http://127.0.0.1:11435"\n',
        encoding="utf-8",
    )
    from deskmate.engine import llm

    base, model, _timeout = llm.resolve_ollama_settings()
    assert model == "from-toml:tag"
    assert base == "http://127.0.0.1:11435"

    monkeypatch.setenv("OLLAMA_MODEL", "from-env:tag")
    base2, model2, _timeout2 = llm.resolve_ollama_settings()
    assert model2 == "from-env:tag"
    assert base2 == "http://127.0.0.1:11435"


def test_activity_feed_curve() -> None:
    from deskmate.a11y.activity_feed import ActivityFeed, ActivityKind

    feed = ActivityFeed()
    feed.record(ActivityKind.KEY_PRESS)
    feed.record(ActivityKind.KEY_PRESS)
    feed.record(ActivityKind.KEY_PRESS)
    params = feed.get_capture_params()
    # keyboard burst (3+ presses within 500ms) → 200ms / 5 FPS
    assert params.interval_ms == 200
    assert params.skip_threshold == 0.02


def test_ocr_output_shape() -> None:
    """OCR shape check: values must be **strings**, not floats."""
    from deskmate.screen.ocr import OcrEngine, perform_ocr
    from PIL import Image

    img = Image.new("RGB", (10, 10), color="white")
    # OFF engine returns empty but well-formed tuple.
    text, jstr, conf = perform_ocr(img, OcrEngine.OFF)
    assert text == "" and jstr == "[]" and conf is None


def test_tesseract_language_mapping() -> None:
    from deskmate.screen.ocr import _tesseract_languages

    assert _tesseract_languages(["en-US"]) == ["eng"]
    assert _tesseract_languages(["zh-CN"]) == ["chi_sim"]
    assert _tesseract_languages(["chi_sim"]) == ["chi_sim"]


def test_winrt_language_mapping() -> None:
    from deskmate.screen.ocr import _winrt_languages

    assert _winrt_languages(["en-US"]) == ["en-US"]
    assert _winrt_languages(["zh-CN"]) == ["zh-Hans"]
    assert _winrt_languages(["chi_sim"]) == ["zh-Hans"]


def test_rapidocr_maps_result_to_normalized_words(monkeypatch) -> None:
    """RapidOCR boxes/txts/scores → stringified, 0–1 normalized word JSON."""
    import json

    import deskmate.screen.ocr as ocr
    from PIL import Image

    class FakeResult:
        # one 100x20 box at (50,10) on a 200x100 image, plus a blank line
        boxes = [
            [[50, 10], [150, 10], [150, 30], [50, 30]],
            [[0, 0], [10, 0], [10, 10], [0, 10]],
        ]
        txts = (" hello ", "  ")  # second is blank → dropped
        scores = (0.95, 0.4)

    # Bypass the real engine/singleton.
    monkeypatch.setattr(ocr, "_rapidocr_engine", lambda: (lambda _arr: FakeResult()))
    ocr._RAPIDOCR_UNAVAILABLE = False

    img = Image.new("RGB", (200, 100), "white")
    text, jstr, conf = ocr._rapidocr(img)

    assert text == "hello"
    words = json.loads(jstr)
    assert len(words) == 1
    w = words[0]
    assert w["text"] == "hello"
    # normalized: left 50/200=0.25, top 10/100=0.1, width 100/200=0.5, height 20/100=0.2
    assert abs(float(w["left"]) - 0.25) < 1e-6
    assert abs(float(w["top"]) - 0.10) < 1e-6
    assert abs(float(w["width"]) - 0.50) < 1e-6
    assert abs(float(w["height"]) - 0.20) < 1e-6
    assert w["conf"] == "0.95"  # stringified
    assert conf == 0.95


def test_rapidocr_unavailable_returns_none(monkeypatch) -> None:
    """When the engine can't load, _rapidocr returns None so perform_ocr falls back."""
    import deskmate.screen.ocr as ocr
    from PIL import Image

    monkeypatch.setattr(ocr, "_rapidocr_engine", lambda: None)
    ocr._RAPIDOCR_UNAVAILABLE = False
    assert ocr._rapidocr(Image.new("RGB", (10, 10), "white")) is None


def test_downscale_only_shrinks() -> None:
    from deskmate.screen.capture import downscale
    from PIL import Image

    big = Image.new("RGB", (3840, 2160), "white")
    out = downscale(big, 1920)
    assert out.size == (1920, 1080)
    # no upscale / no-op when already small or max_width=0
    small = Image.new("RGB", (800, 600), "white")
    assert downscale(small, 1920).size == (800, 600)
    assert downscale(small, 0).size == (800, 600)


def test_pyaudio_stream_cleanup_tolerates_closed_stream() -> None:
    from deskmate.audio.capture import _close_pyaudio_stream

    class ClosedStream:
        def __init__(self) -> None:
            self.closed = False

        def stop_stream(self) -> None:
            raise OSError("Stream not open")

        def close(self) -> None:
            self.closed = True

    stream = ClosedStream()
    _close_pyaudio_stream(stream)
    assert stream.closed is True


def test_transcriber_offsets_vad_segments() -> None:
    from deskmate.audio.transcribe import WhisperTranscriber
    from deskmate.audio.transcribe_backends import RawSegment, TranscribeResult
    from deskmate.audio.vad import SpeechSegment

    class FakeBackend:
        """Stand-in TranscriptionBackend recording the vad_filter it received."""

        name = "fake"

        def __init__(self) -> None:
            self.vad_filters: list[bool] = []

        def transcribe(self, _path: str, *, vad_filter: bool) -> TranscribeResult:
            self.vad_filters.append(vad_filter)
            return TranscribeResult(
                segments=[RawSegment(text=" hello ", start=0.25, end=0.75)],
                detected_language="en",
            )

    transcriber = WhisperTranscriber.__new__(WhisperTranscriber)
    transcriber.languages = ["zh"]
    transcriber._available = True
    transcriber._backend = FakeBackend()
    transcriber._speech_segments = lambda _path: [SpeechSegment(start_s=10.0, end_s=12.0)]
    transcriber._load_clip_source = lambda _source: None
    transcriber._write_clip = lambda source, _segment, _target, *, clip_source=None: source

    out = transcriber.transcribe_segments(Path("fake.wav"))
    assert len(out) == 1
    assert out[0].text == "hello"
    assert out[0].start_time == 10.25
    assert out[0].end_time == 10.75
    # Silero already segmented, so the engine's own VAD must be off for clips.
    assert transcriber._backend.vad_filters == [False]


def test_faster_whisper_backend_builds_transcribe_kwargs() -> None:
    """The onnx_cpu backend forces zh + initial prompt and task=transcribe."""
    from deskmate.audio.transcribe_backends import FasterWhisperBackend, ZH_INITIAL_PROMPT

    class Segment:
        text = " ni hao "
        start = 0.1
        end = 0.5

    class Info:
        language = "zh"

    class Model:
        def transcribe(self, _path: str, **kwargs):
            self.kwargs = kwargs
            return [Segment()], Info()

    backend = FasterWhisperBackend.__new__(FasterWhisperBackend)
    backend.languages = ["zh"]
    backend._model = Model()

    result = backend.transcribe("clip.wav", vad_filter=False)
    assert backend._model.kwargs["task"] == "transcribe"
    assert backend._model.kwargs["vad_filter"] is False
    assert backend._model.kwargs["language"] == "zh"
    assert backend._model.kwargs["initial_prompt"] == ZH_INITIAL_PROMPT
    assert result.detected_language == "zh"
    assert result.segments[0].text == " ni hao "


def test_genai_backend_parses_chunks(monkeypatch) -> None:
    """The openvino_genai backend maps WhisperPipeline result.chunks to RawSegments."""
    import deskmate.audio.transcribe_backends as tb
    from deskmate.audio.transcribe_backends import WhisperGenAIBackend

    class Chunk:
        def __init__(self, text, start, end):
            self.text, self.start_ts, self.end_ts = text, start, end

    class Result:
        chunks = [Chunk(" hello ", 0.0, 1.0), Chunk("", 1.0, 1.2)]

    class Pipeline:
        def generate(self, raw, **kwargs):
            self.raw, self.kwargs = raw, kwargs
            return Result()

    # GenAI takes a raw audio array; stub the file read.
    monkeypatch.setattr(tb, "_read_audio_16k", lambda _p: [0.0, 0.1, 0.2])

    backend = WhisperGenAIBackend.__new__(WhisperGenAIBackend)
    backend.languages = ["zh"]
    backend._model = Pipeline()

    result = backend.transcribe("clip.wav", vad_filter=True)
    # zh forces the special-token language form + return_timestamps. We must NOT
    # send initial_prompt: it crashes the NPU pipeline ("roi_end <= max_dim").
    assert backend._model.kwargs["task"] == "transcribe"
    assert backend._model.kwargs["language"] == "<|zh|>"
    assert backend._model.kwargs["return_timestamps"] is True
    assert "initial_prompt" not in backend._model.kwargs
    assert result.detected_language == "zh"
    assert [s.text for s in result.segments] == [" hello ", ""]
    assert result.segments[0].end == 1.0


def test_backend_fallback_chain() -> None:
    """openvino_genai that fails to load falls back to onnx_cpu."""
    from deskmate.audio.transcribe import WhisperTranscriber
    from deskmate.audio.transcribe_backends import LoadStatus

    transcriber = WhisperTranscriber.__new__(WhisperTranscriber)
    transcriber.model_size = "base"
    transcriber.device = "cpu"
    transcriber.compute_type = "int8"
    transcriber.openvino_genai_model = "OpenVINO/whisper-medium-int8-ov"
    transcriber.openvino_device = "NPU"
    transcriber.openvino_cache_dir = None
    transcriber.languages = []
    transcriber.requested_backend = "openvino_genai"
    transcriber._backend = None
    transcriber.load_error_code = None
    transcriber.load_error_detail = None
    transcriber.user_hint = None

    attempts: list[str] = []

    class LoadableBackend:
        def __init__(self, name: str, ok: bool) -> None:
            self.name = name
            self._ok = ok

        def load(self) -> LoadStatus:
            attempts.append(self.name)
            return LoadStatus.ok() if self._ok else LoadStatus.fail(
                "missing_deps", "nope", "install it"
            )

    def fake_build(name: str):
        return LoadableBackend(name, ok=(name == "onnx_cpu"))

    transcriber._build_backend = fake_build
    available = transcriber._load_backend_chain("openvino_genai")

    assert available is True
    assert attempts == ["openvino_genai", "onnx_cpu"]
    assert transcriber.backend == "onnx_cpu"


def test_set_translate() -> None:
    from deskmate.audio.transcribe import WHISPER_TRANSLATE, _set_translate

    assert WHISPER_TRANSLATE is False
    kwargs: dict = {}
    _set_translate(kwargs, False)
    assert kwargs["task"] == "transcribe"
    _set_translate(kwargs, True)
    assert kwargs["task"] == "translate"


def test_meeting_detector_links_segments(tmp_path: Path) -> None:
    from deskmate.db import DatabaseManager
    from deskmate.meeting import MeetingDetector, detect_meeting

    assert detect_meeting(
        app_name="chrome.exe",
        window_title="Meet",
        browser_url="https://meet.google.com/abc-defg-hij",
        text="Leave call",
    ).in_meeting

    db = DatabaseManager(tmp_path / "meeting.db")
    detector = MeetingDetector(db, end_grace_seconds=999)
    detector.observe(
        app_name="chrome.exe",
        window_title="Meet",
        browser_url="https://meet.google.com/abc-defg-hij",
        text="Leave call",
    )
    assert detector.active_meeting_id is not None

    tid = db.insert_transcript(
        device="mic",
        text="hello meeting",
        language="en",
        speaker_id=None,
        start_time=1.0,
        end_time=2.0,
    )
    seg_id = detector.link_transcript(
        transcription_id=tid,
        speaker_id=None,
        text="hello meeting",
        start_time=1.0,
        end_time=2.0,
    )
    assert seg_id is not None
    meetings = db.list_meetings()
    assert meetings[0]["segment_count"] == 1
    detail = db.list_meeting_segments(meetings[0]["id"])
    assert detail[0]["transcription_id"] == tid
    db.close()


def test_speaker_embedding_roundtrip(tmp_path: Path) -> None:
    from deskmate.db import DatabaseManager

    db = DatabaseManager(tmp_path / "speaker.db")
    speaker_id = db._conn.execute(  # noqa: SLF001
        "INSERT INTO speakers(name, centroid_json, sample_count) VALUES (?, ?, ?)",
        ("Alice", "[1.0, 0.0]", 1),
    ).lastrowid
    embedding_id = db.insert_speaker_embedding(
        speaker_id=int(speaker_id),
        embedding=[1.0, 0.0],
    )
    assert embedding_id > 0
    rows = db._conn.execute("SELECT * FROM speaker_embeddings").fetchall()  # noqa: SLF001
    assert rows[0]["speaker_id"] == speaker_id
    db.close()


def test_accessibility_node_to_dict() -> None:
    from deskmate.a11y.uia_tree import AccessibilityNode, ElementBounds

    node = AccessibilityNode(
        control_type="Button", is_enabled=True, depth=2,
        name="Submit", automation_id="btn-submit",
        bounds=ElementBounds(0.0, 0.0, 10.0, 4.0),
        is_focused=True, on_screen=True,
    )
    d = node.to_dict()
    assert d["control_type"] == "Button"
    assert d["bounds"] == {"x": 0.0, "y": 0.0, "width": 10.0, "height": 4.0}
    assert d["is_focused"] is True
    assert "is_password" not in d  # optional fields skipped when None


def test_safe_control_type_survives_com_errors() -> None:
    from deskmate.a11y.uia_tree import _safe_control_type

    class _Ok:
        ControlTypeName = "Button"

    class _Bad:
        @property
        def ControlTypeName(self) -> str:
            raise OSError("COM stale element")

    assert _safe_control_type(_Ok()) == "Button"
    assert _safe_control_type(_Bad()) is None


def test_workflow_classifier_local() -> None:
    from deskmate.workflow import WorkflowClassifier

    wc = WorkflowClassifier()
    assert wc.classify("chrome.exe", "GitHub - PR review") == "browsing"
    assert wc.classify("Cursor", "main.py") == "coding"
    assert wc.classify("RandomApp", "Whatever") == "other"


def test_pipes_loader(tmp_path: Path) -> None:
    from deskmate.pipes import load_pipes

    (tmp_path / "demo.md").write_text(
        "---\n"
        "name: demo\n"
        "interval_seconds: 60\n"
        "runtime: none\n"
        "permissions:\n"
        "  read_db: true\n"
        "  trigger_capture: false\n"
        "---\n"
        "Hello pipe body.\n",
        encoding="utf-8",
    )
    pipes = load_pipes(tmp_path)
    assert len(pipes) == 1
    assert pipes[0].frontmatter.name == "demo"
    assert pipes[0].frontmatter.interval_seconds == 60
    assert pipes[0].frontmatter.permissions.read_db is True


def test_pipe_runtime_python(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path / "home"))
    from deskmate.db import DatabaseManager
    from deskmate.pipes.loader import Pipe, PipeFrontmatter, PipePermissions
    from deskmate.pipes.runtime import PipeRuntime

    db = DatabaseManager(tmp_path / "pipe.db")
    pipe = Pipe(
        path=tmp_path / "demo.md",
        frontmatter=PipeFrontmatter(
            name="demo",
            runtime="python",
            permissions=PipePermissions(read_db=True),
        ),
        body=(
            "import json, os, pathlib\n"
            "ctx = json.loads(pathlib.Path(os.environ['DESKMATE_PIPE_CONTEXT']).read_text())\n"
            "pathlib.Path(os.environ['DESKMATE_OUTPUT_DIR'], 'result.txt').write_text(ctx['pipe_name'])\n"
            "print(ctx['db_path'] is not None)\n"
        ),
    )
    execution_id = PipeRuntime(db).run(pipe)
    rows = db.list_pipe_executions("demo")
    assert rows[0]["id"] == execution_id
    assert rows[0]["status"] == "success"
    assert "True" in rows[0]["output"]
    assert (Path(rows[0]["session_path"]) / "result.txt").read_text() == "demo"
    db.close()


def test_pixel_redaction_from_ocr(tmp_path: Path) -> None:
    from PIL import Image

    from deskmate.config import Config
    from deskmate.screen.redact_image import redact_image_bytes, regions_from_ocr

    image_path = tmp_path / "frame.jpg"
    Image.new("RGB", (100, 40), color="white").save(image_path)
    words = [
        {"text": "alice@example.com", "left": "0.1", "top": "0.25", "width": "0.5", "height": "0.5"},
    ]
    regions = regions_from_ocr(
        ocr_text="alice@example.com",
        ocr_text_json=json.dumps(words),
        image_width=100,
        image_height=40,
        cfg=Config(),
    )
    assert regions and regions[0].label == "email"
    data = redact_image_bytes(image_path, regions)
    out = tmp_path / "redacted.jpg"
    out.write_bytes(data)
    with Image.open(out) as img:
        assert img.getpixel((15, 15))[0] < 20


def test_video_chunk_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    from deskmate import paths
    from deskmate.screen.video_chunks import video_chunk_path

    paths.ensure_dirs()
    p = video_chunk_path(device_name="Display 1")
    assert p.parent.parent == paths.videos_dir()
    assert p.suffix == ".mp4"
    assert "Display_1" in p.name


def test_ui_static_files_exist() -> None:
    from deskmate.ui import index_file, static_dir

    assert index_file().exists()
    assert (static_dir() / "app.css").exists()
    assert (static_dir() / "app.js").exists()


def test_ui_routes_registered() -> None:
    from deskmate.engine.api import create_app

    app = create_app()
    routes = {getattr(route, "path", "") for route in app.router.routes}
    assert "/" in routes
    assert "/ui" in routes
    assert "/ui/assets" in routes
    assert "/api" in routes
    assert "/meetings" in routes
    assert "/meetings/status" in routes
    assert "/meetings/{meeting_id}" in routes
    assert "/pipes" in routes
    assert "/pipes/{pipe_name}/run" in routes
    assert "/raw_sql" in routes
    assert "/video-chunks" in routes
    assert "/video-chunks/path" in routes
    assert "/video-chunks/register" in routes
    assert "/frames/{frame_id}/text" in routes
    assert "/frames/{frame_id}/context" in routes
    assert "/tags/{content_type}/{item_id}" in routes
    assert "/memories" in routes


def test_classify_model_load_error_ssl() -> None:
    from deskmate.audio.pipeline_status import classify_model_load_error

    code, _hint = classify_model_load_error(
        Exception("SSL: CERTIFICATE_VERIFY_FAILED unable to get local issuer certificate"),
    )
    assert code == "model_download_ssl"


def test_build_audio_status_disabled() -> None:
    from deskmate.audio.pipeline_status import build_audio_status
    from deskmate.config import Config

    cfg = Config()
    cfg.audio.enabled = False
    status = build_audio_status(cfg)
    assert status["error_code"] == "audio_disabled"
    assert "enabled = true" in status["hint"]
