"""Deterministic core of the session journey: activity timeline + error detection.

The LLM "how was it fixed" pass is best-effort and not covered here; these lock
in the parts that must hold with no model: how frames map to activities, how time
is attributed, and how on-screen errors are grouped and judged resolved.
"""

from __future__ import annotations

from deskmate.learning_memory.journey import (
    build_process,
    classify_activity,
    clean_title,
    detect_errors,
    summarize_arc,
)


def test_clean_title_drops_the_app_suffix() -> None:
    """The suffix names the browser; the rest names what was actually open."""
    assert clean_title("2026.2版 OpenVINO 的新功能_哔哩哔哩 - Google Chrome") == "2026.2版 OpenVINO 的新功能_哔哩哔哩"
    assert clean_title("main.py - Visual Studio Code") == "main.py"
    assert clean_title("") == ""


def test_segments_carry_the_dominant_title_and_range() -> None:
    frames = [
        {"ts": "2026-08-17T17:00:00+08:00", "app": "chrome.exe", "window": "OpenVINO 新功能 - Google Chrome", "url": "https://bilibili.com/v"},
        {"ts": "2026-08-17T17:02:00+08:00", "app": "chrome.exe", "window": "OpenVINO 新功能 - Google Chrome", "url": "https://bilibili.com/v"},
        {"ts": "2026-08-17T17:04:00+08:00", "app": "chrome.exe", "window": "别的页面 - Google Chrome", "url": "https://bilibili.com/v"},
    ]
    seg = build_process(frames)["segments"][0]
    assert seg["title"] == "OpenVINO 新功能"  # the title held for most of the stretch
    assert seg["start_ts"] and seg["end_ts"]  # range is kept so evidence can be sliced


def test_classify_activity_covers_the_main_study_modes() -> None:
    assert classify_activity("Code.exe", "main.py — Visual Studio Code") == "code"
    assert classify_activity("WindowsTerminal.exe", "pytest") == "debug"
    assert classify_activity("chrome.exe", "openvino - bilibili", "https://bilibili.com/video/BV1") == "lecture"
    assert classify_activity("chrome.exe", "openvino error", "https://www.google.com/search?q=x") == "search"
    assert classify_activity("chrome.exe", "OpenVINO docs", "https://docs.openvino.ai") == "read"
    assert classify_activity("chrome.exe", "DeskMate", "http://127.0.0.1:8787") == "tool"
    assert classify_activity("chrome.exe", "DeskMate — 学习", "") == "tool"


def test_build_process_allocates_time_and_merges_segments() -> None:
    frames = [
        {"ts": "2026-08-17T17:00:00+08:00", "app": "chrome.exe", "window": "bilibili", "url": "https://bilibili.com/v"},
        {"ts": "2026-08-17T17:02:00+08:00", "app": "chrome.exe", "window": "bilibili", "url": "https://bilibili.com/v"},
        {"ts": "2026-08-17T17:03:00+08:00", "app": "Code.exe", "window": "main.py", "url": ""},
    ]
    process = build_process(frames)
    # Two lecture frames merge into one segment, then a coding segment.
    assert [s["key"] for s in process["segments"]] == ["lecture", "code"]
    keys = {a["key"] for a in process["allocation"]}
    assert keys == {"lecture", "code"}
    assert process["total_min"] > 0


def test_build_process_caps_idle_gaps() -> None:
    # A 2-hour gap between two frames must not count as 2 hours of study.
    frames = [
        {"ts": "2026-08-17T17:00:00+08:00", "app": "Code.exe", "window": "a.py", "url": ""},
        {"ts": "2026-08-17T19:00:00+08:00", "app": "Code.exe", "window": "a.py", "url": ""},
    ]
    assert build_process(frames)["total_min"] < 6  # idle_cap ~2.5 min + tail


def test_detect_errors_groups_and_marks_resolved() -> None:
    ocr = [
        {"ts": "2026-08-17T17:10:00+08:00", "app": "WindowsTerminal.exe", "text": "Traceback (most recent call last) ModuleNotFoundError: No module named 'deskmate'"},
        {"ts": "2026-08-17T17:11:00+08:00", "app": "WindowsTerminal.exe", "text": "ModuleNotFoundError: No module named 'deskmate' again"},
        {"ts": "2026-08-17T17:38:00+08:00", "app": "WindowsTerminal.exe", "text": "npm ERR! missing script build"},
    ]
    errors = detect_errors(ocr, ended_at="2026-08-17T17:40:00+08:00")
    # The repeated ModuleNotFoundError collapses to one problem, seen twice.
    mnfe = next(e for e in errors if "ModuleNotFound" in e["error"])
    assert mnfe["occurrences"] == 2
    # It stopped appearing, but nothing showed it working → not a claimed success.
    assert mnfe["status"] == "likely_resolved"
    npm = next(e for e in errors if "npm ERR" in e["error"])
    assert npm["status"] == "unresolved"  # last seen 17:38, within the 3-min grace


def test_success_signal_upgrades_the_verdict_to_resolved() -> None:
    """'Stopped appearing' and 'observed working again' are different claims."""
    ocr = [
        {"ts": "2026-08-17T17:10:00+08:00", "app": "WindowsTerminal.exe", "text": "ModuleNotFoundError: No module named 'deskmate'"},
        {"ts": "2026-08-17T17:14:00+08:00", "app": "WindowsTerminal.exe", "text": "Successfully installed deskmate-0.1.0"},
    ]
    errors = detect_errors(ocr, ended_at="2026-08-17T17:40:00+08:00")
    assert errors[0]["status"] == "resolved"
    assert errors[0]["success_at"] == "17:14"


def test_errors_on_lecture_slides_are_not_the_learners_errors() -> None:
    """A talk about exceptions puts tracebacks on screen; that is course content."""
    ocr = [{"ts": "2026-08-17T17:10:00+08:00", "app": "chrome.exe",
            "text": "slide: ValueError: invalid literal for int()"}]
    assert detect_errors(ocr, ended_at="2026-08-17T17:40:00+08:00") == []
    # Opt in when the caller really wants reference material included.
    assert len(detect_errors(ocr, ended_at="2026-08-17T17:40:00+08:00", include_reference=True)) == 1


def test_detect_errors_ignores_non_error_text() -> None:
    ocr = [{"ts": "2026-08-17T17:10:00+08:00", "app": "WindowsTerminal.exe", "text": "welcome to the course, click the red button"}]
    assert detect_errors(ocr, ended_at="2026-08-17T17:40:00+08:00") == []


def test_segment_summaries_are_parsed_and_mapped_by_index(monkeypatch) -> None:
    """Covers the model path without a running Ollama — the reply shape is ours."""
    import json as _json
    import sys
    import types

    from deskmate.learning_memory import journey as jn

    fake_pkg = types.ModuleType("deskmate.engine")
    fake_pkg.__path__ = []
    fake = types.ModuleType("deskmate.engine.llm")
    fake.chat_ollama = lambda *a, **k: {"content": _json.dumps({"segments": [
        {"i": 0, "detail": "讲 INT4 KV Cache 压缩与显存收益", "points": ["INT4 KV Cache", "Physical AI API"],
         "doing": "在看讲座", "related": True},
        {"i": 1, "detail": "", "points": []},
    ]}, ensure_ascii=False)}
    fake.strip_thinking = lambda s: s
    monkeypatch.setitem(sys.modules, "deskmate.engine", fake_pkg)
    monkeypatch.setitem(sys.modules, "deskmate.engine.llm", fake)
    segments = [
        {"label": "看讲座/视频", "title": "OpenVINO 新功能", "minutes": 20, "start_ts": "2026-08-17T17:00:00+08:00", "end_ts": "2026-08-17T17:20:00+08:00"},
        {"label": "阅读资料/课件", "title": "docs", "minutes": 5, "start_ts": "2026-08-17T17:20:00+08:00", "end_ts": "2026-08-17T17:25:00+08:00"},
    ]
    audio = [{"ts": "2026-08-17T17:05:00+08:00", "text": "这次讲 INT4 KV Cache 压缩，可以降低显存占用"}]

    out = jn.summarize_segments(segments, [], audio, base="x", model="m", timeout=10)

    assert out[0]["detail"] == "讲 INT4 KV Cache 压缩与显存收益"
    assert out[0]["points"] == ["INT4 KV Cache", "Physical AI API"]
    assert out[0]["related"] is True
    assert out[0]["doing"] == "在看讲座"
    assert 1 not in out  # a stretch with nothing to say is dropped, not blanked


def test_segment_summaries_soft_fail_when_the_model_is_down(monkeypatch) -> None:
    import sys
    import types

    from deskmate.learning_memory import journey as jn

    fake = types.ModuleType("deskmate.engine.llm")

    def _boom(*_a, **_k):
        raise ConnectionError("connection refused")

    fake.chat_ollama = _boom
    fake.strip_thinking = lambda s: s
    fake_pkg = types.ModuleType("deskmate.engine")
    fake_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "deskmate.engine", fake_pkg)
    monkeypatch.setitem(sys.modules, "deskmate.engine.llm", fake)
    segments = [{"label": "看讲座/视频", "title": "t", "start_ts": "2026-08-17T17:00:00+08:00", "end_ts": "2026-08-17T17:20:00+08:00"}]

    assert jn.summarize_segments(segments, [], [], base="x", model="m", timeout=5) == {}


def test_deskmate_ui_is_not_classified_as_reading_courseware() -> None:
    """Looking at DeskMate itself is tooling, not 'reading materials'."""
    frames = [
        {"ts": "2026-08-21T11:03:00+08:00", "app": "chrome.exe",
         "window": "2026.2版 OpenVINO™ 的新功能_哔哩哔哩_bilibili - Google Chrome",
         "url": "https://www.bilibili.com/video/BV1"},
        {"ts": "2026-08-21T11:27:00+08:00", "app": "chrome.exe",
         "window": "DeskMate - Google Chrome", "url": "http://127.0.0.1:8787/"},
        {"ts": "2026-08-21T11:27:36+08:00", "app": "chrome.exe",
         "window": "DeskMate - Google Chrome", "url": "http://127.0.0.1:8787/"},
    ]
    keys = [s["key"] for s in build_process(frames, min_segment_s=0)["segments"]]
    assert keys == ["lecture", "tool"]


def test_build_journey_without_llm_still_flags_tooling_as_unrelated() -> None:
    from deskmate.learning_memory.journey import build_journey

    frames = [
        {"ts": "2026-08-21T11:03:00+08:00", "app": "chrome.exe",
         "window": "OpenVINO 讲座 - Google Chrome", "url": "https://www.bilibili.com/video/BV1"},
        {"ts": "2026-08-21T11:27:00+08:00", "app": "chrome.exe",
         "window": "DeskMate - Google Chrome", "url": "http://127.0.0.1:8787/"},
        {"ts": "2026-08-21T11:27:36+08:00", "app": "chrome.exe",
         "window": "DeskMate - Google Chrome", "url": "http://127.0.0.1:8787/"},
    ]
    out = build_journey(
        frames=frames, ocr_rows=[], audio_rows=[],
        started_at="2026-08-21T11:03:00+08:00", ended_at="2026-08-21T11:28:00+08:00",
        use_llm=False,
    )
    by_key = {s["key"]: s for s in out["process"]["segments"]}
    assert by_key["lecture"]["related"] is True
    assert by_key["tool"]["related"] is False
    assert "DeskMate" in by_key["tool"]["related_why"]
    assert out["arc"] == {}
    assert out["llm_note"] == "disabled"


def test_summarize_arc_tells_the_path_and_the_outcome(monkeypatch) -> None:
    """The wrap-up must say how the session unfolded and what it finished."""
    import json as _json
    import sys
    import types

    from deskmate.learning_memory import journey as jn

    fake_pkg = types.ModuleType("deskmate.engine")
    fake_pkg.__path__ = []
    fake = types.ModuleType("deskmate.engine.llm")
    fake.chat_ollama = lambda *a, **k: {"content": _json.dumps({
        "path": "先看 OpenVINO 新功能讲座，再打开 DeskMate 回看本次学习记录。",
        "outcome": "听完 2026.2 版 OpenVINO 新功能介绍，并用 DeskMate 核对了学习过程。",
        "related_note": "最后一段是 DeskMate 本机界面，属于学习工具而非课件。",
    }, ensure_ascii=False)}
    fake.strip_thinking = lambda s: s
    monkeypatch.setitem(sys.modules, "deskmate.engine", fake_pkg)
    monkeypatch.setitem(sys.modules, "deskmate.engine.llm", fake)
    segments = [
        {"label": "看讲座/视频", "title": "OpenVINO 新功能", "minutes": 20,
         "related": True, "doing": "在看 OpenVINO 讲座",
         "detail": "INT4 KV Cache", "start": "11:03"},
        {"label": "本机工具/界面", "title": "DeskMate", "minutes": 0.7,
         "related": False, "doing": "在看 DeskMate 学习页",
         "detail": "打开学习过程", "start": "11:27"},
    ]
    out = summarize_arc(segments, [], [], session_title="OpenVINO 讲座",
                        base="x", model="m", timeout=10)
    assert "OpenVINO" in out["path"]
    assert "DeskMate" in out["outcome"]
    assert "课件" in out["related_note"]
