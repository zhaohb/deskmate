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

    from deskmate.engine import llm as llm_mod
    from deskmate.learning_memory import journey as jn

    monkeypatch.setattr(
        llm_mod, "chat_ollama",
        lambda *a, **k: {"content": _json.dumps({"segments": [
            {"i": 0, "detail": "讲 INT4 KV Cache 压缩与显存收益", "points": ["INT4 KV Cache", "Physical AI API"]},
            {"i": 1, "detail": "", "points": []},
        ]}, ensure_ascii=False)},
    )
    segments = [
        {"label": "看讲座/视频", "title": "OpenVINO 新功能", "minutes": 20, "start_ts": "2026-08-17T17:00:00+08:00", "end_ts": "2026-08-17T17:20:00+08:00"},
        {"label": "阅读资料/课件", "title": "docs", "minutes": 5, "start_ts": "2026-08-17T17:20:00+08:00", "end_ts": "2026-08-17T17:25:00+08:00"},
    ]
    audio = [{"ts": "2026-08-17T17:05:00+08:00", "text": "这次讲 INT4 KV Cache 压缩，可以降低显存占用"}]

    out = jn.summarize_segments(segments, [], audio, base="x", model="m", timeout=10)

    assert out[0]["detail"] == "讲 INT4 KV Cache 压缩与显存收益"
    assert out[0]["points"] == ["INT4 KV Cache", "Physical AI API"]
    assert 1 not in out  # a stretch with nothing to say is dropped, not blanked


def test_segment_summaries_soft_fail_when_the_model_is_down(monkeypatch) -> None:
    from deskmate.engine import llm as llm_mod
    from deskmate.learning_memory import journey as jn

    def _boom(*_a, **_k):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(llm_mod, "chat_ollama", _boom)
    segments = [{"label": "看讲座/视频", "title": "t", "start_ts": "2026-08-17T17:00:00+08:00", "end_ts": "2026-08-17T17:20:00+08:00"}]

    assert jn.summarize_segments(segments, [], [], base="x", model="m", timeout=5) == {}
