"""Ask agent — meeting tool routing helpers."""

from __future__ import annotations

from pc_assistant.engine.ask import (
    _meeting_overlaps_range,
    _slim_meeting_row,
    _slim_segment,
)


def test_meeting_overlaps_range_full_day() -> None:
    m = {"started_at": "2026-06-01T10:00:00+08:00", "ended_at": "2026-06-01T11:00:00+08:00"}
    assert _meeting_overlaps_range(
        m,
        "2026-06-01T00:00:00+08:00",
        "2026-06-01T23:59:59+08:00",
    )
    assert not _meeting_overlaps_range(
        m,
        "2026-06-01T12:00:00+08:00",
        "2026-06-01T13:00:00+08:00",
    )


def test_meeting_overlaps_range_active_no_end() -> None:
    m = {"started_at": "2026-06-01T15:00:00+08:00", "ended_at": None}
    assert _meeting_overlaps_range(
        m,
        "2026-06-01T14:00:00+08:00",
        "2026-06-01T23:59:59+08:00",
    )


def test_slim_meeting_row_parses_metadata() -> None:
    row = _slim_meeting_row({
        "id": 3,
        "name": "Zoom",
        "started_at": "2026-06-01T09:00:00+08:00",
        "ended_at": "2026-06-01T09:45:00+08:00",
        "segment_count": 12,
        "metadata": '{"profile_name": "Zoom", "app_name": "zoom.exe"}',
    })
    assert row["platform"] == "Zoom"
    assert row["segment_count"] == 12
    assert row["app_name"] == "zoom.exe"


def test_slim_segment_normalizes_fields() -> None:
    seg = _slim_segment({
        "speaker_name": "Alice",
        "start_time": "2026-06-01T09:01:00+08:00",
        "text": "Let's ship the report by Friday…",
    })
    assert seg["speaker"] == "Alice"
    assert seg["start"] == "2026-06-01T09:01:00+08:00"
    assert not seg["text"].endswith("…")


def test_slim_segment_defaults_unknown_speaker() -> None:
    seg = _slim_segment({"text": "hello"})
    assert seg["speaker"] == "Unknown"
    assert seg["text"] == "hello"
