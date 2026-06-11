"""Day-recap data-sufficiency warning banner (⭐7)."""

from __future__ import annotations

from deskmate.apps.agent import _data_sufficiency_warning


def test_rich_data_yields_no_warning() -> None:
    summary = {
        "data_status": "ok",
        "apps": [{"name": "Cursor.exe", "minutes": 120}],
        "audio_summary": {"segment_count": 10},
        "key_texts": [{"text": "x"}] * 8,
        "edited_files": [{"path": "a.py"}],
    }
    assert _data_sufficiency_warning(summary) == ""


def test_not_recording_yields_warning() -> None:
    summary = {
        "data_status": "not_recording",
        "apps": [],
        "audio_summary": {"segment_count": 0},
        "key_texts": [],
        "edited_files": [],
    }
    warning = _data_sufficiency_warning(summary)
    assert "DATA QUALITY WARNING" in warning
    assert "not recording" in warning.lower()


def test_thin_range_yields_warning_and_guards_against_rest_claim() -> None:
    summary = {
        "data_status": "ok",
        "apps": [{"name": "Cursor.exe", "minutes": 3}],
        "audio_summary": {"segment_count": 0},
        "key_texts": [],
        "edited_files": [],
    }
    warning = _data_sufficiency_warning(summary)
    assert "DATA QUALITY WARNING" in warning
    # Must explicitly tell the model not to imply the user rested.
    assert "rest" in warning.lower()


def test_ample_minutes_alone_is_enough() -> None:
    summary = {
        "data_status": "ok",
        "apps": [{"name": "Cursor.exe", "minutes": 200}],
        "audio_summary": {"segment_count": 0},
        "key_texts": [],
        "edited_files": [],
    }
    assert _data_sufficiency_warning(summary) == ""
