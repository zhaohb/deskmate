"""Tests for /activity-summary."""

from __future__ import annotations

from pathlib import Path

from pc_assistant.engine.activity_summary import build_activity_summary, format_summary_for_agent


def test_activity_summary_empty_range(tmp_path: Path) -> None:
    from pc_assistant.db import DatabaseManager

    db = DatabaseManager(tmp_path / "sum.db")
    try:
        out = build_activity_summary(
            db,
            start_time="2099-01-01T00:00:00+00:00",
            end_time="2099-01-01T01:00:00+00:00",
        )
        assert out["data_status"] in ("no_capture_in_range", "not_recording", "unknown")
        assert out["apps"] == []
        assert out["key_texts"] == []
        assert "windows" in out
        assert "audio_summary" in out
        text = format_summary_for_agent(out)
        assert "data_status" in text or "No" in text
    finally:
        db.close()


def test_activity_summary_with_frame_and_text(tmp_path: Path) -> None:
    from pc_assistant.db import DatabaseManager

    db = DatabaseManager(tmp_path / "sum2.db")
    try:
        fid = db.insert_frame(
            monitor_id=1,
            device_name="m1",
            app_name="Cursor.exe",
            window_name="export-report.md - proj",
            browser_url=None,
            focused=True,
            snapshot_path=None,
            width=100,
            height=100,
            capture_trigger="click",
            document_path="c:/proj/export-report.md",
            timestamp="2026-05-28T14:00:00+08:00",
        )
        db.attach_ocr(
            fid,
            text="def main(): pass  # export report module for pc_assistant",
            text_json="[]",
            engine="test",
            confidence=0.9,
        )
        with db._lock:  # noqa: SLF001
            db._conn.execute(  # noqa: SLF001
                """INSERT INTO ui_events
                   (timestamp, relative_ms, event_type, app_name, window_title,
                    browser_url, frame_id, data_json, element_json)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    "2026-05-28T14:05:00+08:00", 0, "text", "Cursor.exe",
                    "export-report.md", None, fid,
                    '{"content": "finished the export report section"}', None,
                ),
            )
        out = build_activity_summary(
            db,
            start_time="2026-05-28T00:00:00+08:00",
            end_time="2026-05-28T23:59:59+08:00",
        )
        assert out["data_status"] == "ok"
        assert out["total_frames"] >= 1
        assert any(a["name"] == "Cursor.exe" for a in out["apps"])
        assert out["edited_files"] and "export-report.md" in out["edited_files"][0]["path"]
        assert any("export" in (kt.get("text") or "").lower() for kt in out["key_texts"])
        md = format_summary_for_agent(out)
        assert "Key texts" in md
        assert "Edited files" in md
        assert "title_change" not in md.lower()
        assert "timeline" in out
        assert len(out["timeline"]) >= 1
    finally:
        db.close()


def test_is_low_value_text_filters_license_nag() -> None:
    from pc_assistant.engine.day_recap_context import is_low_value_text

    assert is_low_value_text("Go to Settings to activate Windows.")
    assert not is_low_value_text(
        "500: Internal Server Error when loading pc_assistant UI export feature"
    )
