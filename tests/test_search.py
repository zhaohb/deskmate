"""Search pipeline tests."""

from __future__ import annotations

from pathlib import Path


def test_sanitize_fts5_query_chinese() -> None:
    from deskmate.db.text_normalizer import sanitize_fts5_query

    assert sanitize_fts5_query("彩虹") == '"彩虹"'


def test_search_audio_empty_and_filtered_queries(tmp_path: Path) -> None:
    from deskmate.db import DatabaseManager

    db = DatabaseManager(tmp_path / "test.db")
    try:
        chunk_id = db.insert_audio_chunk(file_path="a.wav", device_name="mic", duration_ms=1000)
        db.insert_transcript(device="mic", text="hello rainbow world", language="en", audio_chunk_id=chunk_id)

        empty_results = db.search("", "all", limit=20)
        assert any(r.kind.value == "audio" for r in empty_results)

        query_results = db.search("rainbow", "audio", limit=20)
        assert any("rainbow" in (r.payload.get("transcription") or "") for r in query_results)
    finally:
        db.close()


def test_search_frames_fts_sanitize(tmp_path: Path) -> None:
    from deskmate.db import DatabaseManager

    db = DatabaseManager(tmp_path / "test.db")
    try:
        fid = db.insert_frame(
            monitor_id=1,
            device_name="m1",
            app_name="App",
            window_name="w",
            browser_url=None,
            focused=True,
            snapshot_path=None,
            width=0,
            height=0,
            capture_trigger="manual",
        )
        db.attach_ocr(fid, text="screen shows rainbow colors", text_json="[]", engine="winrt", confidence=0.9)
        out = db.search_frames("rainbow")
        assert out and out[0]["frame_id"] == fid
    finally:
        db.close()
