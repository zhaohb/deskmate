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


def test_expand_search_query_splits_compound() -> None:
    from deskmate.db.text_normalizer import expand_search_query

    # camelCase split + prefix wildcard so `submit` reaches `submitButton`.
    expanded = expand_search_query("submitButton")
    assert '"submit"' in expanded
    assert "*" in expanded


def test_search_matches_camelcase_identifier(tmp_path: Path) -> None:
    """`submit` should now find a frame whose OCR text has `submitButton`."""
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
        db.attach_ocr(
            fid, text="click the submitButton to continue",
            text_json="[]", engine="winrt", confidence=0.9,
        )
        results = db.search("submit", "ocr", limit=20)
        assert any(r.payload.get("frame_id") == fid for r in results)
    finally:
        db.close()


def test_ocr_search_ranks_by_relevance_not_recency(tmp_path: Path) -> None:
    """A keyword query returns the most relevant frame first, even if older."""
    from deskmate.db import DatabaseManager

    db = DatabaseManager(tmp_path / "test.db")
    try:
        # Older frame: strong, repeated match for the term.
        relevant = db.insert_frame(
            monitor_id=1, device_name="m1", app_name="App", window_name="w",
            browser_url=None, focused=True, snapshot_path=None,
            width=0, height=0, capture_trigger="manual",
        )
        db.attach_ocr(
            relevant,
            text="rainbow rainbow rainbow over the rainbow valley",
            text_json="[]", engine="winrt", confidence=0.9,
        )
        # Newer frame: weak, single mention buried in unrelated text.
        recent = db.insert_frame(
            monitor_id=1, device_name="m1", app_name="App", window_name="w",
            browser_url=None, focused=True, snapshot_path=None,
            width=0, height=0, capture_trigger="manual",
        )
        db.attach_ocr(
            recent,
            text=(
                "a very long unrelated paragraph about many other topics that "
                "only mentions rainbow once near the very end of the text"
            ),
            text_json="[]", engine="winrt", confidence=0.9,
        )
        results = db.search("rainbow", "ocr", limit=20)
        ids = [r.payload.get("frame_id") for r in results]
        assert relevant in ids and recent in ids
        # BM25 should float the dense-match (older) frame above the sparse one.
        assert ids.index(relevant) < ids.index(recent)
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
