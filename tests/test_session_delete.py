"""Delete learning sessions and meetings through the public API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from deskmate.config import load as load_config
from deskmate.db.manager import DatabaseManager
from deskmate.engine.api import create_app
from deskmate.learning_memory.store import LearningStore


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    db_file = tmp_path / "data.db"
    db = DatabaseManager(db_file)
    return TestClient(create_app(cfg=load_config(), db=db)), LearningStore(db_file), db


def test_delete_learning_session_removes_the_row(client, tmp_path) -> None:
    api_client, store, _ = client
    recap_dir = tmp_path / "apps" / "user-learning" / "output" / "run1"
    recap_dir.mkdir(parents=True)
    (recap_dir / "user-learning.md").write_text("notes", encoding="utf-8")
    session_id = store.upsert_slice_sessions([{
        "kind": "study_other",
        "title": "OpenVINO 课",
        "started_at": "2026-08-16T09:00:00+08:00",
        "ended_at": "2026-08-16T10:00:00+08:00",
        "duration_min": 60,
        "meta": {"recap_path": str(recap_dir)},
    }])[0]
    store.set_session_meta(session_id, {"recap_path": str(recap_dir)})

    response = api_client.delete(f"/learning/sessions/{session_id}")

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert store.get_session(session_id) is None
    assert api_client.get("/learning/sessions").json()["data"] == []
    assert not recap_dir.exists()


def test_delete_unknown_learning_session_is_404(client) -> None:
    api_client, _, _ = client
    assert api_client.delete("/learning/sessions/999999").status_code == 404


def test_delete_meeting_removes_row_and_export_dir(client, tmp_path) -> None:
    api_client, _, db = client
    meeting_id = db.insert_meeting(name="周会")
    db.insert_meeting_segment(
        meeting_id=meeting_id, transcription_id=None, speaker_id=None,
        text="hello", start_time=0.0, end_time=1.0,
    )
    export_dir = tmp_path / "exports" / f"meeting-{meeting_id}"
    export_dir.mkdir(parents=True)
    (export_dir / "summary.json").write_text("{}", encoding="utf-8")

    response = api_client.delete(f"/meetings/{meeting_id}")

    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert db.meeting_by_id(meeting_id) is None
    assert db.list_meeting_segments(meeting_id) == []
    assert api_client.get("/meetings").json() == []
    assert not export_dir.exists()


def test_delete_unknown_meeting_is_404(client) -> None:
    api_client, _, _ = client
    assert api_client.delete("/meetings/999999").status_code == 404


def test_delete_meeting_unlinks_todos(client) -> None:
    api_client, _, db = client
    meeting_id = db.insert_meeting(name="同步会")
    todo_id = db.upsert_todo(text="跟进纪要", source="meeting", meeting_id=meeting_id)

    assert api_client.delete(f"/meetings/{meeting_id}").status_code == 200

    row = db.todo_by_id(todo_id)
    assert row is not None
    assert row["meeting_id"] is None
