"""Per-session learning recap API tests."""

from __future__ import annotations

import json

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
    return TestClient(create_app(cfg=load_config(), db=db)), LearningStore(db_file)


def test_generate_recap_uses_the_exact_session_span(client, monkeypatch) -> None:
    api_client, store = client
    start = "2026-08-16T09:00:00+08:00"
    end = "2026-08-16T10:30:00+08:00"
    session_id = store.upsert_slice_sessions(
        [{
            "kind": "study_other",
            "title": "Historical course",
            "started_at": start,
            "ended_at": end,
            "duration_min": 90,
        }]
    )[0]
    captured = {}

    def fake_trigger(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "queued": False}

    monkeypatch.setattr(
        "deskmate.learning_memory.flush.trigger_user_learning_recap",
        fake_trigger,
    )

    response = api_client.post(f"/learning/sessions/{session_id}/recap")

    assert response.status_code == 200
    assert captured == {
        "verbose": True,
        "background": False,
        "session_id": session_id,
        "start_time": start,
        "end_time": end,
    }


def test_generate_recap_rejects_an_unknown_session(client) -> None:
    api_client, _ = client
    response = api_client.post("/learning/sessions/999999/recap")
    assert response.status_code == 404


def test_get_recap_returns_structured_session_graph(client, tmp_path) -> None:
    api_client, store = client
    session_id = store.upsert_slice_sessions([{
        "kind": "study_other",
        "title": "NLP course",
        "started_at": "2026-08-16T09:00:00+08:00",
        "ended_at": "2026-08-16T10:00:00+08:00",
        "duration_min": 60,
    }])[0]
    recap_dir = tmp_path / "recap"
    recap_dir.mkdir()
    (recap_dir / "user-learning.md").write_text("## 课程总结\nNLP", encoding="utf-8")
    (recap_dir / "learning-enrichment.json").write_text(json.dumps({
        "topics": [{"name": "NLP"}],
        "edges": [{
            "src_name": "Tokenization",
            "dst_name": "Embedding",
            "rel": "leads_to",
            "evidence": "Tokenization leads to Embedding",
        }],
    }), encoding="utf-8")
    store.set_session_meta(session_id, {"recap_path": str(recap_dir)})

    response = api_client.get(f"/learning/sessions/{session_id}/recap")

    assert response.status_code == 200
    assert response.json()["graph"] == {
        "nodes": [
            {"id": "Tokenization", "kind": "concept"},
            {"id": "Embedding", "kind": "concept"},
        ],
        "edges": [{
            "source": "Tokenization",
            "target": "Embedding",
            "relation": "leads_to",
            "evidence": "Tokenization leads to Embedding",
        }],
    }
