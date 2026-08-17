"""Manual-only learning sessions: the auto-detect flag and backfill.

Automatic detection is off by default because "is this person studying?" has no
reliable signal. Every heuristic tried produced the same class of false positive —
an editor in the foreground read as study-coding, DeskMate's own Learning page read
as courseware (its copy is full of 学习 / 课件 / 复习), window furniture read as
errors — and a day of ordinary work generated a dozen bogus sessions.

A session the user starts and ends is unambiguous by construction. These tests pin
that the flag silences detection completely while leaving declared sessions
untouched, and that a span can be recorded after the fact for the case the whole
design otherwise has no answer for: forgetting to press start.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from deskmate.config import load as load_config
from deskmate.db.manager import DatabaseManager
from deskmate.engine.api import create_app
from deskmate.learning_memory.detector import LearningSessionDetector
from deskmate.learning_memory.store import LearningStore

# Surfaces that DO trip the detector when it is enabled, so a silent result
# proves the flag rather than weak input.
DETECTABLE = [
    ("chrome.exe", "PyTorch docs", "https://pytorch.org/docs/stable/index.html"),
    ("chrome.exe", "机器学习教程 第3讲", ""),
]


@pytest.fixture()
def store(tmp_path):
    DatabaseManager(tmp_path / "data.db")
    return LearningStore(tmp_path / "data.db")


def _det(store, **kw):
    kw.setdefault("end_grace_seconds", 0.01)
    return LearningSessionDetector(store, **kw)


# ── the flag ─────────────────────────────────────────────────────────────────

def test_auto_detect_is_off_by_default() -> None:
    assert load_config().learning.auto_detect is False


@pytest.mark.parametrize(("app", "title", "url"), DETECTABLE)
def test_detection_is_silent_when_off(store, app, title, url) -> None:
    det = _det(store, auto_detect=False)
    obs = det.observe(app_name=app, window_title=title, browser_url=url,
                      text="定义 步骤 例题 小结")
    assert obs.kind is None
    assert obs.reason == "manual-only"
    assert store.list_sessions(limit=10) == []


@pytest.mark.parametrize(("app", "title", "url"), DETECTABLE)
def test_the_same_input_is_detected_when_on(store, app, title, url) -> None:
    """Guards the premise: the flag is what silenced it, not thin evidence."""
    det = _det(store, auto_detect=True, end_grace_seconds=600.0)
    assert det.observe(app_name=app, window_title=title, browser_url=url).kind


def test_declared_sessions_are_unaffected_by_the_flag(store) -> None:
    det = _det(store, auto_detect=False)
    det.force_open(title="复习 OpenVINO 推理优化")
    assert det.is_manual is True

    # Observations still arrive; none of them may disturb the session.
    for app, title, url in DETECTABLE:
        det.observe(app_name=app, window_title=title, browser_url=url)
    det.observe(app_name="Code.exe", window_title="hongbo - Visual Studio Code",
                text="def foo(): pass")
    time.sleep(0.05)
    det.expire_if_idle()

    assert det.active_session_id is not None
    assert store.list_sessions(limit=1)[0]["title"] == "复习 OpenVINO 推理优化"


def test_session_start_time_is_readable_for_the_nudge(store) -> None:
    """The reminder loop needs a real age, so this reads the stored row."""
    det = _det(store, auto_detect=False)
    assert det.session_started_at == 0.0
    det.force_open(title="复习")
    assert det.session_started_at > 0
    # Recovered from the DB, so it survives a restart rather than resetting.
    assert _det(store, auto_detect=False).session_started_at > 0


# ── backfill ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    db_file = tmp_path / "data.db"
    DatabaseManager(db_file)
    return TestClient(create_app(cfg=load_config(), db=DatabaseManager(db_file)))


def _span(starts_hours_ago: float, length_h: float) -> tuple[str, str]:
    """A past window: begins ``starts_hours_ago`` back and runs ``length_h``."""
    start = datetime.now().astimezone() - timedelta(hours=starts_hours_ago)
    end = start + timedelta(hours=length_h)
    return start.replace(microsecond=0).isoformat(), end.replace(microsecond=0).isoformat()


def test_backfill_records_a_past_span_as_declared(client) -> None:
    lo, hi = _span(3, 1.5)
    r = client.post("/learning/sessions/backfill",
                    json={"started_at": lo, "ended_at": hi, "title": "复习 OpenVINO"})
    assert r.status_code == 200
    row = r.json()["data"]
    assert row["title"] == "复习 OpenVINO"
    assert row["duration_min"] == pytest.approx(90.0, abs=1.0)
    # Must be marked manual, or none of the declared-session treatment applies:
    # the whole span counting as evidence, and the recap scoping to it.
    assert row["meta"]["detection_source"] == "manual"
    assert row["meta"]["backfilled"] is True
    assert "user-declared" in row["reason"]


def test_backfilled_span_is_visible_to_the_declared_span_lookup(client, tmp_path) -> None:
    lo, hi = _span(3, 1.0)
    client.post("/learning/sessions/backfill", json={"started_at": lo, "ended_at": hi})
    found = LearningStore(tmp_path / "data.db").list_manual_sessions(lo, hi)
    assert len(found) == 1


def test_generate_recap_uses_the_exact_session_span(client, monkeypatch) -> None:
    lo, hi = _span(6, 1.25)
    created = client.post(
        "/learning/sessions/backfill",
        json={"started_at": lo, "ended_at": hi, "title": "历史课程"},
    ).json()
    captured = {}

    def fake_trigger(**kwargs):
        captured.update(kwargs)
        return {"ok": True, "queued": False}

    monkeypatch.setattr(
        "deskmate.learning_memory.flush.trigger_user_learning_recap",
        fake_trigger,
    )
    response = client.post(f"/learning/sessions/{created['session_id']}/recap")

    assert response.status_code == 200
    assert captured["session_id"] == created["session_id"]
    assert captured["start_time"] == lo
    assert captured["end_time"] == hi
    assert captured["background"] is False


def test_generate_recap_rejects_an_unknown_session(client) -> None:
    response = client.post("/learning/sessions/999999/recap")
    assert response.status_code == 404


def test_overlapping_backfill_merges_instead_of_duplicating(client) -> None:
    """Re-recording part of a span must not produce a second session for it."""
    lo, hi = _span(5, 2.0)                      # [now-5h, now-3h]
    first = client.post("/learning/sessions/backfill",
                        json={"started_at": lo, "ended_at": hi, "title": "第一次"})
    mid_lo, mid_hi = _span(4.5, 0.5)            # [now-4.5h, now-4h] — inside it
    second = client.post("/learning/sessions/backfill",
                         json={"started_at": mid_lo, "ended_at": mid_hi, "title": "重叠"})
    assert second.json()["session_id"] == first.json()["session_id"]
    assert len(client.get("/learning/sessions?limit=20").json()["data"]) == 1


@pytest.mark.parametrize(("label", "body"), [
    ("end before start", {"started_at": "2026-08-17T10:00:00+08:00",
                          "ended_at": "2026-08-17T09:00:00+08:00"}),
    ("equal bounds", {"started_at": "2026-08-17T10:00:00+08:00",
                      "ended_at": "2026-08-17T10:00:00+08:00"}),
    ("unparseable", {"started_at": "昨天", "ended_at": "今天"}),
    ("missing fields", {}),
])
def test_backfill_rejects_bad_input(client, label, body) -> None:
    assert client.post("/learning/sessions/backfill", json=body).status_code == 400, label


def test_backfill_refuses_an_implausibly_long_span(client) -> None:
    """A slip of the finger would otherwise mark a month as one sitting."""
    lo, hi = _span(1, 48)
    r = client.post("/learning/sessions/backfill", json={"started_at": lo, "ended_at": hi})
    assert r.status_code == 400
    assert "24 hours" in r.json()["detail"]


def test_live_exposes_the_reminder_cadence(client) -> None:
    """The UI states the real interval instead of hardcoding it."""
    body = client.get("/learning/live").json()
    assert body["auto_detect"] is False
    assert body["nudge_minutes"] == load_config().learning.nudge_minutes
