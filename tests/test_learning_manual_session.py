"""User-declared study sessions: start by hand, collect until stopped.

Automatic detection answers "does this look like studying?", which is the wrong
question once the user has said outright that they are. A declared session is
ground truth, so it behaves differently from a detected one in three ways that
each have a failure mode if missed:

* it never expires on idle — otherwise the quiet stretches the user most wanted
  captured (reading on paper, thinking, notes off screen) end the session
* it keeps the name the user gave it — a window title is not an improvement on
  "复习 OpenVINO 推理优化"
* its whole span counts as study time — re-deriving it from heuristics would
  report the fraction the classifier happened to recognise

Kind may still be promoted: that only sharpens the category, it does not
contradict the user.
"""

from __future__ import annotations

import time

import pytest

from deskmate.db.manager import DatabaseManager
from deskmate.learning_memory.detector import LearningSessionDetector
from deskmate.learning_memory.store import LearningStore

TITLE = "复习 OpenVINO 推理优化"


@pytest.fixture()
def store(tmp_path):
    db_file = tmp_path / "data.db"
    DatabaseManager(db_file)
    return LearningStore(db_file)


def _detector(store, **kw):
    # A grace of ~0 makes idle expiry fire on the very next check, so the
    # "never expires" assertions cannot pass by simply being fast.
    kw.setdefault("end_grace_seconds", 0.01)
    # Auto-detection ships off (see LearningConfig.auto_detect), but this file
    # contrasts declared sessions AGAINST detected ones, so it needs both halves.
    # Declared sessions go through force_open, which ignores the flag either way.
    kw.setdefault("auto_detect", True)
    return LearningSessionDetector(store, **kw)


# ── lifecycle ────────────────────────────────────────────────────────────────

def test_manual_session_opens_and_is_flagged(store) -> None:
    det = _detector(store)
    row = det.force_open(title=TITLE)
    assert det.is_manual is True
    assert det.active_session_id == int(row["id"])
    assert row["meta"]["detection_source"] == "manual"


def test_manual_session_survives_idle_expiry(store) -> None:
    """The headline guarantee: silence does not end a declared session."""
    det = _detector(store)
    det.force_open(title=TITLE)
    time.sleep(0.05)
    det.expire_if_idle()
    assert det.active_session_id is not None


def test_detected_session_still_expires_on_idle(store) -> None:
    """Automatic sessions keep the old cleanup behaviour."""
    det = _detector(store)
    det.observe(
        app_name="chrome.exe", window_title="PyTorch docs",
        browser_url="https://pytorch.org/docs/stable/index.html",
    )
    assert det.active_session_id is not None
    time.sleep(0.05)
    det.expire_if_idle()
    assert det.active_session_id is None


def test_ending_clears_the_manual_flag(store) -> None:
    det = _detector(store)
    det.force_open(title=TITLE)
    sid = det.force_close(trigger_recap=False)
    assert sid is not None
    assert det.is_manual is False
    assert det.active_session_id is None


def test_manual_flag_survives_a_restart(store) -> None:
    """It lives in the stored row, so a daemon restart cannot silently end it."""
    _detector(store).force_open(title=TITLE)
    revived = _detector(store)
    assert revived.is_manual is True
    time.sleep(0.05)
    revived.expire_if_idle()
    assert revived.active_session_id is not None


def test_detected_session_is_not_restored_as_manual(store) -> None:
    det = _detector(store, end_grace_seconds=600.0)
    det.observe(
        app_name="chrome.exe", window_title="PyTorch docs",
        browser_url="https://pytorch.org/docs/stable/index.html",
    )
    assert LearningSessionDetector(store, end_grace_seconds=600.0).is_manual is False


# ── the user's title wins ────────────────────────────────────────────────────

def test_window_titles_never_overwrite_a_manual_title(store) -> None:
    det = _detector(store, end_grace_seconds=600.0)
    det.force_open(title=TITLE)
    det.observe(app_name="Code.exe", window_title="hongbo - Visual Studio Code",
                text="def foo(): pass")
    det.observe(app_name="chrome.exe", window_title="PyTorch docs",
                browser_url="https://pytorch.org/docs/stable/index.html")
    assert store.list_sessions(limit=1)[0]["title"] == TITLE


def test_kind_is_still_promoted_inside_a_manual_session(store) -> None:
    """Sharpening the category does not contradict what the user declared."""
    det = _detector(store, end_grace_seconds=600.0)
    det.force_open(title=TITLE)          # opens as study_other
    det.observe(app_name="chrome.exe", window_title="PyTorch docs",
                browser_url="https://pytorch.org/docs/stable/index.html")
    row = store.list_sessions(limit=1)[0]
    assert row["kind"] == "courseware_view"
    assert row["title"] == TITLE


# ── the whole span counts as study time ──────────────────────────────────────

def test_manual_sessions_are_listed_for_an_overlapping_window(store) -> None:
    det = _detector(store)
    det.force_open(title=TITLE)
    det.force_close(trigger_recap=False)
    row = store.list_sessions(limit=1)[0]
    found = store.list_manual_sessions(row["started_at"], row["ended_at"])
    assert [s["id"] for s in found] == [row["id"]]


def test_open_manual_session_counts_up_to_now(store) -> None:
    """An unfinished session still belongs to any window it started inside."""
    det = _detector(store)
    det.force_open(title=TITLE)
    row = store.list_sessions(limit=1)[0]
    assert store.list_manual_sessions(row["started_at"], "2099-01-01T00:00:00+08:00")


def test_detected_sessions_are_not_reported_as_manual(store) -> None:
    det = _detector(store, end_grace_seconds=600.0)
    det.observe(app_name="chrome.exe", window_title="PyTorch docs",
                browser_url="https://pytorch.org/docs/stable/index.html")
    row = store.list_sessions(limit=1)[0]
    assert store.list_manual_sessions(row["started_at"], "2099-01-01T00:00:00+08:00") == []


def test_window_outside_the_session_finds_nothing(store) -> None:
    det = _detector(store)
    det.force_open(title=TITLE)
    det.force_close(trigger_recap=False)
    assert store.list_manual_sessions(
        "2000-01-01T00:00:00+08:00", "2000-01-02T00:00:00+08:00",
    ) == []


@pytest.mark.parametrize("sep", ["T", " "])
def test_range_lookup_accepts_either_iso_separator(store, sep) -> None:
    """Callers disagree on the separator; string-compared bounds would drop rows."""
    det = _detector(store)
    det.force_open(title=TITLE)
    det.force_close(trigger_recap=False)
    row = store.list_sessions(limit=1)[0]
    lo = str(row["started_at"]).replace("T", sep, 1)
    hi = str(row["ended_at"]).replace("T", sep, 1)
    assert store.list_manual_sessions(lo, hi)


# ── recap merge ──────────────────────────────────────────────────────────────

def test_manual_span_replaces_derived_sessions_inside_it(store, monkeypatch, tmp_path) -> None:
    """A declared hour is one session, not the eleven minutes a classifier saw."""
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    from deskmate.apps.agent import _merge_manual_sessions

    det = _detector(store)
    det.force_open(title=TITLE)
    det.force_close(trigger_recap=False)
    row = store.list_sessions(limit=1)[0]

    derived = [
        {"id": 90, "kind": "code_edit", "started_at": row["started_at"],
         "ended_at": row["ended_at"], "duration_min": 3},          # inside → dropped
        {"id": 91, "kind": "code_edit", "started_at": "2000-01-01T09:00:00+08:00",
         "ended_at": "2000-01-01T09:30:00+08:00", "duration_min": 30},  # outside → kept
    ]
    merged = _merge_manual_sessions(derived, row["started_at"], row["ended_at"])
    ids = [s["id"] for s in merged]
    assert row["id"] in ids
    assert 90 not in ids
    assert 91 in ids
    manual = next(s for s in merged if s["id"] == row["id"])
    assert "user-declared" in manual["reason"]


def test_experiments_inside_a_declared_span_are_kept_as_evidence() -> None:
    """Hands-on work during a declared session is the point of the session.

    "Should this keep a session alive?" and "is this evidence of what the user
    did while studying?" are different questions. Answering both with the same
    classifier meant that once editors stopped counting as a learning signal,
    the experiments run against the lecture disappeared from the study report
    unless they happened to throw an error.
    """
    from deskmate.apps.learning_slice import filter_learning_key_texts

    rows = [
        {"app_name": "chrome.exe", "window_name": "npu shape - Google 搜索",
         "browser_url": "https://www.google.com/search?q=npu",
         "text": "搜索 NPU dynamic shape", "timestamp": "2026-08-17T10:05:00+08:00"},
        {"app_name": "Code.exe", "window_name": "test_openvino.py - Visual Studio Code",
         "text": "core = ov.Core()  # 试一下 NPU", "timestamp": "2026-08-17T10:10:00+08:00"},
        {"app_name": "WindowsTerminal.exe", "window_name": "终端",
         "text": "python test_openvino.py → latency 12ms", "timestamp": "2026-08-17T10:12:00+08:00"},
        {"app_name": "Code.exe", "window_name": "billing.py - company - Visual Studio Code",
         "text": "def billing(): pass", "timestamp": "2026-08-17T15:00:00+08:00"},
    ]
    span = [("2026-08-17T10:00:00+08:00", "2026-08-17T10:30:00+08:00")]
    titles = {r["window_name"] for r in filter_learning_key_texts(rows, declared_spans=span)}

    assert "test_openvino.py - Visual Studio Code" in titles
    assert "终端" in titles
    assert "npu shape - Google 搜索" in titles
    # Unrelated work outside the declared span stays out.
    assert "billing.py - company - Visual Studio Code" not in titles


def test_editor_activity_outside_a_declared_span_is_still_dropped() -> None:
    """Without a declaration, an editor remains no evidence of studying."""
    from deskmate.apps.learning_slice import filter_learning_key_texts

    rows = [{
        "app_name": "Code.exe", "window_name": "billing.py - Visual Studio Code",
        "text": "def billing(): pass", "timestamp": "2026-08-17T15:00:00+08:00",
    }]
    assert filter_learning_key_texts(rows) == []


def test_manual_session_inherits_observations_it_replaced(store, monkeypatch, tmp_path) -> None:
    """Replacing derived spans must not throw away what they observed."""
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    from deskmate.apps.agent import _merge_manual_sessions

    det = _detector(store)
    det.force_open(title=TITLE)
    det.force_close(trigger_recap=False)
    row = store.list_sessions(limit=1)[0]

    derived = [{
        "id": 90, "kind": "courseware_view", "started_at": row["started_at"],
        "ended_at": row["ended_at"], "duration_min": 3,
        "apps": ["chrome.exe"], "urls": ["https://pytorch.org/docs/"],
        "queries": [], "topics": ["pytorch"], "concepts": ["autograd"],
    }]
    merged = _merge_manual_sessions(derived, row["started_at"], row["ended_at"])
    manual = next(s for s in merged if s["id"] == row["id"])
    assert "chrome.exe" in manual["apps"]
    assert "autograd" in manual["concepts"]


def test_merge_is_a_noop_without_manual_sessions(store, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    from deskmate.apps.agent import _merge_manual_sessions

    derived = [{"id": 1, "kind": "code_edit", "started_at": "2026-08-17T10:00:00+08:00"}]
    assert _merge_manual_sessions(
        derived, "2026-08-17T09:00:00+08:00", "2026-08-17T11:00:00+08:00",
    ) == derived
