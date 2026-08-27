"""Evidence pipeline for a lecture: transcript → slice → prompt bundle.

Simulates watching a ~23-minute technical talk (the BV16FKy6kEVk OpenVINO case)
by seeding a realistic transcript, then runs the real collection path. This is
the half of the end-to-end test that needs neither live capture nor a model, so
it can catch integration breakage before anyone spends 23 minutes recording.

It pins the three properties that were previously wrong, each of which silently
degraded the report rather than failing:

* **Volume** — the old path asked the search API for 40 audio rows, so a class
  with hundreds of utterances was summarized from a few of them.
* **Order** — that query is ``ORDER BY timestamp DESC LIMIT n``, so those 40
  rows were the *tail* of the window, not a sample of the lecture.
* **Disclosure** — nothing told the model evidence had been dropped, so a
  partial transcript read as a complete one and produced confident, incomplete
  teaching highlights.

What is deliberately NOT covered: audio loopback capture, Whisper accuracy, OCR
quality and the LLM step. Those need real hardware and a running model server.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

import pytest

from deskmate.apps import agent
from deskmate.apps.agent import (
    _AUDIO_MAX_LINES,
    _collect_courseware_ocr_lines,
    _collect_learning_audio_bits,
)
from deskmate.apps.learning_slice import format_learning_bundle, select_spanning
from deskmate.db.manager import DatabaseManager

# A 23-minute talk, one utterance every ~4s → ~345 rows: the realistic volume
# that the old 40-row cap silently discarded.
LECTURE_START = datetime(2026, 8, 15, 14, 0, 0).astimezone()
LECTURE_MINUTES = 23
UTTERANCE_EVERY_SEC = 4


def _seed_lecture(db: DatabaseManager, *, n: int) -> list[str]:
    """Insert ``n`` evenly spaced transcript rows; return their timestamps."""
    stamps: list[str] = []
    for i in range(n):
        ts = (LECTURE_START + timedelta(seconds=i * UTTERANCE_EVERY_SEC)).replace(
            microsecond=0
        ).isoformat()
        stamps.append(ts)
        # Distinct text per row: the collector de-dupes on a text prefix, so
        # identical lines would collapse and mask a volume regression.
        db._conn.execute(  # noqa: SLF001 - direct seed, no public writer needed
            """INSERT INTO audio_transcriptions
                   (timestamp, transcription, device, text_length)
               VALUES (?,?,?,?)""",
            (ts, f"第{i}段讲解：OpenVINO 2026.2 的新功能与模型转换步骤说明。", "loopback", 30),
        )
    return stamps


@pytest.fixture()
def lecture_db(tmp_path, monkeypatch):
    """A temp DeskMate home seeded with a full lecture transcript."""
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    db = DatabaseManager(tmp_path / "data.db")
    n = (LECTURE_MINUTES * 60) // UTTERANCE_EVERY_SEC
    stamps = _seed_lecture(db, n=n)
    return db, stamps


def _window(stamps: list[str]) -> tuple[str, str]:
    return stamps[0], stamps[-1]


def _minutes(hhmmss: str) -> float:
    """'14:11:32' → minutes past the hour, for span assertions."""
    h, m, s = (int(x) for x in hhmmss.split(":"))
    return h * 60 + m + s / 60


# ── volume: the whole class reaches the prompt, not a 40-row sliver ──────────

def test_lecture_seeds_more_rows_than_the_old_cap(lecture_db) -> None:
    """Guards the premise: without volume, the other assertions prove nothing."""
    db, stamps = lecture_db
    assert len(stamps) > 300
    assert db.count_transcripts_in_range(*_window(stamps)) == len(stamps)


def test_collected_audio_far_exceeds_the_old_40_row_cap(lecture_db) -> None:
    """Coverage is counted in transcript rows, whatever the paragraphing."""
    _, stamps = lecture_db
    bits, stats = _collect_learning_audio_bits(*_window(stamps), {})
    assert stats["source"] == "db"
    assert stats["total"] == len(stamps)
    assert stats["included"] > 200
    assert bits


# ── order + span: teaching order preserved, whole class represented ──────────

def test_collected_audio_is_chronological(lecture_db) -> None:
    _, stamps = lecture_db
    bits, stats = _collect_learning_audio_bits(*_window(stamps), {})
    assert stats["ordered"] is True
    prefixes = [b.split(":")[0] for b in bits]
    assert prefixes == sorted(prefixes)


def test_collected_audio_spans_the_whole_lecture(lecture_db) -> None:
    """The old DESC+LIMIT path returned only the final minutes."""
    _, stamps = lecture_db
    bits, _ = _collect_learning_audio_bits(*_window(stamps), {})
    # Paragraphs are stamped HH:MM:SS, and must bracket the whole class.
    began, ended = _minutes(stamps[0][11:19]), _minutes(stamps[-1][11:19])
    assert bits[0].startswith(stamps[0][11:19])
    assert _minutes(bits[-1][:8]) >= ended - 1
    # Coverage is spread, not clustered at either end: the middle paragraph must
    # come from somewhere near the middle of the class.
    middle = _minutes(bits[len(bits) // 2][:8]) - began
    assert 0.35 <= middle / (ended - began) <= 0.65


def test_short_lecture_is_delivered_complete(tmp_path, monkeypatch) -> None:
    """Under budget, nothing is dropped and nothing claims truncation."""
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    db = DatabaseManager(tmp_path / "data.db")
    stamps = _seed_lecture(db, n=30)
    bits, stats = _collect_learning_audio_bits(*_window(stamps), {})
    assert bits, "a short lecture must still produce paragraphs"
    # Coverage counts source rows, not paragraphs: all 30 rows are present even
    # though they are delivered as a handful of merged paragraphs.
    assert stats["included"] == 30
    assert stats["total"] == 30
    assert stats["truncated"] is False


# ── disclosure: a partial transcript never reads as a complete one ───────────

def test_bundle_declares_partial_coverage_with_real_counts(lecture_db) -> None:
    _, stamps = lecture_db
    bits, stats = _collect_learning_audio_bits(*_window(stamps), {})
    assert stats["truncated"] is True, "this fixture must exceed the budget"

    bundle = format_learning_bundle(
        sessions=[{
            "id": 1, "kind": "courseware_view", "started_at": stamps[0],
            "ended_at": stamps[-1], "duration_min": LECTURE_MINUTES,
            "confidence": 0.95, "title": "2026.2版 OpenVINO™ 的新功能",
            "topics": [], "concepts": [], "apps": [], "queries": [], "urls": [],
            "sample_text": "", "reason": "always-learning rule: openvino中文社区",
        }],
        key_texts=[], edited_files=[], audio_bits=bits,
        range_start=stamps[0], range_end=stamps[-1], audio_stats=stats,
    )

    assert "PARTIAL TRANSCRIPT" in bundle
    assert f"{stats['included']} of {stats['total']}" in bundle
    # The model must be told the gaps are dropped content, not silence, and be
    # required to disclose it — otherwise it reports a complete lecture outline.
    assert "NOT silence" in bundle
    assert "数据说明" in bundle
    assert "CHRONOLOGICAL" in bundle


def test_bundle_does_not_cry_truncation_when_complete(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    db = DatabaseManager(tmp_path / "data.db")
    stamps = _seed_lecture(db, n=25)
    bits, stats = _collect_learning_audio_bits(*_window(stamps), {})
    bundle = format_learning_bundle(
        sessions=[{
            "id": 1, "kind": "courseware_view", "started_at": stamps[0],
            "ended_at": stamps[-1], "duration_min": 2, "confidence": 0.9,
            "title": "t", "topics": [], "concepts": [], "apps": [],
            "queries": [], "urls": [], "sample_text": "", "reason": "",
        }],
        key_texts=[], edited_files=[], audio_bits=bits,
        range_start=stamps[0], range_end=stamps[-1], audio_stats=stats,
    )
    assert "PARTIAL TRANSCRIPT" not in bundle
    assert f"{stats['total']} of {stats['total']} (complete)" in bundle


def test_bundle_flags_unknown_order_on_the_search_fallback() -> None:
    """The /search fallback is relevance-ordered; the model must be told."""
    bundle = format_learning_bundle(
        sessions=[{
            "id": 1, "kind": "courseware_view", "started_at": "a", "ended_at": "b",
            "duration_min": 5, "confidence": 0.9, "title": "t", "topics": [],
            "concepts": [], "apps": [], "queries": [], "urls": [],
            "sample_text": "", "reason": "",
        }],
        key_texts=[], edited_files=[], audio_bits=["x: a", "y: b"],
        range_start="a", range_end="b",
        audio_stats={"total": 0, "included": 2, "truncated": True,
                     "source": "search", "ordered": False},
    )
    assert "ORDER NOT GUARANTEED" in bundle
    assert "total in window unknown" in bundle


def test_manual_session_without_apps_collects_ocr_from_its_exact_span(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    DatabaseManager(tmp_path / "data.db")
    calls: list[tuple[str, str, dict]] = []

    def fake_search(start: str, end: str, **kwargs):
        calls.append((start, end, kwargs))
        return []

    monkeypatch.setattr(agent, "_do_content_search", fake_search)
    session = {
        "kind": "study_other",
        "apps": [],
        "started_at": "2026-08-16T09:00:00+08:00",
        "ended_at": "2026-08-16T10:30:00+08:00",
    }

    assert _collect_courseware_ocr_lines(
        session["started_at"], session["ended_at"], [session]
    ) == []
    assert calls == [(
        session["started_at"],
        session["ended_at"],
        {"limit": 35, "app_name": None, "content_type": "ocr", "verbose": False},
    )]


# ── timestamp-format robustness (the bug that zeroed the auto-recap) ─────────

@pytest.mark.parametrize("sep", ["T", " "])
def test_window_bounds_accept_both_iso_separators(lecture_db, sep) -> None:
    """`flush.py` passed a space separator; ' ' < 'T' silently matched nothing."""
    db, stamps = lecture_db
    lo = stamps[0].replace("T", sep, 1)
    hi = stamps[-1].replace("T", sep, 1)
    assert db.count_transcripts_in_range(lo, hi) == len(stamps)
    assert len(db.transcripts_in_range(lo, hi)) == len(stamps)


def test_window_excludes_rows_outside_the_range(lecture_db) -> None:
    db, stamps = lecture_db
    later = (LECTURE_START + timedelta(hours=5)).replace(microsecond=0).isoformat()
    db._conn.execute(  # noqa: SLF001
        "INSERT INTO audio_transcriptions(timestamp, transcription) VALUES (?,?)",
        (later, "无关的后续录音"),
    )
    assert db.count_transcripts_in_range(*_window(stamps)) == len(stamps)


def test_long_session_is_retrieved_whole(tmp_path, monkeypatch) -> None:
    """Retrieval must not cap. A study session is read end to end or not at all.

    An earlier 2000-row default silently kept only the first ~2 hours, and the
    loss was invisible because the total came from a separate COUNT(*) — the same
    class of bug as the old DESC cap, pointing the other way.
    """
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    db = DatabaseManager(tmp_path / "data.db")
    stamps = _seed_lecture(db, n=2600)          # ~2.9 h at 4 s per utterance
    lo, hi = _window(stamps)
    assert db.count_transcripts_in_range(lo, hi) == 2600
    assert len(db.transcripts_in_range(lo, hi)) == 2600


def test_full_transcript_keeps_everything_the_prompt_dropped(tmp_path, monkeypatch) -> None:
    """The archived transcript is unbudgeted even when the prompt copy is not."""
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    from deskmate.apps.agent import _full_transcript_text

    db = DatabaseManager(tmp_path / "data.db")
    stamps = _seed_lecture(db, n=1500)
    lo, hi = _window(stamps)

    _, stats = _collect_learning_audio_bits(lo, hi, {})
    assert stats["truncated"] is True, "this fixture must exceed the prompt budget"

    full = _full_transcript_text(lo, hi)
    # Every seeded utterance is numbered, so the first and last prove the span
    # survived, and the length proves the middle did too.
    assert "第0段讲解" in full
    assert "第1499段讲解" in full
    assert len(full) > stats["included"] * 10


def test_blank_transcriptions_are_not_counted(lecture_db) -> None:
    db, stamps = lecture_db
    db._conn.execute(  # noqa: SLF001
        "INSERT INTO audio_transcriptions(timestamp, transcription) VALUES (?,?)",
        (stamps[5], "   "),
    )
    assert db.count_transcripts_in_range(*_window(stamps)) == len(stamps)


# ── the sampler itself ───────────────────────────────────────────────────────

def test_select_spanning_keeps_ends_and_preserves_order() -> None:
    xs = list(range(1000))
    got = select_spanning(xs, 50)
    assert len(got) == 50
    assert got[0] == 0 and got[-1] == 999
    assert got == sorted(got)


def test_select_spanning_is_a_noop_under_budget() -> None:
    xs = [1, 2, 3]
    assert select_spanning(xs, 10) == xs
    assert select_spanning(xs, len(xs)) == xs


def test_audio_budget_constant_is_well_above_the_old_cap() -> None:
    """Documents intent: the old behaviour was 40 rows of a whole class."""
    assert _AUDIO_MAX_LINES >= 200
