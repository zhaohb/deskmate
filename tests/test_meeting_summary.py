"""Meeting summaries built from the meeting's own span.

The old flow summarized "the meeting that just ended" from detector-linked
transcript segments, which a hand-declared meeting never has. These cover the
parts that must hold with no model: evidence assembly from speech plus screen,
the reply parsing, and the failure modes that must not fabricate a record.
"""

from __future__ import annotations

import json

from deskmate.meeting.summary import (
    build_evidence,
    build_meeting_summary,
    dedup_key,
    has_enough_evidence,
)

SPEECH = [
    {"ts": "2026-08-18T10:00:00+08:00", "text": "先过一下这周的发布计划"},
    {"ts": "2026-08-18T10:01:00+08:00", "text": "上线前还要跑一轮回归"},
    {"ts": "2026-08-18T10:02:00+08:00", "text": "决定把上线时间推迟到下周三"},
]
SCREEN = [
    {"ts": "2026-08-18T10:01:00+08:00", "text": "Release checklist: staging sign-off, rollback plan"},
]


def test_evidence_keeps_speech_and_screen_apart() -> None:
    """The model must know which lines were spoken and which were on screen."""
    ev = build_evidence(SPEECH, SCREEN)
    assert "SPEECH:" in ev
    assert "SCREEN TEXT:" in ev
    assert "10:00" in ev  # speech carries its clock time
    assert "Release checklist" in ev


def test_evidence_is_empty_when_nothing_was_captured() -> None:
    assert build_evidence([], []) == ""


def test_summary_reports_no_evidence_instead_of_inventing_one() -> None:
    out = build_meeting_summary(
        transcript_rows=[], ocr_rows=[],
        started_at="2026-08-18T10:00:00+08:00", ended_at="2026-08-18T10:30:00+08:00",
    )
    assert out["note"] == "no_evidence"
    assert out["summary"] == ""
    assert out["todos"] == []
    assert out["evidence"] == {"transcript_rows": 0, "screen_rows": 0}


def test_a_stray_screenshot_is_not_a_meeting(monkeypatch) -> None:
    """Observed: one OCR frame of an editor produced a full fabricated record —
    decisions, owners and due dates that were never said. Below the floor the
    model is never asked."""
    from deskmate.engine import llm as llm_mod

    def _must_not_run(*_a, **_k):
        raise AssertionError("the model was asked to summarize nothing")

    monkeypatch.setattr(llm_mod, "chat_ollama", _must_not_run)

    out = build_meeting_summary(
        transcript_rows=[], ocr_rows=[SCREEN[0]],
        started_at="2026-08-18T10:00:00+08:00", ended_at="2026-08-18T10:01:00+08:00",
    )

    assert out["note"] == "no_evidence"
    assert out["todos"] == []


def test_evidence_floor_accepts_a_real_conversation() -> None:
    assert has_enough_evidence(speech_rows=3, screen_rows=0) is True
    assert has_enough_evidence(speech_rows=0, screen_rows=8) is True   # screen-share
    assert has_enough_evidence(speech_rows=2, screen_rows=1) is False  # a blip


def test_summary_parses_the_model_reply(monkeypatch) -> None:
    from deskmate.engine import llm as llm_mod

    monkeypatch.setattr(llm_mod, "resolve_ollama_settings", lambda: ("http://x", "m", 60))
    monkeypatch.setattr(
        llm_mod, "chat_ollama",
        lambda *a, **k: {"content": json.dumps({
            "title": "发布计划评审",
            "summary": "讨论了本周发布计划，并确定推迟上线。",
            "key_points": ["回顾发布检查清单"],
            "decisions": ["上线推迟到下周三"],
            "todos": [{"text": "更新发布时间表", "owner": "张三", "due": "周一"}],
        }, ensure_ascii=False)},
    )

    out = build_meeting_summary(
        transcript_rows=SPEECH, ocr_rows=SCREEN,
        started_at="2026-08-18T10:00:00+08:00", ended_at="2026-08-18T10:30:00+08:00",
    )

    assert out["note"] == ""
    assert out["title"] == "发布计划评审"
    assert out["decisions"] == ["上线推迟到下周三"]
    assert out["todos"] == [{"text": "更新发布时间表", "owner": "张三", "due": "周一"}]
    assert out["evidence"]["transcript_rows"] == 3


def test_summary_soft_fails_when_the_model_is_down(monkeypatch) -> None:
    """A meeting record is never worth crashing on, nor worth faking."""
    from deskmate.engine import llm as llm_mod

    monkeypatch.setattr(llm_mod, "resolve_ollama_settings", lambda: ("http://x", "m", 60))

    def _boom(*_a, **_k):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(llm_mod, "chat_ollama", _boom)

    out = build_meeting_summary(
        transcript_rows=SPEECH, ocr_rows=SCREEN,
        started_at="2026-08-18T10:00:00+08:00", ended_at="2026-08-18T10:30:00+08:00",
    )

    assert out["note"] == "unavailable"
    assert out["summary"] == ""


def test_todo_dedup_key_is_stable_per_meeting() -> None:
    """Regenerating a summary must update follow-ups, not duplicate them."""
    a = dedup_key(7, "更新发布时间表")
    assert a == dedup_key(7, "更新发布时间表 ")  # trailing space is not a new todo
    assert a != dedup_key(8, "更新发布时间表")   # but another meeting's is distinct
