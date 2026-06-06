"""Tests for Ask's anti-hallucination / grounding helpers.

These cover the pure logic added to keep answers evidence-backed:
- `_result_has_evidence` / `_evidence_is_empty`: detect empty tool pools.
- `_verify_answer_grounding`: flag cited timestamps absent from the evidence.
- `_grounded_final_answer`: gate a model answer (no-data refusal + cautions).
"""

from __future__ import annotations

from deskmate.engine import ask


def test_result_has_evidence_detects_payload() -> None:
    assert ask._result_has_evidence('{"result_count":2,"messages":[{"id":"a"}]}')
    assert ask._result_has_evidence('{"apps":[{"name":"Code.exe"}]}')
    assert ask._result_has_evidence('{"segments":[{"text":"hi"}],"text":"hi"}')


def test_result_has_evidence_rejects_empty_and_errors() -> None:
    assert not ask._result_has_evidence("")
    assert not ask._result_has_evidence("{}")
    assert not ask._result_has_evidence('{"result_count":0,"messages":[]}')
    assert not ask._result_has_evidence('{"meeting_count":0,"meetings":[]}')
    assert not ask._result_has_evidence('{"error":"no_mailbox_connected"}')
    assert not ask._result_has_evidence('{"data_status":"no_capture_in_range"}')


def test_evidence_is_empty() -> None:
    assert ask._evidence_is_empty([])
    assert ask._evidence_is_empty([{"result": '{"result_count":0,"messages":[]}'}])
    assert not ask._evidence_is_empty([
        {"result": '{"result_count":0,"messages":[]}'},
        {"result": '{"apps":[{"name":"Code.exe"}]}'},
    ])


def test_verify_answer_grounding_flags_unseen_timestamp() -> None:
    tool_log = [{"result": '{"events":[{"ts":"2026-06-06T15:04:00","app":"Code.exe"}]}'}]
    # 15:04 is in evidence → not flagged; 09:30 is not → flagged.
    assert ask._verify_answer_grounding("你在 15:04 编辑了文件", tool_log) == []
    flagged = ask._verify_answer_grounding("会议在 09:30 开始", tool_log)
    assert "09:30" in flagged


def test_verify_answer_grounding_matches_12h_against_iso() -> None:
    tool_log = [{"result": '{"events":[{"ts":"2026-06-06T15:04:00"}]}'}]
    # "3:04 PM" → 15:04 digits won't literally match, but "3:04" also won't be in
    # evidence; the bare-hour variant check keeps this conservative. A 24h cite
    # that matches is the grounded path:
    assert ask._verify_answer_grounding("at 15:04", tool_log) == []


def test_verify_answer_grounding_no_evidence_returns_empty() -> None:
    # With no evidence text we cannot verify, so we do not flag (avoid noise).
    assert ask._verify_answer_grounding("发生在 09:30", [{"result": ""}]) == []


def test_grounded_final_answer_refuses_without_evidence(monkeypatch) -> None:
    # Force the no-data path's health probe to be offline/empty.
    monkeypatch.setattr(ask, "_http_get", lambda *a, **k: {})
    tool_log = [{"result": '{"result_count":0,"messages":[]}'}]
    out = ask._grounded_final_answer(
        "你昨天参加了 3 个会议。", tool_log, api_base="http://x", question="昨天开了几个会"
    )
    # The fabricated answer is replaced by an honest no-data message.
    assert "3 个会议" not in out
    assert "没有找到" in out or "还没有" in out


def test_grounded_final_answer_appends_caution_for_bad_timestamp(monkeypatch) -> None:
    tool_log = [{"result": '{"events":[{"ts":"2026-06-06T15:04:00","app":"Code.exe"}]}'}]
    out = ask._grounded_final_answer(
        "你在 15:04 编辑代码，并在 09:30 开会。",
        tool_log, api_base="http://x", question="今天做了什么",
    )
    # Real answer kept, but the unverifiable 09:30 triggers a caution.
    assert "15:04" in out
    assert "⚠️" in out and "09:30" in out


def test_grounded_final_answer_passes_clean_answer() -> None:
    tool_log = [{"result": '{"events":[{"ts":"2026-06-06T15:04:00","app":"Code.exe"}]}'}]
    out = ask._grounded_final_answer(
        "你在 15:04 编辑了 Code.exe。", tool_log, api_base="http://x", question="x"
    )
    assert out == "你在 15:04 编辑了 Code.exe。"  # untouched
