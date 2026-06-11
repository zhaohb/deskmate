"""Ask retrieval enhancements: time-range inference (6.3) + snippet rerank (6.2)."""

from __future__ import annotations

from datetime import datetime

from deskmate.engine import ask


_NOW = datetime(2026, 6, 11, 15, 30, 0)  # Thursday afternoon


def _range(question: str):
    return ask._infer_question_time_range(question, _NOW)


def test_infer_yesterday_and_today() -> None:
    s, e = _range("what did I do yesterday")
    assert s.startswith("2026-06-10") and e.startswith("2026-06-11T00:00")
    s, e = _range("今天干了啥")
    assert s.startswith("2026-06-11T00:00")


def test_infer_this_week_and_last_week() -> None:
    # 2026-06-11 is Thursday; ISO week starts Mon 2026-06-08.
    s, e = _range("what did I work on this week")
    assert s.startswith("2026-06-08")
    s, e = _range("上周做了什么")
    assert s.startswith("2026-06-01") and e.startswith("2026-06-08")


def test_infer_morning_afternoon() -> None:
    s, e = _range("上午在忙什么")
    assert s.startswith("2026-06-11T00:00") and e.startswith("2026-06-11T12:00")
    s, e = _range("what did I do this afternoon")
    assert s.startswith("2026-06-11T12:00")


def test_infer_relative_hours_ago() -> None:
    rng = _range("what was I doing 3 hours ago")
    assert rng is not None
    s, e = rng
    # Centered ~12:30, ±30min window.
    assert s.startswith("2026-06-11T12:00") and e.startswith("2026-06-11T13:00")


def test_infer_relative_chinese_minutes() -> None:
    rng = _range("20 分钟前在看什么")
    assert rng is not None
    s, e = rng
    assert s.startswith("2026-06-11T14:55") or s.startswith("2026-06-11T15:0")


def test_infer_unrecognized_returns_none() -> None:
    assert _range("explain how attention works") is None


def test_question_terms_filters_stopwords() -> None:
    terms = ask._question_terms("what did I do with Alice in Slack")
    assert "alice" in terms and "slack" in terms
    assert "what" not in terms and "did" not in terms


def test_rerank_floats_relevant_snippets_first() -> None:
    snippets = [
        {"text": "random youtube video about cats", "app_name": "chrome.exe"},
        {"text": "message to Alice about the launch", "app_name": "slack.exe"},
        {"text": "spreadsheet of numbers", "app_name": "excel.exe"},
    ]
    terms = ask._question_terms("what did I discuss with Alice in Slack")
    ranked = ask._rerank_by_question(snippets, terms)
    assert ranked[0]["app_name"] == "slack.exe"


def test_slim_summary_reranks_before_truncation() -> None:
    # 7 snippets, only the last is relevant; with max 6 the naive slice would
    # drop it, but reranking keeps it.
    snippets = [{"text": f"unrelated {i}", "app_name": "chrome.exe"} for i in range(6)]
    snippets.append({"text": "the Slack thread with Alice", "app_name": "slack.exe"})
    out = ask._slim_summary(
        {"snippets": snippets}, max_snippets=6, question="Alice Slack thread",
    )
    kept = out["snippets"]
    assert any(s["app_name"] == "slack.exe" for s in kept)
    assert out.get("snippets_truncated") is True
