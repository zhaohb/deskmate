"""Agent tool-driven mode tests."""

from __future__ import annotations

from deskmate.apps.agent import (
    PIPE_TOOL_CONFIG,
    ToolSession,
    _check_search_budget,
    _fill_time_range,
    _meets_minimum_tools,
    _parse_tool_arguments,
)


def test_parse_tool_arguments_json_string() -> None:
    assert _parse_tool_arguments('{"limit": 10}') == {"limit": 10}


def test_fill_time_range_defaults() -> None:
    session = ToolSession(start_iso="2026-01-01T00:00:00+08:00", end_iso="2026-01-01T16:00:00+08:00")
    out = _fill_time_range({"limit": 5}, session)
    assert out["start_time"] == session.start_iso
    assert out["end_time"] == session.end_iso


def test_day_recap_not_tool_driven() -> None:
    """day-recap uses prefetch+single-shot, so it should NOT be in PIPE_TOOL_CONFIG."""
    assert "day-recap" not in PIPE_TOOL_CONFIG


def test_search_budget_ai_habits() -> None:
    cfg = PIPE_TOOL_CONFIG["ai-habits"]
    session = ToolSession(
        start_iso="a", end_iso="b", pipe_name="ai-habits", search_calls=cfg.max_search,
    )
    assert _check_search_budget(session) is not None
