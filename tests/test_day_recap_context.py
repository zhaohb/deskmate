"""Tests for multi-day day-recap timestamp helpers."""

from __future__ import annotations

from datetime import date

from pc_assistant.engine.day_recap_context import (
    calendar_days_in_range,
    format_ts_recap,
    range_spans_calendar_days,
)


def test_range_spans_calendar_days() -> None:
    assert range_spans_calendar_days(
        "2026-05-31T00:00:00+08:00",
        "2026-06-02T23:59:59+08:00",
    )
    assert not range_spans_calendar_days(
        "2026-06-02T08:00:00+08:00",
        "2026-06-02T20:00:00+08:00",
    )


def test_calendar_days_in_range_clips_bounds() -> None:
    days = calendar_days_in_range(
        "2026-05-31T12:00:00+08:00",
        "2026-06-02T10:00:00+08:00",
    )
    assert [d[0] for d in days] == [
        date(2026, 5, 31),
        date(2026, 6, 1),
        date(2026, 6, 2),
    ]
    assert days[0][1].startswith("2026-05-31T12:00:00")
    assert days[-1][2].startswith("2026-06-02T10:00:00")


def test_format_ts_recap_with_date() -> None:
    ts = format_ts_recap("2026-05-31T14:30:00+08:00", include_date=True)
    assert ts.startswith("2026-05-31")
    assert "2:30 PM" in ts
