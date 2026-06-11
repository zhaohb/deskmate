"""Tests for user-configurable app schedules."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deskmate.engine.app_schedules import (
    effective_schedule,
    parse_interval_seconds,
    schedules_path,
    validate_schedule_payload,
)


@pytest.fixture()
def schedules_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "apps" / "schedules.json"
    import deskmate.engine.app_schedules as mod

    monkeypatch.setattr(mod, "schedules_path", lambda: path)
    return path


def test_parse_interval_seconds() -> None:
    assert parse_interval_seconds("every 1h") == 3600
    assert parse_interval_seconds("every 30m") == 1800
    assert parse_interval_seconds("every 90s") == 90
    assert parse_interval_seconds("manual") is None


def test_validate_interval_payload() -> None:
    entry = validate_schedule_payload({
        "enabled": True,
        "mode": "interval",
        "interval": "every 2h",
        "hours": "8",
    })
    assert entry == {"enabled": True, "mode": "interval", "interval": "every 2h", "hours": "8"}


def test_validate_daily_payload() -> None:
    entry = validate_schedule_payload({
        "enabled": True,
        "mode": "daily",
        "time": "09:30",
    })
    assert entry["mode"] == "daily"
    assert entry["time"] == "09:30"


def test_user_override_beats_pipe_default(schedules_file: Path) -> None:
    schedules_file.parent.mkdir(parents=True, exist_ok=True)
    schedules_file.write_text(
        json.dumps({"day-recap": {"enabled": True, "mode": "daily", "time": "18:00"}}),
        encoding="utf-8",
    )
    eff = effective_schedule("day-recap", "every 1h")
    assert eff.enabled is True
    assert eff.mode == "daily"
    assert eff.display == "daily 18:00"
    assert eff.source == "user"


def test_user_disable_blocks_pipe_default(schedules_file: Path) -> None:
    schedules_file.parent.mkdir(parents=True, exist_ok=True)
    schedules_file.write_text(
        json.dumps({"day-recap": {"enabled": False, "mode": "manual"}}),
        encoding="utf-8",
    )
    eff = effective_schedule("day-recap", "every 1h")
    assert eff.enabled is False
    assert eff.display == "manual"
    assert eff.source == "user"


def test_pipe_default_when_no_user_entry() -> None:
    eff = effective_schedule("legacy-app", "every 1h")
    assert eff.enabled is True
    assert eff.interval_seconds == 3600
    assert eff.source == "default"
