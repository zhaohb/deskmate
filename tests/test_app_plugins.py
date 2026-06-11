"""User app-plugin discovery: ~/.deskmate/apps/plugins shadowing + guards."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def plugin_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point DESKMATE_HOME at a temp dir and return its plugin root."""
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    plugins = tmp_path / "apps" / "plugins"
    plugins.mkdir(parents=True)
    return plugins


def _make_app(folder: Path, *, schedule: str = "manual", with_app_py: bool = True) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "pipe.md").write_text(
        f"---\ntitle: {folder.name}\nschedule: {schedule}\n---\nbody\n", encoding="utf-8"
    )
    if with_app_py:
        (folder / "app.py").write_text("print('ok')\n", encoding="utf-8")


def test_user_apps_dir_under_home(plugin_home: Path) -> None:
    from deskmate import paths

    assert paths.user_apps_dir() == plugin_home


def test_discovers_user_plugin_alongside_builtins(plugin_home: Path) -> None:
    from deskmate import paths

    _make_app(plugin_home / "my-report")
    names = {p.name for p in paths.discover_app_dirs()}
    assert "my-report" in names
    assert "day-recap" in names  # built-in still discovered


def test_find_app_dir_resolves_plugin(plugin_home: Path) -> None:
    from deskmate import paths

    _make_app(plugin_home / "my-report")
    found = paths.find_app_dir("my-report")
    assert found == plugin_home / "my-report"


def test_user_plugin_shadows_builtin(plugin_home: Path) -> None:
    from deskmate import paths

    _make_app(plugin_home / "day-recap")
    # find_app_dir returns the user copy, not the built-in.
    assert paths.find_app_dir("day-recap") == plugin_home / "day-recap"
    # And discovery lists day-recap exactly once.
    day_recaps = [p for p in paths.discover_app_dirs() if p.name == "day-recap"]
    assert len(day_recaps) == 1
    assert day_recaps[0] == plugin_home / "day-recap"


def test_find_app_dir_rejects_path_traversal(plugin_home: Path) -> None:
    from deskmate import paths

    assert paths.find_app_dir("../secrets") is None
    assert paths.find_app_dir("a/b") is None
    assert paths.find_app_dir("..") is None
    assert paths.find_app_dir("") is None


def test_unknown_app_returns_none(plugin_home: Path) -> None:
    from deskmate import paths

    assert paths.find_app_dir("does-not-exist") is None


def test_scheduled_plugin_is_discovered(plugin_home: Path) -> None:
    from deskmate.engine.app_schedules import discover_scheduled_apps

    _make_app(plugin_home / "hourly-plug", schedule="every 1h")
    scheduled = {name: spec for name, spec in discover_scheduled_apps()}
    assert "hourly-plug" in scheduled
    assert scheduled["hourly-plug"].interval_seconds == 3600


def test_plugin_without_app_py_not_scheduled(plugin_home: Path) -> None:
    from deskmate.engine.app_schedules import discover_scheduled_apps

    _make_app(plugin_home / "broken", schedule="every 1h", with_app_py=False)
    scheduled = {name for name, _ in discover_scheduled_apps()}
    assert "broken" not in scheduled
