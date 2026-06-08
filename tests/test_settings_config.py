"""Tests for the UI-editable settings: generic config writer + endpoints."""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from deskmate.config import load as load_config
from deskmate.db import DatabaseManager
from deskmate.engine.api import create_app


# ── generic config writer ────────────────────────────────────────────────
def test_set_config_value_preserves_comments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        "[ollama]\n# keep me\nmodel = \"old:tag\"\nchat_timeout = 600\n",
        encoding="utf-8",
    )
    from deskmate.config import set_config_value

    set_config_value("ollama", "model", "new:tag")
    text = cfg_path.read_text(encoding="utf-8")
    assert 'model = "new:tag"' in text
    assert "# keep me" in text          # comment preserved
    assert "chat_timeout = 600" in text  # sibling key untouched


def test_set_config_value_creates_missing_section(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[ollama]\nmodel = \"x\"\n", encoding="utf-8")
    from deskmate.config import set_config_value

    set_config_value("retention", "frame_days", 14)
    text = cfg_path.read_text(encoding="utf-8")
    assert "[retention]" in text
    assert "frame_days = 14" in text


def test_set_config_value_renders_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[audio]\nlanguages = [\"zh\"]\n", encoding="utf-8")
    from deskmate.config import set_config_value

    set_config_value("audio", "languages", ["zh", "en"])
    assert 'languages = ["zh", "en"]' in cfg_path.read_text(encoding="utf-8")


# ── settings endpoints ───────────────────────────────────────────────────
def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    cfg = load_config()
    db = DatabaseManager(tmp_path / "data.db")
    return TestClient(create_app(cfg=cfg, db=db, daemon=None))


def test_get_settings_returns_schema_with_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    j = client.get("/config/settings").json()
    groups = j["groups"]
    assert groups and all("title" in g and "fields" in g for g in groups)
    # every field carries its current value + the restart flag
    for g in groups:
        for f in g["fields"]:
            assert "value" in f and "restart" in f and "label" in f


def test_post_settings_validates_and_persists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    res = client.post("/config/settings", json={"values": {
        "ollama.chat_timeout": 900,   # restart-needed, valid
        "ollama.chat_timeout_bad": 1,  # unknown key
    }}).json()
    assert "ollama.chat_timeout" in res["saved"]
    assert "ollama.chat_timeout_bad" in res["errors"]
    assert res["needs_restart"] is True
    # persisted to disk
    assert "chat_timeout = 900" in (tmp_path / "config.toml").read_text(encoding="utf-8")


def test_post_settings_rejects_out_of_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    res = client.post("/config/settings", json={"values": {
        "ollama.chat_timeout": 5,  # below min 30
    }}).json()
    assert res["saved"] == []
    assert "ollama.chat_timeout" in res["errors"]
