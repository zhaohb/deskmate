"""Tests for the training data miner quality gates and the user-profile source.

These build a tiny SQLite DB with the tables the miner reads, then assert that
the mined SFT pairs are clean (length-capped, deduped, language-checked) and
that the new `profile` source synthesizes identity Q&A from habit_profiles.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from deskmate.learning.training.data import (
    _MAX_DUP_OUTPUT,
    _MAX_OUTPUT_CHARS,
    DeskMateTrainingDataMiner,
)


def _make_db(tmp_path: Path) -> Path:
    """Create a minimal DB with the tables the miner queries."""
    db = tmp_path / "t.db"
    con = sqlite3.connect(db)
    con.executescript(
        """
        CREATE TABLE ask_history (
            id INTEGER PRIMARY KEY, question TEXT, answer TEXT,
            feedback INTEGER, created_at TEXT
        );
        CREATE TABLE pipe_executions (
            id INTEGER PRIMARY KEY, pipe_name TEXT, output TEXT,
            status TEXT, started_at TEXT
        );
        CREATE TABLE habit_profiles (
            id INTEGER PRIMARY KEY, day_type TEXT, slot INTEGER, category TEXT,
            top_app TEXT, avg_minutes REAL, frequency REAL, sample_days INTEGER
        );
        CREATE TABLE habit_suggestions (
            id INTEGER PRIMARY KEY, rule_name TEXT, message TEXT,
            context_json TEXT, feedback INTEGER, created_at TEXT
        );
        CREATE TABLE context_events (
            id INTEGER PRIMARY KEY, source TEXT, kind TEXT, app_name TEXT,
            window_title TEXT, summary TEXT, ts TEXT
        );
        """
    )
    con.commit()
    con.close()
    return db


def test_quality_gate_drops_overlong_and_echo(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    con = sqlite3.connect(db)
    # 1 good ask pair, 1 over-length output, 1 echo (input == output).
    con.executemany(
        "INSERT INTO ask_history(question, answer, feedback, created_at) VALUES (?,?,?,?)",
        [
            ("What did I work on today?", "You spent most of today coding in VS Code.", 1, "2026-06-06"),
            ("Give me the long report", "x" * (_MAX_OUTPUT_CHARS + 50), 1, "2026-06-06"),
            ("repeat this please", "repeat this please", 1, "2026-06-06"),
        ],
    )
    con.commit(); con.close()

    miner = DeskMateTrainingDataMiner(db, min_feedback=1, min_chars=8)
    pairs = miner.extract_sft_pairs(sources=["ask"], max_pairs=100)
    miner.close()
    outputs = [p["output"] for p in pairs]
    assert "You spent most of today coding in VS Code." in outputs
    assert all(len(o) <= _MAX_OUTPUT_CHARS for o in outputs)
    assert "repeat this please" not in outputs  # echo dropped


def test_identical_outputs_capped(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    con = sqlite3.connect(db)
    # Many successful pipe runs with the SAME output → should be capped.
    con.executemany(
        "INSERT INTO pipe_executions(pipe_name, output, status, started_at) VALUES (?,?,?,?)",
        [(f"pipe{i}", "The daily report: all systems normal today.", "success", f"2026-06-06T{i:02d}:00")
         for i in range(10)],
    )
    con.commit(); con.close()

    miner = DeskMateTrainingDataMiner(db, min_chars=8)
    pairs = miner.extract_sft_pairs(sources=["pipes"], max_pairs=100)
    miner.close()
    same = [p for p in pairs if "all systems normal" in p["output"]]
    assert len(same) <= _MAX_DUP_OUTPUT


def test_profile_source_synthesizes_identity(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    con = sqlite3.connect(db)
    con.executemany(
        """INSERT INTO habit_profiles
           (day_type, slot, category, top_app, avg_minutes, frequency, sample_days)
           VALUES (?,?,?,?,?,?,?)""",
        [
            ("weekday", 18, "coding", "Code.exe", 40, 0.8, 10),
            ("weekday", 20, "coding", "Code.exe", 30, 0.7, 8),
            ("weekday", 24, "email", "Outlook.exe", 15, 0.6, 7),
            ("weekend", 28, "browsing", "chrome.exe", 25, 0.5, 5),
        ],
    )
    con.commit(); con.close()

    miner = DeskMateTrainingDataMiner(db, min_chars=8)
    pairs = miner.extract_sft_pairs(sources=["profile"], max_pairs=100)
    miner.close()
    assert pairs, "profile source should synthesize identity pairs"
    joined = " ".join(p["output"] for p in pairs)
    assert "Code.exe" in joined  # top app surfaced
    assert any(p["kind"] == "identity" for p in pairs)
    # weekday vs weekend distinction present
    assert any("weekend" in p["input"].lower() for p in pairs)


def test_profile_skipped_when_signal_thin(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    con = sqlite3.connect(db)
    # Only one low-confidence row → not enough to characterize the user.
    con.execute(
        """INSERT INTO habit_profiles
           (day_type, slot, category, top_app, avg_minutes, frequency, sample_days)
           VALUES ('weekday', 18, 'coding', 'Code.exe', 40, 0.5, 1)""",
    )
    con.commit(); con.close()

    miner = DeskMateTrainingDataMiner(db, min_chars=8)
    pairs = miner.extract_sft_pairs(sources=["profile"], max_pairs=100)
    miner.close()
    assert pairs == []


def test_export_jsonl(tmp_path: Path) -> None:
    db = _make_db(tmp_path)
    con = sqlite3.connect(db)
    con.execute(
        "INSERT INTO ask_history(question, answer, feedback, created_at) VALUES (?,?,?,?)",
        ("What's my main editor?", "You mostly use VS Code for development.", 1, "2026-06-06"),
    )
    con.commit(); con.close()

    miner = DeskMateTrainingDataMiner(db, min_chars=8)
    out = tmp_path / "sub" / "dataset.jsonl"
    n = miner.export_jsonl(out, sources=["ask"], max_pairs=100)
    miner.close()
    assert n == 1
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["input"] and obj["output"] and obj["source"] == "ask"


def test_app_outputs_source_mines_markdown(tmp_path, monkeypatch) -> None:
    """App reports on disk become (instruction → report) pairs, long-form OK."""
    import deskmate.learning.training.data as data_mod

    # Point paths.root() at a fake home with two app reports.
    home = tmp_path / "home"
    monkeypatch.setattr(data_mod.paths, "root", lambda: home)
    rec = home / "apps" / "day-recap" / "output" / "20260608T120000"
    rec.mkdir(parents=True)
    long_report = "## Summary\n" + ("我今天主要在 VS Code 里写代码并调试性能。" * 80)
    (rec / "day-recap.md").write_text(long_report, encoding="utf-8")
    prof = home / "apps" / "user-profile" / "output" / "20260608T130000"
    prof.mkdir(parents=True)
    (prof / "user-profile.md").write_text(
        "## 一句话画像\n一位本地大模型开发者。\n\n---\n\n_时间窗：x → y_\n",
        encoding="utf-8",
    )

    db = _make_db(tmp_path)
    miner = DeskMateTrainingDataMiner(db, min_chars=8)
    pairs = miner.extract_sft_pairs(sources=["apps"], max_pairs=100)
    miner.close()

    by_app = {p["app"]: p for p in pairs}
    assert "day-recap" in by_app and "user-profile" in by_app
    # Long-form report kept (over the short-output cap, under the report cap).
    assert len(by_app["day-recap"]["output"]) > _MAX_OUTPUT_CHARS
    # Trailing "_时间窗_" metadata footer stripped from the profile report.
    assert "时间窗" not in by_app["user-profile"]["output"]
    assert all(p["source"] == "app_output" for p in pairs)


def test_app_outputs_empty_when_no_apps_dir(tmp_path, monkeypatch) -> None:
    import deskmate.learning.training.data as data_mod

    monkeypatch.setattr(data_mod.paths, "root", lambda: tmp_path / "empty_home")
    db = _make_db(tmp_path)
    miner = DeskMateTrainingDataMiner(db, min_chars=8)
    pairs = miner.extract_sft_pairs(sources=["apps"], max_pairs=100)
    miner.close()
    assert pairs == []


def test_format_pair_prompt_is_prefix_of_full() -> None:
    """The manual chat format must keep prompt as an exact prefix of full text
    so token-length subtraction yields a correct loss-mask boundary."""
    from deskmate.learning.training.lora import LoRATrainer

    # Build a trainer shell without torch/model: call the formatter directly.
    trainer = LoRATrainer.__new__(LoRATrainer)
    trainer.tokenizer = None  # forces the manual (non-chat-template) path
    prompt, full = trainer._format_pair_with_prompt(
        {"input": "What do I work on?", "output": "Mostly coding."}
    )
    assert full.startswith(prompt)
    assert full[len(prompt):] == "Mostly coding."  # response is exactly the tail
    assert "What do I work on?" in prompt
