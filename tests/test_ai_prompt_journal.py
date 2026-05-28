"""Unit tests for the ai-prompt-journal app helpers.

These tests cover only pure-Python paths (block parsing, prompt-key dedup,
append behaviour) — they do not require Ollama or the pc_assistant API.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

_APP_DIR = Path(__file__).resolve().parents[1] / "apps" / "ai-prompt-journal"
_APPS_DIR = _APP_DIR.parent


def _load_app_module(tmp_home: Path) -> ModuleType:
    """Load apps/ai-prompt-journal/app.py with PC_ASSISTANT_HOME isolated."""
    os.environ["PC_ASSISTANT_HOME"] = str(tmp_home)
    if str(_APPS_DIR) not in sys.path:
        sys.path.insert(0, str(_APPS_DIR))
    # Drop any prior cached load so PC_ASSISTANT_HOME takes effect.
    sys.modules.pop("ai_prompt_journal_app", None)
    spec = importlib.util.spec_from_file_location(
        "ai_prompt_journal_app", _APP_DIR / "app.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["ai_prompt_journal_app"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def app(tmp_path: Path) -> ModuleType:
    return _load_app_module(tmp_path)


def test_prompt_key_normalizes_whitespace_and_quoting(app: ModuleType) -> None:
    body_a = "> hello world\n> how are you?"
    body_b = "> Hello   World\n> how are you?"
    body_c = "hello world how are you?"
    key_a = app._prompt_key(body_a)
    assert key_a == app._prompt_key(body_b), "case + whitespace must not change key"
    assert key_a == app._prompt_key(body_c), "blockquote prefix must be stripped"
    assert len(key_a) <= 80


def test_split_blocks_extracts_each_block(app: ModuleType) -> None:
    report = (
        "## 09:15 — Claude — refactor agent\n"
        "**Category**: coding | **Length**: short\n\n"
        "> can you refactor agent.py to use dataclasses?\n\n"
        "---\n\n"
        "## 09:42 — ChatGPT — sql window function\n"
        "**Category**: coding | **Length**: short\n\n"
        "> how do row_number and rank differ in sqlite?\n\n"
        "---\n"
    )
    blocks = app._split_blocks(report)
    assert len(blocks) == 2
    headers = [h for h, _ in blocks]
    assert headers[0].startswith("09:15")
    assert headers[1].startswith("09:42")
    assert "refactor agent.py" in blocks[0][1]
    assert "row_number" in blocks[1][1]


def test_split_blocks_ignores_no_new_prompts(app: ModuleType) -> None:
    assert app._split_blocks("NO_NEW_PROMPTS") == []
    assert app._split_blocks("") == []


def test_append_new_blocks_dedups_and_writes_header(
    app: ModuleType, tmp_path: Path
) -> None:
    journal = tmp_path / "today.md"

    blocks = [
        (
            "09:15 — Claude — refactor agent",
            "**Category**: coding | **Length**: short\n\n> refactor agent.py please",
        ),
        (
            "09:42 — ChatGPT — sql window",
            "**Category**: coding | **Length**: short\n\n> how do row_number and rank differ?",
        ),
    ]
    appended = app._append_new_blocks(journal, blocks)
    assert appended == 2
    text = journal.read_text(encoding="utf-8")
    assert text.startswith("---\n")  # YAML front matter from _ensure_journal_header
    assert "# AI Prompts" in text
    assert text.count("\n---\n") >= 2  # 1 in front matter + 1 per block separator

    # Re-appending the same blocks must dedup.
    appended_again = app._append_new_blocks(journal, blocks)
    assert appended_again == 0

    # Appending a genuinely new block adds exactly one entry.
    new_blocks = [
        (
            "10:05 — Gemini — explain mtls",
            "**Category**: research | **Length**: short\n\n> explain mutual tls in one paragraph",
        ),
    ]
    appended_new = app._append_new_blocks(journal, new_blocks)
    assert appended_new == 1
    assert "mutual tls" in journal.read_text(encoding="utf-8")


def test_append_zero_blocks_does_not_create_file(
    app: ModuleType, tmp_path: Path
) -> None:
    journal = tmp_path / "today.md"
    appended = app._append_new_blocks(journal, [])
    assert appended == 0
    assert not journal.exists()


def test_today_journal_under_pc_assistant_home(
    app: ModuleType, tmp_path: Path
) -> None:
    path = app._today_journal()
    assert path.parent == tmp_path / "apps" / app.APP_NAME / "journal"
    assert path.suffix == ".md"
