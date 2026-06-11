"""Unit tests for the ai-prompt-journal app helpers.

These tests cover only pure-Python paths (block parsing, prompt-key dedup,
append behaviour) — they do not require Ollama or the DeskMate API.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from datetime import date
from pathlib import Path
from types import ModuleType

import pytest

_APP_DIR = Path(__file__).resolve().parents[1] / "deskmate" / "apps" / "ai-prompt-journal"
_APPS_DIR = _APP_DIR.parent


def _load_app_module(tmp_home: Path) -> ModuleType:
    """Load apps/ai-prompt-journal/app.py with DESKMATE_HOME isolated."""
    os.environ["DESKMATE_HOME"] = str(tmp_home)
    if str(_APPS_DIR) not in sys.path:
        sys.path.insert(0, str(_APPS_DIR))
    # Drop any prior cached load so DESKMATE_HOME takes effect.
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


def test_today_journal_under_deskmate_home(
    app: ModuleType, tmp_path: Path
) -> None:
    path = app._today_journal()
    assert path.parent == tmp_path / "apps" / app.APP_NAME / "journal"
    assert path.suffix == ".md"


def test_header_journal_date_parses_iso_prefix(app: ModuleType) -> None:
    assert app._header_journal_date("2026-05-31 8:40 PM — Cursor — todo") == date(
        2026, 5, 31
    )
    assert app._header_journal_date("8:40 PM — Cursor — todo") is None


def test_append_blocks_by_date_writes_per_day_journal(
    app: ModuleType, tmp_path: Path
) -> None:
    blocks = [
        (
            "2026-05-31 9:00 AM — Cursor — may31",
            "**Category**: other | **Length**: short\n\n> prompt on may 31",
        ),
        (
            "2026-06-02 8:40 PM — Cursor — jun2",
            "**Category**: other | **Length**: short\n\n> prompt on jun 2",
        ),
    ]
    appended = app._append_blocks_by_date(blocks)
    assert appended == 2
    may31 = tmp_path / "apps" / app.APP_NAME / "journal" / "2026-05-31.md"
    jun02 = tmp_path / "apps" / app.APP_NAME / "journal" / "2026-06-02.md"
    assert "may 31" in may31.read_text(encoding="utf-8")
    assert "jun 2" in jun02.read_text(encoding="utf-8")


def test_build_range_display_includes_multi_day_report(app: ModuleType) -> None:
    report = (
        "## 2026-05-31 9:00 AM — Cursor — may31\n"
        "**Category**: other | **Length**: short\n\n"
        "> older prompt\n\n"
        "---\n\n"
        "## 2026-06-02 8:40 PM — Cursor — jun2\n"
        "**Category**: other | **Length**: short\n\n"
        "> newer prompt\n\n"
        "---\n"
    )
    start = "2026-05-31T00:00:00+08:00"
    end = "2026-06-02T23:59:59+08:00"
    display = app._build_range_display(start, end, report)
    assert "2026-05-31" in display and "2026-06-02" in display
    assert "older prompt" in display and "newer prompt" in display
    assert "AI Prompts — 2026-05-31 … 2026-06-02" in display


def test_header_datetime_parses_variants(app: ModuleType) -> None:
    from datetime import date, datetime

    day = date(2026, 6, 6)
    # 12-hour with AM/PM, no date prefix → use journal_day
    assert app._header_datetime("3:11 PM — Tool — t", day) == datetime(2026, 6, 6, 15, 11)
    assert app._header_datetime("11:23 PM — Tool — t", day) == datetime(2026, 6, 6, 23, 23)
    # date prefix overrides journal_day
    assert app._header_datetime("2026-06-02 8:40 PM — X — y", day) == datetime(2026, 6, 2, 20, 40)
    # 24-hour form
    assert app._header_datetime("15:27 — X — y", day) == datetime(2026, 6, 6, 15, 27)
    # unparseable → None (caller keeps the block)
    assert app._header_datetime("no time here", day) is None


def test_merge_journals_filters_to_precise_window(app: ModuleType, tmp_path) -> None:
    """The real bug: 'last 1 hour' must not surface earlier-today prompts."""
    from datetime import date

    journal = app._journal_for_date(date(2026, 6, 6))
    journal.parent.mkdir(parents=True, exist_ok=True)
    journal.write_text(
        "---\ndate: 2026-06-06\n---\n\n# AI Prompts — 2026-06-06\n\n"
        "## 3:11 PM — Copilot — early\n**Category**: other | **Length**: short\n\n"
        "> afternoon prompt\n\n---\n\n"
        "## 11:10 PM — Claude Code (VS Code) — late\n**Category**: other | **Length**: short\n\n"
        "> night prompt\n\n---\n",
        encoding="utf-8",
    )
    # "last 1 hour" window 23:00–23:30 → only the 11:10 PM block
    body = app._merge_journals_in_range(
        "2026-06-06T23:00:00+08:00", "2026-06-06T23:30:00+08:00"
    )
    assert "night prompt" in body
    assert "afternoon prompt" not in body
    # A window covering the afternoon includes only that one
    body2 = app._merge_journals_in_range(
        "2026-06-06T15:00:00+08:00", "2026-06-06T15:30:00+08:00"
    )
    assert "afternoon prompt" in body2
    assert "night prompt" not in body2


# ── agent.py: Claude Code (terminal CLI) detection ───────────────────────────


def _load_agent_module() -> ModuleType:
    """agent.py is a real package module now (deskmate.apps.agent)."""
    import deskmate.apps.agent as agent  # noqa: PLC0415

    sys.modules.setdefault("agent", agent)
    return agent


@pytest.fixture(scope="module")
def agent() -> ModuleType:
    return _load_agent_module()


def _claude_screen(prompt_lines: list[str]) -> str:
    """Build a realistic Claude Code TUI screen with the given composer lines."""
    box = ["\u256d" + "\u2500" * 40 + "\u256e"]
    for i, ln in enumerate(prompt_lines):
        marker = "> " if i == 0 else "  "
        body = (marker + ln).ljust(40)
        box.append("\u2502 " + body + "\u2502")
    box.append("\u2570" + "\u2500" * 40 + "\u256f")
    return (
        "\u273b Welcome to Claude Code\n"
        "\n"
        "  /help for help, /status for your current setup\n"
        "\n"
        "> Previous answer text from Claude here...\n"
        "\n"
        + "\n".join(box)
        + "\n  ? for shortcuts\n"
    )


def test_claude_code_signals_detected(agent: ModuleType) -> None:
    screen = _claude_screen(["refactor agent.py to use dataclasses"])
    assert agent._has_claude_code_signals(screen) is True
    assert agent._has_claude_code_signals("just an ordinary powershell prompt") is False


def test_claude_code_extract_single_line(agent: ModuleType) -> None:
    screen = _claude_screen(["refactor agent.py to use dataclasses"])
    assert (
        agent._extract_claude_code_prompt(screen)
        == "refactor agent.py to use dataclasses"
    )


def test_claude_code_extract_multiline(agent: ModuleType) -> None:
    screen = _claude_screen(["add a feature flag", "then write tests for it"])
    assert (
        agent._extract_claude_code_prompt(screen)
        == "add a feature flag then write tests for it"
    )


def test_claude_code_extract_empty_composer_returns_blank(agent: ModuleType) -> None:
    screen = _claude_screen([""])
    assert agent._extract_claude_code_prompt(screen) == ""


def test_claude_code_extract_placeholder_hint_returns_blank(agent: ModuleType) -> None:
    screen = _claude_screen(['Try "edit the readme"'])
    assert agent._extract_claude_code_prompt(screen) == ""


def test_claude_code_label_is_registered(agent: ModuleType) -> None:
    assert "Claude Code" in agent.AI_PROMPT_JOURNAL_TARGETS


def test_claude_code_vscode_label_is_registered(agent: ModuleType) -> None:
    assert "Claude Code (VS Code)" in agent.AI_PROMPT_JOURNAL_TARGETS
    assert "Claude Code (VS Code)" in agent._AMBIGUOUS_APP_TOOLS


def test_claude_code_vscode_classified_by_focused_name(agent: ModuleType) -> None:
    """The Claude Code extension composer (code.exe) is classified by its name,
    distinct from Copilot, and accepted as a chat context."""
    title = "config.toml - hongbo - Visual Studio Code"
    # Both observed composer names from the VS Code extension.
    for name in (
        "Message input",
        "Chat Input (Agent), edit files in your workspace., Claude Opus 4.8. "
        "Press Enter to send out the request. Use Alt+F1 for Chat Accessibility Help.",
    ):
        tool = agent._classify_tool("code.exe", title, "", "", name)
        assert tool == "Claude Code (VS Code)", name
        assert agent._is_ai_chat_context(tool, title, "", "", name) is True


def test_vscode_copilot_not_misclassified_as_claude(agent: ModuleType) -> None:
    """A Copilot chat composer (no 'claude' in the name) stays VS Code Copilot."""
    title = "main.py - proj - Visual Studio Code"
    tool = agent._classify_tool("code.exe", title, "", "", "Chat Input (Agent), GPT-4o")
    assert tool == "VS Code Copilot"


def test_ordinary_vscode_editing_dropped(agent: ModuleType) -> None:
    """Plain code editing in code.exe (no chat signal) is not a chat context."""
    title = "main.py - proj - Visual Studio Code"
    tool = agent._classify_tool("code.exe", title, "", "", "main.py")
    # It may match code.exe as an ambiguous tool, but without a chat signal it
    # must be rejected as a chat context (so it is never reported as a prompt).
    assert agent._is_ai_chat_context(tool, title, "", "", "main.py") is False

