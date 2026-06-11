"""Unit tests for the todo-list and email-compose output parsers.

These guard the post-processing that turns a small model's markdown into
structured data (DB rows / a sendable draft). The app folders contain hyphens,
so each module is loaded directly from its file path.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_APPS_DIR = Path(__file__).resolve().parents[1] / "deskmate" / "apps"


def _load_app(folder: str, mod_name: str) -> ModuleType:
    # app.py uses absolute deskmate.apps.* imports, so it loads cleanly from
    # file under any module name (the hyphenated folder isn't an importable pkg).
    sys.modules.pop(mod_name, None)
    spec = importlib.util.spec_from_file_location(mod_name, _APPS_DIR / folder / "app.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def todo_app() -> ModuleType:
    return _load_app("todo-list", "todo_list_app")


@pytest.fixture()
def compose_app() -> ModuleType:
    return _load_app("email-compose", "email_compose_app")


# ── todo-list: parse_todos robustness ─────────────────────────────────────

def test_parse_todos_standard_line(todo_app: ModuleType) -> None:
    md = (
        "## Todolist\n"
        "- [ ] Send the budget numbers — from Alice — due 2026-06-03 "
        "— source: email:Outlook — priority: high\n"
    )
    todos = todo_app.parse_todos(md)
    assert len(todos) == 1
    todo = todos[0]
    assert todo["text"] == "Send the budget numbers"
    assert todo["status"] == "open"
    assert todo["source"] == "email"
    assert todo["source_ref"] == "Alice"
    assert todo["due"] == "2026-06-03"
    assert todo["priority"] == "H"


@pytest.mark.parametrize(
    "line",
    [
        "* [x] Ship release notes",       # asterisk bullet, checked
        "- [X] Ship release notes",       # capital X
        "- [✓] Ship release notes",       # unicode check
        "-  [ x ] Ship release notes",    # extra spacing inside brackets
    ],
)
def test_parse_todos_accepts_checked_variants(todo_app: ModuleType, line: str) -> None:
    todos = todo_app.parse_todos(line + "\n")
    assert len(todos) == 1
    assert todos[0]["status"] == "done"
    assert todos[0]["text"] == "Ship release notes"


def test_parse_todos_accepts_empty_brackets(todo_app: ModuleType) -> None:
    # An empty "[]" (no space) is a common small-model variant; it must still
    # parse as an unchecked todo rather than being silently dropped.
    todos = todo_app.parse_todos("- [] Follow up with vendor\n")
    assert len(todos) == 1
    assert todos[0]["status"] == "open"
    assert todos[0]["text"] == "Follow up with vendor"


def test_parse_todos_dedup_key_separates_senders(todo_app: ModuleType) -> None:
    md = (
        "- [ ] Fix bug #123 — from Alice — source: email:Outlook — priority: high\n"
        "- [ ] Fix bug #123 — from Bob — source: email:Outlook — priority: high\n"
    )
    todos = todo_app.parse_todos(md)
    assert len(todos) == 2
    keys = {t["dedup_key"] for t in todos}
    assert len(keys) == 2, "same task from different senders must not collapse"


def test_parse_todos_urgent_priority(todo_app: ModuleType) -> None:
    todos = todo_app.parse_todos(
        "- [ ] Escalate outage — from Ops — source: meeting:Incident — priority: urgent\n"
    )
    assert todos[0]["priority"] == "H"
    assert todos[0]["source"] == "meeting"


# ── email-compose: _parse_draft robustness ────────────────────────────────

def test_parse_draft_standard(compose_app: ModuleType) -> None:
    report = (
        "## Subject\nProject kickoff next week\n\n"
        "## Body\nHi team,\n\nLet's meet Monday.\n\n"
        "## Alternatives\n- Variation A\n"
    )
    subject, body = compose_app._parse_draft(report)
    assert subject == "Project kickoff next week"
    assert "Let's meet Monday." in body
    assert "Variation A" not in body, "Alternatives must not leak into the body"


def test_parse_draft_inline_subject(compose_app: ModuleType) -> None:
    report = "## Subject: Quick sync on budget\n\n## Body\nHello.\n"
    subject, body = compose_app._parse_draft(report)
    assert subject == "Quick sync on budget"
    assert body == "Hello."


def test_parse_draft_strips_markdown_and_quotes(compose_app: ModuleType) -> None:
    report = '## Subject\n- **"Re: Q4 planning"**\n\n## Body\nText.\n'
    subject, _ = compose_app._parse_draft(report)
    assert subject == "Re: Q4 planning"


def test_parse_draft_drops_leading_subject_label(compose_app: ModuleType) -> None:
    report = "## Subject\nSubject: Weekly update\n\n## Body\nText.\n"
    subject, _ = compose_app._parse_draft(report)
    assert subject == "Weekly update"
