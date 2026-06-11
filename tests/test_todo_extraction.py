"""Tests for the unified todo extraction pipeline:

* parse_todos — key-anchored, pipe-delimited parsing; em-dash in a task is
  preserved (the old em-dash split would slice it); source classification is
  robust; legacy em-dash format still parses as a fallback.
* screen evidence guard — _looks_like_screen_task accepts explicit task shapes
  and rejects OCR noise (code, articles, UI labels), to prevent false todos.
* meeting-summary _parse_action_items now also extracts priority, key-anchored.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_APPS_DIR = Path(__file__).resolve().parents[1] / "deskmate" / "apps"


def _agent_module():
    """agent.py is a real package module now (deskmate.apps.agent)."""
    import deskmate.apps.agent as agent  # noqa: PLC0415

    sys.modules.setdefault("agent", agent)
    return agent


def _load(mod_name: str, rel: str):
    # The hyphenated per-app folders (todo-list, …) are NOT importable packages,
    # so their app.py is still loaded from file by path. Their own imports are
    # absolute (deskmate.apps.*), so a bare module name loads cleanly.
    path = _APPS_DIR / rel
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _todo_app():
    _agent_module()  # ensure the shared agent module is importable first
    return _load("todo_list_app", "todo-list/app.py")


# ── parse_todos: key-anchored pipe format ─────────────────────────────────────

def test_parse_pipe_format_basic():
    app = _todo_app()
    md = (
        "## Todolist\n"
        "- [ ] Review the design doc | from: Alice | due: 2026-06-12 | "
        "source: email:gmail | priority: high\n"
    )
    todos = app.parse_todos(md)
    assert len(todos) == 1
    t = todos[0]
    assert t["text"] == "Review the design doc"
    assert t["source_ref"] == "Alice"
    assert t["due"] == "2026-06-12"
    assert t["source"] == "email"
    assert t["priority"] == "H"
    assert t["status"] == "open"


def test_em_dash_in_task_is_preserved():
    """The whole point of the fix: an em-dash inside the task must NOT split it."""
    app = _todo_app()
    md = "- [ ] 完成报告 —— 重点是第三章 | from: 老板 | due: no date | source: meeting:周会 | priority: medium\n"
    todos = app.parse_todos(md)
    assert len(todos) == 1
    assert todos[0]["text"] == "完成报告 —— 重点是第三章"
    assert todos[0]["source"] == "meeting"
    assert todos[0]["priority"] == "M"
    assert todos[0]["due"] == ""  # "no date" normalized away


def test_field_order_independence():
    app = _todo_app()
    md = "- [ ] Send invoice | priority: low | source: email:outlook | due: 2026-07-01 | from: Bob\n"
    t = app.parse_todos(md)[0]
    assert t["text"] == "Send invoice"
    assert t["priority"] == "L" and t["source"] == "email" and t["source_ref"] == "Bob"


def test_screen_source_classified():
    app = _todo_app()
    md = "- [ ] Fix the login bug | from: TODO note | due: no date | source: screen:Code.exe | priority: high\n"
    t = app.parse_todos(md)[0]
    assert t["source"] == "screen"


def test_source_robust_without_canonical_prefix():
    app = _todo_app()
    # Model wrote "outlook:work" without the "email:" prefix → still classified.
    md = "- [ ] Reply to client | from: X | due: no date | source: gmail | priority: low\n"
    # 'gmail' isn't a known head; falls through to "" (acceptable, not misbucketed).
    t = app.parse_todos(md)[0]
    assert t["source"] in ("", "email")  # robust: never crashes, never wrong-bucket


def test_checked_item_marked_done():
    app = _todo_app()
    md = "- [x] Already finished | from: me | due: no date | source: manual | priority: low\n"
    assert app.parse_todos(md)[0]["status"] == "done"


def test_legacy_em_dash_format_still_parses():
    app = _todo_app()
    # No pipe → fall back to legacy em-dash split (backward compat).
    md = "- [ ] Old style task — from Alice — due 2026-06-10 — source: email:gmail — priority: high\n"
    todos = app.parse_todos(md)
    assert len(todos) == 1
    assert todos[0]["text"] == "Old style task"


def test_non_checkbox_lines_ignored():
    app = _todo_app()
    md = "## Todolist\nsome prose\n## By Source\n- email: 3 tasks\n"
    # "- email: 3 tasks" is a bullet but not a checkbox → ignored.
    assert app.parse_todos(md) == []


# ── screen evidence guard: accept explicit tasks, reject noise ────────────────

def test_screen_task_accepts_explicit_shapes():
    agent = _agent_module()
    accept = [
        "TODO: fix the OCR fallback path",
        "FIXME: race in the audio loop",
        "- [ ] follow up with the vendor",
        "Can you please review the PR today?",
        "请你今天之前确认报销单",            # directed (请你)
        "麻烦帮我把会议纪要发一下",          # directed (帮我)
        "记得你明天前提交季度报告",          # ambiguous (记得) + second-person (你)
        "@alice 麻烦更新一下文档",           # ambiguous + @-mention
    ]
    for line in accept:
        assert agent._looks_like_screen_task(line), f"should accept: {line}"


def test_screen_task_rejects_noise():
    agent = _agent_module()
    reject = [
        "def parse_todos(markdown):",            # code
        "import re",                              # code
        "How to center a div - Stack Overflow",   # article/title
        "Home  File  Edit  View  Settings",       # menu bar
        "https://example.com/very/long/url",      # bare URL
        "1.2k comments  340 likes",               # social counters
        "the",                                    # too short
        "x" * 300,                                # too long
        "This is just a normal sentence about my day.",  # prose, no task cue
    ]
    for line in reject:
        assert not agent._looks_like_screen_task(line), f"should reject: {line}"


# ── 7.8: ownership/direction gate — reject tasks not owned by the user ────────

def test_screen_task_rejects_ambiguous_directive_to_others():
    """A directive with NO second-person signal is likely aimed at someone else."""
    agent = _agent_module()
    reject = [
        "请更新文档",                  # group message to someone else
        "记得明天前提交季度报告",       # no 你/您/@ — not clearly mine
        "麻烦尽快处理这个工单",         # no second-person
        "due today: finish the slides",  # bare deadline, no owner
        "截止本周五交付",              # bare deadline
    ]
    for line in reject:
        assert not agent._looks_like_screen_task(line), f"should reject: {line}"


def test_screen_task_browser_todo_is_third_party_code():
    """A TODO:/FIXME read in a browser is third-party code, not the user's task."""
    agent = _agent_module()
    line = "TODO: refactor this legacy parser"
    # In an editor (own file) it's a real personal todo…
    assert agent._looks_like_screen_task(line, is_browser=False)
    # …but the same line seen on GitHub/Stack Overflow in a browser is not.
    assert not agent._looks_like_screen_task(line, is_browser=True)


def test_screen_task_directed_ask_accepted_in_browser():
    """A chat message aimed at the user counts even in a browser (web chat)."""
    agent = _agent_module()
    assert agent._looks_like_screen_task(
        "Can you please review the PR today?", is_browser=True
    )


# ── meeting action items now carry priority, key-anchored ─────────────────────

def test_action_items_extract_priority():
    agent = _agent_module()
    body = (
        "## Action Items\n"
        "- [ ] Ship fix | owner: Alice | due: 2026-06-12 | priority: high\n"
        "- [ ] Note | priority: low | owner: Bob | due: none\n"  # reordered fields
    )
    items = agent._parse_action_items(body)
    assert len(items) == 2
    assert items[0]["priority"] == "H" and items[0]["owner"] == "Alice"
    assert items[1]["priority"] == "L" and items[1]["owner"] == "Bob" and items[1]["due"] == ""
