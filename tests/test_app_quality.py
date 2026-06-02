"""Quality tests for ai-prompt-journal and day-recap output helpers.

Pure-Python paths only — no Ollama / DeskMate API required.

Covers:
- ai-prompt-journal deterministic emission helpers in ``apps/agent.py``
  (``_short_topic``, ``_is_prompt_noise``, ``_emit_prompt_blocks``);
- day-recap OCR denoising in
  ``deskmate.engine.day_recap_context.extract_valuable_lines``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_APPS_DIR = Path(__file__).resolve().parents[1] / "apps"


def _load_agent_module() -> ModuleType:
    if str(_APPS_DIR) not in sys.path:
        sys.path.insert(0, str(_APPS_DIR))
    if "agent" in sys.modules:
        return sys.modules["agent"]
    spec = importlib.util.spec_from_file_location("agent", _APPS_DIR / "agent.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["agent"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def agent() -> ModuleType:
    return _load_agent_module()


# ── ai-prompt-journal: topic extraction ──────────────────────────────────

@pytest.mark.parametrize(
    "prompt, expected",
    [
        ("Can you please help me fix the timeline gap?", "fix the timeline gap"),
        ("How do I refactor agent.py to use dataclasses?",
         "refactor agent.py to use dataclasses"),
        ("Please write a unit test for parse_todos", "write a unit test for"),
        ("explain mutual tls in one paragraph", "mutual tls in one paragraph"),
    ],
)
def test_short_topic_strips_filler(agent: ModuleType, prompt: str, expected: str) -> None:
    assert agent._short_topic(prompt) == expected


def test_short_topic_chinese_filler(agent: ModuleType) -> None:
    # Leading "请帮我" should be stripped, keeping the real intent.
    assert agent._short_topic("请帮我优化day recap效果").startswith("优化day")


def test_short_topic_all_filler_falls_back(agent: ModuleType) -> None:
    # If the whole thing is filler, do not return an empty topic.
    assert agent._short_topic("can you please").strip() != ""
    assert agent._short_topic("   ") == "Untitled prompt"


def test_short_topic_caps_length(agent: ModuleType) -> None:
    long = "implement a brand new caching layer for the search subsystem today"
    topic = agent._short_topic(long)
    assert len(topic) <= 48
    assert len(topic.split(" ")) <= 5


# ── ai-prompt-journal: placeholder noise ─────────────────────────────────

@pytest.mark.parametrize(
    "placeholder",
    [
        "Message Claude",
        "message chatgpt",
        "Ask anything...",
        "How can I help you today?",
        "Reply to Claude",
        "Type a message",
    ],
)
def test_is_prompt_noise_drops_placeholders(agent: ModuleType, placeholder: str) -> None:
    assert agent._is_prompt_noise(placeholder) is True


@pytest.mark.parametrize(
    "real_prompt",
    [
        "Message Claude about the standup meeting tomorrow",
        "Ask the team whether we ship on Friday",
        "Reply to Claude with the refactor plan",
        "how can I parallelize the prefetch calls?",
    ],
)
def test_is_prompt_noise_keeps_real_prompts(agent: ModuleType, real_prompt: str) -> None:
    # Real prompts that merely start with a placeholder word must be kept.
    assert agent._is_prompt_noise(real_prompt) is False


def test_emit_prompt_blocks_uses_clean_topic(agent: ModuleType) -> None:
    prompts = [
        {"ts": "09:15", "tool": "Claude",
         "text": "Can you please refactor agent.py to use dataclasses"},
    ]
    out = agent._emit_prompt_blocks(prompts)
    assert out.startswith("## 09:15 — Claude — ")
    assert "Can you please" not in out.splitlines()[0]
    assert "refactor agent.py" in out.splitlines()[0]
    assert "**Category**: coding" in out


# ── ai-habits: evidence density signal ──────────────────────────────────

@pytest.mark.parametrize(
    "hit_count, expected",
    [(0, "none"), (1, "light"), (3, "light"), (4, "moderate"), (10, "moderate"), (11, "heavy")],
)
def test_usage_intensity_buckets(agent: ModuleType, hit_count: int, expected: str) -> None:
    assert agent._usage_intensity(hit_count) == expected


def test_ai_habits_prefetch_includes_usage_intensity(agent: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_search(start: str, end: str, *, limit: int = 20, app_name: str | None = None,
                    q: str | None = None, verbose: bool = False) -> list[dict]:
        if app_name == "Cursor.exe":
            return [
                {"type": "OCR", "content": {"timestamp": "2026-06-02T09:00:00+08:00", "app_name": "cursor.exe", "window_name": "Cursor", "text": "Refactor agent prefetch logic"}},
                {"type": "OCR", "content": {"timestamp": "2026-06-02T09:05:00+08:00", "app_name": "cursor.exe", "window_name": "Cursor", "text": "Fix tests for app output quality"}},
                {"type": "OCR", "content": {"timestamp": "2026-06-02T09:10:00+08:00", "app_name": "cursor.exe", "window_name": "Cursor", "text": "Run pytest for regression checks"}},
                {"type": "OCR", "content": {"timestamp": "2026-06-02T09:15:00+08:00", "app_name": "cursor.exe", "window_name": "Cursor", "text": "Review ai habits usage intensity"}},
            ]
        return []

    monkeypatch.setattr(agent, "_do_content_search", fake_search)
    data, verified = agent._do_ai_habits_prefetch(
        "2026-06-02T09:00:00+08:00", "2026-06-02T10:00:00+08:00"
    )
    assert "Cursor" in verified
    assert "### Cursor | substantive_hits=4 | usage_intensity=moderate" in data


# ── meeting-summary: generic title detection ─────────────────────────────

@pytest.mark.parametrize(
    "title",
    ["Standup", " Weekly   Standup ", "1:1", "one-on-one", "Sync", "Planning", "Retro"],
)
def test_meeting_generic_titles_expanded(agent: ModuleType, title: str) -> None:
    assert agent._is_generic_meeting_title(title) is True


@pytest.mark.parametrize("title", ["Q2 launch review", "deskmate design discussion"])
def test_meeting_meaningful_titles_are_kept(agent: ModuleType, title: str) -> None:
    assert agent._is_generic_meeting_title(title) is False


# ── video-export: large export hint ──────────────────────────────────────

def test_frames_export_suggests_smaller_range_for_large_exports(
    agent: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        agent,
        "_http_post",
        lambda url, body, timeout=60: {
            "success": True,
            "file_path": "C:/tmp/export.mp4",
            "frame_count": 1000,
        },
    )
    out = agent._do_frames_export("2026-06-02T09:00:00+08:00", "2026-06-02T10:00:00+08:00")
    assert "frame_count: 1000" in out
    assert "suggestion:" in out
    assert "lower fps" in out


# ── day-recap: OCR garble filtering ──────────────────────────────────────

def test_extract_valuable_lines_drops_garble() -> None:
    from deskmate.engine.day_recap_context import extract_valuable_lines

    raw = "\n".join([
        "| || | —— |",            # separator garble, no letters
        "•   ·   •   ·",          # scattered glyphs
        "Fixed the timeline gap in capture pipeline",  # real content
        "############",            # symbol run
    ])
    out = extract_valuable_lines(raw)
    assert "Fixed the timeline gap" in out
    assert "||" not in out
    assert "####" not in out


def test_extract_valuable_lines_keeps_cjk() -> None:
    from deskmate.engine.day_recap_context import extract_valuable_lines

    raw = "优化了 day recap 的输出效果"
    out = extract_valuable_lines(raw)
    assert "优化" in out


def test_extract_valuable_lines_keeps_real_text() -> None:
    from deskmate.engine.day_recap_context import extract_valuable_lines

    raw = "500: Internal Server Error when loading the export feature"
    out = extract_valuable_lines(raw)
    assert "Internal Server Error" in out
