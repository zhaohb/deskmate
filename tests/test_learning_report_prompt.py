"""Contract tests for the user-learning report instructions."""

from __future__ import annotations

from pathlib import Path

PIPE_MD = (
    Path(__file__).parents[1]
    / "deskmate"
    / "apps"
    / "user-learning"
    / "pipe.md"
)
APP_JS = Path(__file__).parents[1] / "deskmate" / "ui" / "static" / "app.js"
AGENT_PY = Path(__file__).parents[1] / "deskmate" / "apps" / "agent.py"


def test_learning_report_requires_synthesis_instead_of_transcript_rewriting() -> None:
    prompt = PIPE_MD.read_text(encoding="utf-8")

    assert "不得按时间顺序复述转录" in prompt
    assert "直接引用最多 2 处" in prompt


def test_learning_report_focuses_on_course_content() -> None:
    prompt = PIPE_MD.read_text(encoding="utf-8")

    expected_headings = [
        "## 是否在学习",
        "## 课程总结",
        "## 主要内容",
        "## 讲了什么",
        "## 课程重点",
        "## 知识图谱",
        "## 掌握状态",
        "## 复习重点",
        "## 下一步学习计划",
        "## 数据说明",
    ]
    assert [line for line in prompt.splitlines() if line.startswith("## ")] == expected_headings


def test_knowledge_graph_is_scoped_to_the_current_session() -> None:
    prompt = PIPE_MD.read_text(encoding="utf-8")

    assert "只使用当前 session" in prompt
    assert "不得混入其他 session" in prompt
    assert "未抽取到可靠的概念关系" in prompt
    assert "MUST COVER" in prompt
    assert "shorter invented graph" in prompt
    assert "课件标题" in prompt


def test_each_learning_session_can_generate_its_own_recap() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    assert 'data-generate-recap="${s.id}"' in source
    assert "`/learning/sessions/${sessionId}/recap`" in source
    assert 'method: "POST"' in source


def test_in_flight_ui_jobs_survive_view_rerender() -> None:
    """Status used to live only in list innerHTML, so switching views hid an
    in-flight recap/summary and looked like the job had finished."""
    source = APP_JS.read_text(encoding="utf-8")

    assert "const uiJobs = new Map()" in source
    assert "function beginUiJob" in source
    assert "function endUiJob" in source
    assert "function paintUiJobs" in source
    assert "function hostHasUiJob" in source
    for token in (
        "lrn-recap:${sessionId}",
        "lrn-video:${sessionId}",
        "lrn-journey:${sessionId}",
        "lrn-transcript:${sessionId}",
        "lrn-ocr:${sessionId}",
        "mtg-summary:${id}",
        "mtg-video:${id}",
        "mtg-transcript:${id}",
        "mtg-ocr:${id}",
        "app-run:${appName}",
        'beginUiJob("training"',
        'beginUiJob("doctor"',
        "paintUiJobs()",
    ):
        assert token in source, token
    sessions = source[
        source.index("async function loadLearningSessions") : source.index("function renderSessionGraph")
    ]
    assert "paintUiJobs()" in sessions
    meetings = source[source.index("function renderMeetings") : source.index("function mtgRenderLines")]
    assert "paintUiJobs()" in meetings


def test_session_ocr_falls_back_to_the_exact_span_without_app_metadata() -> None:
    source = AGENT_PY.read_text(encoding="utf-8")

    assert "search_apps: list[str | None] = app_names[:4] or [None]" in source
    assert 'start, end, limit=search_limit, app_name=app, content_type="ocr"' in source


def test_session_graph_is_rendered_as_svg_ui() -> None:
    source = APP_JS.read_text(encoding="utf-8")

    assert "function renderSessionGraph" in source
    assert 'createElementNS("http://www.w3.org/2000/svg", "svg")' in source
    assert 'classList.add("lrn-graph-node")' in source
    assert 'classList.add("lrn-graph-edge")' in source
