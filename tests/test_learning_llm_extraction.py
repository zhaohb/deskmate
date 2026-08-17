"""LLM-sourced concepts, and the gate that keeps junk off the review schedule.

Rule extraction ranked candidates by word shape and recurrence, which cannot
separate a taught term from a button label — 「复习队列」and 「推理优化」are both
four-character Chinese noun phrases. A real session produced a review queue led by
"SCREEN", "ask" and fragments of DeskMate's own interface copy, and no stoplist
fixed it: patching one round still leaked 「生成学习复盘」and 「纸上记笔记」while
losing `Pipeline` and 「量化」 from the actual lecture.

Deciding whether something was *taught* is a semantic call, so the model makes it.
These tests cover the parsing and the persistence gate — the parts that must hold
whatever the model returns, including when it returns nothing usable.
"""

from __future__ import annotations

import json

import pytest

from deskmate.learning_memory.topics_llm import (
    StructuredExtraction,
    _load_json_object,
    _parse_concepts,
    _parse_structure,
    _parse_topics,
)


def _parse(raw: str) -> StructuredExtraction:
    obj = _load_json_object(raw)
    return StructuredExtraction(
        topics=_parse_topics(obj),
        concepts=_parse_concepts(obj),
        structure=_parse_structure(obj),
    )


GOOD = json.dumps({
    "topics": [{
        "name": "OpenVINO 推理优化", "confidence": 0.9,
        "subtopics": [{"name": "模型量化", "confidence": 0.8}],
        "evidence": ["推理性能上的优化"],
    }],
    "concepts": [
        {"name": "模型量化", "topic": "OpenVINO", "is_definition": True,
         "evidence": ["量化之后的精度损失"]},
        {"name": "Pipeline", "topic": "OpenVINO", "is_code": True,
         "evidence": ["生成的 Pipeline"]},
    ],
    "structure": [
        {"kind": "step", "subject": "创建 Core", "content": "先实例化 Core 对象",
         "ordinal": 1, "evidence": ["首先创建 Core"]},
        {"kind": "definition", "subject": "动态 Shape",
         "content": "输入维度在运行时确定"},
    ],
}, ensure_ascii=False)


# ── parsing ──────────────────────────────────────────────────────────────────

def test_parses_all_three_outputs_from_one_reply() -> None:
    r = _parse(GOOD)
    assert [t.name for t in r.topics] == ["OpenVINO 推理优化"]
    assert [c.name for c in r.concepts] == ["模型量化", "Pipeline"]
    assert {s.kind for s in r.structure} == {"step", "definition"}
    assert r.ok


def test_survives_prose_and_code_fences_around_the_json() -> None:
    """Small models wrap JSON in explanation; the payload still has to come out."""
    r = _parse(f"好的，这是结果：\n```json\n{GOOD}\n```\n希望有帮助。")
    assert [c.name for c in r.concepts] == ["模型量化", "Pipeline"]


def test_concept_flags_reach_the_dataclass() -> None:
    by_name = {c.name: c for c in _parse(GOOD).concepts}
    assert by_name["模型量化"].has_definition is True
    assert by_name["Pipeline"].has_code is True
    assert by_name["Pipeline"].has_definition is False


def test_structure_carries_ordinal_and_llm_source() -> None:
    step = next(s for s in _parse(GOOD).structure if s.kind == "step")
    assert step.ordinal == 1
    # `source` distinguishes model-read structure from the regex patterns.
    assert step.source == "llm"


def test_extraction_prompt_demands_technical_depth() -> None:
    """A course outline is not enough — the prompt must pull technical substance."""
    from deskmate.learning_memory.topics_llm import _SYSTEM

    assert "TECHNICAL substance" in _SYSTEM
    assert "INT4" in _SYSTEM  # asks to keep quantitative facts verbatim
    assert "section title" in _SYSTEM  # rejects table-of-contents concepts
    for verb in ("压缩", "部署到", "量化为"):
        assert verb in _SYSTEM  # technical relations, not vague links


def test_relation_uses_explicit_target_concept() -> None:
    raw = json.dumps({"structure": [{
        "kind": "relation",
        "subject": "INT4 KV Cache Compression",
        "target": "KV Cache",
        "content": "reduces memory consumption",
        "evidence": "INT4 compression reduces the KV Cache footprint",
    }]})

    relation = _parse(raw).structure[0]

    assert relation.subject == "INT4 KV Cache Compression"
    assert relation.content == "KV Cache"


def test_duplicate_and_too_short_concepts_are_dropped() -> None:
    raw = json.dumps({"concepts": [
        {"name": "模型量化"}, {"name": "模型量化"}, {"name": "x"}, {"name": ""},
    ]}, ensure_ascii=False)
    assert [c.name for c in _parse(raw).concepts] == ["模型量化"]


def test_unknown_structure_kinds_are_dropped() -> None:
    raw = json.dumps({"structure": [
        {"kind": "definition", "subject": "梯度", "content": "损失对参数的偏导"},
        {"kind": "garbage", "subject": "反向传播", "content": "链式法则"},
    ]}, ensure_ascii=False)
    assert [s.kind for s in _parse(raw).structure] == ["definition"]


def test_structure_needs_both_a_subject_and_content() -> None:
    """Single characters are noise, not a definition."""
    raw = json.dumps({"structure": [
        {"kind": "definition", "subject": "A", "content": "B"},
        {"kind": "definition", "subject": "梯度", "content": ""},
    ]}, ensure_ascii=False)
    assert _parse(raw).structure == []


def test_concept_without_topic_gets_a_bucket() -> None:
    """`topic` is displayed and grouped on, so it must never be empty."""
    raw = json.dumps({"concepts": [{"name": "反向传播"}]}, ensure_ascii=False)
    assert _parse(raw).concepts[0].topic == "general"


# ── failure modes: nothing usable must mean nothing stored ───────────────────

@pytest.mark.parametrize(("label", "raw"), [
    ("empty", ""),
    ("refusal prose", "抱歉，我无法完成这个请求。"),
    ("truncated json", '{"concepts": ['),
    ("array not object", "[1, 2, 3]"),
    ("wrong shape", '{"concepts": "模型量化"}'),
    ("nulls", '{"topics": null, "concepts": null, "structure": null}'),
])
def test_unusable_replies_yield_nothing(label, raw) -> None:
    r = _parse(raw)
    assert r.concepts == []
    assert r.ok is False, label


def test_ok_is_the_gate_for_persistence() -> None:
    """`ok` is what callers check before scheduling anything.

    SM-2 state accumulates: a concept put on a schedule outlives the session that
    produced it, so a failed pass must schedule nothing rather than fall back to
    the regex candidates it was introduced to replace.
    """
    assert StructuredExtraction().ok is False
    assert _parse(GOOD).ok is True


# ── the pipeline seeds reviews only from model concepts ──────────────────────

def test_pipeline_stores_llm_concepts_not_rule_candidates(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    from deskmate.db.manager import DatabaseManager
    from deskmate.learning_memory import pipeline as pipe_mod
    from deskmate.learning_memory.store import LearningStore

    db_file = tmp_path / "data.db"
    DatabaseManager(db_file)

    # Evidence deliberately full of interface text: the rule extractor happily
    # ranks these, and that is exactly what must not be stored.
    ui_noise = [
        "DeskMate - Google Chrome Minimize Maximize Restore Close SCREEN ask",
        "学习会话 待解决的问题 复习队列 识别到的知识点 生成学习复盘",
    ]
    monkeypatch.setattr(
        pipe_mod, "extract_structured_llm",
        lambda **kw: _parse(GOOD),
    )
    out = pipe_mod.build_learning_enrichment(
        audio_bits=["讲解 OpenVINO 的推理优化与模型量化"],
        courseware_ocr_lines=ui_noise,
        key_text_blobs=ui_noise,
        persist=True,
        db_path=db_file,
    )
    assert out["persisted"] is True

    stored = {c["name"] for c in LearningStore(db_file).list_concepts(limit=50)}
    assert stored == {"模型量化", "Pipeline"}
    for junk in ("SCREEN", "ask", "复习队列", "生成学习复盘"):
        assert junk not in stored


def test_pipeline_schedules_nothing_when_the_model_yields_nothing(
    tmp_path, monkeypatch,
) -> None:
    """An empty queue for one report beats junk on a schedule forever."""
    monkeypatch.setenv("DESKMATE_HOME", str(tmp_path))
    from deskmate.db.manager import DatabaseManager
    from deskmate.learning_memory import pipeline as pipe_mod
    from deskmate.learning_memory.store import LearningStore

    db_file = tmp_path / "data.db"
    DatabaseManager(db_file)
    monkeypatch.setattr(
        pipe_mod, "extract_structured_llm", lambda **kw: StructuredExtraction(),
    )
    pipe_mod.build_learning_enrichment(
        audio_bits=["Minimize Maximize Restore Close SCREEN ask 复习队列"],
        courseware_ocr_lines=["学习会话 生成学习复盘 纸上记笔记"],
        key_text_blobs=[],
        persist=True,
        db_path=db_file,
    )
    store = LearningStore(db_file)
    assert store.list_concepts(limit=50) == []
    assert store.list_due_reviews(limit=50) == []
