"""Session-scoped knowledge graph enrichment tests."""

from __future__ import annotations

from deskmate.learning_memory import pipeline
from deskmate.learning_memory.extract import ConceptHit, ExtractionResult, LectureItem
from deskmate.learning_memory.graph import edges_from_lecture_items
from deskmate.learning_memory.topics_llm import StructuredExtraction, SubtopicHit, TopicHit


def test_enrichment_returns_only_edges_extracted_from_this_session(monkeypatch) -> None:
    result = ExtractionResult(
        concepts=[],
        lecture_items=[
            LectureItem(
                kind="relation",
                subject="Tokenization",
                content="Embedding",
                source="audio",
                evidence="Tokenization leads to Embedding",
            )
        ],
        topic_summary="nlp",
    )
    monkeypatch.setattr(pipeline, "extract_all", lambda **_kwargs: result)
    monkeypatch.setattr(
        pipeline,
        "extract_structured_llm",
        lambda **_kwargs: StructuredExtraction(
            concepts=[
                ConceptHit(name="Tokenization", topic="NLP"),
                ConceptHit(name="Embedding", topic="NLP"),
            ],
            structure=result.lecture_items,
        ),
    )

    enrichment = pipeline.build_learning_enrichment(
        audio_bits=["09:00: Tokenization leads to Embedding"],
        courseware_ocr_lines=[],
        key_text_blobs=[],
        sessions=[{"id": 7}],
        persist=False,
        use_llm_topics=True,
    )

    assert enrichment["edges"] == [{
        "src_name": "Tokenization",
        "dst_name": "Embedding",
        "rel": "leads_to",
        "evidence": "Tokenization leads to Embedding",
        "weight": 1.0,
    }]
    assert "Tokenization -[leads_to]-> Embedding" in enrichment["prompt_block"]


def test_graph_rejects_real_sentence_fragments_and_weak_evidence() -> None:
    approved = {"生成式ai api", "int4 kv cache compression", "kv cache"}
    items = [
        LectureItem("relation", "我是", "英特尔的", evidence="我是来自于英特尔的"),
        LectureItem(
            "relation", "或者是我们要", "RAG来构建一个本地的知识库",
            evidence="或者是我们要基于RAG来构建一个本地的知识库",
        ),
        LectureItem(
            "relation", "INT4 KV Cache Compression", "KV Cache",
            evidence="通",
        ),
        LectureItem(
            "relation", "INT4 KV Cache Compression", "KV Cache",
            evidence="INT4 KV Cache Compression reduces the KV Cache footprint",
        ),
    ]

    edges = edges_from_lecture_items(items, allowed_nodes=approved)

    assert [(edge.src_name, edge.dst_name) for edge in edges] == [
        ("INT4 KV Cache Compression", "KV Cache")
    ]


def test_pipeline_builds_topic_hierarchy_without_regex_relation_noise(monkeypatch) -> None:
    noisy_fallback = ExtractionResult(
        concepts=[],
        lecture_items=[
            LectureItem("relation", "那大家知道现在像", "Transformer架构的模型", evidence="那大家知道现在像基于Transformer架构的模型")
        ],
        topic_summary="general",
    )
    monkeypatch.setattr(pipeline, "extract_all", lambda **_kwargs: noisy_fallback)
    monkeypatch.setattr(
        pipeline,
        "extract_structured_llm",
        lambda **_kwargs: StructuredExtraction(
            topics=[TopicHit(
                name="OpenVINO 2026.2 新特性",
                confidence=0.95,
                subtopics=[
                    SubtopicHit("生成式AI API", 0.9),
                    SubtopicHit("KV缓存压缩技术", 0.88),
                ],
            )],
            concepts=[ConceptHit(name="INT4 KV Cache Compression", topic="KV缓存压缩技术")],
            structure=[],
        ),
    )

    enrichment = pipeline.build_learning_enrichment(
        audio_bits=["OpenVINO 2026.2 adds INT4 KV Cache Compression"],
        courseware_ocr_lines=[],
        key_text_blobs=[],
        persist=False,
    )

    assert {(edge["src_name"], edge["dst_name"], edge["rel"]) for edge in enrichment["edges"]} == {
        ("OpenVINO 2026.2 新特性", "生成式AI API", "contains"),
        ("OpenVINO 2026.2 新特性", "KV缓存压缩技术", "contains"),
        ("KV缓存压缩技术", "INT4 KV Cache Compression", "contains"),
    }
    assert all("那大家知道现在像" not in edge["src_name"] for edge in enrichment["edges"])
