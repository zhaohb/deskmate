"""Orchestrate extract → LLM topics → sessions/events/edges → BKT/SM-2 → prompt."""

from __future__ import annotations

from typing import Any

from ..console import echo_stderr
from .extract import ExtractionResult, extract_all, extraction_to_dict
from .graph import edges_from_lecture_items, hierarchy_edges
from .store import LearningStore
from .topics_llm import (
    StructuredExtraction,
    TopicHit,
    extract_structured_llm,
    topics_to_dict,
)


def _strip_audio_prefix(line: str) -> str:
    """'09:01 [spk]: text' → 'text'."""
    text = (line or "").strip()
    if ": " in text[:40]:
        return text.split(": ", 1)[1].strip()
    return text


def build_learning_enrichment(
    *,
    audio_bits: list[str],
    courseware_ocr_lines: list[str],
    key_text_blobs: list[str],
    sessions: list[dict[str, Any]] | None = None,
    persist: bool = True,
    verbose: bool = False,
    db_path: Any = None,
    use_llm_topics: bool = True,
) -> dict[str, Any]:
    """Run full learning-memory pipeline for user-learning prefetch."""
    audio_texts = [_strip_audio_prefix(a) for a in audio_bits if a]
    ocr_texts = list(courseware_ocr_lines or [])
    other = [t for t in key_text_blobs if t and len(t.strip()) >= 8]
    sessions = sessions or []

    # Merge session concept tags into other texts for denser extraction.
    for s in sessions:
        for c in s.get("concepts") or []:
            other.append(str(c))
        if s.get("sample_text"):
            other.append(str(s["sample_text"]))

    # Rule extraction still runs, but only to seed the model with candidate terms
    # and to keep `topic_summary` for the prompt. Its concepts are NOT what gets
    # stored: ranking by word shape cannot separate a taught term from a button
    # label, which is how a review queue came to be led by "SCREEN" and "ask".
    result = extract_all(audio_texts=audio_texts, ocr_texts=ocr_texts, other_texts=other)

    llm = StructuredExtraction()
    if use_llm_topics:
        llm = extract_structured_llm(
            audio_texts=audio_texts,
            ocr_texts=ocr_texts,
            other_texts=other,
            concept_hints=[c.name for c in result.concepts],
            verbose=verbose,
        )
        # Attach top LLM topic names onto sessions lacking topics.
        top_names = [t.name for t in llm.topics[:3]]
        if top_names:
            for s in sessions:
                if not s.get("topics"):
                    s["topics"] = list(top_names[:2])
    topics: list[TopicHit] = llm.topics

    # What actually gets stored. Lecture structure prefers the model's reading and
    # falls back to the regex patterns, which are harmless here — a wrong
    # definition only weakens one report, whereas a wrong concept is scheduled by
    # SM-2 and persists across every session that follows.
    concepts = llm.concepts
    lecture_items = llm.structure or result.lecture_items
    graph_nodes = {
        name
        for name in (
            [topic.name for topic in topics]
            + [subtopic.name for topic in topics for subtopic in topic.subtopics]
            + [concept.name for concept in concepts]
        )
        if name
    }
    relation_edges = edges_from_lecture_items(llm.structure, allowed_nodes=graph_nodes)
    session_edges = hierarchy_edges(topics, concepts) + relation_edges

    due: list[dict[str, Any]] = []
    edges_out: list[dict[str, Any]] = [
        {
            "src_name": edge.src_name,
            "dst_name": edge.dst_name,
            "rel": edge.rel,
            "evidence": edge.evidence,
            "weight": edge.weight,
        }
        for edge in session_edges
    ]
    events_out: list[dict[str, Any]] = []
    session_ids: list[int] = []
    persisted = False

    if persist and (concepts or lecture_items or topics or sessions):
        try:
            store = LearningStore(db_file=db_path) if db_path else LearningStore()
            ids = store.upsert_concepts(concepts)
            # Map lecture items to session_ref when timestamps roughly match is hard;
            # use first session ref as weak link when only one session.
            session_ref = f"[{sessions[0]['id']}]" if len(sessions) == 1 else ""
            store.insert_lecture_items(
                lecture_items, concept_ids=ids, session_ref=session_ref
            )
            if topics:
                store.upsert_topics(topics)
            # Topic hierarchy is report structure, not a durable review concept.
            store.upsert_edges(relation_edges, ids)
            if sessions:
                session_ids = store.upsert_slice_sessions(sessions, status="closed")
            # Reviews are seeded ONLY from what the model recognised as taught.
            # A failed or empty pass schedules nothing: SM-2 state accumulates, so
            # junk on a schedule outlives the session that produced it, while an
            # empty queue for one report costs nothing.
            due = store.seed_or_update_reviews(concepts, ids) if concepts else []
            events_out = store.list_events(kind=None, status="open", limit=20)
            persisted = True
            if verbose:
                echo_stderr(
                    f"  [learning_memory] concepts={len(concepts)} (llm) "
                    f"lecture={len(lecture_items)} topics={len(topics)} "
                    f"sessions={len(session_ids)} edges={len(session_edges)} "
                    f"events={len(events_out)} due={len(due)} persisted=1"
                )
        except Exception as exc:  # noqa: BLE001
            if verbose:
                echo_stderr(f"  [learning_memory] persist error: {exc}")
            persisted = False
    elif verbose:
        echo_stderr(
            f"  [learning_memory] concepts={len(concepts)} "
            f"lecture={len(lecture_items)} topics={len(topics)} persisted=0"
        )

    if not due and persist:
        try:
            store = LearningStore(db_file=db_path) if db_path else LearningStore()
            due = store.list_due_reviews(limit=20)
            if not events_out:
                events_out = store.list_events(status="open", limit=20)
        except Exception:  # noqa: BLE001
            pass

    prompt_block = format_enrichment_prompt(
        result,
        due_reviews=due,
        topics=topics,
        edges=edges_out,
        events=events_out,
        concepts=concepts,
        lecture_items=lecture_items,
    )
    return {
        "extraction": extraction_to_dict(result),
        "topics": topics_to_dict(topics),
        "due_reviews": due,
        "edges": edges_out,
        "events": events_out,
        "session_ids": session_ids,
        "prompt_block": prompt_block,
        "persisted": persisted,
    }


def format_enrichment_prompt(
    result: ExtractionResult,
    *,
    due_reviews: list[dict[str, Any]],
    topics: list[TopicHit] | None = None,
    edges: list[dict[str, Any]] | None = None,
    events: list[dict[str, Any]] | None = None,
    concepts: list[Any] | None = None,
    lecture_items: list[Any] | None = None,
) -> str:
    """Pre-computed structure the LLM must ground 讲解重点 / 下一步计划 on.

    ``concepts`` / ``lecture_items`` override what is taken from ``result`` so the
    block describes what was actually stored — the model's reading — rather than
    the regex candidates, which are now only hints.
    """
    concepts = result.concepts if concepts is None else concepts
    lecture_items = result.lecture_items if lecture_items is None else lecture_items
    # Ordered most-actionable-first: the prefetch text is tail-truncated to fit
    # the model budget, and 复习重点 / 下一步学习计划 have no other source than
    # these two queues.
    lines: list[str] = [
        "### Pre-computed learning structure (TRUST — do not invent beyond this)",
        f"Topic buckets: {result.topic_summary}",
        "",
    ]

    lines.append(
        "### Review queue — SM-2 + BKT "
        "(cite as 复习队列; prefer OVERDUE / WEAK / low mastery_tier)"
    )
    if due_reviews:
        for i, r in enumerate(due_reviews[:12], 1):
            due = r.get("due_at") or ""
            overdue = "OVERDUE" if r.get("overdue") else "scheduled"
            weak = "WEAK" if r.get("weak_mastery") else "ok"
            tier = r.get("mastery_tier") or "exposure"
            p_know = r.get("decayed_mastery")
            if p_know is None:
                p_know = r.get("p_mastery") or 0
            lines.append(
                f"- Q{i}. {r.get('name')} | topic={r.get('topic')} | "
                f"{overdue}/{weak} tier={tier} due={due} | urgency={r.get('urgency')} | "
                f"p(know)={float(p_know):.2f} "
                f"| EF={float(r.get('ease_factor') or 0):.2f} interval={r.get('interval_days')}d"
            )
    else:
        lines.append("(queue empty — derive gentle next steps from topics/concepts only)")

    events = events or []
    open_problems = [e for e in events if e.get("kind") in {"problem", "ask_later"}]
    lines.append("")
    lines.append("### Problem / ask-later queue (cite as 问题队列)")
    if open_problems:
        for i, e in enumerate(open_problems[:10], 1):
            lines.append(
                f"- P{i}. [{e.get('kind')}] {e.get('summary')} "
                f"(status={e.get('status')})"
            )
    else:
        lines.append("(queue empty)")

    lines.append("")
    topics = topics or []
    lines.append("### LLM topics / subtopics (cite as 主题:… ; confidence 0–1)")
    if topics:
        for i, t in enumerate(topics[:6], 1):
            lines.append(f"- T{i}. **{t.name}** conf={t.confidence:.2f}")
            for s in t.subtopics[:4]:
                lines.append(f"  - sub: {s.name} conf={s.confidence:.2f}")
            if t.evidence:
                lines.append(f"  evidence: {t.evidence[0][:160]}")
    else:
        lines.append("(no LLM topics — fall back to rule concepts / structure)")

    lines.append("")
    if concepts:
        lines.append("### Extracted concepts (rule-based hints)")
        for c in concepts[:20]:
            flags = []
            if c.has_definition:
                flags.append("has_def")
            if c.has_problem:
                flags.append("problem")
            if c.has_code:
                flags.append("code")
            flag_s = f" [{', '.join(flags)}]" if flags else ""
            lines.append(
                f"- {c.name} | topic={c.topic} | hits={c.count}{flag_s}"
            )
            if c.evidence:
                lines.append(f"  evidence: {c.evidence[0][:160]}")
    else:
        lines.append("### Extracted concepts (rule-based hints)")
        lines.append("(none extracted — keep 讲解重点 short and cite OCR/audio only)")

    defs = [i for i in lecture_items if i.kind == "definition"]
    steps = [i for i in lecture_items if i.kind == "step"]
    rels = [i for i in lecture_items if i.kind == "relation"]

    lines.append("")
    lines.append("### Lecture structure — definitions (cite as 结构:定义)")
    if defs:
        for i, it in enumerate(defs[:12], 1):
            lines.append(f"- D{i}. **{it.subject}** = {it.content} _(src={it.source})_")
    else:
        lines.append("(no definition patterns matched)")

    lines.append("")
    lines.append("### Lecture structure — steps (cite as 结构:步骤)")
    if steps:
        for it in steps[:12]:
            lines.append(f"- S{it.ordinal or '?'}. {it.subject}: {it.content} _(src={it.source})_")
    else:
        lines.append("(no step patterns matched)")

    lines.append("")
    lines.append("### Lecture structure — relations (cite as 结构:关系)")
    if rels:
        for i, it in enumerate(rels[:10], 1):
            lines.append(f"- R{i}. {it.subject} ↔ {it.content} _(src={it.source})_")
    else:
        lines.append("(no relation patterns matched)")

    edges = edges or []
    lines.append("")
    lines.append(
        "### Current-session concept graph edges "
        "(cite as 图谱:先决|相关|对比|导致)"
    )
    if edges:
        for e in edges[:12]:
            lines.append(
                f"- {e.get('src_name')} -[{e.get('rel')}]-> {e.get('dst_name')}"
            )
    else:
        lines.append("(no reliable relations extracted from this session)")

    lines.append("")
    lines.append(
        "INSTRUCTION: 讲解重点 prefer LLM topics + 结构:* + 图谱; "
        "理解要点 explain those subjects and prerequisites; "
        "复习重点 / 下一步学习计划 start from OVERDUE/WEAK/exposure-tier and 问题队列; "
        "resolve open problems when evidence supports a fix path."
    )
    return "\n".join(lines)
