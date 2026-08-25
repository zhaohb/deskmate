"""Build concept edges from lecture relation extractions."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .extract import LectureItem, normalize_name
from .topics_llm import TopicHit


@dataclass
class ConceptEdge:
    src_name: str
    dst_name: str
    rel: str  # prerequisite | related | contrasts | leads_to | contains
    evidence: str = ""
    weight: float = 1.0


_CONTRAST = re.compile(r"\b(vs\.?|versus|不同于|compared)\b", re.I)
_PREREQ = re.compile(r"\b(depends on|based on|derived from|基于|依赖于|来自于|建立在)\b", re.I)
_LEADS = re.compile(r"(\bleads?\s+to\b|→|->|=>|导致|产生|用于)", re.I)


def classify_relation(evidence: str, subject: str, content: str) -> str:
    blob = f"{subject} {content} {evidence}"
    if _CONTRAST.search(blob):
        return "contrasts"
    if _PREREQ.search(blob):
        return "prerequisite"
    if _LEADS.search(blob):
        return "leads_to"
    return "related"


def _has_substantive_evidence(value: str) -> bool:
    meaningful = re.sub(r"[^\w\u3400-\u9fff]", "", value or "")
    return len(meaningful) >= 8


def edges_from_lecture_items(
    items: list[LectureItem], *, allowed_nodes: set[str] | None = None
) -> list[ConceptEdge]:
    """Turn model-confirmed relations between known session concepts into edges."""
    out: list[ConceptEdge] = []
    seen: set[str] = set()
    allowed = (
        {normalize_name(name) for name in allowed_nodes}
        if allowed_nodes is not None
        else None
    )
    for it in items:
        if it.kind != "relation":
            continue
        src = (it.subject or "").strip()
        dst = (it.content or "").strip()
        if len(src) < 2 or len(dst) < 2:
            continue
        src_key = normalize_name(src)
        dst_key = normalize_name(dst)
        if src_key == dst_key:
            continue
        if allowed is not None and (src_key not in allowed or dst_key not in allowed):
            continue
        if not _has_substantive_evidence(it.evidence):
            continue
        rel = classify_relation(it.evidence, src, dst)
        key = f"{src_key}|{dst_key}|{rel}"
        if key in seen:
            continue
        seen.add(key)
        out.append(
            ConceptEdge(
                src_name=src[:80],
                dst_name=dst[:80],
                rel=rel,
                evidence=(it.evidence or "")[:200],
                weight=1.0,
            )
        )
    return out


def hierarchy_edges(topics: list[TopicHit], concepts: list[object]) -> list[ConceptEdge]:
    """Build the stable topic → subtopic → concept backbone for one session."""
    out: list[ConceptEdge] = []
    parents: dict[str, str] = {}
    seen: set[tuple[str, str]] = set()

    def add(parent: str, child: str, evidence: str = "") -> None:
        key = (normalize_name(parent), normalize_name(child))
        if not all(key) or key[0] == key[1] or key in seen:
            return
        seen.add(key)
        out.append(ConceptEdge(parent, child, "contains", evidence=evidence))

    for topic in topics:
        topic_evidence = topic.evidence[0] if topic.evidence else ""
        parents[normalize_name(topic.name)] = topic.name
        for subtopic in topic.subtopics:
            parents[normalize_name(subtopic.name)] = subtopic.name
            add(topic.name, subtopic.name, topic_evidence)

    for concept in concepts:
        name = str(getattr(concept, "name", "") or "").strip()
        topic_name = str(getattr(concept, "topic", "") or "").strip()
        parent = parents.get(normalize_name(topic_name))
        if parent and name:
            evidence = next(iter(getattr(concept, "evidence", []) or []), "")
            add(parent, name, evidence)
    return out


def _first_quote(raw: object) -> str:
    if isinstance(raw, str) and raw.strip():
        return raw.strip()[:400]
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str) and item.strip():
                return item.strip()[:400]
    return ""


def graph_from_enrichment(enrichment: dict | None) -> dict[str, list[dict[str, str]]]:
    """Session graph for the recap UI: nodes carry a source quote when we have one."""
    data = enrichment or {}
    topic_names: set[str] = set()
    node_evidence: dict[str, str] = {}
    for topic in data.get("topics") or []:
        name = str((topic or {}).get("name") or "").strip()
        if not name:
            continue
        topic_names.add(name)
        quote = _first_quote((topic or {}).get("evidence"))
        if quote:
            node_evidence[name] = quote
        for subtopic in (topic or {}).get("subtopics") or []:
            subtopic_name = str((subtopic or {}).get("name") or "").strip()
            if not subtopic_name:
                continue
            topic_names.add(subtopic_name)
            sub_quote = _first_quote((subtopic or {}).get("evidence"))
            if sub_quote:
                node_evidence[subtopic_name] = sub_quote
    for concept in (data.get("extraction") or {}).get("concepts") or []:
        name = str((concept or {}).get("name") or "").strip()
        quote = _first_quote((concept or {}).get("evidence"))
        if name and quote and name not in node_evidence:
            node_evidence[name] = quote
    node_kinds: dict[str, str] = {name: "topic" for name in topic_names}
    edges: list[dict[str, str]] = []
    for edge in data.get("edges") or []:
        source = str((edge or {}).get("src_name") or "").strip()
        target = str((edge or {}).get("dst_name") or "").strip()
        if not source or not target:
            continue
        node_kinds.setdefault(source, "topic" if source in topic_names else "concept")
        node_kinds.setdefault(target, "topic" if target in topic_names else "concept")
        edges.append({
            "source": source,
            "target": target,
            "relation": str(edge.get("rel") or "related"),
            "evidence": str(edge.get("evidence") or ""),
        })
    return {
        "nodes": [
            {"id": name, "kind": kind, "evidence": node_evidence.get(name, "")}
            for name, kind in node_kinds.items()
        ],
        "edges": edges,
    }
