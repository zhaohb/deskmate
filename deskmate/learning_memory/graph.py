"""Build concept edges from lecture relation extractions."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .extract import LectureItem, normalize_name


@dataclass
class ConceptEdge:
    src_name: str
    dst_name: str
    rel: str  # prerequisite | related | contrasts | leads_to
    evidence: str = ""
    weight: float = 1.0


_CONTRAST = re.compile(r"\b(vs\.?|versus|不同于|compared)\b", re.I)
_PREREQ = re.compile(r"\b(depends on|based on|derived from|基于|依赖于|来自于|建立在)\b", re.I)
_LEADS = re.compile(r"(→|->|=>|导致|产生|用于)")


def classify_relation(evidence: str, subject: str, content: str) -> str:
    blob = f"{subject} {content} {evidence}"
    if _CONTRAST.search(blob):
        return "contrasts"
    if _PREREQ.search(blob):
        return "prerequisite"
    if _LEADS.search(blob):
        return "leads_to"
    return "related"


def edges_from_lecture_items(items: list[LectureItem]) -> list[ConceptEdge]:
    """Turn relation lecture items into directed concept edges."""
    out: list[ConceptEdge] = []
    seen: set[str] = set()
    for it in items:
        if it.kind != "relation":
            continue
        src = (it.subject or "").strip()
        dst = (it.content or "").strip()
        # content may be longer phrase — keep first token-ish chunk
        if " " in dst and len(dst) > 40:
            dst = dst.split(",")[0].split("，")[0].strip()[:40]
        if len(src) < 2 or len(dst) < 2:
            continue
        if normalize_name(src) == normalize_name(dst):
            continue
        rel = classify_relation(it.evidence, src, dst)
        key = f"{normalize_name(src)}|{normalize_name(dst)}|{rel}"
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
