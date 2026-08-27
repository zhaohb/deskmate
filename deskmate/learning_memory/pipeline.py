"""Orchestrate extract → LLM topics → sessions/events/edges → BKT/SM-2 → prompt."""

from __future__ import annotations

import re
from collections.abc import Callable
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

    must_cover = collect_must_cover(
        topics=topics,
        concepts=concepts,
        due_reviews=due,
        audio_texts=audio_texts,
        ocr_texts=ocr_texts,
    )
    prompt_block = format_enrichment_prompt(
        result,
        due_reviews=due,
        topics=topics,
        edges=edges_out,
        events=events_out,
        concepts=concepts,
        lecture_items=lecture_items,
        cover_names=[c["name"] for c in must_cover],
    )
    return {
        "extraction": extraction_to_dict(result),
        "topics": topics_to_dict(topics),
        "due_reviews": due,
        "edges": edges_out,
        "events": events_out,
        "session_ids": session_ids,
        "prompt_block": prompt_block,
        "must_cover": must_cover,
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
    cover_names: list[str] | None = None,
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

    cover: list[str] = []
    if cover_names:
        cover = [n for n in cover_names if n]
    else:
        for topic in topics or []:
            name = getattr(topic, "name", "") or ""
            if name and name not in cover:
                cover.append(name)
        for concept in concepts or []:
            name = getattr(concept, "name", "") or ""
            if name and name not in cover:
                cover.append(name)

    lines.append("")
    lines.append("### MUST COVER in 讲了什么 / 课程重点 / 知识图谱")
    if cover:
        lines.append(
            "Every name below MUST appear in those three sections, or the section "
            "must explicitly say 材料不足 for that name. Do not drop a short topic "
            "because a longer one was discussed more. Copy the concept-graph edges "
            "into ## 知识图谱; do not invent a shorter substitute graph."
        )
        for name in cover[:24]:
            lines.append(f"- {name}")
    else:
        lines.append("(no extracted concepts — keep those sections short)")

    lines.append("")
    lines.append(
        "INSTRUCTION: 讲解重点 prefer LLM topics + 结构:* + 图谱; "
        "理解要点 explain those subjects and prerequisites; "
        "复习重点 / 下一步学习计划 start from OVERDUE/WEAK/exposure-tier and 问题队列; "
        "resolve open problems when evidence supports a fix path. "
        "MUST COVER names are mandatory in 讲了什么 / 课程重点 / 知识图谱."
    )
    return "\n".join(lines)


# 课程重点 is deliberately absent. That section must say why a point matters,
# which a patcher cannot judge; padding it only duplicated 讲了什么 line for line.
_COVER_SECTIONS = ("## 讲了什么", "## 知识图谱")


def collect_must_cover(
    *,
    topics: list[Any],
    concepts: list[Any],
    due_reviews: list[dict[str, Any]],
    audio_texts: list[str],
    ocr_texts: list[str],
) -> list[dict[str, str]]:
    """Names this session actually taught, to be forced into the recap.

    The writer model compresses 3–7 themes and drops one-minute asides even
    when they are in the prompt. Coverage is computed here from (1) extracted
    concepts, (2) OCR names that also occur in the audio, (3) review-queue
    names grounded in this window — then the recap is patched after generation.
    """
    from deskmate.apps.learning_evidence import (  # noqa: PLC0415
        _IDENT_RE,
        _line_hits_terms,
    )

    audio_blob = "\n".join(audio_texts or [])
    ocr_blob = "\n".join(ocr_texts or [])
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    used_evidence: set[str] = set()
    respell = _slide_speller(ocr_texts)

    def add(name: str, evidence: str = "") -> None:
        # The review queue stores names as ASR heard them, so a slide's "LoRA"
        # comes back as "Laura" and the recap covers one subject twice.
        label = respell((name or "").strip())
        if not _usable_cover_name(label):
            return
        key = label.lower()
        if key in seen:
            return
        seen.add(key)
        # OCR reads the browser and DeskMate's own window alongside the slides,
        # so the extractor returns "bilibili" and "Details" as concepts, with
        # evidence. What separates those from a subject is that a lecturer says
        # a subject out loud. (With no recording at all, this cannot judge.)
        spoken = _context_for(label, audio_texts, [])
        if audio_texts and not spoken:
            return
        # Where the sentence came from decides how the recap may present it: a
        # summary the extractor wrote can stand as an explanation, while text
        # read off a slide is just the screen — often a header or a logo row.
        ev = _clean_evidence(evidence)
        source = "extracted" if ev and not _is_screen_text(evidence) else "transcript"
        if not ev:
            ev = spoken or _context_for(label, [], ocr_texts)
        # One sentence cannot explain two names. PyTorch and PaddlePaddle sat
        # in the same clause, so the recap printed that clause twice.
        if ev in used_evidence:
            ev, source = "", "transcript"
        if ev:
            used_evidence.add(ev)
        items.append({"name": label, "evidence": ev, "source": source})

    def first_evidence(obj: Any) -> str:
        evidence = getattr(obj, "evidence", None)
        if isinstance(evidence, list) and evidence:
            return str(evidence[0])
        return ""

    for topic in topics or []:
        add(getattr(topic, "name", "") or "", first_evidence(topic))
    for concept in concepts or []:
        add(getattr(concept, "name", "") or "", first_evidence(concept))

    for row in due_reviews or []:
        name = str(row.get("name") or "")
        ev = str(row.get("evidence_json") or "")
        if _line_hits_terms(audio_blob, [name]) or _line_hits_terms(ocr_blob, [name]):
            add(name, ev)

    # Slide names that were also spoken — but only if they are rare in the
    # transcript. A name on most lines is the lecture theme; spanning already
    # keeps it. A name on a handful of lines is the aside the writer drops.
    n_audio = max(1, len(audio_texts or []))
    rare_cap = max(2, int(n_audio * 0.15))
    title_lines = [
        t for t in (ocr_texts or []) if str(t).startswith("课件标题:")
    ] or [ocr_blob]
    for title in title_lines:
        # A slide header, a logo wall of supported frameworks, an architecture
        # diagram — each names many things and teaches none of them.
        if _is_enumeration(title):
            continue
        for m in _IDENT_RE.finditer(title):
            tok = m.group(0)
            spoken = [
                line for line in (audio_texts or []) if _line_hits_terms(line, [tok])
            ]
            if not 1 <= len(spoken) <= rare_cap:
                continue
            if all(_is_enumeration(line) for line in spoken):
                continue
            add(tok)

    return _drop_subsumed(items)[:16]


def ensure_must_cover(markdown: str, covers: list[dict[str, str]]) -> str:
    """Append any MUST COVER name the model omitted from the patched sections."""
    if not markdown or not covers:
        return markdown
    out = markdown
    usable = [c for c in covers if _usable_cover_name(c.get("name") or "")]
    for heading in _COVER_SECTIONS:
        body = _section_body(out, heading)
        if body is None:
            continue
        missing = [c for c in usable if not _name_in(body, c["name"])]
        extra = _cover_patch(missing, heading)
        if extra:
            out = _append_in_section(out, heading, extra)
    return out


def _slide_speller(ocr_texts: list[str]) -> Callable[[str], str]:
    """Rewrite a name toward the courseware's spelling of it, as the prompt asks."""
    from deskmate.apps.learning_evidence import (  # noqa: PLC0415
        SlideEvidence,
        canonicalize_against_slides,
    )

    titles: list[str] = []
    body: list[str] = []
    for raw in ocr_texts or []:
        text = str(raw)
        if text.startswith("课件标题:"):
            titles.append(text.removeprefix("课件标题:").strip())
        elif text.startswith("课件OCR:"):
            body.append(text.removeprefix("课件OCR:").strip())
    if not titles and not body:
        return lambda name: name
    slides = SlideEvidence(headlines=titles, lines=body)
    screen = " ".join([*titles, *body]).lower()

    def respell(name: str) -> str:
        words = re.findall(r"\S+", name or "")
        return " ".join(
            # A word the courseware itself uses is already right. Rewriting it
            # toward a lookalike elsewhere on screen turned the good name
            # "Video Generation Pipeline" into "video0 Generation Pipeline".
            word
            if word.lower() in screen
            else canonicalize_against_slides([word], slides)[0]
            for word in words
        )

    return respell


_ENUMERATION_MIN_NAMES = 3


def _is_enumeration(line: str) -> bool:
    """Whether a line rattles off several names instead of teaching one.

    "PyTorch TensorFlow Keras TensorFlowLite PaddlePaddle" is the set of
    formats a runtime reads; no one of them is the subject of that slide.
    """
    from deskmate.apps.learning_evidence import _IDENT_RE  # noqa: PLC0415

    names = {m.group(0).lower() for m in _IDENT_RE.finditer(line or "")}
    return len(names) >= _ENUMERATION_MIN_NAMES


def _drop_subsumed(items: list[dict[str, str]]) -> list[dict[str, str]]:
    """Drop a name that is spelled out inside a longer one already covered.

    "OpenVINO 2026.2" and "OpenVINO 2026.2 新特性" are one subject, and covering
    both printed the same evidence under two headings.
    """
    names = [c["name"].lower() for c in items]
    return [
        c
        for c in items
        if not any(other != c["name"].lower() and c["name"].lower() in other for other in names)
    ]


_EVIDENCE_JSON_RE = re.compile(r'^\s*\[?\s*"(.*)"\s*\]?\s*$', re.S)
_MIN_CONTEXT_LEN = 12
_MAX_CONTEXT_LEN = 140


_SCREEN_PREFIXES = ("课件标题:", "课件OCR:")


def _is_screen_text(raw: str) -> bool:
    """Whether this evidence is words on a slide rather than a spoken sentence."""
    text = " ".join((raw or "").split())
    m = _EVIDENCE_JSON_RE.match(text)
    if m:
        text = m.group(1)
    return text.lstrip("\"'[ ").startswith(_SCREEN_PREFIXES)


def _clean_evidence(raw: str) -> str:
    """Unwrap a stored `["…"]` evidence blob into a plain sentence."""
    text = " ".join((raw or "").split())
    if not text:
        return ""
    m = _EVIDENCE_JSON_RE.match(text)
    if m:
        text = m.group(1).replace('", "', " ").replace('","', " ")
    text = _strip_audio_prefix(text).strip()
    for prefix in _SCREEN_PREFIXES:
        text = text.removeprefix(prefix).strip()
    text = _drop_restated_opening(text)
    text = text.strip("\"'[] 、,，。;；")
    return text[:_MAX_CONTEXT_LEN]


def _drop_restated_opening(text: str) -> str:
    """Collapse a line that restates its own opening.

    Consecutive ASR windows overlap, so one transcript line often reads
    "X … X … 之外" — the same clause twice, the second time continuing.
    """
    for size in range(len(text) // 2, 5, -1):
        again = text.find(text[:size], size)
        if again != -1:
            return text[again:]
    return text


_NAME_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9.+_-]*|[0-9]+(?:\.[0-9]+)*|[一-龥]{2,}")


def _name_tokens(name: str) -> list[str]:
    return [t for t in _NAME_TOKEN_RE.findall(name or "") if len(t) >= 2]


def _line_matches_name(line: str, name: str, tokens: list[str]) -> bool:
    """Whether `line` is talking about `name`.

    A multi-word topic like "OpenVINO 2026.2 新特性" rarely appears verbatim in
    speech, so an exact hit is tried first and a majority of the name's tokens
    is accepted as the fallback.
    """
    from deskmate.apps.learning_evidence import _line_hits_terms  # noqa: PLC0415

    if _line_hits_terms(line, [name]):
        return True
    if len(tokens) < 2:
        return False
    hits = sum(1 for t in tokens if _line_hits_terms(line, [t]))
    return hits >= max(2, (len(tokens) + 1) // 2)


def _context_for(name: str, audio_texts: list[str], ocr_texts: list[str]) -> str:
    """The sentence from this session that introduced `name`.

    Audio first: the lecturer's own wording explains a term, while a slide
    line is usually just the term again. Short hits absorb the next line so
    the bullet reads as a point rather than a fragment.
    """
    tokens = _name_tokens(name)
    for source in (audio_texts or [], ocr_texts or []):
        lines = [_clean_evidence(line) for line in source]
        for i, line in enumerate(lines):
            if not line or not _line_matches_name(line, name, tokens):
                continue
            if line.strip() == name.strip():
                continue
            text = line
            for nxt in lines[i + 1 : i + 3]:
                if len(text) >= _MIN_CONTEXT_LEN:
                    break
                if nxt:
                    text = f"{text} {nxt}"
            if len(text) >= _MIN_CONTEXT_LEN:
                return text[:_MAX_CONTEXT_LEN]
    return ""


def _usable_cover_name(name: str) -> bool:
    s = name.strip()
    if not (3 <= len(s) <= 60):
        return False
    if s.startswith("课件"):
        return False
    low = s.lower()
    if ".exe" in low or "http://" in low or "https://" in low:
        return False
    # A three-letter name is a term only as an acronym (GPU, NPU, FP16). Lower
    # case ones came off the screen — "exe", "com" — and they match as
    # substrings of ordinary words, so they survived every later filter.
    return not (len(s) < 4 and s.isascii() and not s.isupper())


def _name_in(text: str, name: str) -> bool:
    if not text or not name:
        return False
    if name.lower() in text.lower():
        return True
    from deskmate.apps.learning_evidence import _similar  # noqa: PLC0415

    name_toks = re.findall(r"[A-Za-z][A-Za-z0-9+_-]*|[一-龥]{2,}", name)
    text_toks = re.findall(r"[A-Za-z][A-Za-z0-9+_-]*", text)
    for nt in name_toks:
        if len(nt) < 3:
            continue
        if "\u4e00" <= nt[0] <= "\u9fff" and nt in text:
            return True
        for tt in text_toks:
            if tt.lower() == nt.lower() or _similar(tt, nt):
                return True
    return False


def _section_body(markdown: str, heading: str) -> str | None:
    rx = re.compile(
        rf"{re.escape(heading)}[^\n]*\n(.*?)(?=\n## |\Z)",
        re.S,
    )
    m = rx.search(markdown)
    return m.group(1) if m else None


def _append_in_section(markdown: str, heading: str, extra: str) -> str:
    rx = re.compile(
        rf"({re.escape(heading)}[^\n]*\n)(.*?)(?=\n## |\Z)",
        re.S,
    )

    def repl(m: re.Match[str]) -> str:
        return m.group(1) + m.group(2).rstrip() + "\n" + extra + "\n"

    new, n = rx.subn(repl, markdown, count=1)
    return new if n else markdown


def _cover_patch(missing: list[dict[str, str]], heading: str) -> str:
    """Render omitted names without ever faking an explanation.

    Only a summary the extractor wrote earns a full bullet. A line scraped off
    the transcript is raw speech — ASR typos, half a sentence, the lecturer's
    opening greeting — so pasting it under a bold name told the reader nothing
    and buried the model's own prose under dozens of such lines.
    """
    if not missing:
        return ""
    if heading == "## 知识图谱":
        return f"- 节点补全（相关）：{'、'.join(c['name'] for c in missing)}"
    explained = [
        c for c in missing if c.get("source") == "extracted" and c.get("evidence")
    ]
    lines = [f"- **{c['name']}**：{c['evidence']}" for c in explained]
    listed = [c for c in missing if c not in explained]
    if listed:
        lines.append(f"- 本场还提到但未展开：{'、'.join(c['name'] for c in listed)}")
    return "\n".join(lines)
