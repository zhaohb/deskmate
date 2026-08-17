"""One-shot LLM extraction of what a study session actually taught.

Returns topics, concepts and lecture structure from a single Ollama call, using
DeskMate's configured model. Soft-fails to empty results when the LLM is
unavailable or returns unparseable JSON.

Why the LLM owns concepts now: rule extraction ranked candidates by word shape
and recurrence, which cannot tell a taught term from a button label — 「复习队列」
and 「推理优化」 are both four-character Chinese noun phrases. A real session
produced a review queue led by "SCREEN", "ask" and fragments of DeskMate's own
interface copy. No stoplist fixes that; judging whether something was *taught* is
a semantic call, which is exactly what a model can do and a regex cannot.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from ..console import echo_stderr


@dataclass
class SubtopicHit:
    name: str
    confidence: float = 0.5


@dataclass
class TopicHit:
    name: str
    confidence: float = 0.5
    subtopics: list[SubtopicHit] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)


@dataclass
class StructuredExtraction:
    """Everything one LLM pass understood about the session."""

    topics: list[TopicHit] = field(default_factory=list)
    concepts: list[Any] = field(default_factory=list)       # ConceptHit
    structure: list[Any] = field(default_factory=list)      # LectureItem

    @property
    def ok(self) -> bool:
        """Did the call yield anything usable?

        Callers gate persistence on this: seeding SM-2 from a failed pass would
        put junk on a schedule that then persists across every later session,
        which is worse than an empty queue for one report.
        """
        return bool(self.topics or self.concepts or self.structure)


_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.I)

_SYSTEM = (
    "You extract what a study session TAUGHT, from lecture audio and courseware "
    "text. Return ONLY a JSON object (no markdown, no prose) with this shape:\n"
    '{"topics":[{"name":"string","confidence":0.0,'
    '"subtopics":[{"name":"string","confidence":0.0}],'
    '"evidence":["short quote from evidence"]}],'
    '"concepts":[{"name":"string","topic":"string","is_definition":false,'
    '"is_problem":false,"is_code":false,'
    '"evidence":["short quote from evidence"]}],'
    '"structure":[{"kind":"definition|step|relation","subject":"string",'
    '"content":"definition or step text","target":"relation target concept",'
    '"ordinal":0,"evidence":"short verbatim quote"}]}\n'
    "Rules:\n"
    "- Max 6 topics (4 subtopics each), 20 concepts, 20 structure items.\n"
    "- confidence in [0,1] = how clearly the evidence supports the topic.\n"
    "- evidence must be short verbatim quotes from the provided text.\n"
    "- Do not invent anything absent from the evidence.\n"
    "- Prefer the user's language (Chinese if evidence is Chinese).\n"
    "\n"
    "A concept is a subject that was EXPLAINED or PRACTISED. It is never:\n"
    "- interface text: button, menu, tab, window or page labels, app names,\n"
    "  window-manager words (Minimize / Maximize / Restore / Close / 任务栏)\n"
    "- a heading from the tool the user is studying WITH, as opposed to the\n"
    "  material they are studying\n"
    "- a fragment of a sentence, or a phrase containing pronouns or particles\n"
    "  (你/我/的/会/直到/正在) — those are prose, not terms\n"
    "If the evidence is only interface text with nothing taught, return an empty\n"
    "concepts list. An empty list is correct and expected; a list of button\n"
    "labels is a failure.\n"
    "\n"
    "Extract the TECHNICAL substance, not a table of contents. Strongly prefer:\n"
    "- named algorithms, data structures, methods, APIs, model architectures\n"
    "- quantitative facts (numbers, %, latency, memory, precision such as INT4 /\n"
    "  FP16 / 68% / tokens per second) — keep the number inside the evidence quote\n"
    "- hardware, backends, formats, versions (CPU / GPU / NPU / FPGA, IR, ONNX)\n"
    "- mechanisms: how it works, what it optimises, what trade-off it makes\n"
    "A name like 新特性 / 功能介绍 / 课程介绍 is a section title, not a technical\n"
    "concept — drop it and keep the specific technique it introduces instead. For\n"
    "each concept, the evidence quote should state a technical fact about it (what\n"
    "it does, its spec, or its number), not merely repeat its name.\n"
    "\n"
    "structure: `definition` = subject is defined as content; `step` = an ordered\n"
    "procedure step (set ordinal from 1); `relation` links two named concepts: put\n"
    "the source concept in subject and destination concept in target (not content).\n"
    "Both endpoints must exactly match a name in topics, subtopics, or concepts.\n"
    "Prefer relations that carry technical meaning between two named techniques\n"
    "(压缩 / 加速 / 部署到 / 支持 / 依赖 / 量化为 / 替代), e.g. '<technique> 压缩\n"
    "<structure>', '<model> 部署到 <hardware>', '<method> 将 <metric> 降低 <number>'.\n"
    "Never use a clause, benefit, outcome, explanation, pronoun phrase, or sentence\n"
    "fragment as an endpoint. Evidence must be a verbatim quote that states the\n"
    "relation, including any number. Omit uncertain relations instead of guessing.\n"
)


def _clip_corpus(parts: list[str], *, max_chars: int = 4500) -> str:
    chunks: list[str] = []
    used = 0
    for p in parts:
        t = " ".join((p or "").split())
        if len(t) < 12:
            continue
        piece = t[:400]
        if used + len(piece) > max_chars:
            break
        chunks.append(piece)
        used += len(piece)
    return "\n---\n".join(chunks)


def _load_json_object(raw: str) -> dict[str, Any]:
    """Pull the outermost JSON object out of a model reply, or ``{}``."""
    from ..engine import llm as llm_mod  # noqa: PLC0415

    text = llm_mod.strip_thinking(raw or "").strip()
    if not text:
        return {}
    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}


def _clean_quotes(raw: Any, *, limit: int = 3) -> list[str]:
    out: list[str] = []
    for e in (raw or [])[:limit]:
        if isinstance(e, str) and e.strip():
            out.append(e.strip()[:180])
    return out


def _parse_concepts(obj: dict[str, Any]) -> list[Any]:
    """Build ConceptHit rows from the model's `concepts` array."""
    from .extract import ConceptHit  # noqa: PLC0415

    rows = obj.get("concepts")
    if not isinstance(rows, list):
        return []
    out: list[Any] = []
    seen: set[str] = set()
    for row in rows[:20]:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if len(name) < 2 or name.lower() in seen:
            continue
        seen.add(name.lower())
        out.append(
            ConceptHit(
                name=name[:100],
                topic=str(row.get("topic") or "").strip()[:60] or "general",
                count=1,
                evidence=_clean_quotes(row.get("evidence")),
                has_definition=bool(row.get("is_definition")),
                has_problem=bool(row.get("is_problem")),
                has_code=bool(row.get("is_code")),
            )
        )
    return out


def _parse_structure(obj: dict[str, Any]) -> list[Any]:
    """Build LectureItem rows from the model's `structure` array."""
    from .extract import LectureItem  # noqa: PLC0415

    rows = obj.get("structure")
    if not isinstance(rows, list):
        return []
    out: list[Any] = []
    for row in rows[:20]:
        if not isinstance(row, dict):
            continue
        kind = str(row.get("kind") or "").strip().lower()
        if kind not in {"definition", "step", "relation"}:
            continue
        subject = str(row.get("subject") or "").strip()
        raw_content = row.get("target") if kind == "relation" else row.get("content")
        content = str(raw_content or row.get("content") or "").strip()
        if len(subject) < 2 or len(content) < 2:
            continue
        try:
            ordinal = int(row.get("ordinal") or 0)
        except (TypeError, ValueError):
            ordinal = 0
        ev = _clean_quotes(row.get("evidence"), limit=1)
        out.append(
            LectureItem(
                kind=kind,
                subject=subject[:120],
                content=content[:400],
                ordinal=max(0, ordinal),
                source="llm",
                evidence=ev[0] if ev else "",
            )
        )
    return out


def _parse_topics(obj: dict[str, Any]) -> list[TopicHit]:
    rows = obj.get("topics")
    if not isinstance(rows, list):
        return []
    out: list[TopicHit] = []
    for row in rows[:6]:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if len(name) < 2:
            continue
        try:
            conf = float(row.get("confidence", 0.5))
        except (TypeError, ValueError):
            conf = 0.5
        conf = max(0.0, min(1.0, conf))
        subs: list[SubtopicHit] = []
        for s in (row.get("subtopics") or [])[:4]:
            if not isinstance(s, dict):
                continue
            sn = str(s.get("name") or "").strip()
            if len(sn) < 2:
                continue
            try:
                sc = float(s.get("confidence", 0.5))
            except (TypeError, ValueError):
                sc = 0.5
            subs.append(SubtopicHit(name=sn[:80], confidence=max(0.0, min(1.0, sc))))
        out.append(
            TopicHit(
                name=name[:100],
                confidence=conf,
                subtopics=subs,
                evidence=_clean_quotes(row.get("evidence")),
            )
        )
    return out


def extract_structured_llm(
    *,
    audio_texts: list[str],
    ocr_texts: list[str],
    other_texts: list[str],
    concept_hints: list[str] | None = None,
    verbose: bool = False,
) -> StructuredExtraction:
    """One Ollama chat call → topics + concepts + lecture structure.

    A single call on purpose: on a local model each round trip costs real time,
    and these three outputs are the same reading task at different granularity.
    """
    corpus = _clip_corpus(list(audio_texts) + list(ocr_texts) + list(other_texts))
    if len(corpus) < 40:
        if verbose:
            echo_stderr("  [learning_memory] LLM extraction skipped: thin corpus")
        return StructuredExtraction()

    # Rule candidates go in as hints only — they are noisy by nature, and the
    # model's job includes rejecting the interface text among them.
    hints = ", ".join((concept_hints or [])[:16]) or "(none)"
    user = (
        f"Candidate terms found by pattern matching (NOISY — many are interface "
        f"labels; keep only what was actually taught): {hints}\n\n"
        f"Evidence corpus:\n{corpus}\n\n"
        "Extract the JSON now."
    )
    try:
        from ..engine import llm as llm_mod  # noqa: PLC0415

        base, model, timeout = llm_mod.resolve_ollama_settings()
        # Prefer agent override if present (apps set agent.OLLAMA_MODEL).
        try:
            from ..apps import agent as agent_mod  # noqa: PLC0415

            if getattr(agent_mod, "OLLAMA_MODEL", None):
                model = agent_mod.OLLAMA_MODEL
            if getattr(agent_mod, "OLLAMA_BASE", None):
                base = agent_mod.OLLAMA_BASE
        except Exception:  # noqa: BLE001
            pass

        msg = llm_mod.chat_ollama(
            [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user},
            ],
            tools=None,
            base=base,
            model=model,
            # Three arrays now instead of one, so the reply is longer.
            num_predict=2048,
            timeout=min(180, int(timeout or 180)),
        )
        obj = _load_json_object(str(msg.get("content") or ""))
        out = StructuredExtraction(
            topics=_parse_topics(obj),
            concepts=_parse_concepts(obj),
            structure=_parse_structure(obj),
        )
        if verbose:
            echo_stderr(
                f"  [learning_memory] LLM topics={len(out.topics)} "
                f"concepts={len(out.concepts)} structure={len(out.structure)} "
                f"model={model}"
            )
        return out
    except Exception as exc:  # noqa: BLE001
        if verbose:
            echo_stderr(f"  [learning_memory] LLM extraction error: {exc}")
        return StructuredExtraction()


def topics_to_dict(topics: list[TopicHit]) -> list[dict[str, Any]]:
    return [asdict(t) for t in topics]
