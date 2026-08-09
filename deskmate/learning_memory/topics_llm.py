"""One-shot LLM topic/subtopic extraction with confidence + evidence quotes.

Uses DeskMate's configured Ollama model (not Gemini). Soft-fails to [] if the
LLM is unavailable or returns unparseable JSON — rule concepts still work.
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


_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.I)

_SYSTEM = (
    "You extract study topics from classroom/courseware evidence. "
    "Return ONLY a JSON object (no markdown, no prose) with this shape:\n"
    '{"topics":[{"name":"string","confidence":0.0,'
    '"subtopics":[{"name":"string","confidence":0.0}],'
    '"evidence":["short quote from evidence"]}]}\n'
    "Rules:\n"
    "- Max 6 topics; max 4 subtopics each.\n"
    "- confidence in [0,1] = how clearly the evidence supports the topic.\n"
    "- evidence must be short verbatim quotes from the provided text.\n"
    "- Do not invent topics absent from evidence.\n"
    "- Prefer the user's language (Chinese if evidence is Chinese).\n"
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


def _parse_topics_json(raw: str) -> list[TopicHit]:
    from ..engine import llm as llm_mod  # noqa: PLC0415

    text = llm_mod.strip_thinking(raw or "").strip()
    if not text:
        return []
    m = _JSON_FENCE.search(text)
    if m:
        text = m.group(1).strip()
    # Prefer outermost object
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return []
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    rows = obj.get("topics") if isinstance(obj, dict) else None
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
        ev = []
        for e in (row.get("evidence") or [])[:3]:
            if isinstance(e, str) and e.strip():
                ev.append(e.strip()[:180])
        out.append(
            TopicHit(name=name[:100], confidence=conf, subtopics=subs, evidence=ev)
        )
    return out


def extract_topics_llm(
    *,
    audio_texts: list[str],
    ocr_texts: list[str],
    other_texts: list[str],
    concept_hints: list[str] | None = None,
    verbose: bool = False,
) -> list[TopicHit]:
    """One Ollama chat call → topics with confidence. Empty on failure."""
    corpus = _clip_corpus(list(audio_texts) + list(ocr_texts) + list(other_texts))
    if len(corpus) < 40:
        if verbose:
            echo_stderr("  [learning_memory] LLM topics skipped: thin corpus")
        return []

    hints = ", ".join((concept_hints or [])[:16]) or "(none)"
    user = (
        f"Rule-extracted concept hints (may be noisy): {hints}\n\n"
        f"Evidence corpus:\n{corpus}\n\n"
        "Extract topics JSON now."
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
            num_predict=1024,
            timeout=min(120, int(timeout or 120)),
        )
        content = str(msg.get("content") or "")
        topics = _parse_topics_json(content)
        if verbose:
            echo_stderr(f"  [learning_memory] LLM topics={len(topics)} model={model}")
        return topics
    except Exception as exc:  # noqa: BLE001
        if verbose:
            echo_stderr(f"  [learning_memory] LLM topics error: {exc}")
        return []


def topics_to_dict(topics: list[TopicHit]) -> list[dict[str, Any]]:
    return [asdict(t) for t in topics]
