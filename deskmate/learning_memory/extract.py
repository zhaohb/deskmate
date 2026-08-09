"""Deterministic topic/concept + lecture-structure extraction from study text.

Rule-based (no LLM): pulls candidate concepts from audio/OCR, then slots lines
into definition / step / relation buckets — BeePrepared-style structure with
adaptive-learning-agent-style topic tagging.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

_STOP = frozenset({
    "the", "and", "for", "with", "that", "this", "from", "your", "have", "are",
    "was", "were", "been", "being", "will", "would", "could", "should", "about",
    "into", "than", "then", "them", "they", "what", "when", "where", "which",
    "while", "using", "used", "use", "also", "just", "like", "some", "more",
    "most", "such", "only", "other", "over", "after", "before", "between",
    "我们", "你们", "他们", "一个", "这个", "那个", "可以", "因为", "所以",
    "但是", "如果", "已经", "没有", "什么", "怎么", "自己", "时候", "进行",
    "通过", "以及", "或者", "就是", "不是", "还是", "一些", "这些", "那些",
    "今天", "现在", "然后", "首先", "最后", "接下来",
    "定义", "不同", "相比", "比如", "例如", "所以说", "也就是",
    "内容", "方法", "问题", "结果", "过程", "概念", "知识",
})

_DEF_PATTERNS = [
    # EN: "X is/are ..." / "X means ..." / "X refers to ..."
    re.compile(
        r"(?P<sub>[A-Za-z][\w\-/]{1,40}|[\u4e00-\u9fff]{2,20})\s*"
        r"(?:is|are|means|refers to|denotes|定义为|是指|指的是|叫做|称为)\s*"
        r"(?P<body>.{4,180})",
        re.I,
    ),
    # "Definition: X — ..." / "定义：X …"
    re.compile(
        r"(?:definition|定义)\s*[:：]\s*(?P<sub>[^:：,，。]{2,40})\s*[:：,，\-—]?\s*(?P<body>.{4,180})",
        re.I,
    ),
    # "X: a/an/the ..." glossary style
    re.compile(
        r"(?P<sub>[A-Za-z][\w\-]{2,30})\s*[:：]\s*(?P<body>(?:a|an|the)\s+.{4,160})",
        re.I,
    ),
]

_STEP_PATTERNS = [
    re.compile(
        r"(?:^|\n)\s*(?:step\s*)?(?P<ord>\d{1,2})[\.\)、:：]\s*(?P<body>.{4,160})",
        re.I,
    ),
    re.compile(
        r"(?:^|\n)\s*(?P<label>首先|然后|接着|其次|最后|第一步|第二步|第三步|finally|first|second|third|next)\s*[:：,]?\s*(?P<body>.{4,160})",
        re.I,
    ),
]

_REL_PATTERNS = [
    re.compile(
        r"(?P<a>[A-Za-z][\w\-/]{1,40}|[\u4e00-\u9fff]{2,16})\s*"
        r"(?:vs\.?|versus|compared to|compared with|不同于|不同于|与)\s*"
        r"(?P<b>[A-Za-z][\w\-/]{1,40}|[\u4e00-\u9fff]{2,16})",
        re.I,
    ),
    re.compile(
        r"(?P<a>[A-Za-z][\w\-/]{1,40}|[\u4e00-\u9fff]{2,16})\s*"
        r"(?:depends on|based on|derived from|extends|inherits|基于|依赖于|来自于|建立在)\s*"
        r"(?P<b>[A-Za-z][\w\-/]{1,40}|[\u4e00-\u9fff]{2,16}|.{2,40})",
        re.I,
    ),
    re.compile(
        r"(?P<a>[A-Za-z][\w\-/]{1,40}|[\u4e00-\u9fff]{2,16})\s*"
        r"(?:→|->|=>|导致|产生|用于)\s*"
        r"(?P<b>.{2,60})",
        re.I,
    ),
]

_CAMEL = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-zA-Z0-9]+)+\b")
_TECH = re.compile(r"\b[A-Za-z][A-Za-z0-9_\-]{2,30}\b")
_CN_TERM = re.compile(r"[\u4e00-\u9fff]{2,12}")
_PROBLEM_HINT = re.compile(r"(error|exception|traceback|failed|报错|异常|失败)", re.I)
_CODE_HINT = re.compile(r"(\.py|\.c\b|\.cpp|\.ts\b|\.js\b|def |class |import |function )", re.I)


@dataclass
class ConceptHit:
    name: str
    topic: str
    count: int = 1
    evidence: list[str] = field(default_factory=list)
    has_definition: bool = False
    has_problem: bool = False
    has_code: bool = False


@dataclass
class LectureItem:
    kind: str  # definition | step | relation
    subject: str
    content: str
    ordinal: int = 0
    source: str = ""
    evidence: str = ""
    session_ref: str = ""


@dataclass
class ExtractionResult:
    concepts: list[ConceptHit]
    lecture_items: list[LectureItem]
    topic_summary: str


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", (name or "").strip()).lower()


def _topic_bucket(name: str, surrounding: str) -> str:
    blob = f"{name} {surrounding}".lower()
    rules = [
        ("ml/dl", r"softmax|gradient|neural|transformer|embedding|loss|epoch|backprop|卷积|梯度|神经网络"),
        ("systems", r"cache|thread|mutex|latency|throughput|分布式|并发|锁|缓存"),
        ("math", r"matrix|vector|derivative|probability|概率|矩阵|导数|期望"),
        ("programming", r"function|class|api|pointer|async|函数|指针|接口|编译"),
        ("algorithms", r"sort|graph|tree|complexity|动态规划|复杂度|排序|图论"),
    ]
    for topic, pat in rules:
        if re.search(pat, blob, re.I):
            return topic
    if re.search(r"[\u4e00-\u9fff]", name):
        return "general-zh"
    return "general"


def _score_term(term: str, count: int) -> float:
    t = term.strip()
    if len(t) < 2 or t.lower() in _STOP:
        return -1.0
    score = float(count)
    if _CAMEL.fullmatch(t) or (t[:1].isupper() and any(c.isupper() for c in t[1:])):
        score += 2.0
    if re.search(r"[\u4e00-\u9fff]", t) and 2 <= len(t) <= 8:
        score += 1.5
    if t.isupper() and 2 <= len(t) <= 6:  # acronyms
        score += 1.5
    if len(t) > 28:
        score -= 2.0
    return score


def extract_concepts_from_texts(
    texts: list[str],
    *,
    max_concepts: int = 24,
) -> list[ConceptHit]:
    """Rank topic/concept candidates by recurrence + technical shape."""
    counter: Counter[str] = Counter()
    evidence: dict[str, list[str]] = {}
    flags: dict[str, dict[str, bool]] = {}

    for raw in texts:
        text = " ".join((raw or "").split())
        if len(text) < 8:
            continue
        problem = bool(_PROBLEM_HINT.search(text))
        code = bool(_CODE_HINT.search(text))
        candidates: list[str] = []
        candidates.extend(_CAMEL.findall(text))
        candidates.extend(m.group(0) for m in _TECH.finditer(text) if m.group(0).lower() not in _STOP)
        candidates.extend(_CN_TERM.findall(text))
        # Keep unique per line
        seen_line: set[str] = set()
        for c in candidates:
            key = normalize_name(c)
            if not key or key in seen_line or key in _STOP:
                continue
            seen_line.add(key)
            counter[key] += 1
            evidence.setdefault(key, [])
            if len(evidence[key]) < 3:
                evidence[key].append(text[:180])
            fl = flags.setdefault(key, {"problem": False, "code": False})
            fl["problem"] = fl["problem"] or problem
            fl["code"] = fl["code"] or code

    ranked: list[tuple[float, str]] = []
    for key, n in counter.items():
        # Prefer display form from first evidence mention
        display = key
        for ev in evidence.get(key, []):
            m = re.search(re.escape(key), ev, re.I)
            if m:
                display = m.group(0)
                break
            # Chinese exact
            if key in ev:
                display = key
                break
        score = _score_term(display, n)
        if score < 1.5 and n < 2:
            continue
        ranked.append((score, display if display else key))

    ranked.sort(key=lambda x: (-x[0], x[1].lower()))
    out: list[ConceptHit] = []
    for _, display in ranked[:max_concepts]:
        key = normalize_name(display)
        fl = flags.get(key, {})
        out.append(
            ConceptHit(
                name=display,
                topic=_topic_bucket(display, " ".join(evidence.get(key, [])[:1])),
                count=int(counter.get(key, 1)),
                evidence=list(evidence.get(key, [])[:3]),
                has_problem=bool(fl.get("problem")),
                has_code=bool(fl.get("code")),
            )
        )
    return out


def extract_lecture_structure(
    texts: list[tuple[str, str]],
    *,
    concepts: list[ConceptHit] | None = None,
) -> list[LectureItem]:
    """texts: list of (source, text) where source is audio|ocr|mixed."""
    items: list[LectureItem] = []
    concept_names = {normalize_name(c.name): c.name for c in (concepts or [])}

    step_ord = 0
    for source, raw in texts:
        text = (raw or "").strip()
        if len(text) < 6:
            continue

        for pat in _DEF_PATTERNS:
            for m in pat.finditer(text):
                sub = (m.group("sub") or "").strip().rstrip(".,;，。；")
                body = (m.group("body") or "").strip().rstrip(".,;，。；")
                if len(sub) < 2 or len(body) < 4:
                    continue
                if normalize_name(sub) in _STOP:
                    continue
                items.append(
                    LectureItem(
                        kind="definition",
                        subject=sub[:80],
                        content=body[:240],
                        source=source,
                        evidence=text[:200],
                    )
                )
                key = normalize_name(sub)
                if key in concept_names:
                    pass  # linked later in pipeline

        for pat in _STEP_PATTERNS:
            for m in pat.finditer(text):
                body = (m.group("body") or "").strip()
                if len(body) < 4:
                    continue
                if "ord" in m.groupdict() and m.group("ord"):
                    try:
                        step_ord = int(m.group("ord"))
                    except ValueError:
                        step_ord += 1
                else:
                    step_ord += 1
                label = m.groupdict().get("label") or f"step {step_ord}"
                items.append(
                    LectureItem(
                        kind="step",
                        subject=str(label)[:40],
                        content=body[:240],
                        ordinal=step_ord,
                        source=source,
                        evidence=text[:200],
                    )
                )

        for pat in _REL_PATTERNS:
            for m in pat.finditer(text):
                a = (m.group("a") or "").strip()
                b = (m.group("b") or "").strip().rstrip(".,;，。；")
                if len(a) < 2 or len(b) < 2:
                    continue
                if normalize_name(a) in _STOP:
                    continue
                items.append(
                    LectureItem(
                        kind="relation",
                        subject=a[:80],
                        content=b[:160],
                        source=source,
                        evidence=m.group(0)[:200],
                    )
                )

    # Dedup by (kind, subject_norm, content_prefix)
    seen: set[str] = set()
    uniq: list[LectureItem] = []
    for it in items:
        key = f"{it.kind}|{normalize_name(it.subject)}|{normalize_name(it.content)[:60]}"
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)

    # Prefer audio definitions first, then ocr; cap each kind
    def sort_key(it: LectureItem) -> tuple:
        src = 0 if it.source == "audio" else 1
        return ( {"definition": 0, "step": 1, "relation": 2}.get(it.kind, 9), src, it.ordinal )

    uniq.sort(key=sort_key)
    caps = {"definition": 16, "step": 16, "relation": 12}
    counts: dict[str, int] = {}
    out: list[LectureItem] = []
    for it in uniq:
        n = counts.get(it.kind, 0)
        if n >= caps.get(it.kind, 10):
            continue
        counts[it.kind] = n + 1
        out.append(it)

    # Mark concepts that got a definition
    if concepts:
        defined = {normalize_name(i.subject) for i in out if i.kind == "definition"}
        for c in concepts:
            if normalize_name(c.name) in defined:
                c.has_definition = True

    return out


def extract_all(
    *,
    audio_texts: list[str],
    ocr_texts: list[str],
    other_texts: list[str] | None = None,
) -> ExtractionResult:
    other_texts = other_texts or []
    all_texts = [t for t in (audio_texts + ocr_texts + other_texts) if t and t.strip()]
    concepts = extract_concepts_from_texts(all_texts, max_concepts=24)

    paired: list[tuple[str, str]] = []
    paired.extend(("audio", t) for t in audio_texts if t.strip())
    paired.extend(("ocr", t) for t in ocr_texts if t.strip())
    paired.extend(("mixed", t) for t in other_texts if t.strip())
    lecture = extract_lecture_structure(paired, concepts=concepts)

    topics = Counter(c.topic for c in concepts)
    topic_summary = ", ".join(f"{t}({n})" for t, n in topics.most_common(5)) or "(none)"
    return ExtractionResult(concepts=concepts, lecture_items=lecture, topic_summary=topic_summary)


def extraction_to_dict(result: ExtractionResult) -> dict[str, Any]:
    return {
        "topic_summary": result.topic_summary,
        "concepts": [
            {
                "name": c.name,
                "topic": c.topic,
                "count": c.count,
                "evidence": c.evidence,
                "has_definition": c.has_definition,
                "has_problem": c.has_problem,
                "has_code": c.has_code,
            }
            for c in result.concepts
        ],
        "lecture_items": [
            {
                "kind": i.kind,
                "subject": i.subject,
                "content": i.content,
                "ordinal": i.ordinal,
                "source": i.source,
                "evidence": i.evidence,
            }
            for i in result.lecture_items
        ],
    }
