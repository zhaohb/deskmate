"""Slide-aware evidence for the learning recap.

Screen OCR of a lecture is a window tree: widget labels, URLs, and player
chrome come first; the actual slide sits hundreds of characters in. Truncating
that blob from the front is how a title disappears from the prompt.

Nothing here is bound to a site, a product, or a subject. Chrome is whatever
repeats across frames and looks like a widget. Terms are whatever looks like a
name (mixed-case, digits, filenames) plus the words on recovered titles. ASR
is rewritten toward those OCR spellings. The audio budget reserves slots for
terms that are rare in the transcript — a one-minute aside — then spans the
rest.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .learning_slice import select_spanning

# OS / browser *widget* names (whole line or inline). Not lecture vocabulary.
_OS_CHROME_RE = re.compile(
    r"\b(?:Minimize|Maximize|Restore Down|Restore|Close|Back|Forward|Reload|"
    r"New tab|Bookmarks|Extensions|Separator|Address and search bar|"
    r"Chrome Legacy Window|Google Chrome|Microsoft Edge|Mozilla Firefox)\b",
    re.I,
)
_OS_LINE_RE = re.compile(
    r"^(Minimize|Maximize|Restore Down|Restore|Close|Back|Forward|Reload|"
    r"Home|New tab|Bookmarks|Extensions|Separator|Mute|Unmute|Chat|Share|"
    r"Participants|Leave|Settings|Volume)$",
    re.I,
)
_APP_FRAME_RE = re.compile(
    r"\s+[—–-]\s+(Google Chrome|Microsoft Edge|Mozilla Firefox|Safari|"
    r"PowerPoint(?: Presentation)?|Adobe Acrobat|Zoom(?: Meeting)?|"
    r"Microsoft Teams|Visual Studio Code|Notepad)\s*$",
    re.I | re.M,
)
_URL_RE = re.compile(r"https?://\S+", re.I)
# "chrome.exe · 8/25/2026, 2:57:29 PM" — a capture tool's activity feed, which
# is on screen *beside* the lecture. Its rows are not slide text.
_ACTIVITY_ROW_RE = re.compile(
    r"[\w.-]+\.(?:exe|app)\s*[·•・|-]?\s*\d{1,4}[/:-]\d",
    re.I,
)
_FILE_RE = re.compile(
    r"\b[\w./\\-]+\.(?:py|js|ts|tsx|jsx|cpp|cc|cxx|h|hpp|java|kt|go|rs|rb|cs|"
    r"php|swift|md|txt|pdf|pptx?|docx?|ipynb|json|ya?ml|xml|html|css|"
    r"onnx|bin|pt|pth|ckpt|safetensors|npy|npz|csv|sql|sh|bat|ps1|exe)\b",
    re.I,
)
_GLUED_CMD_RE = re.compile(
    r"(?i)\b(python3?|node|npm|npx|pip3?|cargo|java|ruby|perl|go)"
    r"([A-Za-z0-9_./\\-]+\.[A-Za-z0-9]+)\b"
)
# Mixed identifiers / files: GraphQL, iPhone, UTF8, snake_case — not ordinary Titlecase.
_IDENT_RE = re.compile(
    r"\b(?:"
    r"[A-Z]{2,}[a-z][A-Za-z0-9]*|"
    r"[A-Z][a-z]+[A-Z][A-Za-z0-9]*|"
    r"[A-Za-z]+[0-9][A-Za-z0-9_-]*|"
    r"[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+"
    r")\b"
)
_TITLE_CASE_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9+./_-]*(?:\s+[A-Za-z0-9+./_-]+){1,8}\b"
)
_UI_WORDS = frozenset({
    "mute", "unmute", "chat", "share", "leave", "participants", "participant",
    "settings", "volume", "help", "view", "file", "edit", "window", "tools",
    "home", "back", "forward", "reload", "close", "minimize", "maximize",
    "restore", "start", "stop", "pause", "play", "record", "meeting",
    "new", "tab", "bookmarks", "extensions", "search", "address", "zoom",
    "chrome", "edge", "firefox", "safari", "teams", "separator", "bookmarks",
})
_STOP = frozenset({
    "the", "a", "an", "in", "of", "and", "or", "to", "for", "on", "at", "vs",
    "with", "from", "by", "as", "is", "are", "this", "that", "into", "over",
    "under", "plus", "using", "used", "based", "new", "old", "via", "per",
    "的", "了", "在", "是", "与", "和", "及", "或", "对", "中", "被", "从",
    "一个", "以及",
})


@dataclass
class SlideEvidence:
    """Chrome-stripped courseware text ready for the recap prompt."""

    headlines: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)

    def prompt_lines(self) -> list[str]:
        """Headlines first so a later span-sample cannot drop the titles."""
        out: list[str] = []
        seen: set[str] = set()
        for raw in [*self.headlines, *self.lines]:
            key = re.sub(r"\s+", " ", raw).strip().lower()
            if len(key) < 4 or key in seen:
                continue
            seen.add(key)
            prefix = "课件标题: " if raw in self.headlines else "课件OCR: "
            out.append(f"{prefix}{raw.strip()}")
        return out

    def terms(self) -> list[str]:
        """Names that may pin an audio paragraph or rewrite ASR."""
        found: list[str] = []
        seen: set[str] = set()

        def add(token: str) -> None:
            key = token.strip()
            if len(key) < 3:
                return
            norm = key.lower()
            if norm in seen or norm in _STOP or norm in _UI_WORDS:
                return
            seen.add(norm)
            found.append(key)

        blob = " ".join([*self.headlines, *self.lines])
        for rx in (_FILE_RE, _IDENT_RE):
            for m in rx.finditer(blob):
                add(m.group(0))
        for h in self.headlines:
            add(h)
            for word in re.findall(r"[A-Za-z][A-Za-z0-9+_-]*|[一-龥]{2,}", h):
                if len(word) >= 4:
                    add(word)
        return found

    def spellings(self) -> list[str]:
        """Subset of terms safe to rewrite ASR toward (shape, not a word list)."""
        return [t for t in self.terms() if _is_spelling_anchor(t)]


def harvest_slide_evidence(ocr_blobs: list[str]) -> SlideEvidence:
    """Turn raw frame OCR dumps into titles, commands, and filenames."""
    blobs = [b for b in (ocr_blobs or []) if (b or "").strip()]
    chrome = _repeated_ui_lines(blobs)
    headlines: list[str] = []
    body: list[str] = []
    seen: set[str] = set()

    def _keep(text: str, *, headline: bool) -> None:
        key = re.sub(r"\s+", " ", text).strip()
        if len(key) < 4:
            return
        norm = key.lower()
        if norm in seen or norm in _UI_WORDS:
            return
        # A running process is the app hosting the slide, never its title.
        if headline and norm.endswith((".exe", ".app")):
            return
        seen.add(norm)
        (headlines if headline else body).append(key)

    for blob in blobs:
        spam = _intra_blob_spam(blob)
        for piece in _explode_ocr(blob):
            norm = _norm_line(piece)
            if not norm or norm in chrome or norm in spam:
                continue
            if _is_noise_line(piece):
                continue
            glued = _GLUED_CMD_RE.sub(r"\1 \2", piece)
            # A filename on the same dump as a title is common (code on the
            # slide). Keep the file as its own headline, then keep scanning —
            # using the whole chrome-prefixed line as the title, or skipping
            # the rest, is how the slide heading is lost.
            if _FILE_RE.search(glued) or _GLUED_CMD_RE.search(piece):
                for m in _FILE_RE.finditer(glued):
                    _keep(m.group(0), headline=True)
                for m in _GLUED_CMD_RE.finditer(piece):
                    _keep(f"{m.group(1)} {m.group(2)}", headline=True)
            if _looks_like_headline(glued):
                _keep(glued[:160], headline=True)
            for m in _TITLE_CASE_RE.finditer(glued):
                phrase = " ".join(m.group(0).split())
                if _looks_like_headline(phrase) and not _is_noise_line(phrase):
                    _keep(phrase[:160], headline=True)
            if _looks_like_slide_body(glued):
                _keep(glued[:240], headline=False)
            for m in _IDENT_RE.finditer(glued):
                _keep(m.group(0), headline=False)

        for m in _FILE_RE.finditer(blob or ""):
            _keep(m.group(0), headline=True)
        for m in _GLUED_CMD_RE.finditer(blob or ""):
            _keep(f"{m.group(1)} {m.group(2)}", headline=True)
        for m in _IDENT_RE.finditer(blob or ""):
            _keep(m.group(0), headline=False)

    return SlideEvidence(headlines=headlines, lines=body)


def canonicalize_against_slides(texts: list[str], slides: SlideEvidence) -> list[str]:
    """Replace ASR near-misses with OCR spellings of the same name."""
    terms = slides.spellings()
    if not terms:
        return list(texts)
    return [_canonicalize_one(t, terms) for t in texts]


def _canonicalize_one(text: str, terms: list[str]) -> str:
    pattern = re.compile(r"[A-Za-z][A-Za-z0-9+_-]*")
    pieces: list[str] = []
    last = 0
    for m in pattern.finditer(text):
        pieces.append(text[last:m.start()])
        word = m.group(0)
        pieces.append(_best_spelling(word, terms) or word)
        last = m.end()
    pieces.append(text[last:])
    return "".join(pieces)


def _best_spelling(word: str, terms: list[str]) -> str | None:
    """Map one ASR token onto at most one OCR spelling.

    If the token already matches a term ignoring case, only the casing is
    fixed — never swapped for a sibling term that shares a stem.
    Otherwise the unique best fuzzy match wins; an ambiguous pair is left as-is.
    """
    lower = word.lower()
    exact = [t for t in terms if t.lower() == lower]
    if exact:
        return exact[0] if exact[0] != word else None

    scored: list[tuple[float, str]] = []
    for term in terms:
        score = _similarity_score(word, term)
        if score >= 0.82:
            scored.append((score, term))
    if not scored:
        return None
    scored.sort(key=lambda x: (-x[0], -len(x[1])))
    if len(scored) == 1 or scored[0][0] - scored[1][0] >= 0.08:
        winner = scored[0][1]
        return winner if winner != word else None
    return None


def _similarity_score(word: str, term: str) -> float:
    if word == term:
        return 1.0
    a, b = word.lower(), term.lower()
    if a == b:
        return 1.0
    if min(len(a), len(b)) < 4 or abs(len(a) - len(b)) > 3:
        return 0.0
    ratio = SequenceMatcher(None, a, b).ratio()
    if _is_spelling_anchor(term):
        folded = _vowel_fold(a)
        if folded == _vowel_fold(b) and len(folded) >= 2:
            ratio = max(ratio, 0.90)
    return ratio


def budget_paragraphs(
    paragraphs: list[tuple[str, int]],
    *,
    max_lines: int,
    max_chars: int,
    anchor_terms: list[str] | None = None,
) -> list[tuple[str, int]]:
    """Even sample, but reserve slots for terms that are rare in the window.

    A name that lands in most paragraphs is the lecture theme — spanning it is
    enough. A name that lands in a handful of paragraphs is the aside even
    sampling skips; those rows are kept first, then the rest is spanned.
    """
    if not paragraphs:
        return []
    n = len(paragraphs)
    if n <= max_lines and sum(len(p[0]) for p in paragraphs) <= max_chars:
        return list(paragraphs)

    terms = [t for t in (anchor_terms or []) if len(t) >= 3]
    rare = _rare_terms(paragraphs, terms)
    must = [i for i, (line, _) in enumerate(paragraphs) if _line_hits_terms(line, rare)]
    cap_must = max(4, max_lines // 2)
    if len(must) > cap_must:
        must = _pick_rarest_paragraphs(paragraphs, must, rare, cap_must)
    if not must:
        return _span_to_budget(paragraphs, max_lines, max_chars)

    must_set = set(must)
    remaining_slots = max(0, max_lines - len(must))
    others = [paragraphs[i] for i in range(n) if i not in must_set]
    filled = select_spanning(others, remaining_slots) if remaining_slots else []
    chosen_ids = {id(paragraphs[i]) for i in must} | {id(p) for p in filled}
    kept = [p for p in paragraphs if id(p) in chosen_ids]

    while kept and sum(len(x[0]) for x in kept) > max_chars:
        droppable = [p for p in kept if not _line_hits_terms(p[0], rare)]
        if len(droppable) < 2:
            break
        smaller = select_spanning(droppable, max(1, len(droppable) - 1))
        drop_ids = {id(p) for p in droppable} - {id(p) for p in smaller}
        kept = [p for p in kept if id(p) not in drop_ids]
    return kept


def _span_to_budget(
    paragraphs: list[tuple[str, int]],
    max_lines: int,
    max_chars: int,
) -> list[tuple[str, int]]:
    kept = select_spanning(paragraphs, max_lines)
    while kept and sum(len(x[0]) for x in kept) > max_chars:
        nxt = max(1, int(len(kept) * 0.8))
        if nxt >= len(kept):
            break
        kept = select_spanning(kept, nxt)
    return kept


def _explode_ocr(text: str) -> list[str]:
    stripped = _APP_FRAME_RE.sub("", text or "")
    exploded = _URL_RE.sub("\n", stripped)
    exploded = _OS_CHROME_RE.sub("\n", exploded)
    return [ln.strip() for ln in exploded.splitlines() if ln.strip()]


def _norm_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip().lower()


def _repeated_ui_lines(blobs: list[str]) -> set[str]:
    """Lines that recur across frames *and* look like widgets, not slide text.

    A title that stays on screen for the whole session is also repeated; those
    lines are content-shaped and must not be dropped as chrome.
    """
    n = len(blobs)
    if n < 2:
        return set()
    counts: dict[str, int] = {}
    for blob in blobs:
        seen: set[str] = set()
        for piece in _explode_ocr(blob):
            key = _norm_line(piece)
            if not key or key in seen:
                continue
            seen.add(key)
            counts[key] = counts.get(key, 0) + 1
    thresh = max(2, int(n * 0.45))
    chrome: set[str] = set()
    for key, hits in counts.items():
        if hits >= thresh and _looks_like_repeated_chrome(key):
            chrome.add(key)
    return chrome


def _intra_blob_spam(blob: str) -> set[str]:
    counts: dict[str, int] = {}
    for piece in _explode_ocr(blob):
        key = _norm_line(piece)
        if key:
            counts[key] = counts.get(key, 0) + 1
    return {k for k, c in counts.items() if c >= 5}


def _looks_like_ui_line(line: str) -> bool:
    s = line.strip()
    if _OS_LINE_RE.match(s) or _URL_RE.search(s):
        return True
    words = s.split()
    return bool(words) and all(w.lower() in _UI_WORDS for w in words)


def _looks_like_repeated_chrome(line: str) -> bool:
    """Short widget labels that also happen to sit on every frame."""
    if _looks_like_ui_line(line):
        return True
    s = line.strip()
    return len(s) <= 8 and not _IDENT_RE.search(s) and not _FILE_RE.search(s)


def _is_noise_line(line: str) -> bool:
    s = line.strip()
    if len(s) < 4 or s.isdigit():
        return True
    if _OS_LINE_RE.match(s) or _looks_like_ui_line(s):
        return True
    if _ACTIVITY_ROW_RE.search(s):
        return True
    alpha = sum(ch.isalpha() or ("\u4e00" <= ch <= "\u9fff") for ch in s)
    return alpha < 3


def _looks_like_headline(line: str) -> bool:
    s = " ".join(line.split())
    if not (6 <= len(s) <= 120):
        return False
    if _looks_like_ui_line(s):
        return False
    words = s.split()
    if any(w.lower() in _UI_WORDS for w in words) and not _IDENT_RE.search(s) and not _FILE_RE.search(s):
        if len(words) <= 3:
            return False
    if _FILE_RE.search(s) or _IDENT_RE.search(s):
        return len(words) <= 14
    han = re.findall(r"[一-龥]", s)
    if 6 <= len(han) <= 24 and "。" not in s and len(s) <= 48:
        return True
    caps = sum(1 for w in words if w[:1].isupper())
    return caps >= 2 and 3 <= len(words) <= 12 and not s.endswith("。")


def _looks_like_slide_body(line: str) -> bool:
    s = " ".join(line.split())
    if not (12 <= len(s) <= 240):
        return False
    if _is_noise_line(s):
        return False
    return bool(re.search(r"[A-Za-z]{4,}|[一-龥]{4,}", s))


def _is_spelling_anchor(token: str) -> bool:
    t = token.strip()
    if len(t) < 3 or " " in t:
        return False
    if _FILE_RE.search(t) or _IDENT_RE.fullmatch(t):
        return True
    # Title-case words taken off a headline: long enough to be a name.
    return t[:1].isupper() and t[1:].islower() and len(t) >= 6 and t.lower() not in _STOP


def _similar(word: str, term: str) -> bool:
    if word == term:
        return False
    a, b = word.lower(), term.lower()
    if a == b:
        return True
    if min(len(a), len(b)) < 4:
        return False
    if abs(len(a) - len(b)) > 3:
        return False
    # Unusual OCR shape (mixed case / digits / file): extra vowels in ASR
    # often still share the consonant skeleton.
    if _is_spelling_anchor(term):
        folded = _vowel_fold(a)
        if folded == _vowel_fold(b) and len(folded) >= 2:
            return True
    return SequenceMatcher(None, a, b).ratio() >= 0.82


def _vowel_fold(s: str) -> str:
    return re.sub(r"[aeiou]", "", s)


def _rare_terms(
    paragraphs: list[tuple[str, int]],
    terms: list[str],
) -> list[str]:
    n = len(paragraphs)
    if n == 0 or not terms:
        return []
    cap = max(2, int(n * 0.15))
    rare: list[str] = []
    for term in terms:
        hits = sum(1 for line, _ in paragraphs if _line_hits_terms(line, [term]))
        if 1 <= hits <= cap:
            rare.append(term)
    return rare


def _pick_rarest_paragraphs(
    paragraphs: list[tuple[str, int]],
    must: list[int],
    rare: list[str],
    cap: int,
) -> list[int]:
    def score(idx: int) -> int:
        line = paragraphs[idx][0]
        hits = [
            sum(1 for p, _ in paragraphs if _line_hits_terms(p, [t]))
            for t in rare
            if _line_hits_terms(line, [t])
        ]
        return min(hits) if hits else len(paragraphs)

    ranked = sorted(must, key=score)
    return sorted(ranked[:cap])


def _line_hits_terms(line: str, terms: list[str]) -> bool:
    if not terms:
        return False
    low = line.lower()
    for term in terms:
        t = term.lower()
        if t in low:
            return True
        for word in re.findall(r"[A-Za-z][A-Za-z0-9+_-]*", line):
            if _similar(word, term) or word.lower() == t:
                return True
    return False
