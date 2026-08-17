"""Learning-phase detection and evidence slicing for the user-learning app.

Pure functions over activity-summary / search rows. Does not capture — it
selects which already-recorded screen/audio evidence belongs to learning
sessions so the LLM summarizes only that slice.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

# ── strong learning signals ─────────────────────────────────────────────────

_COURSEWARE_HOST_FRAGMENTS = (
    "coursera.org",
    "edx.org",
    "udacity.com",
    "udemy.com",
    "khanacademy.org",
    "classroom.google.com",
    "canvas.",
    "blackboard.",
    "moodle.",
    "icourse163.org",
    "zhihuishu.com",
    "chaoxing.com",
    "xuetangx.com",
    "study.163.com",
    "open.163.com",
    "bilibili.com",
    "youtube.com",
    "youtu.be",
    "docs.microsoft.com",
    "learn.microsoft.com",
    "developer.mozilla.org",
    "pytorch.org",
    "tensorflow.org",
    "huggingface.co",
    "arxiv.org",
    "cnblogs.com",
    "jianshu.com",
    "juejin.cn",
    "zhihu.com",
    "csdn.net",
    "runoob.com",
    "w3schools.com",
    "leetcode.com",
    "leetcode.cn",
    "nowcoder.com",
    "luogu.com.cn",
)

_QUERY_HOST_FRAGMENTS = (
    "google.",
    "bing.com",
    "baidu.com",
    "duckduckgo.com",
    "scholar.google.",
    "zhihu.com/search",
    "stackoverflow.com",
    "stackexchange.com",
    "github.com/search",
    "wikipedia.org",
    "zh.wikipedia.org",
)

_LEARNING_APP_PROCS = frozenset({
    "powerpnt.exe",
    "wps.exe",
    "wpsoffice.exe",
    "acrobat.exe",
    "acrobat reader.exe",
    "acrord32.exe",
    "foxitreader.exe",
    "foxitpdfreader.exe",
    "sumatrapdf.exe",
    "pdfxedit.exe",
    "koodo-reader.exe",
    "anki.exe",
    "obsidian.exe",
})

# Local / desktop video players — adaptive-learning-agent style: treat as
# candidate study surface when title/OCR looks educational (not every movie).
_VIDEO_PLAYER_PROCS = frozenset({
    "potplayer.exe",
    "potplayermini64.exe",
    "potplayermini.exe",
    "vlc.exe",
    "mpc-hc64.exe",
    "mpc-hc.exe",
    "mpc-be64.exe",
    "mpc-be.exe",
    "wmplayer.exe",
    "microsoft.media.player.exe",
    "video.ui.exe",           # Windows Films & TV / Media Player shell
    "applicationframehost.exe",  # UWP host — only with strong title/OCR
    "quicktimeplayer.exe",
    "kmplayer.exe",
    "stormplayer.exe",
    "qqplayer.exe",
    "qqlive.exe",
    "iqiyi.exe",
    "cloudmusic.exe",        # sometimes plays course audio/video
})

_VIDEO_HOST_FRAGMENTS = (
    "bilibili.com",
    "youtube.com",
    "youtu.be",
    "vimeo.com",
    "ted.com",
    "iqiyi.com",
    "youku.com",
    "v.qq.com",
    "ixigua.com",
)

_CODING_LEARN_PROCS = frozenset({
    "cursor.exe",
    "code.exe",
    "devenv.exe",
    "idea64.exe",
    "pycharm64.exe",
    "webstorm64.exe",
    "windowsterminal.exe",
    "powershell.exe",
    "cmd.exe",
    "wt.exe",
})

_TITLE_LEARN_RE = re.compile(
    r"(课件|课程|讲义|作业|习题|考试|复习|tutorial|lecture|courseware|"
    r"lesson|homework|assignment|教材|学堂|慕课|网课|学习|练习|"
    r"第\s*\d+\s*[讲章课回]|week\s*\d+|chapter\s*\d+|lecture\s*\d+|"
    r"公开课|精品课|考研|考公|四级|六级|托福|雅思|cs\d{2,3}|机器学习|深度学习|"
    r"操作系统|数据结构|算法|编译原理)",
    re.I,
)

# OCR / on-screen text that looks like a lecture slide or hard subtitle
# (adaptive-learning-agent: topic from captured text, not app name alone).
_OCR_LECTURE_RE = re.compile(
    r"(定义|定理|证明|引理|公式|例题|练习|小结|本节|本章|本节课|今天我们|"
    r"首先|其次|最后|步骤\s*\d|step\s*\d|"
    r"definition|theorem|lemma|corollary|proof|example|summary|"
    r"learning objectives|agenda|outline|"
    r"所谓|是指|指的是|表示为|推导|等价于|"
    r"attention|softmax|gradient|backprop|transformer|"
    r"第\s*\d+\s*[讲章节页]|slide\s*\d+|页码)",
    re.I,
)

_VIDEO_FILE_RE = re.compile(
    r"\.(mp4|mkv|avi|mov|wmv|flv|webm|m4v|ts|mpeg|mpg)(\b|$)",
    re.I,
)

_PROBLEM_RE = re.compile(
    r"(traceback|exception|error:|failed|failure|wrong answer|compilation error|"
    r"undefined reference|syntaxerror|typeerror|报错|错误|失败|异常)",
    re.I,
)

# ── spoken-lecture cues (audio channel) ──────────────────────────────────────
# Calibrated against a real 35-minute recorded technical talk (630 transcript
# rows, 9327 chars). An earlier version of this list was written from intuition
# — 也就是说 / 换句话说 / 举例来说 — and scored that lecture at ZERO: people
# giving talks do not speak in written-language connectives.
#
# What they actually do is ADDRESS AN AUDIENCE. In that recording 我们 appeared
# 116 times and 大家 112 — together 2.4 per 100 characters. Nobody talking to
# themselves, and nobody in a two-person conversation, says 大家; it presupposes
# a room. That register, not any single phrase, is the usable signal.
_AUDIO_AUDIENCE_RE = re.compile(
    r"(大家|我们|咱们|各位|"
    r"\bwe\b|\bour\b|\bus\b|\byou all\b|\beveryone\b|\bfolks\b)",
    re.I,
)

# Structural moves through material — weaker individually but they confirm a
# prepared talk rather than chat. Colloquial forms come from the observed
# sample; the formal/written connectives are kept too, since a scripted talk, a
# read-aloud paper or an English lecture leans on them even though the recorded
# Chinese speaker did not.
_AUDIO_LECTURE_RE = re.compile(
    # observed in real speech
    r"(首先|接下来|然后|前面讲|前面提到|刚才讲|我们来看|我们再看|大家可以看到|"
    r"大家注意|这里要注意|需要注意|比如|例如|举个例子|总结|小结|"
    r"第[一二三四五六]|这一讲|这一章|这节课|本节课|下一节|"
    # formal / written register
    r"也就是说|换句话说|举例来说|简单来说|具体来说|一般来说|所谓的|"
    r"可以理解为|值得注意的是|由此可见|综上所述|回顾一下|定义为|"
    r"其原理|本质上|从而|因此|此外|另外|最后|"
    # English
    r"let's look|let's take a look|first of all|next we|as you can see|"
    r"for example|for instance|to summarize|in other words|that is to say|"
    r"note that|keep in mind|moving on|in conclusion|it follows that|"
    r"is defined as|in essence|therefore|furthermore|finally)",
    re.I,
)

# Domain vocabulary. A talk that teaches something is dense in subject nouns —
# the reference recording measured 4.2 hits per 100 characters, higher even than
# its audience-address density. Casual conversation and entertainment are not.
# Supporting evidence only: a work discussion is equally technical, so this
# never carries a verdict on its own.
_AUDIO_TECHNICAL_RE = re.compile(
    r"(模型|推理|部署|优化|性能|硬件|平台|版本|生成|训练|算法|架构|框架|接口|"
    r"参数|数据|内存|显存|加速|量化|精度|编译|运行时|插件|函数|变量|指针|"
    r"神经网络|深度学习|机器学习|梯度|卷积|注意力|向量|矩阵|张量|"
    r"数据库|服务器|分布式|并发|线程|进程|缓存|协议|架构师|"
    r"定理|公式|推导|证明|方程|函数图|概率|统计|"
    r"\bAPI\b|\bSDK\b|\bGPU\b|\bCPU\b|\bNPU\b|\bTPU\b|\bpipeline\b|\bruntime\b|"
    r"\bmodel\b|\binference\b|\btraining\b|\bgradient\b|\btensor\b|\bkernel\b|"
    r"\balgorithm\b|\bcompiler\b|\bthroughput\b|\blatency\b|\bquantiz)",
    re.I,
)

# Sustained-speech thresholds over LearningConfig.audio_lookback_seconds (90s).
# The reference talk ran ~18 transcript rows/minute at ~15 chars each, so a
# populated 90-second window holds roughly 400 characters. A chime, a passing
# remark or background music yields almost nothing. Continuity is the main
# defence against "any audio counts as studying" — vocabulary alone is far too
# weak, since a podcast says 大家 just as often as a lecture does.
_AUDIO_MIN_CHARS = 150      # below this, speech is incidental, not a lecture
_AUDIO_RICH_CHARS = 450     # sustained narration filling the lookback window
# Densities (hits per 100 chars) at which each signal counts as fully present.
# Set roughly half the measured values so an ordinary talk clears them without
# the thresholds being trivially met by passing mentions.
_AUDIO_AUDIENCE_DENSITY = 1.2    # reference lecture measured 2.4
_AUDIO_TECHNICAL_DENSITY = 2.0   # reference lecture measured 4.2

_GAP_SEC = 180  # merge learning samples within 3 minutes into one session

# Courseware OCR is the secondary source for 讲解重点; budgeted like audio.
_OCR_MAX_LINES = 60


def select_spanning(items: list[Any], keep: int) -> list[Any]:
    """Keep ``keep`` items spread EVENLY across ``items``, preserving order.

    Used when lecture evidence exceeds the prompt budget. Head/tail truncation
    would hand the model one contiguous slice of a class — the opening or the
    closing — and every concept taught outside that slice becomes invisible.
    Sampling across the whole span keeps coverage roughly uniform, so recall
    degrades gracefully instead of falling off a cliff. First and last items are
    always retained so the reported time range stays honest.
    """
    n = len(items)
    if keep >= n or keep <= 0:
        return list(items)
    if keep == 1:
        return [items[0]]
    step = (n - 1) / (keep - 1)
    picked = sorted({min(n - 1, int(round(i * step))) for i in range(keep)})
    return [items[i] for i in picked]


# Internal alias so this module's call sites read consistently with agent.py.
_select_spanning = select_spanning


def normalize_always_rules(rules: Any) -> tuple[str, ...]:
    """Clean a user-supplied always-learning list into comparable tokens."""
    if isinstance(rules, str):
        rules = [rules]
    out: list[str] = []
    seen: set[str] = set()
    for raw in rules or ():
        token = str(raw or "").strip().lower()
        # A 1-2 char rule would match almost any page; refuse rather than let a
        # stray entry silently turn every window into a study session.
        if len(token) < 3 or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return tuple(out)


def load_always_rules() -> tuple[str, ...]:
    """Read ``[learning] always_learning`` from config, tolerating its absence.

    Callers that already hold a Config (the daemon detector) should pass their
    own value; this is for the app-side paths that have no config handle. Reads
    fresh each call so an edit through the API takes effect on the next recap
    without a restart — the list is tiny and these paths run once per report.
    """
    try:
        from ..config import load  # noqa: PLC0415

        return normalize_always_rules(getattr(load().learning, "always_learning", []))
    except Exception:  # noqa: BLE001
        return ()


def match_always_learning(
    rules: tuple[str, ...] | list[str],
    *,
    app_name: str = "",
    window_name: str = "",
    browser_url: str = "",
    text: str = "",
) -> str:
    """Return the first always-learning rule matching this observation, else ''.

    Matched against URL, host, window title, app name and captured on-screen
    text, because the identifying string lives in different places per source:
    a domain is in the URL, but a channel/UP主 name usually appears only in the
    page text (bilibili window titles are just ``<video>_哔哩哔哩_bilibili``).

    On-screen text is capped before matching so a rule cannot be triggered by
    something buried deep in a long OCR dump of an unrelated page.
    """
    cleaned = normalize_always_rules(rules)
    if not cleaned:
        return ""
    haystacks = (
        (browser_url or "").lower(),
        _host(browser_url or ""),
        (window_name or "").lower(),
        _norm_app(app_name),
        (text or "")[:2000].lower(),
    )
    for rule in cleaned:
        for hay in haystacks:
            if hay and rule in hay:
                return rule
    return ""


def _norm_app(name: str) -> str:
    return (name or "").strip().lower()


def _host(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:  # noqa: BLE001
        return ""


def _path_query(url: str) -> str:
    try:
        p = urlparse(url)
        return f"{p.path or ''}?{p.query or ''}".lower()
    except Exception:  # noqa: BLE001
        return ""


def lecture_content_score(*, title: str = "", text: str = "", pathq: str = "") -> float:
    """How lecture-like on-screen / title text looks (0..1).

    Inspired by adaptive-learning-agent's OCR→topic path: dense educational
    phrases raise the score even when the app is a generic video player.
    """
    blob = f"{title} {pathq} {text}".strip()
    if len(blob) < 4:
        return 0.0
    score = 0.0
    if _TITLE_LEARN_RE.search(title) or _TITLE_LEARN_RE.search(pathq):
        score += 0.55
    if _TITLE_LEARN_RE.search(text or ""):
        score += 0.25
    ocr_hits = _OCR_LECTURE_RE.findall(text or "")
    if ocr_hits:
        score += min(0.55, 0.18 * len(set(h.lower() if isinstance(h, str) else h for h in ocr_hits)))
    if _VIDEO_FILE_RE.search(title) and score >= 0.25:
        score += 0.15
    # Long OCR with multiple lecture cues ≈ slide deck / hardsubs
    if text and len(text) >= 80 and len(ocr_hits) >= 2:
        score += 0.1
    return max(0.0, min(1.0, score))


# Target size of one merged transcript paragraph, in characters.
_PARA_CHARS = 220


def merge_transcript_paragraphs(
    rows: list[dict[str, Any]],
    fmt: Any,
) -> list[tuple[str, int]]:
    """Group consecutive transcript rows into timestamped paragraphs.

    Whisper emits a row every few seconds — around 14 characters of Chinese —
    which are sentence fragments, not sentences. Feeding them one per line costs
    a timestamp per fragment and hands the reader shredded prose.

    Merging consecutive rows until a paragraph is worth reading fixes both: one
    timestamp per paragraph instead of per fragment, and complete sentences. On a
    measured 35-minute talk this is the difference between fitting half the
    lecture in a prompt and fitting all of it.

    Returns ``(paragraph, rows_merged)`` pairs. The row count travels with the
    text so coverage can be reported in the same unit the total is counted in —
    reporting "4 of 25" for four paragraphs holding all twenty-five rows would
    describe a complete transcript as partial.
    """
    out: list[tuple[str, int]] = []
    buf: list[str] = []
    buf_ts = ""
    buf_len = 0
    buf_rows = 0

    def flush() -> None:
        nonlocal buf, buf_ts, buf_len, buf_rows
        if buf:
            line = fmt(buf_ts, " ".join(buf))
            if line:
                out.append((line, buf_rows))
        buf, buf_ts, buf_len, buf_rows = [], "", 0, 0

    for r in rows:
        text = " ".join(
            str(r.get("redacted_transcription") or r.get("transcription") or "").split()
        )
        if not text:
            continue
        if not buf:
            buf_ts = str(r.get("timestamp") or "")
        buf.append(text)
        buf_len += len(text)
        buf_rows += 1
        if buf_len >= _PARA_CHARS:
            flush()
    flush()
    return out


_SELF_UI_HOSTS = ("127.0.0.1:3030", "localhost:3030")
_SELF_UI_TITLE_RE = re.compile(r"\bdeskmate\b", re.I)


def _is_self_ui(*, app_name: str = "", window_name: str = "", browser_url: str = "") -> bool:
    """Is this DeskMate looking at its own interface?

    Its pages describe studying, so they read as study material: the Learning
    page alone contains 学习 / 课件 / 复习 / 知识点 / 讲解重点. Observing them
    produced sessions titled "DeskMate - Google Chrome" whose extracted concepts
    were the page's own button labels.
    """
    url = (browser_url or "").lower()
    if any(h in url for h in _SELF_UI_HOSTS):
        return True
    if "/ui/assets" in url:
        return True
    return bool(_SELF_UI_TITLE_RE.search(window_name or ""))


def looks_like_media_surface(
    *,
    app_name: str = "",
    window_name: str = "",
    browser_url: str = "",
) -> bool:
    """Is this window a video/lecture surface (a player, a video site, a file)?

    Used to remember *which* window is the media one. Frame capture only sees
    the foreground, so when a class plays in a background tab the only title
    available at that moment belongs to an unrelated window. But the media
    window's own title — which names the lecture — does get captured whenever the
    user glances at it, so caching the last one seen recovers the subject that
    audio alone cannot supply.
    """
    app = _norm_app(app_name)
    host = _host(browser_url or "")
    title = window_name or ""
    if app in _VIDEO_PLAYER_PROCS:
        return True
    if host and any(x in host for x in _VIDEO_HOST_FRAGMENTS):
        return True
    if host and any(x in host for x in _COURSEWARE_HOST_FRAGMENTS):
        return True
    return bool(_VIDEO_FILE_RE.search(title))


# Window furniture. Present in every OCR dump, meaningless as evidence, and the
# reason a "problem" could be reported as `Minimize / Maximize / Restore / Close`.
_PROSE_PUNCT_RE = re.compile(r"[，、。；！？]")

_CHROME_LINE_RE = re.compile(
    r"^(minimize|maximize|restore|close|back|forward|reload|new tab|"
    r"最小化|最大化|还原|关闭|后退|前进|刷新|新建标签页|开始|任务栏|"
    r"desktopwindowxamlsource|运行中的应用程序|已固定)\b",
    re.I,
)


def detect_problem_text(*parts: str) -> tuple[str, str]:
    """Find an error on screen. Returns ``(matched_line, marker)``, else ``("","")``.

    Split out of :func:`classify_learning_signal` on purpose: an error says
    "something broke", not "this is what the user is studying". Treating it as a
    session *kind* let one stray word relabel a whole session, because the check
    ran first and outranked every real learning signal.

    Two properties this returns rather than a bare marker:

    * **The line, not the dump.** The caller needs something to store. Passing the
      head of the OCR instead produced problem records reading "hongbo - Visual
      Studio Code / Minimize / Maximize / Restore / Close" — the match itself was
      hundreds of characters further down and never made it into the record.
    * **Line-isolated matching.** A stack trace occupies its own line; the word
      "failed" inside a sentence on a docs page, a chat log or these very notes
      does not. Matching the whole blob is why unrelated screens became problems.
    """
    for part in parts:
        for raw in (part or "").splitlines():
            line = raw.strip()
            if len(line) < 6 or _CHROME_LINE_RE.match(line):
                continue
            if len(line) > 300 or _PROSE_PUNCT_RE.search(line):
                # Prose that merely mentions failure — a lecture explaining error
                # handling, a note about a past bug, these very sentences. Real
                # diagnostics do not carry narrative punctuation; a full-width
                # comma or period is the clearest sign this is someone talking
                # about an error rather than an error being reported.
                continue
            m = _PROBLEM_RE.search(line)
            if not m:
                continue
            return line[:300], m.group(0)
    return "", ""


def lecture_audio_score(
    audio_text: str,
    *,
    lookback_seconds: float = 90.0,
) -> float:
    """How much the recent AUDIO sounds like someone teaching (0..1).

    Separate from :func:`lecture_content_score`, which scores what is *on the
    screen*. The two answer different questions, and conflating them is why a
    lecture playing in the background could never be detected: when a video is
    fullscreen there is almost no on-screen text to score, and when the user is
    working in another window the screen describes the work, not the lecture.

    Four factors, weighted:

    * **Sustained narration** — enough speech in the lookback window that this
      cannot be a passing remark. Keeps chimes and one-liners out.
    * **Audience address** — density of 大家 / 我们 / we / everyone. Speech aimed
      at a room, which is what teaching is. The most reliable single signal.
    * **Structural moves** — 首先 / 接下来 / 也就是说 / to summarize: a prepared
      walk through material rather than conversation.
    * **Domain vocabulary** — subject nouns, dense in any talk that teaches
      something. Supporting evidence only; a work discussion is just as
      technical, so it never carries the verdict alone.

    Honest limit: this cannot separate a technical *podcast* from a technical
    *lecture* — linguistically they are the same thing. Callers treat an
    audio-only verdict as provisional rather than certain, and users can name
    specific sources via the always-learning whitelist.
    """
    text = (audio_text or "").strip()
    if not text:
        return 0.0
    chars = len(text)
    if chars < _AUDIO_MIN_CHARS:
        return 0.0

    # Volume: ramps from 0 at _AUDIO_MIN_CHARS to 1 at _AUDIO_RICH_CHARS.
    span = max(1.0, float(_AUDIO_RICH_CHARS - _AUDIO_MIN_CHARS))
    volume = min(1.0, (chars - _AUDIO_MIN_CHARS) / span)
    # A longer lookback needs proportionally more speech to mean the same thing.
    if lookback_seconds > 0:
        volume *= min(1.0, 90.0 / float(lookback_seconds))

    def _density(rx: re.Pattern[str], per_100: float) -> float:
        return min(1.0, (100.0 * len(rx.findall(text)) / max(1, chars)) / per_100)

    audience = _density(_AUDIO_AUDIENCE_RE, _AUDIO_AUDIENCE_DENSITY)
    technical = _density(_AUDIO_TECHNICAL_RE, _AUDIO_TECHNICAL_DENSITY)

    markers = {m.lower() for m in _AUDIO_LECTURE_RE.findall(text)}
    marker_score = min(1.0, 0.25 * len(markers))

    # Audience address carries the most weight; volume alone must never be
    # enough, or any long stretch of speech would register as a class.
    score = 0.20 * volume + 0.35 * audience + 0.25 * marker_score + 0.20 * technical
    if not markers and audience < 0.5:
        # Dense speech that neither addresses a room nor walks through material:
        # a film, a call, music with lyrics. Hard-capped below the gate even if
        # the vocabulary happens to be technical.
        score = min(score, 0.35)
    return max(0.0, min(1.0, score))


def extract_search_query(url: str) -> str:
    """Best-effort query string from a search / docs URL."""
    if not url:
        return ""
    try:
        p = urlparse(url)
        qs = parse_qs(p.query)
        for key in ("q", "query", "wd", "keyword", "search"):
            if key in qs and qs[key]:
                return unquote(qs[key][0]).strip()[:200]
    except Exception:  # noqa: BLE001
        return ""
    return ""


def classify_learning_signal(
    *,
    app_name: str = "",
    window_name: str = "",
    browser_url: str = "",
    text: str = "",
    always_rules: tuple[str, ...] | list[str] = (),
    audio_text: str = "",
    audio_lookback_seconds: float = 90.0,
) -> tuple[str | None, float, str]:
    """Return (kind, confidence, reason) or (None, 0, '') if not learning.

    kind ∈ {courseware_view, material_query, code_edit, study_other}

    Video path (adaptive-learning-agent inspired):
      player/site is only a *candidate*; title + OCR lecture cues decide.

    ``always_rules`` is the user's always-learning whitelist (``[learning]
    always_learning``). A hit bypasses the lecture-score gate, which otherwise
    rejects real technical talks whose titles don't read like coursework.

    ``audio_text`` is recent speech, kept OUT of ``text`` on purpose. Merging it
    into the on-screen blob (the old behaviour) meant a lecture's transcript
    could only ever nudge the score of whatever window happened to be in front,
    so listening to a class while working was filed under the foreground app.
    Scored separately it can carry its own verdict.

    Precedence, and why:

    1. **whitelist** — the user said so explicitly; nothing should override it.
    2. **foreground courseware** — the user is *looking at* material; direct
       evidence beats inferred.
    3. **lecture audio** — a class is playing though the foreground is
       unrelated. Ranked below 2 so real courseware still wins, above 4 so the
       foreground app cannot bury it.
    4. **learning keyword in the window title** — weakest usable evidence.

    Editors and terminals are not on this list at all; see the note in the body.
    ``problem`` is deliberately absent too: see :func:`detect_problem_text`.
    """
    app = _norm_app(app_name)
    title = window_name or ""
    url = (browser_url or "").strip()
    host = _host(url)
    pathq = _path_query(url)
    blob = f"{title} {text}".strip()
    lec = lecture_content_score(title=title, text=text or "", pathq=pathq)

    matched = match_always_learning(
        always_rules,
        app_name=app_name,
        window_name=window_name,
        browser_url=url,
        text=text,
    )
    if matched:
        # Confidence is pinned above any configured start_confidence so an
        # explicitly trusted source always opens a session; the reason names the
        # rule so the decision stays auditable in learning_sessions.reason.
        kind = "material_query" if extract_search_query(url) else "courseware_view"
        return kind, 0.95, f"always-learning rule: {matched}"

    # DeskMate's own windows are never coursework. Its Learning page is written
    # in exactly the vocabulary the detector looks for — 学习 / 课件 / 复习 /
    # 知识点 / 讲解重点 — so reading one's own dashboard scored as a lecture and
    # then fed the UI's copy back in as "concepts" ("正在记录", "直到你点结束",
    # "接下来该复习什么"). Placed after the whitelist so a user who really does
    # want it counted can say so explicitly.
    if _is_self_ui(app_name=app_name, window_name=window_name, browser_url=url):
        return None, 0.0, ""

    if url:
        q = extract_search_query(url)
        if q and any(h in host or h in pathq for h in _QUERY_HOST_FRAGMENTS):
            return "material_query", 0.9, f"search query: {q}"
        if any(h in host for h in _COURSEWARE_HOST_FRAGMENTS):
            is_video_host = any(x in host for x in _VIDEO_HOST_FRAGMENTS)
            if is_video_host:
                # Bilibili/YouTube: title OR on-screen OCR/subtitles look like class.
                if lec >= 0.45:
                    return (
                        "courseware_view",
                        min(0.92, 0.7 + 0.25 * lec),
                        f"video site + lecture cues ({host})",
                    )
                return None, 0.0, ""
            return "courseware_view", 0.9, f"course/docs host: {host}"
        # Generic video CDN / share page without course host list
        if any(x in host for x in _VIDEO_HOST_FRAGMENTS) and lec >= 0.45:
            return (
                "courseware_view",
                min(0.9, 0.68 + 0.25 * lec),
                f"video host + lecture cues ({host})",
            )
        if q and _TITLE_LEARN_RE.search(q):
            return "material_query", 0.75, f"learning-flavored query: {q}"

    if app in _LEARNING_APP_PROCS:
        return "courseware_view", 0.85, f"reader/office app: {app}"

    # Local video player / UWP media: need lecture-like title or OCR.
    if app in _VIDEO_PLAYER_PROCS or (
        app == "applicationframehost.exe" and (_VIDEO_FILE_RE.search(title) or lec >= 0.5)
    ):
        if lec >= 0.45:
            return (
                "courseware_view",
                min(0.9, 0.72 + 0.2 * lec),
                f"video player + lecture cues: {app or 'player'}",
            )
        # Filename alone: "机器学习第3讲.mp4"
        if _VIDEO_FILE_RE.search(title) and _TITLE_LEARN_RE.search(title):
            return "courseware_view", 0.8, f"video file with learning title: {app}"
        return None, 0.0, ""

    # Browser / other app showing a video file name + lecture OCR (no URL yet)
    if _VIDEO_FILE_RE.search(title) and lec >= 0.5:
        return "courseware_view", 0.78, "video filename + lecture on-screen text"

    # Strong OCR-only lecture surface (slides occupying the screen)
    if lec >= 0.7 and len(text or "") >= 40:
        return "courseware_view", min(0.88, 0.65 + 0.25 * lec), "on-screen lecture OCR/slides"

    # Lecture playing while the foreground is something else — the audio-only
    # study case. Placed after every on-screen courseware signal (looking at
    # material is stronger evidence) but before the IDE/title rules, which would
    # otherwise file "listening to a class while coding" as plain coding.
    #
    # Capped below the on-screen paths: this is an inference from sound alone
    # and cannot tell a lecture from a technical podcast, so it stays reviewable
    # rather than authoritative.
    aud = lecture_audio_score(audio_text, lookback_seconds=audio_lookback_seconds)
    if aud >= 0.55:
        return (
            "courseware_view",
            min(0.82, 0.55 + 0.3 * aud),
            f"lecture audio while foreground is unrelated ({app or 'desktop'})",
        )

    # NOTE: editors and terminals produce no learning signal at all.
    #
    # There used to be three tiers here — a learning word in the title (0.80), a
    # source file open (0.70), an IDE merely in the foreground (0.65). All three
    # sat above keep_confidence, so any of them kept a study session alive
    # indefinitely, and every one of them is just as true on an ordinary workday
    # as it is while studying. Nothing about "an editor is open" distinguishes
    # practising from working.
    #
    # Judging it by content instead (does this code relate to the lecture?) was
    # tried and abandoned: it hinges on the quality of extracted concepts, needs
    # a stoplist of vocabulary common to both, and still misses whenever the
    # learner renames things. The manual "start studying" control covers the
    # real case directly and without guessing — if you are studying in an editor,
    # say so, and the whole span counts.

    if _TITLE_LEARN_RE.search(title):
        return "study_other", 0.7, "learning keyword in window title"

    return None, 0.0, ""

def _parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _fmt_ts(dt: datetime | None) -> str:
    if not dt:
        return ""
    local = dt.astimezone() if dt.tzinfo else dt
    return local.strftime("%Y-%m-%d %H:%M")


def build_learning_sessions(
    summary: dict[str, Any],
    *,
    always_rules: tuple[str, ...] | list[str] | None = None,
) -> list[dict[str, Any]]:
    """Project activity-summary timeline/windows into merged learning sessions."""
    samples: list[dict[str, Any]] = []
    rules = normalize_always_rules(
        always_rules if always_rules is not None else load_always_rules()
    )

    for row in summary.get("timeline") or []:
        kind, conf, reason = classify_learning_signal(
            app_name=str(row.get("app_name") or ""),
            window_name=str(row.get("window_name") or ""),
            browser_url=str(row.get("browser_url") or ""),
            text=str(row.get("text") or ""),
            always_rules=rules,
        )
        if not kind:
            continue
        ts = _parse_ts(str(row.get("timestamp") or ""))
        if not ts:
            continue
        samples.append({
            "ts": ts,
            "kind": kind,
            "confidence": conf,
            "reason": reason,
            "app_name": row.get("app_name") or "",
            "window_name": row.get("window_name") or "",
            "browser_url": row.get("browser_url") or "",
            "text": (row.get("text") or "")[:240],
            "minutes": float(row.get("minutes") or 0),
        })

    # Windows that never appear in the capped timeline still matter.
    for row in summary.get("windows") or []:
        kind, conf, reason = classify_learning_signal(
            app_name=str(row.get("app_name") or ""),
            window_name=str(row.get("window_name") or ""),
            browser_url=str(row.get("browser_url") or ""),
            always_rules=rules,
        )
        if not kind:
            continue
        # Approximate placement using first/last if present; else skip merge seed.
        ts = _parse_ts(str(row.get("first_seen") or row.get("last_seen") or ""))
        if not ts:
            continue
        samples.append({
            "ts": ts,
            "kind": kind,
            "confidence": conf,
            "reason": reason,
            "app_name": row.get("app_name") or "",
            "window_name": row.get("window_name") or "",
            "browser_url": row.get("browser_url") or "",
            "text": "",
            "minutes": float(row.get("minutes") or 0),
        })

    samples.sort(key=lambda s: s["ts"])
    if not samples:
        return []

    sessions: list[dict[str, Any]] = []
    cur: dict[str, Any] | None = None

    for s in samples:
        if cur is None:
            cur = {
                "kind": s["kind"],
                "started_at": s["ts"],
                "ended_at": s["ts"],
                "apps": {s["app_name"]} if s["app_name"] else set(),
                "titles": [s["window_name"]] if s["window_name"] else [],
                "urls": [s["browser_url"]] if s["browser_url"] else [],
                "reasons": [s["reason"]],
                "evidence": [s],
                "confidence": s["confidence"],
                "minutes_hint": float(s.get("minutes") or 0),
            }
            continue

        gap = (s["ts"] - cur["ended_at"]).total_seconds()
        same_kind = s["kind"] == cur["kind"] or {
            s["kind"], cur["kind"],
        } <= {"code_edit", "problem", "study_other"}
        if gap <= _GAP_SEC and same_kind:
            cur["ended_at"] = s["ts"]
            if s["app_name"]:
                cur["apps"].add(s["app_name"])
            if s["window_name"] and s["window_name"] not in cur["titles"]:
                cur["titles"].append(s["window_name"])
            if s["browser_url"] and s["browser_url"] not in cur["urls"]:
                cur["urls"].append(s["browser_url"])
            cur["reasons"].append(s["reason"])
            cur["evidence"].append(s)
            cur["confidence"] = max(cur["confidence"], s["confidence"])
            cur["minutes_hint"] += float(s.get("minutes") or 0)
        else:
            sessions.append(cur)
            cur = {
                "kind": s["kind"],
                "started_at": s["ts"],
                "ended_at": s["ts"],
                "apps": {s["app_name"]} if s["app_name"] else set(),
                "titles": [s["window_name"]] if s["window_name"] else [],
                "urls": [s["browser_url"]] if s["browser_url"] else [],
                "reasons": [s["reason"]],
                "evidence": [s],
                "confidence": s["confidence"],
                "minutes_hint": float(s.get("minutes") or 0),
            }
    if cur:
        sessions.append(cur)

    out: list[dict[str, Any]] = []
    for i, sess in enumerate(sessions, 1):
        span = max(0.0, (sess["ended_at"] - sess["started_at"]).total_seconds() / 60.0)
        duration = max(span, float(sess.get("minutes_hint") or 0))
        # Drop tiny noise blips unless high confidence problem/query.
        if duration < 0.5 and sess["kind"] not in {"problem", "material_query"}:
            if sess["confidence"] < 0.85:
                continue
        title = (sess["titles"][0] if sess["titles"] else "") or (
            sess["urls"][0] if sess["urls"] else sess["kind"]
        )
        queries = []
        for u in sess["urls"]:
            q = extract_search_query(u)
            if q and q not in queries:
                queries.append(q)
        sample = next(
            (e["text"] for e in sess["evidence"] if e.get("text")),
            "",
        )
        # Topic/concept tags on the session (adaptive-learning-agent style) —
        # not just "is learning", but *what* is being studied.
        tag_blobs = [title, sample, *queries, *sess["titles"][:3]]
        topics, concepts = _tag_session_texts(tag_blobs, kind=sess["kind"])
        out.append({
            "id": i,
            "kind": sess["kind"],
            "title": title[:160],
            "started_at": _fmt_ts(sess["started_at"]),
            "ended_at": _fmt_ts(sess["ended_at"]),
            "duration_min": round(duration, 1),
            "apps": sorted(a for a in sess["apps"] if a),
            "urls": sess["urls"][:8],
            "queries": queries[:8],
            "topics": topics,
            "concepts": concepts,
            "confidence": round(float(sess["confidence"]), 2),
            "reason": sess["reasons"][0] if sess["reasons"] else "",
            "sample_text": sample,
        })
    return out


def _tag_session_texts(texts: list[str], *, kind: str = "") -> tuple[list[str], list[str]]:
    """Return (topics, concepts) for a learning session from local evidence."""
    try:
        from deskmate.learning_memory.extract import (  # noqa: PLC0415
            extract_concepts_from_texts,
        )
    except Exception:  # noqa: BLE001
        return [], []
    blobs = [t for t in texts if t and str(t).strip()]
    if not blobs:
        return ([], [kind] if kind else [])
    hits = extract_concepts_from_texts(blobs, max_concepts=8)
    concepts = [h.name for h in hits[:6]]
    topics = []
    for h in hits:
        if h.topic and h.topic not in topics and h.topic not in {"general", "general-zh"}:
            topics.append(h.topic)
    if not topics and kind:
        topics = [kind]
    return topics[:4], concepts


def in_any_span(ts: str, spans: list[tuple[str, str]] | tuple) -> bool:
    """Does ``ts`` fall inside any (start, end) span? Bounds may be blank."""
    stamp = (ts or "").strip().replace(" ", "T", 1)
    if not stamp:
        return False
    for lo, hi in spans or ():
        lo_n = (lo or "").strip().replace(" ", "T", 1)
        hi_n = (hi or "").strip().replace(" ", "T", 1)
        if lo_n and stamp < lo_n:
            continue
        if hi_n and stamp > hi_n + "￿":
            continue
        if lo_n or hi_n:
            return True
    return False


def filter_learning_key_texts(
    key_texts: list[dict[str, Any]],
    *,
    limit: int = 80,
    prefer_courseware: bool = True,
    declared_spans: list[tuple[str, str]] | tuple = (),
) -> list[dict[str, Any]]:
    """Keep key_texts that look like learning / problems / study notes.

    When ``prefer_courseware`` is set, courseware_view / material_query rows are
    sorted ahead of code/problem so lecture OCR fills the prompt budget first.

    ``declared_spans`` are the windows of sessions the user started by hand.
    Screen activity inside one is kept whatever the classifier makes of it,
    because "should this keep a session alive?" and "is this evidence of what
    the user did while studying?" are different questions. Answering both with
    the same classifier meant that once editors stopped counting as a learning
    signal, the experiments someone ran against the lecture — the code, the
    terminal output — vanished from their own study report unless they happened
    to throw an error.
    """
    scored: list[tuple[int, dict[str, Any]]] = []
    for row in key_texts or []:
        kind, conf, _ = classify_learning_signal(
            app_name=str(row.get("app_name") or ""),
            window_name=str(row.get("window_name") or ""),
            browser_url=str(row.get("browser_url") or ""),
            text=str(row.get("text") or ""),
        )
        declared = in_any_span(str(row.get("timestamp") or ""), declared_spans)
        if not kind and declared:
            # Inside a declared session: the user's word outranks the heuristics.
            kind, conf = "study_other", 0.6
        if not kind and not (row.get("text") and _PROBLEM_RE.search(str(row.get("text") or ""))):
            continue
        if not kind:
            kind = "problem"
            conf = 0.8
        rank = {
            "courseware_view": 0,
            "material_query": 1,
            "study_other": 2,
            "code_edit": 3,
            "problem": 4,
        }.get(kind, 5)
        if not prefer_courseware:
            rank = 0
        scored.append((rank, -float(conf), row))
    scored.sort(key=lambda x: (x[0], x[1]))
    return [r for _, __, r in scored[:limit]]


def filter_learning_edited_files(
    edited: list[dict[str, Any]],
    *,
    limit: int = 30,
) -> list[dict[str, Any]]:
    """Prefer source / notebook / notes files as learning artifacts."""
    scored: list[tuple[int, dict[str, Any]]] = []
    for row in edited or []:
        path = str(row.get("path") or "")
        low = path.lower()
        score = 0
        if re.search(r"\.(py|c|cpp|h|js|ts|java|go|rs|ipynb|md|tex|pdf|ppt|pptx)$", low):
            score += 2
        if any(k in low for k in ("homework", "hw", "lab", "course", "lecture", "作业", "实验", "课件")):
            score += 3
        if score:
            scored.append((score, row))
    scored.sort(key=lambda x: (-x[0], -int(x[1].get("frame_count") or 0)))
    return [r for _, r in scored[:limit]]


def format_learning_bundle(
    *,
    sessions: list[dict[str, Any]],
    key_texts: list[dict[str, Any]],
    edited_files: list[dict[str, Any]],
    audio_bits: list[str],
    range_start: str,
    range_end: str,
    courseware_ocr_lines: list[str] | None = None,
    audio_stats: dict[str, Any] | None = None,
) -> str:
    """Deterministic context block for the user-learning LLM prompt.

    ``audio_stats`` (from ``_collect_learning_audio_bits``) carries coverage
    facts — how many transcript lines exist vs. how many are shown, and whether
    they are in teaching order. Without it a partial transcript reads to the
    model as a complete one, and it reports confident 讲解重点 for a class it
    only saw part of. Mirrors the top-level ``_cap_prefetch_text`` watermark,
    but per source.
    """
    lines: list[str] = [
        "### Learning detection (pre-computed — trust this over raw browsing noise)",
        f"Window analyzed: {range_start} → {range_end}",
        f"Learning sessions found: {len(sessions)}",
        "",
    ]
    if not sessions:
        lines.append(
            "NO_LEARNING_SESSION: No courseware / material-query / study-coding / "
            "problem evidence crossed the detector threshold in this range. "
            "The report MUST say learning was not detected and skip inventing a study plan "
            "beyond a gentle suggestion to open course materials or a study IDE next time."
        )
        return "\n".join(lines)

    total_min = sum(float(s.get("duration_min") or 0) for s in sessions)
    by_kind: dict[str, float] = {}
    for s in sessions:
        by_kind[s["kind"]] = by_kind.get(s["kind"], 0.0) + float(s.get("duration_min") or 0)

    stats = audio_stats or {}
    a_total = int(stats.get("total") or 0)
    a_shown = int(stats.get("included") or len(audio_bits))
    if a_total > a_shown:
        coverage = f"{a_shown} of {a_total} (sampled evenly across the session)"
    elif a_total:
        coverage = f"{a_shown} of {a_total} (complete)"
    else:
        coverage = f"{a_shown} (total in window unknown)"

    lines.append(f"Total learning dwell (approx): {total_min:.1f} min")
    lines.append(f"Audio transcript lines available: {coverage}")
    lines.append("By kind:")
    for k, m in sorted(by_kind.items(), key=lambda x: -x[1]):
        lines.append(f"- {k}: {m:.1f} min")
    lines.append("")
    lines.append("### Learning sessions (slice — summarize ONLY these)")
    for s in sessions:
        lines.append(
            f"- [{s['id']}] {s['kind']} | {s['started_at']}–{s['ended_at']} | "
            f"{s['duration_min']} min | conf={s['confidence']}"
        )
        lines.append(f"  title: {s['title']}")
        if s.get("topics"):
            lines.append(f"  topics: {', '.join(s['topics'])}")
        if s.get("concepts"):
            lines.append(f"  concepts: {', '.join(s['concepts'])}")
        if s.get("apps"):
            lines.append(f"  apps: {', '.join(s['apps'])}")
        if s.get("queries"):
            lines.append(f"  queries: {'; '.join(s['queries'])}")
        if s.get("urls"):
            lines.append(f"  urls: {'; '.join(s['urls'][:3])}")
        if s.get("sample_text"):
            lines.append(f"  evidence: {s['sample_text'][:280]}")
        if s.get("reason"):
            lines.append(f"  why_learning: {s['reason']}")

    # Lecture audio first — primary source for course content and key points.
    lines.append("")
    if audio_bits:
        lines.append(
            "### Audio transcripts (lecture) — PRIMARY source for 讲了什么 / 课程重点"
        )
        if stats.get("ordered", True):
            lines.append(
                "Lines are in CHRONOLOGICAL order (oldest first) — treat their "
                "sequence as the teaching order when reconstructing 定义 → 步骤 → 关系."
            )
        else:
            lines.append(
                "⚠️ ORDER NOT GUARANTEED: these lines were retrieved by relevance, "
                "not by time. Do NOT infer teaching order from their sequence."
            )
        if a_total > a_shown:
            lines.append(
                f"⚠️ PARTIAL TRANSCRIPT: {a_shown} of {a_total} lines are shown, "
                "sampled evenly across the session to fit the model budget. "
                "Gaps between consecutive lines are NOT silence — content was "
                "skipped. You MUST say in 数据说明 that transcript coverage was "
                "partial, and you MUST NOT claim the lecture outline is complete."
            )
        lines.append(
            "Summarize what the speaker taught from these lines. Quote key phrases. "
            "Cite as 录音. Do not invent content absent here."
        )
        lines.extend(f"- {a}" for a in audio_bits)
    else:
        lines.append("### Audio transcripts (lecture)")
        lines.append(
            "NO_AUDIO_TRANSCRIPT: No usable lecture audio in this range. "
            "讲了什么/课程重点 must rely on 课件OCR only, or state material is insufficient."
        )

    ocr_lines = courseware_ocr_lines or []
    if ocr_lines:
        # Same silent-truncation hazard as audio: sample across the span rather
        # than keeping the first N slides, and say so when lines were dropped.
        shown_ocr = _select_spanning(ocr_lines, _OCR_MAX_LINES)
        lines.append("")
        lines.append(
            "### Courseware OCR / slides — secondary source for 讲解重点 (cite as 课件OCR)"
        )
        if len(shown_ocr) < len(ocr_lines):
            lines.append(
                f"⚠️ PARTIAL SLIDES: {len(shown_ocr)} of {len(ocr_lines)} OCR lines "
                "are shown, sampled evenly across the session. Say so in 数据说明."
            )
        lines.extend(shown_ocr)

    if key_texts:
        lines.append("")
        lines.append("### Learning-related key texts (OCR / typed — sliced)")
        for row in key_texts[:45]:
            ts = row.get("timestamp") or ""
            app = row.get("app_name") or ""
            win = row.get("window_name") or ""
            text = " ".join(str(row.get("text") or "").split())[:400]
            lines.append(f"- {ts} | {app} | {win}: {text}")

    if edited_files:
        lines.append("")
        lines.append("### Study artifacts (edited files — sliced)")
        for ef in edited_files[:25]:
            lines.append(f"- {ef.get('path', '')} ({ef.get('frame_count', 0)} captures)")

    lines.append("")
    lines.append(
        "INSTRUCTION: Maximize concrete courseware/lecture content in 讲解重点 and "
        "理解要点 when Audio transcripts / Courseware OCR exist. Ignore chat/shopping/"
        "random entertainment unless listed inside a session. Cite session ids, 录音, "
        "or 课件OCR. Never invent lecture points without evidence."
    )
    return "\n".join(lines)
