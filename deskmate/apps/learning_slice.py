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

_GAP_SEC = 180  # merge learning samples within 3 minutes into one session


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
) -> tuple[str | None, float, str]:
    """Return (kind, confidence, reason) or (None, 0, '') if not learning.

    kind ∈ {courseware_view, material_query, code_edit, problem, study_other}

    Video path (adaptive-learning-agent inspired):
      player/site is only a *candidate*; title + OCR lecture cues decide.
    """
    app = _norm_app(app_name)
    title = window_name or ""
    url = (browser_url or "").strip()
    host = _host(url)
    pathq = _path_query(url)
    blob = f"{title} {text}".strip()
    lec = lecture_content_score(title=title, text=text or "", pathq=pathq)

    if blob and _PROBLEM_RE.search(blob):
        return "problem", 0.85, "error/exception pattern in on-screen text"

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

    if app in _CODING_LEARN_PROCS:
        if _TITLE_LEARN_RE.search(title) or _TITLE_LEARN_RE.search(blob):
            return "code_edit", 0.8, "IDE/terminal with learning title"
        if re.search(r"\.(py|c|cpp|h|js|ts|java|go|rs|md|ipynb)\b", title, re.I):
            return "code_edit", 0.7, "IDE editing source file"
        if app in {"cursor.exe", "code.exe", "pycharm64.exe", "idea64.exe"}:
            return "code_edit", 0.65, "IDE foreground (study/practice coding)"

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


def build_learning_sessions(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Project activity-summary timeline/windows into merged learning sessions."""
    samples: list[dict[str, Any]] = []

    for row in summary.get("timeline") or []:
        kind, conf, reason = classify_learning_signal(
            app_name=str(row.get("app_name") or ""),
            window_name=str(row.get("window_name") or ""),
            browser_url=str(row.get("browser_url") or ""),
            text=str(row.get("text") or ""),
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


def filter_learning_key_texts(
    key_texts: list[dict[str, Any]],
    *,
    limit: int = 80,
    prefer_courseware: bool = True,
) -> list[dict[str, Any]]:
    """Keep key_texts that look like learning / problems / study notes.

    When ``prefer_courseware`` is set, courseware_view / material_query rows are
    sorted ahead of code/problem so lecture OCR fills the prompt budget first.
    """
    scored: list[tuple[int, dict[str, Any]]] = []
    for row in key_texts or []:
        kind, conf, _ = classify_learning_signal(
            app_name=str(row.get("app_name") or ""),
            window_name=str(row.get("window_name") or ""),
            browser_url=str(row.get("browser_url") or ""),
            text=str(row.get("text") or ""),
        )
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
) -> str:
    """Deterministic context block for the user-learning LLM prompt."""
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

    lines.append(f"Total learning dwell (approx): {total_min:.1f} min")
    lines.append(f"Audio transcript lines available: {len(audio_bits)}")
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

    # Lecture audio first — primary source for 讲解重点 / 理解要点.
    lines.append("")
    if audio_bits:
        lines.append(
            "### Audio transcripts (lecture) — PRIMARY source for 讲解重点 / 理解要点"
        )
        lines.append(
            "Summarize what the speaker taught from these lines. Quote key phrases. "
            "Cite as 录音. Do not invent content absent here."
        )
        lines.extend(f"- {a}" for a in audio_bits[:40])
    else:
        lines.append("### Audio transcripts (lecture)")
        lines.append(
            "NO_AUDIO_TRANSCRIPT: No usable lecture audio in this range. "
            "讲解重点/理解要点 must rely on 课件OCR only, or state material is insufficient."
        )

    ocr_lines = courseware_ocr_lines or []
    if ocr_lines:
        lines.append("")
        lines.append(
            "### Courseware OCR / slides — secondary source for 讲解重点 (cite as 课件OCR)"
        )
        lines.extend(ocr_lines[:35])

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
