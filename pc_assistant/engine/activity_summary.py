"""Activity summary bundle for LLM agents (/activity-summary)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from ..db import DatabaseManager
from .day_recap_context import (
    count_line_frequencies,
    dedupe_timeline,
    extract_valuable_lines,
    format_ts_local,
    is_low_value_text,
    normalize_text_key,
    reset_line_freq,
)

_GAP_MAX_SEC = 300
_KEY_TEXT_CAP = 30
_TIMELINE_CAP = 45


def _sql_escape(value: str) -> str:
    return value.replace("'", "''")


def _sql_like_escape(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("'", "''")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def _truncate(text: str, max_chars: int) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 3] + "..."


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def _default_range(
    start_time: str | None,
    end_time: str | None,
    *,
    default_hours: float = 16.0,
) -> tuple[str, str]:
    end = end_time or _now_iso()
    if start_time:
        return start_time, end
    end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
    start_dt = end_dt - timedelta(hours=default_hours)
    return start_dt.isoformat(), end


def build_activity_summary(
    db: DatabaseManager,
    *,
    start_time: str | None = None,
    end_time: str | None = None,
    app_name: str | None = None,
    q: str | None = None,
    include_recording: bool = True,
    include_memories: bool = True,
    include_snippets: bool = True,
    include_guidance: bool = True,
    max_snippets: int = 8,
    max_snippet_chars: int = 500,
    max_memories: int = 5,
) -> dict[str, Any]:
    """Build `/activity-summary` JSON response."""
    start, end = _default_range(start_time, end_time)
    app_filter = f" AND app_name = '{_sql_escape(app_name)}'" if app_name else ""
    app_filter_f = f" AND f.app_name = '{_sql_escape(app_name)}'" if app_name else ""
    query_text = (q or "").strip() or None

    with db._lock:  # noqa: SLF001
        conn = db._conn  # noqa: SLF001
        core = _collect_summary_core(conn, start, end, app_filter, app_filter_f)
        recording = (
            _load_recording_status(conn, start, end, app_filter) if include_recording else None
        )
        memories = (
            _load_memories(conn, query_text, max_memories) if include_memories else None
        )
        snippets = (
            _load_snippets(
                conn, core["key_texts"], start, end, query_text,
                max_snippets=min(max(1, max_snippets), 12),
                max_snippet_chars=max(160, min(max_snippet_chars, 1200)),
            )
            if include_snippets
            else None
        )

    snippets_list = snippets or []
    memories_list = memories or []
    data_status = _compute_data_status(core, recording, snippets_list)
    query_status = _compute_query_status(query_text, memories_list, snippets_list)
    guidance = (
        _build_guidance(data_status, query_status, query_text, app_name, recording)
        if include_guidance
        else None
    )

    return {
        "time_range": {"start": start, "end": end},
        "apps": core["apps"],
        "windows": core["windows"],
        "key_texts": core["key_texts"],
        "timeline": core["timeline"],
        "edited_files": core["edited_files"],
        "audio_summary": core["audio_summary"],
        "total_frames": core["total_frames"],
        "data_status": data_status,
        "query_status": query_status,
        "recording": recording,
        "memories": memories,
        "snippets": snippets,
        "guidance": guidance,
    }


def _collect_summary_core(
    conn: Any,
    start: str,
    end: str,
    app_filter: str,
    app_filter_f: str,
) -> dict[str, Any]:
    apps_query = f"""
        SELECT app_name,
               COUNT(*) AS frame_count,
               ROUND(SUM(CASE WHEN gap_sec < {_GAP_MAX_SEC} THEN gap_sec ELSE 0 END) / 60.0, 1) AS minutes,
               MIN(ts) AS first_seen,
               MAX(ts) AS last_seen
          FROM (
            SELECT app_name, timestamp AS ts,
                   (JULIANDAY(LEAD(timestamp) OVER (PARTITION BY app_name ORDER BY timestamp))
                    - JULIANDAY(timestamp)) * 86400 AS gap_sec
              FROM frames
             WHERE timestamp BETWEEN ? AND ?{app_filter}
               AND app_name IS NOT NULL AND app_name != ''
          ) gaps
         GROUP BY app_name
         ORDER BY minutes DESC
         LIMIT 20
    """
    windows_query = f"""
        SELECT app_name, window_name,
               COALESCE(browser_url, '') AS browser_url,
               COUNT(*) AS frame_count,
               ROUND(SUM(CASE WHEN gap_sec < {_GAP_MAX_SEC} THEN gap_sec ELSE 0 END) / 60.0, 1) AS minutes
          FROM (
            SELECT app_name,
                   COALESCE(window_name, '') AS window_name,
                   browser_url,
                   (JULIANDAY(LEAD(timestamp) OVER (PARTITION BY app_name, window_name ORDER BY timestamp))
                    - JULIANDAY(timestamp)) * 86400 AS gap_sec
              FROM frames
             WHERE timestamp BETWEEN ? AND ?{app_filter}
               AND app_name IS NOT NULL AND app_name != ''
               AND window_name IS NOT NULL AND window_name != ''
          ) gaps
         GROUP BY app_name, window_name
         ORDER BY minutes DESC
         LIMIT 30
    """
    # Prefer accessibility text (structured, filterable) over OCR (flat, noisy).
    # Get top 3 per (app, window) to allow Python-side quality ranking.
    a11y_texts_query = f"""
        SELECT a.text AS text, f.app_name,
               COALESCE(f.window_name, '') AS window_name,
               f.timestamp,
               'a11y' AS source
          FROM frames f
          JOIN frame_accessibility a ON a.frame_id = f.id
         WHERE f.timestamp BETWEEN ? AND ?{app_filter_f}
           AND LENGTH(a.text) > 40
         ORDER BY f.timestamp DESC
         LIMIT 60
    """
    ocr_texts_query = f"""
        SELECT o.text AS text, f.app_name,
               COALESCE(f.window_name, '') AS window_name,
               f.timestamp,
               'ocr' AS source
          FROM frames f
          JOIN ocr_text o ON o.frame_id = f.id
         WHERE f.timestamp BETWEEN ? AND ?{app_filter_f}
           AND LENGTH(o.text) BETWEEN 30 AND 800
         ORDER BY f.timestamp DESC
         LIMIT 30
    """
    # Timeline: prefer a11y text (structured), fall back to OCR
    timeline_query = f"""
        SELECT f.timestamp, f.app_name,
               COALESCE(f.window_name, '') AS window_name,
               COALESCE(f.browser_url, '') AS browser_url,
               COALESCE(a.text, o.text, '') AS text
          FROM frames f
          LEFT JOIN frame_accessibility a ON a.frame_id = f.id
          LEFT JOIN ocr_text o ON o.frame_id = f.id
         WHERE f.timestamp BETWEEN ? AND ?{app_filter_f}
           AND LENGTH(COALESCE(a.text, o.text, '')) > 35
         ORDER BY f.timestamp ASC
         LIMIT 80
    """
    ui_texts_query = f"""
        SELECT json_extract(data_json, '$.content') AS text,
               app_name,
               COALESCE(window_title, '') AS window_name,
               timestamp
          FROM ui_events
         WHERE timestamp BETWEEN ? AND ?{app_filter}
           AND event_type IN ('text', 'clipboard')
           AND LENGTH(COALESCE(json_extract(data_json, '$.content'), '')) BETWEEN 15 AND 300
         ORDER BY timestamp DESC
         LIMIT 25
    """
    edited_files_query = f"""
        SELECT document_path AS path, COUNT(*) AS frame_count
          FROM frames
         WHERE timestamp BETWEEN ? AND ?{app_filter}
           AND document_path IS NOT NULL
           AND document_path != ''
         GROUP BY document_path
         ORDER BY frame_count DESC, document_path ASC
         LIMIT 50
    """
    audio_speakers_query = """
        SELECT COALESCE(s.name, 'Unknown') AS speaker_name, COUNT(*) AS segment_count
          FROM audio_transcriptions at
          LEFT JOIN speakers s ON at.speaker_id = s.id
         WHERE at.timestamp BETWEEN ? AND ?
         GROUP BY at.speaker_id
         ORDER BY 2 DESC
         LIMIT 10
    """
    audio_transcripts_query = """
        SELECT at.transcription,
               COALESCE(s.name, 'Unknown') AS speaker,
               at.device,
               at.timestamp
          FROM audio_transcriptions at
          LEFT JOIN speakers s ON at.speaker_id = s.id
         WHERE at.timestamp BETWEEN ? AND ?
           AND TRIM(at.transcription) != ''
           AND LENGTH(at.transcription) > 5
         ORDER BY LENGTH(at.transcription) DESC
         LIMIT 20
    """

    apps_rows = conn.execute(apps_query, (start, end)).fetchall()
    windows_rows = conn.execute(windows_query, (start, end)).fetchall()
    a11y_text_rows = conn.execute(a11y_texts_query, (start, end)).fetchall()
    ocr_text_rows = conn.execute(ocr_texts_query, (start, end)).fetchall()
    screen_text_rows = list(a11y_text_rows) + list(ocr_text_rows)
    timeline_rows = conn.execute(timeline_query, (start, end)).fetchall()

    # Build line-frequency map to filter lines that repeat across many frames
    # (e.g. bookmark bar items, extension names, static sidebar text)
    reset_line_freq()
    all_raw_texts = [
        (row["text"] or "") for row in list(screen_text_rows) + list(timeline_rows)
    ]
    count_line_frequencies(all_raw_texts)
    n_texts = max(len(all_raw_texts), 1)
    # Lines appearing in >30% of all frames are repetitive noise
    freq_threshold = max(3, n_texts // 3)
    ui_text_rows = conn.execute(ui_texts_query, (start, end)).fetchall()
    edited_rows = conn.execute(edited_files_query, (start, end)).fetchall()
    speaker_rows = conn.execute(audio_speakers_query, (start, end)).fetchall()
    transcript_rows = conn.execute(audio_transcripts_query, (start, end)).fetchall()

    apps: list[dict[str, Any]] = []
    total_frames = 0
    for row in apps_rows:
        fc = int(row["frame_count"] or 0)
        total_frames += fc
        apps.append({
            "name": row["app_name"] or "",
            "frame_count": fc,
            "minutes": float(row["minutes"] or 0),
            "first_seen": row["first_seen"] or "",
            "last_seen": row["last_seen"] or "",
        })

    windows: list[dict[str, Any]] = []
    for row in windows_rows:
        win = (row["window_name"] or "").strip()
        if len(win) < 3:
            continue
        windows.append({
            "app_name": row["app_name"] or "",
            "window_name": win,
            "browser_url": row["browser_url"] or "",
            "minutes": float(row["minutes"] or 0),
            "frame_count": int(row["frame_count"] or 0),
        })

    key_texts: list[dict[str, Any]] = []
    seen: set[str] = set()
    # Screen text: filter noise lines, then keep high-value entries
    cleaned_screen: list[tuple[str, Any]] = []
    for row in screen_text_rows:
        raw = (row["text"] or "").strip()
        cleaned = extract_valuable_lines(raw, freq_threshold=freq_threshold)
        if len(cleaned) >= 20:
            cleaned_screen.append((cleaned, row))
    # Sort by cleaned content length (information density after noise removal)
    cleaned_screen.sort(key=lambda x: len(x[0]), reverse=True)

    for cleaned, row in cleaned_screen:
        norm = normalize_text_key(cleaned)
        if norm in seen:
            continue
        seen.add(norm)
        key_texts.append({
            "text": _truncate(cleaned, 500),
            "app_name": row["app_name"] or "",
            "window_name": row["window_name"] or "",
            "timestamp": row["timestamp"] or "",
        })
        if len(key_texts) >= _KEY_TEXT_CAP:
            break

    # UI typed text (already short, no line-filtering needed)
    for row in ui_text_rows:
        if len(key_texts) >= _KEY_TEXT_CAP:
            break
        text = (row["text"] or "").strip()
        if len(text) < 15 or text.lower().startswith(("http", "cdn.")):
            continue
        if is_low_value_text(text):
            continue
        norm = normalize_text_key(text)
        if norm in seen:
            continue
        seen.add(norm)
        key_texts.append({
            "text": _truncate(text, 400),
            "app_name": row["app_name"] or "",
            "window_name": row["window_name"] or "",
            "timestamp": row["timestamp"] or "",
        })

    raw_timeline = []
    for row in timeline_rows:
        raw = (row["text"] or "").strip()
        if not raw:
            continue
        cleaned = extract_valuable_lines(raw, freq_threshold=freq_threshold)
        if len(cleaned) < 20:
            continue
        raw_timeline.append({
            "timestamp": row["timestamp"] or "",
            "app_name": row["app_name"] or "",
            "window_name": row["window_name"] or "",
            "browser_url": row["browser_url"] or "",
            "text": _truncate(cleaned, 500),
        })
    timeline = dedupe_timeline(raw_timeline, max_items=_TIMELINE_CAP)

    edited_files = [
        {"path": row["path"] or "", "frame_count": int(row["frame_count"] or 0)}
        for row in edited_rows
        if (row["path"] or "").strip()
    ]

    speakers = [
        {"name": row["speaker_name"] or "Unknown", "segment_count": int(row["segment_count"] or 0)}
        for row in speaker_rows
    ]
    total_segments = sum(s["segment_count"] for s in speakers)
    top_transcriptions = [
        {
            "transcription": _truncate(row["transcription"] or "", 500),
            "speaker": row["speaker"] or "Unknown",
            "device": row["device"] or "",
            "timestamp": row["timestamp"] or "",
        }
        for row in transcript_rows
    ]

    return {
        "apps": apps,
        "windows": windows,
        "key_texts": key_texts,
        "timeline": timeline,
        "edited_files": edited_files,
        "audio_summary": {
            "segment_count": total_segments,
            "speakers": speakers,
            "top_transcriptions": top_transcriptions,
        },
        "total_frames": total_frames,
    }


def _load_recording_status(
    conn: Any,
    start: str,
    end: str,
    app_filter: str,
) -> dict[str, Any]:
    now = _now_iso()
    row = conn.execute(
        f"""
        SELECT
          (SELECT MAX(timestamp) FROM frames) AS last_frame_at,
          (SELECT MAX(timestamp) FROM audio_transcriptions) AS last_audio_at,
          (SELECT COUNT(*) FROM frames WHERE timestamp BETWEEN ? AND ?{app_filter}) AS frames_in_range,
          (SELECT COUNT(*) FROM audio_transcriptions WHERE timestamp BETWEEN ? AND ?) AS audio_segments_in_range,
          (SELECT ROUND((JULIANDAY(?) - JULIANDAY(MAX(timestamp))) * 86400) FROM frames) AS seconds_since_last_frame,
          (SELECT ROUND((JULIANDAY(?) - JULIANDAY(MAX(timestamp))) * 86400) FROM audio_transcriptions) AS seconds_since_last_audio
        """,
        (start, end, start, end, now, now),
    ).fetchone()
    if not row:
        return {
            "last_frame_at": None,
            "last_audio_at": None,
            "frames_in_range": 0,
            "audio_segments_in_range": 0,
            "recent_capture": False,
        }
    frame_age = row["seconds_since_last_frame"]
    audio_age = row["seconds_since_last_audio"]
    recent = (
        frame_age is not None and 0 <= int(frame_age) <= 600
    ) or (
        audio_age is not None and 0 <= int(audio_age) <= 600
    )
    return {
        "last_frame_at": row["last_frame_at"],
        "last_audio_at": row["last_audio_at"],
        "frames_in_range": int(row["frames_in_range"] or 0),
        "audio_segments_in_range": int(row["audio_segments_in_range"] or 0),
        "recent_capture": recent,
    }


def _load_memories(conn: Any, q: str | None, limit: int) -> list[dict[str, Any]]:
    lim = max(1, min(limit, 20))
    if q:
        rows = conn.execute(
            "SELECT id, content, created_at FROM memories WHERE content LIKE ? ESCAPE '\\' "
            "ORDER BY id DESC LIMIT ?",
            (f"%{_sql_like_escape(q)}%", lim),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT id, content, created_at FROM memories ORDER BY id DESC LIMIT ?",
            (lim,),
        ).fetchall()
    return [
        {
            "id": int(row["id"]),
            "content": _truncate(row["content"] or "", 500),
            "source": "local",
            "tags": [],
            "importance": 0.0,
            "created_at": row["created_at"] or "",
        }
        for row in rows
    ]


def _load_snippets(
    conn: Any,
    key_texts: list[dict[str, Any]],
    start: str,
    end: str,
    q: str | None,
    *,
    max_snippets: int,
    max_snippet_chars: int,
) -> list[dict[str, Any]]:
    screen_limit = max(1, (max_snippets + 1) // 2)
    audio_limit = max(1, max_snippets - screen_limit)
    q_lower = q.lower() if q else None
    snippets: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _push(snippet: dict[str, Any]) -> None:
        norm = (snippet.get("text") or "").lower().strip()
        if len(norm) < 20 or norm in seen:
            return
        seen.add(norm)
        snippets.append(snippet)

    for kt in key_texts:
        text = (kt.get("text") or "").strip()
        if len(text) < 20 or is_low_value_text(text):
            continue
        if q_lower and q_lower not in text.lower():
            continue
        # key_texts are already line-filtered by extract_valuable_lines
        _push({
            "source": "screen",
            "text": _truncate(text, max_snippet_chars),
            "app_name": kt.get("app_name") or None,
            "window_name": kt.get("window_name") or None,
            "speaker": None,
            "timestamp": kt.get("timestamp") or "",
        })
        if len([s for s in snippets if s["source"] == "screen"]) >= screen_limit:
            break

    audio_filter = ""
    audio_args: list[Any] = [start, end]
    if q:
        audio_filter = " AND at.transcription LIKE ? ESCAPE '\\'"
        audio_args.append(f"%{_sql_like_escape(q)}%")
    audio_rows = conn.execute(
        f"""
        SELECT at.transcription, COALESCE(s.name, 'Unknown') AS speaker, at.timestamp
          FROM audio_transcriptions at
          LEFT JOIN speakers s ON at.speaker_id = s.id
         WHERE at.timestamp BETWEEN ? AND ?{audio_filter}
           AND TRIM(at.transcription) != ''
           AND LENGTH(at.transcription) > 5
         ORDER BY at.timestamp DESC
         LIMIT ?
        """,
        (*audio_args, audio_limit),
    ).fetchall()
    for row in audio_rows:
        _push({
            "source": "audio",
            "text": _truncate(row["transcription"] or "", max_snippet_chars),
            "app_name": None,
            "window_name": None,
            "speaker": row["speaker"] or "Unknown",
            "timestamp": row["timestamp"] or "",
        })

    snippets.sort(key=lambda s: s.get("timestamp") or "", reverse=True)
    return snippets[:max_snippets]


def _compute_data_status(
    core: dict[str, Any],
    recording: dict[str, Any] | None,
    snippets: list[dict[str, Any]],
) -> str:
    audio_count = int(core["audio_summary"].get("segment_count") or 0)
    if core["total_frames"] > 0 or audio_count > 0 or snippets:
        return "ok"
    if not recording:
        return "unknown"
    if not recording.get("last_frame_at") and not recording.get("last_audio_at"):
        return "not_recording"
    if recording.get("recent_capture"):
        return "empty_but_recording"
    return "no_capture_in_range"


def _compute_query_status(
    q: str | None,
    memories: list[dict[str, Any]],
    snippets: list[dict[str, Any]],
) -> str:
    if not q:
        return "not_requested"
    if not memories and not snippets:
        return "no_query_matches"
    return "matched"


def _build_guidance(
    data_status: str,
    query_status: str,
    q: str | None,
    app_name: str | None,
    recording: dict[str, Any] | None,
) -> dict[str, Any]:
    searched = ["/activity-summary"]
    next_best: str | None = None
    if query_status == "no_query_matches":
        next_best = (
            "no memories or snippets matched q. retry /activity-summary without q, "
            "then use /search only for verbatim matches."
        )
    elif data_status == "ok":
        next_best = None
    elif data_status == "empty_but_recording":
        next_best = (
            "broaden the time range, remove q/app filters, then retry /activity-summary "
            "before raw /search."
        )
    elif data_status == "no_capture_in_range" and recording:
        last_f = recording.get("last_frame_at") or "never"
        last_a = recording.get("last_audio_at") or "never"
        next_best = (
            f"no captures in this range. last frame: {last_f}; last audio: {last_a}. "
            "try a range around the latest timestamp."
        )
    elif data_status == "not_recording":
        next_best = (
            "no local captures exist yet. start pc-assistant recording before concluding "
            "the user was inactive."
        )
    elif q or app_name:
        next_best = "retry without q/app_name filters before saying no data was found."
    return {"searched_endpoints": searched, "next_best_query": next_best}


def format_summary_for_agent(summary: dict[str, Any], *, include_date: bool = False) -> str:
    """Turn API JSON into markdown sections for the LLM agent."""
    from pc_assistant.engine.day_recap_context import format_ts_recap

    fmt_ts = lambda ts: format_ts_recap(ts, include_date=include_date)  # noqa: E731
    status = summary.get("data_status", "unknown")
    total_frames = summary.get("total_frames", 0)
    audio = summary.get("audio_summary") or {}
    audio_count = int(audio.get("segment_count") or 0)

    if status == "not_recording" and total_frames == 0 and audio_count == 0:
        return "(No activity data was captured in this time range. Start pc-assistant recording.)"

    if status == "no_capture_in_range" and total_frames == 0 and audio_count == 0:
        rec = summary.get("recording") or {}
        return (
            "(No captures in this time range. "
            f"Last frame: {rec.get('last_frame_at')}; last audio: {rec.get('last_audio_at')}.)"
        )

    sections: list[str] = [f"### data_status: {status} (total_frames={total_frames})"]

    guidance = summary.get("guidance")
    if guidance and guidance.get("next_best_query"):
        sections.append(f"### Agent guidance\n{guidance['next_best_query']}")

    apps = summary.get("apps") or []
    if apps:
        lines = []
        for a in apps:
            lines.append(
                f"  [{a.get('first_seen', '')} ~ {a.get('last_seen', '')}] "
                f"{a.get('name', '?')}: {a.get('minutes', 0)} min, {a.get('frame_count', 0)} captures"
            )
        sections.append("### Apps (active time)\n" + "\n".join(lines))

    timeline = summary.get("timeline") or []
    if timeline:
        lines = []
        for e in timeline:
            ts = fmt_ts(e.get("timestamp", ""))
            text = (e.get("text") or "").replace("\n", " ").strip()
            url = e.get("browser_url") or ""
            extra = f" | {url}" if url else ""
            lines.append(
                f"  [{ts}] {e.get('app_name', '')} / {e.get('window_name', '')}{extra}\n"
                f"    {text[:480]}"
            )
        sections.append(
            f"### Chronological timeline ({len(timeline)} samples, use for narrative arc)\n"
            + "\n".join(lines)
        )

    windows = summary.get("windows") or []
    if windows:
        lines = []
        for w in windows[:25]:
            url = w.get("browser_url") or ""
            extra = f" url={url}" if url else ""
            lines.append(
                f"  {w.get('app_name', '?')} / {w.get('window_name', '')}: "
                f"{w.get('minutes', 0)} min{extra}"
            )
        sections.append("### Windows / tabs\n" + "\n".join(lines))

    edited = summary.get("edited_files") or []
    if edited:
        lines = [f"  {e.get('path', '')} ({e.get('frame_count', 0)} captures)" for e in edited[:15]]
        sections.append("### Edited files\n" + "\n".join(lines))

    key_texts = summary.get("key_texts") or []
    if key_texts:
        lines = []
        for kt in key_texts[:30]:
            ts = fmt_ts(kt.get("timestamp", ""))
            text = (kt.get("text") or "").replace("\n", " ").strip()
            lines.append(
                f"  [{ts}] {kt.get('app_name', '')} / {kt.get('window_name', '')}:\n"
                f"    {text}"
            )
        sections.append("### Key texts (screen + typed input)\n" + "\n".join(lines))

    snippets = summary.get("snippets") or []
    if snippets:
        lines = []
        for sn in snippets:
            src = sn.get("source", "?")
            text = (sn.get("text") or "").replace("\n", " ").strip()
            ts = fmt_ts(sn.get("timestamp", ""))
            if src == "audio":
                lines.append(
                    f"  [{ts}] AUDIO ({sn.get('speaker', '')}): {text}"
                )
            else:
                lines.append(
                    f"  [{ts}] SCREEN "
                    f"{sn.get('app_name', '')} / {sn.get('window_name', '')}: {text}"
                )
        sections.append("### Snippets\n" + "\n".join(lines))

    top_tx = audio.get("top_transcriptions") or []
    if top_tx:
        lines = []
        for t in top_tx[:12]:
            ts = fmt_ts(t.get("timestamp", ""))
            lines.append(
                f"  [{ts}] ({t.get('speaker', '')}, {t.get('device', '')}): "
                f"{(t.get('transcription') or '')[:450]}"
            )
        sections.append(f"### Audio ({audio_count} segments)\n" + "\n".join(lines))

    memories = summary.get("memories") or []
    if memories:
        lines = [f"  [{m.get('created_at', '')}] {m.get('content', '')}" for m in memories]
        sections.append("### Memories\n" + "\n".join(lines))

    return "\n\n".join(sections)
