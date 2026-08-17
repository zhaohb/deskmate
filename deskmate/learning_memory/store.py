"""SQLite access for learning_concepts / lecture_items / reviews / topics."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import paths
from ..logger import get
from .bkt import (
    PASSIVE_MIN_DAYS,
    BktParams,
    apply_mastery_decay,
    bkt_update,
    days_between,
    infer_skill_type,
    mastery_tier,
    mastery_urgency,
    passive_exposure_update,
    quality_to_correct,
    seed_mastery_from_evidence,
)
from .extract import ConceptHit, LectureItem, normalize_name
from .graph import ConceptEdge
from .sm2 import Sm2State, due_at_after, seed_quality, sm2_update, urgency_score
from .topics_llm import TopicHit

logger = get("learning_memory.store")


def _dict_factory(cursor: sqlite3.Cursor, row: tuple) -> dict[str, Any]:
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}


def _now_iso() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def _parse_iso(value: str) -> datetime | None:
    """Lenient ISO parse that always yields an aware datetime (local fallback)."""
    text = (value or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.astimezone()


def problem_dedup_key(text: str, *, seen_at: str = "") -> str:
    """Stable per-day key so the live detector and the recap agree on a problem.

    Keyed on content + calendar day rather than session id: the same on-screen
    error surfaces once from the frame detector and again from the recap slice,
    under two different session ids.
    """
    day = (seen_at or _now_iso())[:10]
    return f"problem:{day}:{normalize_name(text)[:80]}"


class LearningStore:
    """Own connection (habits/fusion pattern) to the main DeskMate DB."""

    def __init__(self, db_file: Path | None = None) -> None:
        self.path = Path(db_file) if db_file else paths.db_path()
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = _dict_factory
        with self._lock:
            self._conn.execute("PRAGMA busy_timeout = 5000")
            self._conn.execute("PRAGMA foreign_keys = ON")
            # Ensure tables exist even if daemon hasn't restarted since schema bump.
            from ..db.schema import SCHEMA  # noqa: PLC0415

            self._conn.executescript(SCHEMA)
            self._ensure_review_bkt_columns()

    def _ensure_review_bkt_columns(self) -> None:
        """Additive columns for DBs created before BKT fields existed."""
        cols = {
            r["name"]
            for r in self._conn.execute("PRAGMA table_info(learning_reviews)").fetchall()
        }
        if not cols:
            return
        alters = [
            ("p_mastery", "REAL NOT NULL DEFAULT 0.1"),
            ("bkt_attempts", "INTEGER NOT NULL DEFAULT 0"),
            ("last_bkt_at", "TEXT"),
            ("skill_type", "TEXT NOT NULL DEFAULT ''"),
            ("mastery_tier", "TEXT NOT NULL DEFAULT 'exposure'"),
        ]
        for name, decl in alters:
            if name not in cols:
                self._conn.execute(f"ALTER TABLE learning_reviews ADD COLUMN {name} {decl}")

    def upsert_concepts(
        self,
        concepts: list[ConceptHit],
        *,
        seen_at: str | None = None,
    ) -> dict[str, int]:
        """Insert/update concepts; return name_norm → concept_id."""
        seen_at = seen_at or _now_iso()
        ids: dict[str, int] = {}
        with self._lock:
            for c in concepts:
                key = normalize_name(c.name)
                if not key:
                    continue
                row = self._conn.execute(
                    "SELECT id, hit_count, evidence_json FROM learning_concepts WHERE name_norm = ?",
                    (key,),
                ).fetchone()
                ev = list(c.evidence or [])
                if row:
                    prev = []
                    try:
                        prev = json.loads(row["evidence_json"] or "[]")
                    except json.JSONDecodeError:
                        prev = []
                    merged = []
                    for e in prev + ev:
                        if e and e not in merged:
                            merged.append(e)
                    merged = merged[:6]
                    self._conn.execute(
                        """UPDATE learning_concepts
                              SET name = ?, topic = ?, last_seen = ?, hit_count = hit_count + ?,
                                  evidence_json = ?, updated_at = ?
                            WHERE id = ?""",
                        (
                            c.name,
                            c.topic,
                            seen_at,
                            max(1, int(c.count)),
                            json.dumps(merged, ensure_ascii=False),
                            seen_at,
                            row["id"],
                        ),
                    )
                    ids[key] = int(row["id"])
                else:
                    cur = self._conn.execute(
                        """INSERT INTO learning_concepts
                           (name, name_norm, topic, first_seen, last_seen, hit_count, evidence_json, updated_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            c.name,
                            key,
                            c.topic,
                            seen_at,
                            seen_at,
                            max(1, int(c.count)),
                            json.dumps(ev[:6], ensure_ascii=False),
                            seen_at,
                        ),
                    )
                    ids[key] = int(cur.lastrowid)
        return ids

    def insert_lecture_items(
        self,
        items: list[LectureItem],
        *,
        concept_ids: dict[str, int],
        seen_at: str | None = None,
        session_ref: str = "",
    ) -> int:
        seen_at = seen_at or _now_iso()
        n = 0
        with self._lock:
            for it in items:
                cid = concept_ids.get(normalize_name(it.subject))
                self._conn.execute(
                    """INSERT INTO learning_lecture_items
                       (kind, concept_id, subject, content, ordinal, source, evidence, session_ref, seen_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        it.kind,
                        cid,
                        it.subject,
                        it.content,
                        int(it.ordinal),
                        it.source,
                        (it.evidence or "")[:400],
                        session_ref or it.session_ref,
                        seen_at,
                    ),
                )
                n += 1
        return n

    def upsert_topics(
        self,
        topics: list[TopicHit],
        *,
        seen_at: str | None = None,
    ) -> int:
        """Replace-ish upsert of LLM topics for this run (by name_norm)."""
        if not topics:
            return 0
        seen_at = seen_at or _now_iso()
        n = 0
        with self._lock:
            for t in topics:
                key = normalize_name(t.name)
                if not key:
                    continue
                subs = [
                    {"name": s.name, "confidence": s.confidence}
                    for s in (t.subtopics or [])
                ]
                row = self._conn.execute(
                    "SELECT id FROM learning_topics WHERE name_norm = ?",
                    (key,),
                ).fetchone()
                if row:
                    self._conn.execute(
                        """UPDATE learning_topics
                              SET name = ?, confidence = ?, subtopics_json = ?,
                                  evidence_json = ?, seen_at = ?, updated_at = ?, source = 'llm'
                            WHERE id = ?""",
                        (
                            t.name,
                            float(t.confidence),
                            json.dumps(subs, ensure_ascii=False),
                            json.dumps(list(t.evidence or [])[:4], ensure_ascii=False),
                            seen_at,
                            seen_at,
                            row["id"],
                        ),
                    )
                else:
                    self._conn.execute(
                        """INSERT INTO learning_topics
                           (name, name_norm, confidence, subtopics_json, evidence_json, source, seen_at, updated_at)
                           VALUES (?, ?, ?, ?, ?, 'llm', ?, ?)""",
                        (
                            t.name,
                            key,
                            float(t.confidence),
                            json.dumps(subs, ensure_ascii=False),
                            json.dumps(list(t.evidence or [])[:4], ensure_ascii=False),
                            seen_at,
                            seen_at,
                        ),
                    )
                n += 1
        return n

    def seed_or_update_reviews(
        self,
        concepts: list[ConceptHit],
        concept_ids: dict[str, int],
        *,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Seed SM-2 cards for new concepts, refresh BKT only for existing ones.

        Passive evidence must never reschedule a card: an unattended recap that
        re-observed the same slides would otherwise walk ``repetitions`` up and
        push ``due_at`` weeks out even though the user never reviewed anything.
        Only :meth:`grade_review` may move ease_factor / interval / due_at.
        """
        now = now or datetime.now().astimezone()
        now_iso = now.replace(microsecond=0).isoformat()
        with self._lock:
            for c in concepts:
                key = normalize_name(c.name)
                cid = concept_ids.get(key)
                if not cid:
                    continue
                skill = infer_skill_type(
                    has_code=c.has_code, has_definition=c.has_definition
                )
                row = self._conn.execute(
                    "SELECT * FROM learning_reviews WHERE concept_id = ?",
                    (cid,),
                ).fetchone()
                if row:
                    self._refresh_review_mastery(row, c, skill=skill, now=now, now_iso=now_iso)
                    continue
                q = seed_quality(
                    hit_count=c.count,
                    has_definition=c.has_definition,
                    has_problem=c.has_problem,
                    has_code=c.has_code,
                )
                p_new = seed_mastery_from_evidence(
                    hit_count=c.count,
                    has_definition=c.has_definition,
                    has_problem=c.has_problem,
                    has_code=c.has_code,
                    skill_type=skill or None,
                )
                new_state = sm2_update(Sm2State(), q)
                due = due_at_after(new_state, now=now).replace(microsecond=0).isoformat()
                self._conn.execute(
                    """INSERT INTO learning_reviews
                       (concept_id, ease_factor, interval_days, repetitions, due_at,
                        last_quality, last_reviewed, p_mastery, bkt_attempts,
                        last_bkt_at, skill_type, mastery_tier, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        cid,
                        new_state.ease_factor,
                        new_state.interval_days,
                        new_state.repetitions,
                        due,
                        q,
                        now_iso,
                        p_new,
                        1,
                        now_iso,
                        skill or "",
                        mastery_tier(p_new),
                        now_iso,
                    ),
                )
        return self.list_due_reviews(limit=20, now=now)

    def _refresh_review_mastery(
        self,
        row: dict[str, Any],
        concept: ConceptHit,
        *,
        skill: str,
        now: datetime,
        now_iso: str,
    ) -> None:
        """Decay + one soft passive observation on an existing card.

        The SM-2 schedule is deliberately left untouched. At most one passive
        observation lands per study day so re-running a recap is idempotent.
        """
        last_bkt = row.get("last_bkt_at") or row.get("last_reviewed")
        days = 0.0
        if last_bkt:
            try:
                prev_dt = datetime.fromisoformat(str(last_bkt).replace("Z", "+00:00"))
                days = days_between(prev_dt, now)
            except ValueError:
                days = 0.0
            if days < PASSIVE_MIN_DAYS:
                return
        prev_p = float(row.get("p_mastery") if row.get("p_mastery") is not None else 0.1)
        skill_keep = row.get("skill_type") or skill or ""
        p_new = passive_exposure_update(
            apply_mastery_decay(prev_p, days),
            has_definition=concept.has_definition,
            has_problem=concept.has_problem,
            has_code=concept.has_code,
            hit_count=concept.count,
            skill_type=skill_keep or None,
        )
        self._conn.execute(
            """UPDATE learning_reviews
                  SET p_mastery = ?, bkt_attempts = ?, last_bkt_at = ?,
                      skill_type = ?, mastery_tier = ?, updated_at = ?
                WHERE concept_id = ?""",
            (
                p_new,
                int(row.get("bkt_attempts") or 0) + 1,
                now_iso,
                skill_keep,
                mastery_tier(p_new),
                now_iso,
                int(row["concept_id"]),
            ),
        )

    def grade_review(
        self,
        concept_id: int,
        *,
        quality: int | None = None,
        correct: bool | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any] | None:
        """Interactive grade → SM-2 + BKT update. Returns updated due row or None."""
        now = now or datetime.now().astimezone()
        now_iso = now.replace(microsecond=0).isoformat()
        if quality is None and correct is None:
            return None
        if quality is None:
            quality = 4 if correct else 1
        quality = max(0, min(5, int(quality)))
        is_ok = quality_to_correct(quality) if correct is None else bool(correct)

        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM learning_reviews WHERE concept_id = ?",
                (concept_id,),
            ).fetchone()
            if not row:
                return None
            state = Sm2State(
                ease_factor=float(row["ease_factor"] or 2.5),
                interval_days=float(row["interval_days"] or 0),
                repetitions=int(row["repetitions"] or 0),
            )
            new_state = sm2_update(state, quality)
            prev_p = float(row.get("p_mastery") if row.get("p_mastery") is not None else 0.1)
            last_bkt = row.get("last_bkt_at") or row.get("last_reviewed")
            days = 0.0
            if last_bkt:
                try:
                    prev_dt = datetime.fromisoformat(str(last_bkt).replace("Z", "+00:00"))
                    days = days_between(prev_dt, now)
                except ValueError:
                    days = 0.0
            decayed = apply_mastery_decay(prev_p, days)
            params = BktParams.for_skill(row.get("skill_type") or None)
            p_new = bkt_update(decayed, is_ok, params)
            tier = mastery_tier(p_new)
            due = due_at_after(new_state, now=now).replace(microsecond=0).isoformat()
            self._conn.execute(
                """UPDATE learning_reviews
                      SET ease_factor = ?, interval_days = ?, repetitions = ?,
                          due_at = ?, last_quality = ?, last_reviewed = ?,
                          p_mastery = ?, bkt_attempts = bkt_attempts + 1,
                          last_bkt_at = ?, mastery_tier = ?, updated_at = ?
                    WHERE concept_id = ?""",
                (
                    new_state.ease_factor,
                    new_state.interval_days,
                    new_state.repetitions,
                    due,
                    quality,
                    now_iso,
                    p_new,
                    now_iso,
                    tier,
                    now_iso,
                    concept_id,
                ),
            )
        rows = self.list_due_reviews(limit=100, now=now)
        return next((r for r in rows if int(r.get("concept_id") or 0) == concept_id), None)

    def list_due_reviews(self, *, limit: int = 20, now: datetime | None = None) -> list[dict[str, Any]]:
        now = now or datetime.now().astimezone()
        with self._lock:
            rows = self._conn.execute(
                """SELECT r.*, c.name, c.topic, c.hit_count, c.evidence_json
                     FROM learning_reviews r
                     JOIN learning_concepts c ON c.id = r.concept_id
                    ORDER BY r.due_at ASC
                    LIMIT ?""",
                (max(1, limit * 3),),
            ).fetchall()
        scored: list[tuple[float, dict[str, Any]]] = []
        for r in rows:
            try:
                due = datetime.fromisoformat(str(r["due_at"]).replace("Z", "+00:00"))
            except ValueError:
                due = now
            if due.tzinfo is None:
                due = due.replace(tzinfo=now.tzinfo)
            u_sm2 = urgency_score(due_at=due, now=now, ease=float(r["ease_factor"] or 2.5))
            prev_p = float(r.get("p_mastery") if r.get("p_mastery") is not None else 0.1)
            last_bkt = r.get("last_bkt_at") or r.get("last_reviewed")
            days = 0.0
            if last_bkt:
                try:
                    prev_dt = datetime.fromisoformat(str(last_bkt).replace("Z", "+00:00"))
                    days = days_between(prev_dt, now)
                except ValueError:
                    days = 0.0
            decayed = apply_mastery_decay(prev_p, days)
            tier = mastery_tier(decayed)
            u = u_sm2 + mastery_urgency(decayed)
            item = dict(r)
            item["p_mastery"] = round(prev_p, 4)
            item["decayed_mastery"] = round(decayed, 4)
            item["mastery_tier"] = tier
            item["days_since_bkt"] = round(days, 2)
            item["urgency"] = round(u, 2)
            item["overdue"] = due <= now
            item["weak_mastery"] = decayed < 0.75
            scored.append((u, item))
        scored.sort(key=lambda x: -x[0])
        return [it for _, it in scored[:limit]]

    def list_concepts(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            return self._conn.execute(
                """SELECT * FROM learning_concepts
                    ORDER BY last_seen DESC, hit_count DESC
                    LIMIT ?""",
                (limit,),
            ).fetchall()

    def list_topics(self, *, limit: int = 40) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM learning_topics
                    ORDER BY seen_at DESC, confidence DESC
                    LIMIT ?""",
                (limit,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            item = dict(r)
            try:
                item["subtopics"] = json.loads(r.get("subtopics_json") or "[]")
            except json.JSONDecodeError:
                item["subtopics"] = []
            try:
                item["evidence"] = json.loads(r.get("evidence_json") or "[]")
            except json.JSONDecodeError:
                item["evidence"] = []
            out.append(item)
        return out

    def recent_lecture_items(self, *, limit: int = 40) -> list[dict[str, Any]]:
        with self._lock:
            return self._conn.execute(
                """SELECT * FROM learning_lecture_items
                    ORDER BY id DESC
                    LIMIT ?""",
                (limit,),
            ).fetchall()

    # ── sessions / events / edges ───────────────────────────────────────────

    def upsert_slice_sessions(
        self,
        sessions: list[dict[str, Any]],
        *,
        status: str = "closed",
    ) -> list[int]:
        """Reconcile recap slices against existing sessions, inserting only gaps.

        The frame-level detector owns this table, so a slice that overlaps a row
        is merged into it instead of written again — slice ordinals (``[1]``,
        ``[2]``) are per-run and shift whenever the recap window moves. An open
        row is never closed here; only the detector ends a live session.
        """
        ids: list[int] = []
        now_iso = _now_iso()
        for s in sessions:
            started = str(s.get("started_at") or now_iso)
            ended = str(s.get("ended_at") or "")
            kind = str(s.get("kind") or "")
            match = self._find_overlapping_session(started, ended, kind=kind)
            if match is not None:
                self.touch_session(
                    match,
                    kind=kind,
                    title=str(s.get("title") or "")[:200],
                    apps=[str(a) for a in (s.get("apps") or [])],
                    urls=[str(u) for u in (s.get("urls") or [])],
                    queries=[str(q) for q in (s.get("queries") or [])],
                    topics=[str(t) for t in (s.get("topics") or [])],
                    concepts=[str(c) for c in (s.get("concepts") or [])],
                    confidence=float(s.get("confidence") or 0),
                    reason=str(s.get("reason") or "")[:200],
                    sample_text=str(s.get("sample_text") or "")[:400],
                )
                ids.append(match)
                continue
            with self._lock:
                cur = self._conn.execute(
                    """INSERT INTO learning_sessions
                       (kind, title, status, started_at, ended_at, duration_min,
                        apps_json, urls_json, queries_json, topics_json, concepts_json,
                        confidence, reason, sample_text, slice_ref, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        kind,
                        str(s.get("title") or "")[:200],
                        status,
                        started,
                        ended or None,
                        float(s.get("duration_min") or 0),
                        json.dumps(s.get("apps") or [], ensure_ascii=False),
                        json.dumps(s.get("urls") or [], ensure_ascii=False),
                        json.dumps(s.get("queries") or [], ensure_ascii=False),
                        json.dumps(s.get("topics") or [], ensure_ascii=False),
                        json.dumps(s.get("concepts") or [], ensure_ascii=False),
                        float(s.get("confidence") or 0),
                        str(s.get("reason") or "")[:200],
                        str(s.get("sample_text") or "")[:400],
                        f"[{s.get('id')}]" if s.get("id") is not None else "",
                        now_iso,
                    ),
                )
                ids.append(int(cur.lastrowid))
        return ids

    def _find_overlapping_session(
        self,
        started: str,
        ended: str,
        *,
        kind: str = "",
    ) -> int | None:
        """Id of the recorded session sharing the most time with this slice."""
        s0 = _parse_iso(started)
        if s0 is None:
            return None
        s1 = _parse_iso(ended) or s0
        if s1 < s0:
            s1 = s0
        with self._lock:
            rows = self._conn.execute(
                """SELECT id, kind, status, started_at, ended_at
                     FROM learning_sessions
                    ORDER BY started_at DESC LIMIT 200"""
            ).fetchall()
        best_id: int | None = None
        best_overlap = -1.0
        for r in rows:
            if kind and r.get("kind") and str(r["kind"]) != kind:
                continue
            r0 = _parse_iso(str(r.get("started_at") or ""))
            if r0 is None:
                continue
            fallback = datetime.now().astimezone() if r.get("status") == "open" else r0
            r1 = _parse_iso(str(r.get("ended_at") or "")) or fallback
            if r1 < r0:
                r1 = r0
            overlap = (min(s1, r1) - max(s0, r0)).total_seconds()
            if overlap <= 0 and not (r0 <= s0 and s1 <= r1):
                continue
            if overlap > best_overlap:
                best_overlap = overlap
                best_id = int(r["id"])
        return best_id

    def start_session(
        self,
        *,
        kind: str = "study_other",
        title: str = "",
        topics: list[str] | None = None,
        concepts: list[str] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Manual /start-study — closes any open session first."""
        now_iso = _now_iso()
        with self._lock:
            open_rows = self._conn.execute(
                "SELECT id, started_at FROM learning_sessions WHERE status = 'open'"
            ).fetchall()
            for r in open_rows:
                self._close_session_row(int(r["id"]), ended_at=now_iso, trigger_note="auto_close")
            cur = self._conn.execute(
                """INSERT INTO learning_sessions
                   (kind, title, status, started_at, topics_json, concepts_json, meta_json, updated_at)
                   VALUES (?, ?, 'open', ?, ?, ?, ?, ?)""",
                (
                    kind or "study_other",
                    (title or "Study session")[:200],
                    now_iso,
                    json.dumps(topics or [], ensure_ascii=False),
                    json.dumps(concepts or [], ensure_ascii=False),
                    json.dumps(meta or {}, ensure_ascii=False),
                    now_iso,
                ),
            )
            sid = int(cur.lastrowid)
        return self.get_session(sid) or {"id": sid, "status": "open", "started_at": now_iso}

    def end_session(
        self,
        session_id: int | None = None,
        *,
        ended_at: str | None = None,
    ) -> dict[str, Any] | None:
        """Manual /end-session — close one session (or latest open)."""
        ended_at = ended_at or _now_iso()
        with self._lock:
            if session_id is None:
                row = self._conn.execute(
                    """SELECT id FROM learning_sessions
                        WHERE status = 'open' ORDER BY started_at DESC LIMIT 1"""
                ).fetchone()
                if not row:
                    return None
                session_id = int(row["id"])
            self._close_session_row(session_id, ended_at=ended_at)
        return self.get_session(session_id)

    def _close_session_row(
        self,
        session_id: int,
        *,
        ended_at: str,
        trigger_note: str = "",
    ) -> None:
        row = self._conn.execute(
            "SELECT started_at, meta_json FROM learning_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if not row:
            return
        duration = 0.0
        try:
            start = datetime.fromisoformat(str(row["started_at"]).replace("Z", "+00:00"))
            end = datetime.fromisoformat(ended_at.replace("Z", "+00:00"))
            duration = max(0.0, (end - start).total_seconds() / 60.0)
        except ValueError:
            duration = 0.0
        meta: dict[str, Any] = {}
        try:
            meta = json.loads(row.get("meta_json") or "{}")
        except json.JSONDecodeError:
            meta = {}
        if trigger_note:
            meta["close_note"] = trigger_note
        self._conn.execute(
            """UPDATE learning_sessions
                  SET status = 'closed', ended_at = ?, duration_min = ?,
                      meta_json = ?, updated_at = ?
                WHERE id = ?""",
            (ended_at, round(duration, 1), json.dumps(meta, ensure_ascii=False), ended_at, session_id),
        )

    def touch_session(
        self,
        session_id: int,
        *,
        kind: str = "",
        title: str = "",
        apps: list[str] | None = None,
        urls: list[str] | None = None,
        queries: list[str] | None = None,
        topics: list[str] | None = None,
        concepts: list[str] | None = None,
        confidence: float | None = None,
        reason: str = "",
        sample_text: str = "",
    ) -> None:
        """Merge live observation into an open session (keep-alive path)."""
        now_iso = _now_iso()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM learning_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if not row:
                return

            def _merge(field: str, incoming: list[str] | None, *, cap: int = 12) -> str:
                try:
                    prev = json.loads(row.get(field) or "[]")
                except json.JSONDecodeError:
                    prev = []
                if not isinstance(prev, list):
                    prev = []
                out: list[str] = []
                for x in list(prev) + list(incoming or []):
                    s = str(x).strip()
                    if s and s not in out:
                        out.append(s)
                return json.dumps(out[:cap], ensure_ascii=False)

            apps_json = _merge("apps_json", apps, cap=8)
            urls_json = _merge("urls_json", urls, cap=8)
            queries_json = _merge("queries_json", queries, cap=8)
            topics_json = _merge("topics_json", topics, cap=8)
            concepts_json = _merge("concepts_json", concepts, cap=12)
            self._conn.execute(
                """UPDATE learning_sessions SET
                      kind = COALESCE(NULLIF(?, ''), kind),
                      title = COALESCE(NULLIF(?, ''), title),
                      apps_json = ?, urls_json = ?, queries_json = ?,
                      topics_json = ?, concepts_json = ?,
                      confidence = CASE WHEN ? IS NULL THEN confidence
                                        ELSE MAX(confidence, ?) END,
                      reason = COALESCE(NULLIF(?, ''), reason),
                      sample_text = CASE
                          WHEN length(?) > length(COALESCE(sample_text, '')) THEN ?
                          ELSE sample_text END,
                      updated_at = ?
                    WHERE id = ?""",
                (
                    kind or "",
                    title or "",
                    apps_json,
                    urls_json,
                    queries_json,
                    topics_json,
                    concepts_json,
                    confidence,
                    float(confidence or 0),
                    reason or "",
                    (sample_text or "")[:400],
                    (sample_text or "")[:400],
                    now_iso,
                    session_id,
                ),
            )

    def get_session(self, session_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM learning_sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return self._decode_session(row) if row else None

    def list_sessions(self, *, limit: int = 40, status: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if status:
                rows = self._conn.execute(
                    """SELECT * FROM learning_sessions WHERE status = ?
                        ORDER BY started_at DESC LIMIT ?""",
                    (status, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """SELECT * FROM learning_sessions
                        ORDER BY started_at DESC LIMIT ?""",
                    (limit,),
                ).fetchall()
        return [self._decode_session(r) for r in rows]

    def list_manual_sessions(
        self,
        start_iso: str,
        end_iso: str,
        *,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """User-declared sessions overlapping ``[start_iso, end_iso]``.

        Sessions the user started by hand are ground truth about what counts as
        studying, so a recap must honour their whole span rather than re-deriving
        one from heuristics — that is the difference between "I studied for an
        hour" and "the classifier recognised eleven scattered minutes of it".

        Overlap is computed in Python from parsed timestamps rather than by
        string comparison in SQL: session rows are few, and the stored formats
        vary enough ('T' vs ' ' separators, offsets present or not) that
        lexicographic bounds silently drop whole days.
        """
        lo, hi = _parse_iso(start_iso), _parse_iso(end_iso)
        if lo is None or hi is None:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM learning_sessions ORDER BY started_at DESC LIMIT ?",
                (max(1, limit),),
            ).fetchall()

        out: list[dict[str, Any]] = []
        for raw in rows:
            item = self._decode_session(raw)
            if str((item.get("meta") or {}).get("detection_source") or "") != "manual":
                continue
            began = _parse_iso(str(item.get("started_at") or ""))
            if began is None:
                continue
            # An open session runs up to "now"; a closed one to its end stamp.
            finished = _parse_iso(str(item.get("ended_at") or "")) or hi
            if began <= hi and finished >= lo:
                out.append(item)
        return out

    def _decode_session(self, row: dict[str, Any] | None) -> dict[str, Any]:
        if not row:
            return {}
        item = dict(row)
        for key, field in (
            ("apps", "apps_json"),
            ("urls", "urls_json"),
            ("queries", "queries_json"),
            ("topics", "topics_json"),
            ("concepts", "concepts_json"),
        ):
            try:
                item[key] = json.loads(row.get(field) or "[]")
            except json.JSONDecodeError:
                item[key] = []
        try:
            item["meta"] = json.loads(row.get("meta_json") or "{}")
        except json.JSONDecodeError:
            item["meta"] = {}
        return item

    def insert_event(
        self,
        *,
        kind: str,
        summary: str,
        session_id: int | None = None,
        concept_id: int | None = None,
        app_name: str = "",
        evidence: str = "",
        payload: dict[str, Any] | None = None,
        dedup_key: str = "",
        seen_at: str | None = None,
        status: str = "open",
    ) -> int | None:
        seen_at = seen_at or _now_iso()
        summary = (summary or "").strip()
        if not summary:
            return None
        with self._lock:
            if dedup_key:
                exists = self._conn.execute(
                    "SELECT id FROM learning_events WHERE dedup_key = ?",
                    (dedup_key,),
                ).fetchone()
                if exists:
                    return int(exists["id"])
            cur = self._conn.execute(
                """INSERT INTO learning_events
                   (kind, summary, status, session_id, concept_id, app_name,
                    evidence, payload_json, dedup_key, seen_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    kind,
                    summary[:400],
                    status,
                    session_id,
                    concept_id,
                    app_name or "",
                    (evidence or "")[:400],
                    json.dumps(payload or {}, ensure_ascii=False),
                    dedup_key or "",
                    seen_at,
                    seen_at,
                ),
            )
            return int(cur.lastrowid)

    def list_events(
        self,
        *,
        kind: str | None = None,
        status: str | None = "open",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        clauses = ["1=1"]
        args: list[Any] = []
        if kind:
            clauses.append("kind = ?")
            args.append(kind)
        if status:
            clauses.append("status = ?")
            args.append(status)
        args.append(limit)
        sql = (
            f"SELECT * FROM learning_events WHERE {' AND '.join(clauses)} "
            "ORDER BY seen_at DESC LIMIT ?"
        )
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        out = []
        for r in rows:
            item = dict(r)
            try:
                item["payload"] = json.loads(r.get("payload_json") or "{}")
            except json.JSONDecodeError:
                item["payload"] = {}
            out.append(item)
        return out

    def set_event_status(self, event_id: int, status: str) -> bool:
        with self._lock:
            cur = self._conn.execute(
                """UPDATE learning_events SET status = ?, updated_at = ?
                    WHERE id = ?""",
                (status, _now_iso(), event_id),
            )
            return cur.rowcount > 0

    def upsert_edges(self, edges: list[ConceptEdge], concept_ids: dict[str, int]) -> int:
        """Persist graph edges; auto-create missing endpoint concepts as stubs."""
        if not edges:
            return 0
        seen_at = _now_iso()
        n = 0
        with self._lock:
            for e in edges:
                src_key = normalize_name(e.src_name)
                dst_key = normalize_name(e.dst_name)
                if not src_key or not dst_key:
                    continue
                sid = concept_ids.get(src_key) or self._ensure_stub_concept(e.src_name, seen_at)
                did = concept_ids.get(dst_key) or self._ensure_stub_concept(e.dst_name, seen_at)
                concept_ids[src_key] = sid
                concept_ids[dst_key] = did
                self._conn.execute(
                    """INSERT INTO learning_edges
                       (src_concept_id, dst_concept_id, rel, weight, evidence, source, seen_at)
                       VALUES (?, ?, ?, ?, ?, 'extract', ?)
                       ON CONFLICT(src_concept_id, dst_concept_id, rel) DO UPDATE SET
                         weight = excluded.weight,
                         evidence = excluded.evidence,
                         seen_at = excluded.seen_at""",
                    (sid, did, e.rel, float(e.weight), (e.evidence or "")[:300], seen_at),
                )
                n += 1
        return n

    def _ensure_stub_concept(self, name: str, seen_at: str) -> int:
        key = normalize_name(name)
        row = self._conn.execute(
            "SELECT id FROM learning_concepts WHERE name_norm = ?", (key,)
        ).fetchone()
        if row:
            return int(row["id"])
        cur = self._conn.execute(
            """INSERT INTO learning_concepts
               (name, name_norm, topic, first_seen, last_seen, hit_count, evidence_json, updated_at)
               VALUES (?, ?, '', ?, ?, 1, '[]', ?)""",
            (name[:100], key, seen_at, seen_at, seen_at),
        )
        return int(cur.lastrowid)

    def list_edges(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT e.*,
                          s.name AS src_name,
                          d.name AS dst_name
                     FROM learning_edges e
                     JOIN learning_concepts s ON s.id = e.src_concept_id
                     JOIN learning_concepts d ON d.id = e.dst_concept_id
                    ORDER BY e.seen_at DESC
                    LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def ingest_problem_sessions(
        self,
        sessions: list[dict[str, Any]],
        session_db_ids: list[int],
    ) -> int:
        """Create learning_events(kind=problem) from detector problem sessions."""
        n = 0
        for s, sid in zip(sessions, session_db_ids):
            if s.get("kind") != "problem":
                continue
            text = (s.get("sample_text") or s.get("title") or "problem").strip()
            seen_at = str(s.get("ended_at") or s.get("started_at") or _now_iso())
            eid = self.insert_event(
                kind="problem",
                summary=text[:200],
                session_id=sid,
                app_name=(s.get("apps") or [""])[0] if s.get("apps") else "",
                evidence=text[:300],
                payload={"slice_ref": f"[{s.get('id')}]", "concepts": s.get("concepts") or []},
                dedup_key=problem_dedup_key(text, seen_at=seen_at),
                seen_at=seen_at,
            )
            if eid:
                n += 1
        return n
