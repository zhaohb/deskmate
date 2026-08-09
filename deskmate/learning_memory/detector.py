"""Frame-level learning session open/close (MeetingDetector style).

Watches UI/frame observations, classifies learning signals, and maintains at
most one open ``learning_sessions`` row with an idle grace before close + optional
user-learning flush.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..apps.learning_slice import classify_learning_signal, extract_search_query
from ..logger import get
from .extract import extract_concepts_from_texts
from .store import LearningStore, problem_dedup_key

logger = get("learning_memory.detector")


def _emit(event_name: str, **data: object) -> None:
    try:
        from .. import events as bus  # noqa: PLC0415

        bus.send(bus.EventType(event_name), **data)
    except Exception as exc:  # noqa: BLE001
        logger.debug("learning event %s emit skipped: %s", event_name, exc)


def _now_iso() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class LearningObservation:
    in_learning: bool
    kind: str | None
    confidence: float
    reason: str
    app_name: str
    window_title: str
    browser_url: str | None
    topics: tuple[str, ...]
    concepts: tuple[str, ...]


class LearningSessionDetector:
    """Small FSM: open on strong learning signal, keep alive, grace-close."""

    def __init__(
        self,
        store: LearningStore | None = None,
        *,
        end_grace_seconds: float = 180.0,
        start_confidence: float = 0.75,
        keep_confidence: float = 0.60,
        auto_recap_on_end: bool = True,
        auto_recap_hours: float = 8.0,
        enabled: bool = True,
    ) -> None:
        self.store = store or LearningStore()
        self.end_grace_seconds = float(end_grace_seconds)
        self.start_confidence = float(start_confidence)
        self.keep_confidence = float(keep_confidence)
        self.auto_recap_on_end = bool(auto_recap_on_end)
        self.auto_recap_hours = float(auto_recap_hours)
        self.enabled = bool(enabled)

        self._session_id: int | None = None
        self._kind: str | None = None
        self._last_seen = 0.0
        self._title = ""
        self._restore_open()

    def _restore_open(self) -> None:
        try:
            rows = self.store.list_sessions(status="open", limit=1)
        except Exception:  # noqa: BLE001
            return
        if not rows:
            return
        row = rows[0]
        self._session_id = int(row["id"])
        self._kind = str(row.get("kind") or "") or None
        self._title = str(row.get("title") or "")
        # Treat restored session as recently seen so we don't instantly close.
        self._last_seen = time.time()
        logger.info("restored open learning session id=%s kind=%s", self._session_id, self._kind)

    @property
    def active_session_id(self) -> int | None:
        return self._session_id

    def is_in_learning(self) -> bool:
        self.expire_if_idle()
        return self._session_id is not None

    def observe(
        self,
        *,
        app_name: str,
        window_title: str,
        browser_url: str | None = None,
        text: str = "",
        skip: bool = False,
    ) -> LearningObservation:
        """Observe one UI/frame sample. ``skip=True`` forces idle path (e.g. in meeting)."""
        if not self.enabled or skip:
            obs = LearningObservation(
                in_learning=False,
                kind=None,
                confidence=0.0,
                reason="disabled" if not self.enabled else "skipped",
                app_name=app_name or "",
                window_title=window_title or "",
                browser_url=browser_url,
                topics=(),
                concepts=(),
            )
            if self._session_id is not None and not skip:
                self.expire_if_idle()
            elif skip:
                # Still allow grace expiry while skipped (meeting shouldn't freeze forever).
                self.expire_if_idle()
            return obs

        kind, conf, reason = classify_learning_signal(
            app_name=app_name or "",
            window_name=window_title or "",
            browser_url=browser_url or "",
            text=text or "",
        )
        topics, concepts = self._tag(window_title, browser_url or "", text, kind=kind or "")
        now = time.time()
        active = False

        if kind and conf >= self.keep_confidence:
            if self._session_id is None:
                if conf >= self.start_confidence:
                    self._open(
                        kind=kind,
                        title=(window_title or app_name or kind)[:160],
                        app_name=app_name or "",
                        browser_url=browser_url or "",
                        topics=list(topics),
                        concepts=list(concepts),
                        confidence=conf,
                        reason=reason,
                        sample_text=(text or "")[:400],
                    )
                    active = True
                    self._last_seen = now
            else:
                # Keep-alive (+ soft metadata refresh).
                self._last_seen = now
                active = True
                self._touch(
                    kind=kind,
                    title=(window_title or self._title or kind)[:160],
                    app_name=app_name or "",
                    browser_url=browser_url or "",
                    topics=list(topics),
                    concepts=list(concepts),
                    confidence=conf,
                    reason=reason,
                    sample_text=(text or "")[:400],
                )
                if kind == "problem":
                    self._note_problem(text or window_title or "problem", app_name=app_name or "")
        else:
            self.expire_if_idle(now=now)
            active = self._session_id is not None

        return LearningObservation(
            in_learning=active and kind is not None,
            kind=kind,
            confidence=float(conf),
            reason=reason,
            app_name=app_name or "",
            window_title=window_title or "",
            browser_url=browser_url,
            topics=topics,
            concepts=concepts,
        )

    def expire_if_idle(self, *, now: float | None = None) -> None:
        if self._session_id is None:
            return
        current = time.time() if now is None else now
        if current - self._last_seen < self.end_grace_seconds:
            return
        self._close(trigger_recap=self.auto_recap_on_end)

    def force_open(
        self,
        *,
        kind: str = "study_other",
        title: str = "",
        topics: list[str] | None = None,
        concepts: list[str] | None = None,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Manual /start-study routed through the FSM so state stays in sync.

        Going straight to the store would leave the detector unaware of the open
        row, and its next auto-open would silently close the manual session.
        """
        if self._session_id is not None:
            self._close(trigger_recap=False)
        row = self.store.start_session(
            kind=kind or "study_other",
            title=title or "Study session",
            topics=topics or [],
            concepts=concepts or [],
            meta={**(meta or {}), "detection_source": "manual"},
        )
        self._session_id = int(row["id"])
        self._kind = kind or "study_other"
        self._title = str(row.get("title") or title or "")
        self._last_seen = time.time()
        logger.info("learning session started (manual) id=%s kind=%s", self._session_id, self._kind)
        _emit(
            "learning_session_started",
            session_id=self._session_id,
            kind=self._kind,
            title=self._title,
        )
        return row

    def force_close(self, *, trigger_recap: bool | None = None) -> int | None:
        if self._session_id is None:
            return None
        sid = self._session_id
        self._close(trigger_recap=self.auto_recap_on_end if trigger_recap is None else trigger_recap)
        return sid

    def _open(
        self,
        *,
        kind: str,
        title: str,
        app_name: str,
        browser_url: str,
        topics: list[str],
        concepts: list[str],
        confidence: float,
        reason: str,
        sample_text: str,
    ) -> None:
        row = self.store.start_session(
            kind=kind,
            title=title,
            topics=topics,
            concepts=concepts,
            meta={
                "detection_source": "frame_scan",
                "app_name": app_name,
                "browser_url": browser_url,
                "reason": reason,
                "confidence": confidence,
            },
        )
        sid = int(row["id"])
        # Enrich apps/urls/sample on the fresh open row.
        self.store.touch_session(
            sid,
            kind=kind,
            title=title,
            apps=[app_name] if app_name else [],
            urls=[browser_url] if browser_url else [],
            queries=_queries_from_url(browser_url),
            topics=topics,
            concepts=concepts,
            confidence=confidence,
            reason=reason,
            sample_text=sample_text,
        )
        self._session_id = sid
        self._kind = kind
        self._title = title
        logger.info("learning session started id=%s kind=%s", sid, kind)
        _emit(
            "learning_session_started",
            session_id=sid,
            kind=kind,
            title=title,
        )
        if kind == "problem":
            self._note_problem(sample_text or title, app_name=app_name)

    def _touch(
        self,
        *,
        kind: str,
        title: str,
        app_name: str,
        browser_url: str,
        topics: list[str],
        concepts: list[str],
        confidence: float,
        reason: str,
        sample_text: str,
    ) -> None:
        if self._session_id is None:
            return
        self._kind = kind or self._kind
        self._title = title or self._title
        try:
            self.store.touch_session(
                self._session_id,
                kind=kind,
                title=title,
                apps=[app_name] if app_name else [],
                urls=[browser_url] if browser_url else [],
                queries=_queries_from_url(browser_url),
                topics=topics,
                concepts=concepts,
                confidence=confidence,
                reason=reason,
                sample_text=sample_text,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("touch_session failed: %s", exc)

    def _close(self, *, trigger_recap: bool) -> None:
        sid = self._session_id
        if sid is None:
            return
        kind = self._kind
        try:
            self.store.end_session(sid)
        except Exception as exc:  # noqa: BLE001
            logger.warning("end learning session %s failed: %s", sid, exc)
        logger.info("learning session ended id=%s kind=%s", sid, kind)
        self._session_id = None
        self._kind = None
        self._title = ""
        self._last_seen = 0.0
        _emit(
            "learning_session_ended",
            session_id=sid,
            kind=kind,
            trigger_recap=bool(trigger_recap),
            hours=self.auto_recap_hours,
        )

    def _note_problem(self, summary: str, *, app_name: str) -> None:
        text = (summary or "").strip()
        if not text or self._session_id is None:
            return
        try:
            self.store.insert_event(
                kind="problem",
                summary=text[:200],
                session_id=self._session_id,
                app_name=app_name,
                evidence=text[:300],
                dedup_key=problem_dedup_key(text),
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("problem event insert failed: %s", exc)

    @staticmethod
    def _tag(
        title: str, url: str, text: str, *, kind: str
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        blobs = [title, text, extract_search_query(url) or ""]
        try:
            hits = extract_concepts_from_texts(
                [b for b in blobs if b and b.strip()], max_concepts=6
            )
        except Exception:  # noqa: BLE001
            hits = []
        concepts = tuple(h.name for h in hits[:5])
        topics: list[str] = []
        for h in hits:
            if h.topic and h.topic not in topics and h.topic not in {"general", "general-zh"}:
                topics.append(h.topic)
        if not topics and kind:
            topics = [kind]
        return tuple(topics[:3]), concepts


def _queries_from_url(url: str) -> list[str]:
    q = extract_search_query(url or "")
    return [q] if q else []
