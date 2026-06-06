"""DeskMateTrainingDataMiner — extract supervised (input, output) SFT pairs.

Analog of OpenJarvis's ``TrainingDataMiner`` (which mined an agent trace store).
DeskMate has no agent trace store, so this miner derives SFT pairs from three
existing local data sources, none of which are modified:

* **habits** — ``habit_suggestions`` rows the user marked useful (``feedback``
  >= threshold). The closest analog to OpenJarvis's "trace + feedback": the
  trigger context becomes the input and the accepted coaching message the
  output.
* **pipes** — successful ``pipe_executions``. The pipe invocation becomes the
  input and its produced report the output.
* **timeline** — the unified ``context_events`` stream. The observable window
  state becomes the input and the fused human-readable summary the output
  (kept only when the summary adds information beyond the window title).

Each pair is a dict with keys ``input``, ``output``, ``source`` (provenance)
plus light metadata. Duplicate ``(input, output)`` pairs are collapsed.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from ... import paths
from ...logger import get

logger = get("learning.training.data")

# Valid provenance tags for ``sources=`` filtering.
SOURCES: tuple[str, ...] = ("habits", "pipes", "timeline", "behavior", "ask")

# Only these unified-timeline kinds carry natural-language content worth
# learning from. Low-level kinds (click coordinates, frame triggers, raw
# window titles) are pure noise for SFT and are never mined.
_CONTENT_KINDS: frozenset[str] = frozenset({"transcript", "clipboard", "text"})

# Per-kind instruction template — keeps the (input, output) pair coherent
# instead of the degenerate "describe X -> mouse coordinate" pairs.
_KIND_PROMPT: dict[str, str] = {
    "transcript": "Transcribe what was said{where}.",
    "clipboard": "What text did I copy{where}?",
    "text": "What did I type{where}?",
}

# Junk that must never become a training target even within a content kind:
# mouse-click coordinates, capture triggers, and accessibility boilerplate.
_NOISE_RE = re.compile(
    r"^(left|right|middle)\s*@\(\d|"                      # click coordinates
    r"^(scroll_stop|frame|manual|periodic|idle|timer)\b|"  # capture triggers
    r"not accessible|screen reader|shift\+alt",            # a11y hints
    re.IGNORECASE,
)


def _looks_like_language(text: str) -> bool:
    """True if *text* reads like natural language (>= 2 words, has letters)."""
    if not any(ch.isalpha() for ch in text):
        return False
    return len(text.split()) >= 2


def _dict_factory(cursor: sqlite3.Cursor, row: tuple) -> dict[str, Any]:
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}



class DeskMateTrainingDataMiner:
    """Mine SFT training pairs from DeskMate's local SQLite database.

    Owns its own read-only connection to the same database file, mirroring
    :class:`~deskmate.fusion.store.ContextStore`. WAL mode makes a second
    reader safe alongside the main ``DatabaseManager``.

    Parameters
    ----------
    db_file:
        Path to the SQLite database. ``None`` uses the canonical
        ``~/.deskmate/data.db``.
    min_feedback:
        Minimum ``habit_suggestions.feedback`` for a row to qualify (1 = the
        user pressed "useful").
    min_chars:
        Drop pairs whose input or output is shorter than this (avoids
        degenerate single-token examples).
    """

    def __init__(
        self,
        db_file: Path | None = None,
        *,
        min_feedback: int = 1,
        min_chars: int = 8,
    ) -> None:
        self.path = Path(db_file) if db_file else paths.db_path()
        self._min_feedback = int(min_feedback)
        self._min_chars = int(min_chars)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = _dict_factory
        self._conn.execute("PRAGMA busy_timeout = 5000")

    # -- public API -------------------------------------------------------------

    def extract_sft_pairs(
        self,
        *,
        sources: list[str] | tuple[str, ...] = SOURCES,
        limit_per_source: int = 2000,
        max_pairs: int = 5000,
    ) -> list[dict[str, Any]]:
        """Return deduplicated SFT pairs from the requested *sources*.

        Parameters
        ----------
        sources:
            Subset of ``{"habits", "pipes", "timeline"}``. Unknown names are
            ignored.
        limit_per_source:
            Max rows scanned per source (most recent first).
        max_pairs:
            Hard cap on the total number of returned pairs.
        """
        wanted = [s for s in sources if s in SOURCES]
        pairs: list[dict[str, Any]] = []

        if "habits" in wanted:
            pairs.extend(self._from_habit_suggestions(limit_per_source))
        if "pipes" in wanted:
            pairs.extend(self._from_pipe_executions(limit_per_source))
        if "timeline" in wanted:
            pairs.extend(self._from_context_events(limit_per_source))
        if "behavior" in wanted:
            pairs.extend(self._from_habit_profiles(limit_per_source))
        if "ask" in wanted:
            pairs.extend(self._from_ask_history(limit_per_source))

        # Collapse duplicates, keep first occurrence.
        seen: set[tuple[str, str]] = set()
        deduped: list[dict[str, Any]] = []
        for p in pairs:
            key = (p["input"], p["output"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(p)
            if len(deduped) >= max_pairs:
                break

        logger.info(
            "mined %d SFT pair(s) from %s", len(deduped), wanted or "[]"
        )
        return deduped

    def source_breakdown(
        self,
        *,
        sources: list[str] | tuple[str, ...] = SOURCES,
        limit_per_source: int = 2000,
    ) -> dict[str, int]:
        """Return per-source qualifying pair counts (cheap preview helper)."""
        counts: dict[str, int] = {}
        if "habits" in sources:
            counts["habits"] = len(self._from_habit_suggestions(limit_per_source))
        if "pipes" in sources:
            counts["pipes"] = len(self._from_pipe_executions(limit_per_source))
        if "timeline" in sources:
            counts["timeline"] = len(self._from_context_events(limit_per_source))
        if "behavior" in sources:
            counts["behavior"] = len(self._from_habit_profiles(limit_per_source))
        if "ask" in sources:
            counts["ask"] = len(self._from_ask_history(limit_per_source))
        return counts

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass

    # -- per-source extractors --------------------------------------------------

    def _keep(self, text_in: str, text_out: str) -> bool:
        return len(text_in) >= self._min_chars and len(text_out) >= self._min_chars

    def _from_habit_suggestions(self, limit: int) -> list[dict[str, Any]]:
        """User-approved coaching nudges → (trigger context, message) pairs."""
        try:
            rows = self._conn.execute(
                """SELECT rule_name, message, context_json, feedback, created_at
                       FROM habit_suggestions
                      WHERE feedback IS NOT NULL AND feedback >= ?
                   ORDER BY created_at DESC
                      LIMIT ?""",
                (self._min_feedback, int(limit)),
            ).fetchall()
        except sqlite3.Error as exc:
            logger.debug("habit_suggestions unavailable: %s", exc)
            return []

        pairs: list[dict[str, Any]] = []
        for r in rows:
            message = (r.get("message") or "").strip()
            if not message:
                continue
            rule = (r.get("rule_name") or "").strip()
            ctx = self._render_context(r.get("context_json"))
            if ctx:
                prompt = f"My recent activity shows: {ctx}. What helpful nudge should I get?"
            else:
                prompt = f"The '{rule}' pattern was detected in my activity. What helpful nudge should I get?"
            if not self._keep(prompt, message):
                continue
            pairs.append(
                {
                    "input": prompt,
                    "output": message,
                    "source": "habit_suggestion",
                    "rule": rule,
                    "feedback": r.get("feedback"),
                    "ts": r.get("created_at"),
                }
            )
        return pairs

    def _from_pipe_executions(self, limit: int) -> list[dict[str, Any]]:
        """Successful pipe runs → (invocation, produced report) pairs."""
        try:
            rows = self._conn.execute(
                """SELECT pipe_name, output, started_at
                       FROM pipe_executions
                      WHERE status = 'success' AND output IS NOT NULL AND TRIM(output) <> ''
                   ORDER BY started_at DESC
                      LIMIT ?""",
                (int(limit),),
            ).fetchall()
        except sqlite3.Error as exc:
            logger.debug("pipe_executions unavailable: %s", exc)
            return []

        pairs: list[dict[str, Any]] = []
        for r in rows:
            output = (r.get("output") or "").strip()
            pipe = (r.get("pipe_name") or "").strip()
            if not output or not pipe:
                continue
            prompt = f"Run the '{pipe}' assistant and report the result."
            if not self._keep(prompt, output):
                continue
            pairs.append(
                {
                    "input": prompt,
                    "output": output,
                    "source": "pipe_execution",
                    "pipe": pipe,
                    "ts": r.get("started_at"),
                }
            )
        return pairs

    def _from_context_events(self, limit: int) -> list[dict[str, Any]]:
        """Unified timeline → (recall instruction, content) pairs.

        Only content-bearing kinds (transcripts, clipboard, typed text) are
        mined. Low-level signals (click coordinates, capture triggers, raw
        window titles) and accessibility boilerplate are rejected as noise, so
        the resulting pairs are coherent natural-language examples rather than
        degenerate "describe X -> mouse coordinate" pairs.
        """
        placeholders = ",".join("?" for _ in _CONTENT_KINDS)
        try:
            rows = self._conn.execute(
                f"""SELECT source, kind, app_name, window_title, summary, ts
                       FROM context_events
                      WHERE summary IS NOT NULL AND TRIM(summary) <> ''
                        AND kind IN ({placeholders})
                   ORDER BY ts DESC
                      LIMIT ?""",
                (*sorted(_CONTENT_KINDS), int(limit)),
            ).fetchall()
        except sqlite3.Error as exc:
            logger.debug("context_events unavailable: %s", exc)
            return []

        pairs: list[dict[str, Any]] = []
        for r in rows:
            summary = (r.get("summary") or "").strip()
            kind = (r.get("kind") or "").strip()
            app = (r.get("app_name") or "").strip()
            title = (r.get("window_title") or "").strip()
            if not summary or kind not in _CONTENT_KINDS:
                continue
            # Reject coordinate / trigger / accessibility noise and non-language.
            if _NOISE_RE.search(summary) or not _looks_like_language(summary):
                continue
            # Skip when the content just restates the window title.
            if title and summary.lower() == title.lower():
                continue
            where = f" in {app}" if app else ""
            prompt = _KIND_PROMPT.get(kind, "Describe what happened{where}.").format(where=where)
            if not self._keep(prompt, summary):
                continue
            pairs.append(
                {
                    "input": prompt,
                    "output": summary,
                    "source": f"timeline:{r.get('source')}",
                    "kind": kind,
                    "ts": r.get("ts"),
                }
            )
        return pairs

    def _from_habit_profiles(self, limit: int) -> list[dict[str, Any]]:
        """Learned behavior profiles → (routine question, habit description).

        Each confident ``habit_profiles`` row (a recurring time-slot behavior)
        becomes a natural-language Q&A pair so the model learns the user's
        routine, e.g. "What do I usually do on weekday mornings around 09:00?"
        → "Coding — usually in Code.exe for about 25 min (on 80% of days)."
        Only rows backed by multiple days and a non-trivial frequency are kept.
        """
        try:
            rows = self._conn.execute(
                """SELECT day_type, slot, category, top_app, avg_minutes,
                          frequency, sample_days
                       FROM habit_profiles
                      WHERE sample_days >= 2 AND frequency >= 0.3
                   ORDER BY frequency DESC, sample_days DESC
                      LIMIT ?""",
                (int(limit),),
            ).fetchall()
        except sqlite3.Error as exc:
            logger.debug("habit_profiles unavailable: %s", exc)
            return []

        pairs: list[dict[str, Any]] = []
        for r in rows:
            category = (r.get("category") or "").strip()
            if not category:
                continue
            slot = int(r.get("slot") or 0)
            hour, minute = slot // 2, (30 if slot % 2 else 0)
            day_type = "weekdays" if r.get("day_type") == "weekday" else "weekends"
            top_app = (r.get("top_app") or "").strip()
            avg_min = int(round(float(r.get("avg_minutes") or 0)))
            freq_pct = int(round(float(r.get("frequency") or 0) * 100))
            prompt = f"What do I usually do on {day_type} around {hour:02d}:{minute:02d}?"
            app_part = f", usually in {top_app}," if top_app else ""
            answer = (
                f"Typically {category}{app_part} for about {avg_min} min "
                f"(on {freq_pct}% of days)."
            )
            if not self._keep(prompt, answer):
                continue
            pairs.append(
                {
                    "input": prompt,
                    "output": answer,
                    "source": "behavior",
                    "kind": category,
                    "ts": None,
                }
            )
        return pairs

    def _from_ask_history(self, limit: int) -> list[dict[str, Any]]:
        """Answered Ask queries → (question, grounded answer) pairs.

        The user's own questions paired with the assistant's evidence-based
        answers are the highest-signal SFT source. Empty answers and obvious
        "no data / paused recording" non-answers are dropped.
        """
        try:
            rows = self._conn.execute(
                """SELECT question, answer, created_at
                       FROM ask_history
                      WHERE answer IS NOT NULL AND TRIM(answer) <> ''
                   ORDER BY created_at DESC
                      LIMIT ?""",
                (int(limit),),
            ).fetchall()
        except sqlite3.Error as exc:
            logger.debug("ask_history unavailable: %s", exc)
            return []

        pairs: list[dict[str, Any]] = []
        for r in rows:
            question = (r.get("question") or "").strip()
            answer = (r.get("answer") or "").strip()
            if not question or not answer:
                continue
            if _NOISE_RE.search(answer):
                continue
            if not self._keep(question, answer):
                continue
            pairs.append(
                {
                    "input": question,
                    "output": answer,
                    "source": "ask",
                    "kind": "qa",
                    "ts": r.get("created_at"),
                }
            )
        return pairs

    # -- helpers ----------------------------------------------------------------

    @staticmethod
    def _render_context(context_json: Any) -> str:
        """Render the first few evidence fields of a context_json blob."""
        if not context_json:
            return ""
        try:
            data = json.loads(context_json) if isinstance(context_json, str) else context_json
        except (ValueError, TypeError):
            return ""
        if not isinstance(data, dict) or not data:
            return ""
        parts = [f"{k}={v}" for k, v in list(data.items())[:5] if v not in (None, "")]
        return "; ".join(parts)


__all__ = ["DeskMateTrainingDataMiner", "SOURCES"]
