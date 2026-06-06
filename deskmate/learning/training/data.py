"""DeskMateTrainingDataMiner — extract supervised (input, output) SFT pairs.

DeskMate has no agent trace store, so this miner derives SFT pairs from five
existing local data sources, none of which are modified:

* **habits** — ``habit_suggestions`` rows the user marked useful (``feedback``
  >= threshold). The trigger context becomes the input and the accepted
  coaching message the output.
* **pipes** — successful ``pipe_executions``. The pipe invocation becomes the
  input and its produced report the output.
* **behavior** — statistically stable ``habit_profiles`` slots. A routine
  question becomes the input and the habit description the output.
* **ask** — ``ask_history`` answers the user marked useful (``feedback`` >=
  threshold). The original question becomes the input and the grounded answer
  the output.
* **profile** — aggregated ``habit_profiles`` synthesized into high-level
  "who is this user" identity Q&A (top apps, dominant categories, weekday vs
  weekend rhythm). Gives the fine-tuned model a stable sense of the user
  instead of only isolated routines. Skipped when signal is too thin.

All pairs pass a quality gate (min length, an output-length cap, natural-
language check, input≠output) and are deduplicated whitespace/case-insensitively
with a cap on identical outputs so no handful of rows dominates the gradient.

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
SOURCES: tuple[str, ...] = ("habits", "pipes", "behavior", "ask", "profile")

# Junk that must never become a training target: mouse-click coordinates,
# capture triggers, and accessibility boilerplate.
_NOISE_RE = re.compile(
    r"^(left|right|middle)\s*@\(\d|"                      # click coordinates
    r"^(scroll_stop|frame|manual|periodic|idle|timer)\b|"  # capture triggers
    r"not accessible|screen reader|shift\+alt",            # a11y hints
    re.IGNORECASE,
)


def _looks_like_language(text: str) -> bool:
    """True if *text* reads like natural language.

    Accepts CJK text (where ``split()`` yields one token but the script is
    clearly language) as well as space-delimited scripts with >= 2 words.
    """
    if _has_cjk(text):
        return True
    if not any(ch.isalpha() for ch in text):
        return False
    return len(text.split()) >= 2


def _has_cjk(text: str) -> bool:
    """True if the text contains any CJK ideograph."""
    return any("一" <= ch <= "鿿" for ch in text)


# Outputs longer than this are truncated targets for a small SFT model: a 4000-
# char pipe report teaches length, not preference, and blows up sequence cost.
# We drop pairs whose output exceeds this rather than silently truncating them.
_MAX_OUTPUT_CHARS = 1500

# Cap repeats of an identical output so a handful of rows can't dominate.
_MAX_DUP_OUTPUT = 3

# Whitespace/case-insensitive key used to drop near-duplicate examples that
# would otherwise let a handful of repeated rows dominate the gradient.
def _dedup_key(text: str) -> str:
    return " ".join(text.lower().split())


def _normalize_ws(text: str) -> str:
    """Collapse runs of whitespace; keep single newlines as spaces."""
    return " ".join((text or "").split())


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
            Subset of :data:`SOURCES` (``habits``, ``pipes``, ``behavior``,
            ``ask``, ``profile``). Unknown names are ignored.
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
        if "behavior" in wanted:
            pairs.extend(self._from_habit_profiles(limit_per_source))
        if "ask" in wanted:
            pairs.extend(self._from_ask_history(limit_per_source))
        if "profile" in wanted:
            pairs.extend(self._from_user_profile())

        # Collapse duplicates (whitespace/case-insensitive), keep first. Also cap
        # how many times the SAME output may appear: a few identical pipe reports
        # or routine answers would otherwise dominate the gradient.
        seen: set[tuple[str, str]] = set()
        output_counts: dict[str, int] = {}
        deduped: list[dict[str, Any]] = []
        for p in pairs:
            in_key, out_key = _dedup_key(p["input"]), _dedup_key(p["output"])
            key = (in_key, out_key)
            if key in seen:
                continue
            if output_counts.get(out_key, 0) >= _MAX_DUP_OUTPUT:
                continue
            seen.add(key)
            output_counts[out_key] = output_counts.get(out_key, 0) + 1
            deduped.append(p)
            if len(deduped) >= max_pairs:
                break

        logger.info(
            "mined %d SFT pair(s) from %s", len(deduped), wanted or "[]"
        )
        return deduped

    def export_jsonl(
        self,
        out_path: Path | str,
        *,
        sources: list[str] | tuple[str, ...] = SOURCES,
        limit_per_source: int = 2000,
        max_pairs: int = 5000,
    ) -> int:
        """Mine pairs and write them as JSONL to *out_path* for inspection.

        One JSON object per line with ``input``/``output``/``source`` etc. —
        lets the user eyeball the exact dataset before committing to a training
        run. Returns the number of pairs written."""
        pairs = self.extract_sft_pairs(
            sources=sources, limit_per_source=limit_per_source, max_pairs=max_pairs,
        )
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for p in pairs:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")
        logger.info("exported %d SFT pair(s) to %s", len(pairs), out)
        return len(pairs)

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
        if "behavior" in sources:
            counts["behavior"] = len(self._from_habit_profiles(limit_per_source))
        if "ask" in sources:
            counts["ask"] = len(self._from_ask_history(limit_per_source))
        if "profile" in sources:
            counts["profile"] = len(self._from_user_profile())
        return counts

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass

    # -- per-source extractors --------------------------------------------------

    def _keep(self, text_in: str, text_out: str) -> bool:
        """Quality gate for one (input, output) pair.

        Beyond the min-length floor we reject: outputs longer than
        :data:`_MAX_OUTPUT_CHARS` (length-teaching, not preference), pairs whose
        output isn't natural language, and pairs where input == output (the
        model would learn to echo)."""
        if len(text_in) < self._min_chars or len(text_out) < self._min_chars:
            return False
        if len(text_out) > _MAX_OUTPUT_CHARS:
            return False
        if not _looks_like_language(text_out):
            return False
        if _dedup_key(text_in) == _dedup_key(text_out):
            return False
        return True

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
        """User-approved Ask answers → (question, grounded answer) pairs.

        Only answers the user explicitly marked useful (``feedback`` >=
        threshold) are mined — the same quality gate as ``habit_suggestions`` —
        so a casual or wrong answer never leaks into the training set.
        """
        try:
            rows = self._conn.execute(
                """SELECT question, answer, created_at
                       FROM ask_history
                      WHERE answer IS NOT NULL AND TRIM(answer) <> ''
                        AND feedback IS NOT NULL AND feedback >= ?
                   ORDER BY created_at DESC
                      LIMIT ?""",
                (self._min_feedback, int(limit)),
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

    def _from_user_profile(self) -> list[dict[str, Any]]:
        """Aggregate habit_profiles into high-level 'who is this user' Q&A pairs.

        Where the ``behavior`` source emits one pair per time-slot routine, this
        source synthesizes a handful of *identity* pairs — the user's go-to
        apps, dominant work categories, and weekday/weekend rhythm — so the
        fine-tuned model gains a stable sense of who it is serving rather than
        just memorizing isolated slots. Built only from rows with real
        confidence (multiple sample days), and skipped entirely when there isn't
        enough signal (avoids inventing a persona from one day of data)."""
        try:
            rows = self._conn.execute(
                """SELECT day_type, category, top_app, avg_minutes,
                          frequency, sample_days
                       FROM habit_profiles
                      WHERE sample_days >= 3 AND frequency >= 0.4""",
            ).fetchall()
        except sqlite3.Error as exc:
            logger.debug("habit_profiles unavailable for profile: %s", exc)
            return []

        if len(rows) < 3:
            # Too little signal to characterize the user honestly.
            return []

        app_minutes: dict[str, float] = {}
        cat_minutes: dict[str, float] = {}
        weekday_cats: dict[str, float] = {}
        weekend_cats: dict[str, float] = {}
        for r in rows:
            mins = float(r.get("avg_minutes") or 0) * float(r.get("frequency") or 0)
            app = (r.get("top_app") or "").strip()
            cat = (r.get("category") or "").strip()
            if app:
                app_minutes[app] = app_minutes.get(app, 0.0) + mins
            if cat:
                cat_minutes[cat] = cat_minutes.get(cat, 0.0) + mins
                bucket = weekday_cats if r.get("day_type") == "weekday" else weekend_cats
                bucket[cat] = bucket.get(cat, 0.0) + mins

        def _top(d: dict[str, float], n: int) -> list[str]:
            return [k for k, _ in sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:n]]

        pairs: list[dict[str, Any]] = []

        def _add(prompt: str, answer: str) -> None:
            if self._keep(prompt, answer):
                pairs.append({
                    "input": prompt, "output": answer,
                    "source": "profile", "kind": "identity", "ts": None,
                })

        top_apps = _top(app_minutes, 3)
        if top_apps:
            _add(
                "What apps do I rely on most?",
                "You spend most of your time in " + ", ".join(top_apps)
                + ". These are your primary working tools.",
            )
        top_cats = _top(cat_minutes, 3)
        if top_cats:
            _add(
                "What do I mainly work on?",
                "Your activity is dominated by " + ", ".join(top_cats)
                + ". That's where most of your time goes.",
            )
        wd, we = _top(weekday_cats, 2), _top(weekend_cats, 2)
        if wd:
            _add(
                "What are my weekdays usually like?",
                "On weekdays you mostly focus on " + ", ".join(wd) + ".",
            )
        if we:
            _add(
                "What do I tend to do on weekends?",
                "On weekends your activity shifts toward " + ", ".join(we) + ".",
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
