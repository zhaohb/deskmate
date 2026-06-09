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
* **apps** — markdown reports the local LLM apps wrote to
  ``~/.deskmate/apps/<app>/output/<run>/*.md`` (day-recap, user-profile,
  habit-report, standup, …). Each becomes an (instruction → report) pair so the
  fine-tuned model learns to produce these reports in the user's own style.

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
SOURCES: tuple[str, ...] = ("habits", "pipes", "behavior", "ask", "profile", "apps")

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

# App reports (day-recap / user-profile / habit-report …) are intentionally
# long-form — producing the full report IS the target — so they get a higher
# cap than short nudges/answers. Still bounded to keep sequence cost sane.
_MAX_REPORT_CHARS = 6000

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


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split a leading ``---\\n…\\n---`` YAML frontmatter block off *text*.

    Returns ``(fields, body)``. Only flat ``key: value`` lines are parsed (which
    is all the app reports emit); anything fancier is ignored. When there's no
    frontmatter, returns ``({}, text)`` unchanged. We avoid a YAML dependency on
    purpose — the miner stays import-light."""
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?", text, flags=re.DOTALL)
    if not m:
        return {}, text
    fields: dict[str, str] = {}
    for line in m.group(1).splitlines():
        key, sep, val = line.partition(":")
        if sep and key.strip():
            fields[key.strip()] = val.strip().strip("'\"")
    return fields, text[m.end():].strip()


def _humanize_iso(ts: str) -> str:
    """``2026-06-09T11:46:57+08:00`` → ``2026-06-09 11:46``; pass through other
    shapes (e.g. a bare date) unchanged. Best-effort, no tz math."""
    if not ts:
        return ""
    m = re.match(r"(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})", ts)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    return ts.strip()


def _time_window_label(fm: dict[str, str]) -> str:
    """Build a human time-range label from report frontmatter.

    Prefers an explicit ``window_start``/``window_end`` pair, falls back to a
    single ``date``. Returns ``""`` when there's nothing usable."""
    start = _humanize_iso(fm.get("window_start", ""))
    end = _humanize_iso(fm.get("window_end", ""))
    if start and end:
        return f"{start} → {end}"
    if start:
        return start
    return _humanize_iso(fm.get("date", ""))


def _ts_to_label(run_dir_name: str) -> str:
    """Fallback when a report has no frontmatter: turn a run-dir timestamp like
    ``20260609T124657`` into ``2026-06-09 12:46``. Returns ``""`` if it doesn't
    look like one."""
    m = re.match(r"^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})", run_dir_name or "")
    if not m:
        return ""
    y, mo, d, h, mi = m.groups()
    return f"{y}-{mo}-{d} {h}:{mi}"


# Internal activity-category enum (see db/schema.py:
# coding|browsing|email|meeting|writing|communication|other) → display words in
# each language. Used by the synthetic sources (behavior/profile) and the
# habit-context renderer so a category never leaks as a bare English enum into a
# Chinese sentence (or vice-versa).
_CATEGORY_WORDS = {
    "zh": {
        "coding": "写代码", "browsing": "浏览网页", "email": "处理邮件",
        "meeting": "开会", "writing": "写作", "communication": "沟通交流",
        "other": "其他事务",
    },
    "en": {
        "coding": "coding", "browsing": "browsing", "email": "email",
        "meeting": "meetings", "writing": "writing",
        "communication": "communication", "other": "other tasks",
    },
}


def _category_word(category: str, lang: str) -> str:
    """Map an internal category enum to a display word in *lang* (zh/en).
    Unknown categories pass through unchanged so we never drop information."""
    return _CATEGORY_WORDS.get(lang, {}).get(category, category)


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
        apps: list[str] | tuple[str, ...] | None = None,
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
        apps:
            Optional allow-list of app names for the ``apps`` source. ``None``
            means *all* apps; a list restricts mining to those app folders (lets
            the user exclude low-signal apps like ``ai-prompt-journal`` while
            keeping analytic ones like ``day-recap``). No effect unless ``apps``
            is among *sources*.
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
        if "apps" in wanted:
            pairs.extend(self._from_app_outputs(limit_per_source, apps=apps))

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
        if "apps" in sources:
            counts["apps"] = len(self._from_app_outputs(limit_per_source))
        return counts

    def list_apps(self) -> list[dict[str, Any]]:
        """Discover apps that have on-disk reports, with a per-app pair count.

        Powers the UI's per-app pickers under the ``apps`` source. Returns
        ``[{"app": <name>, "pairs": <n>}]`` sorted by name, where ``pairs`` is
        how many SFT pairs that single app currently contributes."""
        apps_root = paths.root() / "apps"
        if not apps_root.is_dir():
            return []
        names: list[str] = []
        for app_dir in sorted(apps_root.iterdir(), key=lambda p: p.name):
            if app_dir.is_dir() and (app_dir / "output").is_dir():
                names.append(app_dir.name)
        out: list[dict[str, Any]] = []
        for name in names:
            # Reuse the real extractor so the count matches what training sees.
            n = len(self._from_app_outputs(2000, apps=[name]))
            out.append({"app": name, "pairs": n})
        return out

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass

    # -- per-source extractors --------------------------------------------------

    def _keep(self, text_in: str, text_out: str, *, max_out: int = _MAX_OUTPUT_CHARS) -> bool:
        """Quality gate for one (input, output) pair.

        Beyond the min-length floor we reject: outputs longer than *max_out*
        (length-teaching, not preference), pairs whose output isn't natural
        language, and pairs where input == output (the model would learn to
        echo). *max_out* defaults to :data:`_MAX_OUTPUT_CHARS`; long-form
        sources (e.g. app reports) may raise it via :data:`_MAX_REPORT_CHARS`."""
        if len(text_in) < self._min_chars or len(text_out) < self._min_chars:
            return False
        if len(text_out) > max_out:
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
                prompt = f"根据我最近的活动（{ctx}），给我一句有帮助的提醒。"
            else:
                prompt = f"我的活动里出现了「{rule}」的情况，给我一句有帮助的提醒。"
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

    def _from_app_outputs(
        self, limit: int, *, apps: list[str] | tuple[str, ...] | None = None,
    ) -> list[dict[str, Any]]:
        """Local LLM-app reports on disk → (instruction, report) SFT pairs.

        Apps (day-recap, user-profile, habit-report, standup, …) write their
        markdown to ``~/.deskmate/apps/<app>/output/<run>/*.md``. Each report is
        a high-quality, user-grounded long-form target, so the fine-tuned model
        learns to produce these in the user's own style. We read the newest runs
        first, strip any trailing metadata footer, and keep one pair per report.

        *apps*: optional allow-list of app names. ``None`` = all apps; otherwise
        only those folders are mined (so the user can drop low-signal apps).
        """
        apps_root = paths.root() / "apps"
        if not apps_root.is_dir():
            return []
        allow = {a for a in apps} if apps is not None else None

        # Friendly instruction per known app; unknown apps get a generic one.
        instructions = {
            "day-recap": "总结我今天的工作进展、关键时刻和未完成的事项。",
            "user-profile": "根据我近期的活动，总结我的用户画像（角色、兴趣、工作习惯、协作）。",
            "habit-report": "总结我的行为习惯：作息规律、专注节奏和常用工具链。",
            "standup-update": "用 Yesterday / Today / Blockers 的格式写一份站会更新。",
            "time-breakdown": "分析我的时间都花在了哪些事情上。",
            "ai-habits": "总结我使用了哪些 AI 工具以及如何使用的。",
            "ai-prompt-journal": "整理我最近发给各类 AI 工具的提问，按时间汇编成一份 prompt 日志。",
            "email-digest": "汇总我的邮箱概况。",
        }

        # Collect (app, run_dir, md_file) newest-first across all apps.
        runs: list[tuple[str, Path]] = []
        for app_dir in apps_root.iterdir():
            out_root = app_dir / "output"
            if not app_dir.is_dir() or not out_root.is_dir():
                continue
            if allow is not None and app_dir.name not in allow:
                continue  # user excluded this app from the training set
            for run_dir in out_root.iterdir():
                if not run_dir.is_dir():
                    continue
                md_files = sorted(run_dir.glob("*.md"))
                if md_files:
                    runs.append((app_dir.name, md_files[0]))
        # run dir names are timestamps (YYYYmmddTHHMMSS) → newest first by name.
        runs.sort(key=lambda t: t[1].parent.name, reverse=True)

        pairs: list[dict[str, Any]] = []
        for app_name, md_path in runs[: int(limit)]:
            try:
                report = md_path.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
            except OSError as exc:
                logger.debug("app output unreadable %s: %s", md_path, exc)
                continue
            # Pull the YAML frontmatter (date / window_start / window_end …) off
            # the head, both to extract the time window and to keep that block
            # out of the training target (it's metadata noise, not language).
            fm, report = _split_frontmatter(report)
            # Drop a trailing "---\n_…_" metadata footer some apps append
            # (e.g. day-recap's "_时间窗：x → y_"). Tolerant of blank lines / \r.
            report = re.split(r"\n\s*-{3,}\s*\n+\s*_", report, maxsplit=1)[0].strip()
            if not report:
                continue
            base = instructions.get(
                app_name, f"运行「{app_name}」助手并输出结果。"
            )
            # Anchor the instruction in time so the model learns these reports
            # are time-scoped (and which window produced them).
            time_label = _time_window_label(fm) or _ts_to_label(md_path.parent.name)
            prompt = f"{base.rstrip('。.')}（时间范围：{time_label}）。" if time_label else base
            if not self._keep(prompt, report, max_out=_MAX_REPORT_CHARS):
                continue
            pairs.append(
                {
                    "input": prompt,
                    "output": report,
                    "source": "app_output",
                    "app": app_name,
                    "ts": fm.get("date") or md_path.parent.name,
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
            is_weekday = r.get("day_type") == "weekday"
            top_app = (r.get("top_app") or "").strip()
            avg_min = int(round(float(r.get("avg_minutes") or 0)))
            freq_pct = int(round(float(r.get("frequency") or 0) * 100))

            # Synthetic source → emit both a Chinese and an English pair so the
            # fine-tuned model learns this routine in either language.
            cat_en = _category_word(category, "en")
            day_en = "weekdays" if is_weekday else "weekends"
            prompt_en = f"What do I usually do on {day_en} around {hour:02d}:{minute:02d}?"
            app_en = f", usually in {top_app}," if top_app else ""
            answer_en = (
                f"Typically {cat_en}{app_en} for about {avg_min} min "
                f"(on {freq_pct}% of days)."
            )

            cat_zh = _category_word(category, "zh")
            day_zh = "工作日" if is_weekday else "周末"
            prompt_zh = f"{day_zh} {hour:02d}:{minute:02d} 左右我通常在做什么？"
            app_zh = f"，通常用 {top_app}" if top_app else ""
            answer_zh = (
                f"一般是{cat_zh}{app_zh}，大约 {avg_min} 分钟"
                f"（{freq_pct}% 的天数如此）。"
            )

            for prompt, answer in ((prompt_en, answer_en), (prompt_zh, answer_zh)):
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

        # Synthetic source → each identity fact is emitted in BOTH languages.
        def _add(prompt_en: str, answer_en: str, prompt_zh: str, answer_zh: str) -> None:
            for prompt, answer in ((prompt_en, answer_en), (prompt_zh, answer_zh)):
                if self._keep(prompt, answer):
                    pairs.append({
                        "input": prompt, "output": answer,
                        "source": "profile", "kind": "identity", "ts": None,
                    })

        def _cats(keys: list[str], lang: str) -> str:
            return "、".join(_category_word(k, lang) for k in keys) if lang == "zh" \
                else ", ".join(_category_word(k, lang) for k in keys)

        top_apps = _top(app_minutes, 3)
        if top_apps:
            apps_join = ", ".join(top_apps)
            _add(
                "What apps do I rely on most?",
                f"You spend most of your time in {apps_join}. "
                "These are your primary working tools.",
                "我最常用哪些应用？",
                f"你大部分时间花在 {apps_join} 上，这些是你的主力工具。",
            )
        top_cats = _top(cat_minutes, 3)
        if top_cats:
            _add(
                "What do I mainly work on?",
                "Your activity is dominated by " + _cats(top_cats, "en")
                + ". That's where most of your time goes.",
                "我主要在做什么？",
                "你的活动主要集中在" + _cats(top_cats, "zh")
                + "，大部分时间都花在这些上面。",
            )
        wd, we = _top(weekday_cats, 2), _top(weekend_cats, 2)
        if wd:
            _add(
                "What are my weekdays usually like?",
                "On weekdays you mostly focus on " + _cats(wd, "en") + ".",
                "我的工作日通常是什么样的？",
                "工作日你主要专注于" + _cats(wd, "zh") + "。",
            )
        if we:
            _add(
                "What do I tend to do on weekends?",
                "On weekends your activity shifts toward " + _cats(we, "en") + ".",
                "我周末一般做些什么？",
                "周末你的活动更多转向" + _cats(we, "zh") + "。",
            )
        return pairs

    # -- helpers ----------------------------------------------------------------

    @staticmethod
    def _render_context(context_json: Any) -> str:
        """Render a habit-rule ``context_json`` blob as a short natural-language
        clause, e.g. *"I've been browsing in chrome.exe for 15 min (1 min in
        this window), and it's 23:00"*.

        The blob is ``{"state": {category, app_name, continuous_minutes,
        app_minutes, …}, "hour": h, …}`` (see ``habits/rules.py``). The old
        version dumped raw ``key=value`` pairs, which leaked a nested Python
        dict (``state={'category': …}``) into the training prompt — not natural
        language. We pull the few human-meaningful fields and phrase them; if
        the shape is unfamiliar we return ``""`` so the caller falls back to the
        rule-name phrasing rather than emitting noise."""
        if not context_json:
            return ""
        try:
            data = json.loads(context_json) if isinstance(context_json, str) else context_json
        except (ValueError, TypeError):
            return ""
        if not isinstance(data, dict) or not data:
            return ""

        state = data.get("state") if isinstance(data.get("state"), dict) else {}
        category = (state.get("category") or "").strip()
        cat_word = _category_word(category, "zh") if category else ""
        cat_phrase = f"在{cat_word}" if cat_word else ""
        app_name = (state.get("app_name") or "").strip()
        cont = state.get("continuous_minutes")
        app_min = state.get("app_minutes")

        clauses: list[str] = []
        if cat_phrase and app_name:
            clauses.append(f"我一直{cat_phrase}，当前用的是 {app_name}")
        elif cat_phrase:
            clauses.append(f"我一直{cat_phrase}")
        elif app_name:
            clauses.append(f"我当前用的是 {app_name}")

        try:
            if cont is not None and float(cont) >= 1:
                detail = f"已连续使用屏幕约 {int(float(cont))} 分钟"
                if app_min is not None and float(app_min) >= 1:
                    detail += f"（这个应用上约 {int(float(app_min))} 分钟）"
                clauses.append(detail)
        except (TypeError, ValueError):
            pass

        hour = data.get("hour")
        try:
            if hour is not None:
                clauses.append(f"现在是 {int(hour):02d}:00 左右")
        except (TypeError, ValueError):
            pass

        return "；".join(clauses)


__all__ = ["DeskMateTrainingDataMiner", "SOURCES"]
