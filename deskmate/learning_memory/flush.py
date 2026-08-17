"""End-session flush → trigger user-learning recap (study-agent style)."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..console import echo_stderr
from ..logger import get

logger = get("learning_memory.flush")


def _link_recap_to_session(session_id: int, out_dir: Path, *, verbose: bool = False) -> None:
    """Record the report's directory on the session that triggered it.

    Stored in ``learning_sessions.meta_json`` rather than a new column: the field
    already exists, and only sessions whose close fired a recap get a link. An
    automatic session covers a fuzzy window, so no attempt is made to guess which
    report "belongs" to one that did not trigger it.
    """
    try:
        from .store import LearningStore  # noqa: PLC0415

        LearningStore().set_session_meta(session_id, {"recap_path": str(out_dir)})
    except Exception as exc:  # noqa: BLE001
        logger.debug("recap link for session %s failed: %s", session_id, exc)
        if verbose:
            echo_stderr(f"  [learning_flush] recap link failed: {exc}")


def trigger_user_learning_recap(
    *,
    hours: float = 8.0,
    verbose: bool = False,
    background: bool = True,
    session_id: int | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> dict[str, Any]:
    """Run the user-learning app (optionally in a daemon thread).

    ``session_id`` records the finished report against the session that triggered
    it. Explicit bounds scope a manually requested recap to that session instead
    of an approximate recent-hours window.
    """

    def _run() -> dict[str, Any]:
        try:
            from deskmate.apps.agent import G_LEARNING_ENRICHMENT, run_agent  # noqa: PLC0415
            from deskmate.apps.common import output_dir, write_markdown  # noqa: PLC0415

            pipe = Path(__file__).resolve().parents[1] / "apps" / "user-learning" / "pipe.md"
            if start_time and end_time:
                start_iso = start_time
                end_iso = end_time
            else:
                end = datetime.now().astimezone()
                start = end - timedelta(hours=max(0.5, float(hours)))
                start_iso = start.replace(microsecond=0).isoformat()
                end_iso = end.replace(microsecond=0).isoformat()
            report = run_agent(
                pipe,
                verbose=verbose,
                # ISO with the default 'T' separator: stored timestamps are
                # 'YYYY-MM-DDTHH:MM:SS+TZ' and every time filter compares them
                # as strings. A ' ' separator sorts BELOW 'T', so a space-form
                # upper bound silently matched nothing and this auto-recap ran
                # with no audio/OCR evidence at all.
                start_time=start_iso,
                end_time=end_iso,
            )
            out = output_dir("user-learning")
            write_markdown(out / "user-learning.md", report)
            if G_LEARNING_ENRICHMENT:
                (out / "learning-enrichment.json").write_text(
                    json.dumps(
                        {
                            "extraction": G_LEARNING_ENRICHMENT.get("extraction"),
                            "topics": G_LEARNING_ENRICHMENT.get("topics"),
                            "due_reviews": G_LEARNING_ENRICHMENT.get("due_reviews"),
                            "events": G_LEARNING_ENRICHMENT.get("events"),
                            "edges": G_LEARNING_ENRICHMENT.get("edges"),
                            "persisted": G_LEARNING_ENRICHMENT.get("persisted"),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            if session_id:
                _link_recap_to_session(int(session_id), out, verbose=verbose)
            if verbose:
                echo_stderr(f"  [learning_flush] wrote {out / 'user-learning.md'}")
            return {"ok": True, "queued": False, "path": str(out / "user-learning.md")}
        except Exception as exc:  # noqa: BLE001
            logger.warning("user-learning flush failed: %s", exc)
            if verbose:
                echo_stderr(f"  [learning_flush] error: {exc}")
            return {"ok": False, "queued": False, "error": str(exc)}

    if background:
        threading.Thread(target=_run, name="user-learning-flush", daemon=True).start()
        return {"ok": True, "queued": True, "hours": hours}
    return _run()
