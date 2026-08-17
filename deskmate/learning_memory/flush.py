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


def trigger_user_learning_recap(
    *,
    hours: float = 8.0,
    verbose: bool = False,
    background: bool = True,
) -> dict[str, Any]:
    """Run the user-learning app (optionally in a daemon thread)."""

    def _run() -> None:
        try:
            from deskmate.apps.agent import G_LEARNING_ENRICHMENT, run_agent  # noqa: PLC0415
            from deskmate.apps.common import output_dir, write_markdown  # noqa: PLC0415

            pipe = Path(__file__).resolve().parents[1] / "apps" / "user-learning" / "pipe.md"
            end = datetime.now().astimezone()
            start = end - timedelta(hours=max(0.5, float(hours)))
            report = run_agent(
                pipe,
                verbose=verbose,
                # ISO with the default 'T' separator: stored timestamps are
                # 'YYYY-MM-DDTHH:MM:SS+TZ' and every time filter compares them
                # as strings. A ' ' separator sorts BELOW 'T', so a space-form
                # upper bound silently matched nothing and this auto-recap ran
                # with no audio/OCR evidence at all.
                start_time=start.replace(microsecond=0).isoformat(),
                end_time=end.replace(microsecond=0).isoformat(),
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
            if verbose:
                echo_stderr(f"  [learning_flush] wrote {out / 'user-learning.md'}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("user-learning flush failed: %s", exc)
            if verbose:
                echo_stderr(f"  [learning_flush] error: {exc}")

    if background:
        threading.Thread(target=_run, name="user-learning-flush", daemon=True).start()
        return {"ok": True, "queued": True, "hours": hours}
    _run()
    return {"ok": True, "queued": False, "hours": hours}
