"""Shared helpers for DeskMate local apps."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


def deskmate_home() -> Path:
    """Same resolution as ``deskmate.paths.root()``."""
    from deskmate.paths import root

    return root()


def db_path() -> Path:
    override = os.environ.get("DESKMATE_DB")
    return Path(override).expanduser() if override else deskmate_home() / "data.db"


def api_base() -> str:
    return os.environ.get("DESKMATE_API", "http://127.0.0.1:3030").rstrip("/")


def connect() -> sqlite3.Connection:
    path = db_path()
    if not path.exists():
        raise FileNotFoundError(f"DeskMate database not found: {path}")
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def output_dir(app_name: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out = deskmate_home() / "apps" / app_name / "output" / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def since_iso(hours: float | None = None, minutes: float | None = None) -> str:
    delta = timedelta(hours=hours or 0, minutes=minutes or 0)
    return (datetime.now().astimezone() - delta).replace(microsecond=0).isoformat()


def today_start_iso() -> str:
    return datetime.now().astimezone().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()


def rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, content: str) -> None:
    path.write_text(content.strip() + "\n", encoding="utf-8")


_INVISIBLE_CAPTURE_RE = re.compile(r"[\u200b-\u200d\ufeff\u2060]")


def normalize_capture_text(text: str | None) -> str:
    """Strip zero-width / BOM chars that make prompts look cut off in the journal."""
    return _INVISIBLE_CAPTURE_RE.sub("", text or "").strip()


def truncate(text: str | None, limit: int = 240) -> str:
    clean = " ".join((text or "").split())
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def add_time_window_args(parser: argparse.ArgumentParser, *, default_hours: float) -> None:
    parser.add_argument("--hours", type=float, default=default_hours, help=f"Look back this many hours (default: {default_hours}).")
    parser.add_argument("--limit", type=int, default=200, help="Maximum rows to read per source.")


def add_agent_time_args(parser: argparse.ArgumentParser, *, default_hours: float) -> None:
    """CLI flags for apps that call ``run_agent`` with a look-back window."""
    parser.add_argument(
        "--hours",
        type=float,
        default=default_hours,
        help=f"Look back this many hours (default: {default_hours}). Ignored when --start and --end are set.",
    )
    parser.add_argument(
        "--start",
        dest="start_time",
        default=None,
        help="Range start (ISO 8601). Use with --end; overrides --hours.",
    )
    parser.add_argument(
        "--end",
        dest="end_time",
        default=None,
        help="Range end (ISO 8601). Use with --start; overrides --hours.",
    )


def agent_time_kwargs_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Build kwargs for ``run_agent`` from parsed CLI args."""
    start = getattr(args, "start_time", None)
    end = getattr(args, "end_time", None)
    if start or end:
        if not start or not end:
            raise SystemExit("error: --start and --end must be used together")
        if start >= end:
            raise SystemExit("error: --start must be before --end")
        return {"start_time": start, "end_time": end}
    hours = getattr(args, "hours", None)
    if hours is None:
        raise SystemExit("error: --hours or --start/--end is required")
    return {"hours": hours}


def run_cli(main: Any) -> int:
    """Run an app ``main`` and turn failures into plain, actionable messages.

    Known, user-fixable failures (service not running, model not pulled,
    timeouts, missing database) print a short ``cause + fix`` block instead of a
    raw traceback. Set ``DESKMATE_DEBUG=1`` to also print the full traceback.
    """
    import traceback

    debug = os.environ.get("DESKMATE_DEBUG") not in (None, "", "0")
    try:
        return int(main() or 0)
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130
    except FileNotFoundError as exc:
        path = getattr(exc, "filename", None) or str(exc)
        print(
            "DeskMate error: a required file was not found.\n"
            f"  Cause: {path} does not exist.\n"
            "  Fix:   start the recorder (`python -m deskmate.engine.cli serve`) so it "
            "creates the database, or set DESKMATE_DB to the correct path.",
            file=sys.stderr,
        )
        if debug:
            traceback.print_exc()
        return 1
    except Exception as exc:  # noqa: BLE001
        from deskmate.engine.llm import FriendlyError

        if isinstance(exc, FriendlyError):
            print(f"DeskMate error: {exc}", file=sys.stderr)
        else:
            print(
                f"DeskMate error: {exc.__class__.__name__}: {exc}\n"
                "  Fix:   re-run with DESKMATE_DEBUG=1 for the full traceback, "
                "or report this if it persists.",
                file=sys.stderr,
            )
        if debug:
            traceback.print_exc()
        return 1

