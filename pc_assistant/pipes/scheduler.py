"""Background pipe scheduler. Walks `Pipe.frontmatter.interval_seconds` /
`schedule` and executes due pipe bodies through `PipeRuntime`."""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

from ..logger import get
from .loader import Pipe
from .runtime import PipeRuntime

logger = get("pipes.scheduler")


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().replace(microsecond=0).isoformat()


class PipeScheduler:
    def __init__(
        self,
        db,
        pipes: list[Pipe],
        *,
        runtime: PipeRuntime | None = None,
        tick_seconds: float = 1.0,
    ) -> None:
        self.db = db
        self.pipes = pipes
        self.runtime = runtime or PipeRuntime(db)
        self.tick_seconds = tick_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._next_fire: dict[str, float] = {}
        self._init_schedule()

    def _init_schedule(self) -> None:
        now = time.time()
        for p in self.pipes:
            interval = p.frontmatter.interval_seconds
            if interval is None and p.frontmatter.schedule:
                # cron parsing isn't bundled; treat as 5 min default
                interval = 300
            if interval:
                self._next_fire[p.frontmatter.name] = now + interval

    def start(self) -> None:
        if self._thread or not self.pipes:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="PipeScheduler", daemon=True)
        self._thread.start()
        logger.info("pipe scheduler started for %d pipes", len(self.pipes))

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.is_set():
            now = time.time()
            for p in self.pipes:
                name = p.frontmatter.name
                if name not in self._next_fire:
                    continue
                if now >= self._next_fire[name]:
                    self._fire(p)
                    interval = p.frontmatter.interval_seconds or 300
                    self._next_fire[name] = now + interval
            self._stop.wait(self.tick_seconds)

    def _fire(self, pipe: Pipe) -> None:
        self.runtime.run(pipe, trigger="schedule")

    def _record(self, pipe: Pipe, *, status: str, output: str) -> None:
        try:
            execution_id = self.db.insert_pipe_execution(
                pipe_name=pipe.frontmatter.name,
                status="running",
                started_at=_now_iso(),
            )
            self.db.finish_pipe_execution(execution_id, status=status, output=output)
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to record pipe execution: %s", exc)
