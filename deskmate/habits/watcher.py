"""HabitWatcher — background daemon thread tying the three layers together.

* Periodically re-mines ``habit_profiles`` (data → patterns).
* Every ``tick_interval`` evaluates current activity against learned routines
  (patterns + present → timing) and routes any hit through the Notifier
  (suggestion → user).

Owns its own :class:`HabitStore` connection and a single thread, mirroring the
start()/stop() lifecycle of the other daemon components. Fully additive: if
``cfg.habits.enabled`` is false the daemon never constructs it.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any

from ..config import Config
from ..logger import get
from . import rules as rules_mod
from .miner import HabitMiner
from .notifier import Notifier
from .store import HabitStore

logger = get("habits.watcher")


class HabitWatcher:
    def __init__(self, cfg: Config, *, stop: threading.Event | None = None) -> None:
        self.cfg = cfg
        self._hcfg = cfg.habits
        self._stop = stop or threading.Event()
        self._owns_stop = stop is None
        self._thread: threading.Thread | None = None
        self._store: HabitStore | None = None
        self._last_mine: datetime | None = None

    # ─── lifecycle ───────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None:
            return
        self._store = HabitStore()
        try:
            self._store.ensure_rules(rules_mod.DEFAULT_RULES)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not seed default habit rules: %s", exc)
        self._thread = threading.Thread(target=self._loop, name="daemon-habits", daemon=True)
        self._thread.start()
        logger.info("habit watcher started (tick=%dm)", self._hcfg.tick_interval_min)

    def stop(self) -> None:
        if self._owns_stop:
            self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=3.0)
        if self._store is not None:
            self._store.close()
            self._store = None
        self._thread = None

    # ─── loop ────────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        # Mine once shortly after start so the UI has data; then on the tick.
        self._safe_mine()
        interval_sec = max(60, self._hcfg.tick_interval_min * 60)
        while not self._stop.is_set():
            if self._stop.wait(interval_sec):
                break
            self._maybe_mine()
            self._safe_tick()

    def _maybe_mine(self) -> None:
        now = datetime.now().astimezone()
        if self._last_mine is None:
            self._safe_mine()
            return
        elapsed_h = (now - self._last_mine).total_seconds() / 3600.0
        if elapsed_h >= self._hcfg.mine_interval_hours:
            self._safe_mine()

    def _safe_mine(self) -> None:
        if self._store is None:
            return
        try:
            HabitMiner(
                self._store,
                lookback_days=self._hcfg.mine_lookback_days,
                min_frequency=self._hcfg.min_frequency,
                min_sample_days=self._hcfg.min_sample_days,
            ).mine()
            self._last_mine = datetime.now().astimezone()
        except Exception as exc:  # noqa: BLE001
            logger.warning("habit mining failed: %s", exc)

    def _safe_tick(self) -> None:
        try:
            self._tick()
        except Exception as exc:  # noqa: BLE001
            logger.warning("habit tick failed: %s", exc)

    def _tick(self) -> dict[str, Any]:
        store = self._store
        if store is None:
            return {"status": "no_store"}
        now = datetime.now().astimezone()

        state = rules_mod.read_current_state(store, now, window_minutes=self.cfg.habits.tick_interval_min)
        if state is None:
            return {"status": "no_activity"}

        day_type, slot = rules_mod.current_slot_key(now)
        profiles = store.profiles_for_slot(day_type, slot)

        notifier = Notifier(
            store,
            daily_quota=self._hcfg.daily_quota,
            toast_enabled=self._hcfg.toast_enabled,
        )

        # Highest-priority hit wins; at most one suggestion per tick.
        for rule in store.enabled_rules():
            hit, message, ctx = rules_mod.evaluate_rule(rule, state, profiles, now)
            if hit:
                result = notifier.deliver(rule, message, ctx, now)
                return {"status": "evaluated", "fired": rule["name"], "result": result}
        return {"status": "evaluated", "fired": None}
