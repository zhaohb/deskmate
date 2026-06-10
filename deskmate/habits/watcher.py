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
            # Retire superseded rules BEFORE seeding so a stale break_reminder
            # doesn't linger, then seed the current 3-tier default set.
            self._store.delete_rules(rules_mod.RETIRED_RULE_NAMES)
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
        # Store setting is authoritative (the UI writes it there so daemon + API
        # agree without a restart); fall back to the config default.
        lang = store.get_setting(
            "reminder_lang", getattr(self._hcfg, "reminder_lang", "zh")
        ) or "zh"

        notifier = Notifier(
            store,
            daily_quota=self._hcfg.daily_quota,
            toast_enabled=self._hcfg.toast_enabled,
            toast_title="DeskMate 小助手" if lang == "zh" else "DeskMate",
        )

        # Interruptibility: don't nudge during a meeting / presentation / DND.
        # Fail-open — any probe error leaves busy_reason None (reminders fire).
        busy_reason: str | None = None
        if getattr(self._hcfg, "respect_presence", True):
            try:
                from . import presence as presence_mod  # noqa: PLC0415

                in_meeting = store.has_open_meeting(now.replace(microsecond=0).isoformat())
                busy_reason = presence_mod.busy_reason(in_meeting=in_meeting)
            except Exception as exc:  # noqa: BLE001
                logger.debug("presence check failed: %s", exc)
                busy_reason = None

        # At most one suggestion per tick. The first rule that is actually
        # *delivered* wins — NOT merely the first rule that hits. A rule can hit
        # on every tick (e.g. break_reminder once screen time passes its
        # threshold) yet be in its own cooldown; that must not starve a
        # lower-priority rule that is ready to fire. So a per-rule cooldown skip
        # falls through to the next rule. Global gates (quiet hours / daily
        # quota) suppress every rule alike, so we stop as soon as one hits them.
        # Global suppressors stop ALL rules this tick (no point trying the rest):
        # the master switch, presence (meeting/fullscreen/DND), quiet hours, and
        # the daily quota gate every rule alike. Per-rule skips (cooldown, snooze,
        # auto-disabled) fall through so a ready lower-priority rule can fire.
        global_reasons = ("globally_disabled", "in_meeting", "fullscreen",
                          "focus_assist", "quiet_hours", "daily_quota")
        skipped: list[dict[str, Any]] = []
        for rule in store.enabled_rules():
            hit, message, ctx = rules_mod.evaluate_rule(rule, state, profiles, now, lang)
            if not hit:
                continue
            result = notifier.deliver(rule, message, ctx, now, busy_reason=busy_reason)
            status, reason = result.get("status"), result.get("reason")
            if status == "sent":
                return {"status": "evaluated", "fired": rule["name"], "result": result}
            if status == "suppressed" and reason in global_reasons:
                return {"status": "evaluated", "fired": None, "result": result}
            # cooldown or auto-disabled → this rule isn't deliverable now; let a
            # ready lower-priority rule have its turn.
            skipped.append(result)
        return {"status": "evaluated", "fired": None, "skipped": skipped}
