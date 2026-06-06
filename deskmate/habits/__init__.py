"""Additive habits module: learn routines, evaluate the present, suggest proactively.

This package is self-contained and does not modify any existing DeskMate
behavior. It reads from the existing ``frames`` / ``ui_events`` tables and
writes only to the ``habit_*`` tables added to the schema.

Components
----------
* :class:`~deskmate.habits.store.HabitStore` — data access (own SQLite handle).
* :class:`~deskmate.habits.miner.HabitMiner` — frames → ``habit_profiles``.
* :mod:`~deskmate.habits.rules` — pure rule evaluation + defaults.
* :class:`~deskmate.habits.notifier.Notifier` — dedup / quiet hours / channels.
* :class:`~deskmate.habits.watcher.HabitWatcher` — background daemon thread.
"""

from __future__ import annotations

from .miner import HabitMiner
from .notifier import Notifier
from .store import HabitStore
from .watcher import HabitWatcher

__all__ = ["HabitStore", "HabitMiner", "Notifier", "HabitWatcher"]
