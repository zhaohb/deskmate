"""SM-2 spaced repetition (SuperMemo 2) for learning concept reviews.

Quality scale (0–5) follows the classic SM-2 convention. New concepts are seeded
from evidence strength rather than a quiz (no interactive grade yet).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class Sm2State:
    ease_factor: float = 2.5
    interval_days: float = 0.0
    repetitions: int = 0


def sm2_update(state: Sm2State, quality: int) -> Sm2State:
    """Return next SM-2 state after a review graded ``quality`` (0..5)."""
    q = max(0, min(5, int(quality)))
    ef = state.ease_factor
    reps = state.repetitions
    interval = state.interval_days

    if q < 3:
        reps = 0
        interval = 1.0
    else:
        if reps == 0:
            interval = 1.0
        elif reps == 1:
            interval = 6.0
        else:
            interval = max(1.0, interval * ef)
        reps += 1

    ef = ef + (0.1 - (5 - q) * (0.08 + (5 - q) * 0.02))
    if ef < 1.3:
        ef = 1.3

    return Sm2State(ease_factor=ef, interval_days=interval, repetitions=reps)


def due_at_after(state: Sm2State, *, now: datetime | None = None) -> datetime:
    now = now or datetime.now().astimezone()
    days = max(0.0, float(state.interval_days))
    return now + timedelta(days=days)


def seed_quality(
    *,
    hit_count: int,
    has_definition: bool,
    has_problem: bool,
    has_code: bool,
) -> int:
    """Map passive evidence to an initial SM-2 quality without a quiz.

    Problems pull quality down (needs review sooner); definitions + repeated
    hits push it up.
    """
    q = 3
    if has_definition:
        q += 1
    if hit_count >= 3:
        q += 1
    if has_code:
        q += 0  # practice seen, keep baseline
    if has_problem:
        q -= 2
    return max(1, min(5, q))


def urgency_score(*, due_at: datetime, now: datetime | None = None, ease: float = 2.5) -> float:
    """Higher = more urgent. Overdue cards outrank future ones; low ease boosts."""
    now = now or datetime.now().astimezone()
    if due_at.tzinfo is None:
        due_at = due_at.replace(tzinfo=now.tzinfo)
    hours = (now - due_at).total_seconds() / 3600.0
    return hours + max(0.0, (2.5 - ease) * 6.0)
