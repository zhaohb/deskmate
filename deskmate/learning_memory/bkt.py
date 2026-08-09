"""4-parameter Bayesian Knowledge Tracing + Ebbinghaus-style mastery decay.

Standard BKT (Corbett & Anderson) with a daily exponential decay on p(know).
Used alongside SM-2: BKT answers "how well known?", SM-2 answers "when due?".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime


DEFAULT_PARAMS = {
    "p_init": 0.1,
    "p_learn": 0.3,
    "p_slip": 0.1,
    "p_guess": 0.25,
}

SKILL_PARAMS = {
    "conceptual": {"p_init": 0.15, "p_learn": 0.35, "p_slip": 0.08, "p_guess": 0.25},
    "procedural": {"p_init": 0.05, "p_learn": 0.25, "p_slip": 0.12, "p_guess": 0.20},
}

DECAY_RATE = 0.02  # per day in m * exp(-rate * days)
MASTERY_THRESHOLD = 0.75

# Passive evidence (re-reading slides, re-watching a lecture) is weak proof of
# recall, so it saturates below the recall tier and is only counted once per
# study day. Only a graded review may push p(know) past the ceiling.
PASSIVE_CEILING = 0.60
PASSIVE_MIN_DAYS = 0.5


@dataclass
class BktParams:
    p_init: float = 0.1
    p_learn: float = 0.3
    p_slip: float = 0.1
    p_guess: float = 0.25

    @classmethod
    def for_skill(cls, skill_type: str | None = None) -> BktParams:
        raw = SKILL_PARAMS.get(skill_type or "", DEFAULT_PARAMS)
        return cls(
            p_init=float(raw["p_init"]),
            p_learn=float(raw["p_learn"]),
            p_slip=float(raw["p_slip"]),
            p_guess=float(raw["p_guess"]),
        )


def bkt_update(p_mastery: float, is_correct: bool, params: BktParams | None = None) -> float:
    """One observation update → posterior p(L), then learning transition."""
    params = params or BktParams()
    p_l = max(0.0, min(1.0, float(p_mastery)))
    p_s = params.p_slip
    p_g = params.p_guess
    p_t = params.p_learn

    if is_correct:
        p_obs_l = 1.0 - p_s
        p_obs_not = p_g
    else:
        p_obs_l = p_s
        p_obs_not = 1.0 - p_g

    p_obs = p_obs_l * p_l + p_obs_not * (1.0 - p_l)
    if p_obs < 1e-10:
        p_obs = 1e-10
    p_l_given = (p_obs_l * p_l) / p_obs
    p_new = p_l_given + (1.0 - p_l_given) * p_t
    return max(0.0, min(1.0, p_new))


def apply_mastery_decay(mastery: float, days_since_last: float) -> float:
    """Ebbinghaus-inspired: m * exp(-DECAY_RATE * days)."""
    m = max(0.0, min(1.0, float(mastery)))
    d = max(0.0, float(days_since_last))
    return max(0.0, m * math.exp(-DECAY_RATE * d))


def days_between(earlier: datetime, later: datetime) -> float:
    if earlier.tzinfo is None and later.tzinfo is not None:
        earlier = earlier.replace(tzinfo=later.tzinfo)
    if later.tzinfo is None and earlier.tzinfo is not None:
        later = later.replace(tzinfo=earlier.tzinfo)
    return max(0.0, (later - earlier).total_seconds() / 86400.0)


def seed_mastery_from_evidence(
    *,
    hit_count: int,
    has_definition: bool,
    has_problem: bool,
    has_code: bool,
    skill_type: str | None = None,
) -> float:
    """Map passive study evidence to an initial p(know) without a quiz.

    Definitions / repeats / code practice ≈ soft correct observations;
    on-screen errors ≈ soft incorrect.
    """
    params = BktParams.for_skill(skill_type)
    p = params.p_init
    n_ok = 0
    if has_definition:
        n_ok += 1
    if hit_count >= 2:
        n_ok += 1
    if hit_count >= 4:
        n_ok += 1
    if has_code:
        n_ok += 1
    for _ in range(n_ok):
        p = bkt_update(p, True, params)
    if has_problem:
        p = bkt_update(p, False, params)
    return p


def quality_to_correct(quality: int) -> bool:
    """SM-2 quality 0–5 → binary BKT observation (correct if >= 3)."""
    return int(quality) >= 3


def mastery_urgency(decayed_mastery: float) -> float:
    """Extra urgency when decayed p(know) is below the review threshold."""
    gap = MASTERY_THRESHOLD - max(0.0, min(1.0, decayed_mastery))
    return max(0.0, gap) * 10.0


def infer_skill_type(*, has_code: bool, has_definition: bool) -> str:
    if has_code and not has_definition:
        return "procedural"
    if has_definition:
        return "conceptual"
    return ""


def mastery_tier(decayed_mastery: float) -> str:
    """Nebula-style Exposure→Recall ladder from decayed p(know).

    exposure < recognition < recall < fluent
    """
    p = max(0.0, min(1.0, float(decayed_mastery)))
    if p < 0.25:
        return "exposure"
    if p < 0.50:
        return "recognition"
    if p < 0.75:
        return "recall"
    return "fluent"


def passive_exposure_update(
    p_mastery: float,
    *,
    has_definition: bool,
    has_problem: bool,
    has_code: bool,
    hit_count: int,
    skill_type: str | None = None,
) -> float:
    """Soft BKT observation from passive study evidence (no quiz was taken).

    On-screen errors count as a full incorrect observation and may drop p(know)
    freely; merely reading or repeating counts as correct but saturates at
    ``PASSIVE_CEILING`` so re-watching material never imitates graded recall.
    """
    params = BktParams.for_skill(skill_type)
    p = max(0.0, min(1.0, float(p_mastery)))
    if has_problem:
        return bkt_update(p, False, params)
    if not (has_definition or has_code or hit_count >= 2):
        return p
    return min(bkt_update(p, True, params), max(p, PASSIVE_CEILING))

