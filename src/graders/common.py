from __future__ import annotations

SCORE_EPSILON = 0.0001


def clamp_task_score(score: float, *, precision: int = 4) -> float:
    bounded = max(SCORE_EPSILON, min(float(score), 1.0 - SCORE_EPSILON))
    return round(bounded, precision)
