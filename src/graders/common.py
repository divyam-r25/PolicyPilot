from __future__ import annotations

SCORE_EPSILON = 0.0001


def clamp_task_score(score: float, *, precision: int = 4) -> float:
    safe_precision = max(int(precision), 1)
    rounding_step = 10 ** (-safe_precision)
    epsilon = max(SCORE_EPSILON, rounding_step)
    bounded = max(epsilon, min(float(score), 1.0 - epsilon))
    rounded = round(bounded, safe_precision)

    # Guard against a rounded boundary value (for example precision=2 -> 1.00).
    if rounded <= 0.0:
        return round(epsilon, safe_precision)
    if rounded >= 1.0:
        return round(1.0 - epsilon, safe_precision)
    return rounded
