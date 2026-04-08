from __future__ import annotations


def clamp_task_score(score: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """
    Clamp task score into a valid closed interval.

    The benchmark expects scores in [0, 1]. This helper keeps that contract
    centralized so runner/API code cannot emit out-of-range values.
    """
    bounded = max(lower, min(score, upper))
    return round(bounded, 4)
