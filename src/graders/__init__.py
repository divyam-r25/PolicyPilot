from __future__ import annotations

from typing import Callable, Dict

from ..env.models import Scenario
from ..env.state import EnvState
from .common import clamp_task_score
from .easy import grade_case as grade_easy_case
from .hard import grade_case as grade_hard_case
from .medium import grade_case as grade_medium_case

GRADE_FUNCTIONS: Dict[str, Callable[[EnvState, Scenario], Dict[str, object]]] = {
    "easy": grade_easy_case,
    "medium": grade_medium_case,
    "hard": grade_hard_case,
}


def grade_episode(state: EnvState, scenario: Scenario) -> Dict[str, object]:
    difficulty = scenario.difficulty
    if difficulty not in GRADE_FUNCTIONS:
        raise ValueError(f"No grader registered for difficulty '{difficulty}'.")
    result = GRADE_FUNCTIONS[difficulty](state, scenario)
    score = max(0.0, min(float(result["score"]), 1.0))
    result["score"] = round(score, 4)
    if "components" in result and "subscores" not in result:
        result["subscores"] = dict(result["components"])
    result["success_threshold"] = 0.85
    result["success"] = score >= 0.85
    return result


__all__ = ["grade_episode", "clamp_task_score"]
