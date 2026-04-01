from __future__ import annotations

from typing import Dict

from ..env.models import Scenario
from ..env.state import EnvState


def grade_case(state: EnvState, scenario: Scenario) -> Dict[str, object]:
    actions = [entry.get("action", {}).get("action_type") for entry in state.history]
    reasons = [entry.get("action", {}).get("reason", "") for entry in state.history]

    violation_detection = (
        1.0
        if any(action in {"reject_case", "escalate_case", "request_missing_info"} for action in actions)
        else 0.0
    )

    if state.decision == scenario.gold.decision:
        correct_action = 1.0
    elif state.decision == "escalate_case":
        correct_action = 0.7
    else:
        correct_action = 0.0

    reason_text = " ".join(reasons).lower()
    if any(keyword in reason_text for keyword in ("meal", "limit", "policy")):
        correct_reason = 1.0
    elif reason_text.strip():
        correct_reason = 0.5
    else:
        correct_reason = 0.0

    score = (0.4 * violation_detection) + (0.4 * correct_action) + (0.2 * correct_reason)
    return {
        "difficulty": "easy",
        "score": round(min(score, 1.0), 4),
        "success": score >= 0.85,
        "components": {
            "correct_violation_detection": round(violation_detection, 4),
            "correct_action": round(correct_action, 4),
            "correct_reason": round(correct_reason, 4),
        },
    }
