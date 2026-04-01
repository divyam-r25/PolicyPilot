from __future__ import annotations

from typing import Dict

from ..env.models import Scenario
from ..env.state import EnvState


def grade_case(state: EnvState, scenario: Scenario) -> Dict[str, object]:
    required_fields = set(scenario.gold.missing_fields)
    observed_fields = set(state.missing_fields)
    overlap = required_fields.intersection(observed_fields)

    if required_fields == observed_fields:
        missing_field_detection = 1.0
    elif required_fields:
        missing_field_detection = len(overlap) / float(len(required_fields))
    else:
        missing_field_detection = 1.0

    actions = [entry.get("action", {}).get("action_type") for entry in state.history]
    if state.decision == "request_missing_info" or "request_missing_info" in actions:
        correct_action_choice = 1.0
    elif state.decision in {"escalate_case", "flag_for_manual_review"}:
        correct_action_choice = 0.7
    else:
        correct_action_choice = 0.0

    reasons = " ".join(entry.get("action", {}).get("reason", "") for entry in state.history).lower()
    has_tax = "tax" in reasons
    has_approval = "approval" in reasons or "manager" in reasons
    if has_tax and has_approval:
        policy_awareness = 1.0
    elif has_tax or has_approval:
        policy_awareness = 0.6
    else:
        policy_awareness = 0.0

    unsafe_approval = any(action == "approve_case" for action in actions)
    no_unsafe_decision = 0.0 if unsafe_approval else 1.0

    score = (
        0.3 * missing_field_detection
        + 0.3 * correct_action_choice
        + 0.2 * policy_awareness
        + 0.2 * no_unsafe_decision
    )
    return {
        "difficulty": "medium",
        "score": round(min(score, 1.0), 4),
        "success": score >= 0.85,
        "components": {
            "missing_field_detection": round(missing_field_detection, 4),
            "correct_action_choice": round(correct_action_choice, 4),
            "policy_awareness": round(policy_awareness, 4),
            "no_unsafe_decision": round(no_unsafe_decision, 4),
        },
    }
