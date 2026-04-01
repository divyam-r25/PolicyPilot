from __future__ import annotations

from typing import Dict

from ..env.models import Scenario
from ..env.state import EnvState


def grade_case(state: EnvState, scenario: Scenario) -> Dict[str, object]:
    policy_interpretation = 1.0 if state.policy_used_correctly else 0.5 if state.policy_trace else 0.0

    conflict_flags = [
        bool(entry.get("analysis", {}).get("conflict_detected"))
        for entry in state.history
    ]
    if any(conflict_flags):
        conflict_resolution = (
            1.0 if state.decision in {"escalate_case", "flag_for_manual_review"} else 0.4
        )
    else:
        conflict_resolution = 1.0

    if state.decision == scenario.gold.decision:
        final_decision = 1.0
    elif state.decision == "flag_for_manual_review":
        final_decision = 0.7
    else:
        final_decision = 0.0

    invalid_actions = sum(
        1 for entry in state.history if not entry.get("validation", {}).get("is_valid", True)
    )
    workflow_correctness = 1.0 if invalid_actions == 0 and state.step_count <= state.max_steps else 0.0

    reasons = " ".join(entry.get("action", {}).get("reason", "") for entry in state.history).lower()
    has_audit_keywords = any(
        keyword in reasons for keyword in ("policy", "hierarchy", "escalate", "receipt", "approval", "threshold")
    )
    audit_compliance = (
        1.0 if has_audit_keywords and (state.audit_notes or reasons.strip()) else 0.3 if reasons.strip() else 0.0
    )

    score = (
        0.2 * policy_interpretation
        + 0.2 * conflict_resolution
        + 0.2 * final_decision
        + 0.2 * workflow_correctness
        + 0.2 * audit_compliance
    )
    return {
        "difficulty": "hard",
        "score": round(min(score, 1.0), 4),
        "success": score >= 0.85,
        "components": {
            "policy_interpretation": round(policy_interpretation, 4),
            "conflict_resolution": round(conflict_resolution, 4),
            "final_decision": round(final_decision, 4),
            "workflow_correctness": round(workflow_correctness, 4),
            "audit_compliance": round(audit_compliance, 4),
        },
    }
