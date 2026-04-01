from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .actions import ActionValidator, ValidationResult
from .models import Action, Scenario
from .policy_engine import PolicyAnalysis
from .state import EnvState


@dataclass(slots=True)
class RewardResult:
    reward: float
    components: Dict[str, float]
    penalties: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "reward": self.reward,
            "components": dict(self.components),
            "penalties": list(self.penalties),
        }


class RewardEngine:
    VIOLATION_WEIGHT = 0.25
    EVIDENCE_WEIGHT = 0.25
    DECISION_WEIGHT = 0.30
    AUDIT_WEIGHT = 0.20

    def calculate(
        self,
        state: EnvState,
        scenario: Scenario,
        validation: ValidationResult,
        analysis: PolicyAnalysis,
        validator: ActionValidator,
    ) -> RewardResult:
        if not validation.is_valid or validation.action is None:
            return RewardResult(
                reward=-0.2,
                components={
                    "violation_detection": 0.0,
                    "evidence_handling": 0.0,
                    "decision_correctness": 0.0,
                    "audit_quality": 0.0,
                },
                penalties=["invalid_action(-0.2)"],
            )

        action = validation.action
        violation_detection = self._violation_detection_score(action, analysis)
        evidence_handling = self._evidence_score(action, analysis)
        decision_correctness = self._decision_score(action, analysis, scenario)
        audit_quality = self._audit_quality_score(action)

        base_reward = (
            self.VIOLATION_WEIGHT * violation_detection
            + self.EVIDENCE_WEIGHT * evidence_handling
            + self.DECISION_WEIGHT * decision_correctness
            + self.AUDIT_WEIGHT * audit_quality
        )

        penalties: List[str] = []
        penalty_value = 0.0

        if action.action_type == "approve_case" and not analysis.safe_to_approve:
            penalty_value -= 0.5
            penalties.append("unsafe_approval(-0.5)")

        signature = validator.action_signature(action)
        if (
            state.last_action_signature
            and state.last_action_signature == signature
            and action.action_type in {"request_missing_info", "add_audit_note"}
        ):
            penalty_value -= 0.1
            penalties.append("repeated_useless_action(-0.1)")

        required_steps_not_done = bool(analysis.required_missing_fields) and (
            analysis.recommended_action == "request_missing_info"
        )
        if required_steps_not_done and action.action_type in {"approve_case", "reject_case"}:
            penalty_value -= 0.3
            penalties.append("skipped_required_step(-0.3)")

        total_reward = round(base_reward + penalty_value, 4)
        return RewardResult(
            reward=total_reward,
            components={
                "violation_detection": round(violation_detection, 4),
                "evidence_handling": round(evidence_handling, 4),
                "decision_correctness": round(decision_correctness, 4),
                "audit_quality": round(audit_quality, 4),
            },
            penalties=penalties,
        )

    @staticmethod
    def _violation_detection_score(action: Action, analysis: PolicyAnalysis) -> float:
        issues_present = bool(analysis.detected_violations or analysis.required_missing_fields)
        if issues_present and action.action_type in {
            "reject_case",
            "request_missing_info",
            "escalate_case",
            "flag_for_manual_review",
        }:
            return 1.0
        if issues_present and action.action_type == "approve_case":
            return 0.0
        if not issues_present and action.action_type == "approve_case":
            return 1.0
        return 0.4

    @staticmethod
    def _evidence_score(action: Action, analysis: PolicyAnalysis) -> float:
        required = set(analysis.required_missing_fields)
        if not required:
            return 1.0 if action.action_type != "request_missing_info" else 0.5

        if action.action_type == "request_missing_info":
            provided = set(action.payload.get("fields", []))
            if not provided:
                return 0.0
            if required.issubset(provided):
                return 1.0
            return len(required.intersection(provided)) / float(len(required))

        if action.action_type in {"escalate_case", "flag_for_manual_review"}:
            if analysis.recommended_action in {"escalate_case", "flag_for_manual_review"}:
                return 0.9
            return 0.4

        return 0.0

    @staticmethod
    def _decision_score(action: Action, analysis: PolicyAnalysis, scenario: Scenario) -> float:
        if action.action_type == scenario.gold.decision:
            return 1.0
        if action.action_type == analysis.recommended_action:
            return 0.8
        if action.action_type == "approve_case" and not analysis.safe_to_approve:
            return 0.0
        if action.action_type == "add_audit_note":
            return 0.4
        return 0.2

    @staticmethod
    def _audit_quality_score(action: Action) -> float:
        reason = action.reason.strip().lower()
        if not reason:
            return 0.0

        score = 0.2
        policy_keywords = (
            "policy",
            "threshold",
            "approval",
            "tax",
            "receipt",
            "limit",
            "escalate",
            "compliance",
        )
        if any(keyword in reason for keyword in policy_keywords):
            score += 0.4
        if len(reason) >= 30:
            score += 0.4
        return min(score, 1.0)
