from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

ALLOWED_ACTIONS: tuple[str, ...] = (
    "approve_case",
    "reject_case",
    "request_missing_info",
    "escalate_case",
    "flag_for_manual_review",
    "add_audit_note",
)

FINAL_ACTIONS: set[str] = {
    "approve_case",
    "reject_case",
    "escalate_case",
    "flag_for_manual_review",
}


@dataclass(slots=True)
class TaskMetadata:
    name: str = "compliance_review"
    difficulty: str = "easy"
    step: int = 0
    max_steps: int = 8


@dataclass(slots=True)
class CaseData:
    id: str
    type: str
    amount: float
    currency: str = "USD"
    line_items: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DocumentData:
    type: str
    fields: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PolicyConfig:
    max_meal: float = 75.0
    requires_tax_breakdown_above: float = 300.0
    requires_manager_approval_above: float = 500.0
    require_full_scope_approval_above: float = 500.0
    escalate_above: float = 500.0
    reject_on_missing_docs: bool = False
    mixed_expense_requires_escalation: bool = True
    escalate_on_duplicate_claim: bool = True
    pending_exception_requires_escalation: bool = True
    require_fx_rate_for_non_usd: bool = True
    rule_hierarchy: List[str] = field(
        default_factory=lambda: [
            "reject_fraudulent_document",
            "flag_duplicate_claim",
            "escalate_above_threshold",
            "escalate_pending_exception",
            "escalate_mixed_expense",
            "request_missing_info_first",
            "reject_meal_violation",
            "reject_missing_docs",
            "approve_clean_case",
        ]
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ObservationState:
    missing_fields: List[str] = field(default_factory=list)
    decision: Optional[str] = None
    reviewed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Observation:
    task: TaskMetadata
    case: CaseData
    documents: List[DocumentData]
    policy: PolicyConfig
    state: ObservationState
    allowed_actions: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": asdict(self.task),
            "case": self.case.to_dict(),
            "documents": [doc.to_dict() for doc in self.documents],
            "policy": self.policy.to_dict(),
            "state": self.state.to_dict(),
            "allowed_actions": list(self.allowed_actions),
        }


@dataclass(slots=True)
class GoldStandard:
    decision: str
    missing_fields: List[str] = field(default_factory=list)
    policy_refs: List[str] = field(default_factory=list)
    safe_actions: List[str] = field(default_factory=list)
    unsafe_actions: List[str] = field(default_factory=list)


@dataclass(slots=True)
class Scenario:
    id: str
    name: str
    difficulty: str
    case: CaseData
    documents: List[DocumentData]
    policy: PolicyConfig
    allowed_actions: List[str]
    max_steps: int
    gold: GoldStandard
    description: str = ""

    def to_observation(
        self,
        step: int,
        missing_fields: Optional[List[str]] = None,
        decision: Optional[str] = None,
        reviewed: bool = False,
    ) -> Observation:
        return Observation(
            task=TaskMetadata(
                name=self.name,
                difficulty=self.difficulty,
                step=step,
                max_steps=self.max_steps,
            ),
            case=self.case,
            documents=self.documents,
            policy=self.policy,
            state=ObservationState(
                missing_fields=missing_fields or [],
                decision=decision,
                reviewed=reviewed,
            ),
            allowed_actions=self.allowed_actions,
        )


@dataclass(slots=True)
class Action:
    action_type: str
    case_id: str
    payload: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
