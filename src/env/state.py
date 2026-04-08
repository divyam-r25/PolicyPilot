from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(slots=True)
class EnvState:
    case_id: str
    difficulty: str
    step_count: int
    max_steps: int
    decision: Optional[str] = None
    missing_fields: List[str] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    reward_accumulated: float = 0.0
    done: bool = False
    critical_failure: bool = False
    policy_used_correctly: bool = False
    reviewed: bool = False
    audit_notes: List[str] = field(default_factory=list)
    policy_trace: List[str] = field(default_factory=list)
    expected_decision: Optional[str] = None
    expected_missing_fields: List[str] = field(default_factory=list)
    last_action_signature: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
