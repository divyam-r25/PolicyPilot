from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .models import ALLOWED_ACTIONS, Action


@dataclass(slots=True)
class ValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    action: Optional[Action] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": list(self.errors),
            "action": self.action.to_dict() if self.action else None,
        }


class ActionValidator:
    required_fields = {"action_type", "case_id", "payload", "reason"}

    def validate(
        self,
        raw_action: Union[Action, Dict[str, Any]],
        case_id: str,
        allowed_actions: List[str],
    ) -> ValidationResult:
        if isinstance(raw_action, Action):
            data: Dict[str, Any] = raw_action.to_dict()
        elif isinstance(raw_action, dict):
            data = raw_action
        else:
            return ValidationResult(
                is_valid=False,
                errors=["Action must be an object or Action dataclass instance."],
            )

        missing_top_level = sorted(self.required_fields.difference(data.keys()))
        errors: List[str] = []
        if missing_top_level:
            errors.append(f"Missing top-level action fields: {missing_top_level}.")

        action_type = data.get("action_type")
        action_case_id = data.get("case_id")
        payload = data.get("payload")
        reason = data.get("reason")

        if action_type not in ALLOWED_ACTIONS:
            errors.append(f"Unsupported action_type '{action_type}'.")
        elif action_type not in allowed_actions:
            errors.append(f"Action '{action_type}' is not allowed in this step.")

        if action_case_id != case_id:
            errors.append(f"case_id mismatch: expected '{case_id}', received '{action_case_id}'.")

        if not isinstance(payload, dict):
            errors.append("payload must be an object.")

        if not isinstance(reason, str) or not reason.strip():
            errors.append("reason must be a non-empty string.")

        if isinstance(payload, dict):
            if action_type == "request_missing_info":
                fields = payload.get("fields")
                if not isinstance(fields, list) or not fields:
                    errors.append("request_missing_info requires payload.fields as a non-empty list.")
                elif not all(isinstance(field_name, str) and field_name for field_name in fields):
                    errors.append("payload.fields must contain non-empty strings only.")

            if action_type == "add_audit_note":
                note = payload.get("note")
                if note is not None and not isinstance(note, str):
                    errors.append("add_audit_note payload.note must be a string when supplied.")

        if errors:
            return ValidationResult(is_valid=False, errors=errors)

        normalized = Action(
            action_type=str(action_type),
            case_id=str(action_case_id),
            payload=payload if isinstance(payload, dict) else {},
            reason=reason.strip(),
        )
        return ValidationResult(is_valid=True, errors=[], action=normalized)

    @staticmethod
    def action_signature(action: Optional[Action]) -> Optional[str]:
        if action is None:
            return None
        stable_payload = json.dumps(action.payload, sort_keys=True, separators=(",", ":"))
        return f"{action.action_type}|{stable_payload}"
