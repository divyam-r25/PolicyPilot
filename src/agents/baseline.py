from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple


class BaselineComplianceAgent:
    """
    Baseline deterministic policy-aware agent.

    Logging format (strict):
    POLICYPILOT_LOG|<iso_timestamp>|case=<case_id>|action=<action_type>|policy_ref=<policy_ref>|reason=<reason>
    """

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        action, _ = self.act_with_log(observation)
        return action

    def act_with_log(self, observation: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        case = observation.get("case", {})
        policy = observation.get("policy", {})
        documents = observation.get("documents", [])
        case_id = str(case.get("id", "UNKNOWN"))
        amount = float(case.get("amount", 0.0))
        line_items = case.get("line_items", [])

        receipt_fields = [
            doc.get("fields", {}) for doc in documents if doc.get("type") == "receipt"
        ]
        all_fields = [doc.get("fields", {}) for doc in documents]

        missing_fields: List[str] = []
        tax_threshold = float(policy.get("requires_tax_breakdown_above", 300.0))
        approval_threshold = float(policy.get("requires_manager_approval_above", 500.0))
        escalate_threshold = float(policy.get("escalate_above", 500.0))
        max_meal = float(policy.get("max_meal", 75.0))
        reject_on_missing_docs = bool(policy.get("reject_on_missing_docs", False))
        hierarchy = list(policy.get("rule_hierarchy", []))

        has_tax_breakdown = any(
            ("tax" in fields and fields.get("tax") is not None)
            or fields.get("tax_breakdown")
            for fields in receipt_fields
        )
        has_manager_approval = any(
            fields.get("manager_approval") in {True, "approved", "APPROVED"}
            or fields.get("approval_status") in {"approved", "APPROVED"}
            for fields in all_fields
        )
        has_partial_receipt = any(fields.get("is_partial") is True for fields in receipt_fields)

        if amount > tax_threshold and not has_tax_breakdown:
            missing_fields.append("receipt_tax")
        if amount > approval_threshold and not has_manager_approval:
            missing_fields.append("manager_approval")
        if reject_on_missing_docs and has_partial_receipt:
            missing_fields.append("itemized_receipt")

        meal_violation = any(
            "meal" in str(item.get("category", "")).lower()
            and float(item.get("amount", 0.0)) > max_meal
            for item in line_items
        )
        categories = {str(item.get("category", "")).lower() for item in line_items}
        mixed_expense = any("personal" in c for c in categories) and any(
            "business" in c or "travel" in c or "meal" in c for c in categories
        )

        action_type = "approve_case"
        payload: Dict[str, Any] = {}
        reason = "Case appears compliant under current policy."
        policy_ref = "approve_clean_case"

        if mixed_expense and amount > escalate_threshold:
            action_type = "escalate_case"
            reason = (
                "Conflicting risk signals detected (mixed personal/business plus high amount); escalating per hierarchy."
            )
            policy_ref = "escalate_above_threshold"
        elif amount > escalate_threshold and reject_on_missing_docs and missing_fields:
            action_type = "escalate_case"
            reason = "High-value case with missing documentation requires escalation under policy hierarchy."
            policy_ref = "escalate_above_threshold"
        elif missing_fields:
            action_type = "request_missing_info"
            payload = {"fields": sorted(set(missing_fields))}
            reason = "Required evidence is missing for policy thresholds; requesting fields before any approval."
            policy_ref = "request_missing_info_first"
        elif meal_violation:
            action_type = "reject_case"
            reason = "Meal amount exceeds policy cap and no override is present."
            policy_ref = "reject_meal_violation"
        elif mixed_expense:
            action_type = "escalate_case"
            reason = "Mixed personal/business claim requires manual escalation."
            policy_ref = "escalate_mixed_expense"

        if hierarchy:
            # Keep the baseline deterministic with explicit hierarchy fallback.
            if action_type == "request_missing_info" and "request_missing_info_first" not in hierarchy:
                action_type = "escalate_case"
                payload = {}
                reason = "Escalating because policy hierarchy does not prioritize missing-info requests."
                policy_ref = "escalate_above_threshold"

        action = {
            "action_type": action_type,
            "case_id": case_id,
            "payload": payload,
            "reason": reason,
        }
        log_line = self._build_log_line(case_id, action_type, policy_ref, reason)
        return action, log_line

    @staticmethod
    def _build_log_line(case_id: str, action_type: str, policy_ref: str, reason: str) -> str:
        timestamp = datetime.now(timezone.utc).isoformat()
        escaped_reason = reason.replace("|", "/").strip()
        return (
            f"POLICYPILOT_LOG|{timestamp}|case={case_id}|action={action_type}|"
            f"policy_ref={policy_ref}|reason={escaped_reason}"
        )

    @staticmethod
    def action_to_json(action: Dict[str, Any]) -> str:
        return json.dumps(action, sort_keys=True, separators=(",", ":"))
