from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from .models import Scenario


@dataclass(slots=True)
class PolicyAnalysis:
    detected_violations: List[str] = field(default_factory=list)
    required_missing_fields: List[str] = field(default_factory=list)
    recommended_action: str = "approve_case"
    safe_to_approve: bool = True
    policy_trace: List[str] = field(default_factory=list)
    conflict_detected: bool = False
    hierarchy_applied: bool = False

    def to_dict(self) -> dict:
        return {
            "detected_violations": list(self.detected_violations),
            "required_missing_fields": list(self.required_missing_fields),
            "recommended_action": self.recommended_action,
            "safe_to_approve": self.safe_to_approve,
            "policy_trace": list(self.policy_trace),
            "conflict_detected": self.conflict_detected,
            "hierarchy_applied": self.hierarchy_applied,
        }


class PolicyEngine:
    def analyze(self, scenario: Scenario) -> PolicyAnalysis:
        analysis = PolicyAnalysis()
        policy = scenario.policy
        case = scenario.case
        documents = scenario.documents

        receipt_documents = [doc for doc in documents if doc.type == "receipt"]
        receipt_fields = [doc.fields for doc in receipt_documents]
        other_fields = [doc.fields for doc in documents if doc.type != "receipt"]
        all_fields = receipt_fields + other_fields

        has_tax_breakdown = any(
            ("tax" in fields and fields.get("tax") is not None)
            or ("tax_breakdown" in fields and fields.get("tax_breakdown"))
            for fields in receipt_fields
        )
        has_manager_approval = any(
            fields.get("manager_approval") in {True, "approved", "APPROVED"}
            or fields.get("approval_status") in {"approved", "APPROVED"}
            for fields in all_fields
        ) or any(doc.type == "manager_approval" for doc in documents)
        has_full_scope_approval = any(
            str(fields.get("approval_scope", "")).lower() in {"full_report", "all_items", "entire_claim", "full"}
            for fields in all_fields
        )

        has_itemized_receipt = any(
            fields.get("itemized_receipt") in {True, "yes", "YES"}
            or fields.get("is_itemized") is True
            or fields.get("is_partial") is False
            for fields in receipt_fields
        )

        meal_items = [item for item in case.line_items if "meal" in str(item.get("category", "")).lower()]
        meal_violation = any(float(item.get("amount", 0.0)) > policy.max_meal for item in meal_items)
        if meal_violation:
            analysis.detected_violations.append("meal_limit_exceeded")
            analysis.policy_trace.append(
                f"Meal line item exceeds max_meal={policy.max_meal:.2f}."
            )

        categories = {str(item.get("category", "")).lower() for item in case.line_items}
        has_personal = any("personal" in category for category in categories)
        has_business = any("business" in category or "travel" in category or "meal" in category for category in categories)
        mixed_expense = has_personal and has_business
        if mixed_expense:
            analysis.detected_violations.append("mixed_expense_detected")
            analysis.policy_trace.append("Case contains mixed business and personal expenses.")

        duplicate_claim = any(
            fields.get("duplicate_claim") is True
            or str(fields.get("duplicate_status", "")).lower() == "duplicate"
            for fields in all_fields
        ) or any(
            doc.type == "duplicate_check"
            and str(doc.fields.get("status", "")).lower() in {"duplicate", "match_found"}
            for doc in documents
        )
        if duplicate_claim:
            analysis.detected_violations.append("duplicate_claim_detected")
            analysis.policy_trace.append("Potential duplicate claim detected from historical records.")

        fraudulent_document = any(
            fields.get("is_fraudulent") is True
            or str(fields.get("receipt_authenticity", "")).lower() in {"fake", "fraudulent", "suspect", "false"}
            for fields in receipt_fields
        )
        if fraudulent_document:
            analysis.detected_violations.append("fraudulent_document_detected")
            analysis.policy_trace.append("Receipt authenticity check indicates a fraudulent or suspect document.")

        pending_exception = any(
            doc.type == "exception_request"
            and str(doc.fields.get("status", doc.fields.get("exception_status", ""))).lower()
            in {"pending", "requested", "unclear"}
            for doc in documents
        )
        if pending_exception:
            analysis.detected_violations.append("pending_exception_request")
            analysis.policy_trace.append("Policy exception request is pending and requires escalation path.")

        if case.amount > policy.requires_tax_breakdown_above and not has_tax_breakdown:
            analysis.required_missing_fields.append("receipt_tax")
            analysis.policy_trace.append(
                f"Tax breakdown required above {policy.requires_tax_breakdown_above:.2f}."
            )

        if case.amount > policy.requires_manager_approval_above and not has_manager_approval:
            analysis.required_missing_fields.append("manager_approval")
            analysis.policy_trace.append(
                f"Manager approval required above {policy.requires_manager_approval_above:.2f}."
            )
        elif (
            case.amount > policy.require_full_scope_approval_above
            and has_manager_approval
            and any("approval_scope" in fields for fields in all_fields)
            and not has_full_scope_approval
        ):
            analysis.required_missing_fields.append("full_scope_approval")
            analysis.policy_trace.append("Manager approval exists but does not cover the full claim scope.")

        receipt_marked_partial = any(fields.get("is_partial") is True for fields in receipt_fields)
        if policy.reject_on_missing_docs and (receipt_marked_partial or not has_itemized_receipt):
            analysis.required_missing_fields.append("itemized_receipt")
            analysis.policy_trace.append("Itemized receipt is required for this policy context.")

        if policy.require_fx_rate_for_non_usd and case.currency.upper() != "USD":
            has_fx_proof = any(
                fields.get("fx_rate") is not None
                or fields.get("exchange_rate") is not None
                or fields.get("currency_conversion_proof") is True
                for fields in all_fields
            )
            if not has_fx_proof:
                analysis.required_missing_fields.append("fx_rate_proof")
                analysis.policy_trace.append("FX conversion evidence required for non-USD reimbursement.")

        analysis.required_missing_fields = sorted(set(analysis.required_missing_fields))

        rule_candidates: List[Tuple[str, str]] = []
        if fraudulent_document:
            rule_candidates.append(("reject_fraudulent_document", "reject_case"))
        if duplicate_claim and policy.escalate_on_duplicate_claim:
            rule_candidates.append(("flag_duplicate_claim", "flag_for_manual_review"))
        if case.amount > policy.escalate_above:
            rule_candidates.append(("escalate_above_threshold", "escalate_case"))
        if pending_exception and policy.pending_exception_requires_escalation:
            rule_candidates.append(("escalate_pending_exception", "escalate_case"))
        if mixed_expense and policy.mixed_expense_requires_escalation:
            rule_candidates.append(("escalate_mixed_expense", "escalate_case"))
        if analysis.required_missing_fields:
            if policy.reject_on_missing_docs:
                rule_candidates.append(("reject_missing_docs", "reject_case"))
            else:
                rule_candidates.append(("request_missing_info_first", "request_missing_info"))
        if meal_violation:
            rule_candidates.append(("reject_meal_violation", "reject_case"))
        if not rule_candidates:
            rule_candidates.append(("approve_clean_case", "approve_case"))

        distinct_actions = {candidate_action for _, candidate_action in rule_candidates}
        analysis.conflict_detected = len(distinct_actions) > 1

        hierarchy = policy.rule_hierarchy
        priority = {rule: idx for idx, rule in enumerate(hierarchy)}
        selected_rule, selected_action = min(
            rule_candidates, key=lambda item: priority.get(item[0], len(priority) + 50)
        )
        analysis.recommended_action = selected_action
        analysis.hierarchy_applied = analysis.conflict_detected or len(rule_candidates) > 1
        analysis.policy_trace.append(
            f"Selected rule '{selected_rule}' -> action '{selected_action}'."
        )

        analysis.safe_to_approve = (
            analysis.recommended_action == "approve_case"
            and not analysis.required_missing_fields
            and not analysis.detected_violations
        )
        return analysis
