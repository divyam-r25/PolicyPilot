from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

from ..env.policy_engine import PolicyAnalysis, PolicyEngine


class BaselineComplianceAgent:
    """
    Baseline deterministic policy-aware agent.

    Logging format (strict):
    POLICYPILOT_LOG|<iso_timestamp>|case=<case_id>|action=<action_type>|policy_ref=<policy_ref>|reason=<reason>
    """

    def __init__(self) -> None:
        self._policy_engine = PolicyEngine()

    def act(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        action, _ = self.act_with_log(observation)
        return action

    def act_with_log(self, observation: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
        case = observation.get("case", {})
        case_id = str(case.get("id", "UNKNOWN"))
        scenario = self._build_scenario_view(observation)
        analysis = self._policy_engine.analyze(scenario)

        action_type = analysis.recommended_action
        payload: Dict[str, Any] = {}
        if action_type == "request_missing_info":
            payload = {"fields": list(analysis.required_missing_fields)}
        reason = self._build_reason(action_type, analysis)
        policy_ref = self._extract_policy_ref(analysis)

        action = {
            "action_type": action_type,
            "case_id": case_id,
            "payload": payload,
            "reason": reason,
        }
        log_line = self._build_log_line(case_id, action_type, policy_ref, reason)
        return action, log_line

    @staticmethod
    def _build_scenario_view(observation: Dict[str, Any]) -> SimpleNamespace:
        case = observation.get("case", {})
        policy = observation.get("policy", {})
        documents = observation.get("documents", [])

        scenario_case = SimpleNamespace(
            amount=float(case.get("amount", 0.0)),
            currency=str(case.get("currency", "USD")),
            line_items=list(case.get("line_items", [])),
        )
        scenario_policy = SimpleNamespace(
            max_meal=float(policy.get("max_meal", 75.0)),
            requires_tax_breakdown_above=float(policy.get("requires_tax_breakdown_above", 300.0)),
            requires_manager_approval_above=float(policy.get("requires_manager_approval_above", 500.0)),
            require_full_scope_approval_above=float(policy.get("require_full_scope_approval_above", 500.0)),
            escalate_above=float(policy.get("escalate_above", 500.0)),
            reject_on_missing_docs=bool(policy.get("reject_on_missing_docs", False)),
            mixed_expense_requires_escalation=bool(policy.get("mixed_expense_requires_escalation", True)),
            escalate_on_duplicate_claim=bool(policy.get("escalate_on_duplicate_claim", True)),
            pending_exception_requires_escalation=bool(policy.get("pending_exception_requires_escalation", True)),
            require_fx_rate_for_non_usd=bool(policy.get("require_fx_rate_for_non_usd", True)),
            rule_hierarchy=list(policy.get("rule_hierarchy", [])),
        )
        scenario_documents = [
            SimpleNamespace(
                type=str(doc.get("type", "")),
                fields=dict(doc.get("fields", {})),
            )
            for doc in documents
            if isinstance(doc, dict)
        ]
        return SimpleNamespace(
            case=scenario_case,
            policy=scenario_policy,
            documents=scenario_documents,
        )

    @staticmethod
    def _build_reason(action_type: str, analysis: PolicyAnalysis) -> str:
        violations = set(analysis.detected_violations)

        if action_type == "flag_for_manual_review":
            if "duplicate_claim_detected" in violations:
                return (
                    "Potential duplicate claim detected from historical records; flagging for manual review per policy hierarchy."
                )
            return "Risk signals require manual review under policy hierarchy."

        if action_type == "escalate_case":
            if "pending_exception_request" in violations:
                return "Policy exception is pending; escalating for manual resolution under policy hierarchy."
            if "mixed_expense_detected" in violations:
                return "Mixed personal and business expenses detected; escalating per policy hierarchy."
            if analysis.required_missing_fields:
                return "High-risk case has unresolved evidence gaps; escalating under policy thresholds."
            return "Risk signals exceed policy thresholds; escalating for manual decision."

        if action_type == "request_missing_info":
            return "Required evidence is missing for policy thresholds; requesting fields before any approval."

        if action_type == "reject_case":
            if "fraudulent_document_detected" in violations:
                return "Receipt authenticity appears fraudulent under policy controls; rejecting the claim."
            if "meal_limit_exceeded" in violations:
                return "Meal amount exceeds policy cap and no override is present."
            if analysis.required_missing_fields:
                return "Required documentation is missing and policy mandates rejection."
            return "Policy violations detected; rejecting claim under compliance rules."

        return "Case appears compliant under current policy."

    @staticmethod
    def _extract_policy_ref(analysis: PolicyAnalysis) -> str:
        for trace_line in reversed(analysis.policy_trace):
            match = re.search(r"Selected rule '([^']+)'", trace_line)
            if match:
                return match.group(1)

        fallback = {
            "approve_case": "approve_clean_case",
            "reject_case": "reject_case",
            "request_missing_info": "request_missing_info_first",
            "escalate_case": "escalate_case",
            "flag_for_manual_review": "flag_for_manual_review",
        }
        return fallback.get(analysis.recommended_action, "baseline_policy_engine")

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
