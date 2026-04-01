from __future__ import annotations

from ..env.models import CaseData, DocumentData, GoldStandard, PolicyConfig, Scenario


def build_scenario() -> Scenario:
    return Scenario(
        id="C-102",
        name="compliance_review",
        difficulty="medium",
        case=CaseData(
            id="C-102",
            type="travel_expense",
            amount=540.0,
            currency="USD",
            line_items=[
                {
                    "category": "business_travel_lodging",
                    "amount": 540.0,
                    "description": "Hotel stay for client on-site visit",
                }
            ],
        ),
        documents=[
            DocumentData(
                type="receipt",
                fields={
                    "merchant": "Hotel Aurora",
                    "subtotal": 500.0,
                    "is_partial": False,
                },
            )
        ],
        policy=PolicyConfig(
            max_meal=75.0,
            requires_tax_breakdown_above=300.0,
            requires_manager_approval_above=500.0,
            escalate_above=700.0,
            reject_on_missing_docs=False,
            mixed_expense_requires_escalation=True,
            rule_hierarchy=[
                "request_missing_info_first",
                "escalate_above_threshold",
                "reject_missing_docs",
                "reject_meal_violation",
                "approve_clean_case",
            ],
        ),
        allowed_actions=[
            "approve_case",
            "reject_case",
            "request_missing_info",
            "escalate_case",
            "flag_for_manual_review",
            "add_audit_note",
        ],
        max_steps=8,
        gold=GoldStandard(
            decision="request_missing_info",
            missing_fields=["manager_approval", "receipt_tax"],
            policy_refs=["requires_tax_breakdown_above", "requires_manager_approval_above"],
            safe_actions=["request_missing_info", "escalate_case"],
            unsafe_actions=["approve_case"],
        ),
        description="Missing tax breakdown and manager approval in a high-value travel claim.",
    )
