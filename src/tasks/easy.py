from __future__ import annotations

from ..env.models import CaseData, DocumentData, GoldStandard, PolicyConfig, Scenario


def build_scenario() -> Scenario:
    return Scenario(
        id="C-101",
        name="compliance_review",
        difficulty="easy",
        case=CaseData(
            id="C-101",
            type="meal_expense",
            amount=120.0,
            currency="USD",
            line_items=[
                {
                    "category": "business_meal",
                    "amount": 120.0,
                    "description": "Client dinner",
                }
            ],
        ),
        documents=[
            DocumentData(
                type="receipt",
                fields={
                    "merchant": "Bistro Nova",
                    "subtotal": 108.0,
                    "tax": 12.0,
                    "is_itemized": True,
                },
            )
        ],
        policy=PolicyConfig(
            max_meal=75.0,
            requires_tax_breakdown_above=300.0,
            requires_manager_approval_above=500.0,
            escalate_above=500.0,
            reject_on_missing_docs=False,
            mixed_expense_requires_escalation=True,
            rule_hierarchy=[
                "reject_meal_violation",
                "request_missing_info_first",
                "escalate_above_threshold",
                "escalate_mixed_expense",
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
        max_steps=6,
        gold=GoldStandard(
            decision="reject_case",
            missing_fields=[],
            policy_refs=["max_meal"],
            safe_actions=["reject_case", "escalate_case"],
            unsafe_actions=["approve_case"],
        ),
        description="Single-rule meal limit violation where rejection is expected.",
    )
