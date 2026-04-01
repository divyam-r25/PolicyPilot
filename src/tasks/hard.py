from __future__ import annotations

from typing import List

from ..env.models import CaseData, DocumentData, GoldStandard, PolicyConfig, Scenario


def _allowed_actions() -> List[str]:
    return [
        "approve_case",
        "reject_case",
        "request_missing_info",
        "escalate_case",
        "flag_for_manual_review",
        "add_audit_note",
    ]


def build_scenarios() -> List[Scenario]:
    return [
        Scenario(
            id="C-103",
            name="compliance_review",
            difficulty="hard",
            case=CaseData(
                id="C-103",
                type="mixed_expense_report",
                amount=640.0,
                currency="USD",
                line_items=[
                    {
                        "category": "business_travel_lodging",
                        "amount": 280.0,
                        "description": "Conference lodging",
                    },
                    {
                        "category": "business_meal",
                        "amount": 300.0,
                        "description": "Team dinner",
                    },
                    {
                        "category": "personal_upgrade",
                        "amount": 60.0,
                        "description": "Personal room upgrade",
                    },
                ],
            ),
            documents=[
                DocumentData(
                    type="receipt",
                    fields={
                        "merchant": "Summit Hotel",
                        "subtotal": 640.0,
                        "is_partial": True,
                    },
                ),
                DocumentData(
                    type="approval_thread",
                    fields={
                        "approval_status": "unclear",
                    },
                ),
            ],
            policy=PolicyConfig(
                max_meal=75.0,
                requires_tax_breakdown_above=300.0,
                requires_manager_approval_above=500.0,
                escalate_above=500.0,
                reject_on_missing_docs=True,
                mixed_expense_requires_escalation=True,
                rule_hierarchy=[
                    "escalate_above_threshold",
                    "escalate_mixed_expense",
                    "reject_missing_docs",
                    "reject_meal_violation",
                    "request_missing_info_first",
                    "approve_clean_case",
                ],
            ),
            allowed_actions=_allowed_actions(),
            max_steps=8,
            gold=GoldStandard(
                decision="escalate_case",
                missing_fields=["itemized_receipt", "manager_approval", "receipt_tax"],
                policy_refs=["reject_on_missing_docs", "escalate_above", "rule_hierarchy"],
                safe_actions=["escalate_case", "flag_for_manual_review"],
                unsafe_actions=["approve_case"],
            ),
            description=(
                "Conflicting rules with mixed expenses, partial documentation, and escalation threshold."
            ),
        ),
        Scenario(
            id="C-104",
            name="compliance_review",
            difficulty="hard",
            case=CaseData(
                id="C-104",
                type="international_travel_expense",
                amount=910.0,
                currency="EUR",
                line_items=[
                    {
                        "category": "business_travel_lodging",
                        "amount": 650.0,
                        "description": "International conference hotel",
                    },
                    {
                        "category": "personal_minibar",
                        "amount": 85.0,
                        "description": "Non-reimbursable minibar charge",
                    },
                    {
                        "category": "business_transport",
                        "amount": 175.0,
                        "description": "Airport transfer and local transit",
                    },
                ],
            ),
            documents=[
                DocumentData(
                    type="receipt",
                    fields={
                        "merchant": "EuroStay",
                        "subtotal": 910.0,
                        "is_partial": True,
                    },
                ),
                DocumentData(
                    type="manager_approval",
                    fields={
                        "manager_approval": True,
                        "approval_scope": "lodging_only",
                    },
                ),
                DocumentData(
                    type="duplicate_check",
                    fields={"status": "clean"},
                ),
            ],
            policy=PolicyConfig(
                max_meal=75.0,
                requires_tax_breakdown_above=300.0,
                requires_manager_approval_above=500.0,
                require_full_scope_approval_above=500.0,
                escalate_above=700.0,
                reject_on_missing_docs=False,
                mixed_expense_requires_escalation=True,
                require_fx_rate_for_non_usd=True,
                rule_hierarchy=[
                    "escalate_above_threshold",
                    "escalate_mixed_expense",
                    "request_missing_info_first",
                    "reject_missing_docs",
                    "approve_clean_case",
                ],
            ),
            allowed_actions=_allowed_actions(),
            max_steps=8,
            gold=GoldStandard(
                decision="escalate_case",
                missing_fields=["full_scope_approval", "fx_rate_proof", "receipt_tax"],
                policy_refs=[
                    "require_full_scope_approval_above",
                    "require_fx_rate_for_non_usd",
                    "escalate_above",
                ],
                safe_actions=["escalate_case", "request_missing_info"],
                unsafe_actions=["approve_case"],
            ),
            description=(
                "International mixed claim with partial approval scope and missing FX/tax evidence above escalation threshold."
            ),
        ),
        Scenario(
            id="C-105",
            name="compliance_review",
            difficulty="hard",
            case=CaseData(
                id="C-105",
                type="duplicate_reimbursement_claim",
                amount=420.0,
                currency="USD",
                line_items=[
                    {
                        "category": "business_travel_lodging",
                        "amount": 420.0,
                        "description": "Repeat hotel reimbursement",
                    }
                ],
            ),
            documents=[
                DocumentData(
                    type="receipt",
                    fields={
                        "merchant": "Metro Hotel",
                        "subtotal": 390.0,
                        "tax": 30.0,
                        "is_itemized": True,
                    },
                ),
                DocumentData(
                    type="duplicate_check",
                    fields={
                        "status": "duplicate",
                        "source_case_id": "C-088",
                    },
                ),
            ],
            policy=PolicyConfig(
                max_meal=75.0,
                requires_tax_breakdown_above=300.0,
                requires_manager_approval_above=500.0,
                escalate_above=800.0,
                reject_on_missing_docs=False,
                mixed_expense_requires_escalation=False,
                escalate_on_duplicate_claim=True,
                rule_hierarchy=[
                    "flag_duplicate_claim",
                    "escalate_pending_exception",
                    "request_missing_info_first",
                    "approve_clean_case",
                ],
            ),
            allowed_actions=_allowed_actions(),
            max_steps=8,
            gold=GoldStandard(
                decision="flag_for_manual_review",
                missing_fields=[],
                policy_refs=["flag_duplicate_claim", "duplicate_check"],
                safe_actions=["flag_for_manual_review", "escalate_case"],
                unsafe_actions=["approve_case"],
            ),
            description="Duplicate-claim signal with otherwise complete docs should be manually flagged.",
        ),
        Scenario(
            id="C-106",
            name="compliance_review",
            difficulty="hard",
            case=CaseData(
                id="C-106",
                type="suspected_fraud_claim",
                amount=280.0,
                currency="USD",
                line_items=[
                    {
                        "category": "business_supplies",
                        "amount": 280.0,
                        "description": "Office supplies reimbursement",
                    }
                ],
            ),
            documents=[
                DocumentData(
                    type="receipt",
                    fields={
                        "merchant": "Paper Planet",
                        "subtotal": 250.0,
                        "tax": 30.0,
                        "is_itemized": True,
                        "receipt_authenticity": "fraudulent",
                    },
                ),
                DocumentData(
                    type="manager_approval",
                    fields={"manager_approval": True, "approval_scope": "full_report"},
                ),
            ],
            policy=PolicyConfig(
                max_meal=75.0,
                requires_tax_breakdown_above=300.0,
                requires_manager_approval_above=500.0,
                escalate_above=700.0,
                reject_on_missing_docs=False,
                mixed_expense_requires_escalation=False,
                rule_hierarchy=[
                    "reject_fraudulent_document",
                    "flag_duplicate_claim",
                    "request_missing_info_first",
                    "approve_clean_case",
                ],
            ),
            allowed_actions=_allowed_actions(),
            max_steps=8,
            gold=GoldStandard(
                decision="reject_case",
                missing_fields=[],
                policy_refs=["reject_fraudulent_document"],
                safe_actions=["reject_case", "flag_for_manual_review"],
                unsafe_actions=["approve_case"],
            ),
            description="Fraudulent receipt should be rejected despite manager approval and complete formatting.",
        ),
        Scenario(
            id="C-107",
            name="compliance_review",
            difficulty="hard",
            case=CaseData(
                id="C-107",
                type="exception_pending_claim",
                amount=760.0,
                currency="USD",
                line_items=[
                    {
                        "category": "business_travel_lodging",
                        "amount": 520.0,
                        "description": "Client visit lodging",
                    },
                    {
                        "category": "business_meal",
                        "amount": 140.0,
                        "description": "Client meal over policy cap with exception request",
                    },
                    {
                        "category": "business_transport",
                        "amount": 100.0,
                        "description": "Local transport",
                    },
                ],
            ),
            documents=[
                DocumentData(
                    type="receipt",
                    fields={
                        "merchant": "City Inn",
                        "subtotal": 700.0,
                        "tax": 60.0,
                        "is_itemized": True,
                    },
                ),
                DocumentData(
                    type="manager_approval",
                    fields={"manager_approval": True, "approval_scope": "full_report"},
                ),
                DocumentData(
                    type="exception_request",
                    fields={"status": "pending", "reason": "VIP-client hosted dinner"},
                ),
            ],
            policy=PolicyConfig(
                max_meal=75.0,
                requires_tax_breakdown_above=300.0,
                requires_manager_approval_above=500.0,
                escalate_above=700.0,
                reject_on_missing_docs=False,
                pending_exception_requires_escalation=True,
                mixed_expense_requires_escalation=False,
                rule_hierarchy=[
                    "escalate_pending_exception",
                    "escalate_above_threshold",
                    "reject_meal_violation",
                    "request_missing_info_first",
                    "approve_clean_case",
                ],
            ),
            allowed_actions=_allowed_actions(),
            max_steps=8,
            gold=GoldStandard(
                decision="escalate_case",
                missing_fields=[],
                policy_refs=["escalate_pending_exception", "exception_request"],
                safe_actions=["escalate_case"],
                unsafe_actions=["approve_case"],
            ),
            description="Meal exception is pending; escalation is required instead of premature rejection or approval.",
        ),
    ]


def build_scenario(index: int = 0) -> Scenario:
    scenarios = build_scenarios()
    if not scenarios:
        raise ValueError("Hard scenario list is empty.")
    return scenarios[index % len(scenarios)]
