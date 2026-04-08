from src.agents.baseline import BaselineComplianceAgent
from src.env.core import PolicyPilotEnv


def test_baseline_flags_duplicate_claim_case() -> None:
    env = PolicyPilotEnv(seed=42)
    observation = env.reset(difficulty="hard")

    assert observation["case"]["id"] == "C-105"

    action = BaselineComplianceAgent().act(observation)

    assert action["action_type"] == "flag_for_manual_review"
    assert "duplicate" in action["reason"].lower()
