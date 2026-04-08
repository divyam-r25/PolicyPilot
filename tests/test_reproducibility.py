from src.agents.baseline import BaselineComplianceAgent
from src.env.core import PolicyPilotEnv


def test_same_seed_produces_same_hard_case_and_grade() -> None:
    env_a = PolicyPilotEnv(seed=42)
    obs_a = env_a.reset(difficulty="hard")
    scenario_a = env_a.current_scenario
    assert scenario_a is not None
    env_a.step(
        {
            "action_type": scenario_a.gold.decision,
            "case_id": obs_a["case"]["id"],
            "payload": {},
            "reason": "Deterministic policy-grounded decision.",
        }
    )
    grade_a = env_a.grade()

    env_b = PolicyPilotEnv(seed=42)
    obs_b = env_b.reset(difficulty="hard")
    scenario_b = env_b.current_scenario
    assert scenario_b is not None
    env_b.step(
        {
            "action_type": scenario_b.gold.decision,
            "case_id": obs_b["case"]["id"],
            "payload": {},
            "reason": "Deterministic policy-grounded decision.",
        }
    )
    grade_b = env_b.grade()

    assert obs_a["case"]["id"] == obs_b["case"]["id"]
    assert grade_a["score"] == grade_b["score"]


def test_baseline_hard_score_is_not_higher_than_easy() -> None:
    baseline = BaselineComplianceAgent()

    easy_env = PolicyPilotEnv(seed=42)
    easy_observation = easy_env.reset(difficulty="easy")
    easy_action = baseline.act(easy_observation)
    easy_env.step(easy_action)
    easy_score = easy_env.grade()["score"]

    hard_env = PolicyPilotEnv(seed=42)
    hard_observation = hard_env.reset(difficulty="hard")
    hard_action = baseline.act(hard_observation)
    hard_env.step(hard_action)
    hard_score = hard_env.grade()["score"]

    assert 0.0 < easy_score < 1.0
    assert 0.0 < hard_score < 1.0
    assert hard_score <= easy_score
