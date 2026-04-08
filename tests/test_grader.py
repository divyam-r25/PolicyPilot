from src.env.core import PolicyPilotEnv


def test_grader_easy_medium_hard() -> None:
    env = PolicyPilotEnv()

    easy_observation = env.reset(difficulty="easy")
    env.step(
        {
            "action_type": "reject_case",
            "case_id": easy_observation["case"]["id"],
            "payload": {},
            "reason": "Meal exceeds policy limit.",
        }
    )
    easy_grade = env.grade()
    assert easy_grade["score"] >= 0.85
    assert 0.0 < easy_grade["score"] < 1.0
    assert "subscores" in easy_grade

    medium_observation = env.reset(difficulty="medium")
    env.step(
        {
            "action_type": "request_missing_info",
            "case_id": medium_observation["case"]["id"],
            "payload": {"fields": ["receipt_tax", "manager_approval"]},
            "reason": "Policy requires tax and manager approval above threshold.",
        }
    )
    medium_grade = env.grade()
    assert medium_grade["score"] >= 0.85
    assert 0.0 < medium_grade["score"] < 1.0

    hard_observation = env.reset(difficulty="hard")
    hard_decision = env.current_scenario.gold.decision  # type: ignore[union-attr]
    env.step(
        {
            "action_type": hard_decision,
            "case_id": hard_observation["case"]["id"],
            "payload": {},
            "reason": "Applying policy hierarchy, threshold checks, approval evidence, and receipt controls.",
        }
    )
    hard_grade = env.grade()
    assert hard_grade["score"] >= 0.85
    assert 0.0 < hard_grade["score"] < 1.0
    assert "episode_trace" in hard_grade
