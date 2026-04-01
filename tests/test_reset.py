from src.env.core import PolicyPilotEnv


def test_reset() -> None:
    env = PolicyPilotEnv()
    observation = env.reset(difficulty="medium")

    assert observation["task"]["name"] == "compliance_review"
    assert observation["task"]["difficulty"] == "medium"
    assert observation["case"]["id"] == "C-102"
    assert observation["state"]["decision"] is None
    assert observation["state"]["reviewed"] is False

    internal_state = env.state()
    assert internal_state["step_count"] == 0
    assert internal_state["done"] is False
    assert internal_state["case_id"] == "C-102"
    assert internal_state["history"] == []
    assert internal_state["episode_trace"] == []


def test_reset_clears_previous_episode_state() -> None:
    env = PolicyPilotEnv()
    first_observation = env.reset(difficulty="easy")
    env.step(
        {
            "action_type": "reject_case",
            "case_id": first_observation["case"]["id"],
            "payload": {},
            "reason": "Meal exceeds policy cap.",
        }
    )
    state_after_step = env.state()
    assert state_after_step["step_count"] == 1
    assert state_after_step["done"] is True
    assert len(state_after_step["history"]) == 1

    second_observation = env.reset(difficulty="medium")
    reset_state = env.state()
    assert second_observation["case"]["id"] == "C-102"
    assert reset_state["step_count"] == 0
    assert reset_state["decision"] is None
    assert reset_state["history"] == []
    assert reset_state["reward_accumulated"] == 0.0
