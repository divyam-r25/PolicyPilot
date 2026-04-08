from src.env.core import PolicyPilotEnv


def test_step_valid() -> None:
    env = PolicyPilotEnv()
    observation = env.reset(difficulty="easy")

    action = {
        "action_type": "reject_case",
        "case_id": observation["case"]["id"],
        "payload": {},
        "reason": "Meal exceeds policy max_meal limit.",
    }
    observation, reward, done, info = env.step(action)

    assert done is True
    assert reward > 0.5
    assert observation["state"]["decision"] == "reject_case"
    assert info["validation"]["is_valid"] is True
    assert info["analysis"]["recommended_action"] in {"reject_case", "escalate_case"}
    assert 0.0 < info["score"] < 1.0
    assert env.state()["score"] == info["score"]
