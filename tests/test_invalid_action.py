from src.env.core import PolicyPilotEnv


def test_invalid_action() -> None:
    env = PolicyPilotEnv()
    observation = env.reset(difficulty="easy")

    invalid_action = {
        "action_type": "approve_case",
        "case_id": observation["case"]["id"],
        "reason": "I forgot payload and this should be invalid.",
    }
    observation, reward, done, info = env.step(invalid_action)

    assert done is False
    assert reward == -0.2
    assert observation["state"]["decision"] is None
    assert info["validation"]["is_valid"] is False
    assert info["penalties"] == ["invalid_action(-0.2)"]
