from src.env.core import PolicyPilotEnv


def test_reward_penalizes_unsafe_approval() -> None:
    env = PolicyPilotEnv()
    observation = env.reset(difficulty="easy")

    _, reward, done, info = env.step(
        {
            "action_type": "approve_case",
            "case_id": observation["case"]["id"],
            "payload": {},
            "reason": "Approving without policy checks.",
        }
    )

    assert done is True
    assert reward < 0.0
    assert "unsafe_approval(-0.5)" in info["penalties"]


def test_reward_positive_for_good_decision() -> None:
    env = PolicyPilotEnv()
    observation = env.reset(difficulty="easy")

    _, reward, _, info = env.step(
        {
            "action_type": "reject_case",
            "case_id": observation["case"]["id"],
            "payload": {},
            "reason": "Rejected because meal exceeds policy max and no exception is present.",
        }
    )

    assert reward > 0.5
    assert info["penalties"] == []
