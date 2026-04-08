import inference
from fastapi.testclient import TestClient

from src.env.core import PolicyPilotEnv


def _fresh_client() -> TestClient:
    inference.env = PolicyPilotEnv(seed=42)
    inference._cached_openai_client.cache_clear()
    return TestClient(inference.app, raise_server_exceptions=False)


def test_api_pre_reset_routes_return_400_not_500() -> None:
    client = _fresh_client()

    state_response = client.get("/state")
    grade_response = client.get("/grade")
    act_response = client.post("/act")

    assert state_response.status_code == 400
    assert grade_response.status_code == 400
    assert act_response.status_code == 400

    assert state_response.json()["detail"] == inference.ENV_NOT_INITIALIZED_DETAIL
    assert grade_response.json()["detail"] == inference.ENV_NOT_INITIALIZED_DETAIL
    assert act_response.json()["detail"] == inference.ENV_NOT_INITIALIZED_DETAIL


def test_api_post_reset_routes_return_200_and_valid_score() -> None:
    client = _fresh_client()

    reset_response = client.post("/reset", json={"difficulty": "easy"})
    assert reset_response.status_code == 200

    state_response = client.get("/state")
    grade_response = client.get("/grade")
    act_response = client.post("/act")

    assert state_response.status_code == 200
    assert grade_response.status_code == 200
    assert act_response.status_code == 200

    state_payload = state_response.json()
    grade_payload = grade_response.json()
    act_payload = act_response.json()

    assert state_payload["case_id"] == "C-101"
    assert state_payload["step_count"] == 0

    assert 0.0 < grade_payload["score"] < 1.0

    assert {"action_type", "case_id", "payload", "reason"}.issubset(
        set(act_payload["action"].keys())
    )
