from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

import inference
from src.env.core import PolicyPilotEnv


def _offline_client():
    return None, "offline-model", "disabled-in-tests"


def test_run_benchmark_works_in_baseline_fallback(monkeypatch) -> None:
    monkeypatch.setattr(inference, "_cached_openai_client", _offline_client)
    result = inference.run_benchmark(
        difficulties=["easy", "medium", "hard"],
        max_steps=3,
        seed=42,
        require_remote_llm=False,
    )

    assert result["env"] == "policypilot"
    assert result["client_mode"] == "baseline_fallback"
    assert len(result["results"]) == 3
    for item in result["results"]:
        assert 0.0 <= item["score"] <= 1.0
        assert "task" in item


def test_act_requires_reset_first(monkeypatch) -> None:
    monkeypatch.setattr(inference, "_cached_openai_client", _offline_client)
    inference.env = PolicyPilotEnv(seed=42)
    client = TestClient(inference.app)

    response = client.post("/act")
    assert response.status_code == 400
    assert "reset" in response.json()["detail"].lower()


def test_act_returns_valid_action_after_reset(monkeypatch) -> None:
    monkeypatch.setattr(inference, "_cached_openai_client", _offline_client)
    inference.env = PolicyPilotEnv(seed=42)
    client = TestClient(inference.app)

    reset_response = client.post("/reset", json={"difficulty": "easy"})
    assert reset_response.status_code == 200

    act_response = client.post("/act")
    assert act_response.status_code == 200
    payload = act_response.json()

    action = payload["action"]
    assert action["action_type"] in {
        "approve_case",
        "reject_case",
        "request_missing_info",
        "escalate_case",
        "flag_for_manual_review",
        "add_audit_note",
    }
    assert action["case_id"] == "C-101"
    assert isinstance(action["payload"], dict)
    assert isinstance(action["reason"], str) and action["reason"].strip()
    assert payload["error"] is None


def test_state_and_grade_require_reset_first() -> None:
    inference.env = PolicyPilotEnv(seed=42)
    client = TestClient(inference.app)

    state_response = client.get("/state")
    assert state_response.status_code == 400
    assert "reset" in state_response.json()["detail"].lower()

    grade_response = client.get("/grade")
    assert grade_response.status_code == 400
    assert "reset" in grade_response.json()["detail"].lower()


def test_run_benchmark_rejects_empty_difficulties(monkeypatch) -> None:
    monkeypatch.setattr(inference, "_cached_openai_client", _offline_client)
    with pytest.raises(ValueError, match="difficulties"):
        inference.run_benchmark(
            difficulties=["", "  "],
            max_steps=2,
            seed=42,
        )


def test_run_benchmark_route_invalid_difficulty_returns_400(monkeypatch) -> None:
    monkeypatch.setattr(inference, "_cached_openai_client", _offline_client)
    client = TestClient(inference.app)

    response = client.post(
        "/run_benchmark",
        json={"difficulties": ["impossible"], "max_steps": 2, "seed": 42},
    )

    assert response.status_code == 400
    assert "unsupported difficulty" in response.json()["detail"].lower()


def test_benchmark_log_format_contains_required_prefixes(monkeypatch, capsys) -> None:
    monkeypatch.setattr(inference, "_cached_openai_client", _offline_client)
    inference.run_benchmark(difficulties=["easy"], max_steps=1, seed=42)
    output = capsys.readouterr().out

    assert "[STEP]  step=" in output
    assert "[END]   success=" in output
