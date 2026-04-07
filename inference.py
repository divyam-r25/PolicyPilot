from __future__ import annotations

import argparse
import json
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.agents.baseline import BaselineComplianceAgent
from src.env.core import PolicyPilotEnv
from src.tasks import list_difficulties

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional runtime dependency path
    OpenAI = None  # type: ignore[assignment]


ENV_NAME = "policypilot"
DEFAULT_MODEL_NAME = "gpt-4.1-mini"
REQUIRED_ENV_VARS = ("API_BASE_URL", "API_KEY")


class ResetRequest(BaseModel):
    difficulty: str = Field(default="easy", description="easy | medium | hard")
    variant: Optional[int] = Field(default=None, description="Optional deterministic scenario variant index")


class StepRequest(BaseModel):
    action: Dict[str, Any]


class ActRequest(BaseModel):
    observation: Dict[str, Any]


class RunEpisodeRequest(BaseModel):
    difficulty: str = Field(default="easy")
    max_steps: int = Field(default=8, ge=1, le=20)
    variant: Optional[int] = Field(default=None)


class RunBenchmarkRequest(BaseModel):
    difficulties: List[str] = Field(default_factory=lambda: ["easy", "medium", "hard"])
    max_steps: int = Field(default=8, ge=1, le=20)
    seed: int = Field(default=42)


app = FastAPI(
    title="PolicyPilot Compliance Review Benchmark API",
    version="1.1.0",
    description="OpenEnv-compatible enterprise compliance review benchmark.",
)

env = PolicyPilotEnv(seed=42)
baseline_agent = BaselineComplianceAgent()


def _compact_json(value: Dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sanitize_error_text(message: Optional[str]) -> str:
    if not message:
        return "null"
    cleaned = re.sub(r"\s+", " ", message.strip())
    return cleaned if cleaned else "null"


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    stripped = text.strip()
    try:
        loaded = json.loads(stripped)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        pass

    candidates = re.findall(r"\{.*\}", stripped, flags=re.DOTALL)
    for candidate in candidates:
        try:
            loaded = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            return loaded
    return None


def _normalize_action(action: Dict[str, Any], observation: Dict[str, Any]) -> Dict[str, Any]:
    case_id = str(observation.get("case", {}).get("id", "UNKNOWN"))
    normalized = {
        "action_type": action.get("action_type"),
        "case_id": action.get("case_id", case_id),
        "payload": action.get("payload", {}),
        "reason": action.get("reason", "Policy-grounded decision."),
    }
    if not isinstance(normalized["payload"], dict):
        normalized["payload"] = {}
    if not isinstance(normalized["reason"], str) or not normalized["reason"].strip():
        normalized["reason"] = "Policy-grounded decision."
    return normalized


def _fallback_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    action = baseline_agent.act(observation)
    return _normalize_action(action, observation)


def _resolve_model_name() -> str:
    for key in ("MODEL_NAME", "MODEL", "OPENAI_MODEL"):
        value = os.getenv(key, "").strip()
        if value:
            return value
    return DEFAULT_MODEL_NAME


def _make_openai_client() -> Tuple[Optional[Any], str, Optional[str]]:
    api_base_url = os.getenv("API_BASE_URL", "").strip()
    model_name = _resolve_model_name()
    api_key = os.getenv("API_KEY", "").strip()

    missing = [key for key in REQUIRED_ENV_VARS if not os.getenv(key, "").strip()]
    if missing:
        return None, model_name, f"Missing env vars: {', '.join(missing)}"
    if OpenAI is None:
        return None, model_name, "openai package unavailable; using baseline fallback."
    try:
        client = OpenAI(base_url=api_base_url, api_key=api_key)
    except Exception as exc:  # pragma: no cover - runtime path
        return None, model_name, f"OpenAI client init failed: {exc}"
    return client, model_name, None


@lru_cache(maxsize=1)
def _cached_openai_client() -> Tuple[Optional[Any], str, Optional[str]]:
    return _make_openai_client()


def _llm_action(client: Any, model_name: str, observation: Dict[str, Any]) -> Dict[str, Any]:
    system_prompt = (
        "You are a compliance officer. Return ONLY one JSON object with keys: "
        "action_type, case_id, payload, reason. "
        "Never approve when required evidence is missing."
    )
    user_prompt = (
        "Observation JSON:\n"
        f"{json.dumps(observation, sort_keys=True)}\n\n"
        "Choose one action from allowed_actions."
    )
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content
    if isinstance(content, list):
        text_chunks = [chunk.get("text", "") for chunk in content if isinstance(chunk, dict)]
        content_text = "\n".join(text_chunks)
    else:
        content_text = str(content or "")

    parsed = _extract_json_object(content_text)
    if parsed is None:
        raise ValueError("Model response did not contain valid JSON action object.")
    return _normalize_action(parsed, observation)


def _select_action(
    observation: Dict[str, Any],
    llm_client: Optional[Any],
    model_name: str,
) -> Tuple[Dict[str, Any], Optional[str]]:
    if llm_client is None:
        return _fallback_action(observation), None
    try:
        return _llm_action(llm_client, model_name, observation), None
    except Exception as exc:
        return _fallback_action(observation), f"llm_error:{exc}"


def _print_start(task_name: str, model_name: str) -> None:
    print(f"[START] task={task_name} env={ENV_NAME} model={model_name}", flush=True)


def _print_step(step_index: int, action: Dict[str, Any], reward: float, done: bool, error: Optional[str]) -> None:
    action_str = _compact_json(action)
    error_text = _sanitize_error_text(error)
    print(
        f"[STEP]  step={step_index} action={action_str} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_text}",
        flush=True,
    )


def _print_end(success: bool, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{value:.2f}" for value in rewards)
    print(
        f"[END]   success={str(success).lower()} steps={len(rewards)} rewards={rewards_str}",
        flush=True,
    )


def run_single_task(
    difficulty: str,
    max_steps: int,
    seed: int,
    model_name: str,
    llm_client: Optional[Any] = None,
) -> Dict[str, Any]:
    task_name = f"compliance_review_{difficulty}"
    _print_start(task_name=task_name, model_name=model_name)

    runner_env = PolicyPilotEnv(seed=seed)
    rewards: List[float] = []
    grade: Optional[Dict[str, Any]] = None
    last_error: Optional[str] = None
    done = False

    try:
        observation = runner_env.reset(difficulty=difficulty)
    except Exception as exc:
        last_error = f"reset_error:{exc}"
        _print_end(success=False, rewards=rewards)
        return {
            "task": difficulty,
            "success": False,
            "score": 0.0,
            "steps": 0,
            "rewards": rewards,
            "grade": None,
            "error": last_error,
        }

    for step_index in range(1, max_steps + 1):
        action, action_error = _select_action(
            observation=observation,
            llm_client=llm_client,
            model_name=model_name,
        )
        try:
            observation, reward, done, info = runner_env.step(action)
            rewards.append(reward)
            validation = info.get("validation", {})
            validation_errors = validation.get("errors") or []
            step_error = action_error or (validation_errors[0] if validation_errors else None)
            _print_step(
                step_index=step_index,
                action=action,
                reward=reward,
                done=done,
                error=step_error,
            )
        except Exception as exc:
            rewards.append(0.0)
            last_error = f"step_error:{exc}"
            _print_step(
                step_index=step_index,
                action=action,
                reward=0.0,
                done=True,
                error=last_error,
            )
            done = True
            break

        if done:
            try:
                grade = runner_env.grade()
            except Exception as exc:
                last_error = f"grade_error:{exc}"
                grade = None
            break

    success = bool(grade and grade.get("success"))
    _print_end(success=success, rewards=rewards)
    return {
        "task": difficulty,
        "success": success,
        "score": float(grade["score"]) if grade else 0.0,
        "steps": len(rewards),
        "rewards": rewards,
        "grade": grade,
        "error": last_error,
    }


def run_benchmark(
    difficulties: List[str],
    max_steps: int = 8,
    seed: int = 42,
) -> Dict[str, Any]:
    llm_client, model_name, startup_error = _make_openai_client()
    results: List[Dict[str, Any]] = []

    for difficulty in difficulties:
        result = run_single_task(
            difficulty=difficulty,
            max_steps=max_steps,
            seed=seed,
            model_name=model_name,
            llm_client=llm_client,
        )
        results.append(result)

    average_score = round(sum(item["score"] for item in results) / float(len(results)), 4) if results else 0.0
    return {
        "env": ENV_NAME,
        "model": model_name,
        "seed": seed,
        "difficulties": list(difficulties),
        "average_score": average_score,
        "results": results,
        "client_mode": "openai" if llm_client is not None else "baseline_fallback",
        "startup_error": startup_error,
    }


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "PolicyPilot",
        "description": "Compliance Review Benchmark (OpenEnv)",
        "endpoints": [
            "/health",
            "/reset",
            "/step",
            "/state",
            "/grade",
            "/act",
            "/run_episode",
            "/run_benchmark",
        ],
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/reset")
def reset(request: Optional[ResetRequest] = None) -> Dict[str, Any]:
    difficulty = request.difficulty if request else "easy"
    variant = request.variant if request else None
    try:
        observation = env.reset(difficulty=difficulty, scenario_variant=variant)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"observation": observation, "state": env.state()}


@app.post("/step")
def step(request: StepRequest) -> Dict[str, Any]:
    try:
        observation, reward, done, info = env.step(request.action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "observation": observation,
        "reward": reward,
        "done": done,
        "info": info,
        "state": env.state(),
    }


@app.get("/state")
def state() -> Dict[str, Any]:
    try:
        return env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/grade")
def grade() -> Dict[str, Any]:
    try:
        return env.grade()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/act")
def act(request: ActRequest) -> Dict[str, Any]:
    llm_client, model_name, startup_error = _cached_openai_client()
    action, action_error = _select_action(
        observation=request.observation,
        llm_client=llm_client,
        model_name=model_name,
    )
    error_parts = [part for part in (startup_error, action_error) if part]
    if error_parts:
        log_line = " | ".join(error_parts)
    else:
        log_line = f"llm_action:model={model_name}"
    return {"action": action, "log": log_line}


@app.post("/run_episode")
def run_episode(request: RunEpisodeRequest) -> Dict[str, Any]:
    llm_client, model_name, startup_error = _cached_openai_client()
    try:
        observation = env.reset(difficulty=request.difficulty, scenario_variant=request.variant)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    trajectory: List[Dict[str, Any]] = []
    done = False
    info: Dict[str, Any] = {}

    for _ in range(request.max_steps):
        action, action_error = _select_action(
            observation=observation,
            llm_client=llm_client,
            model_name=model_name,
        )
        error_parts = [part for part in (startup_error, action_error) if part]
        if error_parts:
            log_line = " | ".join(error_parts)
        else:
            log_line = f"llm_action:model={model_name}"
        observation, reward, done, info = env.step(action)
        trajectory.append(
            {
                "action": action,
                "log": log_line,
                "reward": reward,
                "done": done,
                "termination_reason": info.get("termination_reason"),
                "reward_breakdown": info.get("reward_breakdown"),
            }
        )
        if done:
            break

    return {
        "final_observation": observation,
        "done": done,
        "trajectory": trajectory,
        "info": info,
        "state": env.state(),
        "grade": env.grade() if done else None,
    }


@app.post("/run_benchmark")
def run_benchmark_endpoint(request: RunBenchmarkRequest) -> Dict[str, Any]:
    requested = [difficulty.lower().strip() for difficulty in request.difficulties]
    supported = set(list_difficulties())
    invalid = [difficulty for difficulty in requested if difficulty not in supported]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported difficulties: {invalid}. Supported: {sorted(supported)}.",
        )
    return run_benchmark(
        difficulties=requested,
        max_steps=request.max_steps,
        seed=request.seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PolicyPilot inference runner and API server.")
    parser.add_argument("--serve", action="store_true", help="Run FastAPI server mode.")
    parser.add_argument(
        "--run-benchmark",
        action="store_true",
        help="Run benchmark mode (default when --serve is not used).",
    )
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "7860")))
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--difficulties",
        type=str,
        default="easy,medium,hard",
        help="Comma-separated list of difficulties.",
    )
    args = parser.parse_args()

    if args.serve:
        uvicorn.run("inference:app", host="0.0.0.0", port=args.port, reload=False)
        return

    selected = [item.strip().lower() for item in args.difficulties.split(",") if item.strip()]
    supported = set(list_difficulties())
    difficulties = [difficulty for difficulty in selected if difficulty in supported]
    if not difficulties:
        difficulties = ["easy", "medium", "hard"]

    summary = run_benchmark(
        difficulties=difficulties,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
