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

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

ENV_NAME = "policypilot"
DEFAULT_MODEL_NAME = "gpt-4.1-mini"
DEFAULT_HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
API_KEY_ENV_CANDIDATES = (
    "API_KEY",
    "HF_TOKEN",
    "HUGGINGFACEHUB_API_TOKEN",
    "HUGGING_FACE_HUB_TOKEN",
)
PLACEHOLDER_TOKEN_HINTS = (
    "your_actual_token_here",
    "your_token_here",
    "<token>",
    "replace_me",
    "changeme",
    "paste_token_here",
)


def _resolve_model_name() -> str:
    return os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)


def _env_truthy(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_api_key() -> Tuple[Optional[str], Optional[str]]:
    for env_name in API_KEY_ENV_CANDIDATES:
        value = os.getenv(env_name)
        if value and value.strip():
            return value.strip(), env_name
    return None, None


def _resolve_api_base_url(api_key: Optional[str]) -> Optional[str]:
    explicit_base_url = os.getenv("API_BASE_URL")
    if explicit_base_url and explicit_base_url.strip():
        return explicit_base_url.strip()

    if api_key:
        return DEFAULT_HF_ROUTER_BASE_URL

    return None


def _looks_like_placeholder_token(token: str) -> bool:
    value = token.strip().lower()
    if not value:
        return True
    if any(hint in value for hint in PLACEHOLDER_TOKEN_HINTS):
        return True
    return value in {"hf_xxx", "hf_***", "token"}


def _should_require_remote_llm(cli_flag: bool = False) -> bool:
    if cli_flag:
        return True
    return _env_truthy("REQUIRE_REMOTE_LLM") or _env_truthy("STRICT_PROXY_MODE")


def _make_openai_client() -> Tuple[Optional[Any], str, Optional[str]]:
    """
    Build OpenAI-compatible client using injected hackathon proxy vars.

    Important behavior:
    - If proxy env vars exist, we ATTEMPT a real proxy call so validator can detect usage.
    - If that call fails, we DO NOT crash the entire script.
      We log a warning and return fallback mode.
    """
    api_key, api_key_env = _resolve_api_key()
    api_base_url = _resolve_api_base_url(api_key)
    model_name = _resolve_model_name()

    if not (api_base_url and api_key):
        return (
            None,
            model_name,
            "Missing remote LLM config. Set API_KEY (or HF_TOKEN) and optionally API_BASE_URL.",
        )

    if _looks_like_placeholder_token(api_key):
        hint_env = api_key_env or "API_KEY"
        return (
            None,
            model_name,
            (
                f"{hint_env} appears to be a placeholder token. "
                "Set a real Hugging Face token (starts with hf_) before running remote LLM mode."
            ),
        )

    if "router.huggingface.co" in api_base_url and not api_key.startswith("hf_"):
        hint_env = api_key_env or "API_KEY"
        return (
            None,
            model_name,
            (
                f"{hint_env} does not look like a Hugging Face token. "
                "Expected prefix 'hf_' for router.huggingface.co."
            ),
        )

    if OpenAI is None:
        return None, model_name, "openai package not installed"

    try:
        client = OpenAI(
            base_url=api_base_url,
            api_key=api_key,
        )

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": "Say hello in one word."},
            ],
            temperature=0,
            max_tokens=5,
        )

        text = (response.choices[0].message.content or "").strip()
        print(f"[INFO] proxy_check={text}", flush=True)
        print("[INFO] Proxy verification successful", flush=True)
        return client, model_name, None

    except Exception as exc:
        raw_error = str(exc)
        if "401" in raw_error and "Invalid username or password." in raw_error:
            raw_error = (
                f"{raw_error} Check API_KEY/HF_TOKEN value and token permissions on Hugging Face."
            )
        error_msg = f"LLM proxy failed: {raw_error}"
        print(f"[WARN] {error_msg}", flush=True)
        return None, model_name, error_msg


@lru_cache(maxsize=1)
def _cached_openai_client() -> Tuple[Optional[Any], str, Optional[str]]:
    return _make_openai_client()


app = FastAPI(title="PolicyPilot OpenEnv API", version="1.0.0")
env = PolicyPilotEnv(seed=42)
baseline_agent = BaselineComplianceAgent()


class ResetRequest(BaseModel):
    difficulty: str = Field(default="easy")
    scenario_variant: Optional[int] = None


class StepRequest(BaseModel):
    action_type: str
    case_id: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    reason: str = Field(default="Policy-grounded decision.")


class RunEpisodeRequest(BaseModel):
    difficulty: str = Field(default="easy")
    max_steps: int = Field(default=8)
    seed: int = Field(default=42)
    require_remote_llm: bool = Field(default=False)


class RunBenchmarkRequest(BaseModel):
    difficulties: List[str] = Field(default_factory=lambda: ["easy", "medium", "hard"])
    max_steps: int = Field(default=8)
    seed: int = Field(default=42)
    require_remote_llm: bool = Field(default=False)


def _compact_json(value: Dict[str, Any]) -> str:
    return json.dumps(value, separators=(",", ":"))


def _sanitize_error_text(message: Optional[str]) -> str:
    if not message:
        return "null"
    return re.sub(r"\s+", " ", message.strip())


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract first valid JSON object from model output.
    """
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    matches = re.findall(r"\{.*\}", text, re.DOTALL)
    for m in matches:
        try:
            obj = json.loads(m)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None


def _normalize_action(action: Dict[str, Any], observation: Dict[str, Any]) -> Dict[str, Any]:
    case_id = observation.get("case", {}).get("id", "UNKNOWN")
    return {
        "action_type": action.get("action_type"),
        "case_id": action.get("case_id", case_id),
        "payload": action.get("payload", {}),
        "reason": action.get("reason", "Policy-grounded decision."),
    }


def _fallback_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    return _normalize_action(baseline_agent.act(observation), observation)


def _llm_action(client: Any, model_name: str, observation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ask LLM for action. Must return valid normalized JSON action.
    """
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        max_tokens=300,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a compliance review agent.\n"
                    "Return ONLY a valid JSON object with keys:\n"
                    "action_type, case_id, payload, reason.\n"
                    "Do not include markdown or explanations."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(observation),
            },
        ],
    )

    text = (response.choices[0].message.content or "").strip()
    parsed = _extract_json_object(text)

    if parsed is None:
        raise RuntimeError(f"Invalid JSON from LLM: {text[:300]}")

    return _normalize_action(parsed, observation)


def _select_action(observation: Dict[str, Any], llm_client: Optional[Any], model_name: str) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Select action using LLM if available, otherwise baseline fallback.

    IMPORTANT:
    If LLM fails, we DO NOT crash benchmark execution.
    We fallback and log the error in structured stdout.
    """
    if llm_client is None:
        return _fallback_action(observation), None

    try:
        return _llm_action(llm_client, model_name, observation), None
    except Exception as exc:
        return _fallback_action(observation), f"llm_error:{exc}"


def _print_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={ENV_NAME} model={model}", flush=True)


def _print_step(i: int, action: Dict[str, Any], reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={i} action={_compact_json(action)} reward={reward:.2f} "
        f"done={str(done).lower()} error={_sanitize_error_text(error)}",
        flush=True,
    )


def _print_end(task: str, score: float, success: bool, rewards: List[float]) -> None:
    print(
        f"[END] task={task} score={score:.2f} steps={len(rewards)} "
        f"success={str(success).lower()} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


def run_single_task(
    difficulty: str,
    max_steps: int,
    seed: int,
    model_name: str,
    llm_client: Optional[Any],
) -> Dict[str, Any]:
    task = f"compliance_review_{difficulty}"
    _print_start(task, model_name)

    local_env = PolicyPilotEnv(seed=seed)
    obs = local_env.reset(difficulty=difficulty)

    rewards: List[float] = []
    done = False
    grade: Dict[str, Any] = {}

    for i in range(1, max_steps + 1):
        action, err = _select_action(obs, llm_client, model_name)

        obs, reward, done, info = local_env.step(action)
        rewards.append(reward)

        _print_step(i, action, reward, done, err)

        if done:
            break

    grade = local_env.grade()
    success = grade.get("success", False)
    score = grade.get("score", 0.0)

    _print_end(task, score, success, rewards)

    return {
        "task": difficulty,
        "score": score,
        "success": success,
        "components": grade.get("components", {}),
        "subscores": grade.get("subscores", {}),
        "episode_trace": grade.get("episode_trace", {}),
    }


def run_benchmark(
    difficulties: List[str],
    max_steps: int = 8,
    seed: int = 42,
    require_remote_llm: bool = False,
) -> Dict[str, Any]:
    """
    Benchmark entrypoint.

    Behavior:
    - Attempt proxy usage if env vars exist.
    - Never crash if proxy init fails unless strict mode is enabled.
    - Always produce structured stdout.
    """
    strict_remote_llm = _should_require_remote_llm(require_remote_llm)
    startup_error = None

    try:
        llm_client, model_name, startup_error = _cached_openai_client()
        client_mode = "openai" if llm_client is not None else "baseline_fallback"
    except Exception as exc:
        startup_error = str(exc)
        print(f"[WARN] startup_error={startup_error}", flush=True)

        llm_client = None
        model_name = _resolve_model_name()
        client_mode = "baseline_fallback"

    if strict_remote_llm and llm_client is None:
        raise RuntimeError(startup_error or "Remote LLM mode required but proxy client was not initialized.")

    results = []
    for difficulty in difficulties:
        results.append(
            run_single_task(
                difficulty=difficulty,
                max_steps=max_steps,
                seed=seed,
                model_name=model_name,
                llm_client=llm_client,
            )
        )

    return {
        "env": ENV_NAME,
        "model": model_name,
        "client_mode": client_mode,
        "require_remote_llm": strict_remote_llm,
        "startup_error": startup_error,
        "results": results,
    }


def run_episode(
    difficulty: str = "easy",
    max_steps: int = 8,
    seed: int = 42,
    require_remote_llm: bool = False,
) -> Dict[str, Any]:
    result = run_benchmark(
        [difficulty],
        max_steps=max_steps,
        seed=seed,
        require_remote_llm=require_remote_llm,
    )
    return result["results"][0]


@app.get("/")
def root():
    return {
        "name": "PolicyPilot OpenEnv API",
        "status": "ok",
        "endpoints": ["/health", "/reset", "/step", "/state", "/grade", "/act", "/run_episode", "/run_benchmark"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
def reset(req: Optional[ResetRequest] = None):
    """
    Must support:
    - POST /reset
    - POST /reset with {}
    - POST /reset with {"difficulty":"easy"}
    """
    try:
        parsed_req = req or ResetRequest()
        observation = env.reset(
            difficulty=parsed_req.difficulty,
            scenario_variant=parsed_req.scenario_variant,
        )
        return {
            "observation": observation,
            "state": env.state(),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step")
def step(req: StepRequest):
    try:
        action = {
            "action_type": req.action_type,
            "case_id": req.case_id,
            "payload": req.payload,
            "reason": req.reason,
        }
        observation, reward, done, info = env.step(action)
        return {
            "observation": observation,
            "reward": reward,
            "done": done,
            "info": info,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state")
def state():
    return env.state()


@app.get("/grade")
def grade():
    return env.grade()


@app.post("/act")
def act():
    """
    Generate one action for the CURRENT env state using remote LLM if configured,
    otherwise baseline fallback.
    """
    try:
        observation = {
            "task": env.current_task,
            "case": env.current_case,
            "documents": env.current_documents,
            "policy": env.current_policy,
            "state": env.state(),
            "allowed_actions": env.allowed_actions(),
        }

        try:
            llm_client, model_name, _ = _cached_openai_client()
        except Exception:
            llm_client, model_name = None, _resolve_model_name()

        action, error = _select_action(observation, llm_client, model_name)
        return {
            "action": action,
            "error": error,
            "model": model_name,
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/run_episode")
def run_episode_route(req: RunEpisodeRequest):
    try:
        return run_episode(
            difficulty=req.difficulty,
            max_steps=req.max_steps,
            seed=req.seed,
            require_remote_llm=req.require_remote_llm,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/run_benchmark")
def run_benchmark_route(req: RunBenchmarkRequest):
    try:
        return run_benchmark(
            difficulties=req.difficulties,
            max_steps=req.max_steps,
            seed=req.seed,
            require_remote_llm=req.require_remote_llm,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true", help="Run FastAPI server")
    parser.add_argument("--run-benchmark", action="store_true", help="Run benchmark locally")
    parser.add_argument("--difficulties", type=str, default="easy,medium,hard")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--require-remote-llm",
        action="store_true",
        help="Fail fast if remote LLM proxy initialization does not succeed.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)

    args = parser.parse_args()

    if args.serve:
        if _should_require_remote_llm(args.require_remote_llm):
            llm_client, _, startup_error = _cached_openai_client()
            if llm_client is None:
                print(f"[FATAL] {startup_error}", flush=True)
                raise SystemExit(2)
        uvicorn.run(app, host=args.host, port=args.port)
        return

    difficulties = [d.strip() for d in args.difficulties.split(",") if d.strip()]
    try:
        result = run_benchmark(
            difficulties=difficulties,
            max_steps=args.max_steps,
            seed=args.seed,
            require_remote_llm=args.require_remote_llm,
        )
    except RuntimeError as exc:
        print(f"[FATAL] {exc}", flush=True)
        raise SystemExit(2)

    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

