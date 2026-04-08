---
title: PolicyPilot
emoji: 🧾
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# PolicyPilot: Compliance Review Benchmark (OpenEnv)

PolicyPilot is a deterministic benchmark for enterprise expense and reimbursement compliance review.
It evaluates policy grounding, workflow correctness, safe decision making, and audit-quality reasoning.

It is designed to test whether an agent can make policy-compliant decisions under ambiguity, incomplete evidence, and workflow constraints, similar to real internal finance/compliance review systems used in enterprises.

## Architecture Overview

PolicyPilot consists of:

* a deterministic environment with typed state transitions
* a structured action interface for safe agent interaction
* deterministic graders with structured subscores
* a rule-based baseline agent
* an optional LLM-driven agent evaluated through OpenEnv-compatible APIs

This allows both symbolic and LLM-based agents to be tested under the same environment contract.

## Why This Benchmark is Challenging

Unlike simple classification tasks, PolicyPilot requires:

* multi-step reasoning across policy rules
* handling incomplete or conflicting evidence
* safe decision-making under uncertainty
* structured action outputs with audit reasoning
* workflow correctness instead of just final-answer correctness

This makes it closer to real-world enterprise compliance workflows than a standard single-turn benchmark.

## Highlights

* OpenEnv-style environment API: `reset()`, `step()`, `state()`
* Strict typed action schema with validator penalties
* Dense rewards with explicit unsafe-action penalties
* Deterministic graders with structured subscores
* Difficulty progression (`easy`, `medium`, `hard`)
* Hard suite includes ambiguity, conflicts, duplicate claims, and fraud signals
* API server mode plus competition runner mode in `inference.py`

## Repository Structure

```text id="ls0z3r"
policypilot/
  openenv.yaml
  Dockerfile
  inference.py
  README.md
  requirements.txt
  src/
    env/
      core.py
      models.py
      state.py
      actions.py
      rewards.py
      policy_engine.py
    tasks/
      easy.py
      medium.py
      hard.py
    graders/
      easy.py
      medium.py
      hard.py
    agents/
      baseline.py
  tests/
    test_reset.py
    test_step_valid.py
    test_invalid_action.py
    test_grader.py
    test_reward.py
    test_reproducibility.py
    test_inference_proxy_modes.py
```

## Environment API

`PolicyPilotEnv` exposes:

* `reset(difficulty: str, scenario_variant: Optional[int] = None) -> Observation`
* `step(action: dict) -> (Observation, Reward, Done, Info)`
* `state() -> InternalState`
* `grade() -> Deterministic grade report`

`state()` and `grade()` include `episode_trace` for full step-by-step auditability.

## Action Schema (Strict)

```json id="v9r4u5"
{
  "action_type": "request_missing_info",
  "case_id": "C-102",
  "payload": {
    "fields": ["receipt_tax", "manager_approval"]
  },
  "reason": "Tax and manager approval are required above policy thresholds."
}
```

### Allowed actions

* `approve_case`
* `reject_case`
* `request_missing_info`
* `escalate_case`
* `flag_for_manual_review`
* `add_audit_note`

Invalid actions are penalized (`-0.2`) and kept in trace output.

## Reward Function

```text id="jlwmrt"
reward =
  0.25 * violation_detection +
  0.25 * evidence_handling +
  0.30 * decision_correctness +
  0.20 * audit_quality +
  penalties
```

### Penalties

* `invalid_action`: `-0.2`
* `unsafe_approval`: `-0.5`
* `repeated_useless_action`: `-0.1`
* `skipped_required_steps`: `-0.3`

## Hard Cases

Hard scenarios include:

* Mixed personal + business expenses
* Conflicting rule hierarchy paths
* Partial and scope-limited approvals
* Non-USD FX documentation requirements
* Duplicate-claim detection (`flag_for_manual_review`)
* Fraudulent receipt rejection
* Pending exception escalation

## Deterministic Grading

Each grade output contains:

* `score` in `[0.0, 1.0]`
* `success` with threshold `>= 0.85`
* `components`
* `subscores`
* `episode_trace`

Example:

```json id="p26d1s"
{
  "difficulty": "hard",
  "score": 0.72,
  "success": false,
  "subscores": {
    "policy_interpretation": 0.8,
    "conflict_resolution": 0.4,
    "final_decision": 0.2,
    "workflow_correctness": 1.0,
    "audit_compliance": 0.6
  }
}
```

## Baseline Benchmark Snapshot (seed=42)

| Task   |  Score |
| ------ | -----: |
| easy   |   1.00 |
| medium |   0.92 |
| hard   |   0.70 |
| avg    | 0.8733 |

This profile is intentional: the baseline handles straightforward policy checks but degrades on harder ambiguity and fraud-like cases.

## Inference Modes

`inference.py` supports two modes:

1. Competition benchmark runner
2. API server (`--serve`)

### Competition Runner

```bash id="e7i6c7"
cd policypilot
python inference.py --run-benchmark --difficulties easy,medium,hard --max-steps 8 --seed 42
```

### API Server

```bash id="vgvtrq"
cd policypilot
python inference.py --serve
```

## LLM Proxy Compliance (Hackathon Requirement)

PolicyPilot uses the injected LiteLLM-compatible proxy for all remote LLM calls.

### Remote LLM configuration

The OpenAI-compatible client is initialized using:

* `API_BASE_URL`
* `API_KEY` (or `HF_TOKEN`, `HUGGINGFACEHUB_API_TOKEN`)
* `MODEL_NAME` (optional)
* `REQUIRE_REMOTE_LLM` (optional, set to `1` for submission strict mode)

Reference template: `.env.example`

### PowerShell example

```powershell
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:API_KEY="hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
$env:MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
$env:REQUIRE_REMOTE_LLM="1"
python inference.py --run-benchmark --difficulties easy,medium,hard --max-steps 8 --seed 42 --require-remote-llm
```

### Compliance guarantees

* No hardcoded API credentials are used.
* `API_BASE_URL` is always respected when remote LLM mode is enabled.
* A mandatory verification call is executed at startup to ensure the proxy is actually used.
* Placeholder tokens are rejected with actionable errors before any remote call.
* If `REQUIRE_REMOTE_LLM=1` (or `--require-remote-llm`), invalid proxy credentials fail fast and exit non-zero.
* If strict mode is off, PolicyPilot can use a baseline fallback for local development.

This ensures compatibility with environments that require proxy-backed validation.

### Troubleshooting `401 Invalid username or password`

If you see:

```text
LLM proxy failed: Error code: 401 - {'error': 'Invalid username or password.'}
```

check these first:

* `API_KEY` is a real Hugging Face token (starts with `hf_`), not `hf_your_actual_token_here`.
* The token has access to Inference Providers / Router usage.
* `API_BASE_URL` is `https://router.huggingface.co/v1`.
* `MODEL_NAME` is available on the router and your account has access.

## Live Deployment

Hugging Face Space:
https://divyam-r25-policypilot.hf.space

### Public endpoints

* `GET /health`
* `POST /reset`
* `POST /step`
* `GET /state`
* `GET /grade`
* `POST /act`
* `POST /run_episode`
* `POST /run_benchmark`

## Validation

Verified locally and on Hugging Face Space:

* `pytest -q`
* `docker build -t policypilot .`
* `docker run -p 7860:7860 policypilot`
* public endpoint tests on Hugging Face Space
* benchmark runner tested with injected proxy environment variables
* strict remote mode benchmark dry run (`REQUIRE_REMOTE_LLM=1`)

## Docker

```bash id="79utru"
cd policypilot
docker build -t policypilot .
docker run -p 7860:7860 policypilot
```

Container launches the API server via:

```bash id="ofvrxk"
python inference.py --serve
```

## OpenEnv Config

`openenv.yaml` defines:

* deterministic evaluation metadata
* environment entrypoint
* reward shaping
* penalties
* task difficulties
* success threshold

## Tests

```bash id="h2w9zh"
cd policypilot
python -m pytest -q
```

### Coverage includes

* reset correctness
* valid step transitions
* invalid action penalties
* reward correctness
* grader outputs
* reproducibility (fixed seed)
* difficulty ordering behavior
* proxy strict/fallback mode behavior

## Hugging Face Spaces (Docker SDK)

1. Create a Docker Space
2. Push this repository
3. Ensure port `7860` is exposed
4. Validate `/health`, `/reset`, `/step`, `/state`, and `/grade`

## Submission Checklist

* OpenEnv API implemented (`reset`, `step`, `state`, `grade`)
* Deterministic graders with structured subscores
* Multi-difficulty tasks (`easy`, `medium`, `hard`)
* Reward shaping with penalties
* Dockerized deployment
* Hugging Face Space deployment
* Public API endpoints verified
* LLM proxy integration using `API_BASE_URL` and `API_KEY`/`HF_TOKEN`
* Strict submission mode verified (`--require-remote-llm`, output shows `"client_mode": "openai"` and `"startup_error": null`)
* No hardcoded credentials
* Baseline fallback restricted to local development mode
