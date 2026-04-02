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

## Highlights

- OpenEnv-style environment API: `reset()`, `step()`, `state()`
- Strict typed action schema with validator penalties
- Dense rewards with explicit unsafe-action penalties
- Deterministic graders with structured subscores
- Difficulty progression (`easy`, `medium`, `hard`)
- Hard suite includes ambiguity, conflicts, duplicate claims, and fraud signals
- API server mode plus competition runner mode in `inference.py`

## Repository Structure

```text
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
```

## Environment API

`PolicyPilotEnv`:

- `reset(difficulty: str, scenario_variant: Optional[int] = None) -> Observation`
- `step(action: dict) -> (Observation, Reward, Done, Info)`
- `state() -> InternalState`
- `grade() -> Deterministic grade report`

`state()` and `grade()` include `episode_trace` for full step-by-step auditability.

## Action Schema (Strict)

```json
{
  "action_type": "request_missing_info",
  "case_id": "C-102",
  "payload": {
    "fields": ["receipt_tax", "manager_approval"]
  },
  "reason": "Tax and manager approval are required above policy thresholds."
}
```

Allowed actions:

- `approve_case`
- `reject_case`
- `request_missing_info`
- `escalate_case`
- `flag_for_manual_review`
- `add_audit_note`

Invalid actions are penalized (`-0.2`) and kept in trace output.

## Reward Function

```text
reward =
  0.25 * violation_detection +
  0.25 * evidence_handling +
  0.30 * decision_correctness +
  0.20 * audit_quality +
  penalties
```

Penalties:

- `invalid_action`: `-0.2`
- `unsafe_approval`: `-0.5`
- `repeated_useless_action`: `-0.1`
- `skipped_required_steps`: `-0.3`

## Hard Cases

Hard scenarios include:

- Mixed personal + business expenses
- Conflicting rule hierarchy paths
- Partial and scope-limited approvals
- Non-USD FX documentation requirements
- Duplicate-claim detection (`flag_for_manual_review`)
- Fraudulent receipt rejection
- Pending exception escalation

## Deterministic Grading

Each grade output contains:

- `score` in `[0.0, 1.0]`
- `success` with threshold `>= 0.85`
- `components`
- `subscores`
- `episode_trace`

Example:

```json
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

| Task | Score |
|------|------:|
| easy | 1.00 |
| medium | 0.92 |
| hard | 0.70 |
| avg | 0.8733 |

This profile is intentional: the baseline handles straightforward policy checks but degrades on harder ambiguity and duplicate/fraud cases.

## Inference Modes

`inference.py` supports two modes:

1. Competition benchmark runner (default)
2. API server (`--serve`)

### Competition Runner

```bash
cd policypilot
python inference.py --run-benchmark --difficulties easy,medium,hard --max-steps 8 --seed 42
```

Runner uses env vars:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

If those are not provided or the client fails to initialize, it automatically falls back to the baseline agent.

Required log format emitted to stdout:

```text
[START] task=<task_name> env=<benchmark> model=<model_name>
[STEP]  step=<n> action=<action_json> reward=<0.00> done=<true|false> error=<msg|null>
[END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
```
## Live Deployment

Hugging Face Space:
https://divyam-r25-policypilot.hf.space

### Public endpoints
- `GET /health`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /grade`

### API Server

## Validation

Verified locally and on Hugging Face Space:
- `pytest -q`
- `docker build -t policypilot .`
- `docker run -p 7860:7860 policypilot`
- Public endpoint tests on Hugging Face Space

```bash
cd policypilot
python inference.py --serve
```

Endpoints:

- `GET /health`
- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /grade`
- `POST /act`
- `POST /run_episode`
- `POST /run_benchmark`

## Docker

Build and run:

```bash
cd policypilot
docker build -t policypilot .
docker run -p 7860:7860 policypilot
```

Container launches API mode via `python inference.py --serve`.

## OpenEnv Config

`openenv.yaml` includes:

- deterministic evaluation metadata
- environment entrypoint
- reward weights and penalties
- task difficulties and success threshold

## Tests

Run:

```bash
cd policypilot
python -m pytest -q
```

Covered checks include:

- `reset()` behavior and state clearing
- valid step transition
- invalid action penalties
- reward behavior (unsafe approval and correct decision)
- grader outputs/subscores
- reproducibility with fixed seed
- baseline hard score not higher than easy score

## HuggingFace Spaces (Docker SDK)

1. Create a new Docker Space.
2. Push this directory.
3. Ensure exposed port is `7860`.
4. Validate `/health`, `/reset`, `/step` on deployed URL.
