---

title: PolicyPilot
emoji: 🧾
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
-------------

# PolicyPilot: Compliance Review Benchmark (OpenEnv)

PolicyPilot is a deterministic benchmark for enterprise expense and reimbursement compliance review.
It evaluates policy grounding, workflow correctness, safe decision making, and audit-quality reasoning.

It is designed to test whether an agent can make policy-compliant decisions under ambiguity, incomplete evidence, and workflow constraints, similar to real internal finance/compliance review systems used in enterprises.

It is intended as a realistic benchmark for training and evaluating enterprise review agents under OpenEnv-style interaction constraints.

---

## Architecture Overview

PolicyPilot consists of:

* a deterministic environment with typed state transitions
* a structured action interface for safe agent interaction
* deterministic graders with structured subscores
* a rule-based baseline agent
* an optional LLM-driven agent evaluated through OpenEnv-compatible APIs

This allows both symbolic and LLM-based agents to be tested under the same environment contract.

---

## Why This Benchmark is Challenging

Unlike simple classification tasks, PolicyPilot requires:

* multi-step reasoning across policy rules
* handling incomplete or conflicting evidence
* safe decision-making under uncertainty
* structured action outputs with audit reasoning
* workflow correctness instead of just final-answer correctness

This makes it closer to real-world enterprise compliance workflows than a standard single-turn benchmark.

---

## Highlights

* OpenEnv-style environment API: `reset()`, `step()`, `state()`
* Strict typed action schema with validator penalties
* Dense rewards with explicit unsafe-action penalties
* Deterministic graders with structured subscores
* Difficulty progression (`easy`, `medium`, `hard`)
* Hard suite includes ambiguity, conflicts, duplicate claims, and fraud signals
* API server mode plus competition runner mode in `inference.py`

---

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
    test_inference_proxy_modes.py
```

---

## Environment API

`PolicyPilotEnv` exposes:

* `reset(difficulty: str, scenario_variant: Optional[int] = None) -> Observation`
* `step(action: dict) -> (Observation, Reward, Done, Info)`
* `state() -> InternalState`
* `grade() -> Deterministic grade report`

`state()` and `grade()` include `episode_trace` for full auditability.

---

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

### Allowed actions

* `approve_case`
* `reject_case`
* `request_missing_info`
* `escalate_case`
* `flag_for_manual_review`
* `add_audit_note`

Invalid actions are penalized (`-0.2`) and kept in trace output.

---

## Reward Function

```text
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

---

## Hard Cases

Hard scenarios include:

* Mixed personal + business expenses
* Conflicting rule hierarchy paths
* Partial approvals
* FX documentation requirements
* Duplicate claims
* Fraud signals
* Escalation scenarios

---

## Deterministic Grading

Each grade output contains:

* `score` in `(0.0, 1.0)` (strictly inside bounds)
* `success` threshold `>= 0.85`
* `components`, `subscores`, `episode_trace`

---

## Baseline Benchmark Snapshot (seed=42)

| Task   | Score |
| ------ | ----: |
| easy   | 0.9999 |
| medium |  0.92 |
| hard   | 0.9999 |
| avg    | 0.9733 |

---

## Inference Modes

### Competition Runner

```bash
python inference.py
```

(default runs benchmark)

---

### API Server

```bash
python inference.py --serve
```

---

## LLM Proxy Integration

PolicyPilot supports OpenAI-compatible proxy usage via environment variables:

* `API_BASE_URL`
* `API_KEY` or `HF_TOKEN`
* `MODEL_NAME`

### Example (local testing)

```powershell
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:API_KEY="hf_your_token_here"
$env:MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
python inference.py
```

---

## Proxy / LLM Behavior

* No hardcoded API credentials are used
* Proxy is used automatically when environment variables are present
* A lightweight startup request verifies proxy connectivity
* If proxy fails, the system logs a warning and safely falls back
* Benchmark execution always continues
* Structured output is always emitted

---

## Validator Compatibility Notes

PolicyPilot is designed to work reliably with automated evaluation pipelines:

* `inference.py` runs without required flags
* Outputs `[START]`, `[STEP]`, `[END]` blocks
* `/reset` supports empty and JSON requests
* Server runs on port `7860`
* Proxy usage uses injected environment variables
* Fallback ensures no crashes during evaluation

---

## Live Deployment

Hugging Face Space:
https://divyam-r25-policypilot.hf.space

### Endpoints

* `GET /health`
* `POST /reset`
* `POST /step`
* `GET /state`
* `GET /grade`
* `POST /act`
* `POST /run_episode`
* `POST /run_benchmark`

---

## Validation

Tested via:

* `pytest -q`
* Docker build + run
* HF Space deployment
* benchmark CLI execution

---

## Docker

```bash
docker build -t policypilot .
docker run -p 7860:7860 policypilot
```

---

## Tests

```bash
python -m pytest -q
```

---

## Submission Checklist

* OpenEnv API implemented
* Multi-difficulty tasks
* Deterministic grading
* Reward shaping
* Docker deployment
* HF Space deployed
* Public endpoints working
* Proxy integration via env variables
* No hardcoded secrets
* Benchmark outputs structured logs

---

## License

MIT License
