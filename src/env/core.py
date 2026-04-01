from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ..graders import grade_episode
from ..tasks import get_scenario
from .actions import ActionValidator
from .models import FINAL_ACTIONS, Scenario
from .policy_engine import PolicyEngine
from .rewards import RewardEngine
from .state import EnvState


class PolicyPilotEnv:
    """
    OpenEnv-style benchmark environment.

    Required API:
      - reset() -> Observation
      - step(action) -> (Observation, Reward, Done, Info)
      - state() -> InternalState
    """

    def __init__(self, difficulty: str = "easy", seed: int = 42) -> None:
        self.default_difficulty = difficulty
        self.seed = seed
        self.validator = ActionValidator()
        self.policy_engine = PolicyEngine()
        self.reward_engine = RewardEngine()
        self.current_scenario: Optional[Scenario] = None
        self._state: Optional[EnvState] = None
        self._episode_counter_by_difficulty: Dict[str, int] = {}

    def reset(
        self,
        difficulty: Optional[str] = None,
        scenario_variant: Optional[int] = None,
    ) -> Dict[str, Any]:
        selected_difficulty = (difficulty or self.default_difficulty).lower().strip()
        episode_index = self._episode_counter_by_difficulty.get(selected_difficulty, 0)
        scenario = get_scenario(
            selected_difficulty,
            seed=self.seed,
            episode_index=episode_index,
            variant=scenario_variant,
        )
        self._episode_counter_by_difficulty[selected_difficulty] = episode_index + 1
        self.current_scenario = scenario
        self._state = EnvState(
            case_id=scenario.case.id,
            difficulty=scenario.difficulty,
            step_count=0,
            max_steps=scenario.max_steps,
            decision=None,
            missing_fields=[],
            history=[],
            reward_accumulated=0.0,
            done=False,
            critical_failure=False,
            policy_used_correctly=False,
            reviewed=False,
            audit_notes=[],
            policy_trace=[],
            expected_decision=scenario.gold.decision,
            expected_missing_fields=list(scenario.gold.missing_fields),
            last_action_signature=None,
        )
        return self._build_observation()

    def step(self, raw_action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.current_scenario is None or self._state is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")

        if self._state.done:
            return self._build_observation(), 0.0, True, {"message": "Episode is already done."}

        scenario = self.current_scenario
        state = self._state

        analysis = self.policy_engine.analyze(scenario)
        validation = self.validator.validate(
            raw_action,
            case_id=scenario.case.id,
            allowed_actions=scenario.allowed_actions,
        )
        reward_result = self.reward_engine.calculate(
            state=state,
            scenario=scenario,
            validation=validation,
            analysis=analysis,
            validator=self.validator,
        )

        state.step_count += 1
        termination_reason: Optional[str] = None

        if validation.is_valid and validation.action is not None:
            action = validation.action
            state.reviewed = True
            state.policy_trace = list(analysis.policy_trace)
            state.policy_used_correctly = action.action_type in {
                analysis.recommended_action,
                scenario.gold.decision,
            }

            if action.action_type == "add_audit_note":
                note = str(action.payload.get("note", action.reason))
                state.audit_notes.append(note)

            if action.action_type == "request_missing_info":
                state.missing_fields = list(analysis.required_missing_fields)
                requested_fields = set(action.payload.get("fields", []))
                expected_fields = set(analysis.required_missing_fields)
                if expected_fields and expected_fields.issubset(requested_fields):
                    # For this benchmark, a complete request can be a valid terminal workflow action.
                    state.decision = "request_missing_info"
                    state.done = True
                    termination_reason = "missing_info_requested"
            elif action.action_type in FINAL_ACTIONS:
                state.decision = action.action_type
                state.done = True
                termination_reason = "final_decision"
                if action.action_type == "approve_case" and not analysis.safe_to_approve:
                    state.critical_failure = True
                    termination_reason = "critical_failure_unsafe_approval"

            state.last_action_signature = self.validator.action_signature(action)
        else:
            state.policy_used_correctly = False

        if state.step_count >= state.max_steps and not state.done:
            state.done = True
            termination_reason = "max_steps_reached"

        state.reward_accumulated = round(state.reward_accumulated + reward_result.reward, 4)

        history_entry = {
            "step": state.step_count,
            "action": validation.action.to_dict() if validation.action else raw_action,
            "validation": validation.to_dict(),
            "analysis": analysis.to_dict(),
            "reward": reward_result.to_dict(),
            "state_snapshot": {
                "decision": state.decision,
                "missing_fields": list(state.missing_fields),
                "done": state.done,
            },
        }
        state.history.append(history_entry)

        info: Dict[str, Any] = {
            "validation": validation.to_dict(),
            "analysis": analysis.to_dict(),
            "reward_breakdown": reward_result.components,
            "penalties": reward_result.penalties,
            "recommended_action": analysis.recommended_action,
            "termination_reason": termination_reason,
            "episode_trace": list(state.history),
        }
        if state.done:
            info["grade"] = self.grade()

        return self._build_observation(), reward_result.reward, state.done, info

    def state(self) -> Dict[str, Any]:
        if self._state is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        scenario = self.current_scenario
        payload = self._state.to_dict()
        payload["scenario_id"] = scenario.id if scenario else None
        payload["success_metric_threshold"] = 0.85
        payload["episode_trace"] = list(self._state.history)
        return payload

    def grade(self) -> Dict[str, Any]:
        if self._state is None or self.current_scenario is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        result = grade_episode(self._state, self.current_scenario)
        result["episode_trace"] = list(self._state.history)
        return result

    def _build_observation(self) -> Dict[str, Any]:
        if self.current_scenario is None or self._state is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")

        step_number = min(self._state.step_count + 1, self._state.max_steps)
        observation = self.current_scenario.to_observation(
            step=step_number,
            missing_fields=self._state.missing_fields,
            decision=self._state.decision,
            reviewed=self._state.reviewed,
        )
        return observation.to_dict()
