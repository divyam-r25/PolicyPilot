from __future__ import annotations

from typing import Callable, Dict, List, Optional

from ..env.models import Scenario
from .easy import build_scenario as build_easy_scenario
from .hard import build_scenarios as build_hard_scenarios
from .medium import build_scenario as build_medium_scenario

SCENARIO_POOLS: Dict[str, Callable[[], List[Scenario]]] = {
    "easy": lambda: [build_easy_scenario()],
    "medium": lambda: [build_medium_scenario()],
    "hard": build_hard_scenarios,
}


def get_scenario(
    difficulty: str,
    seed: int = 42,
    episode_index: int = 0,
    variant: Optional[int] = None,
) -> Scenario:
    normalized = difficulty.lower().strip()
    if normalized not in SCENARIO_POOLS:
        raise ValueError(
            f"Unsupported difficulty '{difficulty}'. Choose from {sorted(SCENARIO_POOLS)}."
        )
    scenarios = SCENARIO_POOLS[normalized]()
    if not scenarios:
        raise ValueError(f"No scenarios available for difficulty '{difficulty}'.")

    if variant is not None:
        selected_index = int(variant) % len(scenarios)
    elif len(scenarios) == 1:
        selected_index = 0
    else:
        selected_index = (seed + episode_index + sum(ord(ch) for ch in normalized)) % len(scenarios)
    return scenarios[selected_index]


def list_difficulties() -> List[str]:
    return sorted(SCENARIO_POOLS.keys())


def scenario_count(difficulty: str) -> int:
    normalized = difficulty.lower().strip()
    if normalized not in SCENARIO_POOLS:
        raise ValueError(
            f"Unsupported difficulty '{difficulty}'. Choose from {sorted(SCENARIO_POOLS)}."
        )
    return len(SCENARIO_POOLS[normalized]())
