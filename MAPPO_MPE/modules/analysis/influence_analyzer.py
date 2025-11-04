"""Utilities for selecting influential timesteps in MAPPO analysis."""
from __future__ import annotations

from typing import List, Tuple


class InfluenceAnalyzer:
    """Analyze influence histories to identify candidate attack timesteps."""

    @staticmethod
    def find_influence_timesteps(
        action_influences_history: List[List[List[float]]],
        agent_i: int,
        agent_j: int,
        atk_steps_limit: int,
        k_steps: int = 1,
    ) -> Tuple[List[int], List[int]]:
        timesteps = min(atk_steps_limit, len(action_influences_history))
        influence_values: List[Tuple[int, float]] = []

        for t in range(timesteps):
            influence = action_influences_history[t][agent_j][agent_i]
            influence_values.append((t, influence))

        if not influence_values:
            return [], []

        sorted_by_value = sorted(influence_values, key=lambda item: item[1], reverse=True)
        max_influences = sorted(sorted_by_value[:k_steps])

        sorted_by_value_asc = sorted(influence_values, key=lambda item: item[1])
        min_influences = sorted(sorted_by_value_asc[:k_steps])

        return [t for t, _ in max_influences], [t for t, _ in min_influences]
