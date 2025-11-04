"""Experiment logging utilities for MAPPO analysis."""
from __future__ import annotations

import json
from typing import Dict, List, Optional


class ExperimentDataLogger:
    """Collect and persist experiment level statistics."""

    def __init__(self, nagents: int):
        self.nagents = nagents
        self.cumulative_influences: Dict[str, List[float]] = {}
        self.directional_derivatives: Dict[str, List[float]] = {}
        self.attack_metrics: List[Dict] = []

    def log_cumulative_influences(self, action_influences_history, seed: int, episode_length: int):
        for t in range(min(len(action_influences_history), episode_length)):
            influences = action_influences_history[t]
            for i in range(self.nagents):
                for j in range(self.nagents):
                    if i == j:
                        continue
                    key = f'seed{seed}_i{i}_j{j}'
                    value = float(influences[j][i])
                    self.cumulative_influences.setdefault(key, []).append(value)

    def log_directional_derivatives(self, directional_derivatives_history, seed: int, episode_length: int):
        for t in range(min(len(directional_derivatives_history), episode_length)):
            derivatives = directional_derivatives_history[t]
            for i in range(self.nagents):
                for j in range(self.nagents):
                    if i == j:
                        continue
                    key = f'seed{seed}_i{i}_j{j}'
                    value = float(derivatives[j][i])
                    self.directional_derivatives.setdefault(key, []).append(value)

    def log_attack_metrics(self, metrics: Dict, pair_info: Optional[Dict] = None):
        entry = metrics.copy()
        if pair_info:
            entry.update(pair_info)
        self.attack_metrics.append(entry)

    def save_to_json(self, path: str):
        data = {
            'cumulative_influences': self.cumulative_influences,
            'directional_derivatives': self.directional_derivatives,
            'attack_metrics': self.attack_metrics,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
