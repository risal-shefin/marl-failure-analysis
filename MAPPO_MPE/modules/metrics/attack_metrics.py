"""Attack metrics computation for MAPPO fault analysis."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from ..constants import K_SIGMA

WATCH_WINDOW = 15


class AttackMetricsComputer:
    """Compute metrics assessing the effect of targeted attacks."""

    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma

    def compute_attack_metrics(
        self,
        attack_results: Dict,
        normal_values: List[List[float]],
        normal_rewards: List[List[float]],
        ref_vals: List[List[float]],
        ref_std_devs: List[List[float]],
        observe_agent_j: int,
    ) -> Dict[str, float]:
        attack_timesteps = attack_results['attack_timesteps']
        values_history = attack_results['q_values_history']
        rewards_history = attack_results['rewards_history']
        taylor_errors_history = attack_results['taylor_errors_history']
        episode_length = attack_results['episode_length']

        if not attack_timesteps:
            return {}

        window_start = min(attack_timesteps)
        window_end = min(window_start + WATCH_WINDOW, episode_length - 1)

        metrics = {
            'max_value_drop': 0.0,
            'weighted_value_drop_sum': 0.0,
            'max_reward_drop': 0.0,
            'weighted_reward_drop_sum': 0.0,
            'max_abs_taylor_deviation': 0.0,
            'weighted_taylor_deviation_sum': 0.0,
            'exceed_rate': 0.0,
            'window_length': window_end - window_start + 1,
        }

        exceed_count = 0
        window_steps = 0

        for t in range(window_start, window_end + 1):
            if t >= len(values_history) or t >= len(normal_values):
                break
            weight = self.gamma ** (t - window_start)
            window_steps += 1

            value_drop = normal_values[t][observe_agent_j] - values_history[t][observe_agent_j]
            metrics['max_value_drop'] = max(metrics['max_value_drop'], value_drop)
            metrics['weighted_value_drop_sum'] += weight * value_drop

            reward_drop = normal_rewards[t][observe_agent_j] - rewards_history[t][observe_agent_j]
            metrics['max_reward_drop'] = max(metrics['max_reward_drop'], reward_drop)
            metrics['weighted_reward_drop_sum'] += weight * reward_drop

            if t < len(taylor_errors_history) and t < len(ref_vals[observe_agent_j]):
                taylor_error = taylor_errors_history[t][observe_agent_j]
                ref_mean = ref_vals[observe_agent_j][t]
                ref_std = ref_std_devs[observe_agent_j][t]
                deviation = abs(taylor_error - ref_mean)
                metrics['max_abs_taylor_deviation'] = max(metrics['max_abs_taylor_deviation'], deviation)
                metrics['weighted_taylor_deviation_sum'] += weight * deviation
                threshold = K_SIGMA * ref_std
                if deviation > threshold and not np.isclose(taylor_error, ref_mean, rtol=1e-5, atol=1e-5):
                    exceed_count += 1

        if window_steps > 0:
            metrics['exceed_rate'] = exceed_count / window_steps

        return metrics
