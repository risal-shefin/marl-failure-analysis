"""Episode execution helpers for MAPPO analysis."""
from __future__ import annotations

import random
from collections import deque
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

from ..constants import K_SIGMA
from ..metrics import (
    AttackMetricsComputer,
    collect_agent_values,
    compute_pairwise_frob_norms,
    compute_taylor_error_policy,
)


class EpisodeRunner:
    """Run normal and attacked episodes for MAPPO analysis."""

    def __init__(self, runner, env, epsilon: float = 0.01):
        self.runner = runner
        self.env = env
        self.epsilon = epsilon
        self.attack_metrics = AttackMetricsComputer()
        self._last_action_masks: Optional[Iterable] = None

    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    # ------------------------------------------------------------------
    # Environment helpers
    # ------------------------------------------------------------------
    def _reset_env(self, seed: Optional[int] = None) -> Tuple[Iterable, Optional[Iterable]]:
        if seed is not None:
            try:
                result = self.env.reset(seed=seed)
            except TypeError:
                if hasattr(self.env, 'seed'):
                    self.env.seed(seed)
                result = self.env.reset()
        else:
            result = self.env.reset()

        states, masks = self._extract_states_and_masks(result)
        self._last_action_masks = masks
        return states, masks

    def _step_env(self, actions: Iterable) -> Tuple[Iterable, Iterable, Iterable, Dict, Optional[Iterable]]:
        result = self.env.step(actions)
        if not isinstance(result, tuple):
            raise ValueError('Environment step is expected to return a tuple')

        states = result[0]
        rewards = result[1]
        dones = result[2]
        info = result[3] if len(result) > 3 else {}
        masks = None

        if len(result) > 4:
            for extra in result[4:]:
                if self._is_mask_like(extra, states):
                    masks = extra
                    break

        self._last_action_masks = masks
        return states, rewards, dones, info, masks

    @staticmethod
    def _extract_states_and_masks(
        result: Union[Iterable, Tuple[Iterable, ...]]
    ) -> Tuple[Iterable, Optional[Iterable]]:
        if isinstance(result, tuple):
            states = result[0]
            masks = None
            for extra in result[1:]:
                if EpisodeRunner._is_mask_like(extra, states):
                    masks = extra
                    break
            return states, masks
        return result, None

    @staticmethod
    def _is_mask_like(candidate, states: Iterable) -> bool:
        if candidate is None:
            return False
        try:
            return len(candidate) == len(states)
        except TypeError:
            return False

    def _get_action_mask(self, agent_id: int) -> Optional[Iterable]:
        if self._last_action_masks is None:
            return None
        try:
            return self._last_action_masks[agent_id]
        except (IndexError, TypeError):
            return None

    def _select_action(self, state, agent_id: int):
        mask = self._get_action_mask(agent_id)
        action, dist = self.runner.agent_n.select_action(
            state,
            agent_id,
            evaluate=True,
            action_mask=mask,
            return_dist=True,
        )
        return int(action), dist

    def run_normal_episode(self, seed: int, collect_frames: bool = False) -> Dict:
        self._set_seed(seed)
        states, _ = self._reset_env(seed)
        action_influences_history: List[List[List[float]]] = []
        directional_derivatives_history: List[List[List[float]]] = []
        values_history: List[List[float]] = []
        rewards_history: List[List[float]] = []
        taylor_history: List[List[float]] = []
        frames = [] if collect_frames else None
        timestep = 0

        if collect_frames:
            frame = self.env.render()
            frames.append(frame)

        while True:
            influences = compute_pairwise_frob_norms(self.runner, states)
            action_influences_history.append(influences)
            directional_derivatives_history.append(influences)
            values = collect_agent_values(self.runner, states)
            values_history.append(values)
            taylor_errors = compute_taylor_error_policy(self.runner, states, self.epsilon)
            taylor_history.append(taylor_errors)

            actions = []
            for agent_id in range(self.runner.args.N):
                action, _ = self._select_action(states[agent_id], agent_id)
                actions.append(action)

            next_states, rewards, dones, _, _ = self._step_env(actions)
            rewards_array = np.array(rewards)
            rewards_history.append(list(rewards_array.squeeze()))

            if collect_frames:
                frame = self.env.render()
                frames.append(frame)

            states = next_states
            timestep += 1

            done_flag = dones.all() if hasattr(dones, 'all') else all(dones)
            if done_flag:
                break

        result = {
            'action_influences_history': action_influences_history,
            'directional_derivatives_history': directional_derivatives_history,
            'q_values_history': values_history,
            'rewards_history': rewards_history,
            'taylor_errors_history': taylor_history,
            'episode_length': timestep,
        }
        if collect_frames:
            result['frames'] = frames
        return result

    def run_attacked_episode(
        self,
        seed: int,
        attack_agent_i: int,
        attack_timesteps: List[int],
        ref_vals: List[List[float]],
        ref_std_devs: List[List[float]],
        observe_agent: Optional[int] = None,
        collect_frames: bool = False,
    ) -> Dict:
        self._set_seed(seed)
        states, _ = self._reset_env(seed)
        frames = [] if collect_frames else None
        result_deques = [deque(maxlen=5) for _ in range(self.runner.args.N)]
        fault_timeline = []
        fault_first_detected: Dict[int, int] = {}
        action_influences_history: List[List[List[float]]] = []
        directional_derivatives_history: List[List[List[float]]] = []
        values_history: List[List[float]] = []
        rewards_history: List[List[float]] = []
        taylor_history: List[List[float]] = []
        attack_timesteps = sorted(attack_timesteps)
        observe_agent_id = observe_agent if observe_agent is not None else attack_agent_i
        timestep = 0

        if collect_frames:
            frames.append(self.env.render())

        while True:
            influences = compute_pairwise_frob_norms(self.runner, states)
            action_influences_history.append(influences)
            directional_derivatives_history.append(influences)
            values = collect_agent_values(self.runner, states)
            values_history.append(values)
            taylor_errors = compute_taylor_error_policy(self.runner, states, self.epsilon)
            taylor_history.append(taylor_errors)

            actions = []
            for agent_id in range(self.runner.args.N):
                action, dist = self._select_action(states[agent_id], agent_id)
                if timestep in attack_timesteps and agent_id == attack_agent_i:
                    probs = dist.probs.squeeze()
                    worst_action = int(torch.argmin(probs).item())
                    actions.append(worst_action)
                else:
                    actions.append(int(action))

            for agent_id in range(self.runner.args.N):
                result_deques[agent_id].append(taylor_errors[agent_id])
                if timestep < len(ref_vals[agent_id]):
                    detection_value = np.mean(result_deques[agent_id])
                    threshold = K_SIGMA * ref_std_devs[agent_id][timestep] if timestep < len(ref_std_devs[agent_id]) else 0.0
                    exceeds = abs(detection_value - ref_vals[agent_id][timestep]) > threshold and not np.isclose(
                        detection_value, ref_vals[agent_id][timestep], rtol=1e-5, atol=1e-5)
                    if exceeds:
                        if agent_id not in fault_first_detected:
                            fault_first_detected[agent_id] = timestep
                        fault_timeline.append({
                            'agent': agent_id,
                            't': timestep,
                            'taylor_deviation': abs(detection_value - ref_vals[agent_id][timestep]),
                        })

            next_states, rewards, dones, _, _ = self._step_env(actions)
            rewards_array = np.array(rewards)
            rewards_history.append(list(rewards_array.squeeze()))

            if collect_frames:
                frames.append(self.env.render())

            states = next_states
            timestep += 1

            done_flag = dones.all() if hasattr(dones, 'all') else all(dones)
            if done_flag:
                break

        result = {
            'fault_timeline': fault_timeline,
            'q_values_history': values_history,
            'rewards_history': rewards_history,
            'taylor_errors_history': taylor_history,
            'action_influences_history': action_influences_history,
            'directional_derivatives_history': directional_derivatives_history,
            'episode_length': timestep,
            'episode_reward': float(sum(sum(r) for r in rewards_history)),
            'attack_timesteps': attack_timesteps,
            'attacked_agent': attack_agent_i,
            'observed_agent': observe_agent_id,
        }
        if collect_frames:
            result['frames'] = frames
        return result

    def compute_attack_metrics(
        self,
        attack_results: Dict,
        normal_values: List[List[float]],
        normal_rewards: List[List[float]],
        ref_vals: List[List[float]],
        ref_std_devs: List[List[float]],
        observe_agent: int,
    ) -> Dict[str, float]:
        return self.attack_metrics.compute_attack_metrics(
            attack_results,
            normal_values,
            normal_rewards,
            ref_vals,
            ref_std_devs,
            observe_agent,
        )
