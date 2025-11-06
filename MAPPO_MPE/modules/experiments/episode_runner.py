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
    compute_pairwise_action_influences,
    collect_agent_q_values,
    collect_agent_q_value,
    compute_pairwise_action_directional_second_derivatives,
    compute_taylor_delta_policy,
)
ATTACK_WINDOW_LENGTH = 3


class EpisodeRunner:
    """Run normal and attacked episodes for MAPPO analysis."""

    def __init__(self, runner, env, epsilon: float = 0.01):
        self.runner = runner
        self.env = env
        self.epsilon = epsilon
        self.attack_metrics = AttackMetricsComputer()
        self._last_action_masks: Optional[Iterable] = None
        self.directional_derivatives_history = []

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
        """
        Run a normal episode without attacks to collect action influences.
        
        Args:
            seed: Random seed for the episode
            collect_frames: Whether to collect RGB frames for visualization
            
        Returns:
            Dictionary containing episode data including action influences
        """
        # Reset directional derivatives history for this episode
        self.directional_derivatives_history = []
        
        self._set_seed(seed)
        states, _ = self._reset_env(seed)
        action_influences_history: List[List[List[float]]] = []
        q_values_history: List[List[float]] = []
        rewards_history: List[List[float]] = []
        taylor_history: List[List[float]] = []
        frames = [] if collect_frames else None
        timestep = 0
        episode_reward = 0

        if collect_frames:
            try:
                frame = self.env.render(mode='rgb_array')
            except TypeError:
                frame = self.env.render()
            frames.append(frame)

        while True:
            # Get actions for all agents
            actions = []
            for agent_id in range(self.runner.args.N):
                action, _ = self._select_action(states[agent_id], agent_id)
                actions.append(action)
            
            # Compute global state (concatenation of observations)
            global_state = np.concatenate(states)
            
            # Collect Q-values using centralized Q if available
            q_values = collect_agent_q_values(
                self.runner.agent_n,
                global_state,
                states,
                actions,
                None  # action_spaces not needed for MAPPO
            )
            q_values_history.append(q_values)
            
            # Compute pairwise action influences
            action_influences_matrix = compute_pairwise_action_influences(
                self.runner.agent_n,
                global_state,
                states,
                actions,
                None  # action_spaces not needed for MAPPO
            )
            action_influences_history.append(action_influences_matrix)
            
            # Compute directional second derivatives
            directional_derivatives_matrix = compute_pairwise_action_directional_second_derivatives(
                self.runner.agent_n,
                global_state,
                states,
                actions,
                None  # action_spaces not needed for MAPPO
            )
            self.directional_derivatives_history.append(directional_derivatives_matrix)
            
            # Compute Taylor delta policy
            taylor_errors = compute_taylor_delta_policy(
                self.runner.agent_n,
                states,
                self.epsilon
            )
            taylor_history.append(taylor_errors)

            # Take environment step
            next_states, rewards, dones, _, _ = self._step_env(actions)
            rewards_array = np.array(rewards)
            rewards_history.append(list(rewards_array.squeeze()))
            episode_reward += np.sum(rewards_array)

            if collect_frames:
                try:
                    frame = self.env.render(mode='rgb_array')
                except TypeError:
                    frame = self.env.render()
                frames.append(frame)

            states = next_states
            timestep += 1

            done_flag = dones.all() if hasattr(dones, 'all') else all(dones)
            if done_flag:
                break

        print(f"Episode reward (normal): {episode_reward}")
        result = {
            'action_influences_history': action_influences_history,
            'directional_derivatives_history': self.directional_derivatives_history,
            'q_values_history': q_values_history,
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
        """
        Run an episode with attack on specific agent at specific timesteps.
        
        Args:
            seed: Random seed for the episode
            attack_agent_i: Agent to attack (the influencing agent)
            attack_timesteps: List of timesteps to attack
            ref_vals: Reference Taylor error values
            ref_std_devs: Reference standard deviations
            observe_agent: Agent to observe impact on (the influenced agent). 
                         If None, observe attack_agent_i
            collect_frames: Whether to collect RGB frames for visualization
            
        Returns:
            Dictionary containing attack episode results
        """
        
        self._set_seed(seed)
        states, _ = self._reset_env(seed)
        
        # Determine which agent to observe impact on
        observe_agent_id = observe_agent if observe_agent is not None else attack_agent_i
        
        episode_reward = 0
        frames = [] if collect_frames else None
        result_deques = [deque(maxlen=5) for _ in range(self.runner.args.N)]
        fault_timeline = []
        fault_first_detected: Dict[int, int] = {}
        q_values_history: List[List[float]] = []
        rewards_history: List[List[float]] = []
        taylor_errors_history: List[List[float]] = []
        directional_derivatives_history = []
        action_influences_history = []
        attack_timesteps = sorted(attack_timesteps)
        timestep = 0

        if collect_frames:
            try:
                frame = self.env.render(mode='rgb_array')
            except TypeError:
                frame = self.env.render()
            frames.append(frame)

        while True:
            # Get actions for all agents
            actions = []
            action_dists = []
            for agent_id in range(self.runner.args.N):
                action, dist = self._select_action(states[agent_id], agent_id)
                actions.append(action)
                action_dists.append(dist)
            
            # Compute global state (concatenation of observations)
            global_state = np.concatenate(states)
            
            # Collect Q-values using centralized Q if available
            q_values = collect_agent_q_values(
                self.runner.agent_n,
                global_state,
                states,
                actions,
                None  # action_spaces not needed for MAPPO
            )
            q_values_history.append(q_values)

            # Compute pairwise action influences
            action_influences_matrix = compute_pairwise_action_influences(
                self.runner.agent_n,
                global_state,
                states,
                actions,
                None  # action_spaces not needed for MAPPO
            )
            action_influences_history.append(action_influences_matrix)

            # Compute directional second derivatives
            directional_derivatives_matrix = compute_pairwise_action_directional_second_derivatives(
                self.runner.agent_n,
                global_state,
                states,
                actions,
                None  # action_spaces not needed for MAPPO
            )
            directional_derivatives_history.append(directional_derivatives_matrix)
            
            # Compute Taylor delta policy for fault detection
            taylor_results = compute_taylor_delta_policy(
                self.runner.agent_n,
                states,
                self.epsilon
            )
            taylor_errors_history.append(taylor_results)
            
            # Update fault detection deques
            for agent_id in range(self.runner.args.N):
                result_deques[agent_id].append(taylor_results[agent_id])
            
            # Check for fault detection by threshold
            for agent_id in range(self.runner.args.N):
                if timestep < len(ref_vals[agent_id]) and timestep < len(ref_std_devs[agent_id]):
                    detection_value = np.mean(result_deques[agent_id])
                    threshold_exceeded = abs(detection_value - ref_vals[agent_id][timestep]) > K_SIGMA * ref_std_devs[agent_id][timestep] \
                                         and not np.isclose(detection_value, ref_vals[agent_id][timestep], rtol=1e-5, atol=1e-5)
                    
                    if threshold_exceeded:
                        if agent_id not in fault_first_detected:
                            fault_first_detected[agent_id] = timestep
                        fault_timeline.append({
                            'agent': agent_id,
                            't': timestep,
                            'contribs': {},
                            'taylor_deviation': abs(detection_value - ref_vals[agent_id][timestep])
                        })
            
            # Apply attack if current timestep is in any attack window
            attack_active = False
            for attack_t in attack_timesteps:
                if attack_t <= timestep < attack_t + ATTACK_WINDOW_LENGTH:
                    attack_active = True
                    break
            
            if attack_active:
                # Inject fault - find worst action for the attacked agent wrt the observed agent
                # that minimizes the observed agent's Q-value
                worst_action = 0
                worst_q_value = float('inf')
                
                for action in range(self.runner.args.action_dim):
                    temp_actions = actions.copy()
                    temp_actions[attack_agent_i] = action
                    observe_agent_q_val = collect_agent_q_value(
                        self.runner.agent_n,
                        observe_agent_id,
                        global_state,
                        states,
                        temp_actions,
                        None  # action_spaces not needed for MAPPO
                    )
                    if observe_agent_q_val < worst_q_value:
                        worst_q_value = observe_agent_q_val
                        worst_action = action
                
                actions[attack_agent_i] = worst_action

            # Take environment step
            next_states, rewards, dones, _, _ = self._step_env(actions)
            rewards_array = np.array(rewards)
            rewards_history.append(list(rewards_array.squeeze()))
            episode_reward += np.sum(rewards_array)

            if collect_frames:
                try:
                    frame = self.env.render(mode='rgb_array')
                except TypeError:
                    frame = self.env.render()
                frames.append(frame)

            states = next_states
            timestep += 1

            done_flag = dones.all() if hasattr(dones, 'all') else all(dones)
            if done_flag:
                break

        result = {
            'fault_timeline': fault_timeline,
            'q_values_history': q_values_history,
            'rewards_history': rewards_history,
            'taylor_errors_history': taylor_errors_history,
            'directional_derivatives_history': directional_derivatives_history,
            'action_influences_history': action_influences_history,
            'episode_length': timestep,
            'episode_reward': episode_reward,
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
