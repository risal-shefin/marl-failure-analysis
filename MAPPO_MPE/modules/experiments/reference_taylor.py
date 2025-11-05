"""Reference Taylor value management for MAPPO analysis."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from ..metrics import compute_taylor_error_policy


class ReferenceTaylorManager:
    """Manage reference Taylor approximation statistics for MAPPO."""

    def __init__(self, runner, env, config):
        self.runner = runner
        self.env = env
        self.config = config
        self.cache_dir = getattr(config, 'taylor_cache_dir', None)
        self._cache: Dict[int, Tuple[List[List[float]], List[List[float]]]] = {}

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

    def _cache_path(self, seed: int) -> str:
        if not self.cache_dir:
            return ''
        return os.path.join(self.cache_dir, f'ref_taylor_seed{seed}.json')

    def load_cache(self):
        if not self.cache_dir:
            return
        for filename in os.listdir(self.cache_dir):
            if filename.startswith('ref_taylor_seed') and filename.endswith('.json'):
                seed = int(filename.split('seed')[1].split('.json')[0])
                path = os.path.join(self.cache_dir, filename)
                with open(path, 'r') as f:
                    data = json.load(f)
                self._cache[seed] = (data['means'], data['stds'])

    def get_reference_values(self, seed: int) -> Tuple[List[List[float]], List[List[float]]]:
        if seed in self._cache:
            return self._cache[seed]

        path = self._cache_path(seed)
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
            self._cache[seed] = (data['means'], data['stds'])
            return self._cache[seed]

        means, stds = self._compute_reference(seed)
        if path:
            with open(path, 'w') as f:
                json.dump({'means': means, 'stds': stds}, f)
        self._cache[seed] = (means, stds)
        return means, stds

    def _compute_reference(self, seed: int) -> Tuple[List[List[float]], List[List[float]]]:
        episodes = getattr(self.config, 'ref_episodes', 10)
        epsilon = getattr(self.config, 'taylor_epsilon', 0.01)
        nagents = self.runner.args.N
        per_agent: List[Dict[int, List[float]]] = [defaultdict(list) for _ in range(nagents)]

        for episode in range(episodes):
            states, masks = self._reset_env(seed + episode)
            done = np.array([False for _ in range(nagents)])
            timestep = 0

            while not done.all():
                errors = compute_taylor_error_policy(self.runner, states, epsilon)
                for agent_id, value in enumerate(errors):
                    per_agent[agent_id][timestep].append(value)

                actions = []
                for agent_id in range(nagents):
                    mask = masks[agent_id] if masks is not None else None
                    action, _ = self.runner.agent_n.select_action(
                        states[agent_id], agent_id, evaluate=True, return_dist=True, action_mask=mask
                    )
                    actions.append(int(action))

                next_states, _, done, _, masks = self._step_env(actions)
                states = next_states
                timestep += 1

        means: List[List[float]] = []
        stds: List[List[float]] = []
        max_timestep = max((max(agent_data.keys()) for agent_data in per_agent if agent_data), default=0)

        for agent_id in range(nagents):
            agent_means = []
            agent_stds = []
            for t in range(max_timestep + 1):
                values = per_agent[agent_id].get(t, [0.0])
                agent_means.append(float(np.mean(values)))
                agent_stds.append(float(np.std(values)))
            means.append(agent_means)
            stds.append(agent_stds)

        return means, stds

    # ------------------------------------------------------------------
    # Environment helpers shared with the episode runner
    # ------------------------------------------------------------------
    def _reset_env(self, seed: int):
        try:
            result = self.env.reset(seed=seed)
        except TypeError:
            if hasattr(self.env, 'seed'):
                self.env.seed(seed)
            result = self.env.reset()
        return self._extract_states_and_masks(result)

    def _step_env(self, actions):
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
        return states, rewards, dones, info, masks

    @staticmethod
    def _extract_states_and_masks(result):
        if isinstance(result, tuple):
            states = result[0]
            masks = None
            for extra in result[1:]:
                if ReferenceTaylorManager._is_mask_like(extra, states):
                    masks = extra
                    break
            return states, masks
        return result, None

    @staticmethod
    def _is_mask_like(candidate, states):
        if candidate is None:
            return False
        try:
            return len(candidate) == len(states)
        except TypeError:
            return False
