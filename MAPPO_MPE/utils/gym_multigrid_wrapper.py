"""Wrapper around gym-multigrid environments for MAPPO evaluation."""
import sys
from pathlib import Path

import numpy as np
import gym


class GymMultiGridWrapper:
    def __init__(self, env, do_flat_obs: bool = False):
        self.env = env
        self.do_flat_obs = do_flat_obs
        self.nagents = len(env.agents)
        self.action_space = [env.action_space for _ in range(self.nagents)]

        if self.do_flat_obs:
            self.observation_space = [
                gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(int(np.prod(env.observation_space.shape)),),
                    dtype=env.observation_space.dtype,
                )
                for _ in range(self.nagents)
            ]
        else:
            self.observation_space = [env.observation_space for _ in range(self.nagents)]

        self.agent_types = ["agent" for _ in range(self.nagents)]
        self.possible_agents = [i for i in range(self.nagents)]

    @staticmethod
    def wrap_env(env, do_flat_obs: bool = False):
        return GymMultiGridWrapper(env, do_flat_obs=do_flat_obs)

    @staticmethod
    def make_and_wrap_env(env_id: str, do_flat_obs: bool = False):
        envs_root = Path(__file__).resolve().parents[2] / 'maddpg-pytorch' / 'envs' / 'gym-multigrid'
        if envs_root.exists() and str(envs_root) not in sys.path:
            sys.path.append(str(envs_root))

        from gym_multigrid.envs import CollectGame4HEnv10x10N2, SoccerGame4HEnv10x15N2  # pylint: disable=import-error

        if env_id == 'soccer':
            env = SoccerGame4HEnv10x15N2()
        elif env_id == 'collect':
            env = CollectGame4HEnv10x10N2()
        else:
            raise ValueError(f"Unknown MultiGrid environment ID: {env_id}")

        return GymMultiGridWrapper.wrap_env(env, do_flat_obs)

    def reset(self, seed=None):
        if seed is not None:
            self.env.seed(seed)
        obs = self.env.reset()
        if self.do_flat_obs:
            obs = [obs[i].flatten() for i in range(self.nagents)]
        return obs

    def step(self, actions):
        if isinstance(actions, dict):
            actions = list(actions.values())
        obs, rewards, done, info = self.env.step(actions)
        if self.do_flat_obs:
            obs = [obs[i].flatten() for i in range(self.nagents)]
        dones = [done for _ in range(self.nagents)]
        infos = [info for _ in range(self.nagents)]
        rewards = np.array([rewards])
        dones = np.array([dones])
        infos = np.array([infos])
        return obs, rewards, dones, infos

    def render(self, mode='rgb_array'):
        if hasattr(self.env, 'render'):
            return self.env.render(mode=mode)
        return None

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()

    def __getattr__(self, name):
        return getattr(self.env, name)
