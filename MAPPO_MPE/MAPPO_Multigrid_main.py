"""MAPPO runner for MultiGrid environments."""
import os
from datetime import datetime

import numpy as np
import torch
from gym.spaces import Box, Discrete

from mappo import MAPPO
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer


class Runner_MAPPO_Multigrid:
    def __init__(self, args, env_id: str, number: int, seed: int, flatten_obs: bool = False):
        from utils.gym_multigrid_wrapper import GymMultiGridWrapper

        self.args = args
        self.env_id = env_id
        self.number = number
        self.seed = seed

        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.env = GymMultiGridWrapper.make_and_wrap_env(env_id, do_flat_obs=flatten_obs)
        self.args.N = self.env.nagents

        self.args.obs_dim_n = []
        for obs_space in self.env.observation_space:
            if isinstance(obs_space, Box):
                self.args.obs_dim_n.append(obs_space.shape[0])
            else:
                raise ValueError(f"Unsupported observation space: {obs_space}")

        self.args.action_dim_n = []
        for act_space in self.env.action_space:
            if isinstance(act_space, Discrete):
                self.args.action_dim_n.append(act_space.n)
            else:
                raise ValueError(f"Unsupported action space: {act_space}")

        self.args.obs_dim = self.args.obs_dim_n[0]
        self.args.action_dim = self.args.action_dim_n[0]
        self.args.state_dim = np.sum(self.args.obs_dim_n)

        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.output_dir = os.path.join(args.output_dir, 'train', f'multigrid_{env_id}', timestamp)
        self.data_dir = os.path.join(self.output_dir, 'data')
        self.model_dir = os.path.join(self.output_dir, 'models')
        self.tensorboard_dir = os.path.join(self.output_dir, 'tensorboard')

        for directory in [self.output_dir, self.data_dir, self.model_dir, self.tensorboard_dir]:
            os.makedirs(directory, exist_ok=True)

        self.agent_n = MAPPO(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        if self.args.use_reward_norm:
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()
