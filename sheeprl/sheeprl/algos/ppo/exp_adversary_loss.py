from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, Sequence

import gymnasium as gym
import numpy as np
import torch
from lightning import Fabric
from lightning.fabric.wrappers import _FabricModule
from torch import Tensor
from sheeprl.utils.utils import gae
from sheeprl.algos.ppo.loss import entropy_loss, policy_loss, value_loss

from sheeprl.algos.ppo.agent import PPOPlayer, build_agent
from sheeprl.utils.env import make_env
from sheeprl.utils.imports import _IS_MLFLOW_AVAILABLE
from sheeprl.utils.utils import unwrap_fabric

if TYPE_CHECKING:
    from mlflow.models.model import ModelInfo

AGGREGATOR_KEYS = {"Rewards/rew_avg", "Game/ep_len_avg", "Loss/value_loss", "Loss/policy_loss", "Loss/entropy_loss"}
MODELS_TO_REGISTER = {"agent"}


def prepare_obs(
    fabric: Fabric, obs: Dict[str, np.ndarray], *, cnn_keys: Sequence[str] = [], num_envs: int = 1, **kwargs
) -> Dict[str, Tensor]:
    torch_obs = {}
    for k in obs.keys():
        torch_obs[k] = torch.from_numpy(obs[k].copy()).to(fabric.device).float()
        if k in cnn_keys:
            torch_obs[k] = torch_obs[k].reshape(num_envs, -1, *torch_obs[k].shape[-2:])
        else:
            torch_obs[k] = torch_obs[k].reshape(num_envs, -1)
    return normalize_obs(torch_obs, cnn_keys, obs.keys())

def normalize_obs(
    obs: Dict[str, np.ndarray | Tensor], cnn_keys: Sequence[str], obs_keys: Sequence[str]
) -> Dict[str, np.ndarray | Tensor]:
    return {k: obs[k] / 255 - 0.5 if k in cnn_keys else obs[k] for k in obs_keys}

def perturb_obs(obs: Dict[str, np.ndarray | Tensor]) -> Dict[str, np.ndarray | Tensor]:
    return obs

def get_episode_data(agent: PPOPlayer, fabric: Fabric, cfg: Dict[str, Any], log_dir: str):
    env = make_env(cfg, None, 0, log_dir, "test", vector_env_idx=0)()
    agent.eval()
    done = False
    cumulative_rew = 0
    obs = env.reset(seed=cfg.seed)[0]
    episode_data = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'values': [], 'dones': [], 
                    'next_states': [], 'next_values': []}

    while not done:
        torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder)

        # Act greedly through the environment
        actions = agent.get_actions(torch_obs, greedy=True)
        if agent.actor.is_continuous:
            actions = torch.cat(actions, dim=-1)
        else:
            actions = torch.cat([act.argmax(dim=-1) for act in actions], dim=-1)

        # Single environment step
        obs, reward, done, truncated, _ = env.step(actions.cpu().numpy().reshape(env.action_space.shape))
        done = done or truncated
        cumulative_rew += reward
        next_torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder)
        _, logprob, value = agent(torch_obs)

        episode_data['states'].append(obs)
        episode_data['actions'].append(actions)
        episode_data['logprobs'].append(logprob)
        episode_data['rewards'].append(reward)
        episode_data['values'].append(value)
        episode_data['dones'].append(done)
        episode_data['next_states'].append(next_torch_obs)
        episode_data['next_values'].append(agent.get_values(next_torch_obs))

        if cfg.dry_run:
            done = True
    
    fabric.print(f"Episode Reward = {cumulative_rew}")
    env.close()
    return episode_data

def exp_loss_run(agent: PPOPlayer, fabric: Fabric, cfg: Dict[str, Any], log_dir: str):
    episode_data = get_episode_data(agent, fabric, cfg, log_dir)
    returns, advantages = gae(torch.tensor(episode_data['rewards']), torch.tensor(episode_data['values']), 
                              torch.tensor(episode_data['dones']), torch.tensor(episode_data['next_values'][-1]), 
                              cfg.algo.rollout_steps, cfg.algo.gamma, cfg.algo.gae_lambda)

    # Policy loss
    # We are evaluating same network. so same log prob for old and new log probs
    pg_loss = policy_loss(torch.tensor(episode_data['logprobs']), torch.tensor(episode_data['logprobs']), advantages, cfg.algo.clip_coef, reduction="none")
    
    # Value loss
    # We are evaluating same network. so same critic value for old and new values
    v_loss = value_loss(torch.tensor(episode_data['values']), torch.tensor(episode_data['values']), returns, cfg.algo.clip_coef, cfg.algo.clip_vloss, reduction="none")

    fabric.print(f"Policy Loss = {pg_loss}")
    fabric.print(f"Value Loss = {v_loss}")
