from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, Sequence

import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from lightning import Fabric
from lightning.fabric.wrappers import _FabricModule
from torch import Tensor
import os
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


def perturb_obs_fgsm(agent: PPOPlayer, env, torch_obs: Dict[str, Tensor], fabric: Fabric, cfg: Dict[str, any], epsilon: float
) -> Dict[str, Tensor]:
    """ Compute perturbed obs using FGSM attack """
    
    torch_obs = {k: v for k, v in torch_obs.items()}    # copy
    for key in cfg.algo.cnn_keys.encoder:
        if key not in torch_obs:
            continue
        torch_obs[key] = torch_obs[key].detach().requires_grad_(True)

    saved_env_state = env.unwrapped.clone_state() # Save the Current State Before Rollout
    _, logprob, value = agent(torch_obs)
    action = agent.get_actions(torch_obs, greedy=True)
    if agent.actor.is_continuous:
        action = torch.cat(action, dim=-1)
    else:
        action = torch.cat([act.argmax(dim=-1) for act in action], dim=-1)
    next_obs, reward, done, truncated, _ = env.step(action)
    next_obs = prepare_obs(fabric, next_obs, cnn_keys=cfg.algo.cnn_keys.encoder)
    next_value = agent.get_values(next_obs)
    _, advantages = gae(torch.tensor([reward]), value.unsqueeze(0), torch.tensor([done]), next_value.unsqueeze(0),
            1, cfg.algo.gamma, cfg.algo.gae_lambda)
    env.unwrapped.restore_state(saved_env_state)  # Restore the Environment Back to the Saved State

    pg_loss = policy_loss(logprob, logprob, advantages, cfg.algo.clip_coef, reduction="none")

    perturbed_obs = {k: v for k, v in torch_obs.items()} # copy
    for key in cfg.algo.cnn_keys.encoder:
        if key not in torch_obs:
            continue

        # The gradient in terms of loss
        grad_J = torch.autograd.grad(pg_loss, torch_obs[key], create_graph=True)[0]
        # Compute η_i (adversarial perturbation direction)
        eta_i = epsilon * grad_J.sign() / torch.max(grad_J.norm(p=2), torch.tensor(1.0))

        # Perturbed state
        perturbed_obs[key] = torch_obs[key] + eta_i

    return perturbed_obs

def perturb_obs_random_noise(torch_obs: Dict[str, Tensor], cfg: Dict[str, any], epsilon: float
) -> Dict[str, Tensor]:
    torch_obs = {k: v for k, v in torch_obs.items()}    # copy
    for key in cfg.algo.cnn_keys.encoder:
        if key in torch_obs:
            noise = torch.randn_like(torch_obs[key]) * epsilon
            torch_obs[key] = torch_obs[key] + noise
    return torch_obs

def so_inrd(agent: PPOPlayer, env, torch_obs: Dict[str, Tensor], fabric: Fabric, cfg: Dict[str, any], epsilon: float
) -> Tensor:
    """ Compute gradient of J with respect to s """
    torch_obs = {k: v for k, v in torch_obs.items()}    # copy
    for key in cfg.algo.cnn_keys.encoder:
        if key not in torch_obs:
            continue
        torch_obs[key] = torch_obs[key].detach().requires_grad_(True)

    saved_env_state = env.unwrapped.clone_state() # Save the Current State Before Rollout
    _, logprob, value = agent(torch_obs)
    action = agent.get_actions(torch_obs, greedy=True)
    if agent.actor.is_continuous:
        action = torch.cat(action, dim=-1)
    else:
        action = torch.cat([act.argmax(dim=-1) for act in action], dim=-1)
    next_obs, reward, done, truncated, _ = env.step(action)
    next_obs = prepare_obs(fabric, next_obs, cnn_keys=cfg.algo.cnn_keys.encoder)
    next_value = agent.get_values(next_obs)
    _, advantages = gae(torch.tensor([reward]), value.unsqueeze(0), torch.tensor([done]), next_value.unsqueeze(0),
            1, cfg.algo.gamma, cfg.algo.gae_lambda)
    env.unwrapped.restore_state(saved_env_state)  # Restore the Environment Back to the Saved State

    pg_loss = policy_loss(logprob, logprob, advantages, cfg.algo.clip_coef, reduction="none")

    perturbed_obs = {k: v for k, v in torch_obs.items()}    # copy
    J_tilde_li = list()
    for key in cfg.algo.cnn_keys.encoder:
        if key not in torch_obs:
            continue

        # The gradient in terms of loss
        grad_J = torch.autograd.grad(pg_loss, torch_obs[key], create_graph=True)[0]
        # Compute η_i (adversarial perturbation direction)
        eta_i = epsilon * grad_J.sign() / torch.max(grad_J.norm(p=2), torch.tensor(1.0))
        # Compute J tilde
        J_tilde_li.append(pg_loss + torch.dot(grad_J.flatten(), eta_i.flatten()))
    
        # Perturbed state
        perturbed_obs[key] = torch_obs[key] + eta_i

    J_tilde = torch.tensor(J_tilde_li).mean()
    p_value = agent.get_values(perturbed_obs)
    _, p_advantages = gae(torch.tensor([reward]), p_value.unsqueeze(0), torch.tensor([done]), next_value.unsqueeze(0),
            1, cfg.algo.gamma, cfg.algo.gae_lambda)
    perturbed_policy_loss = policy_loss(logprob, logprob, p_advantages, cfg.algo.clip_coef, reduction="none")

    # Compute L
    L = perturbed_policy_loss - J_tilde
    return L


def get_episode_data(agent: PPOPlayer, fabric: Fabric, cfg: Dict[str, Any], log_dir: str, do_attack: bool):
    env = make_env(cfg, None, 0, log_dir, "test", vector_env_idx=0)()
    agent.eval()
    done = False
    cumulative_rew = 0
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)  # PyTorch CPU seed
    torch.cuda.manual_seed(cfg.seed)  # PyTorch GPU seed
    obs = env.reset(seed=cfg.seed)[0]

    episode_data = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'values': [], 'dones': [], 
                    'next_states': [], 'next_values': [], 'so_inrd_l': [], 'attack_flag': []}

    so_eps = 0.1
    step_counter = 0
    while not done:
        step_counter += 1
        torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder)
        
        is_attacked = False
        if do_attack and agent.get_values(torch_obs) > 0.7:
        # if do_attack and np.random.random() < 0.5:
            torch_obs = perturb_obs_fgsm(agent, env, torch_obs, fabric, cfg, so_eps)
            # torch_obs = perturb_obs_random_noise(torch_obs, cfg, so_eps)
            is_attacked = True

        _, logprob, value = agent(torch_obs)
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

        episode_data['states'].append(obs)
        episode_data['actions'].append(actions)
        episode_data['logprobs'].append(logprob)
        episode_data['rewards'].append(reward)
        episode_data['values'].append(value)
        episode_data['dones'].append(done)
        episode_data['next_states'].append(next_torch_obs)
        episode_data['next_values'].append(agent.get_values(next_torch_obs))
        episode_data['so_inrd_l'].append(so_inrd(agent, env, torch_obs, fabric, cfg, so_eps).item())
        episode_data['attack_flag'].append(is_attacked)

        if cfg.dry_run:
            done = True
    
    fabric.print(f"Episode Reward = {cumulative_rew}")
    episode_data['sum_reward'] = cumulative_rew
    env.close()
    return episode_data
    # return cumulative_rew

def exp_loss_run(agent: PPOPlayer, fabric: Fabric, cfg: Dict[str, Any], log_dir: str):
    episode_data = get_episode_data(agent, fabric, cfg, log_dir, False)
    episode_data_attacked = get_episode_data(agent, fabric, cfg, log_dir, True)
    plot(episode_data, episode_data_attacked, log_dir)

    # returns, advantages = gae(torch.tensor(episode_data['rewards']), torch.tensor(episode_data['values']), 
    #                           torch.tensor(episode_data['dones']), torch.tensor(episode_data['next_values'][-1]), 
    #                           cfg.algo.rollout_steps, cfg.algo.gamma, cfg.algo.gae_lambda)

    # Policy loss
    # We are evaluating same network. so same log prob for old and new log probs
    # pg_loss = policy_loss(torch.tensor(episode_data['logprobs']), torch.tensor(episode_data['logprobs']), advantages, cfg.algo.clip_coef, reduction="none")
    
    # # Value loss
    # # We are evaluating same network. so same critic value for old and new values
    # v_loss = value_loss(torch.tensor(episode_data['values']), torch.tensor(episode_data['values']), returns, cfg.algo.clip_coef, cfg.algo.clip_vloss, reduction="none")

    # fabric.print(f"Policy Loss = {pg_loss}")
    # fabric.print(f"Value Loss = {v_loss}")
    # fabric.print(f"L_SO_INRD = {episode_data['so_inrd_l']}")


def plot(episode_data, episode_data_attacked, log_dir: str):
    fig = plt.figure()
    plt.plot(np.arange(len(episode_data['so_inrd_l'])), episode_data['so_inrd_l'], 
             label="Unattacked", linestyle=":")
    plt.plot(np.arange(len(episode_data_attacked['so_inrd_l'])), episode_data_attacked['so_inrd_l'], 
             label="Attacked", linestyle="--")
    # Add a dashed vertical line at step_x
    attack_flags = episode_data_attacked['attack_flag']
    attacked_steps = np.arange(len(episode_data_attacked['so_inrd_l']))[attack_flags]
    attacked_values = np.zeros(len(episode_data_attacked['so_inrd_l']))[attack_flags]
    # plt.scatter(attacked_steps, attacked_values, color='r', marker='x', label="Attacked Step")
    # plt.axvline(x=index, color='r', linestyle='--', label=f'Attacked Step')
    plt.xlabel("Steps")
    plt.ylabel("SO INRD L value")
    plt.title("Env: Boxing, FGSM Attack When V(s) > 0.7")
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'so_inrd_comparison.png'), dpi=300, format='png',bbox_inches='tight')
    plt.close(fig)
