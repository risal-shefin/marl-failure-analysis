from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Dict, Sequence
import os
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
import matplotlib.pyplot as plt



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

def get_episode_data(agent: PPOPlayer, fabric: Fabric, cfg: Dict[str, Any], log_dir: str, perturb_flag=False,random_state=100,epsilon=0.1):
    env = make_env(cfg, None, 0, log_dir, "test", vector_env_idx=0)()
    agent.eval()
    done = False
    cumulative_rew = 0
    obs = env.reset(seed=100)[0]
    episode_data = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'values': [], 'dones': [], 
                    'next_states': [], 'next_values': []}
    state_count=0
    while not done:
        state_count+=1
        if perturb_flag and state_count ==random_state:
            fabric.print(f"Perturbation intiated at step : {state_count}")
            torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder)
            # fabric.print(f"Before Perturbation: {torch_obs}")
            for key in cfg.algo.cnn_keys.encoder:
                if key in torch_obs:
                    noise = torch.randn_like(torch_obs[key]) * epsilon
                    print(f"Noise : {noise}")
                    torch_obs[key] = torch_obs[key] + noise
                    
            fabric.print(f"After perturbation State  : {torch_obs}")
            # exit()
        if state_count==random_state and perturb_flag==False:
            print(f"State:{state_count} Value\n:{torch_obs}")
        else:
            torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder)
        # perturbed state(torch_obs)
        # actions = agent.get_actions(perturbed, greedy=True)
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
def plot_losses(normal_pg_loss, normal_v_loss, attack_pg_loss, attack_v_loss,
                random_state: int = 10,
                save_figures: bool = False,
                save_dir: str = ".",
                policy_loss_filename: str = "policy_loss_comparison.png",
                value_loss_filename: str = "value_loss_comparison.png") -> None:
    import os
    import matplotlib.pyplot as plt

    if save_figures and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Create separate x-axes for normal and attack episodes.
    x_axis_normal = range(1, len(normal_pg_loss) + 1)
    x_axis_attack = range(1, len(attack_pg_loss) + 1)
    
    # --- Plot Policy Loss ---
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis_normal, normal_pg_loss, label="Normal Policy Loss")
    plt.plot(x_axis_attack, attack_pg_loss, label="Attack Policy Loss")
    plt.axvline(x=random_state, color='red', linestyle='--', label="Perturbation State")
    plt.xlabel("State Index")
    plt.ylabel("Policy Loss")
    plt.title("Policy Loss Comparison")
    plt.legend()
    if save_figures:
        policy_loss_path = os.path.join(save_dir, policy_loss_filename)
        plt.savefig(policy_loss_path)
    plt.show()
    
    # --- Plot Value Loss ---
    x_axis_normal_val = range(1, len(normal_v_loss) + 1)
    x_axis_attack_val = range(1, len(attack_v_loss) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis_normal_val, normal_v_loss, label="Normal Value Loss")
    plt.plot(x_axis_attack_val, attack_v_loss, label="Attack Value Loss")
    plt.axvline(x=random_state, color='red', linestyle='--', label="Perturbation State")
    plt.xlabel("State Index")
    plt.ylabel("Value Loss")
    plt.title("Value Loss Comparison")
    plt.legend()
    if save_figures:
        value_loss_path = os.path.join(save_dir, value_loss_filename)
        plt.savefig(value_loss_path)
    plt.show()

def exp_loss_run(agent: PPOPlayer, fabric: Fabric, cfg: Dict[str, Any], log_dir: str):
    random_state=100
    import random
    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    episode_data = get_episode_data(agent, fabric, cfg, log_dir,perturb_flag=False,random_state=random_state)
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
    fabric.print(f"Value ={episode_data['values']}")
    fabric.print(f"Action = {episode_data['actions']}")
    # print(f"######## ATTACK SCENARIO (FO) #######")
    episode_data = get_episode_data(agent, fabric, cfg, log_dir,perturb_flag=True,random_state=random_state,epsilon=1.0)
    returns, advantages = gae(torch.tensor(episode_data['rewards']), torch.tensor(episode_data['values']), 
                              torch.tensor(episode_data['dones']), torch.tensor(episode_data['next_values'][-1]), 
                              cfg.algo.rollout_steps, cfg.algo.gamma, cfg.algo.gae_lambda)

    # Policy loss
    # We are evaluating same network. so same log prob for old and new log probs
    pg_loss_attack = policy_loss(torch.tensor(episode_data['logprobs']), torch.tensor(episode_data['logprobs']), advantages, cfg.algo.clip_coef, reduction="none")
    
    # Value loss
    # We are evaluating same network. so same critic value for old and new values
    v_loss_attack = value_loss(torch.tensor(episode_data['values']), torch.tensor(episode_data['values']), returns, cfg.algo.clip_coef, cfg.algo.clip_vloss, reduction="none")

    fabric.print(f"Policy Loss = {pg_loss_attack}")
    fabric.print(f"Value Loss = {v_loss_attack}")
    fabric.print(f"Action = {episode_data['actions']}")
    fabric.print(f"Value ={episode_data['values']}")
    sample=6
    plot_losses(pg_loss,v_loss,
                pg_loss_attack,
                v_loss_attack,
                random_state=random_state,
                save_figures=True,
                save_dir="/deac/csc/vanbastelaerGrp/guptd23/RL_Project/AdversaryLoss/sheeprl/logs/runs/ppo/BoxingNoFrameskip-v4/Figure",
                policy_loss_filename=f"TEST_POLICY_{sample}.png",
                value_loss_filename=f'TEST_VALUE_{sample}.png'
                )
    
