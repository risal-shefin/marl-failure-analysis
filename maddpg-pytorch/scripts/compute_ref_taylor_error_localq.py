import argparse
import torch
import time
import imageio
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from utils.make_env import make_env
from algorithms.maddpg import MADDPG
import os
from datetime import datetime
from utils.pettingzoo_wrapper import PettingZooWrapper
from utils.misc import gumbel_softmax
import pettingzoo.mpe as mpe
import pettingzoo.sisl as sisl
import pettingzoo.atari as atari
import matplotlib.pyplot as plt
from PIL import Image
from collections import deque
import supersuit
import csv
import math
from tqdm import tqdm

USE_CUDA = torch.cuda.is_available()
DEVICE = 'gpu' if USE_CUDA else 'cpu'
torch_device = torch.device("cuda" if USE_CUDA else "cpu")

def preprocess_env_atari(env):
    # as per openai baseline's MaxAndSKip wrapper, maxes over the last 2 frames
    # to deal with frame flickering
    env = supersuit.max_observation_v0(env, 2)
    # skip frames for faster processing and less control
    # to be compatible with gym, use frame_skip(env, (2,5))
    env = supersuit.frame_skip_v0(env, 4)
    # downscale observation for faster processing
    env = supersuit.resize_v1(env, 84, 84)
    # allow agent to see everything on the screen despite Atari's flickering screen problem
    env = supersuit.frame_stack_v1(env, 4)
    return env


def compute_taylor_delta_localq(maddpg, obs, actions, action_spaces, epsilon):
    # Convert discrete actions to one-hot encoding
    if maddpg.discrete_action:
        one_hot_actions = []
        for i, action in enumerate(actions):
            one_hot = np.zeros(action_spaces[i].n)
            one_hot[action] = 1.0
            one_hot_actions.append(one_hot)
        actions = one_hot_actions

    torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    actions = [Variable(torch.Tensor([actions[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    vf_in = torch.cat((*torch_obs, *actions), dim=1)

    delta_errors = []

    for i, agent_i in enumerate(maddpg.agents):
        local_vf_in = torch.cat((torch_obs[i], actions[i]), dim=1)
        local_critic_val = agent_i.local_critic(local_vf_in).mean()
        # grad_i = torch.autograd.grad(-local_critic_val, local_vf_in, create_graph=True, retain_graph=True)[0]
        grad_i = torch.autograd.grad(-local_critic_val, actions[i], create_graph=True, retain_graph=True)[0]

        eta_i = epsilon * grad_i.sign() / torch.max(grad_i.norm(p=2), torch.tensor(1e-6))
        
        # Second-order Taylor approximation: f(x + η) ≈ f(x) + ∇f(x)^T η + 0.5 η^T H η
        j_tilde = local_critic_val + torch.dot(grad_i.flatten(), eta_i.flatten())# + 0.5 * torch.dot(eta_i.flatten(), hvp.flatten())
        
        # p_local_vf_in = local_vf_in + eta_i
        p_action = actions[i] + eta_i
        p_local_vf_in = torch.cat((torch_obs[i], p_action), dim=1)
        j_perturbed = agent_i.local_critic(p_local_vf_in).mean()
        delta_error = abs(j_perturbed - j_tilde).item()
        delta_errors.append(delta_error)

    return delta_errors

def compute_taylor_delta_policy(maddpg, obs, actions, action_spaces, epsilon):
    # Convert discrete actions to one-hot encoding
    if maddpg.discrete_action:
        one_hot_actions = []
        for i, action in enumerate(actions):
            one_hot = np.zeros(action_spaces[i].n)
            one_hot[action] = 1.0
            one_hot_actions.append(one_hot)
        actions = one_hot_actions

    torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    actions = [Variable(torch.Tensor([actions[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    vf_in = torch.cat((*torch_obs, *actions), dim=1)

    delta_errors = []

    for i, agent_i in enumerate(maddpg.agents):
        if maddpg.discrete_action:
            action_logits_i = agent_i.policy(torch_obs[i])
            action_log_probs = torch.log_softmax(action_logits_i, dim=-1)
            max_action_idx = torch.argmax(action_log_probs, dim=-1)
            target_val = action_log_probs.gather(-1, max_action_idx.unsqueeze(-1)).squeeze()
        else:
            target_val = agent_i.policy(torch_obs[i]).sum()
        grad_i = torch.autograd.grad(target_val, torch_obs[i], create_graph=True, retain_graph=True)[0]
        # grad_i = torch.autograd.grad(critic_val, actions[i], create_graph=True, retain_graph=True)[0]

        eta_i = epsilon * grad_i.sign() / torch.max(grad_i.norm(p=2), torch.tensor(1e-6))
        
        # First-order Taylor approximation: f(x + η) ≈ f(x) + ∇f(x)^T η
        j_tilde = target_val + torch.dot(grad_i.flatten(), eta_i.flatten())

        p_torch_obs_i = torch_obs[i] + eta_i
        if maddpg.discrete_action:
            p_action_logits_i = agent_i.policy(p_torch_obs_i)
            p_action_log_probs = torch.log_softmax(p_action_logits_i, dim=-1)
            p_max_action_idx = torch.argmax(p_action_log_probs, dim=-1)
            j_perturbed = p_action_log_probs.gather(-1, p_max_action_idx.unsqueeze(-1)).squeeze()
        else:
            j_perturbed = agent_i.policy(p_torch_obs_i).sum()
            
        delta_error = abs(j_perturbed - j_tilde).item()
        delta_errors.append(delta_error)

    return delta_errors


def get_episode_data(env, maddpg, config, logdir, do_attack=False, atk_agent_id=-1, seed=None):
    # obs = env.reset()
    obs = env.reset(seed=seed) if seed else env.reset()
    # obs = env.reset(seed=12345) # better for speaker_listener_v3
    episode_reward = 0
    frames = []
    # initialize deque buffers for last batch_size observations
    result_deques_policy = [deque(maxlen=5) for _ in range(maddpg.nagents)]
    result_deques_localq = [deque(maxlen=5) for _ in range(maddpg.nagents)] if maddpg.local_q else None
    metric_vals = []
    cnt = 0

    while True:
        torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=False) for i in range(maddpg.nagents)]
        torch_agent_actions = maddpg.step(torch_obs, explore=False)
        agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions]
        if maddpg.discrete_action:
            actions = {agent_name: agent_actions[i].argmax() for i, agent_name in enumerate(env.possible_agents)}
        else:
            actions = {agent_name: agent_actions[i].squeeze() for i, agent_name in enumerate(env.possible_agents)}
        
        # Compute policy Taylor error
        results_policy = compute_taylor_delta_policy(maddpg, obs, list(actions.values()), env.action_space, 0.001)
        for i in range(maddpg.nagents):
            result_deques_policy[i].append(results_policy[i])
        
        # Compute local Q Taylor error if local_q is enabled
        policy_means = [np.mean(result_deques_policy[i]) for i in range(maddpg.nagents)]
        if maddpg.local_q:
            results_localq = compute_taylor_delta_localq(maddpg, obs, list(actions.values()), env.action_space, 0.001)
            for i in range(maddpg.nagents):
                result_deques_localq[i].append(results_localq[i])
            localq_means = [np.mean(result_deques_localq[i]) for i in range(maddpg.nagents)]
            # Combine policy and local Q metrics
            metric_vals.append([policy_means, localq_means])
        else:
            # Only policy metrics
            metric_vals.append([policy_means])

        next_obs, rewards, dones, infos = env.step(actions)
        episode_reward += np.sum([rewards[:,i] if env.agent_types[i] != 'adversary' else np.zeros_like(rewards[:,i]) for i in range(maddpg.nagents)])

        obs = next_obs
        cnt += 1
        if dones.all():
            break

    # print(f"Episode reward: {episode_reward}")

    return metric_vals


def save_matrix_to_files(matrix, agent_id, logdir, suffix="", has_local_q=False):
    """
    Save matrix data to a CSV file for all timesteps.
    
    Args:
        matrix: List of timesteps, where each timestep contains metrics data including:
                mean, variance, std_dev, q1, q3, median, and MAD for policy and optionally local Q
        agent_id: ID of the agent
        logdir: Directory to save the file
        suffix: Optional suffix for the filename
        has_local_q: Whether local Q metrics are included
    """
    filename = f"maddpg_taylor_error_atk_free_agent_{suffix}.csv"
    filepath = os.path.join(logdir, filename)
    
    # Create header row
    if has_local_q:
        header = ["agent", "timestep", 
                  "mean", "variance", "std_dev", "q1", "q3", 
                  "median", "mad", "diff_mean", "diff_std",
                  "localq_mean", "localq_variance", "localq_std_dev", "localq_q1", "localq_q3",
                  "localq_median", "localq_mad", "localq_diff_mean", "localq_diff_std"]
    else:
        header = ["agent", "timestep", "mean", "variance", "std_dev", "q1", "q3", "median", "mad", "diff_mean", "diff_std"]
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        for timestep, timestep_metrics in enumerate(matrix):
            row = [agent_id, timestep]
            if has_local_q:
                # Policy metrics first, then local Q metrics
                for value in timestep_metrics[0]:  # Policy metrics
                    row.append(value)
                for value in timestep_metrics[1]:  # Local Q metrics
                    row.append(value)
            else:
                # Only policy metrics
                for value in timestep_metrics:
                    row.append(value)
            writer.writerow(row)
    
    print(f"Saved {len(matrix)} timestep matrices to {filepath}")


def run(config):
    maddpg = MADDPG.init_from_save(config.model_path, test_mode=True)

    # create a log directory under runs/<env_id>/<timestamp> using os and getcwd
    cwd = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(cwd, 'runs', f"{config.env_id}_{'discrete' if maddpg.discrete_action else 'continuous'}", f"{timestamp}_metrics_taylor_error")
    os.makedirs(logdir, exist_ok=True)
    total_episodes = 50000

    try:
        env_func = getattr(mpe, config.env_id)
        if config.env_id == 'simple_spread_v3':
            env = env_func.parallel_env(continuous_actions= not maddpg.discrete_action, render_mode='rgb_array', N=maddpg.nagents)
        else:
            env = env_func.parallel_env(continuous_actions= not maddpg.discrete_action, render_mode='rgb_array')
    except:
        try:
            env_func = getattr(sisl, config.env_id)
            env = env_func.parallel_env(n_pursuers=5, render_mode='rgb_array') if config.env_id == 'waterworld_v4' else env_func.parallel_env(render_mode='rgb_array')
        except:
            env_func = getattr(atari, config.env_id)
            env = env_func.parallel_env(render_mode='rgb_array')
            env = preprocess_env_atari(env)

    env = PettingZooWrapper.wrap_env(env)
    env.reset()

    # maddpg.prep_rollouts(device=DEVICE)
    maddpg.prep_training(device=DEVICE)

    # Separate datasets for policy and local Q metrics
    if maddpg.local_q:
        result_dataset_policy = [{} for _ in range(maddpg.nagents)]
        result_dataset_localq = [{} for _ in range(maddpg.nagents)]
    else:
        result_dataset_policy = [{} for _ in range(maddpg.nagents)]
    
    for i in tqdm(range(total_episodes), desc="Processing episodes"):
        results = get_episode_data(env, maddpg, config, logdir)
        for timestep in range(len(results)):
            for agent_id in range(maddpg.nagents):
                if timestep not in result_dataset_policy[agent_id]:
                    result_dataset_policy[agent_id][timestep] = []
                if maddpg.local_q:
                    if timestep not in result_dataset_localq[agent_id]:
                        result_dataset_localq[agent_id][timestep] = []
                    # results[timestep] contains [policy_means, localq_means]
                    result_dataset_policy[agent_id][timestep].append(results[timestep][0][agent_id])
                    result_dataset_localq[agent_id][timestep].append(results[timestep][1][agent_id])
                else:
                    # results[timestep] contains [policy_means]
                    result_dataset_policy[agent_id][timestep].append(results[timestep][0][agent_id])

    for agent_id in range(maddpg.nagents):
        # Compute mean and variance for each agent across all episodes
        print(f"\n---- Agent {agent_id}:")
        metrics_mat = []
        sorted_timesteps = sorted(result_dataset_policy[agent_id].keys())
        
        for i, timestep in enumerate(sorted_timesteps):
            print(f"\n ---- Timestep {timestep}:")
            
            # Policy metrics
            policy_timestep_values = result_dataset_policy[agent_id][timestep]
            policy_mean_val = np.mean(policy_timestep_values)
            policy_var_val = np.var(policy_timestep_values)
            policy_std_dev_val = np.std(policy_timestep_values)
            policy_q1, policy_q3 = np.percentile(policy_timestep_values, [25, 75])
            policy_median_val = np.median(policy_timestep_values)
            policy_mad_val = np.median(np.abs(np.array(policy_timestep_values) - policy_median_val))
            
            # Compute policy difference metrics (e_i - e_{i-1})
            if i == 0:  # For timestep 0, differences are 0
                policy_diff_mean = 0.0
                policy_diff_std = 0.0
            else:
                prev_timestep = sorted_timesteps[i-1]
                curr_policy_vals = result_dataset_policy[agent_id][timestep]
                prev_policy_vals = result_dataset_policy[agent_id][prev_timestep]
                policy_differences = [curr - prev for curr, prev in zip(curr_policy_vals, prev_policy_vals)]
                policy_diff_mean = np.mean(policy_differences)
                policy_diff_std = np.std(policy_differences)
            
            policy_metrics = [policy_mean_val, policy_var_val, policy_std_dev_val, policy_q1, policy_q3, 
                            policy_median_val, policy_mad_val, policy_diff_mean, policy_diff_std]
            
            print(f"Policy - Agent {agent_id}: mean = {policy_mean_val:.4f}, variance = {policy_var_val:.4f}, std_dev = {policy_std_dev_val:.4f}, IQR = {policy_q3 - policy_q1:.4f}, median = {policy_median_val:.4f}, MAD = {policy_mad_val:.4f}, diff_mean = {policy_diff_mean:.4f}, diff_std = {policy_diff_std:.4f}")
            
            if maddpg.local_q:
                # Local Q metrics
                localq_timestep_values = result_dataset_localq[agent_id][timestep]
                localq_mean_val = np.mean(localq_timestep_values)
                localq_var_val = np.var(localq_timestep_values)
                localq_std_dev_val = np.std(localq_timestep_values)
                localq_q1, localq_q3 = np.percentile(localq_timestep_values, [25, 75])
                localq_median_val = np.median(localq_timestep_values)
                localq_mad_val = np.median(np.abs(np.array(localq_timestep_values) - localq_median_val))
                
                # Compute local Q difference metrics (e_i - e_{i-1})
                if i == 0:  # For timestep 0, differences are 0
                    localq_diff_mean = 0.0
                    localq_diff_std = 0.0
                else:
                    prev_timestep = sorted_timesteps[i-1]
                    curr_localq_vals = result_dataset_localq[agent_id][timestep]
                    prev_localq_vals = result_dataset_localq[agent_id][prev_timestep]
                    localq_differences = [curr - prev for curr, prev in zip(curr_localq_vals, prev_localq_vals)]
                    localq_diff_mean = np.mean(localq_differences)
                    localq_diff_std = np.std(localq_differences)
                
                localq_metrics = [localq_mean_val, localq_var_val, localq_std_dev_val, localq_q1, localq_q3,
                                localq_median_val, localq_mad_val, localq_diff_mean, localq_diff_std]
                
                print(f"Local Q - Agent {agent_id}: mean = {localq_mean_val:.4f}, variance = {localq_var_val:.4f}, std_dev = {localq_std_dev_val:.4f}, IQR = {localq_q3 - localq_q1:.4f}, median = {localq_median_val:.4f}, MAD = {localq_mad_val:.4f}, diff_mean = {localq_diff_mean:.4f}, diff_std = {localq_diff_std:.4f}")
                
                # Combine policy and local Q metrics
                combined_metrics = [policy_metrics, localq_metrics]
                metrics_mat.append(combined_metrics)
            else:
                # Only policy metrics
                metrics_mat.append(policy_metrics)

        save_matrix_to_files(metrics_mat, agent_id, logdir, suffix=f"{agent_id}", has_local_q=maddpg.local_q)

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_path",
                        help="model directory")

    config = parser.parse_args()

    run(config)
