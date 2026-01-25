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
from matplotlib.patches import Patch
from itertools import combinations, chain
import copy

USE_CUDA = torch.cuda.is_available()
DEVICE = 'gpu' if USE_CUDA else 'cpu'
torch_device = torch.device("cuda" if USE_CUDA else "cpu")
K_SIGMA = 0.9
ATTACKED_AGENT = None
# SEED = 3276
# SEED = 13123
SEED = 42


def compute_episode_shapley_values(config, maddpg, action_history, seed, attacked_agent_id=None, attacked_steps=None):
    """
    Compute Shapley values for the entire episode at once.
    
    This is more efficient than computing at each timestep since we can reuse
    environment runs and compute cumulative rewards.
    
    Args:
        config: Configuration object containing env_id and other settings
        maddpg: MADDPG instance
        action_history: List of action lists for each timestep (used for obs reconstruction)
        seed: Seed to use for environment creation
        attacked_agent_id: ID of the attacked agent (None if no attack)
        attacked_steps: List of timesteps when attack occurs (None if no attack)
        
    Returns:
        list of numpy arrays containing Shapley values for each timestep
    """
    N = maddpg.nagents
    T = len(action_history)
    
    if T == 0:
        print("Warning: No action history available for Shapley computation")
        return []
    
    def create_fresh_env():
        """Create a fresh environment with the same configuration as the main environment."""
        try:
            env_func_ref = getattr(mpe, config.env_id)
            if config.env_id == "simple_spread_v3":
                test_env = env_func_ref.parallel_env(continuous_actions=not maddpg.discrete_action, render_mode='rgb_array', N=maddpg.nagents)
            else:
                test_env = env_func_ref.parallel_env(continuous_actions=not maddpg.discrete_action, render_mode='rgb_array')
        except:
            try:
                env_func_ref = getattr(sisl, config.env_id)
                test_env = env_func_ref.parallel_env(n_pursuers=5, render_mode='rgb_array') if config.env_id == 'waterworld_v4' else env_func_ref.parallel_env(render_mode='rgb_array')
            except:
                env_func_ref = getattr(atari, config.env_id)
                test_env = env_func_ref.parallel_env(render_mode='rgb_array')
                test_env = preprocess_env_atari(test_env)
        
        test_env = PettingZooWrapper.wrap_env(test_env)
        return test_env

    def eval_coalition_cumulative_rewards(coalition_mask):
        """Run full episode with coalition actions and return cumulative rewards per timestep."""
        test_env = create_fresh_env()
        obs = test_env.reset(seed=seed)
        
        step_rewards = []
        total_reward = 0.0
        
        for step in range(T):
            # Get observations and compute actions using target policy
            torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=False) 
                        for i in range(maddpg.nagents)]
            torch_agent_actions = maddpg.step(torch_obs, explore=False)
            agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions]
            
            # Create actions dict based on coalition mask
            coalition_actions = {}
            for i, agent_name in enumerate(test_env.possible_agents):
                if coalition_mask[i]:
                    # Coalition agent uses target policy action
                    action = agent_actions[i].argmax()
                    
                    # Apply worst action attack if this is the attacked agent and step is in attacked_steps
                    if i == attacked_agent_id and step in attacked_steps:
                        action = agent_actions[i].argmin()
                    
                    coalition_actions[agent_name] = action
                else:
                    # Non-coalition agent uses action 0
                    coalition_actions[agent_name] = 0
            
            obs, rewards, dones, _ = test_env.step(coalition_actions)
            step_reward = np.array(rewards).sum()
            total_reward += step_reward
            step_rewards.append(step_reward)
            
            if dones.all():
                break
        
        test_env.close()
        return step_rewards

    print("Computing Shapley values for entire episode...")
    
    # Store Shapley values for each timestep
    shapley_history = []
    
    # For efficiency, we compute Shapley values based on cumulative rewards
    # Then convert to per-timestep marginal contributions
    
    # Compute exact Shapley values for each agent
    agent_cumulative_rewards = {}  # agent_i -> coalition_mask -> cumulative_rewards_list
    
    for agent_i in range(N):
        print(f"Computing Shapley for agent {agent_i}...")
        agent_cumulative_rewards[agent_i] = {}
        
        other_agents = [j for j in range(N) if j != agent_i]
        
        # Enumerate all possible coalitions of other agents
        for r in range(len(other_agents) + 1):
            for coalition in combinations(other_agents, r):
                # Coalition without agent i
                coalition_mask_without = [False] * N
                for j in coalition:
                    coalition_mask_without[j] = True
                
                coalition_key_without = tuple(coalition_mask_without)
                if coalition_key_without not in agent_cumulative_rewards[agent_i]:
                    agent_cumulative_rewards[agent_i][coalition_key_without] = eval_coalition_cumulative_rewards(coalition_mask_without)
                
                # Coalition with agent i
                coalition_mask_with = coalition_mask_without.copy()
                coalition_mask_with[agent_i] = True
                
                coalition_key_with = tuple(coalition_mask_with)
                if coalition_key_with not in agent_cumulative_rewards[agent_i]:
                    agent_cumulative_rewards[agent_i][coalition_key_with] = eval_coalition_cumulative_rewards(coalition_mask_with)
    
    # Now compute per-timestep Shapley values
    for t in range(T):
        shapley_step = np.zeros(N, dtype=float)
        
        for agent_i in range(N):
            marginal_contribs = []
            other_agents = [j for j in range(N) if j != agent_i]
            
            for r in range(len(other_agents) + 1):
                for coalition in combinations(other_agents, r):
                    coalition_mask_without = [False] * N
                    for j in coalition:
                        coalition_mask_without[j] = True
                    
                    coalition_mask_with = coalition_mask_without.copy()
                    coalition_mask_with[agent_i] = True
                    
                    coalition_key_without = tuple(coalition_mask_without)
                    coalition_key_with = tuple(coalition_mask_with)
                    
                    # Get cumulative rewards up to timestep t
                    v_without = agent_cumulative_rewards[agent_i][coalition_key_without][t]
                    v_with = agent_cumulative_rewards[agent_i][coalition_key_with][t]

                    marginal_contribs.append(v_with - v_without)
            
            shapley_step[agent_i] = float(np.mean(marginal_contribs))
        
        shapley_history.append(shapley_step)
        if (t + 1) % 10 == 0 or t + 1 == T:
            print(f"Completed timestep {t + 1}/{T}")
    
    return shapley_history


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


def fgsm_attack(maddpg, obs, actions, attacked_agent_id, epsilon):
    # Convert to tensors with gradient tracking
    torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    torch_actions = [Variable(torch.Tensor([actions[i]]).to(torch_device), requires_grad=False) for i in range(maddpg.nagents)]
    # Concatenate for critic input
    vf_in = torch.cat((*torch_obs, *torch_actions), dim=1)
    # Loss to maximize (degrade agent performance)
    loss = -(maddpg.agents[attacked_agent_id].critic(vf_in)).mean()  # Negative to maximize via gradient ascent
    # Compute gradient
    grad = torch.autograd.grad(loss, torch_obs[attacked_agent_id], retain_graph=True)[0]
    # FGSM perturbation: move in direction of gradient sign
    perturbation = epsilon * grad.sign()
    # Apply perturbation element-wise
    obs_perturbed = obs[attacked_agent_id] + perturbation.squeeze().cpu().numpy()
    return obs_perturbed


def compute_taylor_delta_policy(maddpg, obs, epsilon):
    torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]

    delta_errors = []

    for i, agent_i in enumerate(maddpg.agents):
        action_logits_i = agent_i.policy(torch_obs[i])
        action_log_probs = torch.log_softmax(action_logits_i, dim=-1)
        max_action_idx = torch.argmax(action_log_probs, dim=-1)
        critic_val = action_log_probs.gather(-1, max_action_idx.unsqueeze(-1)).squeeze()
        grad_i = torch.autograd.grad(critic_val, torch_obs[i], create_graph=True, retain_graph=True)[0]
        # grad_i = torch.autograd.grad(critic_val, actions[i], create_graph=True, retain_graph=True)[0]

        eta_i = epsilon * grad_i.sign() / torch.max(grad_i.norm(p=2), torch.tensor(1e-6))
        
        # Second-order Taylor approximation: f(x + η) ≈ f(x) + ∇f(x)^T η + 0.5 η^T H η
        j_tilde = critic_val + torch.dot(grad_i.flatten(), eta_i.flatten())# + 0.5 * torch.dot(eta_i.flatten(), hvp.flatten())
        p_torch_obs_i = torch_obs[i] + eta_i
        p_action_logits_i = agent_i.policy(p_torch_obs_i)
        p_action_log_probs = torch.log_softmax(p_action_logits_i, dim=-1)
        p_max_action_idx = torch.argmax(p_action_log_probs, dim=-1)
        j_perturbed = p_action_log_probs.gather(-1, p_max_action_idx.unsqueeze(-1)).squeeze()
        delta_error = abs(j_perturbed - j_tilde).item()
        delta_errors.append(delta_error)

    return delta_errors

def compute_frob_norms(maddpg, obs, actions, action_spaces, vulnerable_agent_id):
    # if not maddpg.discrete_action:
    #     raise NotImplementedError("This function is only implemented for discrete action spaces.")
    
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
    
    results = []

    for i, agent_i in enumerate(maddpg.agents):
        critic_val = agent_i.critic(vf_in).mean()
        grad_i = torch.autograd.grad(critic_val, torch_obs[i], create_graph=True, retain_graph=True)[0]

        # Compute Hessian matrix
        hessian_matrix = []
        for k in range(grad_i.shape[1]):
            # Compute ∂²Q/∂obs_i[k]∂obs_j
            second_grad = torch.autograd.grad(
                grad_i[0, k], 
                torch_obs[vulnerable_agent_id], 
                retain_graph=True, 
                allow_unused=True
            )[0]
            
            hessian_matrix.append(second_grad.flatten())

        H = torch.stack(hessian_matrix)
        hessian_frob_norm = torch.norm(H, p='fro')
        results.append(hessian_frob_norm.item())

    return results


def compute_pairwise_frob_norms(maddpg, obs, actions, action_spaces):
    """
    Compute Frobenius norms of cross-agent Hessian blocks for all (i, j).
    Returns an N x N list where entry [i][j] approximates || \partial^2 v_i / (\partial obs_i \partial obs_j) ||_F.
    """
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

    N = maddpg.nagents
    results = [[0.0 for _ in range(N)] for _ in range(N)]

    for i in range(N):
        critic_val = maddpg.agents[i].critic(vf_in).mean()
        grad_i = torch.autograd.grad(critic_val, torch_obs[i], create_graph=True, retain_graph=True)[0]

        for j in range(N):
            hessian_matrix = []
            for k in range(grad_i.shape[1]):
                second_grad = torch.autograd.grad(
                    grad_i[0, k],
                    torch_obs[j],
                    retain_graph=True,
                    allow_unused=True
                )[0]
                if second_grad is None:
                    second_grad = torch.zeros_like(torch_obs[j])
                hessian_matrix.append(second_grad.flatten())

            H = torch.stack(hessian_matrix) if len(hessian_matrix) > 0 else torch.zeros(1, 1)
            results[i][j] = H.norm(p='fro').item()

    return results


# second order directional derivative
def compute_2nd_ord_dir_derivatives(maddpg, obs, actions, action_spaces, vulnerable_agent_id):
    # if not maddpg.discrete_action:
    #     raise NotImplementedError("This function is only implemented for discrete action spaces.")
    
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
    
    results = []

    for i, agent_i in enumerate(maddpg.agents):
        critic_val = agent_i.critic(vf_in).mean()
        grad_i = torch.autograd.grad(critic_val, torch_obs[i], create_graph=True, retain_graph=True)[0]
        v = grad_i / torch.max(grad_i.norm(p=2), torch.tensor(1e-6))

        # Compute Hessian-vector product (HVP) of grad_i and v with respect to torch_obs[j]
        hvp = torch.autograd.grad(
            outputs=grad_i,
            inputs=torch_obs[vulnerable_agent_id],
            grad_outputs=v,
            retain_graph=True,
            allow_unused=True
        )[0]

        # Compute u^T * H * v (quadratic form)
        grad_j = torch.autograd.grad(-critic_val, torch_obs[vulnerable_agent_id], create_graph=True, retain_graph=True)[0]
        u = grad_j / torch.max(grad_j.norm(p=2), torch.tensor(1e-6))
        curvature_val = torch.dot(u.flatten(), hvp.flatten())
        results.append(curvature_val.item())

    return results


def get_episode_data(env, test_env, maddpg, config, logdir, ref_vals, ref_std_devs, detection_method='mean_std', do_attack=False, atk_agent_id=-1, seed=None):
    # obs = env.reset()
    obs = env.reset(seed=seed) if seed else env.reset()
    # obs = env.reset(seed=12345) # better for speaker_listener_v3
    episode_reward = 0
    episode_rewards = [0 for _ in range(maddpg.nagents)]
    frames = []
    # initialize deque buffers for last batch_size observations
    result_deques = [deque(maxlen=5) for _ in range(maddpg.nagents)]
    frob_norms_deques = [deque(maxlen=1) for _ in range(maddpg.nagents)]
    sec_dir_derivatives_deques = [deque(maxlen=1) for _ in range(maddpg.nagents)]
    metric_vals = []
    cnt = 0
    vulnerable_agent_id = None
    attacked_steps = []
    frob_norms_list = []
    sec_dir_derivatives = []
    do_start_attack = False
    attack_step_remaining = 15

    # Fault detection tracking
    fault_first_detected = {}  # agent_id -> timestep first detected
    fault_timeline = []  # list of dicts: {agent, t, contribs: {f: c}}
    frob_norms_matrix_history = []  # list over timesteps of N x N frob norm matrices

    # Store observation history for Shapley value computation
    obs_history = []
    action_history = []
    # Store per-timestep Shapley values computed via env stepping (deepcopied env)
    shapley_env_history = []

    prev_errors = [0 for i in range(maddpg.nagents)]

    while True:
        # Store current observation for Shapley computation
        obs_history.append([obs[i].copy() for i in range(maddpg.nagents)])
        
        # add Gaussian noise to an agent's observation
        # noise_scale = 0.0  # adjust the standard deviation of the noise as needed
        # obs[attacked_agent] = obs[attacked_agent] + np.random.randn(*obs[attacked_agent].shape) * noise_scale

        # FGSM attack
        if do_attack and cnt >= config.atk_start_step and cnt <= config.atk_end_step and False:
            temp_torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=False) for i in range(maddpg.nagents)]
            temp_torch_agent_actions = maddpg.step(temp_torch_obs, explore=False)
            agent_actions = [ac.data.cpu().numpy() for ac in temp_torch_agent_actions]
            temp_actions = [agent_actions[i].squeeze() for i, agent_name in enumerate(env.possible_agents)]
            obs[atk_agent_id] = fgsm_attack(maddpg, obs, temp_actions, atk_agent_id, 0.1)
        
        torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=False) for i in range(maddpg.nagents)]
        torch_agent_actions = maddpg.step(torch_obs, explore=False)
        agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions]
        if maddpg.discrete_action:
            actions = {agent_name: agent_actions[i].argmax() for i, agent_name in enumerate(env.possible_agents)}
        else:
            actions = {agent_name: agent_actions[i].squeeze() for i, agent_name in enumerate(env.possible_agents)}

        # store current actions (ordered list) for Shapley computation
        action_list = [actions[env.possible_agents[i]] for i in range(maddpg.nagents)]
        action_history.append([a for a in action_list])

        # random attack
        if do_attack and False:
            actions[env.possible_agents[atk_agent_id]] = env.action_spaces[env.possible_agents[atk_agent_id]].sample()
        
        # Action Space Attacks
        if maddpg.discrete_action:
            # Compute entropy of action logits
            action_logits = maddpg.get_action_logits(torch_obs)
            atk_agent_action_probs = torch.softmax(action_logits[atk_agent_id].squeeze(), dim=0)
            atk_agent_log_probs = torch.log(atk_agent_action_probs)
            atk_agent_entropy = -torch.sum(atk_agent_action_probs * atk_agent_log_probs)
            if do_attack and atk_agent_entropy < 0.5 and cnt >= 5:
                do_start_attack = True
            # worst action attack for discrete action space
            # if do_attack and np.random.rand() < 0.75:
            # if do_attack and cnt >= config.atk_start_step and cnt <= config.atk_end_step:
            if do_start_attack and attack_step_remaining > 0:
                actions[env.possible_agents[atk_agent_id]] = torch.argmin(action_logits[atk_agent_id]).item()
                attacked_steps.append(cnt)
                attack_step_remaining -= 1
        else:
            if do_attack and cnt >= 5:
                do_start_attack = True
            # random action attack
            if do_start_attack and attack_step_remaining > 0:
                # actions[env.possible_agents[atk_agent_id]] = env.action_spaces[env.possible_agents[atk_agent_id]].sample()
                # attacked_steps.append(cnt)
                attack_step_remaining -= 1

        if config.save_gifs:
            frames.append(Image.fromarray(env.render()))
        
        # results = compute_taylor_delta(maddpg, obs, list(actions.values()), env.action_space, 0.1)
        results = compute_taylor_delta_policy(maddpg, obs, 0.01)
        # results = compute_eigen(maddpg, obs, list(actions.values()), env.action_space, 0.1)
        results_frob_norms = compute_frob_norms(maddpg, obs, list(actions.values()), env.action_space, atk_agent_id)
        # Pairwise Frobenius norms across all agent pairs for cascading impact analysis
        pairwise_frobs = compute_pairwise_frob_norms(maddpg, obs, list(actions.values()), env.action_space)
        frob_norms_matrix_history.append(pairwise_frobs)
        results_sec_dir_derivatives = compute_2nd_ord_dir_derivatives(maddpg, obs, list(actions.values()), env.action_space, atk_agent_id)

        for i in range(maddpg.nagents):
            result_deques[i].append(results[i])
            
            # Apply different detection methods
            if detection_method == 'mean_std':
                detection_value = np.mean(result_deques[i])
                threshold_exceeded = abs(detection_value - ref_vals[i][cnt]) > K_SIGMA * ref_std_devs[i][cnt]
            elif detection_method == 'median_mad':
                detection_value = np.mean(result_deques[i])
                threshold_exceeded = abs(detection_value - ref_vals[i][cnt]) > K_SIGMA * ref_std_devs[i][cnt]
            elif detection_method == 'diff':
                if cnt > 0:
                    current_diff = results[i] - prev_errors[i]
                    threshold_exceeded = abs(current_diff - ref_vals[i][cnt]) > K_SIGMA * ref_std_devs[i][cnt]
                    detection_value = current_diff
                else:
                    threshold_exceeded = False
                    detection_value = 0.0
            else:
                raise ValueError(f"Unknown detection method: {detection_method}")
            
            if threshold_exceeded:
                if i not in fault_first_detected:
                    print(f" [!!!] Anomaly detected for agent {i} at timestep: {cnt}. Method: {detection_method}. Value: {detection_value:.6f}")
                    fault_first_detected[i] = cnt
                    # Cascading Impact Analysis
                    prev_faults = [(f, tf) for f, tf in fault_first_detected.items() if f != i and tf < cnt]
                    contribs = {}
                    if len(prev_faults) > 0:
                        for f, tf in prev_faults:
                            # Mean Frobenius norm from t_f to current t for H_{i,f}
                            values_over_time = [frob_norms_matrix_history[tau][i][f] for tau in range(tf, cnt + 1) if tau < len(frob_norms_matrix_history)]
                            if len(values_over_time) > 0:
                                contribs[f] = float(np.mean(values_over_time))
                        if len(contribs) > 0:
                            ranked = sorted(contribs.items(), key=lambda x: x[1], reverse=True)
                            print(f"     >> Potential contributors to fault in agent {i} (mean ||H_{{i,f}}||_F from t_f to {cnt}): {ranked}")
                    fault_timeline.append({
                        'agent': i,
                        't': cnt,
                        'contribs': contribs
                    })
            frob_norms_deques[i].append(results_frob_norms[i])
            sec_dir_derivatives_deques[i].append(results_sec_dir_derivatives[i])

        metric_vals.append([np.mean(result_deques[i]) for i in range(maddpg.nagents)])
        prev_errors = results
        frob_norms_list.append([np.mean(frob_norms_deques[i]) for i in range(maddpg.nagents)])
        sec_dir_derivatives.append([np.mean(sec_dir_derivatives_deques[i]) for i in range(maddpg.nagents)])

        next_obs, rewards, dones, infos = env.step(actions)
        episode_reward += np.sum([rewards[:,i] if env.agent_types[i] != 'adversary' else np.zeros_like(rewards[:,i]) for i in range(maddpg.nagents)])
        episode_rewards = [episode_rewards[i] + rewards[:,i].sum() for i in range(maddpg.nagents)]

        obs = next_obs
        cnt += 1
        if dones.all():
            break

    # Compute Shapley values once at the end of the episode
    print("Computing episode Shapley values...")
    shapley_env_history = compute_episode_shapley_values(config, maddpg, action_history, seed, 
                                                         attacked_agent_id=atk_agent_id if do_attack else None,
                                                         attacked_steps=attacked_steps if do_attack else None)
    print(f"Episode Shapley values shape: {np.array(shapley_env_history).shape}")

    print(f"Episode reward: {episode_reward}")
    print(f"Episode rewards: {episode_rewards}")
    if config.save_gifs:
        imageio.mimsave(os.path.join(logdir, f'{config.env_id}_episode_atk_{atk_agent_id if do_attack else "free"}.gif'), frames, duration=125)
        print(f"Saved gif of episode to {logdir}")
    print("")
    return metric_vals, attacked_steps, frob_norms_list, sec_dir_derivatives, frob_norms_matrix_history, fault_timeline, obs_history, action_history, fault_first_detected, shapley_env_history


def plot_results(results_attacked, attacked_steps, atk_agent_id, ref_vals, ref_std_devs, logdir, detection_method='mean_std'):
    n = len(results_attacked[0])  # number of agents
    t = len(results_attacked)     # number of time steps
    
    # Create n subplots in a row
    max_per_row = 3
    rows = math.ceil(n / max_per_row)
    cols = min(n, max_per_row)
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten()  # so you can index axes[i] easily
    fig.suptitle(f'Taylor Error ({detection_method.upper().replace("_", "+")} | Worst Action Attack | Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.95)
    
    # Ensure axes is iterable even for single agent case
    if n == 1:
        axes = [axes]
    
    for i in range(n):
        ax = axes[i]
        
        # Extract time series for agent i
        attacked_series = [results_attacked[t][i] for t in range(len(results_attacked))]
        
        # For 'diff' detection method, plot the differences instead of raw values
        if detection_method == 'diff':
            # Calculate differences for plotting (skip first timestep as it has no previous value)
            diff_series = []
            for t in range(1, len(attacked_series)):
                diff = attacked_series[t] - attacked_series[t-1]
                diff_series.append(diff)
            
            # Update series to plot differences
            attacked_series = diff_series
            steps_length = len(attacked_series)
            steps = range(1, steps_length + 1)  # Start from timestep 1
        else:
            # Plot the curves normally
            steps_length = len(attacked_series)
            steps = range(steps_length)
        ref_vals[i] = ref_vals[i][:steps_length]
        ref_std_devs[i] = ref_std_devs[i][:steps_length]

        # Add green region using ref_vals and ref_std_devs
        ref_lower = [ref_vals[i][t] - K_SIGMA*ref_std_devs[i][t] for t in range(len(ref_vals[i]))]
        ref_upper = [ref_vals[i][t] + K_SIGMA*ref_std_devs[i][t] for t in range(len(ref_vals[i]))]
        ax.fill_between(steps, ref_lower, ref_upper, alpha=0.1, color='green')
        
        ax.plot(steps, attacked_series, 'r-', label='Observed', linewidth=2)
        ax.plot(steps, ref_vals[i], 'g--', label='Reference', linewidth=2)
        
        # Mark attacked timesteps with vertical lines
        if i == atk_agent_id and attacked_steps:
            # for attack_step in attacked_steps:
            #     ax.axvline(x=attack_step, color='orange', linestyle=':', alpha=0.5, linewidth=1.5)
            # # Add legend entry for attack markers
            # ax.axvline(x=attacked_steps[0], color='orange', linestyle=':', alpha=0.5, linewidth=1.5, label='Attacked Steps')
            start = min(attacked_steps)
            end = max(attacked_steps)
            ax.axvspan(start, end, color='red', alpha=0.1, label='Attacked Region')
        
        ax.set_xlabel('Step')
        if detection_method == 'diff':
            ax.set_ylabel('Taylor Error Difference')
        else:
            ax.set_ylabel('Taylor Delta Error')
        ax.set_title(f'Agent {i}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # hide the unused axes
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(logdir, f'plot_analysis_{detection_method}_attacked_{atk_agent_id}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved analysis plot to {logdir}")


def plot_frobs(frobs_normal, frobs_atk, attacked_steps, atk_agent_id, logdir):
    n = len(frobs_normal[0])  # number of agents
    t = len(frobs_normal)     # number of time steps
    
    # Create n subplots in a row
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    fig.suptitle(f'Frobenius Norms (Worst Action Attack | Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.95)
    
    # Ensure axes is iterable even for single agent case
    if n == 1:
        axes = [axes]
    
    for i in range(n):
        ax = axes[i]
        
        # Extract time series for agent i
        normal_series = [frobs_normal[t][i] for t in range(len(frobs_normal))]
        attacked_series = [frobs_atk[t][i] for t in range(len(frobs_atk))]
        
        # Plot the curves
        normal_steps = range(len(normal_series))
        attacked_steps = range(len(attacked_series))
        ax.plot(attacked_steps, attacked_series, 'r-', label='Under Attack', linewidth=2)
        ax.plot(normal_steps, normal_series, 'g-', label='Normal', linewidth=2)

        # Mark attacked timesteps with vertical lines
        if attacked_steps:
            for attack_step in attacked_steps:
                ax.axvline(x=attack_step, color='orange', linestyle=':', alpha=0.5, linewidth=1.5)
            # Add legend entry for attack markers
            ax.axvline(x=attacked_steps[0], color='orange', linestyle=':', alpha=0.5, linewidth=1.5, label='Attacked Steps')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Frobenius Norm')
        ax.set_title(f'Agent {i}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(logdir, f'plot_frobs_attacked_{atk_agent_id}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved frobenius norms plot to {logdir}")


def plot_sec_dir_derivatives(s_dir_derv_normal, s_dir_derv_atk, attacked_steps, atk_agent_id, logdir):
    n = len(s_dir_derv_normal[0])  # number of agents
    t = len(s_dir_derv_normal)     # number of time steps
    
    # Create n subplots in a row
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    fig.suptitle(f'2nd Ord. Dir. Derivatives (Worst Action Attack | Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.95)
    
    # Ensure axes is iterable even for single agent case
    if n == 1:
        axes = [axes]
    
    for i in range(n):
        ax = axes[i]
        
        # Extract time series for agent i
        normal_series = [s_dir_derv_normal[t][i] for t in range(len(s_dir_derv_normal))]
        attacked_series = [s_dir_derv_atk[t][i] for t in range(len(s_dir_derv_atk))]
        
        # Plot the curves
        normal_steps = range(len(normal_series))
        attacked_steps = range(len(attacked_series))

        ax.plot(normal_steps, normal_series, 'g-', label='Normal', linewidth=2)
        ax.plot(attacked_steps, attacked_series, 'r-', label='Under Attack', linewidth=2)
        
        # Highlight region under y < 0 in red
        y_min = min(min(normal_series), min(attacked_series))
        if y_min < 0:
            ax.axhspan(y_min * 1.1, 0, alpha=0.2, color='red')
        
        # Mark attacked timesteps with vertical lines
        if attacked_steps:
            for attack_step in attacked_steps:
                ax.axvline(x=attack_step, color='orange', linestyle=':', alpha=0.5, linewidth=1.5)
            # Add legend entry for attack markers
            ax.axvline(x=attacked_steps[0], color='orange', linestyle=':', alpha=0.5, linewidth=1.5, label='Attacked Steps')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('2nd Ord. Dir. Derivative')
        ax.set_title(f'Agent {i}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(logdir, f'plot_sec_dir_derivatives_attacked_{atk_agent_id}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved 2nd ord. dir. derivatives plot to {logdir}")


def plot_fault_timeline(fault_timeline, total_agents, logdir):
    if len(fault_timeline) == 0:
        print("No faults detected; skipping fault timeline plot.")
        return

    k = len(fault_timeline)
    fig = plt.figure(figsize=(max(6, 3*k), 5))  # reduce height from 6 → 5
    gs = fig.add_gridspec(
        3, k,
        height_ratios=[0.8, 2, 0.1],  # smaller top & bottom rows
        hspace=0.1  # tighter vertical spacing
    )

    cmap = plt.get_cmap('tab20')
    agent_colors = {i: cmap(i % 20) for i in range(total_agents)}

    # --- Timeline axis (top row) ---
    ax_timeline = fig.add_subplot(gs[0, :])
    ax_timeline.axis('off')

    # Horizontal arrow for timeline
    arrow_y = 0.5
    ax_timeline.annotate(
        '', xy=(1, arrow_y), xytext=(0, arrow_y),
        arrowprops=dict(arrowstyle='-|>', lw=2, color='gray'),
        xycoords='axes fraction', textcoords='axes fraction'
    )

    # Milestones
    for i, event in enumerate(fault_timeline):
        frac_x = (i + 0.5) / k  # evenly spaced

        # Circle marker
        ax_timeline.plot(frac_x, arrow_y, 'o', color='darkred', markersize=10, transform=ax_timeline.transAxes)

        # Faulty agent label above
        ax_timeline.text(frac_x, arrow_y + 0.15,
                         f"Faulty agent {event['agent']}",
                         ha='center', va='bottom',
                         fontsize=11, fontweight='bold',
                         transform=ax_timeline.transAxes)

        # Timestep label below
        ax_timeline.text(frac_x, arrow_y - 0.15,
                         f"t = {event['t']}",
                         ha='center', va='top',
                         fontsize=10, color='darkblue',
                         transform=ax_timeline.transAxes)

    # --- Contributor charts (middle row) ---
    for col, event in enumerate(fault_timeline):
        ax = fig.add_subplot(gs[1, col])
        contribs = event.get('contribs', {})

        if len(contribs) == 0:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No prior faults',
                    ha='center', va='center', fontsize=10, style='italic')
        else:
            vals = np.array(list(contribs.values()), dtype=float)
            if vals.sum() > 0:
                vals /= vals.sum()
            colors = [agent_colors[a] for a in contribs.keys()]

            wedges, _, autotexts = ax.pie(
                vals, autopct='%1.1f%%', startangle=90, colors=colors,
                wedgeprops=dict(width=0.35, edgecolor='w')
            )
            for at in autotexts:
                at.set_fontsize(9)
                at.set_fontweight('bold')
            ax.set_title('Contributors', fontsize=11, pad=5)
            ax.set_aspect('equal')

    # --- Legend row (bottom row) ---
    ax_legend = fig.add_subplot(gs[2, :])
    ax_legend.axis('off')
    legend_elements = [Patch(facecolor=agent_colors[i], label=f"Agent {i}") for i in range(total_agents)]
    ax_legend.legend(handles=legend_elements, loc='center', ncol=total_agents,
                     fontsize=9, frameon=False)

    fig.suptitle('Fault Detection Timeline and Contributors',
                 fontsize=14, fontweight='bold', y=0.96)

    out_path = os.path.join(logdir, 'fault_timeline.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved fault timeline plot to {out_path}")


def plot_contributor_barchart(fault_timeline, total_agents, logdir):
    if len(fault_timeline) == 0:
        print("No faults detected; skipping contributor bar chart.")
        return

    k = len(fault_timeline)
    # Increase figure width for better spacing, especially with many events
    fig = plt.figure(figsize=(max(8, 4*k), 6))  # Increased from 3*k to 4*k width and 5 to 6 height
    gs = fig.add_gridspec(
        3, k,
        height_ratios=[0.8, 2.5, 0.2],  # Give more space to the middle row
        hspace=0.15,  # Increase vertical spacing
        wspace=0.3    # Add horizontal spacing between subplots
    )

    cmap = plt.get_cmap('tab20')
    agent_colors = {i: cmap(i % 20) for i in range(total_agents)}

    # --- Timeline axis (top row) ---
    ax_timeline = fig.add_subplot(gs[0, :])
    ax_timeline.axis('off')

    arrow_y = 0.5
    ax_timeline.annotate(
        '', xy=(1, arrow_y), xytext=(0, arrow_y),
        arrowprops=dict(arrowstyle='-|>', lw=2, color='gray'),
        xycoords='axes fraction', textcoords='axes fraction'
    )

    for i, event in enumerate(fault_timeline):
        frac_x = (i + 0.5) / k
        ax_timeline.plot(frac_x, arrow_y, 'o', color='darkred', markersize=10, transform=ax_timeline.transAxes)
        ax_timeline.text(frac_x, arrow_y + 0.15,
                         f"Faulty agent {event['agent']}",
                         ha='center', va='bottom',
                         fontsize=11, fontweight='bold',
                         transform=ax_timeline.transAxes)
        ax_timeline.text(frac_x, arrow_y - 0.15,
                         f"t = {event['t']}",
                         ha='center', va='top',
                         fontsize=10, color='darkblue',
                         transform=ax_timeline.transAxes)

    # --- Contributor bar charts (middle row) ---
    for col, event in enumerate(fault_timeline):
        ax = fig.add_subplot(gs[1, col])
        contribs = event.get('contribs', {})

        if len(contribs) == 0:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No prior faults',
                    ha='center', va='center', fontsize=10, style='italic')
        else:
            agents = list(contribs.keys())
            scores = np.array(list(contribs.values()), dtype=float)

            colors = [agent_colors[a] for a in agents]

            # Use narrower bars with proper spacing
            bar_width = 0.6  # Make bars narrower
            x_positions = range(len(agents))
            bars = ax.bar(x_positions, scores, color=colors, width=bar_width, 
                         edgecolor='black', linewidth=0.5, alpha=0.8)

            # Set appropriate x limits with padding
            if len(agents) > 1:
                ax.set_xlim(-0.8, len(agents) - 0.2)
            else:
                ax.set_xlim(-0.8, 0.8)

            # Improved label handling
            ax.set_xticks(x_positions)
            if len(agents) <= 3:
                # For few agents, use normal labels
                ax.set_xticklabels([f"Agent {i}" for i in agents], fontsize=9)
            else:
                # For many agents, use abbreviated labels with rotation
                ax.set_xticklabels([f"A{i}" for i in agents], rotation=45, ha='right', fontsize=8)

            # Add value labels on top of bars for clarity
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(scores)*0.01,
                       f'{score:.3f}', ha='center', va='bottom', fontsize=7)

            ax.set_ylabel("Contribution", fontsize=9)
            ax.set_title('Contributors', fontsize=11, pad=10)

            # Grid for readability
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Set y-axis to start from 0 for better visual comparison
            ax.set_ylim(bottom=0)

    # --- Legend row (bottom row) ---
    ax_legend = fig.add_subplot(gs[2, :])
    ax_legend.axis('off')
    legend_elements = [Patch(facecolor=agent_colors[i], label=f"Agent {i}") for i in range(total_agents)]
    ax_legend.legend(handles=legend_elements, loc='center', ncol=total_agents,
                     fontsize=9, frameon=False)

    fig.suptitle('Fault Detection Timeline and Contributor Scores',
                 fontsize=14, fontweight='bold', y=0.97)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # Adjusted margins for better label visibility

    out_path = os.path.join(logdir, 'fault_contributor_barchart.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.2)  # Added padding
    plt.show()
    print(f"Saved contributor bar chart to {out_path}")


def plot_shapley_analysis(shapley_values, fault_first_detected, attacked_agent_id, total_agents, logdir, is_attack_scenario=True):
    """
    Create a comprehensive visualization of Shapley value analysis.
    
    Args:
        shapley_values: Dictionary or list of Shapley values per agent
        fault_first_detected: Dictionary of agent_id -> first detection timestep
        attacked_agent_id: ID of attacked agent (None for normal scenario)
        total_agents: Total number of agents
        logdir: Directory to save plots
        is_attack_scenario: Whether this is an attack scenario or normal scenario
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Shapley Values Bar Chart
    agents = list(range(total_agents))
    values = [shapley_values[i] for i in agents]
    
    if is_attack_scenario and attacked_agent_id is not None:
        colors = ['red' if i == attacked_agent_id else 'lightblue' for i in agents]
        title = 'Shapley Values (Attack Scenario)'
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label=f'Actual Attacked Agent ({attacked_agent_id})'),
            Patch(facecolor='lightblue', alpha=0.7, label='Other Agents')
        ]
    else:
        colors = ['lightblue' for i in agents]
        title = 'Shapley Values'
        legend_elements = [
            Patch(facecolor='lightblue', alpha=0.7, label='All Agents')
        ]
    
    bars = ax1.bar(agents, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    
    # Add detection markers only for attack scenario
    if is_attack_scenario:
        for i, bar in enumerate(bars):
            if i in fault_first_detected:
                ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(values)*0.01,
                        f'Det@{fault_first_detected[i]}', ha='center', va='bottom', 
                        fontsize=8, color='red', fontweight='bold')
    
    ax1.set_xlabel('Agent ID')
    ax1.set_ylabel('Shapley Value')
    ax1.set_title(title)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Plot 2: Detection Timeline vs Shapley Ranking (only for attack scenarios)
    if is_attack_scenario and fault_first_detected:
        # Sort agents by Shapley values (descending)
        sorted_by_shapley = sorted(range(total_agents), key=lambda x: shapley_values[x], reverse=True)
        
        detection_times = [fault_first_detected.get(i, -1) for i in sorted_by_shapley]
        shapley_ranks = list(range(1, len(sorted_by_shapley) + 1))
        
        # Create scatter plot
        detected_agents = [i for i in sorted_by_shapley if i in fault_first_detected]
        detected_times = [fault_first_detected[i] for i in detected_agents]
        detected_ranks = [sorted_by_shapley.index(i) + 1 for i in detected_agents]
        
        colors_scatter = ['red' if i == attacked_agent_id else 'blue' for i in detected_agents]
        
        ax2.scatter(detected_ranks, detected_times, c=colors_scatter, s=100, alpha=0.7, edgecolors='black')
        
        # Label points with agent IDs
        for i, (rank, time) in enumerate(zip(detected_ranks, detected_times)):
            agent_id = detected_agents[i]
            ax2.annotate(f'Agent {agent_id}', (rank, time), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9)
        
        ax2.set_xlabel('Shapley Value Rank (1 = Highest)')
        ax2.set_ylabel('First Detection Timestep')
        ax2.set_title('Detection Time vs Shapley Ranking')
        ax2.grid(True, alpha=0.3)
        
        # Add ideal line if attacked agent was detected
        if attacked_agent_id is not None and attacked_agent_id in fault_first_detected:
            attacked_rank = sorted_by_shapley.index(attacked_agent_id) + 1
            attacked_detection_time = fault_first_detected[attacked_agent_id]
            ax2.axvline(x=attacked_rank, color='red', linestyle='--', alpha=0.5, 
                       label=f'Actual Attacked Agent (Rank {attacked_rank})')
            ax2.legend()
    else:
        if is_attack_scenario:
            ax2.text(0.5, 0.5, 'No faults detected', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=14, style='italic')
            ax2.set_title('Detection Timeline (No Faults Detected)')
        else:
            # For normal scenario, show Shapley value distribution
            ax2.hist(values, bins=min(10, total_agents), alpha=0.7, color='lightblue', edgecolor='black')
            ax2.set_xlabel('Shapley Value')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Shapley Value Distribution')
            ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    scenario_suffix = 'normal' if not is_attack_scenario else f'atk_{attacked_agent_id}'
    out_path = os.path.join(logdir, f'shapley_analysis_{scenario_suffix}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved Shapley analysis plot to {out_path}")


def plot_shapley_timeseries(shapley_ts, attacked_agent_id, total_agents, logdir, is_attack_scenario=True):
    """
    Plot timestep-wise Shapley values for all agents in a single figure.
    
    Args:
        shapley_ts: Time series of Shapley values (T x N)
        attacked_agent_id: ID of attacked agent (None for normal scenario)
        total_agents: Total number of agents
        logdir: Directory to save plots
        is_attack_scenario: Whether this is an attack scenario or normal scenario
    """
    T, N = shapley_ts.shape
    plt.figure(figsize=(max(8, T/5), 6))
    
    for i in range(N):
        if is_attack_scenario and attacked_agent_id is not None and i == attacked_agent_id:
            # Highlight attacked agent in attack scenarios
            plt.plot(range(T), shapley_ts[:, i], label=f'Agent {i} (Attacked)', 
                    linewidth=2.5, color='red')
        else:
            plt.plot(range(T), shapley_ts[:, i], label=f'Agent {i}', linewidth=1.5)
    
    plt.xlabel('Timestep')
    plt.ylabel('Shapley Value')
    
    if is_attack_scenario and attacked_agent_id is not None:
        title = f'Timestep-wise Shapley Values (Attacked Agent: {attacked_agent_id})'
    else:
        title = 'Timestep-wise Shapley Values'
    
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.grid(alpha=0.3)
    
    scenario_suffix = 'normal' if not is_attack_scenario else f'atk_{attacked_agent_id}'
    out_path = os.path.join(logdir, f'shapley_timeseries_{scenario_suffix}.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved shapley timeseries to {out_path}")


def plot_shapley_mean_barchart(mean_shapley, attacked_agent_id, total_agents, logdir, is_attack_scenario=True):
    """
    Plot mean Shapley values across episode as a bar chart.
    
    Args:
        mean_shapley: Mean Shapley values per agent
        attacked_agent_id: ID of attacked agent (None for normal scenario)
        total_agents: Total number of agents
        logdir: Directory to save plots
        is_attack_scenario: Whether this is an attack scenario or normal scenario
    """
    agents = list(range(total_agents))
    values = mean_shapley
    
    # Create different colors for each agent using a colormap
    cmap = plt.get_cmap('tab10')
    colors = [cmap(i % 10) for i in range(total_agents)]
    
    # Highlight attacked agent with red color in attack scenarios
    if is_attack_scenario and attacked_agent_id is not None:
        colors[attacked_agent_id] = 'red'
        title = f'Mean Shapley Value per Agent (Attacked: {attacked_agent_id})'
    else:
        title = 'Mean Shapley Value per Agent'
    
    plt.figure(figsize=(max(6, total_agents), 5))
    bars = plt.bar(agents, values, color=colors, edgecolor='black', alpha=0.8)
    
    # Fix x-axis to show integer agent IDs
    plt.xticks(agents)
    plt.xlabel('Agent ID')
    plt.ylabel('Mean Shapley Value')
    plt.title(title)
    
    # Add value labels on top of bars
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(values)*0.01 if max(values)!=0 else 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Create legend
    legend_labels = [f'Agent {i}' for i in range(total_agents)]
    if is_attack_scenario and attacked_agent_id is not None:
        legend_labels[attacked_agent_id] = f'Agent {attacked_agent_id} (Attacked)'
    
    legend_patches = [plt.matplotlib.patches.Patch(color=colors[i], label=legend_labels[i]) for i in range(total_agents)]
    plt.legend(handles=legend_patches, loc='upper right', fontsize=9)
    
    scenario_suffix = 'normal' if not is_attack_scenario else f'atk_{attacked_agent_id}'
    out_path = os.path.join(logdir, f'shapley_mean_barchart_{scenario_suffix}.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved mean shapley barchart to {out_path}")


def plot_influence_pies(frob_matrix_history, attacked_agent_id, total_agents, logdir, is_attack_scenario=True):
    """
    For each agent i, plot a pie chart showing influence (mean frob_ij across episode) of other agents on i.
    
    Args:
        frob_matrix_history: list of T elements, each is N x N matrix
        attacked_agent_id: ID of attacked agent (None for normal scenario)
        total_agents: Total number of agents
        logdir: Directory to save plots
        is_attack_scenario: Whether this is an attack scenario or normal scenario
    """
    if len(frob_matrix_history) == 0:
        print("No frobenius history; skipping influence pies.")
        return

    T = len(frob_matrix_history)
    N = total_agents
    # compute mean across time for each (i,j)
    mean_matrix = np.zeros((N, N), dtype=float)
    for t in range(T):
        mean_matrix += np.array(frob_matrix_history[t])
    mean_matrix /= float(T)

    # Create subplots: one pie per agent
    cols = min(4, N)
    rows = int(math.ceil(N / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.8*cols, 4*rows))
    axes = axes.flatten() if N > 1 else [axes]

    cmap = plt.get_cmap('tab10')
    agent_colors = [cmap(i % 10) for i in range(N)]

    for i in range(N):
        ax = axes[i]
        # influence of j on i is mean_matrix[i][j]
        vals = mean_matrix[i, :]
        # avoid all zeros
        if vals.sum() <= 0:
            ax.text(0.5, 0.5, 'No influence data', ha='center', va='center')
            ax.axis('off')
            continue

        # normalize to percent
        vals_norm = vals / vals.sum()
        colors = [agent_colors[j] for j in range(N)]
        
        # Highlight attacked agent in attack scenarios
        if is_attack_scenario and attacked_agent_id is not None:
            # Make attacked agent slice more prominent
            colors = [agent_colors[j] if j != attacked_agent_id else 'red' for j in range(N)]
            explode = [0.1 if j == attacked_agent_id else 0 for j in range(N)]
            wedges, texts, autotexts = ax.pie(vals_norm, colors=colors, autopct='%1.1f%%', startangle=90, explode=explode)
        else:
            wedges, texts, autotexts = ax.pie(vals_norm, colors=colors, autopct='%1.1f%%', startangle=90)
        
        ax.set_title(f'Influence on Agent {i}')
        ax.axis('equal')

    # remove any extra axes
    for k in range(N, len(axes)):
        fig.delaxes(axes[k])

    # Create a single legend at the bottom of the figure
    legend_labels = [f'Agent {j}' for j in range(N)]
    legend_colors = agent_colors.copy()
    
    if is_attack_scenario and attacked_agent_id is not None:
        legend_labels[attacked_agent_id] = f'Agent {attacked_agent_id} (Attacked)'
        legend_colors[attacked_agent_id] = 'red'
    
    # Create legend patches
    legend_patches = [plt.matplotlib.patches.Patch(color=legend_colors[j], label=legend_labels[j]) for j in range(N)]
    fig.legend(handles=legend_patches, loc='lower center', ncol=min(N, 5), 
               bbox_to_anchor=(0.5, -0.05), fontsize=10)

    if is_attack_scenario and attacked_agent_id is not None:
        title = f'Inter-Agent Influence (Attacked Agent: {attacked_agent_id})'
        scenario_suffix = f'atk_{attacked_agent_id}'
    else:
        title = 'Inter-Agent Influence'
        scenario_suffix = 'normal'
    
    plt.suptitle(title)
    out_path = os.path.join(logdir, f'influence_pies_{scenario_suffix}.png')
    plt.tight_layout(rect=[0, 0.1, 1, 0.96])  # Leave space for legend at bottom
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved influence pies to {out_path}")


def save_matrix_to_files(matrix, attacked_steps, attacked_agent_id, total_agents, logdir, filename):
    """
    Save matrix data to a CSV file for all timesteps.
    
    Args:
        matrix: List of timesteps, where each timestep contains n_agent data
        attacked_agent_id: ID of the attacked agent
        total_agents: Total number of agents
        logdir: Directory to save the file
    """
    filepath = os.path.join(logdir, filename)
    
    # Create header row
    # header = ["timestep", "attacked_agent"]
    header = ["timestep", "is_attacked", "attacked_agent"]
    for i in range(total_agents):
        header.append(f"agent_{i}")
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        for timestep, timestep_data in enumerate(matrix):
            is_attacked = 1 if timestep in attacked_steps else 0
            row = [timestep, is_attacked, attacked_agent_id]
            for i in range(total_agents):
                row.append(timestep_data[i])
            writer.writerow(row)
    
    print(f"Saved {len(matrix)} timestep matrices to {filepath}")


def run(config):
    maddpg = MADDPG.init_from_save(config.model_path, test_mode=True)

    # create a log directory under runs/<env_id>/<timestamp> using os and getcwd
    cwd = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(cwd, 'runs', f"{config.env_id}_{'discrete' if maddpg.discrete_action else 'continuous'}", timestamp)
    os.makedirs(logdir, exist_ok=True)

    try:
        env_func = getattr(mpe, config.env_id)
        if config.env_id == "simple_spread_v3":
            env = env_func.parallel_env(continuous_actions= not maddpg.discrete_action, render_mode='rgb_array', N=maddpg.nagents)
            test_env = env_func.parallel_env(continuous_actions= not maddpg.discrete_action, render_mode='rgb_array', N=maddpg.nagents)
        else:
            env = env_func.parallel_env(continuous_actions= not maddpg.discrete_action, render_mode='rgb_array')
            test_env = env_func.parallel_env(continuous_actions= not maddpg.discrete_action, render_mode='rgb_array')
    except:
        try:
            env_func = getattr(sisl, config.env_id)
            env = env_func.parallel_env(n_pursuers=5, render_mode='rgb_array') if config.env_id == 'waterworld_v4' else env_func.parallel_env(render_mode='rgb_array')
            test_env = env_func.parallel_env(n_pursuers=5, render_mode='rgb_array') if config.env_id == 'waterworld_v4' else env_func.parallel_env(render_mode='rgb_array')
        except:
            env_func = getattr(atari, config.env_id)
            env = env_func.parallel_env(render_mode='rgb_array')
            env = preprocess_env_atari(env)
            test_env = env_func.parallel_env(render_mode='rgb_array')
            test_env = preprocess_env_atari(test_env)

    env = PettingZooWrapper.wrap_env(env)
    env.reset()
    test_env = PettingZooWrapper.wrap_env(test_env)
    test_env.reset()

    # maddpg.prep_rollouts(device=DEVICE)
    maddpg.prep_training(device=DEVICE)

    # Read reference values from CSV files if provided
    ref_vals = [[] for _ in range(maddpg.nagents)]
    ref_std_devs = [[] for _ in range(maddpg.nagents)]

    for agent_id in range(maddpg.nagents):
        csv_filename = f"maddpg_taylor_error_atk_free_agent_{agent_id}.csv"
        csv_path = os.path.join(config.ref_val_dir, csv_filename)
        
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                if config.detection_method == 'mean_std':
                    # Use mean and std_dev columns
                    ref_vals[agent_id].append(float(row[2]))  # mean
                    ref_std_devs[agent_id].append(float(row[4]))  # std_dev
                elif config.detection_method == 'median_mad':
                    # Use median and MAD columns
                    ref_vals[agent_id].append(float(row[7]))  # median
                    ref_std_devs[agent_id].append(float(row[8]))  # MAD
                elif config.detection_method == 'diff':
                    # Use diff_mean and diff_std columns
                    ref_vals[agent_id].append(float(row[9]))  # diff_mean
                    ref_std_devs[agent_id].append(float(row[10]))  # diff_std
                else:
                    raise ValueError(f"Unknown detection method: {config.detection_method}")

    attacked_agent_id = config.attack_agent_id  # specify the agent to attack
    global ATTACKED_AGENT
    ATTACKED_AGENT = attacked_agent_id
    seed = SEED

    results_normal, _, frob_norms_normal, sec_dir_derivatives_normal, frob_norms_matrix_history_normal, _, obs_history_normal, action_history_normal, _, shapley_env_normal = get_episode_data(env, test_env, maddpg, config, logdir, ref_vals, ref_std_devs, config.detection_method, do_attack=False, atk_agent_id=attacked_agent_id, seed=seed)

    results_attacked, attacked_steps, frob_norms_atk, sec_dir_derivatives_atk, frob_norms_matrix_history, fault_timeline, obs_history_attacked, action_history_attacked, fault_first_detected, shapley_env_attacked = get_episode_data(env, test_env, maddpg, config, logdir, ref_vals, ref_std_devs, config.detection_method, do_attack=True, atk_agent_id=attacked_agent_id, seed=seed)

    # Use per-timestep Shapley values computed at end of episode for efficiency
    print("\n" + "="*60)
    print("SHAPLEY VALUE ANALYSIS (episode-based, computed once at end)")
    print("="*60)

    # shapley_env_attacked is a list of per-timestep numpy arrays (T x N)
    shapley_ts_attacked = np.vstack(shapley_env_attacked)
    mean_shapley_attacked = np.mean(shapley_ts_attacked, axis=0)
    
    # shapley_env_normal is also a list of per-timestep numpy arrays (T x N)
    shapley_ts_normal = np.vstack(shapley_env_normal)
    mean_shapley_normal = np.mean(shapley_ts_normal, axis=0)

    # Save Shapley values (mean per agent) to file for attacked scenario
    shapley_csv_path = os.path.join(logdir, f'shapley_values_atk_{attacked_agent_id}.csv')
    with open(shapley_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['agent_id', 'shapley_value', 'first_detected_timestep', 'actual_attacked_agent'])
        for agent_id in range(maddpg.nagents):
            detection_time = fault_first_detected.get(agent_id, -1)
            is_actual_attacker = 1 if agent_id == attacked_agent_id else 0
            writer.writerow([agent_id, mean_shapley_attacked[agent_id], detection_time, is_actual_attacker])
    print(f"Saved Shapley values (attacked) to {shapley_csv_path}")
    
    # Save Shapley values (mean per agent) to file for normal scenario
    shapley_csv_path_normal = os.path.join(logdir, f'shapley_values_normal.csv')
    with open(shapley_csv_path_normal, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['agent_id', 'shapley_value'])
        for agent_id in range(maddpg.nagents):
            writer.writerow([agent_id, mean_shapley_normal[agent_id]])
    print(f"Saved Shapley values (normal) to {shapley_csv_path_normal}")

    # Save per-timestep Shapley timeseries to CSV for attacked scenario
    shapley_ts_path = os.path.join(logdir, f'shapley_timeseries_atk_{attacked_agent_id}.csv')
    with open(shapley_ts_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['timestep'] + [f'agent_{i}' for i in range(maddpg.nagents)]
        writer.writerow(header)
        for t, row in enumerate(shapley_ts_attacked):
            writer.writerow([t] + row.tolist())
    print(f"Saved Shapley timeseries (attacked) to {shapley_ts_path}")
    
    # Save per-timestep Shapley timeseries to CSV for normal scenario
    shapley_ts_path_normal = os.path.join(logdir, f'shapley_timeseries_normal.csv')
    with open(shapley_ts_path_normal, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['timestep'] + [f'agent_{i}' for i in range(maddpg.nagents)]
        writer.writerow(header)
        for t, row in enumerate(shapley_ts_normal):
            writer.writerow([t] + row.tolist())
    print(f"Saved Shapley timeseries (normal) to {shapley_ts_path_normal}")
    
    save_matrix_to_files(results_attacked, attacked_steps, attacked_agent_id, maddpg.nagents, logdir, f'maddpg_taylor_error_atk_{attacked_agent_id}.csv')
    save_matrix_to_files(frob_norms_atk, attacked_steps, attacked_agent_id, maddpg.nagents, logdir, f'maddpg_frobenius_norms_atk_{attacked_agent_id}.csv')
    save_matrix_to_files(sec_dir_derivatives_atk, attacked_steps, attacked_agent_id, maddpg.nagents, logdir, f'maddpg_sec_dir_derivatives_atk_{attacked_agent_id}.csv')

    plot_results(results_attacked, attacked_steps, attacked_agent_id, ref_vals, ref_std_devs, logdir, config.detection_method)
    plot_frobs(frob_norms_normal, frob_norms_atk, attacked_steps, attacked_agent_id, logdir)
    plot_sec_dir_derivatives(sec_dir_derivatives_normal, sec_dir_derivatives_atk, attacked_steps, attacked_agent_id, logdir)
    plot_fault_timeline(fault_timeline, maddpg.nagents, logdir)
    plot_contributor_barchart(fault_timeline, maddpg.nagents, logdir)
    
    # Plot Shapley analysis for both normal and attacked scenarios
    print("\nGenerating Shapley value plots...")
    
    # Normal scenario plots
    plot_shapley_timeseries(shapley_ts_normal, None, maddpg.nagents, logdir, is_attack_scenario=False)
    plot_shapley_mean_barchart(mean_shapley_normal, None, maddpg.nagents, logdir, is_attack_scenario=False)
    plot_influence_pies(frob_norms_matrix_history_normal, None, maddpg.nagents, logdir, is_attack_scenario=False)
    plot_shapley_analysis(dict(enumerate(mean_shapley_normal)), {}, None, maddpg.nagents, logdir, is_attack_scenario=False)
    
    # Attacked scenario plots
    plot_shapley_timeseries(shapley_ts_attacked, attacked_agent_id, maddpg.nagents, logdir, is_attack_scenario=True)
    plot_shapley_mean_barchart(mean_shapley_attacked, attacked_agent_id, maddpg.nagents, logdir, is_attack_scenario=True)
    plot_influence_pies(frob_norms_matrix_history, attacked_agent_id, maddpg.nagents, logdir, is_attack_scenario=True)
    plot_shapley_analysis(dict(enumerate(mean_shapley_attacked)), fault_first_detected, attacked_agent_id, maddpg.nagents, logdir, is_attack_scenario=True)
    
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_path",
                        help="model directory")
    parser.add_argument("--save_gifs", action="store_true",
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--ref_val_dir", type=str, default='',)
    parser.add_argument("--attack_agent_id", type=int, default=0,)
    parser.add_argument("--atk_start_step", type=int, default=-math.inf)
    parser.add_argument("--atk_end_step", type=int, default=math.inf)
    parser.add_argument("--detection_method", type=str, default='mean_std', 
                        choices=['mean_std', 'median_mad', 'diff'],
                        help="Detection method to use: 'mean_std', 'median_mad', or 'diff'")

    config = parser.parse_args()

    run(config)
