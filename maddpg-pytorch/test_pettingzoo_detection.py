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

USE_CUDA = torch.cuda.is_available()
DEVICE = 'gpu' if USE_CUDA else 'cpu'
torch_device = torch.device("cuda" if USE_CUDA else "cpu")
K_SIGMA = 0.9

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


def get_episode_data(env, maddpg, config, logdir, ref_vals, ref_std_devs, detection_method='mean_std', do_attack=False, atk_agent_id=-1, seed=None):
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

    prev_errors = [0 for i in range(maddpg.nagents)]

    while True:
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
        results = compute_taylor_delta_policy(maddpg, obs, list(actions.values()), env.action_space, 0.01)
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

    print(f"Episode reward: {episode_reward}")
    print(f"Episode rewards: {episode_rewards}")
    if config.save_gifs:
        imageio.mimsave(os.path.join(logdir, f'{config.env_id}_episode_atk_{atk_agent_id if do_attack else "free"}.gif'), frames, duration=125)
        print(f"Saved gif of episode to {logdir}")
    print("")
    return metric_vals, attacked_steps, frob_norms_list, sec_dir_derivatives, frob_norms_matrix_history, fault_timeline


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
            env = env_func.parallel_env(continuous_actions= not maddpg.discrete_action, render_mode='rgb_array', N=5)
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
    seed = 53

    results_normal, _, frob_norms_normal, sec_dir_derivatives_normal, _, _ = get_episode_data(env, maddpg, config, logdir, ref_vals, ref_std_devs, config.detection_method, do_attack=False, atk_agent_id=attacked_agent_id, seed=seed)

    results_attacked, attacked_steps, frob_norms_atk, sec_dir_derivatives_atk, frob_norms_matrix_history, fault_timeline = get_episode_data(env, maddpg, config, logdir, ref_vals, ref_std_devs, config.detection_method, do_attack=True, atk_agent_id=attacked_agent_id, seed=seed)
    save_matrix_to_files(results_attacked, attacked_steps, attacked_agent_id, maddpg.nagents, logdir, f'maddpg_taylor_error_atk_{attacked_agent_id}.csv')
    save_matrix_to_files(frob_norms_atk, attacked_steps, attacked_agent_id, maddpg.nagents, logdir, f'maddpg_frobenius_norms_atk_{attacked_agent_id}.csv')
    save_matrix_to_files(sec_dir_derivatives_atk, attacked_steps, attacked_agent_id, maddpg.nagents, logdir, f'maddpg_sec_dir_derivatives_atk_{attacked_agent_id}.csv')

    plot_results(results_attacked, attacked_steps, attacked_agent_id, ref_vals, ref_std_devs, logdir, config.detection_method)
    plot_frobs(frob_norms_normal, frob_norms_atk, attacked_steps, attacked_agent_id, logdir)
    plot_sec_dir_derivatives(sec_dir_derivatives_normal, sec_dir_derivatives_atk, attacked_steps, attacked_agent_id, logdir)
    plot_fault_timeline(fault_timeline, maddpg.nagents, logdir)
    plot_contributor_barchart(fault_timeline, maddpg.nagents, logdir)
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
