from collections import deque
from make_env_pettingzoo import make_env
from datetime import datetime
import torch
import torch.nn as nn
import argparse
import gymnasium
import numpy as np
import os
import imageio
from PIL import Image, ImageDraw
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import csv
import math

from MAPPO_MPE_main import Runner_MAPPO_MPE


def perturb_random_noise(states, perturb_agent_id, noise_std=0.1):
    perturbed_states = states.copy()
    perturbed_states[perturb_agent_id] = states[perturb_agent_id] + noise_std * torch.randn_like(states[perturb_agent_id])
    return perturbed_states

def perturb_fgsm(states, perturb_agent_id, perturb_eps=0.1):
    p_states_tensor = torch.cat([torch.tensor(state, dtype=torch.float32, requires_grad=True) for state in states], dim=0)
    # Compute the gradient of the value function with respect to the observation
    values = runner.agent_n.compute_value(p_states_tensor).squeeze(-1)  # shape: (N,)
    # Compute the gradient for the perturbed agent. Taking negative of the value since fgsm is a maximization attack
    grad = torch.autograd.grad(-values[perturb_agent_id], p_states_tensor, create_graph=True)[0]
    # Perturb the observation in the direction of the gradient
    agent_state_dim = len(states[perturb_agent_id])
    return states[perturb_agent_id] + perturb_eps * np.sign(grad[perturb_agent_id*agent_state_dim : (perturb_agent_id+1)*agent_state_dim].detach().cpu().numpy())


def compute_frob_norms(runner: Runner_MAPPO_MPE, states, vulnerable_agent_id):
    states_tensors = [torch.tensor(state, dtype=torch.float32, requires_grad=True) for state in states]
    states_tensor = torch.cat(states_tensors, dim=0)
    values = runner.agent_n.compute_value(states_tensor).squeeze(-1)  # shape: (N,)

    # Store eigenvalues for each agent pair (i, j)
    results = []

    for i in range(runner.args.N):
        # Compute first-order gradient with respect to agent i's observation
        grad_i = torch.autograd.grad(values[i], states_tensors[i], create_graph=True, retain_graph=True)[0]

        # Compute cross-agent Hessian matrix for agent pair (i, j)
        # This represents ∂²v/∂obs_i∂obs_j
        hessian_matrix = []
        
        for k in range(grad_i.shape[0]):  # For each dimension of agent i's observation (has shape (obs_dim,))
            # Compute ∂²v/∂obs_i[k]∂obs_j
            second_grad = torch.autograd.grad(
                grad_i[k], 
                states_tensors[vulnerable_agent_id],
                retain_graph=True, 
                allow_unused=True
            )[0]
            hessian_matrix.append(second_grad.flatten())

        # Convert to tensor and compute eigenvalues
        H = torch.stack(hessian_matrix)

        # eigenvals = torch.linalg.eigvals(H)
        # eigenval = torch.max(eigenvals.real).item()
        # results.append(eigenval)
        # # results.append(torch.trace(H).item())  # Trace of the Hessian matrix
        # continue
        # # Compute Frobenius norm of the Hessian matrix
        results.append(H.norm(p='fro').item())

    return results

def compute_pairwise_frob_norms(runner: Runner_MAPPO_MPE, states):
    """
    Compute Frobenius norms of cross-agent Hessian blocks for all (i, j).
    Returns an N x N list where entry [i][j] approximates || \partial^2 v_i / (\partial obs_i \partial obs_j) ||_F.
    """
    states_tensors = [torch.tensor(state, dtype=torch.float32, requires_grad=True) for state in states]
    states_tensor = torch.cat(states_tensors, dim=0)
    values = runner.agent_n.compute_value(states_tensor).squeeze(-1)  # shape: (N,)

    N = runner.args.N
    results = [[0.0 for _ in range(N)] for _ in range(N)]

    for i in range(N):
        # Gradient wrt agent i's observation
        grad_i = torch.autograd.grad(values[i], states_tensors[i], create_graph=True, retain_graph=True)[0]

        for j in range(N):
            hessian_matrix = []
            for k in range(grad_i.shape[0]):
                second_grad = torch.autograd.grad(
                    grad_i[k],
                    states_tensors[j],
                    retain_graph=True,
                    allow_unused=True
                )[0]
                hessian_matrix.append(second_grad.flatten())

            H = torch.stack(hessian_matrix)
            results[i][j] = H.norm(p='fro').item()

    return results

# second order directional derivative
def compute_2nd_ord_dir_derivatives(runner: Runner_MAPPO_MPE, states, vulnerable_agent_id):
    states_tensors = [torch.tensor(state, dtype=torch.float32, requires_grad=True) for state in states]
    states_tensor = torch.cat(states_tensors, dim=0)
    values = runner.agent_n.compute_value(states_tensor).squeeze(-1)  # shape: (N,)

    # Store eigenvalues for each agent pair (i, j)
    results = []

    for i in range(runner.args.N):
        # Compute first-order gradient with respect to agent i's observation
        grad_i = torch.autograd.grad(values[i], states_tensors[i], create_graph=True, retain_graph=True)[0]
        v = grad_i / torch.max(grad_i.norm(p=2), torch.tensor(1e-6))

        # Compute Hessian-vector product (HVP) of grad_i and v with respect to states_tensors[j]
        hvp = torch.autograd.grad(
            outputs=grad_i,
            inputs=states_tensors[vulnerable_agent_id],
            grad_outputs=v,
            retain_graph=True,
            allow_unused=True
        )[0]

        # Compute u^T * H * v (quadratic form)
        # grad_j = torch.autograd.grad(values[i], states_tensors[vulnerable_agent_id], create_graph=True, retain_graph=True)[0]
        grad_j = torch.autograd.grad(values[i], states_tensors[vulnerable_agent_id], create_graph=True, retain_graph=True)[0]
        u = -grad_j / torch.max(grad_j.norm(p=2), torch.tensor(1e-6))
        curvature_val = torch.dot(u.flatten(), hvp.flatten())
        results.append(curvature_val.item())

    return results


def compute_taylor_error_policy(runner: Runner_MAPPO_MPE, states, epsilon):
    states_tensors = [torch.tensor(state, dtype=torch.float32, requires_grad=True) for state in states]
    states_tensor = torch.cat(states_tensors, dim=0)
    values = runner.agent_n.compute_value(states_tensor).squeeze(-1)  # shape: (N,)

    delta_errors = []

    for i in range(runner.args.N):
        obs = states_tensors[i].unsqueeze(0)  # shape: (1, obs_dim)
        action, dist = runner.agent_n.compute_action(obs, i, evaluate=True, return_dist=True)
        target_val = dist.log_prob(action)
        grad_i = torch.autograd.grad(target_val, obs, create_graph=True, retain_graph=True)[0]
        # grad_i = torch.autograd.grad(critic_val, actions[i], create_graph=True, retain_graph=True)[0]

        eta_i = epsilon * grad_i.sign() / torch.max(grad_i.norm(p=2), torch.tensor(1e-6))
        
        # Second-order Taylor expansion using Hessian-vector product (HVP)
        # Instead of computing full Hessian, compute H * eta_i directly
        # hvp = torch.autograd.grad(
        #     outputs=grad_i.flatten(), 
        #     inputs=torch_obs[i], 
        #     grad_outputs=eta_i.flatten(),
        #     retain_graph=True
        # )[0]
        
        # Second-order Taylor approximation: f(x + η) ≈ f(x) + ∇f(x)^T η + 0.5 η^T H η
        j_tilde = target_val + torch.dot(grad_i.flatten(), eta_i.flatten())# + 0.5 * torch.dot(eta_i.flatten(), hvp.flatten())
        p_state = obs + eta_i
        p_action, p_dist = runner.agent_n.compute_action(p_state, i, evaluate=True, return_dist=True)
        j_perturbed = p_dist.log_prob(p_action)
        delta_error = abs(j_perturbed - j_tilde).item()
        delta_errors.append(delta_error)

    return delta_errors

def compute_eigen_policy(runner: Runner_MAPPO_MPE, states):
    states_tensor = torch.stack([torch.tensor(state, dtype=torch.float32, requires_grad=True) for state in states])

    results = []

    for i in range(runner.args.N):
        obs = states_tensor[i].unsqueeze(0)  # shape: (1, obs_dim)
        action, dist = runner.agent_n.compute_action(obs, i, evaluate=True, return_dist=True)
        target_val = dist.log_prob(action)
        grad_i = torch.autograd.grad(target_val, obs, create_graph=True, retain_graph=True)[0]

        # Compute Hessian matrix
        hessian_flat = []
        grad_i_flat = grad_i.flatten()
        for j in range(grad_i.numel()):
            grad2 = torch.autograd.grad(
                outputs=grad_i_flat[j], 
                inputs=obs, 
                retain_graph=True,
                create_graph=False
            )[0]
            hessian_flat.append(grad2.flatten())
        
        hessian = torch.stack(hessian_flat)

        # # Compute Frobenius norm of Hessian
        # hessian_frob_norm = torch.norm(hessian, p='fro')
        # results.append(hessian_frob_norm)
        # continue
        
        # # Compute eigenvalues
        eigenvals = torch.linalg.eigvals(hessian)
        eigenval = torch.min(eigenvals.real).item() 
        results.append(eigenval)
        continue

    return results


def get_episode_data(env, runner: Runner_MAPPO_MPE, ref_vals, ref_std_devs, do_attack: bool, attacked_agent_id: str, detection_method='mean_std'):

    # Run one episode and perturb the observation of the "adversary" agent
    state = env.reset(seed=runner.seed)
    done = [False for agent_id in range(runner.args.N)]
    episode_reward = {agent_id: 0.0 for agent_id in range(runner.args.N)}

    iter_count = 0
    frames = []  # List to collect frames

    # initialize deque buffers for last batch_size observations
    result_deques = [deque(maxlen=5) for _ in range(runner.args.N)]
    frob_norms_deques = [deque(maxlen=5) for _ in range(runner.args.N)]
    sec_dir_derivatives_deques = [deque(maxlen=5) for _ in range(runner.args.N)]
    metric_vals = []
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

    prev_errors = [0 for i in range(runner.args.N)]

    while not all(done):
        # Get actions from the agent (in evaluation mode, training=False)
        actions = []
        
        for id in range(runner.args.N):
            # fgsm
            # if do_attack and id == attacked_agent_id and np.random.rand() < args.attack_rate:
            #     state[id] = perturb_fgsm(state, id, args.perturb_eps)

            action, dist = runner.agent_n.select_action(state[id], id, evaluate=True, return_dist=True)
            # action space attack
            # if do_attack and id == attacked_agent_id and np.random.rand() < args.attack_rate:
            # if do_attack and id == attacked_agent_id and dist.entropy() < 0.5:
            if do_attack and id == attacked_agent_id and dist.entropy() < 0.5 and iter_count >= 5:
                do_start_attack = True
            # if do_attack and id == attacked_agent_id and runner.args.atk_step_start <= iter_count <= runner.args.atk_step_end:
            if do_start_attack and id == attacked_agent_id and attack_step_remaining > 0:
                # # random action
                # action = env.action_space[attacked_agent_id].sample()
                # worst action attack
                action = torch.argmin(dist.probs).item()
                # print(" >> attacked")
                attacked_steps.append(iter_count)
                attack_step_remaining -= 1
            # if id > 2 and iter_count >= 3:
            #     action = torch.argmin(dist.probs).item()
            actions.append(action)

        results = compute_taylor_error_policy(runner, state, 0.01)
        results_frob_norms = compute_frob_norms(runner, state, attacked_agent_id)
        # Pairwise Frobenius norms across all agent pairs for cascading impact analysis
        pairwise_frobs = compute_pairwise_frob_norms(runner, state)
        frob_norms_matrix_history.append(pairwise_frobs)
        results_sec_dir_derivatives = compute_2nd_ord_dir_derivatives(runner, state, attacked_agent_id)
        # results = compute_eigen_policy(runner, state)
        for i in range(runner.args.N):
            result_deques[i].append(results[i])
            
            # Apply different detection methods
            if detection_method == 'mean_std':
                detection_value = np.mean(result_deques[i])
                threshold_exceeded = abs(detection_value - ref_vals[i][iter_count]) > 0.6 * ref_std_devs[i][iter_count]
            elif detection_method == 'median_mad':
                detection_value = np.mean(result_deques[i])
                threshold_exceeded = abs(detection_value - ref_vals[i][iter_count]) > 0.6 * ref_std_devs[i][iter_count]
            elif detection_method == 'diff':
                if iter_count > 0:
                    current_diff = results[i] - prev_errors[i]
                    threshold_exceeded = abs(current_diff - ref_vals[i][iter_count]) > 0.6 * ref_std_devs[i][iter_count]
                    detection_value = current_diff
                else:
                    threshold_exceeded = False
                    detection_value = 0.0
            else:
                raise ValueError(f"Unknown detection method: {detection_method}")
            
            if threshold_exceeded:
                if i not in fault_first_detected:
                    print(f" [!!!] Anomaly detected for agent {i} at timestep: {iter_count}. Method: {detection_method}. Value: {detection_value:.6f}")
                    fault_first_detected[i] = iter_count
                    # Cascading Impact Analysis
                    prev_faults = [(f, tf) for f, tf in fault_first_detected.items() if f != i and tf < iter_count]
                    contribs = {}
                    if len(prev_faults) > 0:
                        for f, tf in prev_faults:
                            # Mean Frobenius norm from t_f to current t for H_{i,f}
                            values_over_time = [frob_norms_matrix_history[tau][i][f] for tau in range(tf, iter_count + 1) if tau < len(frob_norms_matrix_history)]
                            if len(values_over_time) > 0:
                                contribs[f] = float(np.mean(values_over_time))
                        if len(contribs) > 0:
                            ranked = sorted(contribs.items(), key=lambda x: x[1], reverse=True)
                            print(f"     >> Potential contributors to fault in agent {i} (mean ||H_{{i,f}}||_F from t_f to {iter_count}): {ranked}")
                    fault_timeline.append({
                        'agent': i,
                        't': iter_count,
                        'contribs': contribs
                    })
            frob_norms_deques[i].append(results_frob_norms[i])
            sec_dir_derivatives_deques[i].append(results_sec_dir_derivatives[i])

        metric_vals.append([np.mean(result_deques[i]) for i in range(runner.args.N)])
        prev_errors = results
        frob_norms_list.append([np.mean(frob_norms_deques[i]) for i in range(runner.args.N)])
        sec_dir_derivatives.append([np.mean(sec_dir_derivatives_deques[i]) for i in range(runner.args.N)])

        next_state, reward, done, info = env.step(actions)
        
        for agent_id in range(runner.args.N):
            episode_reward[agent_id] += reward[agent_id]
        
        state = next_state
        iter_count += 1
    
    print("Episode finished. Rewards:", episode_reward, " Steps:", iter_count)
    return metric_vals, attacked_steps, frob_norms_list, sec_dir_derivatives, frob_norms_matrix_history, fault_timeline


def plot_results(results_attacked, attacked_steps, atk_agent_id, ref_vals, ref_std_devs, logdir, detection_method='mean_std'):
    n = len(results_attacked[0])  # number of agents
    t = len(results_attacked)     # number of time steps
    
    # Create n subplots in a row
    # fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    # --- Instead of: fig, axes = plt.subplots(1, n, figsize=(4*n, 4)) ---
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
        
        # Adjust reference data to match the series length
        ref_vals[i] = ref_vals[i][:steps_length]
        ref_std_devs[i] = ref_std_devs[i][:steps_length]
        
        # Add green region using ref_vals and ref_std_devs
        ref_lower = [ref_vals[i][t] - 0.6*ref_std_devs[i][t] for t in range(len(ref_vals[i]))]
        ref_upper = [ref_vals[i][t] + 0.6*ref_std_devs[i][t] for t in range(len(ref_vals[i]))]
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
    plt.savefig(os.path.join(logdir, f'plot_analysis_attacked_{atk_agent_id}.png'), dpi=300, bbox_inches='tight')
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
        steps = range(len(normal_series))
        ax.plot(steps, attacked_series, 'r-', label='Under Attack', linewidth=2)
        ax.plot(steps, normal_series, 'g-', label='Normal', linewidth=2)
        
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


def plot_sec_dir_derivatives(s_dir_derv_normal, s_dir_derv_atk, attacked_steps, atk_agent_id, ref_vals, ref_std_devs, logdir):
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
        steps = range(len(normal_series))

        # ref_lower = [ref_vals[i][t] - ref_std_devs[i][t] for t in range(len(ref_vals[i]))]
        # ref_upper = [ref_vals[i][t] + ref_std_devs[i][t] for t in range(len(ref_vals[i]))]
        ref_lower = ref_vals[i]
        ref_upper = ref_std_devs[i]
        # ax.fill_between(steps, ref_lower, ref_upper, alpha=0.1, color='green')
        
        ax.plot(steps, attacked_series, 'r-', label='Under Attack', linewidth=2)
        # ax.plot(steps, normal_series, 'g-', label='Normal', linewidth=2)
        # ax.plot(steps, ref_vals[i], 'g--', label='Normal (Mean)', linewidth=2)
        
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

from matplotlib.patches import Patch

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


def main(runner: Runner_MAPPO_MPE, env, args):
    attacked_agent_id = args.attack_agent_id
    cwd = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(cwd, 'runs', f"{args.env_id}_{'discrete' if args.discrete_action else 'continuous'}", timestamp)
    os.makedirs(logdir, exist_ok=True)
    print(f"Logging directory: {logdir}")

    # Read reference values from CSV files if provided
    ref_vals = [[] for _ in range(runner.args.N)]
    ref_std_devs = [[] for _ in range(runner.args.N)]
    ref_sdd_vals = [[] for _ in range(runner.args.N)]
    ref_sdd_std_devs = [[] for _ in range(runner.args.N)]

    for agent_id in range(runner.args.N):
        csv_filename = f"mappo_taylor_error_atk_free_agent_{agent_id}.csv"
        csv_path = os.path.join(args.ref_val_dir, csv_filename)

        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                if args.detection_method == 'mean_std':
                    # Use mean and std_dev columns
                    ref_vals[agent_id].append(float(row[2]))  # mean
                    ref_std_devs[agent_id].append(float(row[4]))  # std_dev
                elif args.detection_method == 'median_mad':
                    # Use median and MAD columns
                    ref_vals[agent_id].append(float(row[7]))  # median
                    ref_std_devs[agent_id].append(float(row[8]))  # MAD
                elif args.detection_method == 'diff':
                    # Use diff_mean and diff_std columns
                    ref_vals[agent_id].append(float(row[9]))  # diff_mean
                    ref_std_devs[agent_id].append(float(row[10]))  # diff_std
                else:
                    raise ValueError(f"Unknown detection method: {args.detection_method}")

        # Load 2nd order directional derivative reference data
        # sdd_csv_filename = f"mappo_sdd_agent_{agent_id}_vs_all_episodes_10000.csv"
        # sdd_csv_path = os.path.join(args.ref_val_dir, sdd_csv_filename)

        # with open(sdd_csv_path, 'r') as csvfile:
        #     reader = csv.reader(csvfile)
        #     header = next(reader)  # Read header to find the right columns
            
        #     # Find the column for agent_id vs atk_agent_id
        #     # mean_col_name = f"agent_{agent_id}_vs_{attacked_agent_id}_mean"
        #     # std_col_name = f"agent_{agent_id}_vs_{attacked_agent_id}_std_dev"
        #     mean_col_name = f"agent_{agent_id}_vs_{attacked_agent_id}_q1"
        #     std_col_name = f"agent_{agent_id}_vs_{attacked_agent_id}_q3"
            
        #     mean_col_idx = header.index(mean_col_name)
        #     std_col_idx = header.index(std_col_name)
            
        #     for row in reader:
        #         ref_sdd_means[agent_id].append(float(row[mean_col_idx]))
        #         ref_sdd_std_devs[agent_id].append(float(row[std_col_idx]))

    # episode_data_unattacked = get_episode_data(env, runner, False, None)
    # save_matrix_to_files(episode_data_unattacked, None, runner.args.N, logdir)

    results_normal, _, frob_norms_normal, sec_dir_derivatives_normal, _, _ = get_episode_data(env, runner, ref_vals, ref_std_devs, False, attacked_agent_id, args.detection_method)

    results_attacked, attacked_steps, frob_norms_atk, sec_dir_derivatives_atk, frob_norms_matrix_history, fault_timeline = get_episode_data(env, runner, ref_vals, ref_std_devs, True, attacked_agent_id, args.detection_method)
    save_matrix_to_files(results_attacked, attacked_steps, attacked_agent_id, runner.args.N, logdir, f'mappo_taylor_error_atk_{attacked_agent_id}.csv')
    save_matrix_to_files(frob_norms_atk, attacked_steps, attacked_agent_id, runner.args.N, logdir, f'mappo_frobenius_norms_atk_{attacked_agent_id}.csv')
    save_matrix_to_files(sec_dir_derivatives_atk, attacked_steps, attacked_agent_id, runner.args.N, logdir, f'mappo_sec_dir_derivatives_atk_{attacked_agent_id}.csv')

    plot_results(results_attacked, attacked_steps, attacked_agent_id, ref_vals, ref_std_devs, logdir, args.detection_method)
    plot_frobs(frob_norms_normal, frob_norms_atk, attacked_steps, attacked_agent_id, logdir)
    plot_sec_dir_derivatives(sec_dir_derivatives_normal, sec_dir_derivatives_atk, attacked_steps, attacked_agent_id, ref_sdd_vals, ref_sdd_std_devs, logdir)
    plot_fault_timeline(fault_timeline, runner.args.N, logdir)
    plot_contributor_barchart(fault_timeline, runner.args.N, logdir)
    env.close()


if __name__ == '__main__':    
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help="Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=25, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the rnn")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the mlp")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling. Here, we do not use it.")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=float, default=False, help="Whether to use relu, if False, we will use tanh")
    parser.add_argument("--use_rnn", type=bool, default=False, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=float, default=False, help="Whether to add agent_id. Here, we do not use it.")
    parser.add_argument("--use_value_clip", type=float, default=False, help="Whether to use value clip.")
    parser.add_argument("--attack_rate", type=float, default=0.5, help="Attack probability when attacking (0.0-1.0)")
    parser.add_argument("--perturb_eps", type=float, default=0.1, help="Perturbation epsilon value for attacks")
    parser.add_argument("--attack_agent_id", type=int, default=0, help="Whether to add agent_id. Here, we do not use it.")
    # Add output directory argument
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save all output files")
    parser.add_argument("--env_id", type=str, required=True, help="Environment ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--discrete_action", type=bool, default=True, help="Whether the action space is discrete or continuous")
    parser.add_argument("--atk_step_start", type=int, default=-math.inf, help="Attack start step")
    parser.add_argument("--atk_step_end", type=int, default=math.inf, help="Attack end step")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory from load the trained model")
    parser.add_argument("--ref_val_dir", type=str, required=True, help="Directory to fetch reference value files")
    parser.add_argument("--detection_method", type=str, default="mean_std", choices=['mean_std', 'median_mad', 'diff'], help="Detection method to use")

    args = parser.parse_args()
    env = make_env(env_name=args.env_id)
    runner = Runner_MAPPO_MPE(args, env_name=args.env_id, number=1, seed=args.seed)

    # runner.agent_n.load_model_from_directory("/deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/MAPPO_MPE/model/MAPPO_actor_env_simple_spread_number_1_seed_0_step_1215k.pth")
    # runner.agent_n.load_model_from_directory("/deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/MAPPO_MPE/runs/train_simple_spread_v3_20250729-203446/models/MAPPO_seed_0_score_-30.94.pt")
    runner.agent_n.load_model_from_directory(args.model_dir)
    main(runner, env, args)
    # runner = Runner_MAPPO_MPE(args, env_name="simple_spread_v3", number=1, seed=0)
    # runner.run()
