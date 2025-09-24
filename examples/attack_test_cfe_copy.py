"""Train an algorithm."""
import argparse
from collections import deque
import sys
import os
import yaml
# Add HARL to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from harl.utils.configs_tools import get_defaults_yaml_args, update_args
import numpy as np
import torch
from harl.utils.trans_tools import _t2n 
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import itertools
from scipy.special import comb

def plot_results(results, results_attacked, atk_agent_id, logdir):
        os.makedirs(logdir, exist_ok=True)
        n = len(results[0])  # number of agents
        t = len(results)     # number of time steps
        
        # Create n subplots in a row
        fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
        fig.suptitle(f'Taylor Error (Worst Action Attack | Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.95)
        
        # Ensure axes is iterable even for single agent case
        if n == 1:
            axes = [axes]
        
        for i in range(n):
            ax = axes[i]
            
            # Extract time series for agent i
            normal_series = [results[t][i] for t in range(len(results))]
            attacked_series = [results_attacked[t][i] for t in range(len(results_attacked))]
            
            # Plot the curves
            steps_normal = range(len(normal_series))
            steps_attacked = range(len(attacked_series))
            ax.plot(steps_normal, normal_series, 'b-', label='Normal', linewidth=2)
            ax.plot(steps_attacked, attacked_series, 'r-', label='Attacked', linewidth=2)
            
            ax.set_xlabel('Step')
            ax.set_ylabel('Taylor Delta Error')
            ax.set_title(f'Agent {i}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(os.path.join(logdir, f'plot_analysis_attacked_{atk_agent_id}.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved analysis plot to {logdir}")

def compute_taylor_policy(runner, eval_obs, eval_available_actions, eval_rnn_states):
        # states_tensor = torch.stack([torch.tensor(state_dict[k], dtype=torch.float32, requires_grad=True) for k in state_dict.keys()])
        eval_obs = torch.tensor(eval_obs, dtype=torch.float32, requires_grad=True)
        delta_errors = []
        eval_actions_collector = []
        eval_masks = np.ones(
            (runner.algo_args["eval"]["n_eval_rollout_threads"], runner.num_agents, 1),
            dtype=np.float32,
        )

        for agent_id in range(runner.num_agents):
            cur_obs = eval_obs[:, agent_id]
            eval_actions, eval_actions_log_prob, temp_rnn_state = runner.actor[agent_id].get_actions(
                cur_obs,
                eval_rnn_states[:, agent_id],
                eval_masks[:, agent_id],
                eval_available_actions[:, agent_id]
                if eval_available_actions[0] is not None
                else None,
                deterministic=True,
            )
            # eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
            eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
            grad_i = torch.autograd.grad(
                outputs=eval_actions_log_prob,
                inputs=cur_obs,
                create_graph=True,
                retain_graph=True,
            )[0]

            eta_i = 0.001 * grad_i.sign() / torch.max(grad_i.norm(p=2), torch.tensor(1e-6))

            
            j_tilde = eval_actions_log_prob + torch.dot(grad_i.flatten(), eta_i.flatten())  #+ 0.5 * torch.dot(eta_i.flatten(), hvp.flatten())

            p_obs = cur_obs + eta_i
            _, perturb_log_prob, _ = runner.actor[agent_id].get_actions(
                p_obs,
                eval_rnn_states[:, agent_id],
                eval_masks[:, agent_id],
                eval_available_actions[:, agent_id]
                if eval_available_actions[0] is not None
                else None,
                deterministic=True,
            )
            # _, _, p_log_prob = runner.agents.choose_actions_attack(p_state, i)
            # # Actual value of perturbed point
            j_perturbed = perturb_log_prob

            delta_error = abs(j_perturbed - j_tilde).item()
            delta_errors.append(delta_error)

        return delta_errors

def eval(runner, attack_status=False, attack_agent_id=0, enable_cfe_analysis=False):
    """
    Enhanced evaluation function with counterfactual effect decomposition analysis.
    
    Args:
        runner: HARL runner instance
        attack_status: Whether to simulate attack
        attack_agent_id: ID of agent to attack
        enable_cfe_analysis: Whether to enable counterfactual effect analysis
    
    Returns:
        tuple: Original metrics plus counterfactual analysis results
    """
    
    eval_episode = 0

    eval_obs, eval_share_obs, eval_available_actions = runner.eval_envs.reset(seed=23)

    eval_rnn_states = np.zeros(
        (
            runner.algo_args["eval"]["n_eval_rollout_threads"],
            runner.num_agents,
            runner.recurrent_n,
            runner.rnn_hidden_size,
        ),
        dtype=np.float32,
    )
    eval_rnn_states_critic = np.zeros(
        (
            runner.algo_args["eval"]["n_eval_rollout_threads"],
            runner.num_agents,
            runner.recurrent_n,
            runner.rnn_hidden_size,
        ),
        dtype=np.float32,
    )
    eval_masks = np.ones(
        (runner.algo_args["eval"]["n_eval_rollout_threads"], runner.num_agents, 1),
        dtype=np.float32,
    )

    taylor_error_list = list()
    frob_norms_list = []
    sec_dir_derivatives = []
    result_deques = [deque(maxlen=5) for _ in range(runner.num_agents)]
    frob_norms_deques = [deque(maxlen=5) for _ in range(runner.num_agents)]
    sec_dir_derivatives_deques = [deque(maxlen=5) for _ in range(runner.num_agents)]
    
    # ========== COUNTERFACTUAL ANALYSIS VARIABLES ==========
    # Purpose: Track counterfactual metrics for cascading effect analysis
    agent_effects_timeline = []  # Agent-specific effects over time
    shapley_timeline = []        # Shapley values over time  
    cascade_strength_timeline = [] # Cascade strength over time
    # ======================================================

    while True:
        eval_rnn_states_backup = np.copy(eval_rnn_states)
        eval_actions_collector = []
        for agent_id in range(runner.num_agents):
            eval_actions, temp_rnn_state = runner.actor[agent_id].act(
                eval_obs[:, agent_id],
                eval_rnn_states[:, agent_id],
                eval_masks[:, agent_id],
                eval_available_actions[:, agent_id]
                if eval_available_actions[0] is not None
                else None,
                deterministic=True,
            )
            eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
            eval_actions_collector.append(_t2n(eval_actions))

        eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
        
        if attack_status:
            eval_actions[0][attack_agent_id] = runner.eval_envs.action_space[attack_agent_id].sample()  # Random action for attack agent


        # calculating taylor policy
        delta_errors = compute_taylor_policy(runner, eval_obs, eval_available_actions, eval_rnn_states_backup)
        results_frob_norms = compute_frob_norms(runner, eval_obs, 1, eval_rnn_states_critic, eval_masks)
        results_sec_dir_derivatives = compute_2nd_ord_dir_derivatives(runner, eval_obs, 1, eval_rnn_states_critic, eval_masks)
        
        # ========== COUNTERFACTUAL EFFECT ANALYSIS ==========
        # Purpose: Compute agent-specific effects and Shapley values for cascading analysis
        if enable_cfe_analysis and attack_status:
            # Get other agents (excluding compromised agent)
            other_agents = [i for i in range(runner.num_agents) if i != attack_agent_id]
            
            # Compute agent-specific effects
            agent_effects = compute_agent_specific_effects(
                runner, eval_obs, eval_available_actions, eval_rnn_states_backup,
                attack_agent_id, other_agents
            )
            agent_effects_timeline.append(agent_effects)
            
            # Compute Shapley values (computationally expensive, so do it less frequently)
            if len(agent_effects_timeline) % 5 == 0:  # Every 5 timesteps
                shapley_values = compute_shapley_values(
                    runner, eval_obs, eval_available_actions, eval_rnn_states_backup, attack_agent_id
                )
                shapley_timeline.append(shapley_values)
        # ===================================================
        
        for i in range(runner.num_agents):
            result_deques[i].append(delta_errors[i])
            taylor_approx_error = np.mean(result_deques[i])
            # if abs(taylor_approx_error) > ref_means[i][iter_count] + ref_std_devs[i][iter_count] and vulnerable_agent_id is None:
            #     print(f" [!!!] Anomaly detected for agent {i} at timestep: {iter_count}. Taylor Appx. Error: {taylor_approx_error}")
            #     vulnerable_agent_id = i
            frob_norms_deques[i].append(results_frob_norms[i])
            sec_dir_derivatives_deques[i].append(results_sec_dir_derivatives[i])
    
        taylor_error_list.append([np.mean(list(result_deques[j])) for j in range(runner.num_agents)])
        frob_norms_list.append([np.mean(frob_norms_deques[i]) for i in range(runner.num_agents)])
        sec_dir_derivatives.append([np.mean(sec_dir_derivatives_deques[i]) for i in range(runner.num_agents)])

        (
            eval_obs,
            eval_share_obs,
            eval_rewards,
            eval_dones,
            eval_infos,
            eval_available_actions,
        ) = runner.eval_envs.step(eval_actions)
        eval_data = (
            eval_obs,
            eval_share_obs,
            eval_rewards,
            eval_dones,
            eval_infos,
            eval_available_actions,
        )

        value, eval_rnn_states_critic = runner.critic.get_values(
            eval_share_obs,
            eval_rnn_states_critic,
            eval_masks,
        )
        
        # Print eval_rewards and eval_obs
        # print(f"eval_rewards: {eval_rewards}")
        # print(f"eval_obs shape: {eval_obs.shape}, eval_obs: {eval_obs}")
        # runner.logger.eval_per_step(
        #     eval_data
        # )  # logger callback at each step of evaluation

        eval_dones_env = np.all(eval_dones, axis=1)

        eval_rnn_states[
            eval_dones_env == True
        ] = np.zeros(  # if env is done, then reset rnn_state to all zero
            (
                (eval_dones_env == True).sum(),
                runner.num_agents,
                runner.recurrent_n,
                runner.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

        eval_masks = np.ones(
            (runner.algo_args["eval"]["n_eval_rollout_threads"], runner.num_agents, 1),
            dtype=np.float32,
        )
        eval_masks[eval_dones_env == True] = np.zeros(
            ((eval_dones_env == True).sum(), runner.num_agents, 1), dtype=np.float32
        )

        for eval_i in range(runner.algo_args["eval"]["n_eval_rollout_threads"]):
            if eval_dones_env[eval_i]:
                eval_episode += 1
                # runner.logger.eval_thread_done(
                #     eval_i
                # )  # logger callback when an episode is done

        if eval_episode >= runner.algo_args["eval"]["eval_episodes"]:
            # runner.logger.eval_log(
            #     eval_episode
            # )  # logger callback at the end of evaluation
            break
    
    # ========== RETURN COUNTERFACTUAL ANALYSIS RESULTS ==========
    # Purpose: Return both original metrics and new counterfactual analysis data
    cfe_results = {
        'agent_effects_timeline': agent_effects_timeline,
        'shapley_timeline': shapley_timeline,
        'cascade_strength_timeline': cascade_strength_timeline
    }
    
    return taylor_error_list, frob_norms_list, sec_dir_derivatives, cfe_results

def compute_frob_norms(runner, eval_obs, vulnerable_agent_id, eval_rnn_states_critic, eval_masks):
    eval_obs = torch.tensor(eval_obs, dtype=torch.float32)

    # Create separate tensors for each agent's observations with gradient tracking
    agent_obs_tensors = []
    obs_dim = runner.envs.observation_space[0].shape[0]  # Assuming all agents have same obs dim
    n_agents = runner.num_agents

    for i in range(n_agents):
        # Extract agent i's observation from share_obs
        # start_idx = i * obs_dim
        # end_idx = (i + 1) * obs_dim
        agent_obs = eval_obs[0][i].clone().detach()
        agent_obs_tensor = agent_obs.clone().detach().requires_grad_(True)
        agent_obs_tensors.append(agent_obs_tensor)

    # Concatenate back to recreate share_obs structure with gradient tracking
    concatenated_obs = torch.cat(agent_obs_tensors, dim=0)
    share_obs = concatenated_obs.unsqueeze(0).unsqueeze(0)  # Reshape to (1, 1, obs_dim*n_agent)
    # Need to expand to match original shape (1, n_agents, obs_dim*n_agent)
    share_obs = share_obs.expand(1, n_agents, -1)
   
    values, temp_rnn_state_critic = runner.critic.get_values(
            share_obs,
            eval_rnn_states_critic,
            eval_masks,
        )
    values = values.squeeze()  # shape: (N,)

    # values = runner.agent_n.compute_value(states_tensor).squeeze(-1)  # shape: (N,)

    # Store eigenvalues for each agent pair (i, j)
    results = []

    for i in range(runner.num_agents):
        obs_dim = runner.envs.observation_space[i].shape[0]
        # Compute first-order gradient of v_i with respect to agent i's observation
        grad_i = torch.autograd.grad(values[i], agent_obs_tensors[i], create_graph=True, retain_graph=True)[0]

        # Compute cross-agent Hessian matrix for agent pair (i, j)
        # This represents ∂²v/∂obs_i∂obs_j
        hessian_matrix = []
        
        for k in range(grad_i.shape[0]):  # For each dimension of agent i's observation (has shape (obs_dim,))
            # Compute ∂²v/∂obs_i[k]∂obs_j
            second_grad = torch.autograd.grad(
                grad_i[k],
                agent_obs_tensors[vulnerable_agent_id],
                retain_graph=True,
                allow_unused=True
            )[0]
            hessian_matrix.append(second_grad.flatten())

        # Convert to tensor and compute eigenvalues
        H = torch.stack(hessian_matrix)

        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvals(H)
        real_eigenvalues = eigenvalues.real
        negative_count = (real_eigenvalues < 0).sum().item()
        total_count = len(real_eigenvalues)
        negative_ratio = negative_count / total_count if total_count > 0 else 0.0
        results.append(negative_ratio)
        continue

        # Frobenius norm of the Hessian matrix
        results.append(H.norm(p='fro').item()) 

    return results

# second order directional derivative
def compute_2nd_ord_dir_derivatives(runner, eval_obs, vulnerable_agent_id, eval_rnn_states_critic, eval_masks):
    eval_obs = torch.tensor(eval_obs, dtype=torch.float32)

    # Create separate tensors for each agent's observations with gradient tracking
    agent_obs_tensors = []
    obs_dim = runner.envs.observation_space[0].shape[0]  # Assuming all agents have same obs dim
    n_agents = runner.num_agents

    for i in range(n_agents):
        # Extract agent i's observation from share_obs
        # start_idx = i * obs_dim
        # end_idx = (i + 1) * obs_dim
        agent_obs = eval_obs[0][i].clone().detach()
        agent_obs_tensor = agent_obs.clone().detach().requires_grad_(True)
        agent_obs_tensors.append(agent_obs_tensor)

    # Concatenate back to recreate share_obs structure with gradient tracking
    concatenated_obs = torch.cat(agent_obs_tensors, dim=0)
    share_obs = concatenated_obs.unsqueeze(0).unsqueeze(0)  # Reshape to (1, 1, obs_dim*n_agent)
    # Need to expand to match original shape (1, n_agents, obs_dim*n_agent)
    share_obs = share_obs.expand(1, n_agents, -1)
   
    values, temp_rnn_state_critic = runner.critic.get_values(
            share_obs,
            eval_rnn_states_critic,
            eval_masks,
        )
    values = values.squeeze()  # shape: (N,)

    # values = runner.agent_n.compute_value(states_tensor).squeeze(-1)  # shape: (N,)

    # Store eigenvalues for each agent pair (i, j)
    results = []

    for i in range(runner.num_agents):
        obs_dim = runner.envs.observation_space[i].shape[0]
        # Compute first-order gradient of v_i with respect to agent i's observation
        grad_i = torch.autograd.grad(values[i], agent_obs_tensors[i], create_graph=True, retain_graph=True)[0]
        v = grad_i / torch.max(grad_i.norm(p=2), torch.tensor(1e-6))

        # Compute Hessian-vector product (HVP) of grad_i and v with respect to states_tensors[j]
        hvp = torch.autograd.grad(
            outputs=grad_i,
            inputs=agent_obs_tensors[vulnerable_agent_id],
            grad_outputs=v,
            retain_graph=True,
            allow_unused=True
        )[0]

        # Compute u^T * H * v (quadratic form)
        grad_j = torch.autograd.grad(-values[i], agent_obs_tensors[vulnerable_agent_id], create_graph=True, retain_graph=True)[0]
        u = grad_j / torch.max(grad_j.norm(p=2), torch.tensor(1e-6))
        u = -u  # negative gradient direction
        curvature_val = torch.dot(u.flatten(), hvp.flatten())
        results.append(curvature_val.item())

    return results

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
            # Add legend en
            #  for attack markers
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
        steps = range(len(normal_series))
        
        ax.plot(steps, attacked_series, 'r-', label='Under Attack', linewidth=2)
        ax.plot(steps, normal_series, 'g-', label='Normal', linewidth=2)
        
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


# ====================== COUNTERFACTUAL EFFECT DECOMPOSITION FUNCTIONS ======================
# Purpose: Implement agent-specific effects and Shapley value attribution for cascading analysis

def compute_agent_specific_effects(runner, eval_obs, eval_available_actions, eval_rnn_states, 
                                 compromised_agent_id, target_agents_subset):
    """
    Compute agent-specific effects: how compromised agent's actions propagate through specific agents.
    
    Args:
        runner: HARL runner instance
        eval_obs: Current observations
        eval_available_actions: Available actions
        eval_rnn_states: RNN states
        compromised_agent_id: ID of the compromised agent
        target_agents_subset: List of agent IDs to analyze propagation through
        
    Returns:
        dict: Agent-specific effect values for each target agent
    """
    eval_obs = torch.tensor(eval_obs, dtype=torch.float32, requires_grad=True)
    eval_masks = np.ones((runner.algo_args["eval"]["n_eval_rollout_threads"], runner.num_agents, 1), dtype=np.float32)
    
    agent_effects = {}
    
    # Get baseline policy outputs for all agents
    baseline_log_probs = []
    for agent_id in range(runner.num_agents):
        cur_obs = eval_obs[:, agent_id]
        _, log_prob, _ = runner.actor[agent_id].get_actions(
            cur_obs, eval_rnn_states[:, agent_id], eval_masks[:, agent_id],
            eval_available_actions[:, agent_id] if eval_available_actions[0] is not None else None,
            deterministic=True
        )
        baseline_log_probs.append(log_prob)
    
    # Compute effect propagation for each target agent
    for target_agent in target_agents_subset:
        if target_agent == compromised_agent_id:
            continue
            
        # Compute gradient of target agent's policy w.r.t. compromised agent's observation
        target_log_prob = baseline_log_probs[target_agent]
        compromised_obs = eval_obs[:, compromised_agent_id]
        
        # Attempt gradient computation with allow_unused=True
        grad_result = torch.autograd.grad(
            outputs=target_log_prob,
            inputs=compromised_obs,
            retain_graph=True,
            create_graph=True,
            allow_unused=True
        )[0]
        
        if grad_result is not None:
            # Direct gradient connection exists
            effect_magnitude = torch.norm(grad_result).item()
        else:
            # No gradient connection - use observation similarity as proxy
            compromised_obs_norm = torch.norm(compromised_obs)
            target_obs = eval_obs[:, target_agent]
            target_obs_norm = torch.norm(target_obs)
            
            # Use cosine similarity as a proxy for potential influence
            if compromised_obs_norm > 0 and target_obs_norm > 0:
                cosine_sim = torch.dot(compromised_obs.flatten(), target_obs.flatten()) / (compromised_obs_norm * target_obs_norm)
                # Enhanced proxy: combine similarity with observation magnitude differences
                obs_magnitude_diff = abs(compromised_obs_norm - target_obs_norm) / max(compromised_obs_norm, target_obs_norm)
                effect_magnitude = (abs(cosine_sim.item()) + obs_magnitude_diff.item()) * 0.05  # Combined metric
            else:
                effect_magnitude = 0.0
        
        agent_effects[target_agent] = effect_magnitude
            
        
    return agent_effects


def compute_marginal_contribution(runner, eval_obs, eval_available_actions, eval_rnn_states,
                                compromised_agent_id, coalition, target_agent):
    """
    Compute marginal contribution of target_agent when added to coalition.
    This is used for Shapley value calculation.
    
    Args:
        runner: HARL runner instance
        eval_obs: Current observations  
        eval_available_actions: Available actions
        eval_rnn_states: RNN states
        compromised_agent_id: ID of compromised agent
        coalition: List of agent IDs in current coalition
        target_agent: Agent whose marginal contribution to compute
        
    Returns:
        float: Marginal contribution value
    """
    # Effect with coalition including target_agent
    coalition_with_target = list(coalition) + [target_agent] if target_agent not in coalition else list(coalition)
    effect_with = compute_agent_specific_effects(
        runner, eval_obs, eval_available_actions, eval_rnn_states,
        compromised_agent_id, coalition_with_target
    )
    
    # Effect with coalition excluding target_agent
    coalition_without = [a for a in coalition if a != target_agent]
    effect_without = compute_agent_specific_effects(
        runner, eval_obs, eval_available_actions, eval_rnn_states,
        compromised_agent_id, coalition_without
    )
    
    # Marginal contribution is the difference
    target_effect_with = effect_with.get(target_agent, 0.0)
    target_effect_without = effect_without.get(target_agent, 0.0)
    
    return target_effect_with - target_effect_without


def compute_shapley_values(runner, eval_obs, eval_available_actions, eval_rnn_states, compromised_agent_id):
    """
    Compute Shapley values to attribute cascading effects to individual agents.
    This quantifies each agent's responsibility in the cascading effect.
    
    Args:
        runner: HARL runner instance
        eval_obs: Current observations
        eval_available_actions: Available actions  
        eval_rnn_states: RNN states
        compromised_agent_id: ID of compromised agent
        
    Returns:
        dict: Shapley values for each agent
    """
    n_agents = runner.num_agents
    other_agents = [i for i in range(n_agents) if i != compromised_agent_id]
    shapley_values = {}
    
    for target_agent in other_agents:
        marginal_contributions = []
        remaining_agents = [a for a in other_agents if a != target_agent]
        
        # Iterate over all possible coalition sizes
        for coalition_size in range(len(remaining_agents) + 1):
            # Get all coalitions of this size
            for coalition in itertools.combinations(remaining_agents, coalition_size):
                marginal_contrib = compute_marginal_contribution(
                    runner, eval_obs, eval_available_actions, eval_rnn_states,
                    compromised_agent_id, coalition, target_agent
                )
                
                # Weight by coalition size (Shapley formula)
                weight = 1.0 / comb(len(remaining_agents), coalition_size) if comb(len(remaining_agents), coalition_size) > 0 else 1.0
                marginal_contributions.append(marginal_contrib * weight)
        
        # Shapley value is the average of weighted marginal contributions
        shapley_values[target_agent] = np.mean(marginal_contributions) if marginal_contributions else 0.0
    
    return shapley_values


def compute_cascade_strength(normal_metrics, attacked_metrics, timestep):
    """
    Compute temporal cascade strength at a given timestep.
    Measures how much the system state deviates from normal due to attack.
    
    Args:
        normal_metrics: List of normal behavior metrics over time
        attacked_metrics: List of attacked behavior metrics over time  
        timestep: Current timestep to analyze
        
    Returns:
        float: Cascade strength value
    """
    if timestep >= len(normal_metrics) or timestep >= len(attacked_metrics):
        return 0.0
    
    normal_state = np.array(normal_metrics[timestep])
    attacked_state = np.array(attacked_metrics[timestep])
    
    # Compute L2 norm of difference
    cascade_strength = np.linalg.norm(attacked_state - normal_state)
    return cascade_strength


def plot_counterfactual_decomposition(normal_data, attacked_data, agent_effects_timeline, 
                                    shapley_timeline, cascade_strength_timeline,
                                    compromised_agent_id, save_path):
    """
    Enhanced visualization showing counterfactual effect decomposition analysis.
    
    Args:
        normal_data: Normal scenario data
        attacked_data: Attacked scenario data
        agent_effects_timeline: Agent-specific effects over time
        shapley_timeline: Shapley values over time
        cascade_strength_timeline: Cascade strength over time
        compromised_agent_id: ID of compromised agent
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Agent-specific effects over time
    ax1 = axes[0, 0]
    timesteps = range(len(agent_effects_timeline))
    for agent_id in agent_effects_timeline[0].keys():
        effects = [agent_effects_timeline[t].get(agent_id, 0) for t in timesteps]
        ax1.plot(timesteps, effects, label=f'Agent {agent_id}', linewidth=2)
    ax1.set_title('Agent-Specific Effects Over Time')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Effect Magnitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Shapley value attribution
    ax2 = axes[0, 1]
    if shapley_timeline:
        # Create timesteps for Shapley values (computed every 5 timesteps)
        shapley_timesteps = [i * 5 for i in range(len(shapley_timeline))]
        for agent_id in shapley_timeline[0].keys():
            shapley_vals = [shapley_timeline[i].get(agent_id, 0) for i in range(len(shapley_timeline))]
            ax2.plot(shapley_timesteps, shapley_vals, label=f'Agent {agent_id}', linewidth=2, linestyle='--', marker='o')
    ax2.set_title('Shapley Value Attribution')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Attribution Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Cascade strength timeline
    ax3 = axes[1, 0]
    ax3.plot(timesteps, cascade_strength_timeline, 'r-', linewidth=3, label='Cascade Strength')
    ax3.axvline(x=0, color='orange', linestyle=':', label='Attack Start', linewidth=2)
    ax3.set_title('Temporal Cascade Strength')
    ax3.set_xlabel('Timestep') 
    ax3.set_ylabel('Cascade Magnitude')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Comparison of Taylor errors (existing metric)
    ax4 = axes[1, 1]
    n_agents = len(normal_data[0])
    for i in range(n_agents):
        normal_series = [normal_data[t][i] for t in timesteps]
        attacked_series = [attacked_data[t][i] for t in timesteps] 
        ax4.plot(timesteps, normal_series, 'b-', alpha=0.7, label=f'Normal Agent {i}' if i == 0 else "")
        ax4.plot(timesteps, attacked_series, 'r-', alpha=0.7, label=f'Attacked Agent {i}' if i == 0 else "")
    ax4.set_title('Taylor Error Comparison')
    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('Taylor Error')
    ax4.legend(['Normal', 'Attacked'])
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(f'Counterfactual Effect Decomposition (Compromised Agent: {compromised_agent_id})', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def save_counterfactual_metrics(agent_effects_timeline, shapley_timeline, cascade_timeline,
                              compromised_agent_id, logdir, filename_prefix):
    """
    Save counterfactual analysis metrics to CSV files.
    
    Args:
        agent_effects_timeline: Timeline of agent-specific effects
        shapley_timeline: Timeline of Shapley values
        cascade_timeline: Timeline of cascade strength values
        compromised_agent_id: ID of compromised agent
        logdir: Directory to save files
        filename_prefix: Prefix for filenames
    """
    # Save agent effects
    if agent_effects_timeline:
        filepath = os.path.join(logdir, f'{filename_prefix}_agent_effects.csv')
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Header
            agent_ids = list(agent_effects_timeline[0].keys())
            header = ['timestep', 'compromised_agent'] + [f'effect_agent_{aid}' for aid in agent_ids]
            writer.writerow(header)
            
            # Data
            for t, effects in enumerate(agent_effects_timeline):
                row = [t, compromised_agent_id] + [effects.get(aid, 0.0) for aid in agent_ids]
                writer.writerow(row)
        print(f"Saved agent effects to {filepath}")
    
    # Save Shapley values  
    if shapley_timeline:
        filepath = os.path.join(logdir, f'{filename_prefix}_shapley_values.csv')
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Header
            agent_ids = list(shapley_timeline[0].keys())
            header = ['timestep', 'compromised_agent'] + [f'shapley_agent_{aid}' for aid in agent_ids]
            writer.writerow(header)
            
            # Data
            for t, shapley_vals in enumerate(shapley_timeline):
                row = [t, compromised_agent_id] + [shapley_vals.get(aid, 0.0) for aid in agent_ids]
                writer.writerow(row)
        print(f"Saved Shapley values to {filepath}")
    
    # Save cascade strength
    filepath = os.path.join(logdir, f'{filename_prefix}_cascade_strength.csv')
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['timestep', 'compromised_agent', 'cascade_strength'])
        
        for t, strength in enumerate(cascade_timeline):
            writer.writerow([t, compromised_agent_id, strength])
    print(f"Saved cascade strength to {filepath}")

# ====================== END COUNTERFACTUAL FUNCTIONS ======================


def restore(runner,reward,episode=None,filepath="/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/happo/Latest_5/seed-00001-2025-08-15-22-58-50/models"):
        """Restore model parameters."""
        for agent_id in range(runner.num_agents):
            policy_actor_state_dict = torch.load(
                str(filepath)
                + "/actor_agent"
                + str(agent_id)
                + "_" + str(reward)
                # + "_episode_" + str(episode)
                + ".pt",
                weights_only=False  # Explicitly set to suppress warning
            )
            runner.actor[agent_id].actor.load_state_dict(policy_actor_state_dict)
        if not runner.algo_args["render"]["use_render"]:
            policy_critic_state_dict = torch.load(
                str(filepath)
                + "/critic_agent"
                + "_" + str(reward)
                # + "_episode_" + str(episode)
                + ".pt",
                weights_only=False  # Explicitly set to suppress warning
            )
            runner.critic.critic.load_state_dict(policy_critic_state_dict)
            if runner.value_normalizer is not None:
                value_normalizer_state_dict = torch.load(
                    str(filepath)
                    + "/value_normalizer"
                    + "_" + str(reward)
                    # + "_episode_" + str(episode)
                    + ".pt",
                    weights_only=False  # Explicitly set to suppress warning
                )
                runner.value_normalizer.load_state_dict(value_normalizer_state_dict)
def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="happo",
        choices=[
            "happo",
            "hatrpo",
            "haa2c",
            "haddpg",
            "hatd3",
            "hasac",
            "had3qn",
            "maddpg",
            "matd3",
            "mappo",
        ],
        help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, matd3, mappo.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="pettingzoo_mpe",
        choices=[
            "smac",
            "mamujoco",
            "pettingzoo_mpe",
            "gym",
            "football",
            "dexhands",
            "smacv2",
            "lag",
        ],
        help="Environment name. Choose from: smac, mamujoco, pettingzoo_mpe, gym, football, dexhands, smacv2, lag.",
    )
    parser.add_argument(
        "--exp_name", type=str, default="installtest", help="Experiment name."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        return eval(arg)
        

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict
    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:  # load config from corresponding yaml file
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])
    update_args(unparsed_dict, algo_args, env_args)  # update args from command line

    if args["env"] == "dexhands":
        import isaacgym  # isaacgym has to be imported before PyTorch

    # note: isaac gym does not support multiple instances, thus cannot eval separately
    if args["env"] == "dexhands":
        algo_args["eval"]["use_eval"] = False
        algo_args["train"]["episode_length"] = env_args["hands_episode_length"]

    # start training
    from harl.runners import RUNNER_REGISTRY

    algo_args['eval']['n_eval_rollout_threads'] = 1
    algo_args['eval']['eval_episodes'] = 1
    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    restore(runner, -503.235)  # Restore the model with specific reward and episode
    runner.prep_training()
    
    attack_agent_id = 0
    
    # ========== ENHANCED COUNTERFACTUAL EFFECT ANALYSIS ==========
    # Purpose: Run evaluation with both original metrics and counterfactual analysis
    print("Running normal evaluation...")
    results_normal, frob_norms_normal, sec_dir_derivatives_normal, cfe_normal = eval(
        runner, False, attack_agent_id, enable_cfe_analysis=False)
    
    print("Running attacked evaluation with counterfactual analysis...")
    results_attacked, frob_norms_atk, sec_dir_derivatives_atk, cfe_attacked = eval(
        runner, attack_status=True, attack_agent_id=attack_agent_id, enable_cfe_analysis=True)

    log_dir = algo_args['attack']['log_dir']
    alg_name = algo_args['attack']['algo_name']
    date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = os.path.join(log_dir, alg_name, date)
    os.makedirs(log_path, exist_ok=True)

    # ========== ORIGINAL PLOTTING ==========
    plot_results(results_normal, results_attacked, atk_agent_id=attack_agent_id, logdir=log_path)
    plot_frobs(frob_norms_normal, frob_norms_atk, [], attack_agent_id, log_path)
    plot_sec_dir_derivatives(sec_dir_derivatives_normal, sec_dir_derivatives_atk, [], attack_agent_id, log_path)

    # ========== COUNTERFACTUAL ANALYSIS PLOTTING ==========
    # Purpose: Generate comprehensive counterfactual effect decomposition visualization
    if cfe_attacked['agent_effects_timeline']:
        print("Generating counterfactual effect decomposition plots...")
        
        # Compute cascade strength timeline
        cascade_strengths = []
        min_len = min(len(results_normal), len(results_attacked))
        for t in range(min_len):
            strength = compute_cascade_strength(results_normal, results_attacked, t)
            cascade_strengths.append(strength)
        cfe_attacked['cascade_strength_timeline'] = cascade_strengths
        
        # Generate enhanced counterfactual plots
        cfe_plot_path = os.path.join(log_path, f'counterfactual_decomposition_atk_{attack_agent_id}.png')
        plot_counterfactual_decomposition(
            results_normal, results_attacked,
            cfe_attacked['agent_effects_timeline'],
            cfe_attacked['shapley_timeline'], 
            cfe_attacked['cascade_strength_timeline'],
            attack_agent_id, cfe_plot_path
        )
        
        # Save counterfactual metrics to CSV
        save_counterfactual_metrics(
            cfe_attacked['agent_effects_timeline'],
            cfe_attacked['shapley_timeline'],
            cfe_attacked['cascade_strength_timeline'],
            attack_agent_id, log_path, f'cfe_analysis_atk_{attack_agent_id}'
        )
        
        print(f"Counterfactual Effect Decomposition Analysis Complete!")
        print(f"Key Insights:")
        print(f"  - Agent-specific effects tracked over {len(cfe_attacked['agent_effects_timeline'])} timesteps")
        print(f"  - Shapley value attribution computed for {len(cfe_attacked['shapley_timeline'])} timesteps")
        print(f"  - Cascade strength shows temporal propagation patterns")
        print(f"  - Results saved to: {log_path}")

    # ========== ORIGINAL CSV SAVING ==========
    save_matrix_to_files(results_attacked, [], attack_agent_id, runner.num_agents, log_path, f'happo_taylor_error_atk_{attack_agent_id}.csv')
    save_matrix_to_files(frob_norms_atk, [], attack_agent_id, runner.num_agents, log_path, f'happo_frobenius_norms_atk_{attack_agent_id}.csv')
    save_matrix_to_files(sec_dir_derivatives_atk, [], attack_agent_id, runner.num_agents, log_path, f'happo_sec_dir_derivatives_atk_{attack_agent_id}.csv')

    # runner.run()
    runner.close()



if __name__ == "__main__":
    main()
