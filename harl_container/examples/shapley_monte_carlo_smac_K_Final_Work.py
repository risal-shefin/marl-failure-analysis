#!/usr/bin/env python3
"""
Monte Carlo Approximation of Shapley Values for Multi-Agent RL
Implementation of Algorithm 1 from the research paper
Normal scenario (no attacks) - pure Shapley value computation
"""

import argparse
import os
import csv
import json
import itertools
import random
from math import factorial
from collections import defaultdict
from datetime import datetime
import csv

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import imageio

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harl.utils.configs_tools import get_defaults_yaml_args, update_args
from harl.utils.trans_tools import _t2n
from harl.runners import RUNNER_REGISTRY
from tqdm import tqdm

import matplotlib.patches
import math
from collections import deque

GIF_FRAMES=list()
# ------------------------------- Utility Functions -------------------------------

def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
def calculate_edge_scores(runner, average_matrix=None):
    """
    Calculate edge scores (outbound influence) for each agent
    Edge score = sum of agent's influence on all other agents
    
    Args:
        runner: HARL runner instance
        average_matrix: Pre-computed average Frobenius matrix (if None, will compute it)
        num_episodes: Number of episodes to run (only used if average_matrix is None)
        seed: Random seed (only used if average_matrix is None)
    """
    # If no pre-computed matrix is provided, calculate it
    
    # Convert to numpy array for easier manipulation
    average_matrix = np.array(average_matrix)
    n_agents = runner.num_agents
    
    # Edge scores calculate করব (প্রতিটা agent এর inbound influence)
    edge_scores = {}
    for i in range(n_agents):
        # Agent i এর উপর অন্য agents এর influence এর sum (নিজের থেকে নিজের উপর influence বাদ)
        edge_scores[i] = sum(average_matrix[j][i] for j in range(n_agents) if j != i)
    print(f"Average Influence Matrix:\n{average_matrix}")
    print(f"Edge Scores (Outbound Influence): {edge_scores}")
    return edge_scores, average_matrix.tolist()

def calculate_cascade_risk_index(shapley_values, edge_scores):
    """
    Calculate Cascade Risk Index using the formula:
    CRI_i = max(0, -a_i) * I_i
    where a_i = contribution (Shapley value) and I_i = outbound influence
    """
    cri_scores = {}
    
    for agent_id in shapley_values.keys():
        contribution = shapley_values[agent_id]  # a_i
        outbound_influence = edge_scores[agent_id]  # I_i
        
        # CRI formula: max(0, -a_i) * I_i
        # যদি contribution negative হয়, তাহলে ওটা risky
        cri_scores[agent_id] = max(0, -contribution) * outbound_influence
    
    return cri_scores

def plot_edge_scores(edge_scores, save_path, title="Edge Scores (Outbound Influence)"):
    """Plot bar chart of edge scores"""
    agents = list(edge_scores.keys())
    values = list(edge_scores.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(agents)), values, alpha=0.7, color='orange')
    plt.xlabel('Agent ID')
    plt.ylabel('Edge Score (Outbound Influence)')
    plt.title(title)
    plt.xticks(range(len(agents)), [f'Agent {a}' for a in agents])
    plt.grid(True, alpha=0.3)
    
    # Value labels on bars
    for i, v in enumerate(values):
        plt.text(i, v + max(values) * 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_cri_scores(cri_scores, save_path, title="Cascade Risk Index (CRI)"):
    """Plot bar chart of CRI scores"""
    agents = list(cri_scores.keys())
    values = list(cri_scores.values())
    
    plt.figure(figsize=(10, 6))
    # Risk এর জন্য red color ব্যবহার করব
    plt.bar(range(len(agents)), values, alpha=0.7, color='red')
    plt.xlabel('Agent ID')
    plt.ylabel('Cascade Risk Index')
    plt.title(title)
    plt.xticks(range(len(agents)), [f'Agent {a}' for a in agents])
    plt.grid(True, alpha=0.3)
    
    # Value labels
    for i, v in enumerate(values):
        plt.text(i, v + max(values) * 0.01 if max(values) > 0 else 0.001, 
                f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_metrics(shapley_values, edge_scores, cri_scores, save_path):
    """Plot all three metrics in subplots"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    agents = list(shapley_values.keys())
    
    # Node scores (Shapley values)
    shapley_vals = list(shapley_values.values())
    axes[0].bar(range(len(agents)), shapley_vals, alpha=0.7, color='blue')
    axes[0].set_title('Node Scores (Shapley Values)')
    axes[0].set_xlabel('Agent ID')
    axes[0].set_ylabel('Contribution')
    axes[0].set_xticks(range(len(agents)))
    axes[0].set_xticklabels([f'Agent {a}' for a in agents])
    axes[0].grid(True, alpha=0.3)
    
    # Edge scores
    edge_vals = list(edge_scores.values())
    axes[1].bar(range(len(agents)), edge_vals, alpha=0.7, color='orange')
    axes[1].set_title('Edge Scores (Outbound Influence)')
    axes[1].set_xlabel('Agent ID')
    axes[1].set_ylabel('Influence')
    axes[1].set_xticks(range(len(agents)))
    axes[1].set_xticklabels([f'Agent {a}' for a in agents])
    axes[1].grid(True, alpha=0.3)
    
    # CRI scores
    cri_vals = list(cri_scores.values())
    axes[2].bar(range(len(agents)), cri_vals, alpha=0.7, color='red')
    axes[2].set_title('Cascade Risk Index (CRI)')
    axes[2].set_xlabel('Agent ID')
    axes[2].set_ylabel('Risk Level')
    axes[2].set_xticks(range(len(agents)))
    axes[2].set_xticklabels([f'Agent {a}' for a in agents])
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def ensure_dir(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)


def slice_avail(avail, agent_id):
    """Extract available actions for a specific agent"""
    if avail is None:
        return None
    first = avail[0]
    if first is None:
        return None
    return avail[:, agent_id]


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

            eta_i = 0.01 * grad_i.sign() / torch.max(grad_i.norm(p=2), torch.tensor(1e-6))

            
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


def compute_pairwise_frob_norms_from_attack_test(runner, eval_obs, eval_rnn_states_critic, eval_masks):
    """
    Compute Frobenius norms of cross-agent Hessian blocks for all (i, j).
    Returns an N x N matrix where entry [i][j] approximates || \partial^2 v_i / (\partial obs_i \partial obs_j) ||_F.
    This is the version from attack_test.py
    """
    eval_obs = torch.tensor(eval_obs, dtype=torch.float32)

    agent_obs_tensors = []
    n_agents = runner.num_agents
    # assume eval_obs shape (1, n_agents, obs_dim)
    for i in range(n_agents):
        agent_obs = eval_obs[0][i].clone().detach()
        agent_obs_tensor = torch.tensor(agent_obs, dtype=torch.float32, requires_grad=True)
        agent_obs_tensors.append(agent_obs_tensor)

    concatenated_obs = torch.cat(agent_obs_tensors, dim=0)
    share_obs = concatenated_obs.unsqueeze(0).unsqueeze(0)
    share_obs = share_obs.expand(1, n_agents, -1)
    # print(f"Shape of share_obs: {share_obs.shape}")

    # exit("Exiting after one episode for edge score calculation.")
    
    values, temp_rnn_state_critic = runner.critic.get_values_with_grad(
        share_obs,
        eval_rnn_states_critic,
        eval_masks,
    )
    values = values.squeeze()

    N = n_agents
    results = [[0.0 for _ in range(N)] for _ in range(N)]


    for i in range(N):
        # gradient of v_i wrt agent i obs using the individual tensor
        """
        obs tensor:
            0 agent: [agent 0, agent 1, agent 2]
            1 agent: [agent 0, agent 1, agent 2]
            2 agent: [agent 0, agent 1, agent 2]
        0 agent : values[i][0*70:(0+1)*70]
        j agent : values[i][j*70:(j+1)*70]
        """
        
        grad_i = torch.autograd.grad(values[i], agent_obs_tensors[i], create_graph=True, retain_graph=True)[0]

        for j in range(N):
            # hessian_matrix = []
            hessian_matrix = torch.autograd.grad(
                grad_i.squeeze(),
                agent_obs_tensors[j],
                grad_outputs=torch.eye(grad_i.shape[0]).to(grad_i.device),
                retain_graph=True,
                is_grads_batched=True,
                allow_unused=True,
            )[0]
            results[i][j] = torch.norm(hessian_matrix, p='fro').item()
    # Normalize each row by its sum
    for i in range(N):
        row_sum = sum(results[i])
        if row_sum > 0:
            for j in range(N):
                results[i][j] /= row_sum
    return results

def eval_frobenius_single_episode(runner, use_seed=False, seed=42,args=None):
    """
    Modified eval function from attack_test.py for calculating Frobenius norms in a single episode.
    Returns the list of pairwise Frobenius norm matrices for each timestep.
    """
    if args.env=='smac' or args.env=='smacv2':
        eval_obs, eval_share_obs, eval_available_actions = runner.eval_envs.reset()
    else:
        eval_obs, eval_share_obs, eval_available_actions = runner.eval_envs.reset(seed=seed)

    # print(f"Shape of eval_obs: {eval_obs.shape}, eval_share_obs: {eval_share_obs.shape}")
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

    frob_norms_matrix_history = []  # list of N x N pairwise frob matrices per timestep

    while True:
        # Get actions for all agents
        eval_actions_collector = []
        for agent_id in tqdm(range(runner.num_agents), desc="Eval Agent Actions"):
            # print(f"Agent {agent_id} eval_share_obs shape: {eval_share_obs[:, agent_id].shape}")
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
        
        # Calculate pairwise Frobenius norms for this timestep
        pairwise_frobs = compute_pairwise_frob_norms_from_attack_test(
            runner, eval_obs, eval_rnn_states_critic, eval_masks
        )
        frob_norms_matrix_history.append(pairwise_frobs)

        # Step the environment
        (
            eval_obs,
            eval_share_obs,
            eval_rewards,
            eval_dones,
            eval_infos,
            eval_available_actions,
        ) = runner.eval_envs.step(eval_actions)

        # Update critic states
        value, eval_rnn_states_critic = runner.critic.get_values(
            eval_share_obs,
            eval_rnn_states_critic,
            eval_masks,
        )

        eval_dones_env = np.all(eval_dones, axis=1)

        eval_rnn_states[eval_dones_env == True] = np.zeros(
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

        # Check if episode is done
        if eval_dones_env[0]:
            break

    return frob_norms_matrix_history

def calculate_average_frobenius_norms(runner, num_episodes=1, seed=42,args=None):
    """
    Calculate Frobenius norms over multiple episodes and return the average.
    
    Args:
        runner: HARL runner instance
        num_episodes: Number of episodes to run
        seed: Random seed (only used if num_episodes == 1)
        
    Returns:
        Average Frobenius norm matrix across all episodes and timesteps
    """
    all_episode_matrices = []
    
    for episode in tqdm(range(num_episodes), desc="Calculating Frobenius norms"):
        use_seed = (num_episodes == 1)
        print(f"use_seed={use_seed}, seed={seed}")
        episode_matrices = eval_frobenius_single_episode(runner, use_seed=use_seed, seed=seed,args=args)
        all_episode_matrices.extend(episode_matrices)
    
    if len(all_episode_matrices) == 0:
        print("No Frobenius norm matrices calculated!")
        return None
    
    # Calculate average across all timesteps from all episodes
    n_agents = runner.num_agents
    average_matrix = np.zeros((n_agents, n_agents))
    
    for matrix in all_episode_matrices:
        average_matrix += np.array(matrix)
    
    average_matrix /= len(all_episode_matrices)
    
    print(f"Calculated Frobenius norms over {num_episodes} episode(s), "
          f"total timesteps: {len(all_episode_matrices)}")
    
    return average_matrix.tolist()

def plot_influence_pies(frob_matrix_history, attacked_agent_id, total_agents, save_path, is_attack_scenario=True):
    """
    For each agent i, plot a pie chart showing influence (mean frob_ij across episode) of other agents on i.
    """
    if len(frob_matrix_history) == 0:
        print("No frobenius history; skipping influence pies.")
        return

    T = len(frob_matrix_history)
    N = total_agents
    
    # Compute mean across time for each (i,j)
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
        vals = mean_matrix[i, :]
        
        if vals.sum() <= 0:
            ax.text(0.5, 0.5, 'No influence data', ha='center', va='center')
            ax.axis('off')
            continue

        vals_norm = vals / vals.sum()
        colors = [agent_colors[j] for j in range(N)]
        
        if is_attack_scenario and attacked_agent_id is not None:
            colors = [agent_colors[j] if j != attacked_agent_id else 'red' for j in range(N)]
            explode = [0.1 if j == attacked_agent_id else 0 for j in range(N)]
            wedges, texts, autotexts = ax.pie(vals_norm, colors=colors, autopct='%1.1f%%', 
                                            startangle=90, explode=explode)
        else:
            wedges, texts, autotexts = ax.pie(vals_norm, colors=colors, autopct='%1.1f%%', 
                                            startangle=90)
        
        ax.set_title(f'Influence on Agent {i}')
        ax.axis('equal')

    # Remove extra axes
    for k in range(N, len(axes)):
        fig.delaxes(axes[k])

    # Create legend
    legend_labels = [f'Agent {j}' for j in range(N)]
    legend_colors = agent_colors.copy()
    
    if is_attack_scenario and attacked_agent_id is not None:
        legend_labels[attacked_agent_id] = f'Agent {attacked_agent_id} (Attacked)'
        legend_colors[attacked_agent_id] = 'red'
    
    legend_patches = [plt.matplotlib.patches.Patch(color=legend_colors[j], 
                                                  label=legend_labels[j]) for j in range(N)]
    fig.legend(handles=legend_patches, loc='lower center', ncol=min(N, 5), 
               bbox_to_anchor=(0.5, -0.05), fontsize=10)

    title = f'Inter-Agent Influence (Attacked Agent: {attacked_agent_id})' if is_attack_scenario and attacked_agent_id is not None else 'Inter-Agent Influence'
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.1, 1, 0.96])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved influence pies to {save_path}")


# def calculate_attack_reward(runner,attack_status=False,attacked_agent_id=None,seed=42,args=None):
#     """
#     Calculate reward when doing attack or not doing attack.
#     When attack_status is True, the attacked_agent_id will take the worst possible action by evaluating Q-values.
#     When attack_status is False, all agents act optimally.
#     """
#     # Reset environment with specified seed
#     set_all_seeds(seed)
#     if args.env=='smac' or args.env=='smacv2':   
#         eval_obs, eval_share_obs, eval_avail = runner.eval_envs.reset()
#     else:
#         eval_obs, eval_share_obs, eval_avail = runner.eval_envs.reset(seed=seed)

#     n_agents = runner.num_agents
#     rnn = np.zeros((1, n_agents, runner.recurrent_n, runner.rnn_hidden_size), dtype=np.float32)
#     masks = np.ones((1, n_agents, 1), dtype=np.float32)

#     total_reward = 0.0
#     step_count = 0

#     while True:
#         actions_col = []
        
#         for agent_id in range(n_agents):
#             if attack_status and agent_id == attacked_agent_id:
#                 # Attacked agent: evaluate Q-values and take worst action (argmin)
#                 if hasattr(runner.eval_envs.action_space[agent_id], 'n'):
#                     # Discrete action space - evaluate Q-values for all actions
#                     n_actions = runner.eval_envs.action_space[agent_id].n
#                     avail_slice = slice_avail(eval_avail, agent_id)
                    
#                     if avail_slice is not None and avail_slice[0] is not None:
#                         available_actions = np.where(avail_slice[0] > 0.5)[0]
#                     else:
#                         available_actions = list(range(n_actions))
                    
#                     # print(f"Available actions for attacked agent {agent_id}: {available_actions}")
#                     # print(f"Available actions slice: {avail_slice}")
#                     # Get Q-values for all available actions
#                     obs_tensor = torch.FloatTensor(eval_obs[:, agent_id])
#                     rnn_tensor = torch.FloatTensor(rnn[:, agent_id])
#                     mask_tensor = torch.FloatTensor(masks[:, agent_id])
                    
#                     # Get action logits/Q-values from the actor
#                     with torch.no_grad():
#                         # action_logits = runner.actor[agent_id].actor.act.get_logits(torch.tensor(eval_obs).to(runner.device))
#                         action_log_probs, dist_entropy, action_distribution = runner.actor[agent_id].evaluate_actions(
#                             obs_tensor.to(runner.device),
#                             rnn_tensor.to(runner.device),
#                             available_actions,
#                             mask_tensor.to(runner.device),
#                             slice_avail(eval_avail, agent_id),
#                             None
#                         )
#                         # Extract action probabilities and take argmin
#                         q_values = action_log_probs.squeeze()
                        
#                         # # Mask unavailable actions with high values (so they won't be selected as minimum)
#                         # masked_q_values = q_values.clone()
#                         # if avail_slice is not None and avail_slice[0] is not None:
#                         #     for a in range(n_actions):
#                         #         if a not in available_actions:
#                         #             masked_q_values[a] = float('inf')
                        
#                         # Take argmin to get worst action
#                         print(f"Log probabilities of actions for attacked agent {agent_id}: {q_values}")
#                         print(f"Q-values shape: {q_values.shape}, Available actions: {len(available_actions)}")

#                         # Handle case where agent is dead (only one action available)
#                         if q_values.numel() == 1 or len(available_actions) == 1:
#                             print(f"Agent {agent_id} appears to be dead or has only one action. Using index 0.")
#                             action_index = 0
#                         else:
#                             # Normal case with multiple actions
#                             if args.worst_action == 'worst':
#                                 action_index = torch.argmin(q_values).item()
#                             elif args.worst_action == '2nd_worst':
#                                 if len(available_actions) >= 2:
#                                     action_index = torch.topk(q_values, 2, largest=False).indices[1].item()
#                                 else:
#                                     print(f"Not enough actions for 2nd worst, using worst action for agent {agent_id}")
#                                     action_index = torch.argmin(q_values).item()
#                             elif args.best_action == '2nd_best':
#                                 if len(available_actions) >= 2:
#                                     action_index = torch.topk(q_values, 2, largest=True).indices[1].item()
#                                 else:
#                                     print(f"Not enough actions for 2nd best, using best action for agent {agent_id}")
#                                     action_index = torch.argmax(q_values).item()
#                             elif args.best_action == '3rd_best':
#                                 if len(available_actions) >= 3:
#                                     action_index = torch.topk(q_values, 3, largest=True).indices[2].item()
#                                 else:
#                                     print(f"Not enough actions for 3rd best, using best available action for agent {agent_id}")
#                                     action_index = torch.argmax(q_values).item()
#                             else:
#                                 # taking best action always
#                                 print(f"Warning: BEST ACTION IS TAKEN FOR ATTACKED AGENT {agent_id}")
#                                 action_index = torch.argmax(q_values).item()
                        
#                         action_array = np.array([[available_actions[action_index]]], dtype=np.int64)
#                         # print(f"Chosen worst action for attacked agent {agent_id}: {action_array}")
#                         # exit("Exiting after printing chosen worst action for debugging.")
#                         actions_col.append(action_array)
#             else:
#                 # Normal agent: act optimally using trained policy
#                 action, rnn_next = runner.actor[agent_id].act(
#                     eval_obs[:, agent_id],
#                     rnn[:, agent_id],
#                     masks[:, agent_id],
#                     slice_avail(eval_avail, agent_id),
#                     deterministic=True
#                 )
#                 rnn[:, agent_id] = _t2n(rnn_next)
#                 actions_col.append(_t2n(action))
        
#         # Transpose to get proper action format
#         actions = np.array(actions_col).transpose(1, 0, 2)
        
#         # Step environment
#         eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_avail = runner.eval_envs.step(actions)
#         total_reward += float(eval_rewards.sum())
#         step_count += 1
        
#         # Check termination conditions
#         if np.all(eval_dones):
#             break
        
#         # Update masks for done environments
#         done_env = np.all(eval_dones, axis=1)
#         rnn[done_env == True] = 0
#         masks[:] = 1.0
#         masks[done_env == True] = 0.0

#     return total_reward

# def plot_aggregated_taylor_error_barchart(mean_normal_taylor, 
#                                         zero_mean_attack_taylor.first_mean_attack, logdir, n_agents):
#     """
#     Plot aggregated Taylor error analysis results as a bar chart comparing normal vs attacked scenarios.
    
#     Args:
#         mean_normal_taylor: Mean Taylor errors per agent in normal scenario across seeds
#         std_normal_taylor: Std Taylor errors per agent in normal scenario across seeds
#         mean_attack_taylor: Mean Taylor errors matrix - for each attacked agent, the mean errors of all agents across seeds
#         std_attack_taylor: Std Taylor errors matrix - for each attacked agent, the std errors of all agents across seeds
#         logdir: Directory to save the plot
#         n_agents: Number of agents
#     """
#     # Create subplots: one for each attacked agent scenario
#     # Calculate how to distribute n_agents+1 columns across 2 rows
#     total_plots = n_agents + 1
#     cols_per_row = 3
#     rows = math.ceil(total_plots / cols_per_row)
#     fig, axes = plt.subplots(rows, cols_per_row, figsize=(5 * cols_per_row, 12))
    
#     # Flatten axes array to handle 2D subplot grid properly
#     if total_plots == 1:
#         axes = [axes]
#     else:
#         axes = axes.flatten()
    
#     # Get consistent color palette
#     agent_colors = get_agent_colors(n_agents)
    
#     # Calculate global y-axis limits for consistency across all subplots
#     all_values = list(mean_normal_taylor)
#     for attack_means in mean_attack_taylor:
#         all_values.extend(attack_means)
    
#     global_max = max(all_values) if all_values else 0.01
#     global_text_offset = 0.001
#     global_y_limit_upper = global_max * 1.1 if global_max > 0 else 0.01  # Extra space for text labels
    
#     # Plot 1: Normal scenario (all agents)
#     ax = axes[0]
#     agents = list(range(n_agents))
#     colors = [agent_colors[i] for i in range(n_agents)]
    
#     bars = ax.bar(agents, mean_normal_taylor, color=colors, alpha=0.8, edgecolor='black')
    
#     # Add error bars for standard deviation
#     # ax.errorbar(agents, mean_normal_taylor, yerr=std_normal_taylor, fmt='none', color='black', capsize=3)
#     ax.errorbar(agents, mean_normal_taylor, fmt='none', color='black', capsize=3)
    
#     # Add value labels on top of bars
#     for bar, val, std_val in zip(bars, mean_normal_taylor, std_normal_taylor):
#         height = bar.get_height()
#         ax.text(bar.get_x() + bar.get_width()/2., height + global_text_offset,
#                 f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
#     ax.set_xlabel('Agent ID')
#     ax.set_ylabel('Mean Taylor Error')
#     ax.set_title('Normal Scenario')
#     ax.grid(axis='y', alpha=0.3)
#     ax.set_xticks(agents)
#     ax.set_ylim(0, global_y_limit_upper)
    
#     # Plot 2-N+1: Attack scenarios (for each attacked agent)
#     for attacked_agent_id in range(n_agents):
#         ax = axes[attacked_agent_id + 1]
#         attack_means = mean_attack_taylor[attacked_agent_id]
#         attack_stds = std_attack_taylor[attacked_agent_id]
        
#         # Use same colors, but highlight the attacked agent in red
#         colors_attack = [agent_colors[i] if i != attacked_agent_id else 'red' for i in range(n_agents)]
        
#         bars = ax.bar(agents, attack_means, color=colors_attack, alpha=0.8, edgecolor='black')
        
#         # Add error bars for standard deviation
#         # ax.errorbar(agents, attack_means, yerr=attack_stds, fmt='none', color='black', capsize=3)
#         ax.errorbar(agents, attack_means, fmt='none', color='black', capsize=3)
        
#         # Add value labels on top of bars using global text offset
#         for bar, val, std_val in zip(bars, attack_means, attack_stds):
#             height = bar.get_height()
#             ax.text(bar.get_x() + bar.get_width()/2., height + global_text_offset,
#                     f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
#         ax.set_xlabel('Agent ID')
#         ax.set_ylabel('Mean Taylor Error')
#         ax.set_title(f'Agent {attacked_agent_id} Attacked')
#         ax.grid(axis='y', alpha=0.3)
#         ax.set_xticks(agents)
#         ax.set_ylim(0, global_y_limit_upper)  # Use global y-limit for consistency
    
#     # Hide unused subplots
#     for j in range(total_plots, len(axes)):
#         axes[j].set_visible(False)
    
#     plt.suptitle('Aggregated Taylor Error Analysis: Normal vs Attack Scenarios', fontsize=16, fontweight='bold')
#     plt.tight_layout()
    
#     # Save plot
#     barchart_path = os.path.join(logdir, 'aggregated_taylor_error_analysis_barchart.png')
#     plt.savefig(barchart_path, dpi=300, bbox_inches='tight')
#     plt.show()
#     print(f"Saved aggregated Taylor error analysis bar chart to {barchart_path}")

def calculate_attack_reward(runner,attack_status=False,attacked_agent_id=None,seed=42,args=None,calculate_taylor=False):
    """
    Calculate reward when doing attack or not doing attack.
    When attack_status is True, the attacked_agent_id will take the worst possible action by evaluating Q-values.
    When attack_status is False, all agents act optimally.
    """
    # Reset environment with specified seed
    set_all_seeds(seed)
    if args.env=='smac' or args.env=='smacv2':
        eval_obs, eval_share_obs, eval_avail = runner.eval_envs.reset()
    else:
        eval_obs, eval_share_obs, eval_avail = runner.eval_envs.reset(seed=seed)
    n_agents = runner.num_agents
    # rnn = np.zeros((1, n_agents, runner.recurrent_n, runner.rnn_hidden_size), dtype=np.float32)
    # masks = np.ones((1, n_agents, 1), dtype=np.float32)
    eval_rnn_states = np.zeros(
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

    total_reward = 0.0
    step_count = 0
    result_list = [[] for _ in range(n_agents)]
    
    while True:
        actions_col = []
        eval_rnn_states_backup = eval_rnn_states.copy()
        for agent_id in range(n_agents):
            if attack_status and agent_id == attacked_agent_id:
                # Attacked agent: evaluate Q-values and take worst action (argmin)
                if hasattr(runner.eval_envs.action_space[agent_id], 'n'):
                    # Discrete action space - evaluate Q-values for all actions
                    n_actions = runner.eval_envs.action_space[agent_id].n
                    avail_slice = slice_avail(eval_avail, agent_id)
                    
                    if avail_slice is not None and avail_slice[0] is not None:
                        available_actions = np.where(avail_slice[0] > 0.5)[0]
                    else:
                        available_actions = list(range(n_actions))
                    
                    # print(f"Available actions for attacked agent {agent_id}: {available_actions}")
                    # print(f"Available actions slice: {avail_slice}")
                    # Get Q-values for all available actions
                    obs_tensor = torch.FloatTensor(eval_obs[:, agent_id])
                    rnn_tensor = torch.FloatTensor(eval_rnn_states[:, agent_id])
                    mask_tensor = torch.FloatTensor(eval_masks[:, agent_id])
                    
                    # Get action logits/Q-values from the actor
                    with torch.no_grad():
                        # action_logits = runner.actor[agent_id].actor.act.get_logits(torch.tensor(eval_obs).to(runner.device))
                        action_log_probs, dist_entropy, action_distribution = runner.actor[agent_id].evaluate_actions(
                            obs_tensor.to(runner.device),
                            rnn_tensor.to(runner.device),
                            available_actions,
                            mask_tensor.to(runner.device),
                            slice_avail(eval_avail, agent_id),
                            None
                        )
                        # Extract action probabilities and take argmin
                        q_values = action_log_probs.squeeze()
                        
                        # # Mask unavailable actions with high values (so they won't be selected as minimum)
                        # masked_q_values = q_values.clone()
                        # if avail_slice is not None and avail_slice[0] is not None:
                        #     for a in range(n_actions):
                        #         if a not in available_actions:
                        #             masked_q_values[a] = float('inf')
                        
                        # Take argmin to get worst action
                        # print(f"Log probabilities of actions for attacked agent {agent_id}: {q_values}")
                        # print(f"Q-values shape: {q_values.shape}, Available actions: {len(available_actions)}")

                        # Handle case where agent is dead (only one action available)
                        if q_values.numel() == 1 or len(available_actions) == 1:
                            print(f"Agent {agent_id} appears to be dead or has only one action. Using index 0.")
                            action_index = 0
                        else:
                            # Normal case with multiple actions
                            if args.worst_action == 'worst':
                                action_index = torch.argmin(q_values).item()
                            elif args.worst_action == '2nd_worst':
                                if len(available_actions) >= 2:
                                    action_index = torch.topk(q_values, 2, largest=False).indices[1].item()
                                else:
                                    print(f"Not enough actions for 2nd worst, using worst action for agent {agent_id}")
                                    action_index = torch.argmin(q_values).item()
                            elif args.best_action == '2nd_best':
                                if len(available_actions) >= 2:
                                    action_index = torch.topk(q_values, 2, largest=True).indices[1].item()
                                else:
                                    print(f"Not enough actions for 2nd best, using best action for agent {agent_id}")
                                    action_index = torch.argmax(q_values).item()
                            elif args.best_action == '3rd_best':
                                if len(available_actions) >= 3:
                                    action_index = torch.topk(q_values, 3, largest=True).indices[2].item()
                                else:
                                    print(f"Not enough actions for 3rd best, using best available action for agent {agent_id}")
                                    action_index = torch.argmax(q_values).item()
                            else:
                                # taking best action always
                                print(f"Warning: BEST ACTION IS TAKEN FOR ATTACKED AGENT {agent_id}")
                                action_index = torch.argmax(q_values).item()
                        
                        action_array = np.array([[available_actions[action_index]]], dtype=np.int64)
                        # print(f"Chosen worst action for attacked agent {agent_id}: {action_array}")
                        # exit("Exiting after printing chosen worst action for debugging.")
                        actions_col.append(action_array)
            else:
                # Normal agent: act optimally using trained policy
                action, rnn_next = runner.actor[agent_id].act(
                    eval_obs[:, agent_id],
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    slice_avail(eval_avail, agent_id),
                    deterministic=True
                )
                eval_rnn_states[:, agent_id] = _t2n(rnn_next)
                actions_col.append(_t2n(action))
        
        if calculate_taylor:
            delta_errors = compute_taylor_policy(runner, eval_obs, eval_avail, eval_rnn_states_backup)
            # print(f"Taylor errors at step {step_count}: {delta_errors}")
            for i in range(runner.num_agents):
                result_list[i].append(delta_errors[i]) # episode
            
    
        # Transpose to get proper action format
        actions = np.array(actions_col).transpose(1, 0, 2)
        
        # Step environment
        eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_avail = runner.eval_envs.step(actions)
        total_reward += float(eval_rewards.sum())
        step_count += 1
        
        # Check termination conditions
        if np.all(eval_dones):
            break
        
        # Update masks for done environments
        done_env = np.all(eval_dones, axis=1)
        eval_rnn_states[done_env == True] = 0
        eval_masks[:] = 1.0
        eval_masks[done_env == True] = 0.0
    # print(f"result_list: {result_list}")
    taylor_error_episode_mean = [np.mean(result_list[j]) for j in range(runner.num_agents)]
    # print(f"Taylor error mean per agent for the episode: {taylor_error_episode_mean}")
    
    return total_reward, taylor_error_episode_mean if calculate_taylor else total_reward


# ------------------------------- Coalition Value Functions -------------------------------

def sample_coalition(agents):
    """
    Sample a random coalition from the list of agents
    Returns a frozenset representing the coalition
    """
    coalition_size = random.randint(1, len(agents))
    coalition = random.sample(agents, coalition_size)
    return frozenset(coalition)


def remove_from_list(coalition, agents):
    """
    Remove coalition members from agents list
    Equivalent to agents \ coalition
    """
    return [agent for agent in agents if agent not in coalition]


def save_gif_from_frames(frames, filepath, duration=200):
    """
    Save a list of frames as a GIF
    
    Args:
        frames: List of numpy arrays representing frames
        filepath: Path to save the GIF
        duration: Duration between frames in milliseconds
    """
    if not frames:
        print(f"Warning: No frames captured for GIF {filepath}")
        return
    
    
    # Convert frames to PIL Images
    pil_frames = []
    for frame in frames:
        # Check for batch dimension and remove it
        if len(frame.shape) == 4 and frame.shape[0] == 1:
            frame = frame.squeeze(0)  # Remove batch dimension: (1,H,W,C) -> (H,W,C)
        # Handle different frame formats
        if isinstance(frame, np.ndarray):
            if frame.dtype != np.uint8:
                # Normalize to 0-255 range if needed
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            # Handle different shapes
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # RGB image
                pil_frames.append(Image.fromarray(frame))
            elif len(frame.shape) == 3 and frame.shape[2] == 4:
                # RGBA image, convert to RGB
                rgb_frame = frame[:, :, :3]
                pil_frames.append(Image.fromarray(rgb_frame))
            elif len(frame.shape) == 2:
                # Grayscale, convert to RGB
                rgb_frame = np.stack([frame] * 3, axis=2)
                pil_frames.append(Image.fromarray(rgb_frame))
            else:
                print(f"Warning: Unsupported frame shape {frame.shape}")
                continue
    
    if pil_frames:
        # Save as GIF
        pil_frames[0].save(
            filepath,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0
        )
        print(f"GIF saved: {filepath} ({len(pil_frames)} frames)")
    else:
        print(f"Warning: No valid frames to save for {filepath}")
            
    


def rollout(runner, coalition, seed,args=None, episode_length=None, save_gif=False, gif_path=None, compute_frob=False):
    """Modified rollout to optionally compute Frobenius norms"""
    # Reset environment with specified seed
    if args.env=='smac' or args.env=='smacv2':
        eval_obs, eval_share_obs, eval_avail = runner.eval_envs.reset()
    else:
        eval_obs, eval_share_obs, eval_avail = runner.eval_envs.reset(seed=seed)
    
    n_agents = runner.num_agents
    rnn = np.zeros((1, n_agents, runner.recurrent_n, runner.rnn_hidden_size), dtype=np.float32)
    masks = np.ones((1, n_agents, 1), dtype=np.float32)
    
    total_reward = 0.0
    step_count = 0
    frames = []
    frob_history = [] if compute_frob else None
    
    while True:
        # Capture frame for GIF if requested
        if save_gif:
            try:
                frame = None
                
                # Method 1: Try standard gym/gymnasium RGB array rendering
                if hasattr(runner.eval_envs, 'render'):
                    try:
                        frame = runner.eval_envs.render(mode='rgb_array')
                    except (TypeError, AttributeError):
                        # Method 2: Try without mode (PettingZoo style)
                        try:
                            frame = runner.eval_envs.render()
                        except:
                            frame = None
                
                # Method 3: Try accessing wrapped environment
                if frame is None and hasattr(runner.eval_envs, 'env'):
                    try:
                        if hasattr(runner.eval_envs.env, 'render'):
                            try:
                                frame = runner.eval_envs.env.render(mode='rgb_array')
                            except (TypeError, AttributeError):
                                frame = runner.eval_envs.env.render()
                    except:
                        pass
                
                # Method 4: Try individual environment in vectorized wrapper
                if frame is None and hasattr(runner.eval_envs, 'envs') and len(runner.eval_envs.envs) > 0:
                    try:
                        env0 = runner.eval_envs.envs[0]
                        if hasattr(env0, 'render'):
                            try:
                                frame = env0.render(mode='rgb_array')
                            except (TypeError, AttributeError):
                                frame = env0.render()
                    except:
                        pass
                
                # Method 5: Try the base PettingZoo environment
                if frame is None:
                    try:
                        # Navigate through wrapper layers
                        base_env = runner.eval_envs
                        while hasattr(base_env, 'env') and base_env.env is not base_env:
                            base_env = base_env.env
                        
                        if hasattr(base_env, 'render'):
                            try:
                                frame = base_env.render(mode='rgb_array')
                            except (TypeError, AttributeError):
                                frame = base_env.render()
                                
                        # For PettingZoo environments, try accessing pygame surface
                        if frame is None and hasattr(base_env, 'screen'):
                            try:
                                import pygame
                                if base_env.screen is not None:
                                    frame = pygame.surfarray.array3d(base_env.screen)
                                    frame = np.transpose(frame, (1, 0, 2))  # Correct orientation
                            except ImportError:
                                pass
                            except:
                                pass
                                
                    except:
                        pass
                
                # Store frame if we got one
                if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
                    frames.append(frame.copy())
                elif step_count == 0:  # Only warn once per episode
                    print(f"Warning: Environment does not support visual rendering for GIF creation")
                    
            except Exception as e:
                if step_count == 0:  # Only warn once
                    print(f"Warning: Could not capture frame for GIF: {e}")
        
        actions_col = []
        
        for agent_id in range(n_agents):
            if agent_id in coalition:
                # Agent in coalition: act optimally using trained policy
                action, rnn_next = runner.actor[agent_id].act(
                    eval_obs[:, agent_id],
                    rnn[:, agent_id],
                    masks[:, agent_id],
                    slice_avail(eval_avail, agent_id),
                    deterministic=True
                )
                rnn[:, agent_id] = _t2n(rnn_next)
                actions_col.append(_t2n(action))
            else:
                # action_array = np.array([[1]], dtype=np.int64)
                # actions_col.append(action_array)
                # Agent not in coalition: take random action
                if hasattr(runner.eval_envs.action_space[agent_id], 'n'):
                    # Discrete action space
                    n_actions = runner.eval_envs.action_space[agent_id].n
                    avail_slice = slice_avail(eval_avail, agent_id)
                    
                    if avail_slice is not None and avail_slice[0] is not None:
                        # Use availability mask
                        available_actions = np.where(avail_slice[0] > 0.5)[0]
                        if len(available_actions) > 0:
                            random_action = np.random.choice(available_actions)
                        else:
                            random_action = np.random.randint(0, n_actions)
                    else:
                        random_action = np.random.randint(0, n_actions)
                    
                    action_array = np.array([[random_action]], dtype=np.int64)
                    actions_col.append(action_array)
                else:
                    # Continuous action space
                    action_dim = runner.eval_envs.action_space[agent_id].shape[0]
                    random_action = np.random.uniform(
                        runner.eval_envs.action_space[agent_id].low,
                        runner.eval_envs.action_space[agent_id].high,
                        size=(1, action_dim)
                    )
                    actions_col.append(random_action)
        
        # Transpose to get proper action format
        actions = np.array(actions_col).transpose(1, 0, 2)
        
        # Step environment
        eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_avail = runner.eval_envs.step(actions)
        total_reward += float(eval_rewards.sum())
        step_count += 1
        
        
        # Check termination conditions
        if np.all(eval_dones):
            break
        
        if episode_length is not None and step_count >= episode_length:
            break
        
        # Update masks for done environments
        done_env = np.all(eval_dones, axis=1)
        rnn[done_env == True] = 0
        masks[:] = 1.0
        masks[done_env == True] = 0.0
    
    # Save GIF if requested
    if save_gif and gif_path and frames:
        save_gif_from_frames(frames, gif_path)
    
    # Return both reward and Frobenius history
    return total_reward


# ------------------------------- Shapley Value Computation -------------------------------

def monte_carlo_shapley_values(runner, agents, M=1000, seed=42,args=None, save_gifs=False, gif_dir=None):
    """
    Monte Carlo approximation of Shapley values (Algorithm 1)
    
    Args:
        runner: HARL runner instance
        agents: List of agent IDs
        M: Number of coalition permutations to sample
        seed: Base random seed
        save_gifs: whether to save GIFs for some sample episodes
        gif_dir: directory to save GIF files
        
    Returns:
        Dictionary mapping agent_id -> shapley_value
    """
    # Initialize Shapley values (line 1)
    shapley_values = {agent: 0.0 for agent in agents}
    all_frob_matrices = []
    # # Set random seed for reproducible sampling
    # random.seed(seed)
    # np.random.seed(seed)
    
    # Determine which episodes to save as GIFs (save first few and some spread throughout)
    gif_episodes = set()
    if save_gifs and gif_dir:
        ensure_dir(gif_dir)
        # Save first 5 episodes and every M//10 episodes
        gif_episodes.update(range(min(5, M)))
        gif_episodes.update(range(0, M, max(1, M//10)))
    
    # Sample M coalition permutations (lines 2-13)
    for m in tqdm(range(M), desc="Monte Carlo Shapley Sampling"):
        # Line 4: Initialize marginal contributions
        marginal_contributions = {agent: 0.0 for agent in agents}
        
        # Line 5: Sample a random coalition
        
        
        # Line 6: Remove coalition members from agents list
        # scoal_no_i = remove_from_list(coal_i, agents)
        
        # Line 7: Compute reward with coalition
        current_seed = seed # + m * 1000  # Different seed for each rollout uncomment it to make different play
        
        # Determine if we should save GIF for this episode
        
        
        # r_i = rollout(runner, coal_i, current_seed, save_gif=should_save_gif, gif_path=gif_path)
        # Computing the rollout and Frobenius norms
        
        
        # Lines 8-9: Compute marginal contribution for each agent
        for agent in agents:
            
            should_save_gif = save_gifs and m in gif_episodes
            gif_path = None
            coal_i = sample_coalition(agents)
            # print(f"Sampled coalition (episode {m}): {sorted(coal_i)}")
            if should_save_gif:
                coalition_str = "_".join(map(str, sorted(coal_i))) if coal_i else "empty"
                gif_path = os.path.join(gif_dir, f"episode_{m:04d}_coalition_{coalition_str}.gif")
            # Compute reward with this agent
            coal_with_agent = frozenset(coal_i | {agent})
            # print(f"  Evaluating coalition with Agent {agent}: {sorted(coal_with_agent)}")
            r_i = rollout(runner, coal_with_agent, current_seed,args=args, 
                                       save_gif=should_save_gif, gif_path=gif_path, 
                                       compute_frob=True)
        
            # Compute reward without this agent
            coal_without_agent = frozenset(coal_with_agent - {agent}) # if agent in coal_i else coal_i
            # print(f"  Evaluating coalition without Agent {agent}: {sorted(coal_without_agent)}")
            # Don't save GIF for marginal contribution rollouts (too many)
            r_neg_i = rollout(runner, coal_without_agent, current_seed,args=args)   #+ agent + 1

            # Marginal contribution
            marginal_contributions[agent] = r_i - r_neg_i
        
        # Lines 11-12: Update Shapley values
        for agent in agents:
            shapley_values[agent] += marginal_contributions[agent]
    
    # Line 14: Average over all samples and return
    for agent in agents:
        shapley_values[agent] /= M
    
    return shapley_values


def exact_shapley_values(runner, agents, seed=42, save_gifs=False, gif_dir=None):
    """
    Compute exact Shapley values using all possible coalitions
    Only feasible for small numbers of agents
    
    Args:
        runner: HARL runner instance
        agents: List of agent IDs
        seed: Random seed
        save_gifs: whether to save GIFs for some coalitions
        gif_dir: directory to save GIF files
        
    Returns:
        Dictionary mapping agent_id -> shapley_value
    """
    n = len(agents)
    if n > 6:
        print(f"Warning: Exact computation with {n} agents may be very slow!")
    
    # Cache for coalition values to avoid recomputation
    coalition_values = {}
    coalition_counter = 0  # Counter for GIF naming
    
    # Prepare GIF directory if needed
    if save_gifs and gif_dir:
        exact_gif_dir = os.path.join(gif_dir, "exact_computation")
        ensure_dir(exact_gif_dir)
    
    def get_coalition_value(coalition_set):
        """Get value of a coalition, using cache"""
        nonlocal coalition_counter
        coalition_frozen = frozenset(coalition_set)
        if coalition_frozen not in coalition_values:
            current_seed = seed #+ hash(coalition_frozen) % 10000
            
            # Save GIF for some interesting coalitions in exact computation
            should_save_gif = (save_gifs and gif_dir and 
                             (len(coalition_set) == 0 or len(coalition_set) == len(agents) or 
                              coalition_counter % max(1, 2**len(agents) // 5) == 0))
            
            gif_path = None
            if should_save_gif:
                coalition_str = "_".join(map(str, sorted(coalition_set))) if coalition_set else "empty"
                gif_path = os.path.join(exact_gif_dir, f"coalition_{coalition_str}.gif")
            
            coalition_values[coalition_frozen] = rollout(
                runner, coalition_frozen, current_seed, 
                save_gif=should_save_gif, gif_path=gif_path
            )
            coalition_counter += 1
        return coalition_values[coalition_frozen]
    
    shapley_values = {agent: 0.0 for agent in agents}
    
    # Compute Shapley value for each agent
    for agent in agents:
        others = [a for a in agents if a != agent]
        
        # Sum over all subsets of other agents
        for subset_size in tqdm(range(len(others) + 1), desc=f"Computing Shapley for Agent {agent}"):
            for subset in itertools.combinations(others, subset_size):
                subset_set = set(subset)
                
                # Coalition without agent
                v_S = get_coalition_value(subset_set)
                
                # Coalition with agent
                v_S_with_agent = get_coalition_value(subset_set | {agent})
                
                # Marginal contribution
                marginal = v_S_with_agent - v_S
                
                # Shapley weight
                weight = factorial(subset_size) * factorial(n - subset_size - 1) / factorial(n)
                
                shapley_values[agent] += weight * marginal
    
    return shapley_values


# ------------------------------- Visualization and Output -------------------------------

def plot_shapley_values(shapley_values, save_path, title="Shapley Values"):
    """Plot bar chart of Shapley values"""
    agents = list(shapley_values.keys())
    values = list(shapley_values.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(agents)), values, alpha=0.7)
    plt.xlabel('Agent ID')
    plt.ylabel('Shapley Value')
    plt.title(title)
    plt.xticks(range(len(agents)), [f'Agent {a}' for a in agents])
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(values):
        plt.text(i, v + max(values) * 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def get_agent_colors(n_agents):
    """
    Get consistent color palette for agents across all plots.
    
    Args:
        n_agents: Number of agents
        
    Returns:
        dict: Dictionary mapping agent index to color
    """
    cmap = plt.get_cmap('tab20')
    agent_colors = {i: cmap(i % 20) for i in range(n_agents)}
    return agent_colors

def plot_frob_full_details(normal, first_attack, second_attack, third_attack,save_path,attack_type):
    colors_dict = get_agent_colors(3)
    colors = [colors_dict[i] for i in range(3)]  # Convert dict to list
    print(f"Colors used for agents: {colors_dict}")
    """
    Plot full details of Frobenius norms for Normal and Attack scenarios
    Shows Taylor expansion errors for each agent across different scenarios
    """
    total_plots = 3 + 1
    rows = math.ceil(total_plots / 3)
    fig, axes = plt.subplots(rows, 3, figsize=(5 * 3, 12))
    all_values = list(normal)
    all_values.extend(first_attack)
    all_values.extend(second_attack)
    all_values.extend(third_attack)
    global_max = max(all_values) if all_values else 0.01
    global_text_offset = 0.001
    global_y_limit_upper = global_max * 1.1 if global_max > 0 else 0.01  # Extra space for text labels
    # Flatten axes array to handle 2D subplot grid properly
    if total_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    ax = axes[0]
    agents = list(range(3))
    bars = ax.bar(agents, normal, color=colors, alpha=0.8, edgecolor='black')
    for bar, val in zip(bars, normal):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    mean_attack = [first_attack, second_attack, third_attack]
    for attacked_agent_id in range(3):
        ax = axes[attacked_agent_id + 1]
        attack_means = mean_attack[attacked_agent_id]


        # Use same colors, but highlight the attacked agent in red
        colors_attack = [colors[i] if i != attacked_agent_id else 'red' for i in range(3)]
        
        bars = ax.bar(agents, attack_means, color=colors_attack, alpha=0.8, edgecolor='black')
        
        # Add error bars for standard deviation
        # ax.errorbar(agents, attack_means, yerr=attack_stds, fmt='none', color='black', capsize=3)
        ax.errorbar(agents, attack_means, fmt='none', color='black', capsize=3)
        
        # Add value labels on top of bars using global text offset
        for bar, val in zip(bars, attack_means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        ax.set_xlabel('Agent ID')
        ax.set_ylabel('Mean Taylor Error')
        ax.set_title(f'Agent {attacked_agent_id} Attacked')
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticks(agents)
        ax.set_ylim(0, global_y_limit_upper)  # Use global y-limit for consistency
    # Hide unused subplots
    for j in range(total_plots, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Aggregated Taylor Error Analysis: Normal vs Attack Scenarios', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    barchart_path = os.path.join(save_path, attack_type)
    ensure_dir(barchart_path)
    plt.savefig(f"{barchart_path}/aggregated_taylor_error_analysis_barchart.png", dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved aggregated Taylor error analysis bar chart to {barchart_path}")

def save_shapley_to_csv(shapley_values, filepath):
    """Save Shapley values to CSV file"""
    with open(filepath, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Create header with agent columns
        agents = sorted(shapley_values.keys())
        header = [f'agent_{agent_id}' for agent_id in agents]
        writer.writerow(header)
        
        # Write shapley values in a single row
        values = [shapley_values[agent_id] for agent_id in agents]
        writer.writerow(values)


def compare_methods(mc_values, exact_values=None):
    """Compare Monte Carlo vs exact Shapley values if available"""
    if exact_values is None:
        return
    
    print("\n" + "="*50)
    print("COMPARISON: Monte Carlo vs Exact Shapley Values")
    print("="*50)
    print(f"{'Agent ID':<10} {'Monte Carlo':<15} {'Exact':<15} {'Difference':<15}")
    print("-"*55)
    
    total_diff = 0
    for agent_id in mc_values.keys():
        mc_val = mc_values[agent_id]
        exact_val = exact_values[agent_id]
        diff = abs(mc_val - exact_val)
        total_diff += diff
        print(f"{agent_id:<10} {mc_val:<15.6f} {exact_val:<15.6f} {diff:<15.6f}")
    
    print("-"*55)
    print(f"Total Absolute Difference: {total_diff:.6f}")
    print(f"Average Absolute Difference: {total_diff/len(mc_values):.6f}")


# ------------------------------- Configuration and Restore -------------------------------

def restore_model(runner, restore_dir, reward):
    """Restore trained model from checkpoint"""
    for agent_id in range(runner.num_agents):
        policy_actor_state_dict = torch.load(
            os.path.join(restore_dir, f"actor_agent{agent_id}_{reward}.pt"),
            weights_only=False
        )
        runner.actor[agent_id].actor.load_state_dict(policy_actor_state_dict)
    
    if not runner.algo_args["render"]["use_render"]:
        policy_critic_state_dict = torch.load(
            os.path.join(restore_dir, f"critic_agent_{reward}.pt"),
            weights_only=False
        )
        runner.critic.critic.load_state_dict(policy_critic_state_dict)
        
        if runner.value_normalizer is not None:
            value_normalizer_state_dict = torch.load(
                os.path.join(restore_dir, f"value_normalizer_{reward}.pt"),
                weights_only=False
            )
            runner.value_normalizer.load_state_dict(value_normalizer_state_dict)


# ------------------------------- Main Function -------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo Shapley Values for Multi-Agent RL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Algorithm and environment
    parser.add_argument("--algo", type=str, default="happo",
                        choices=["happo", "hatrpo", "haa2c", "haddpg", "hatd3", "hasac", 
                                "had3qn", "maddpg", "matd3", "mappo"])
    parser.add_argument("--env", type=str, default="pettingzoo_mpe",
                        choices=["smac", "mamujoco", "pettingzoo_mpe", "gym", "football", 
                                "dexhands", "smacv2", "lag"])
    parser.add_argument("--exp_name", type=str, default="shapley_mc")
    parser.add_argument("--dir_name", type=str, default="Simple_Spread_V3")
    parser.add_argument("--load_config", type=str, default="")
    
    # Model restoration
    parser.add_argument("--restore_dir", type=str, default="/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/happo/smac_3s_vs_3z/seed-00001-2025-08-29-08-50-05/models")
    parser.add_argument("--restore_reward", type=str, default="60.492")
    parser.add_argument("--restore_episode", type=str, default="")
    
    # Shapley computation parameters
    parser.add_argument("--M", type=int, default=1000,
                        help="Number of coalition permutations for Monte Carlo")
    parser.add_argument("--exact", action="store_true",
                        help="Also compute exact Shapley values (only for small number of agents)")
    parser.add_argument("--save_gifs", action="store_true",
                        help="Save GIFs of game episodes for visualization")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Frobenius norm calculation parameters
    parser.add_argument("--compute_frobenius", action="store_true",
                        help="Compute and plot Frobenius norms for inter-agent influence analysis")
    parser.add_argument("--frobenius_episodes", type=int, default=1,
                        help="Number of episodes to run for Frobenius norm calculation")

    parser.add_argument("--total_seeds", type=int, default=2,
                        help="Number of seeds to run for Frobenius norm calculation")

    #worst action type
    parser.add_argument("--worst_action", type=str, default="None",
                        choices=["worst", "2nd_worst", "None"],
                        help="Type of worst action to take for attacked agent")
    parser.add_argument("--best_action", type=str, default="None",
                        choices=["2nd_best", "3rd_best", "None"],
                        help="Type of best action to take for attacked agent")
    
    # Output
    parser.add_argument("--save_dir", type=str, default="shapley_results",
                        help="Directory to save results")
    parser.add_argument("--output_dir", type=str, default="output_test",
                        help="Directory to save correlation analysis results (CSV and text files)")
    
    args, unparsed_args = parser.parse_known_args()
    
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # Initialize ranking lists outside the seed loop
    reward_drop_list = []
    shapley_ranking_list = []
    edge_score_ranking_list = []
    
    # Initialize list to collect all Shapley values from each seed
    all_shapley_values = []
    
    # Initialize list to collect all edge scores from each seed
    all_edge_scores = []
    
    # Initialize accuracy counters for analysis
    shapley_index_zero_acc = 0
    shapley_index_one_acc = 0
    shapley_index_two_acc = 0
    edge_index_zero_acc = 0
    edge_index_one_acc = 0
    edge_index_two_acc = 0
    
    # List to store analysis data for CSV
    analysis_data = []
    args.output_dir = f"{args.output_dir}/{timestamp}"
    ensure_dir(args.output_dir)
    log_path = args.output_dir
    print(f"Analysis results will be saved to: {log_path}")
    final_normal_taylor_error = [[] for _ in range(3)]  # Assuming max 3 agents for normal
    final_attack_0_taylor_error = [[] for _ in range(3)]  # Assuming max 3 agents for attack 0
    final_attack_1_taylor_error = [[] for _ in range(3)]  # Assuming max 3 agents for attack 1
    final_attack_2_taylor_error = [[] for _ in range(3)]  # Assuming max 3 agents for attack 2
    # Set seeds
    for seed in range(args.total_seeds):
        args.seed = seed
        print(f"\n\n===== Running with Seed {args.seed} =====")
        set_all_seeds(args.seed)
        
        # if seed == args.total_seeds - 1:
        #     # Create output directory
        #     log_path = os.path.join(args.save_dir, args.dir_name,str(args.seed),timestamp)
        #     ensure_dir(log_path)
        
        # Parse additional config overrides
        unparsed_dict = {}
        if len(unparsed_args) >= 2:
            keys = [k[2:] for k in unparsed_args[0::2]]  # Remove '--' prefix
            values = [v for v in unparsed_args[1::2]]
            unparsed_dict = {k: v for k, v in zip(keys, values)}
        
        # Load configuration
        if args.load_config != "":
            with open(args.load_config, encoding="utf-8") as file:
                all_config = json.load(file)
            main_args = all_config["main_args"]
            algo_args = all_config["algo_args"]
            env_args = all_config["env_args"]
            main_args["exp_name"] = args.exp_name
            main_args["algo"] = args.algo or main_args["algo"]
            main_args["env"] = args.env or main_args["env"]
        else:
            algo_args, env_args = get_defaults_yaml_args(args.algo, args.env)
            main_args = {"algo": args.algo, "env": args.env, "exp_name": args.exp_name}
        
        # Set evaluation parameters
        algo_args["eval"]["n_eval_rollout_threads"] = 1
        algo_args["eval"]["eval_episodes"] = 1
        if args.env == 'smac' or args.env == 'smacv2':
            algo_args["seed"]["seed"]=args.seed
        
        update_args(unparsed_dict, algo_args, env_args)
        
        # Special handling for dexhands environment
        if main_args["env"] == "dexhands":
            try:
                import isaacgym  # noqa: F401
            except ImportError:
                print("Warning: isaacgym not available for dexhands environment")
            algo_args["eval"]["use_eval"] = False
            algo_args["train"]["episode_length"] = env_args["hands_episode_length"]
        
        # Create runner
        runner = RUNNER_REGISTRY[main_args["algo"]](main_args, algo_args, env_args)
        
        # Restore model if specified
        if args.restore_dir and args.restore_reward:
            restore_model(runner, args.restore_dir, args.restore_reward)

        runner.prep_training()
        
        
        # Get list of agents
        agents = list(range(runner.num_agents))
        
        # print("="*50)
        # print("MONTE CARLO SHAPLEY VALUES COMPUTATION")
        # print("="*50)
        # print(f"Environment: {main_args['env']}")
        # print(f"Algorithm: {main_args['algo']}")
        # print(f"Number of agents: {len(agents)}")
        # print(f"Monte Carlo samples (M): {args.M}")
        # print(f"Random seed: {args.seed}")
        # print(f"Save GIFs: {args.save_gifs}")
        # print(f"Compute Frobenius norms: {args.compute_frobenius}")
        # if args.compute_frobenius:
        #     print(f"Frobenius episodes: {args.frobenius_episodes}")
        #     print(f"Use seed for Frobenius: {args.frobenius_episodes == 1}")
        # # print(f"Results will be saved to: {log_path}")
        # print("="*50)
        
        # # Setup GIF directory if needed
        # gif_dir = None
        # if args.save_gifs:
        #     gif_dir = os.path.join(log_path, "gifs")
        #     ensure_dir(gif_dir)
        #     print(f"GIFs will be saved to: {gif_dir}")
            
        #     # Check if environment supports rendering
        #     print("Checking environment rendering capabilities...")
        #     try:
        #         # Try to enable rendering if the environment supports it
        #         if hasattr(runner.eval_envs, 'enable_rendering'):
        #             runner.eval_envs.enable_rendering()
                
        #         test_obs, test_share_obs, test_avail = runner.eval_envs.reset()
        #         # Try to get a test frame
        #         test_frame = None
        #         try:
        #             test_frame = runner.eval_envs.render(mode='rgb_array')
        #         except:
        #             try:
        #                 test_frame = runner.eval_envs.render()
        #             except:
        #                 pass
                
        #         if test_frame is not None and isinstance(test_frame, np.ndarray):
        #             print(f"✓ Environment supports rendering (frame shape: {test_frame.shape})")
        #         else:
        #             print("⚠ Environment may not support visual rendering - GIFs may be empty")
        #             print("  Note: PettingZoo MPE environments often don't have built-in visual rendering")
        #             print("  Try installing: pip install pygame")
        #     except Exception as e:
        #         print(f"⚠ Could not test rendering: {e}")
        #         print("  GIF creation will be attempted but may not work")
        # Compute Monte Carlo Shapley values
        print("\nComputing Monte Carlo Shapley values...")
        mc_shapley_values = monte_carlo_shapley_values(
            runner=runner,
            agents=agents,
            M=args.M,
            seed=args.seed,
            args=args,
            save_gifs=args.save_gifs,
            # gif_dir=gif_dir
        )
        
        print("\nMonte Carlo Shapley Values:")
        print("-" * 30)
        total_value = 0
        for agent_id, value in mc_shapley_values.items():
            print(f"Agent {agent_id}: {value:.6f}")
            total_value += value
        print("-" * 30)
        print(f"Total: {total_value:.6f}")
        
        # Rank agents by Shapley values (descending: highest to lowest contribution)
        shapley_sorted = sorted(mc_shapley_values.items(), key=lambda x: x[1], reverse=True)
        shapley_ranking = [agent_id for agent_id, _ in shapley_sorted]
        shapley_ranking_list.append(shapley_ranking)
        
        # Compute exact Shapley values if requested and feasible
        # exact_shapley_values = None
        # if args.exact:
        #     if len(agents) <= 6:
        #         print(f"\nComputing exact Shapley values...")
        #         exact_shapley_values = exact_shapley_values(
        #             runner=runner,
        #             agents=agents,
        #             seed=args.seed,
        #             save_gifs=args.save_gifs,
        #             gif_dir=gif_dir
        #         )
                
        #         print("\nExact Shapley Values:")
        #         print("-" * 30)
        #         exact_total = 0
        #         for agent_id, value in exact_shapley_values.items():
        #             print(f"Agent {agent_id}: {value:.6f}")
        #             exact_total += value
        #         print("-" * 30)
        #         print(f"Total: {exact_total:.6f}")
                
        #         # Compare methods
        #         compare_methods(mc_shapley_values, exact_shapley_values)
        #     else:
        #         print(f"\nSkipping exact computation: too many agents ({len(agents)} > 6)")
        
        # # Save results
        # print(f"\nSaving results to {log_path}...")
        
        # Save Monte Carlo results
        print(f"Mc_shapley values: {mc_shapley_values}")
        
        # Store Shapley values for this seed
        all_shapley_values.append(mc_shapley_values.copy())
        
        # Save individual seed results for reference (CSV for each seed)
        # save_shapley_to_csv(mc_shapley_values, os.path.join(log_path, f"shapley_monte_carlo_seed_{args.seed}.csv"))
        
        # Save averaged Shapley values to CSV (updated each seed)
        if len(all_shapley_values) > 0:
            # Calculate current average
            num_agents = len(all_shapley_values[0])
            current_averaged_shapley = {}
            for agent_id in range(num_agents):
                agent_values = [shapley_dict[agent_id] for shapley_dict in all_shapley_values]
                current_averaged_shapley[agent_id] = np.mean(agent_values)
            
            # Save current averaged results to CSV
            avg_shapley_csv_path = os.path.join(log_path, f"averaged_shapley_values_{timestamp}.csv")
            save_shapley_to_csv(current_averaged_shapley, avg_shapley_csv_path)
            
            # Save all individual Shapley values to CSV (updated each seed)
            all_shapley_csv_path = os.path.join(log_path, f"all_shapley_values_{timestamp}.csv")
            with open(all_shapley_csv_path, 'w', newline='') as csvfile:
                # Create header with agent columns
                fieldnames = ['seed'] + [f'agent_{i}' for i in range(num_agents)]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for seed_idx, shapley_dict in enumerate(all_shapley_values):
                    row = {'seed': seed_idx}
                    for agent_id in range(num_agents):
                        row[f'agent_{agent_id}'] = shapley_dict[agent_id]
                    writer.writerow(row)

        # Save exact results if computed
        # if exact_shapley_values is not None:
        #     save_shapley_to_csv(exact_shapley_values, os.path.join(log_path, "shapley_exact.csv"))
        #     plot_shapley_values(exact_shapley_values, os.path.join(log_path, "shapley_exact.png"),
        #                     "Exact Shapley Values")
        
        # Compute and plot Frobenius norms if requested
        frob_matrices_history = []
        average_frob_matrix = None
        
        if args.compute_frobenius:
            print(f"\nComputing Frobenius norms over {args.frobenius_episodes} episode(s)...")
            average_frob_matrix = calculate_average_frobenius_norms(
                runner, 
                num_episodes=args.frobenius_episodes, 
                seed=args.seed,
                args=args
            )
            # print("\nAverage Frobenius Norm Matrix:")
            # print(average_frob_matrix)
            # exit("Exiting after Frobenius norm calculation for debugging.")
            if average_frob_matrix is not None:
                # Convert average matrix to the format expected by plot_influence_pies
                frob_matrices_history = [average_frob_matrix]

                if seed == args.total_seeds - 1:
                    print("Generating influence pie charts...")
                    influence_pie_path = os.path.join(log_path, "influence_pies.png")
                    plot_influence_pies(
                        frob_matrices_history,
                        None,  # No attacked agent in normal Shapley computation
                    len(agents), 
                    influence_pie_path, 
                    is_attack_scenario=False
                    )
                
                # Save Frobenius matrices to CSV for analysis
                frob_csv_path = os.path.join(log_path, "frobenius_matrices.csv")
                
                # Check if file exists to determine if we need to write header
                file_exists = os.path.exists(frob_csv_path)
                
                with open(frob_csv_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    
                    # Only write header if file doesn't exist (first time)
                    if not file_exists:
                        header = ['metric'] + [f'agent_{i}_to_{j}' for i in range(len(agents)) for j in range(len(agents))]
                        writer.writerow(header)
                    
                    row = ['average_frobenius'] + [average_frob_matrix[i][j] for i in range(len(agents)) for j in range(len(agents))]
                    writer.writerow(row)
                print(f"Saved Frobenius matrices to {frob_csv_path}")
            else:
                print("Failed to compute Frobenius norms.")
        
        # Part 1 Task Implementation শুরু হবে Shapley values এর পরে
        # If Frobenius norms weren't computed above but we need them for edge scores,
        # # compute them now with 1000 episodes
        # if average_frob_matrix is None:
        #     print(f"\nComputing Frobenius norms for edge scores over 1000 episodes...")
        #     average_frob_matrix = calculate_average_frobenius_norms(
        #         runner, 
        #         num_episodes=1000, 
        #         seed=args.seed,
        #         args=args
        #     )

        # Step 1: Edge Scores (Outbound Influence) calculation
        print("\nCalculating Edge Scores (Outbound Influence)...")
        edge_scores, influence_matrix = calculate_edge_scores(
            runner, 
            average_matrix=average_frob_matrix,  # Pass the pre-computed matrix
        )

        if edge_scores is not None:
            print("\nEdge Scores (Outbound Influence):")
            print("-" * 40)
            for agent_id, score in edge_scores.items():
                print(f"Agent {agent_id}: {score:.6f}")
            
            # Store edge scores for this seed
            all_edge_scores.append(edge_scores.copy())
            
            # Rank agents by edge scores (descending: highest to lowest influence)
            edge_score_sorted = sorted(edge_scores.items(), key=lambda x: x[1], reverse=True)
            edge_score_ranking = [agent_id for agent_id, _ in edge_score_sorted]
            edge_score_ranking_list.append(edge_score_ranking)
            
            # Save edge scores to consolidated CSV (append mode)
            edge_csv_path = os.path.join(log_path, "edge_scores_all_seeds.csv")
            
            # Check if file exists to determine if we need to write header
            file_exists = os.path.exists(edge_csv_path)
            
            with open(edge_csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Only write header if file doesn't exist (first time)
                if not file_exists:
                    num_agents = len(edge_scores)
                    header = ['seed'] + [f'agent_{i}' for i in range(num_agents)]
                    writer.writerow(header)
                
                # Write edge scores for current seed
                row = [args.seed] + [edge_scores[agent_id] for agent_id in sorted(edge_scores.keys())]
                writer.writerow(row)
            
            print("Part 1 Task Completed Successfully!")
            print(f"Results saved in: {log_path}")
        else:
            print("Failed to calculate edge scores.")
        
        # # Save configuration
        # config_info = {
        #     "main_args": main_args,
        #     "computation_args": {
        #         "M": args.M,
        #         "seed": args.seed,
        #         "save_gifs": args.save_gifs,
        #         "compute_frobenius": args.compute_frobenius,
        #         "frobenius_episodes": args.frobenius_episodes,
        #         "exact_computed": exact_shapley_values is not None,
        #         "num_agents": len(agents),
        #     },
        #     "results": {
        #         "monte_carlo_shapley": {int(k): float(v) for k, v in mc_shapley_values.items()},
        #         "exact_shapley": ({int(k): float(v) for k, v in exact_shapley_values.items()} 
        #                         if exact_shapley_values is not None else None)
        #     }
        # }
        
        # with open(os.path.join(log_path, "config_and_results.json"), "w") as f:
        #     json.dump(config_info, f, indent=2)
        
        print("Done!")
        print(f"Results saved in: {log_path}")
        
        if args.save_gifs and gif_dir:
            try:
                gif_files = [f for f in os.listdir(gif_dir) if f.endswith('.gif')]
                if gif_files:
                    print(f"GIFs created: {len(gif_files)} (saved in {gif_dir})")
                    # Show a few examples
                    for gif_file in sorted(gif_files)[:3]:
                        print(f"  - {gif_file}")
                    if len(gif_files) > 3:
                        print(f"  ... and {len(gif_files) - 3} more")
                else:
                    print(f"No GIFs were created - environment may not support visual rendering")
                    print("Note: PettingZoo MPE environments often require additional setup for visual output")
            except Exception as e:
                print(f"Could not check GIF directory: {e}")

        normal_reward,normal_taylor_episode_mean = calculate_attack_reward(runner,attack_status=False,attacked_agent_id=None,seed=args.seed,args=args,calculate_taylor=True)
        print(f"Normal episode reward (no attack): {normal_reward:.6f}")
        attack_rewards = []
        
        
        attack_agent_zero_reward, attack_agent_zero_taylor_mean = calculate_attack_reward(runner,attack_status=True,attacked_agent_id=0,seed=args.seed,args=args,calculate_taylor=True)
        print(f"Episode reward with Agent 0 attacked: {attack_agent_zero_reward:.6f}")
        attack_rewards.append((0, attack_agent_zero_reward))
        
        attack_agent_one_reward, attack_agent_one_taylor_mean = calculate_attack_reward(runner,attack_status=True,attacked_agent_id=1,seed=args.seed,args=args,calculate_taylor=True)
        print(f"Episode reward with Agent 1 attacked: {attack_agent_one_reward:.6f}")
        attack_rewards.append((1, attack_agent_one_reward))
        attack_agent_two_reward, attack_agent_two_taylor_mean = calculate_attack_reward(runner,attack_status=True,attacked_agent_id=2,seed=args.seed,args=args,calculate_taylor=True)
        print(f"Episode reward with Agent 2 attacked: {attack_agent_two_reward:.6f}")
        attack_rewards.append((2, attack_agent_two_reward))

        # Sort attack_rewards by reward (ascending: worst to least) and extract agent IDs
        attack_rewards_sorted = sorted(attack_rewards, key=lambda x: x[1])
        agent_ranking = [agent_id for agent_id, _ in attack_rewards_sorted]
        reward_drop_list.append(agent_ranking)

        # Perform ranking analysis
        current_shapley_ranking = shapley_ranking_list[-1]  # Get the most recent shapley ranking
        current_edge_ranking = edge_score_ranking_list[-1]  # Get the most recent edge ranking
        current_reward_ranking = agent_ranking  # Current reward drop ranking
        print(f"Current Shapley Ranking: {current_shapley_ranking}")
        print(f"Current Reward Drop Ranking: {current_reward_ranking}")
        print(f"Normal Taylor Error per Agent: {normal_taylor_episode_mean}")
        print(f"Attack Agent 0 Taylor Error per Agent: {attack_agent_zero_taylor_mean}")
        print(f"Attack Agent 1 Taylor Error per Agent: {attack_agent_one_taylor_mean}")
        print(f"Attack Agent 2 Taylor Error per Agent: {attack_agent_two_taylor_mean}")
        # Append Taylor errors for each agent
        for i in range(3):
            final_normal_taylor_error[i].append(normal_taylor_episode_mean[i])
            final_attack_0_taylor_error[i].append(attack_agent_zero_taylor_mean[i])
            final_attack_1_taylor_error[i].append(attack_agent_one_taylor_mean[i])
            final_attack_2_taylor_error[i].append(attack_agent_two_taylor_mean[i])
        # Compare rankings position by position
        for i in range(3):  # Assuming 3 agents
            # Shapley vs Reward comparison
            if current_shapley_ranking[i] == current_reward_ranking[i]:
                if i == 0:
                    shapley_index_zero_acc += 1
                elif i == 1:
                    shapley_index_one_acc += 1
                elif i == 2:
                    shapley_index_two_acc += 1
            
            # Edge score vs Reward comparison
            if current_edge_ranking[i] == current_reward_ranking[i]:
                if i == 0:
                    edge_index_zero_acc += 1
                elif i == 1:
                    edge_index_one_acc += 1
                elif i == 2:
                    edge_index_two_acc += 1
        
        # Store analysis data for this seed
        analysis_data.append({
            'seed': args.seed,
            'shapley_list': current_shapley_ranking,
            'edge_list': current_edge_ranking,
            'reward_drop_list': current_reward_ranking,
            'shapley_index_zero_acc': shapley_index_zero_acc,
            'shapley_index_one_acc': shapley_index_one_acc,
            'shapley_index_two_acc': shapley_index_two_acc,
            'edge_index_zero_acc': edge_index_zero_acc,
            'edge_index_one_acc': edge_index_one_acc,
            'edge_index_two_acc': edge_index_two_acc
        })

        # Close runner
        # Plot reward comparison under different attack scenarios (only every 1000th seed)
        if args.seed == args.total_seeds - 1:
            attack_scenarios = ['Normal', 'Attacked Agent 0', 'Attacked Agent 1', 'Attacked Agent 2']
            reward_values = [normal_reward, attack_agent_zero_reward, attack_agent_one_reward, attack_agent_two_reward]
            
            plt.figure(figsize=(12, 8))
            plt.bar(range(len(attack_scenarios)), reward_values, alpha=0.7, 
                        color=['green', 'red', 'red', 'red'])
            plt.xlabel('Attack Scenario')
            plt.ylabel('Reward')
            if args.worst_action != "None":
                plt.title(f'Reward Under Attack vs Normal (Attacked Agent takes {args.worst_action} action)')
            elif args.best_action != "None":
                plt.title(f'Reward Under Attack vs Normal (Attacked Agent takes {args.best_action} action)')
            else:
                plt.title('Reward Under Attack vs Normal')
            plt.xticks(range(len(attack_scenarios)), attack_scenarios, rotation=45, ha='right')
            plt.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, v in enumerate(reward_values):
                plt.text(i, v + max(reward_values) * 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig(os.path.join(log_path, "reward_inception.png"), dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Reward comparison plot saved to: {os.path.join(log_path, 'reward_inception.png')}")
        else:
            print(f"Skipping plot generation for seed {args.seed} (plots saved only every 1000th seed)")
        runner.close()


    
    # Print all rankings outside the args.seed loop
    # print("\n" + "="*60)
    # print("AGENT RANKINGS SUMMARY")
    # print("="*60)
    
    # print("\nReward Drop List (Agent ranking from most vulnerable to least vulnerable):")
    # for i, ranking in enumerate(reward_drop_list):
    #     print(f"Seed {i}: {ranking}")
        
    # print("\nShapley Values Ranking (Agent ranking from highest to lowest contribution):")
    # for i, ranking in enumerate(shapley_ranking_list):
    #     print(f"Seed {i}: {ranking}")
    normal_mean = []
    attack_0_mean = []
    attack_1_mean = []
    attack_2_mean = []
    for i in range(3):
        normal_mean.append(np.mean(final_normal_taylor_error[i]))

        attack_0_mean.append(np.mean(final_attack_0_taylor_error[i]))

        attack_1_mean.append(np.mean(final_attack_1_taylor_error[i]))

        attack_2_mean.append(np.mean(final_attack_2_taylor_error[i]))

        print(f"\nAgent {i} Taylor Error Analysis:")
        print(f"  Normal scenario: {normal_mean[i]:.6f}")
        print(f"  Attack Agent 0: {attack_0_mean[i]:.6f}")
        print(f"  Attack Agent 1: {attack_1_mean[i]:.6f}")
        print(f"  Attack Agent 2: {attack_2_mean[i]:.6f}")

    plot_frob_full_details(normal_mean, attack_0_mean, attack_1_mean, attack_2_mean, save_path=args.output_dir,attack_type=args.worst_action if args.worst_action != "None" else args.best_action)
    print("\nEdge Scores Ranking (Agent ranking from highest to lowest influence):")
    for i, ranking in enumerate(edge_score_ranking_list):
        print(f"Seed {i}: {ranking}")
    
    # Calculate averaged Shapley values across all seeds
    print("\n" + "="*60)
    print("AVERAGED SHAPLEY VALUES ACROSS ALL SEEDS")
    print("="*60)
    
    if all_shapley_values:
        # Get number of agents from first Shapley values dictionary
        num_agents = len(all_shapley_values[0])
        
        # Initialize averaged Shapley values dictionary
        averaged_shapley_values = {}
        
        # Calculate average for each agent
        for agent_id in range(num_agents):
            agent_values = [shapley_dict[agent_id] for shapley_dict in all_shapley_values]
            averaged_shapley_values[agent_id] = np.mean(agent_values)
        
        # Print averaged results
        print(f"\nAveraged Shapley Values over {len(all_shapley_values)} seeds:")
        print("-" * 50)
        total_avg_value = 0
        for agent_id, avg_value in averaged_shapley_values.items():
            # Calculate standard deviation for this agent across seeds
            agent_values = [shapley_dict[agent_id] for shapley_dict in all_shapley_values]
            std_value = np.std(agent_values)
            print(f"Agent {agent_id}: {avg_value:.6f} ± {std_value:.6f}")
            total_avg_value += avg_value
        print("-" * 50)
        print(f"Total: {total_avg_value:.6f}")
        
        # Plot averaged Shapley values using existing function (only plotting here, CSV already saved in loop)
        avg_shapley_plot_path = os.path.join(args.output_dir, f"averaged_shapley_values_{timestamp}.png")
        plot_shapley_values(
            averaged_shapley_values, 
            avg_shapley_plot_path,
            f"Averaged Shapley Values (M={args.M}, Seeds={len(all_shapley_values)})"
        )
        print(f"Saved averaged Shapley values plot to: {avg_shapley_plot_path}")
        
    else:
        print("No Shapley values collected - cannot compute averages")
    
    # Calculate averaged Edge Scores across all seeds
    print("\n" + "="*60)
    print("AVERAGED EDGE SCORES ACROSS ALL SEEDS")
    print("="*60)
    
    if all_edge_scores:
        # Get number of agents from first edge scores dictionary
        num_agents = len(all_edge_scores[0])
        
        # Initialize averaged edge scores dictionary
        averaged_edge_scores = {}
        
        # Calculate average for each agent
        for agent_id in range(num_agents):
            agent_edge_scores = [edge_dict[agent_id] for edge_dict in all_edge_scores]
            averaged_edge_scores[agent_id] = np.mean(agent_edge_scores)
        
        # Print averaged results
        print(f"\nAveraged Edge Scores over {len(all_edge_scores)} seeds:")
        print("-" * 50)
        for agent_id, avg_score in averaged_edge_scores.items():
            # Calculate standard deviation for this agent across seeds
            agent_edge_scores = [edge_dict[agent_id] for edge_dict in all_edge_scores]
            std_score = np.std(agent_edge_scores)
            print(f"Agent {agent_id}: {avg_score:.6f} ± {std_score:.6f}")
        
        # Save averaged edge scores to CSV
        avg_edge_csv_path = os.path.join(args.output_dir, f"averaged_edge_scores_{timestamp}.csv")
        with open(avg_edge_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['agent_id', 'averaged_edge_score', 'std_edge_score'])
            for agent_id, avg_score in averaged_edge_scores.items():
                agent_edge_scores = [edge_dict[agent_id] for edge_dict in all_edge_scores]
                std_score = np.std(agent_edge_scores)
                writer.writerow([agent_id, avg_score, std_score])
        print(f"Saved averaged edge scores to: {avg_edge_csv_path}")
        
        # Save all individual edge scores to CSV for further analysis
        all_edge_csv_path = os.path.join(args.output_dir, f"all_edge_scores_{timestamp}.csv")
        with open(all_edge_csv_path, 'w', newline='') as csvfile:
            # Create header with agent columns
            fieldnames = ['seed'] + [f'agent_{i}_edge_score' for i in range(num_agents)]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for seed_idx, edge_dict in enumerate(all_edge_scores):
                row = {'seed': seed_idx}
                for agent_id in range(num_agents):
                    row[f'agent_{agent_id}_edge_score'] = edge_dict[agent_id]
                writer.writerow(row)
        print(f"Saved all individual edge scores to: {all_edge_csv_path}")
        
        # Plot averaged edge scores using existing function
        avg_edge_plot_path = os.path.join(args.output_dir, f"averaged_edge_scores_{timestamp}.png")
        plot_edge_scores(
            averaged_edge_scores, 
            avg_edge_plot_path,
            f"Averaged Edge Scores (Seeds={len(all_edge_scores)})"
        )
        print(f"Saved averaged edge scores plot to: {avg_edge_plot_path}")
        
    else:
        print("No edge scores collected - cannot compute averages")
    
    # Calculate final accuracy percentages
    total_seeds = args.total_seeds
    print("\n" + "="*60)
    print("RANKING CORRELATION ANALYSIS")
    print("="*60)
    print(f"\nShapley Values vs Reward Drop Correlation:")
    print(f"  Index 0 (Most important vs Most vulnerable): {shapley_index_zero_acc}/{total_seeds} = {shapley_index_zero_acc/total_seeds:.2%}")
    print(f"  Index 1 (2nd important vs 2nd vulnerable): {shapley_index_one_acc}/{total_seeds} = {shapley_index_one_acc/total_seeds:.2%}")
    print(f"  Index 2 (Least important vs Least vulnerable): {shapley_index_two_acc}/{total_seeds} = {shapley_index_two_acc/total_seeds:.2%}")
    
    print(f"\nEdge Scores vs Reward Drop Correlation:")
    print(f"  Index 0 (Highest influence vs Most vulnerable): {edge_index_zero_acc}/{total_seeds} = {edge_index_zero_acc/total_seeds:.2%}")
    print(f"  Index 1 (2nd influence vs 2nd vulnerable): {edge_index_one_acc}/{total_seeds} = {edge_index_one_acc/total_seeds:.2%}")
    print(f"  Index 2 (Lowest influence vs Least vulnerable): {edge_index_two_acc}/{total_seeds} = {edge_index_two_acc/total_seeds:.2%}")
    
    # Create output directory if it doesn't exist

    
    
    # Create correlation plots
    # Plot 1: Shapley vs Reward Drop Correlation
    shapley_correlations = [
        shapley_index_zero_acc / total_seeds,
        shapley_index_one_acc / total_seeds,
        shapley_index_two_acc / total_seeds
    ]
    
    plt.figure(figsize=(10, 6))
    ranks = ['Rank 0', 'Rank 1', 'Rank 2']
    bars = plt.bar(ranks, shapley_correlations, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Ranking Position')
    plt.ylabel('Correlation Accuracy')
    plt.title('Shapley Values vs Reward Drop Correlation')
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for bar, corr in zip(bars, shapley_correlations):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{corr:.2%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    shapley_corr_plot_path = os.path.join(args.output_dir, f"shapley_correlation_{timestamp}.png")
    plt.savefig(shapley_corr_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Edge Score vs Reward Drop Correlation
    edge_correlations = [
        edge_index_zero_acc / total_seeds,
        edge_index_one_acc / total_seeds,
        edge_index_two_acc / total_seeds
    ]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(ranks, edge_correlations, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Ranking Position')
    plt.ylabel('Correlation Accuracy')
    plt.title('Edge Scores vs Reward Drop Correlation')
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for bar, corr in zip(bars, edge_correlations):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{corr:.2%}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    edge_corr_plot_path = os.path.join(args.output_dir, f"edge_correlation_{timestamp}.png")
    plt.savefig(edge_corr_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved Shapley correlation plot to: {shapley_corr_plot_path}")
    print(f"Saved Edge correlation plot to: {edge_corr_plot_path}")
    
    # Save correlation analysis to text file
    txt_filename = f"correlation_analysis_{timestamp}.txt"
    txt_path = os.path.join(args.output_dir, txt_filename)
    
    
    with open(txt_path, 'w') as txtfile:
        txtfile.write("="*60 + "\n")
        txtfile.write("RANKING CORRELATION ANALYSIS\n")
        txtfile.write("="*60 + "\n\n")
        txtfile.write("Shapley Values vs Reward Drop Correlation:\n")
        txtfile.write(f"  Index 0 (Most important vs Most vulnerable): {shapley_index_zero_acc}/{total_seeds} = {shapley_index_zero_acc/total_seeds:.2%}\n")
        txtfile.write(f"  Index 1 (2nd important vs 2nd vulnerable): {shapley_index_one_acc}/{total_seeds} = {shapley_index_one_acc/total_seeds:.2%}\n")
        txtfile.write(f"  Index 2 (Least important vs Least vulnerable): {shapley_index_two_acc}/{total_seeds} = {shapley_index_two_acc/total_seeds:.2%}\n")
        txtfile.write(f"\nEdge Scores vs Reward Drop Correlation:\n")
        txtfile.write(f"  Index 0 (Highest influence vs Most vulnerable): {edge_index_zero_acc}/{total_seeds} = {edge_index_zero_acc/total_seeds:.2%}\n")
        txtfile.write(f"  Index 1 (2nd influence vs 2nd vulnerable): {edge_index_one_acc}/{total_seeds} = {edge_index_one_acc/total_seeds:.2%}\n")
        txtfile.write(f"  Index 2 (Lowest influence vs Least vulnerable): {edge_index_two_acc}/{total_seeds} = {edge_index_two_acc/total_seeds:.2%}\n")
    
    # Save analysis to CSV
    csv_filename = f"ranking_analysis_{timestamp}.csv"
    csv_path = os.path.join(args.output_dir, csv_filename)
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['seed', 'shapley_list', 'edge_list', 'reward_drop_list', 
                     'shapley_index_zero_acc', 'shapley_index_one_acc', 'shapley_index_two_acc',
                     'edge_index_zero_acc', 'edge_index_one_acc', 'edge_index_two_acc']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for data in analysis_data:
            writer.writerow(data)
    
    print(f"\nCorrelation analysis text file saved to: {txt_path}")
    print(f"Ranking analysis CSV file saved to: {csv_path}")


if __name__ == "__main__":
    main()
