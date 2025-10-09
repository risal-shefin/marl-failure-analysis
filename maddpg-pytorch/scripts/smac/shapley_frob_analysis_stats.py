#!/usr/bin/env python3
"""
Multi-Seed Integrated Shapley Values, Frobenius Norm Analysis, Taylor Error Analysis, and Attack Analysis for Multi-Agent RL (SMAC Version)

This script runs comprehensive agent influence and risk assessment across multiple seeds and computes
aggregated statistics with matching accuracy analysis. It includes:
1. Monte Carlo Shapley values computation (100 episodes per seed)
2. Pairwise Frobenius norm analysis (1 episode per seed)
3. Outbound influence score I_i^out for each agent
4. Cascade Risk Index (CRI) for each agent
5. Attack vs No-Attack analysis (episodic rewards under attack scenarios)
6. Taylor Error Analysis comparing normal vs attacked scenarios for each agent
7. Agent ranking and matching accuracy computation
8. Aggregated visualization across multiple seeds

Features:
- Multi-seed analysis with configurable iteration count
- Agent ranking by Shapley values, outbound influence, and attack impact
- Matching accuracy computation between different ranking methods
- Mean and standard deviation computation across seeds
- Aggregated visualizations with error bars
- Cascade Risk Index for each agent
- Taylor Error Analysis with barchart visualization comparing normal and attacked scenarios
- Consistent color scheme across all plots using get_agent_colors function
- SMAC environment support

New Multi-Seed Analysis Features:
- Command line argument for max iterations instead of seed
- Ranking computation: agents sorted by Shapley values, outbound influence, and reward drop
- Matching accuracy: position-wise comparison between rankings
- Aggregated statistics: mean ± std for all metrics
- Comprehensive multi-seed visualizations

Taylor Error Analysis Features:
- Computes Taylor error approximations for policy perturbations during normal episodes
- Analyzes Taylor errors when specific agents are under attack (worst action attack)
- Provides comparative barchart visualization showing normal vs attacked scenarios
- Shows impact of attacks on Taylor error approximations across all agents
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from datetime import datetime
from torch.autograd import Variable
from tqdm import tqdm
import random
from collections import defaultdict
import json
import imageio
from PIL import Image
import math

from algorithms.maddpg import MADDPG
from utils.smac_wrapper import SmacWrapper

USE_CUDA = torch.cuda.is_available()
DEVICE = 'gpu' if USE_CUDA else 'cpu'
torch_device = torch.device("cuda" if USE_CUDA else "cpu")

# Fixed episode counts
SHAPLEY_EPISODES = 20
FROBENIUS_EPISODES = 1


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


def create_environment(config):
    """Create a fresh SMAC environment instance"""
    env = SmacWrapper.make_env(config.map_name)
    return env


def sample_coalition(agent_i, other_agents):
    """
    Sample a random coalition that includes agent_i but excludes agent_i from the sampling process.
    This is used to create coal_i (coalition including agent_i) and coal_no_i (coalition without agent_i).
    
    Args:
        agent_i: The agent index we're computing Shapley value for
        other_agents: List of other agent indices (excluding agent_i)
        
    Returns:
        tuple: (coalition_with_i, coalition_without_i) as boolean masks
    """
    n_other_agents = len(other_agents)
    
    # Randomly choose which other agents to include in coalition
    coalition_mask_others = [random.choice([True, False]) for _ in range(n_other_agents)]
    
    n_total_agents = len(other_agents) + 1  # +1 for agent_i
    
    # Coalition without agent i
    coalition_without_i = [False] * n_total_agents
    for j, include in enumerate(coalition_mask_others):
        coalition_without_i[other_agents[j]] = include
    
    # Coalition with agent i
    coalition_with_i = coalition_without_i.copy()
    coalition_with_i[agent_i] = True
    
    return coalition_with_i, coalition_without_i


def rollout_coalition(env, maddpg, coalition_mask):
    """
    Run a single episode rollout with the given coalition.
    Agents in the coalition use their learned policy, others use default action (0).
    
    Args:
        env: Environment instance
        maddpg: MADDPG model
        coalition_mask: Boolean list indicating which agents are in the coalition
        
    Returns:
        float: Total episode reward (shared payout)
    """
    obs, action_masks = env.reset()
    total_reward = 0.0
    
    while True:
        # Get actions for all agents
        torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=False) 
                    for i in range(maddpg.nagents)]
        torch_masks = [Variable(torch.Tensor(action_masks[i]).to(torch_device), requires_grad=False)
                       if action_masks[i] is not None else None for i in range(maddpg.nagents)]
        
        # Get policy actions
        torch_agent_actions = maddpg.step(torch_obs, explore=False, action_masks=torch_masks)
        agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions]
        
        # Create action dict based on coalition membership
        actions_dict = {}
        for i, agent_name in enumerate(env.possible_agents):
            if coalition_mask[i]:
                # Agent in coalition: use learned policy
                action = agent_actions[i].argmax()  # SMAC uses discrete actions
            else:
                # Agent not in coalition: check if no-op (0) is available, otherwise use stop (1)
                if action_masks[i] is not None:
                   action = 0 if action_masks[i][0] else 1
                else:
                    action = 0  # Default fallback
            
            actions_dict[agent_name] = action
        
        # Step environment
        obs, rewards, dones, infos, action_masks = env.step(actions_dict)
        
        # Compute shared reward (sum of all agent rewards)
        step_reward = np.sum(rewards)
        total_reward += step_reward
        
        # Check if episode is done
        if dones.all():
            break
    
    return total_reward


def monte_carlo_shapley(config, maddpg, logdir, env):
    """
    Compute Shapley values using Monte Carlo approximation.
    
    Args:
        config: Configuration object
        maddpg: MADDPG model
        logdir: Log directory for saving files
        
    Returns:
        tuple: (final_shapley_values, running_means_history)
    """
    n_agents = maddpg.nagents
    M = SHAPLEY_EPISODES
    
    # Initialize storage for marginal contributions
    marginal_contributions = [[] for _ in range(n_agents)]
    
    # Storage for running means to track convergence
    running_means_history = [[] for _ in range(n_agents)]
    
    print(f"Computing Shapley values using Monte Carlo approximation with {M} iterations...")
    print(f"Number of agents: {n_agents}")
    
    # Monte Carlo iterations
    for m in tqdm(range(M), desc="Monte Carlo iterations"):
        # For each agent i
        for agent_i in range(n_agents):
            # Get other agents (all except agent_i)
            other_agents = [j for j in range(n_agents) if j != agent_i]

            # Sample a random coalition
            coalition_with_i, coalition_without_i = sample_coalition(agent_i, other_agents)
            
            # Rollout with coalition including agent i
            r_plus_i = rollout_coalition(env, maddpg, coalition_with_i)
            
            # Rollout with coalition excluding agent i
            r_minus_i = rollout_coalition(env, maddpg, coalition_without_i)
            
            # Compute marginal contribution
            marginal_contrib = r_plus_i - r_minus_i
            marginal_contributions[agent_i].append(marginal_contrib)

        # Update running means after each iteration
        for agent_i in range(n_agents):
            if len(marginal_contributions[agent_i]) > 0:
                running_mean = np.mean(marginal_contributions[agent_i])
                running_means_history[agent_i].append(running_mean)
            else:
                # If no contributions yet, append 0
                running_means_history[agent_i].append(0.0)
    
    # Compute final Shapley values
    final_shapley_values = []
    for agent_i in range(n_agents):
        if len(marginal_contributions[agent_i]) > 0:
            shapley_value = np.mean(marginal_contributions[agent_i])
        else:
            shapley_value = 0.0
        final_shapley_values.append(shapley_value)
    
    print(f"Final Shapley values: {final_shapley_values}")
    
    return final_shapley_values, running_means_history


def compute_taylor_delta_policy(maddpg, obs, actions, action_spaces, epsilon):
    """
    Compute Taylor error delta for policy approximation.
    Adapted from test_pettingzoo_detection.py for SMAC environments.
    
    Args:
        maddpg: MADDPG model
        obs: List of observations for each agent
        actions: List of actions for each agent
        action_spaces: List of action spaces for each agent
        epsilon: Perturbation magnitude
        
    Returns:
        list: Taylor error delta for each agent
    """
    # Convert discrete actions to one-hot encoding for SMAC
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

        eta_i = epsilon * grad_i.sign() / torch.max(grad_i.norm(p=2), torch.tensor(1e-6))
        
        # Second-order Taylor approximation: f(x + η) ≈ f(x) + ∇f(x)^T η
        j_tilde = critic_val + torch.dot(grad_i.flatten(), eta_i.flatten())
        p_torch_obs_i = torch_obs[i] + eta_i
        p_action_logits_i = agent_i.policy(p_torch_obs_i)
        p_action_log_probs = torch.log_softmax(p_action_logits_i, dim=-1)
        p_max_action_idx = torch.argmax(p_action_log_probs, dim=-1)
        j_perturbed = p_action_log_probs.gather(-1, p_max_action_idx.unsqueeze(-1)).squeeze()
        delta_error = abs(j_perturbed - j_tilde).item()
        delta_errors.append(delta_error)

    return delta_errors


def compute_pairwise_frob_norms(maddpg, obs, actions, action_spaces):
    """
    Compute Frobenius norms of cross-agent Hessian blocks for all (i, j).
    Returns an N x N list where entry [i][j] approximates || \partial^2 v_i / (\partial obs_i \partial obs_j) ||_F.
    """
    # Convert discrete actions to one-hot encoding for SMAC
    one_hot_actions = []
    for i, action in enumerate(actions):
        one_hot = np.zeros(action_spaces[i].n)
        one_hot[action] = 1.0
        one_hot_actions.append(one_hot)
    actions = one_hot_actions

    torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    torch_actions = [Variable(torch.Tensor([actions[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    vf_in = torch.cat((*torch_obs, *torch_actions), dim=1)

    N = maddpg.nagents
    results = np.zeros((N, N))

    for i in range(N):
        critic_val = maddpg.agents[i].critic(vf_in).mean()
        grad_i = torch.autograd.grad(critic_val, torch_obs[i], create_graph=True, retain_graph=True)[0]
        obs_dim = grad_i.shape[1]

        for j in range(N):
            # Batch compute all second derivatives at once
            hessian_rows = torch.autograd.grad(
                grad_i.squeeze(), 
                torch_obs[j], 
                grad_outputs=torch.eye(obs_dim, device=torch_device, dtype=grad_i.dtype),
                retain_graph=True, 
                allow_unused=True,
                is_grads_batched=True
            )[0]

            frob_norm = torch.norm(hessian_rows, p='fro')
            results[i][j] = frob_norm.item()

        # normalize the values in agent i's row
        results[i] = results[i] / (np.sum(results[i]) + 1e-10)

    return results


def run_frobenius_analysis(config, maddpg, logdir, env):
    """
    Run Frobenius analysis for 1 episode.
    
    Args:
        config: Configuration object
        maddpg: MADDPG model
        logdir: Log directory for saving files
        
    Returns:
        numpy.ndarray: N x N matrix of average Frobenius norms
    """
    n_agents = maddpg.nagents
    num_episodes = FROBENIUS_EPISODES
    
    # Initialize storage for Frobenius norms
    total_frob_norms = np.zeros((n_agents, n_agents))
    total_timesteps = 0
    
    print(f"Running {num_episodes} episode(s) to compute pairwise Frobenius norms...")
    print(f"Number of agents: {n_agents}")
    
    # Run episodes
    for episode in tqdm(range(num_episodes), desc="Frobenius episodes"):
        obs, action_masks = env.reset()
        episode_reward = 0
        
        while True:
            # Get actions for all agents
            torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=False) 
                        for i in range(maddpg.nagents)]
            torch_masks = [Variable(torch.Tensor(action_masks[i]).to(torch_device), requires_grad=False)
                           if action_masks[i] is not None else None for i in range(maddpg.nagents)]
            
            # Get policy actions
            torch_agent_actions = maddpg.step(torch_obs, explore=False, action_masks=torch_masks)
            agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions]
            
            # Create action dict
            actions_dict = {}
            for i, agent_name in enumerate(env.possible_agents):
                action = agent_actions[i].argmax()  # SMAC uses discrete actions
                actions_dict[agent_name] = action
            
            # Convert actions to list format for Frobenius computation
            actions_list = [actions_dict[agent_name] for agent_name in env.possible_agents]
            
            # Compute pairwise Frobenius norms
            frob_norms = compute_pairwise_frob_norms(maddpg, obs, actions_list, env.action_space)
            
            # Accumulate Frobenius norms
            for i in range(n_agents):
                for j in range(n_agents):
                    total_frob_norms[i, j] += frob_norms[i][j]
            total_timesteps += 1
            
            # Step environment
            obs, rewards, dones, infos, action_masks = env.step(actions_dict)
            episode_reward += sum(rewards)
            
            # Check if episode is done
            if dones.all():
                break
        
        print(f"Episode completed with total reward: {episode_reward}")
    
    # Compute average Frobenius norms
    if total_timesteps > 0:
        avg_frob_norms = total_frob_norms / total_timesteps
    else:
        avg_frob_norms = total_frob_norms
    
    print(f"Completed analysis with {total_timesteps} total timesteps")
    print(f"Average Frobenius norms shape: {avg_frob_norms.shape}")
    
    return avg_frob_norms


def compute_outbound_influence(avg_frob_norms):
    """
    Compute outbound influence I_i^out for each agent i.
    I_i^out = sum of frob[j][i] for all j != i
    
    Args:
        avg_frob_norms: N x N matrix of average Frobenius norms
        
    Returns:
        list: Outbound influence scores for each agent
    """
    n_agents = avg_frob_norms.shape[0]
    outbound_influence = []
    
    for i in range(n_agents):
        # Sum of influences from agent i to all other agents j (j != i)
        influence_sum = 0.0
        for j in range(n_agents):
            if j != i:
                influence_sum += avg_frob_norms[j][i]  # frob[j][i] - influence of i on j
        outbound_influence.append(influence_sum)
    
    return outbound_influence


def compute_cascade_risk_index(shapley_values, outbound_influence):
    """
    Compute Cascade Risk Index (CRI) for each agent i.
    CRI_i = max(0, -shapley_i) * I_i^out
    
    Args:
        shapley_values: List of Shapley values for each agent
        outbound_influence: List of outbound influence scores for each agent
        
    Returns:
        list: Cascade Risk Index values for each agent
    """
    cascade_risk = []
    
    for i in range(len(shapley_values)):
        risk = max(0, -shapley_values[i]) * outbound_influence[i]
        cascade_risk.append(risk)
    
    return cascade_risk


def compute_ranking_and_matching(shapley_values, outbound_influence, normal_reward, attack_rewards):
    """
    Compute agent rankings based on different metrics and their matching accuracies.
    
    Args:
        shapley_values: List of Shapley values for each agent
        outbound_influence: List of outbound influence scores for each agent
        normal_reward: Reward from normal episode
        attack_rewards: List of rewards when each agent is attacked
        
    Returns:
        dict: Dictionary containing rankings and matching accuracies
    """
    n_agents = len(shapley_values)
    
    # Compute reward drops (impact of attacking each agent)
    reward_drops = [normal_reward - attack_reward for attack_reward in attack_rewards]
    
    # Create agent indices
    agent_indices = list(range(n_agents))
    
    # Sort agents by different metrics (descending order)
    shapley_ranking = sorted(agent_indices, key=lambda i: shapley_values[i], reverse=True)
    outbound_ranking = sorted(agent_indices, key=lambda i: outbound_influence[i], reverse=True)
    reward_drop_ranking = sorted(agent_indices, key=lambda i: reward_drops[i], reverse=True)
    
    # Compute position-wise matching accuracies
    def compute_position_wise_matching(ranking1, ranking2):
        """Compute position-wise matching for each index/rank"""
        position_matches = []
        for i in range(len(ranking1)):
            matches = 1 if ranking1[i] == ranking2[i] else 0
            position_matches.append(matches)
        return position_matches
    
    shapley_vs_reward_position_matches = compute_position_wise_matching(shapley_ranking, reward_drop_ranking)
    outbound_vs_reward_position_matches = compute_position_wise_matching(outbound_ranking, reward_drop_ranking)
    
    return {
        'shapley_ranking': shapley_ranking,
        'outbound_ranking': outbound_ranking,
        'reward_drop_ranking': reward_drop_ranking,
        'reward_drops': reward_drops,
        'shapley_vs_reward_position_matches': shapley_vs_reward_position_matches,
        'outbound_vs_reward_position_matches': outbound_vs_reward_position_matches
    }


def plot_convergence(running_means_history, logdir, n_agents):
    """Plot the convergence of Shapley values over Monte Carlo iterations"""
    plt.figure(figsize=(12, 8))
    
    # Get consistent color palette
    agent_colors = get_agent_colors(n_agents)
    
    for agent_i in range(n_agents):
        iterations = range(1, len(running_means_history[agent_i]) + 1)
        plt.plot(iterations, running_means_history[agent_i], 
                label=f'Agent {agent_i}', color=agent_colors[agent_i], linewidth=2)
    
    plt.xlabel('Monte Carlo Iterations')
    plt.ylabel('Running Mean Shapley Value')
    plt.title('Convergence of Shapley Values (Monte Carlo Approximation)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    convergence_path = os.path.join(logdir, 'shapley_convergence.png')
    plt.savefig(convergence_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved convergence plot to {convergence_path}")


def plot_shapley_barchart(shapley_values, logdir, n_agents):
    """Plot Shapley values as a bar chart"""
    plt.figure(figsize=(10, 6))
    
    agents = list(range(n_agents))
    # Get consistent color palette
    agent_colors = get_agent_colors(n_agents)
    colors = [agent_colors[i] for i in range(n_agents)]
    
    bars = plt.bar(agents, shapley_values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on top of bars
    for bar, val in zip(bars, shapley_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(abs(min(shapley_values)), max(shapley_values)),
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Agent ID')
    plt.ylabel('Shapley Value')
    plt.title('Shapley Values (Monte Carlo Approximation)')
    plt.xticks(agents)
    plt.grid(axis='y', alpha=0.3)
    
    # Add legend
    legend_labels = [f'Agent {i}' for i in range(n_agents)]
    legend_patches = [plt.matplotlib.patches.Patch(color=colors[i], label=legend_labels[i]) 
                     for i in range(n_agents)]
    plt.legend(handles=legend_patches, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    barchart_path = os.path.join(logdir, 'shapley_values_barchart.png')
    plt.savefig(barchart_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved Shapley values bar chart to {barchart_path}")


def plot_outbound_influence_barchart(outbound_influence, logdir, n_agents):
    """Plot outbound influence scores as a bar chart"""
    plt.figure(figsize=(10, 6))
    
    agents = list(range(n_agents))
    # Get consistent color palette
    agent_colors = get_agent_colors(n_agents)
    colors = [agent_colors[i] for i in range(n_agents)]
    
    bars = plt.bar(agents, outbound_influence, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on top of bars
    for bar, val in zip(bars, outbound_influence):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(outbound_influence),
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Agent ID')
    plt.ylabel('Outbound Influence Score')
    plt.title('Outbound Influence Scores')
    plt.xticks(agents)
    plt.grid(axis='y', alpha=0.3)
    
    # Add legend
    legend_labels = [f'Agent {i}' for i in range(n_agents)]
    legend_patches = [plt.matplotlib.patches.Patch(color=colors[i], label=legend_labels[i]) 
                     for i in range(n_agents)]
    plt.legend(handles=legend_patches, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    barchart_path = os.path.join(logdir, 'outbound_influence_barchart.png')
    plt.savefig(barchart_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved outbound influence bar chart to {barchart_path}")


def plot_cascade_risk_barchart(cascade_risk, logdir, n_agents):
    """Plot Cascade Risk Index values as a bar chart"""
    plt.figure(figsize=(10, 6))
    
    agents = list(range(n_agents))
    # Get consistent color palette
    agent_colors = get_agent_colors(n_agents)
    colors = [agent_colors[i] for i in range(n_agents)]
    
    bars = plt.bar(agents, cascade_risk, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on top of bars
    for bar, val in zip(bars, cascade_risk):
        height = bar.get_height()
        if max(cascade_risk) > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(cascade_risk),
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            plt.text(bar.get_x() + bar.get_width()/2., 0.001,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Agent ID')
    plt.ylabel('Cascade Risk Index (CRI)')
    plt.title('Cascade Risk Index')
    plt.xticks(agents)
    plt.grid(axis='y', alpha=0.3)
    
    # Add legend
    legend_labels = [f'Agent {i}' for i in range(n_agents)]
    legend_patches = [plt.matplotlib.patches.Patch(color=colors[i], label=legend_labels[i]) 
                     for i in range(n_agents)]
    plt.legend(handles=legend_patches, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    barchart_path = os.path.join(logdir, 'cascade_risk_index_barchart.png')
    plt.savefig(barchart_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved Cascade Risk Index bar chart to {barchart_path}")


def create_influence_heatmap(avg_frob_norms, logdir, n_agents):
    """Create a heatmap visualization of the Frobenius influence matrix."""
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    im = plt.imshow(avg_frob_norms, cmap='viridis', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Average Frobenius Norm', rotation=270, labelpad=20)
    
    # Set ticks and labels
    plt.xticks(range(n_agents), [f'Agent {i}' for i in range(n_agents)])
    plt.yticks(range(n_agents), [f'Agent {i}' for i in range(n_agents)])
    
    # Add text annotations
    for i in range(n_agents):
        for j in range(n_agents):
            text = plt.text(j, i, f'{avg_frob_norms[i, j]:.3f}',
                           ha="center", va="center", 
                           color="black" if avg_frob_norms[i, j] > np.max(avg_frob_norms)/2 else "white")
    
    plt.title('Agent Influence Matrix\n(Average Frobenius Norms)', fontsize=14, fontweight='bold')
    plt.xlabel('Influencing Agent (j)', fontsize=12)
    plt.ylabel('Influenced Agent (i)', fontsize=12)
    
    # Save plot
    heatmap_path = os.path.join(logdir, 'frobenius_influence_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved Frobenius influence heatmap to {heatmap_path}")


def create_influence_pie_charts(avg_frob_norms, logdir, n_agents):
    """
    Create pie charts showing the influence of other agents on each agent.
    
    Args:
        avg_frob_norms: N x N matrix of average Frobenius norms
        logdir: Directory to save the plots
        n_agents: Number of agents
    """
    # Create subplots for each agent
    max_per_row = 3
    rows = math.ceil(n_agents / max_per_row)
    cols = min(n_agents, max_per_row)
    
    fig, axes = plt.subplots(rows, cols, figsize=(2.8*cols, 4*rows))
    if n_agents == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.flatten()
    
    # Get consistent color palette
    agent_colors = get_agent_colors(n_agents)
    
    for i in range(n_agents):
        ax = axes[i]
        
        # Get influences of all agents (including self) on agent i
        influences = []
        labels = []
        colors_for_agent = []
        
        for j in range(n_agents):
            influences.append(avg_frob_norms[i, j])
            if i == j:
                labels.append(f'Agent {j} (self)')
            else:
                labels.append(f'Agent {j}')
            colors_for_agent.append(agent_colors[j])
        
        # Only create pie chart if there are influences
        if len(influences) > 0 and sum(influences) > 0:
            # Remove zero influences for cleaner visualization
            non_zero_indices = [k for k, val in enumerate(influences) if val > 1e-10]
            if non_zero_indices:
                clean_influences = [influences[k] for k in non_zero_indices]
                clean_labels = [labels[k] for k in non_zero_indices]
                clean_colors = [colors_for_agent[k] for k in non_zero_indices]
                
                # Create pie chart (without labels, using legend instead)
                wedges, texts, autotexts = ax.pie(
                    clean_influences, 
                    colors=clean_colors,
                    autopct='%1.1f%%',
                    startangle=90
                )
                
                # Improve text readability
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
            else:
                # No non-zero influences
                ax.text(0.5, 0.5, 'No significant\ninfluences', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
        else:
            # No influences at all
            ax.text(0.5, 0.5, 'No influences\ndetected', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
        
        ax.set_title(f'Influences on Agent {i}', fontsize=14, fontweight='bold')
    
    # Hide unused subplots
    for j in range(n_agents, len(axes)):
        axes[j].set_visible(False)
    
    # Create a single legend at the bottom of the figure
    legend_labels = []
    legend_colors = []
    for j in range(n_agents):
        legend_labels.append(f'Agent {j}')
        legend_colors.append(agent_colors[j])
    
    # Create legend patches
    legend_patches = [plt.matplotlib.patches.Patch(color=legend_colors[j], label=legend_labels[j]) for j in range(n_agents)]
    fig.legend(handles=legend_patches, loc='lower center', ncol=min(n_agents, 5), 
               bbox_to_anchor=(0.5, -0.05), fontsize=10)
    
    plt.suptitle('Agent Influence Analysis (Pairwise Frobenius Norms)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Leave space for legend at bottom
    
    # Save plot
    pie_chart_path = os.path.join(logdir, 'agent_influence_pie_charts.png')
    plt.savefig(pie_chart_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved influence pie charts to {pie_chart_path}")


def rollout_normal_episode(env, maddpg, seed, collect_taylor_errors=False, epsilon=0.01):
    """
    Run a normal episode where all agents use their learned policies.
    
    Args:
        env: Environment instance
        maddpg: MADDPG model
        seed: Random seed for episode
        collect_taylor_errors: Whether to collect Taylor error data at each timestep
        epsilon: Perturbation magnitude for Taylor error computation
        
    Returns:
        float or tuple: Total episode reward, or (total_reward, taylor_errors_per_timestep) if collect_taylor_errors=True
    """
    obs, action_masks = env.reset()
    total_reward = 0.0
    step_count = 0
    taylor_errors_per_timestep = [] if collect_taylor_errors else None
    
    while True:
        # Get actions for all agents
        torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=False) 
                    for i in range(maddpg.nagents)]
        torch_masks = [Variable(torch.Tensor(action_masks[i]).to(torch_device), requires_grad=False)
                       if action_masks[i] is not None else None for i in range(maddpg.nagents)]
        
        # Get policy actions
        torch_agent_actions = maddpg.step(torch_obs, explore=False, action_masks=torch_masks)
        agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions]
        
        # Create action dict with all agents using learned policies
        actions_dict = {}
        actions_list = []
        for i, agent_name in enumerate(env.possible_agents):
            action = agent_actions[i].argmax()  # SMAC uses discrete actions
            actions_dict[agent_name] = action
            actions_list.append(action)
        
        # Collect Taylor error data if requested
        if collect_taylor_errors:
            nagents = len(env.possible_agents)
            action_spaces = [env.action_space[i] for i in range(nagents)]
            taylor_errors = compute_taylor_delta_policy(maddpg, obs, actions_list, action_spaces, epsilon)
            taylor_errors_per_timestep.append(taylor_errors)
        
        # Step environment
        obs, rewards, dones, infos, action_masks = env.step(actions_dict)
        
        # Compute total reward
        step_reward = np.sum(rewards)
        total_reward += step_reward
        step_count += 1
        
        # Check if episode is done
        if dones.all():
            break
    
    if collect_taylor_errors:
        return total_reward, taylor_errors_per_timestep
    else:
        return total_reward


def rollout_attacked_episode(env, maddpg, attacked_agent_id, seed, collect_taylor_errors=False, epsilon=0.01):
    """
    Run an episode where one specific agent is attacked (performs worst actions).
    
    Args:
        env: Environment instance
        maddpg: MADDPG model
        attacked_agent_id: Index of the agent to attack
        seed: Random seed for episode
        collect_taylor_errors: Whether to collect Taylor error data at each timestep
        epsilon: Perturbation magnitude for Taylor error computation
        
    Returns:
        float or tuple: Total episode reward, or (total_reward, taylor_errors_per_timestep) if collect_taylor_errors=True
    """
    obs, action_masks = env.reset()
    total_reward = 0.0
    step_count = 0
    taylor_errors_per_timestep = [] if collect_taylor_errors else None
    
    while True:
        # Get actions for all agents
        torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=False) 
                    for i in range(maddpg.nagents)]
        torch_masks = [Variable(torch.Tensor(action_masks[i]).to(torch_device), requires_grad=False)
                       if action_masks[i] is not None else None for i in range(maddpg.nagents)]
        
        # Get policy actions
        torch_agent_actions = maddpg.step(torch_obs, explore=False, action_masks=torch_masks)
        agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions]
        
        # Create action dict
        actions_dict = {}
        actions_list = []
        for i, agent_name in enumerate(env.possible_agents):
            if i == attacked_agent_id:
                action_logits = maddpg.get_action_logits(torch_obs, torch_masks)

                # Worst action attack
                # Apply action mask - set invalid actions to very high values so they won't be selected as minimum
                masked_logits = action_logits[attacked_agent_id].clone().squeeze()
                # masked_logits[action_masks[i] == 0] = float('inf')
                # action = torch.argmin(masked_logits).item()

                # action = torch.argmin(action_logits[attacked_agent_id]).item()

                # kth best action
                masked_logits[action_masks[i] == 0] = float('inf') # set invalid actions to very low values so they won't be selected as maximum
                indices = torch.argsort(masked_logits, descending=False)
                action = indices[1].item() # 3rd best action
            else:
                # Normal agent: use learned policy
                action = agent_actions[i].argmax()
            
            actions_dict[agent_name] = action
            actions_list.append(action)
        
        # Collect Taylor error data if requested
        if collect_taylor_errors:
            nagents = len(env.possible_agents)
            action_spaces = [env.action_space[i] for i in range(nagents)]
            taylor_errors = compute_taylor_delta_policy(maddpg, obs, actions_list, action_spaces, epsilon)
            taylor_errors_per_timestep.append(taylor_errors)
        
        # Step environment
        obs, rewards, dones, infos, action_masks = env.step(actions_dict)
        
        # Compute total reward
        step_reward = np.sum(rewards)
        total_reward += step_reward
        step_count += 1
        
        # Check if episode is done
        if dones.all():
            break
    
    if collect_taylor_errors:
        return total_reward, taylor_errors_per_timestep
    else:
        return total_reward


def run_attack_analysis(config, maddpg, logdir, env):
    """
    Run attack vs no-attack analysis and collect Taylor error data.
    
    Args:
        config: Configuration object
        maddpg: MADDPG model
        logdir: Log directory for saving files
        
    Returns:
        tuple: (normal_reward, attack_rewards, normal_taylor_errors, attack_taylor_errors) 
               where attack_rewards and attack_taylor_errors are lists for each attacked agent
    """
    n_agents = maddpg.nagents
    
    print(f"Running attack vs no-attack analysis with Taylor error collection...")
    print(f"Number of agents: {n_agents}")
    
    # Step 1: Run normal episode and collect both reward and Taylor errors
    print("Running normal episode (no attacks) and collecting Taylor errors...")
    normal_reward, normal_taylor_errors = rollout_normal_episode(env, maddpg, config.seed, collect_taylor_errors=True)
    print(f"Normal episode reward: {normal_reward:.3f}")
    
    # Compute mean Taylor errors across timesteps for each agent (normal scenario)
    normal_mean_taylor_errors = []
    for agent_id in range(n_agents):
        agent_errors = [timestep_errors[agent_id] for timestep_errors in normal_taylor_errors]
        mean_error = np.mean(agent_errors) if agent_errors else 0.0
        normal_mean_taylor_errors.append(mean_error)
    
    print(f"Normal scenario mean Taylor errors: {[f'{e:.6f}' for e in normal_mean_taylor_errors]}")
    
    # Step 2: Run episodes with each agent attacked and collect both rewards and Taylor errors
    attack_rewards = []
    attack_mean_taylor_errors = []
    
    for agent_id in range(n_agents):
        print(f"Running episode with Agent {agent_id} attacked and collecting Taylor errors...")
        attacked_reward, attack_taylor_errors = rollout_attacked_episode(env, maddpg, agent_id, config.seed, collect_taylor_errors=True)
        attack_rewards.append(attacked_reward)
        
        # Compute mean Taylor errors across timesteps for each agent (attack scenario)
        attack_mean_errors_per_agent = []
        for other_agent_id in range(n_agents):
            agent_errors = [timestep_errors[other_agent_id] for timestep_errors in attack_taylor_errors]
            mean_error = np.mean(agent_errors) if agent_errors else 0.0
            attack_mean_errors_per_agent.append(mean_error)
        
        attack_mean_taylor_errors.append(attack_mean_errors_per_agent)
        
        print(f"Episode reward when Agent {agent_id} attacked: {attacked_reward:.3f}")
        print(f"Attack scenario (Agent {agent_id} attacked) mean Taylor errors: {[f'{e:.6f}' for e in attack_mean_errors_per_agent]}")
    
    return normal_reward, attack_rewards, normal_mean_taylor_errors, attack_mean_taylor_errors


def plot_attack_analysis_barchart(normal_reward, attack_rewards, logdir, n_agents):
    """
    Plot attack analysis results as a bar chart.
    
    Args:
        normal_reward: Reward from normal episode
        attack_rewards: List of rewards when each agent is attacked
        logdir: Directory to save the plot
        n_agents: Number of agents
    """
    plt.figure(figsize=(12, 8))
    
    # Prepare data
    categories = ['Normal'] + [f'Agent {i} Attacked' for i in range(n_agents)]
    rewards = [normal_reward] + attack_rewards
    
    # Get consistent color palette
    colors = ['green'] + ['red' for _ in range(n_agents)]  # Normal in green, attacks in red
    
    # Create bar chart
    bars = plt.bar(categories, rewards, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on top of bars
    for bar, val in zip(bars, rewards):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(abs(min(rewards)), max(rewards)),
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Customize plot
    plt.xlabel('Scenario', fontsize=12)
    plt.ylabel('Episode Reward', fontsize=12)
    plt.title('Attack vs No-Attack Analysis', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add a horizontal line for normal performance reference
    plt.axhline(y=normal_reward, color='green', linestyle='--', alpha=0.7, label=f'Normal Performance: {normal_reward:.3f}')
    
    # Calculate and show impact
    impacts = [(normal_reward - attack_reward) for attack_reward in attack_rewards]
    max_impact = max(impacts) if impacts else 0
    max_impact_agent = impacts.index(max_impact) if impacts else 0
    
    # plt.text(0.02, 0.98, f'Max Impact: Agent {max_impact_agent} ({max_impact:.3f})', 
    #          transform=plt.gca().transAxes, verticalalignment='top',
    #          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.legend()
    plt.tight_layout()
    
    # Save plot
    barchart_path = os.path.join(logdir, 'attack_analysis_barchart.png')
    plt.savefig(barchart_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved attack analysis bar chart to {barchart_path}")
    
    return impacts


def plot_taylor_error_barchart(normal_mean_errors, attack_mean_errors, logdir, n_agents):
    """
    Plot Taylor error analysis results as a bar chart comparing normal vs attacked scenarios.
    
    Args:
        normal_mean_errors: List of mean Taylor errors per agent in normal scenario
        attack_mean_errors: List of lists - for each attacked agent, the mean Taylor errors of all agents
        logdir: Directory to save the plot
        n_agents: Number of agents
    """
    # Create subplots: one for each attacked agent scenario
    # Calculate how to distribute n_agents+1 columns across 2 rows
    total_plots = n_agents + 1
    cols_per_row = math.ceil(total_plots / 2)
    fig, axes = plt.subplots(2, cols_per_row, figsize=(2.8*cols_per_row, 4*2))
    
    # Flatten axes array to handle 2D subplot grid properly
    if total_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Get consistent color palette
    agent_colors = get_agent_colors(n_agents)
    
    # Plot 1: Normal scenario (all agents)
    ax = axes[0]
    agents = list(range(n_agents))
    colors = [agent_colors[i] for i in range(n_agents)]
    
    bars = ax.bar(agents, normal_mean_errors, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on top of bars
    for bar, val in zip(bars, normal_mean_errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(abs(min(normal_mean_errors)), max(normal_mean_errors)),
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    ax.set_xlabel('Agent ID')
    ax.set_ylabel('Mean Taylor Error')
    ax.set_title('Normal Scenario')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(agents)
    
    # Plot 2-N+1: Attack scenarios (for each attacked agent)
    for attacked_agent_id in range(n_agents):
        ax = axes[attacked_agent_id + 1]
        attack_errors = attack_mean_errors[attacked_agent_id]
        
        # Use same colors, but highlight the attacked agent in red
        colors_attack = [agent_colors[i] if i != attacked_agent_id else 'red' for i in range(n_agents)]
        
        bars = ax.bar(agents, attack_errors, color=colors_attack, alpha=0.8, edgecolor='black')
        
        # Add value labels on top of bars
        for bar, val in zip(bars, attack_errors):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(abs(min(attack_errors)), max(attack_errors)),
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        ax.set_xlabel('Agent ID')
        ax.set_ylabel('Mean Taylor Error')
        ax.set_title(f'Agent {attacked_agent_id} Attacked')
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticks(agents)
    
    plt.suptitle('Taylor Error Analysis: Normal vs Attack Scenarios', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    barchart_path = os.path.join(logdir, 'taylor_error_analysis_barchart.png')
    plt.savefig(barchart_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved Taylor error analysis bar chart to {barchart_path}")


def plot_aggregated_taylor_error_barchart(mean_normal_taylor, std_normal_taylor, 
                                        mean_attack_taylor, std_attack_taylor, logdir, n_agents):
    """
    Plot aggregated Taylor error analysis results as a bar chart comparing normal vs attacked scenarios.
    
    Args:
        mean_normal_taylor: Mean Taylor errors per agent in normal scenario across seeds
        std_normal_taylor: Std Taylor errors per agent in normal scenario across seeds
        mean_attack_taylor: Mean Taylor errors matrix - for each attacked agent, the mean errors of all agents across seeds
        std_attack_taylor: Std Taylor errors matrix - for each attacked agent, the std errors of all agents across seeds
        logdir: Directory to save the plot
        n_agents: Number of agents
    """
    # Create subplots: one for each attacked agent scenario
    # Calculate how to distribute n_agents+1 columns across 2 rows
    total_plots = n_agents + 1
    cols_per_row = 3
    rows = math.ceil(total_plots / cols_per_row)
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(5 * cols_per_row, 12))
    
    # Flatten axes array to handle 2D subplot grid properly
    if total_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Get consistent color palette
    agent_colors = get_agent_colors(n_agents)
    
    # Calculate global y-axis limits for consistency across all subplots
    all_values = list(mean_normal_taylor)
    for attack_means in mean_attack_taylor:
        all_values.extend(attack_means)
    
    global_max = max(all_values) if all_values else 0.01
    global_text_offset = 0.001
    global_y_limit_upper = global_max * 1.1 if global_max > 0 else 0.01  # Extra space for text labels
    
    # Plot 1: Normal scenario (all agents)
    ax = axes[0]
    agents = list(range(n_agents))
    colors = [agent_colors[i] for i in range(n_agents)]
    
    bars = ax.bar(agents, mean_normal_taylor, color=colors, alpha=0.8, edgecolor='black')
    
    # Add error bars for standard deviation
    # ax.errorbar(agents, mean_normal_taylor, yerr=std_normal_taylor, fmt='none', color='black', capsize=3)
    ax.errorbar(agents, mean_normal_taylor, fmt='none', color='black', capsize=3)
    
    # Add value labels on top of bars
    for bar, val, std_val in zip(bars, mean_normal_taylor, std_normal_taylor):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + global_text_offset,
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    ax.set_xlabel('Agent ID')
    ax.set_ylabel('Mean Taylor Error')
    ax.set_title('Normal Scenario')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(agents)
    ax.set_ylim(0, global_y_limit_upper)
    
    # Plot 2-N+1: Attack scenarios (for each attacked agent)
    for attacked_agent_id in range(n_agents):
        ax = axes[attacked_agent_id + 1]
        attack_means = mean_attack_taylor[attacked_agent_id]
        attack_stds = std_attack_taylor[attacked_agent_id]
        
        # Use same colors, but highlight the attacked agent in red
        colors_attack = [agent_colors[i] if i != attacked_agent_id else 'red' for i in range(n_agents)]
        
        bars = ax.bar(agents, attack_means, color=colors_attack, alpha=0.8, edgecolor='black')
        
        # Add error bars for standard deviation
        # ax.errorbar(agents, attack_means, yerr=attack_stds, fmt='none', color='black', capsize=3)
        ax.errorbar(agents, attack_means, fmt='none', color='black', capsize=3)
        
        # Add value labels on top of bars using global text offset
        for bar, val, std_val in zip(bars, attack_means, attack_stds):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + global_text_offset,
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
    barchart_path = os.path.join(logdir, 'aggregated_taylor_error_analysis_barchart.png')
    plt.savefig(barchart_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved aggregated Taylor error analysis bar chart to {barchart_path}")


def plot_aggregated_shapley_barchart(mean_shapley_values, std_shapley_values, logdir, n_agents):
    """Plot mean Shapley values as a bar chart"""
    plt.figure(figsize=(10, 6))
    
    agents = list(range(n_agents))
    # Get consistent color palette
    agent_colors = get_agent_colors(n_agents)
    colors = [agent_colors[i] for i in range(n_agents)]
    
    bars = plt.bar(agents, mean_shapley_values,
                   color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on top of bars
    for bar, val in zip(bars, mean_shapley_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(abs(min(mean_shapley_values)), max(mean_shapley_values)),
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Agent ID')
    plt.ylabel('Mean Shapley Value')
    plt.title('Mean Shapley Values Across Multiple Seeds')
    plt.xticks(agents)
    plt.grid(axis='y', alpha=0.3)
    
    # Add legend
    legend_labels = [f'Agent {i}' for i in range(n_agents)]
    legend_patches = [plt.matplotlib.patches.Patch(color=colors[i], label=legend_labels[i]) 
                     for i in range(n_agents)]
    plt.legend(handles=legend_patches, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    barchart_path = os.path.join(logdir, 'mean_shapley_values_barchart.png')
    plt.savefig(barchart_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved mean Shapley values bar chart to {barchart_path}")


def plot_aggregated_outbound_influence_barchart(mean_outbound_influence, std_outbound_influence, logdir, n_agents):
    """Plot mean outbound influence scores as a bar chart"""
    plt.figure(figsize=(10, 6))
    
    agents = list(range(n_agents))
    # Get consistent color palette
    agent_colors = get_agent_colors(n_agents)
    colors = [agent_colors[i] for i in range(n_agents)]
    
    bars = plt.bar(agents, mean_outbound_influence,
                   color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on top of bars
    for bar, val in zip(bars, mean_outbound_influence):
        height = bar.get_height()
        if max(mean_outbound_influence) > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(mean_outbound_influence),
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Agent ID')
    plt.ylabel('Mean Outbound Influence Score')
    plt.title('Mean Outbound Influence Scores Across Multiple Seeds')
    plt.xticks(agents)
    plt.grid(axis='y', alpha=0.3)
    
    # Add legend
    legend_labels = [f'Agent {i}' for i in range(n_agents)]
    legend_patches = [plt.matplotlib.patches.Patch(color=colors[i], label=legend_labels[i]) 
                     for i in range(n_agents)]
    plt.legend(handles=legend_patches, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    barchart_path = os.path.join(logdir, 'mean_outbound_influence_barchart.png')
    plt.savefig(barchart_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved mean outbound influence bar chart to {barchart_path}")


def create_aggregated_influence_pie_charts(mean_frob_norms, logdir, n_agents):
    """
    Create pie charts showing the mean influence of other agents on each agent.
    
    Args:
        mean_frob_norms: N x N matrix of mean Frobenius norms
        logdir: Directory to save the plots
        n_agents: Number of agents
    """
    # Create subplots for each agent
    max_per_row = 3
    rows = math.ceil(n_agents / max_per_row)
    cols = min(n_agents, max_per_row)
    
    fig, axes = plt.subplots(rows, cols, figsize=(2.8*cols, 4*rows))
    if n_agents == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes if isinstance(axes, (list, np.ndarray)) else [axes]
    else:
        axes = axes.flatten()
    
    # Get consistent color palette
    agent_colors = get_agent_colors(n_agents)
    
    for i in range(n_agents):
        ax = axes[i]
        
        # Get influences of all agents (including self) on agent i
        influences = []
        labels = []
        colors_for_agent = []
        
        for j in range(n_agents):
            influences.append(mean_frob_norms[i, j])
            if i == j:
                labels.append(f'Agent {j} (self)')
            else:
                labels.append(f'Agent {j}')
            colors_for_agent.append(agent_colors[j])
        
        # Only create pie chart if there are influences
        if len(influences) > 0 and sum(influences) > 0:
            # Remove zero influences for cleaner visualization
            non_zero_indices = [k for k, val in enumerate(influences) if val > 1e-10]
            if non_zero_indices:
                clean_influences = [influences[k] for k in non_zero_indices]
                clean_labels = [labels[k] for k in non_zero_indices]
                clean_colors = [colors_for_agent[k] for k in non_zero_indices]
                
                # Create pie chart (without labels, using legend instead)
                wedges, texts, autotexts = ax.pie(
                    clean_influences, 
                    colors=clean_colors,
                    autopct='%1.1f%%',
                    startangle=90,
                    textprops={'fontsize': 8}
                )
            else:
                ax.text(0.5, 0.5, 'No significant\ninfluences', 
                       ha='center', va='center', transform=ax.transAxes, fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No influences', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
        
        ax.set_title(f'Influences on Agent {i}', fontsize=14, fontweight='bold')
    
    # Hide unused subplots
    for j in range(n_agents, len(axes)):
        axes[j].set_visible(False)
    
    # Create a single legend at the bottom of the figure
    legend_labels = []
    legend_colors = []
    for j in range(n_agents):
        legend_labels.append(f'Agent {j}')
        legend_colors.append(agent_colors[j])
    
    # Create legend patches
    legend_patches = [plt.matplotlib.patches.Patch(color=legend_colors[j], label=legend_labels[j]) for j in range(n_agents)]
    fig.legend(handles=legend_patches, loc='lower center', ncol=min(n_agents, 5), 
               bbox_to_anchor=(0.5, -0.05), fontsize=10)
    
    plt.suptitle('Agent Influence Analysis (Pairwise Frobenius Norms)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Leave space for legend at bottom
    
    # Save plot
    pie_chart_path = os.path.join(logdir, 'mean_agent_influence_pie_charts.png')
    plt.savefig(pie_chart_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved mean influence pie charts to {pie_chart_path}")


def plot_shapley_reward_matching_barchart(mean_position_accuracies, std_position_accuracies, logdir, n_agents):
    """Plot position-wise matching accuracies between Shapley ranking and reward drop ranking"""
    plt.figure(figsize=(10, 6))
    
    positions = list(range(n_agents))
    # Get consistent color palette
    agent_colors = get_agent_colors(n_agents)
    colors = [agent_colors[i] for i in range(n_agents)]
    
    bars = plt.bar(positions, mean_position_accuracies,
                   color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on top of bars
    for bar, val in zip(bars, mean_position_accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Ranking Position')
    plt.ylabel('Mean Matching Accuracy')
    plt.title('Position-wise Matching Accuracy: Shapley vs Reward Drop Rankings')
    plt.xticks(positions, [f'Rank {i+1}' for i in positions])
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    
    # Save plot
    barchart_path = os.path.join(logdir, 'shapley_reward_matching_barchart.png')
    plt.savefig(barchart_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved Shapley-reward matching accuracy bar chart to {barchart_path}")


def plot_outbound_reward_matching_barchart(mean_position_accuracies, std_position_accuracies, logdir, n_agents):
    """Plot position-wise matching accuracies between outbound influence ranking and reward drop ranking"""
    plt.figure(figsize=(10, 6))
    
    positions = list(range(n_agents))
    # Get consistent color palette
    agent_colors = get_agent_colors(n_agents)
    colors = [agent_colors[i] for i in range(n_agents)]
    
    bars = plt.bar(positions, mean_position_accuracies,
                   color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on top of bars
    for bar, val in zip(bars, mean_position_accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Ranking Position')
    plt.ylabel('Mean Matching Accuracy')
    plt.title('Position-wise Matching Accuracy: Outbound Influence vs Reward Drop Rankings')
    plt.xticks(positions, [f'Rank {i+1}' for i in positions])
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    
    # Save plot
    barchart_path = os.path.join(logdir, 'outbound_reward_matching_barchart.png')
    plt.savefig(barchart_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved outbound-reward matching accuracy bar chart to {barchart_path}")


def plot_matching_accuracy_barchart(mean_shapley_accuracy, std_shapley_accuracy, 
                                   mean_outbound_accuracy, std_outbound_accuracy, logdir):
    """Plot overall matching accuracies as a comparison bar chart"""
    plt.figure(figsize=(8, 6))
    
    categories = ['Shapley vs\nReward Drop', 'Outbound vs\nReward Drop']
    means = [mean_shapley_accuracy, mean_outbound_accuracy]
    colors = ['blue', 'orange']
    
    bars = plt.bar(categories, means, color=colors, alpha=0.8, 
                   edgecolor='black')
    
    # Add value labels on top of bars
    for bar, val in zip(bars, means):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.ylabel('Mean Overall Matching Accuracy')
    plt.title('Overall Matching Accuracies Between Ranking Methods')
    plt.grid(axis='y', alpha=0.3)
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    
    # Save plot
    barchart_path = os.path.join(logdir, 'overall_matching_accuracy_barchart.png')
    plt.savefig(barchart_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved overall matching accuracy bar chart to {barchart_path}")


def save_results(shapley_values, running_means_history, avg_frob_norms, 
                outbound_influence, cascade_risk, normal_reward, attack_rewards, 
                attack_impacts, logdir, config):
    """Save all results to files for later analysis"""
    
    # Save comprehensive results
    results_file = os.path.join(logdir, 'integrated_analysis_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'shapley_values': shapley_values,
            'outbound_influence_scores': outbound_influence,
            'cascade_risk_index': cascade_risk,
            'frobenius_matrix': avg_frob_norms.tolist(),
            'convergence_data': running_means_history,
            'attack_analysis': {
                'normal_reward': normal_reward,
                'attack_rewards': attack_rewards,
                'attack_impacts': attack_impacts
            },
            'num_agents': len(shapley_values),
            'shapley_episodes': SHAPLEY_EPISODES,
            'frobenius_episodes': FROBENIUS_EPISODES,
            'map_name': config.map_name,
            'model_path': config.model_path,
            'seed': config.seed
        }, f, indent=2)
    
    # Save Frobenius matrix as CSV
    frob_csv_path = os.path.join(logdir, 'frobenius_norms_matrix.csv')
    np.savetxt(frob_csv_path, avg_frob_norms, delimiter=',', fmt='%.6f')
    
    print(f"Saved comprehensive results to {results_file}")
    print(f"Saved Frobenius matrix to {frob_csv_path}")


def run_multi_seed_analysis(config):
    """Run multi-seed analysis"""

    # Create env
    env = create_environment(config)
    
    # Load the trained MADDPG model
    print(f"Loading MADDPG model from {config.model_path}")
    maddpg = MADDPG.init_from_save(config.model_path, test_mode=True)
    maddpg.prep_training(device=DEVICE)
    
    n_agents = maddpg.nagents
    
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(
        os.getcwd(), 
        'runs', 
        f"smac_{config.map_name}_multi_seed_shapley_frob_analysis", 
        f"{timestamp}_max_{config.max_iterations}_iterations"
    )
    os.makedirs(logdir, exist_ok=True)
    
    print(f"Results will be saved to: {logdir}")
    print(f"Environment: {config.map_name}")
    print(f"Number of agents: {n_agents}")
    print(f"Max iterations: {config.max_iterations}")
    print(f"Shapley episodes per iteration: {SHAPLEY_EPISODES}")
    print(f"Frobenius episodes per iteration: {FROBENIUS_EPISODES}")
    
    # Initialize storage for all iterations
    all_shapley_values = []
    all_frob_norms = []
    all_outbound_influence = []
    all_matching_results = []
    all_normal_taylor_errors = []
    all_attack_taylor_errors = []
    
    # Run multiple iterations with different seeds
    for iteration in range(config.max_iterations):
        print(f"\n" + "="*60)
        print(f"ITERATION {iteration + 1} of {config.max_iterations}")
        print("="*60)
        
        # Use iteration as seed for reproducibility
        seed = iteration
        env.seed(seed)
        
        # Create a temporary config for this iteration
        temp_config = type('Config', (), {})()
        temp_config.map_name = config.map_name
        temp_config.model_path = config.model_path
        temp_config.seed = seed
        
        # Set random seeds for this iteration
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        print(f"Set random seed to {seed}")
        
        # Step 1: Compute Shapley values
        print(f"\nSTEP 1 (Iteration {iteration + 1}): Computing Shapley Values")
        shapley_values, running_means_history = monte_carlo_shapley(temp_config, maddpg, logdir, env)
        
        # Step 2: Compute Frobenius norms  
        print(f"\nSTEP 2 (Iteration {iteration + 1}): Computing Frobenius Norms")
        avg_frob_norms = run_frobenius_analysis(temp_config, maddpg, logdir, env)
        
        # Step 3: Compute outbound influence scores
        outbound_influence = compute_outbound_influence(avg_frob_norms)
        
        # Step 4: Run attack analysis and collect Taylor error data
        normal_reward, attack_rewards, normal_taylor_errors, attack_taylor_errors = run_attack_analysis(temp_config, maddpg, logdir, env)
        
        # Compute rankings and matching accuracies
        matching_results = compute_ranking_and_matching(
            shapley_values, outbound_influence, normal_reward, attack_rewards
        )
        
        # Store results
        all_shapley_values.append(shapley_values)
        all_frob_norms.append(avg_frob_norms)
        all_outbound_influence.append(outbound_influence)
        all_matching_results.append(matching_results)
        all_normal_taylor_errors.append(normal_taylor_errors)
        all_attack_taylor_errors.append(attack_taylor_errors)
        
        print(f"Iteration {iteration + 1} completed successfully")
        print(f"  Shapley position matches: {matching_results['shapley_vs_reward_position_matches']}")
        print(f"  Outbound position matches: {matching_results['outbound_vs_reward_position_matches']}")
        
    
    # Compute aggregated statistics
    print("\n" + "="*60)
    print("COMPUTING AGGREGATED STATISTICS")
    print("="*60)
    
    if not all_shapley_values:
        print("No successful iterations completed!")
        return
    
    # Convert to numpy arrays for easier computation
    shapley_array = np.array(all_shapley_values)  # (iterations, agents)
    frob_array = np.array(all_frob_norms)  # (iterations, agents, agents)
    outbound_array = np.array(all_outbound_influence)  # (iterations, agents)
    
    # Compute means and standard deviations
    mean_shapley = np.mean(shapley_array, axis=0)
    std_shapley = np.std(shapley_array, axis=0)
    
    mean_frob_norms = np.mean(frob_array, axis=0)
    std_frob_norms = np.std(frob_array, axis=0)
    
    mean_outbound = np.mean(outbound_array, axis=0)
    std_outbound = np.std(outbound_array, axis=0)
    
    # Compute Taylor error aggregated statistics
    normal_taylor_array = np.array(all_normal_taylor_errors)  # (iterations, agents)
    attack_taylor_array = np.array(all_attack_taylor_errors)  # (iterations, attacked_agents, agents)
    
    mean_normal_taylor = np.mean(normal_taylor_array, axis=0)
    std_normal_taylor = np.std(normal_taylor_array, axis=0)
    
    mean_attack_taylor = np.mean(attack_taylor_array, axis=0)  # (attacked_agents, agents)
    std_attack_taylor = np.std(attack_taylor_array, axis=0)  # (attacked_agents, agents)
    
    # Compute cascade risk index from mean values
    cascade_risk = compute_cascade_risk_index(mean_shapley, mean_outbound)
    
    # Compute position-wise matching accuracy statistics
    shapley_position_matches = [result['shapley_vs_reward_position_matches'] for result in all_matching_results]
    outbound_position_matches = [result['outbound_vs_reward_position_matches'] for result in all_matching_results]
    
    # Convert to numpy arrays for easier computation (iterations x positions)
    shapley_position_array = np.array(shapley_position_matches)  # (iterations, n_agents)
    outbound_position_array = np.array(outbound_position_matches)  # (iterations, n_agents)
    
    # Compute mean and std for each position
    mean_shapley_position_accuracy = np.mean(shapley_position_array, axis=0)
    std_shapley_position_accuracy = np.std(shapley_position_array, axis=0)
    
    mean_outbound_position_accuracy = np.mean(outbound_position_array, axis=0)
    std_outbound_position_accuracy = np.std(outbound_position_array, axis=0)
    
    # Also compute overall accuracies for summary
    overall_shapley_accuracies = np.mean(shapley_position_array, axis=1)  # Mean across positions for each iteration
    overall_outbound_accuracies = np.mean(outbound_position_array, axis=1)  # Mean across positions for each iteration
    
    mean_overall_shapley_accuracy = np.mean(overall_shapley_accuracies)
    std_overall_shapley_accuracy = np.std(overall_shapley_accuracies)
    
    mean_overall_outbound_accuracy = np.mean(overall_outbound_accuracies)
    std_overall_outbound_accuracy = np.std(overall_outbound_accuracies)
    
    # Create visualizations
    print("\n" + "="*60)
    print("CREATING AGGREGATED VISUALIZATIONS")
    print("="*60)
    
    plot_aggregated_shapley_barchart(mean_shapley, std_shapley, logdir, n_agents)
    plot_aggregated_outbound_influence_barchart(mean_outbound, std_outbound, logdir, n_agents)
    create_aggregated_influence_pie_charts(mean_frob_norms, logdir, n_agents)
    plot_cascade_risk_barchart(cascade_risk, logdir, n_agents)  # Use existing function
    
    # Plot aggregated Taylor error analysis
    plot_aggregated_taylor_error_barchart(mean_normal_taylor, std_normal_taylor, 
                                        mean_attack_taylor, std_attack_taylor, logdir, n_agents)
    
    # Plot position-wise matching accuracies
    plot_shapley_reward_matching_barchart(mean_shapley_position_accuracy, std_shapley_position_accuracy, logdir, n_agents)
    plot_outbound_reward_matching_barchart(mean_outbound_position_accuracy, std_outbound_position_accuracy, logdir, n_agents)
    
    # Plot overall matching accuracies
    plot_matching_accuracy_barchart(mean_overall_shapley_accuracy, std_overall_shapley_accuracy, 
                                   mean_overall_outbound_accuracy, std_overall_outbound_accuracy, logdir)
    
    # Save comprehensive results
    print("\n" + "="*60)
    print("SAVING COMPREHENSIVE RESULTS")
    print("="*60)
    
    results_file = os.path.join(logdir, 'multi_seed_analysis_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'aggregated_statistics': {
                'mean_shapley_values': mean_shapley.tolist(),
                'std_shapley_values': std_shapley.tolist(),
                'mean_outbound_influence': mean_outbound.tolist(),
                'std_outbound_influence': std_outbound.tolist(),
                'mean_frob_norms': mean_frob_norms.tolist(),
                'std_frob_norms': std_frob_norms.tolist(),
                'cascade_risk_index': cascade_risk,
                'mean_overall_shapley_accuracy': mean_overall_shapley_accuracy,
                'std_overall_shapley_accuracy': std_overall_shapley_accuracy,
                'mean_overall_outbound_accuracy': mean_overall_outbound_accuracy,
                'std_overall_outbound_accuracy': std_overall_outbound_accuracy,
                'mean_shapley_position_accuracy': mean_shapley_position_accuracy.tolist(),
                'std_shapley_position_accuracy': std_shapley_position_accuracy.tolist(),
                'mean_outbound_position_accuracy': mean_outbound_position_accuracy.tolist(),
                'std_outbound_position_accuracy': std_outbound_position_accuracy.tolist(),
                'mean_normal_taylor_errors': mean_normal_taylor.tolist(),
                'std_normal_taylor_errors': std_normal_taylor.tolist(),
                'mean_attack_taylor_errors': mean_attack_taylor.tolist(),
                'std_attack_taylor_errors': std_attack_taylor.tolist()
            },
            'raw_data': {
                'all_shapley_values': shapley_array.tolist(),
                'all_outbound_influence': outbound_array.tolist(),
                'all_normal_taylor_errors': normal_taylor_array.tolist(),
                'all_attack_taylor_errors': attack_taylor_array.tolist(),
                'all_frob_norms': frob_array.tolist(),
                'all_matching_results': all_matching_results
            },
            'configuration': {
                'max_iterations': config.max_iterations,
                'num_agents': n_agents,
                'shapley_episodes': SHAPLEY_EPISODES,
                'frobenius_episodes': FROBENIUS_EPISODES,
                'map_name': config.map_name,
                'model_path': config.model_path
            }
        }, f, indent=2)
    
    print(f"Saved comprehensive results to {results_file}")
    
    # Final summary
    print("\n" + "="*70)
    print("MULTI-SEED ANALYSIS COMPLETED!")
    print("="*70)
    print(f"Successfully completed {len(all_shapley_values)}/{config.max_iterations} iterations")
    print(f"Mean Shapley values: {[f'{val:.3f}±{std:.3f}' for val, std in zip(mean_shapley, std_shapley)]}")
    print(f"Mean outbound influence: {[f'{val:.3f}±{std:.3f}' for val, std in zip(mean_outbound, std_outbound)]}")
    print(f"Mean normal Taylor errors: {[f'{val:.6f}±{std:.6f}' for val, std in zip(mean_normal_taylor, std_normal_taylor)]}")
    print(f"Cascade Risk Index: {[f'{val:.3f}' for val in cascade_risk]}")
    print(f"Overall shapley-reward matching accuracy: {mean_overall_shapley_accuracy:.3f}±{std_overall_shapley_accuracy:.3f}")
    print(f"Overall outbound-reward matching accuracy: {mean_overall_outbound_accuracy:.3f}±{std_overall_outbound_accuracy:.3f}")
    print(f"Position-wise shapley matching: {[f'{val:.3f}±{std:.3f}' for val, std in zip(mean_shapley_position_accuracy, std_shapley_position_accuracy)]}")
    print(f"Position-wise outbound matching: {[f'{val:.3f}±{std:.3f}' for val, std in zip(mean_outbound_position_accuracy, std_outbound_position_accuracy)]}")
    print(f"\nResults saved to: {logdir}")
    
    # Identify highest risk agent
    if cascade_risk and max(cascade_risk) > 0:
        highest_risk_agent = np.argmax(cascade_risk)
        max_risk_value = cascade_risk[highest_risk_agent]
        print(f"Highest Cascade Risk Agent: Agent {highest_risk_agent} (CRI = {max_risk_value:.3f})")
    else:
        print("No agent shows significant cascade risk.")

    env.close()

def run(config):
    """Main execution function"""
    
    # Load the trained MADDPG model
    print(f"Loading MADDPG model from {config.model_path}")
    maddpg = MADDPG.init_from_save(config.model_path, test_mode=True)
    maddpg.prep_training(device=DEVICE)
    
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(
        os.getcwd(), 
        'runs', 
        f"smac_{config.map_name}_shapley_frob_analysis", 
        f"{timestamp}_seed_{config.seed}"
    )
    os.makedirs(logdir, exist_ok=True)
    
    print(f"Results will be saved to: {logdir}")
    print(f"Environment: {config.map_name}")
    print(f"Number of agents: {maddpg.nagents}")
    print(f"Shapley episodes: {SHAPLEY_EPISODES}")
    print(f"Frobenius episodes: {FROBENIUS_EPISODES}")
    
    # Set random seed for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    print(f"Set random seed to {config.seed}")
    
    # Step 1: Compute Shapley values
    print("\n" + "="*50)
    print("STEP 1: Computing Shapley Values")
    print("="*50)
    shapley_values, running_means_history = monte_carlo_shapley(config, maddpg, logdir)
    
    # Step 2: Compute Frobenius norms
    print("\n" + "="*50)
    print("STEP 2: Computing Frobenius Norms")
    print("="*50)
    avg_frob_norms = run_frobenius_analysis(config, maddpg, logdir)
    
    # Step 3: Compute outbound influence scores
    print("\n" + "="*50)
    print("STEP 3: Computing Outbound Influence Scores")
    print("="*50)
    outbound_influence = compute_outbound_influence(avg_frob_norms)
    print(f"Outbound influence scores: {outbound_influence}")
    
    # Step 4: Compute Cascade Risk Index
    print("\n" + "="*50)
    print("STEP 4: Computing Cascade Risk Index")
    print("="*50)
    cascade_risk = compute_cascade_risk_index(shapley_values, outbound_influence)
    print(f"Cascade Risk Index values: {cascade_risk}")
    
    # Step 5: Run Attack Analysis and Taylor Error Analysis
    print("\n" + "="*50)
    print("STEP 5: Attack vs No-Attack Analysis with Taylor Error Collection")
    print("="*50)
    normal_reward, attack_rewards, normal_taylor_errors, attack_taylor_errors = run_attack_analysis(config, maddpg, logdir)
    attack_impacts = plot_attack_analysis_barchart(normal_reward, attack_rewards, logdir, maddpg.nagents)
    plot_taylor_error_barchart(normal_taylor_errors, attack_taylor_errors, logdir, maddpg.nagents)
    
    # Step 6: Create all other visualizations
    print("\n" + "="*50)
    print("STEP 6: Creating Other Visualizations")
    print("="*50)
    
    # Shapley-related plots
    plot_convergence(running_means_history, logdir, maddpg.nagents)
    plot_shapley_barchart(shapley_values, logdir, maddpg.nagents)
    
    # Frobenius-related plots
    create_influence_heatmap(avg_frob_norms, logdir, maddpg.nagents)
    create_influence_pie_charts(avg_frob_norms, logdir, maddpg.nagents)
    
    # New feature plots
    plot_outbound_influence_barchart(outbound_influence, logdir, maddpg.nagents)
    plot_cascade_risk_barchart(cascade_risk, logdir, maddpg.nagents)
    
    # Step 7: Save results
    print("\n" + "="*50)
    print("STEP 7: Saving Results")
    print("="*50)
    save_results(shapley_values, running_means_history, avg_frob_norms, 
                outbound_influence, cascade_risk, normal_reward, attack_rewards, 
                attack_impacts, logdir, config)
    
    # Final summary
    print("\n" + "="*60)
    print("INTEGRATED ANALYSIS COMPLETED!")
    print("="*60)
    print(f"Shapley values: {[f'{val:.3f}' for val in shapley_values]}")
    print(f"Outbound influence: {[f'{val:.3f}' for val in outbound_influence]}")
    print(f"Cascade Risk Index: {[f'{val:.3f}' for val in cascade_risk]}")
    print(f"Normal episode reward: {normal_reward:.3f}")
    print(f"Attack impacts: {[f'{val:.3f}' for val in attack_impacts]}")
    print(f"\nResults saved to: {logdir}")
    
    # Identify highest risk agent
    if cascade_risk and max(cascade_risk) > 0:
        max_risk_agent = np.argmax(cascade_risk)
        max_risk_value = cascade_risk[max_risk_agent]
        print(f"\nHighest Risk Agent (CRI): Agent {max_risk_agent} (CRI = {max_risk_value:.3f})")
    else:
        print(f"\nNo agents with significant cascade risk detected.")
    
    # Identify most vulnerable agent (highest attack impact)
    if attack_impacts and max(attack_impacts) > 0:
        most_vulnerable_agent = np.argmax(attack_impacts)
        max_impact_value = attack_impacts[most_vulnerable_agent]
        print(f"Most Vulnerable Agent (Attack): Agent {most_vulnerable_agent} (Impact = {max_impact_value:.3f})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Seed Integrated Shapley Values, Frobenius Analysis, and Attack Analysis for Multi-Agent RL (SMAC)')
    
    parser.add_argument("map_name", help="Name of SMAC map")
    parser.add_argument("model_path", help="Path to trained MADDPG model directory")
    parser.add_argument("--max_iterations", type=int, default=10,
                        help="Maximum number of iterations (seeds) to run for multi-seed analysis (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for single-seed analysis (default: 42)")
    parser.add_argument("--single_seed", action="store_true",
                        help="Run single-seed analysis instead of multi-seed analysis")
    
    config = parser.parse_args()
    
    if config.single_seed:
        run(config)
    else:
        run_multi_seed_analysis(config)
