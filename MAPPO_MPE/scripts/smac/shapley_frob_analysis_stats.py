#!/usr/bin/env python3
"""
Multi-Seed Integrated Shapley Values, Frobenius Norm Analysis, Taylor Error Analysis, and Attack Analysis for Multi-Agent RL (MAPPO - SMAC)

This script runs comprehensive agent influence and risk assessment across multiple seeds and computes
aggregated statistics with matching accuracy analysis. It includes:
1. Monte Carlo Shapley values computation (50 episodes per seed)
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
- SMAC StarCraft II environment support

New Multi-Seed Analysis Features:
- Command line argument for max iterations instead of seed
- Ranking computation: agents sorted by Shapley values, outbound influence, and reward drop
- Matching accuracy: position-wise comparison between rankings
- Aggregated statistics: mean values for all metrics
- Comprehensive multi-seed visualizations

Taylor Error Analysis Features:
- Computes Taylor error approximations for policy perturbations during normal episodes
- Analyzes Taylor errors when specific agents are under attack (worst action attack)
- Provides comparative barchart visualization showing normal vs attacked scenarios
- Shows impact of attacks on Taylor error approximations across all agents
- Multi-seed aggregation of Taylor error statistics with mean and standard deviation
- Integrated visualization in both single-seed and multi-seed analysis modes

Adapted for MAPPO from the original MADDPG implementation.
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
import math
from PIL import Image

from utils.smac_wrapper import SmacWrapper
from MAPPO_SMAC_main import Runner_MAPPO_SMAC

USE_CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if USE_CUDA else 'cpu'
torch_device = torch.device("cuda" if USE_CUDA else "cpu")

# Fixed episode counts
SHAPLEY_EPISODES = 2
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


def sample_coalition(agent_i, other_agents):
    """
    Sample a random coalition that includes agent_i but excludes agent_i from the sampling process.
    This is used to create coalition_with_i and coalition_without_i.
    
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


def rollout_coalition(env, runner, coalition_mask, seed=None):
    """
    Run a single episode rollout with the given coalition.
    Agents in the coalition use their learned policy, others use default action (0).
    
    Args:
        env: Environment instance
        runner: MAPPO runner with trained model
        coalition_mask: Boolean list indicating which agents are in the coalition
        seed: Random seed for episode (not used in SMAC wrapper)
        
    Returns:
        float: Total episode reward (shared payout)
    """
    obs, action_masks = env.reset()
    total_reward = 0.0
    step_count = 0
    
    while True:
        # Get actions for all agents
        actions = []
        for i in range(runner.args.N):
            if coalition_mask[i]:
                mask = action_masks[i]
                action = runner.agent_n.select_action(obs[i], i, evaluate=True, action_mask=mask)
                actions.append(action)
            else:
                # Agent not in coalition: check if no-op (0) is available, otherwise use stop (1)
                if action_masks[i] is not None:
                    action = 0 if action_masks[i][0] else 1
                else:
                    action = 0  # Default fallback
                actions.append(action)
        
        # Step environment
        actions_dict = {agent_name: actions[i] for i, agent_name in enumerate(env.possible_agents)}
        obs, rewards, dones, infos, action_masks = env.step(actions_dict)

        # Compute shared reward (sum of all agent rewards)
        step_reward = np.sum(rewards)
        total_reward += step_reward
        step_count += 1
        
        # Check if episode is done
        if dones.all() or step_count >= runner.args.episode_limit:
            break
    
    return total_reward


def monte_carlo_shapley(env, runner, args, logdir):
    """
    Compute Shapley values using Monte Carlo approximation.
    
    Args:
        env: Environment instance
        runner: MAPPO runner with trained model
        args: Arguments containing configuration
        logdir: Log directory for saving files
        
    Returns:
        tuple: (final_shapley_values, running_means_history)
    """
    n_agents = runner.args.N
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
            r_plus_i = rollout_coalition(env, runner, coalition_with_i)
            
            # Rollout with coalition excluding agent i  
            r_minus_i = rollout_coalition(env, runner, coalition_without_i)
            
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


def compute_pairwise_frob_norms(runner: Runner_MAPPO_SMAC, states):
    """
    Compute Frobenius norms of cross-agent Hessian blocks for all (i, j).
    Returns an N x N list where entry [i][j] approximates || \partial^2 v_i / (\partial obs_i \partial obs_j) ||_F.
    """
    states_tensors = [torch.tensor(state, dtype=torch.float32, requires_grad=True) for state in states]
    states_tensor = torch.cat(states_tensors, dim=0)
    
    # Try to get values from the critic
    values = runner.agent_n.compute_value(states_tensor).squeeze(-1)  # shape: (N,)

    N = runner.args.N
    results = [[0.0 for _ in range(N)] for _ in range(N)]

    for i in range(N):
        # Gradient wrt agent i's observation
        grad_i = torch.autograd.grad(values[i], states_tensors[i], create_graph=True, retain_graph=True)[0]

        for j in range(N):
            # Compute second derivatives (Hessian) between agent i's value and agent j's observations
            hessian_matrix = []
            for k in range(grad_i.shape[0]):  # For each element in the gradient
                second_grad = torch.autograd.grad(grad_i[k], states_tensors[j], retain_graph=True, allow_unused=True)[0]
                hessian_matrix.append(second_grad.flatten())
            
            H = torch.stack(hessian_matrix)
            frob_norm = torch.norm(H, p='fro')
            results[i][j] = frob_norm.item()

    # Normalize results for each agent
    for i in range(N):
        row_sum = sum(results[i]) + 1e-10
        results[i] = [val / row_sum for val in results[i]]

    return results


def run_frobenius_analysis(env, runner, args, logdir):
    """
    Run Frobenius analysis for 1 episode.
    
    Args:
        env: Environment instance
        runner: MAPPO runner with trained model
        args: Arguments containing configuration
        logdir: Log directory for saving files
        
    Returns:
        numpy.ndarray: N x N matrix of average Frobenius norms
    """
    n_agents = runner.args.N
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
        step_count = 0
        
        while True:
            # Get actions for all agents using learned policy
            actions = []
            for i in range(runner.args.N):
                action = runner.agent_n.select_action(obs[i], i, evaluate=True, action_mask=action_masks[i])
                actions.append(action)
            
            # Compute pairwise Frobenius norms
            frob_norms = compute_pairwise_frob_norms(runner, obs)
            
            # Accumulate Frobenius norms
            for i in range(n_agents):
                for j in range(n_agents):
                    total_frob_norms[i, j] += frob_norms[i][j]
            total_timesteps += 1
            
            # Step environment
            actions_dict = {agent_name: actions[i] for i, agent_name in enumerate(env.possible_agents)}
            obs, rewards, dones, infos, action_masks = env.step(actions_dict)
            episode_reward += np.sum(rewards)
            step_count += 1
            
            # Check if episode is done
            if dones.all() or step_count >= runner.args.episode_limit:
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


def compute_taylor_error_policy(runner: Runner_MAPPO_SMAC, states, epsilon=0.01):
    states_tensor = torch.stack([torch.tensor(state, dtype=torch.float32, requires_grad=True) for state in states])

    delta_errors = []

    for i in range(runner.args.N):
        obs = states_tensor[i].unsqueeze(0)  # shape: (1, obs_dim)
        action, dist = runner.agent_n.compute_action(obs, i, evaluate=True, return_dist=True)
        target_val = dist.log_prob(action)
        grad_i = torch.autograd.grad(target_val, obs, create_graph=True, retain_graph=True)[0]
        # grad_i = torch.autograd.grad(critic_val, actions[i], create_graph=True, retain_graph=True)[0]

        eta_i = epsilon * grad_i.sign() / torch.max(grad_i.norm(p=2), torch.tensor(1e-6))
        
        # First-order Taylor approximation: f(x + η) ≈ f(x) + ∇f(x)^T η
        j_tilde = target_val + torch.dot(grad_i.flatten(), eta_i.flatten())
        p_state = obs + eta_i
        p_action, p_dist = runner.agent_n.compute_action(p_state, i, evaluate=True, return_dist=True)
        j_perturbed = p_dist.log_prob(p_action)
        delta_error = abs(j_perturbed - j_tilde).item()
        delta_errors.append(delta_error)

    return delta_errors


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


def create_influence_heatmap(avg_frob_norms, logdir, n_agents):
    """Create a heatmap visualization of the influence matrix."""
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
                           ha="center", va="center", color="w", fontweight='bold')
    
    plt.title('Agent Influence Matrix\n(Average Frobenius Norms)', fontsize=14, fontweight='bold')
    plt.xlabel('Influencing Agent (j)', fontsize=12)
    plt.ylabel('Influenced Agent (i)', fontsize=12)
    
    # Save plot
    heatmap_path = os.path.join(logdir, 'agent_influence_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved influence heatmap to {heatmap_path}")


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


def plot_outbound_influence_barchart(outbound_influence, logdir, n_agents):
    """Plot outbound influence scores as a bar chart"""
    plt.figure(figsize=(10, 6))
    
    agents = list(range(n_agents))
    # Get consistent color palette
    agent_colors = get_agent_colors(n_agents)
    colors = [agent_colors[i] for i in range(n_agents)]
    
    bars = plt.bar(agents, outbound_influence, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on top of bars
    max_influence = max(outbound_influence) if outbound_influence else 0
    for bar, val in zip(bars, outbound_influence):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max_influence,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Agent ID')
    plt.ylabel('Outbound Influence Score')
    plt.title('Outbound Influence Scores (I_i^out)')
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
    """Plot cascade risk index values as a bar chart"""
    plt.figure(figsize=(10, 6))
    
    agents = list(range(n_agents))
    # Get consistent color palette
    agent_colors = get_agent_colors(n_agents)
    colors = [agent_colors[i] for i in range(n_agents)]
    
    bars = plt.bar(agents, cascade_risk, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on top of bars
    max_risk = max(cascade_risk) if cascade_risk else 0
    for bar, val in zip(bars, cascade_risk):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max_risk,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Agent ID')
    plt.ylabel('Cascade Risk Index')
    plt.title('Cascade Risk Index (CRI = max(0, -Shapley) × Outbound Influence)')
    plt.xticks(agents)
    plt.grid(axis='y', alpha=0.3)
    
    # Add legend
    legend_labels = [f'Agent {i}' for i in range(n_agents)]
    legend_patches = [plt.matplotlib.patches.Patch(color=colors[i], label=legend_labels[i]) 
                     for i in range(n_agents)]
    plt.legend(handles=legend_patches, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    barchart_path = os.path.join(logdir, 'cascade_risk_barchart.png')
    plt.savefig(barchart_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved cascade risk bar chart to {barchart_path}")


def save_results(shapley_values, running_means_history, avg_frob_norms, 
                outbound_influence, cascade_risk, normal_reward, attack_rewards, 
                attack_impacts, normal_taylor_errors, attack_taylor_errors, logdir, args):
    """Save all results to files for later analysis"""
    
    # Save comprehensive results
    results_file = os.path.join(logdir, 'integrated_analysis_results.json')
    with open(results_file, 'w') as f:
        results_data = {
            'shapley_values': shapley_values.tolist() if isinstance(shapley_values, np.ndarray) else shapley_values,
            'outbound_influence_scores': outbound_influence.tolist() if isinstance(outbound_influence, np.ndarray) else outbound_influence,
            'cascade_risk_index': cascade_risk.tolist() if isinstance(cascade_risk, np.ndarray) else cascade_risk,
            'frobenius_matrix': avg_frob_norms.tolist(),
            'convergence_data': [[val.tolist() if isinstance(val, np.ndarray) else val for val in row] if isinstance(row, list) else row.tolist() if isinstance(row, np.ndarray) else row for row in running_means_history],
            'attack_analysis': {
                'normal_reward': float(normal_reward) if isinstance(normal_reward, (np.ndarray, np.number)) else normal_reward,
                'attack_rewards': attack_rewards.tolist() if isinstance(attack_rewards, np.ndarray) else attack_rewards,
                'attack_impacts': attack_impacts.tolist() if isinstance(attack_impacts, np.ndarray) else attack_impacts
            },
            'taylor_error_analysis': {
                'normal_taylor_errors': normal_taylor_errors.tolist() if isinstance(normal_taylor_errors, np.ndarray) else normal_taylor_errors,
                'attack_taylor_errors': attack_taylor_errors.tolist() if isinstance(attack_taylor_errors, np.ndarray) else attack_taylor_errors
            },
            'num_agents': len(shapley_values),
            'shapley_episodes': SHAPLEY_EPISODES,
            'frobenius_episodes': FROBENIUS_EPISODES,
            'map_name': args.map_name,
            'model_dir': args.model_dir,
            'seed': args.seed
        }
        json.dump(results_data, f, indent=2)
    
    # Save Frobenius matrix as CSV
    frob_csv_path = os.path.join(logdir, 'frobenius_norms_matrix.csv')
    np.savetxt(frob_csv_path, avg_frob_norms, delimiter=',', fmt='%.6f')
    
    print(f"Saved comprehensive results to {results_file}")
    print(f"Saved Frobenius matrix to {frob_csv_path}")


def rollout_normal_episode(env, runner, seed, collect_taylor_errors=False, epsilon=0.01):
    """
    Run a normal episode where all agents use their learned policies.
    
    Args:
        env: Environment instance
        runner: MAPPO runner with trained model
        seed: Random seed for episode
        collect_taylor_errors: Whether to collect Taylor error data
        epsilon: Perturbation magnitude for Taylor error computation
        
    Returns:
        tuple: (total_reward, taylor_errors) where taylor_errors is list if collect_taylor_errors=True, else (total_reward, None)
    """
    obs, action_masks = env.reset()
    total_reward = 0.0
    step_count = 0
    taylor_errors_per_step = [] if collect_taylor_errors else None
    
    while True:
        # Collect Taylor errors if requested
        if collect_taylor_errors:
            step_taylor_errors = compute_taylor_error_policy(runner, obs, epsilon)
            taylor_errors_per_step.append(step_taylor_errors)
        
        # Get actions for all agents using learned policies
        actions = []
        for i in range(runner.args.N):
            action = runner.agent_n.select_action(obs[i], i, evaluate=True, action_mask=action_masks[i])
            actions.append(action)
        
        # Step environment
        actions_dict = {agent_name: actions[i] for i, agent_name in enumerate(env.possible_agents)}
        next_obs, rewards, dones, infos, next_action_masks = env.step(actions_dict)
        
        # Compute total reward
        step_reward = np.sum(rewards)
        total_reward += step_reward
        step_count += 1
        
        obs = next_obs
        action_masks = next_action_masks
        
        # Check if episode is done
        if dones.all():
            break
    
    # Aggregate Taylor errors if collected
    if collect_taylor_errors and taylor_errors_per_step:
        # Compute mean Taylor error per agent across all steps
        mean_taylor_errors = []
        for agent_i in range(runner.args.N):
            agent_errors = [step_errors[agent_i] for step_errors in taylor_errors_per_step]
            mean_taylor_errors.append(np.mean(agent_errors))
        return total_reward, mean_taylor_errors
    
    return total_reward, None


def rollout_attacked_episode(env, runner, attacked_agent_id, seed, collect_taylor_errors=False, epsilon=0.01):
    """
    Run an episode where one specific agent is attacked (performs worst actions).
    
    Args:
        env: Environment instance
        runner: MAPPO runner with trained model
        attacked_agent_id: Index of the agent to attack
        seed: Random seed for episode
        collect_taylor_errors: Whether to collect Taylor error data
        epsilon: Perturbation magnitude for Taylor error computation
        
    Returns:
        tuple: (total_reward, taylor_errors) where taylor_errors is list if collect_taylor_errors=True, else (total_reward, None)
    """
    obs, action_masks = env.reset()
    total_reward = 0.0
    step_count = 0
    taylor_errors_per_step = [] if collect_taylor_errors else None
    
    while True:
        # Collect Taylor errors if requested
        if collect_taylor_errors:
            step_taylor_errors = compute_taylor_error_policy(runner, obs, epsilon)
            taylor_errors_per_step.append(step_taylor_errors)
        
        # Get actions for all agents
        actions = []
        for i in range(runner.args.N):
            if i == attacked_agent_id:
                # Worst action attack: get action distribution and choose worst valid action
                action, dist = runner.agent_n.select_action(obs[i], i, evaluate=True, 
                                                          action_mask=action_masks[i], return_dist=True)
                
                # Apply action mask to probabilities (set invalid actions to inf)
                valid_probs = dist.probs.clone().squeeze()
                # valid_probs[action_masks[i] == 0] = float('inf')  # Set invalid actions to inf so they won't be minimum
                
                # Choose worst valid action
                # action = torch.argmin(valid_probs).item()

                # Get the third best action
                valid_probs[action_masks[i] == 0] = float('-inf')
                sorted_probs, sorted_indices = torch.sort(valid_probs, descending=True)
                action = sorted_indices[2].item()
            else:
                # Normal agent: use learned policy
                action = runner.agent_n.select_action(obs[i], i, evaluate=True, action_mask=action_masks[i])
            
            actions.append(action)
        
        # Step environment
        actions_dict = {agent_name: actions[i] for i, agent_name in enumerate(env.possible_agents)}
        next_obs, rewards, dones, infos, next_action_masks = env.step(actions_dict)
        
        # Compute total reward
        step_reward = np.sum(rewards)
        total_reward += step_reward
        step_count += 1
        
        obs = next_obs
        action_masks = next_action_masks
        
        # Check if episode is done
        if dones.all():
            break
    
    # Aggregate Taylor errors if collected
    if collect_taylor_errors and taylor_errors_per_step:
        # Compute mean Taylor error per agent across all steps
        mean_taylor_errors = []
        for agent_i in range(runner.args.N):
            agent_errors = [step_errors[agent_i] for step_errors in taylor_errors_per_step]
            mean_taylor_errors.append(np.mean(agent_errors))
        return total_reward, mean_taylor_errors
    
    return total_reward, None


def run_attack_analysis(env, runner, args, logdir, collect_taylor_errors=False, epsilon=0.01):
    """
    Run attack vs no-attack analysis.
    
    Args:
        env: Environment instance
        runner: MAPPO runner with trained model
        args: Arguments containing configuration
        logdir: Log directory for saving files
        collect_taylor_errors: Whether to collect Taylor error data
        epsilon: Perturbation magnitude for Taylor error computation
        
    Returns:
        tuple: (normal_reward, attack_rewards, normal_taylor_errors, attack_taylor_errors) 
               where taylor_errors are None if collect_taylor_errors=False
    """
    n_agents = runner.args.N
    
    print(f"Running attack vs no-attack analysis...")
    print(f"Number of agents: {n_agents}")
    
    # Step 1: Run normal episode
    print("Running normal episode (no attacks)...")
    normal_reward, normal_taylor_errors = rollout_normal_episode(env, runner, args.seed, 
                                                               collect_taylor_errors, epsilon)
    print(f"Normal episode reward: {normal_reward:.3f}")
    if collect_taylor_errors and normal_taylor_errors:
        print(f"Normal Taylor errors: {[f'{err:.6f}' for err in normal_taylor_errors]}")
    
    # Step 2: Run episodes with each agent attacked
    attack_rewards = []
    attack_taylor_errors = []
    for agent_id in range(n_agents):
        print(f"Running episode with Agent {agent_id} attacked...")
        attacked_reward, attacked_taylor = rollout_attacked_episode(env, runner, agent_id, args.seed,
                                                                   collect_taylor_errors, epsilon)
        attack_rewards.append(attacked_reward)
        if collect_taylor_errors:
            attack_taylor_errors.append(attacked_taylor)
        print(f"Episode reward when Agent {agent_id} attacked: {attacked_reward:.3f}")
        if collect_taylor_errors and attacked_taylor:
            print(f"Attacked Taylor errors: {[f'{err:.6f}' for err in attacked_taylor]}")
    
    return normal_reward, attack_rewards, normal_taylor_errors, attack_taylor_errors


def plot_attack_analysis_barchart(normal_reward, attack_rewards, logdir, n_agents):
    """
    Plot attack analysis results as a bar chart.
    
    Args:
        normal_reward: Reward from normal episode
        attack_rewards: List of rewards when each agent is attacked
        logdir: Directory to save the plot
        n_agents: Number of agents
        
    Returns:
        list: Attack impacts (normal_reward - attack_reward for each agent)
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
    plt.axhline(y=normal_reward, color='green', linestyle='--', alpha=0.7, 
                label=f'Normal Performance: {normal_reward:.3f}')
    
    # Calculate and show impact
    impacts = [(normal_reward - attack_reward) for attack_reward in attack_rewards]
    max_impact = max(impacts) if impacts else 0
    max_impact_agent = impacts.index(max_impact) if impacts else 0
    
    # Add text box with impact information
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
    if not normal_mean_errors or not attack_mean_errors:
        print("No Taylor error data to plot")
        return
    
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
    
    # Hide unused subplots
    for j in range(total_plots, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Taylor Error Analysis: Normal vs Attack Scenarios', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    taylor_path = os.path.join(logdir, 'taylor_error_barchart.png')
    plt.savefig(taylor_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved Taylor error bar chart to {taylor_path}")


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
    
    y_min = min(all_values) - 0.1 * abs(min(all_values))
    y_max = max(all_values) + 0.1 * abs(max(all_values))
    
    # Plot 1: Normal scenario (all agents)
    ax = axes[0]
    agents = list(range(n_agents))
    colors = [agent_colors[i] for i in range(n_agents)]
    
    bars = ax.bar(agents, mean_normal_taylor,
                  color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on top of bars
    for bar, val in zip(bars, mean_normal_taylor):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * (y_max - y_min),
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    ax.set_xlabel('Agent ID')
    ax.set_ylabel('Mean Taylor Error')
    ax.set_title('Normal Scenario')
    ax.grid(axis='y', alpha=0.3)
    ax.set_xticks(agents)
    ax.set_ylim(y_min, y_max)
    
    # Plot 2-N+1: Attack scenarios (for each attacked agent)
    for attacked_agent_id in range(n_agents):
        ax = axes[attacked_agent_id + 1]
        attack_means = mean_attack_taylor[attacked_agent_id]
        attack_stds = std_attack_taylor[attacked_agent_id]
        
        # Use same colors, but highlight the attacked agent in red
        colors_attack = [agent_colors[i] if i != attacked_agent_id else 'red' for i in range(n_agents)]
        
        bars = ax.bar(agents, attack_means,
                      color=colors_attack, alpha=0.8, edgecolor='black')
        
        # Add value labels on top of bars
        for bar, val in zip(bars, attack_means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * (y_max - y_min),
                    f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        ax.set_xlabel('Agent ID')
        ax.set_ylabel('Mean Taylor Error')
        ax.set_title(f'Agent {attacked_agent_id} Attacked')
        ax.grid(axis='y', alpha=0.3)
        ax.set_xticks(agents)
        ax.set_ylim(y_min, y_max)
    
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
        position_matches = []
        for pos in range(n_agents):
            if ranking1[pos] == ranking2[pos]:
                position_matches.append(1.0)
            else:
                position_matches.append(0.0)
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


def plot_aggregated_shapley_barchart(mean_shapley_values, std_shapley_values, logdir, n_agents):
    """Plot aggregated Shapley values across multiple seeds"""
    plt.figure(figsize=(10, 6))
    
    agents = list(range(n_agents))
    # Get consistent color palette
    agent_colors = get_agent_colors(n_agents)
    colors = [agent_colors[i] for i in range(n_agents)]
    
    bars = plt.bar(agents, mean_shapley_values, color=colors, 
                   alpha=0.8, edgecolor='black')
    
    # Add value labels on top of bars
    for i, (bar, mean_val) in enumerate(zip(bars, mean_shapley_values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mean_shapley_values)*0.02,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Agent ID')
    plt.ylabel('Shapley Value')
    plt.title('Aggregated Shapley Values')
    plt.xticks(agents)
    plt.grid(axis='y', alpha=0.3)
    
    # Add legend
    legend_labels = [f'Agent {i}' for i in range(n_agents)]
    legend_patches = [plt.matplotlib.patches.Patch(color=colors[i], label=legend_labels[i]) 
                     for i in range(n_agents)]
    plt.legend(handles=legend_patches, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    barchart_path = os.path.join(logdir, 'aggregated_shapley_values_barchart.png')
    plt.savefig(barchart_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved aggregated Shapley values bar chart to {barchart_path}")


def plot_aggregated_outbound_influence_barchart(mean_outbound_influence, std_outbound_influence, logdir, n_agents):
    """Plot aggregated outbound influence scores across multiple seeds"""
    plt.figure(figsize=(10, 6))
    
    agents = list(range(n_agents))
    # Get consistent color palette
    agent_colors = get_agent_colors(n_agents)
    colors = [agent_colors[i] for i in range(n_agents)]
    
    bars = plt.bar(agents, mean_outbound_influence, color=colors, 
                   alpha=0.8, edgecolor='black')
    
    # Add value labels on top of bars
    for i, (bar, mean_val) in enumerate(zip(bars, mean_outbound_influence)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mean_outbound_influence)*0.02,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Agent ID')
    plt.ylabel('Outbound Influence Score')
    plt.title('Aggregated Outbound Influence Scores')
    plt.xticks(agents)
    plt.grid(axis='y', alpha=0.3)
    
    # Add legend
    legend_labels = [f'Agent {i}' for i in range(n_agents)]
    legend_patches = [plt.matplotlib.patches.Patch(color=colors[i], label=legend_labels[i]) 
                     for i in range(n_agents)]
    plt.legend(handles=legend_patches, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    barchart_path = os.path.join(logdir, 'aggregated_outbound_influence_barchart.png')
    plt.savefig(barchart_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved aggregated outbound influence bar chart to {barchart_path}")


def plot_aggregated_cascade_risk_barchart(mean_cascade_risk, std_cascade_risk, logdir, n_agents):
    """Plot aggregated cascade risk index values across multiple seeds"""
    plt.figure(figsize=(10, 6))
    
    agents = list(range(n_agents))
    # Get consistent color palette
    agent_colors = get_agent_colors(n_agents)
    colors = [agent_colors[i] for i in range(n_agents)]
    
    bars = plt.bar(agents, mean_cascade_risk, color=colors, 
                   alpha=0.8, edgecolor='black')
    
    # Add value labels on top of bars
    for i, (bar, mean_val) in enumerate(zip(bars, mean_cascade_risk)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mean_cascade_risk)*0.02,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Agent ID')
    plt.ylabel('Cascade Risk Index')
    plt.title('Aggregated Cascade Risk Index')
    plt.xticks(agents)
    plt.grid(axis='y', alpha=0.3)
    
    # Add legend
    legend_labels = [f'Agent {i}' for i in range(n_agents)]
    legend_patches = [plt.matplotlib.patches.Patch(color=colors[i], label=legend_labels[i]) 
                     for i in range(n_agents)]
    plt.legend(handles=legend_patches, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    barchart_path = os.path.join(logdir, 'aggregated_cascade_risk_barchart.png')
    plt.savefig(barchart_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved aggregated cascade risk bar chart to {barchart_path}")


def create_aggregated_influence_pie_charts(mean_frob_norms, logdir, n_agents):
    """
    Create pie charts showing the aggregated influence of other agents on each agent across seeds.
    
    Args:
        mean_frob_norms: N x N matrix of mean Frobenius norms across seeds
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
        
        # Get influences from other agents on agent i (row i, excluding diagonal)
        influences = []
        labels = []
        colors = []
        
        for j in range(n_agents):
            influences.append(mean_frob_norms[i][j])
            # labels.append(f'Agent {j}')
            colors.append(agent_colors[j])
        
        # Only create pie chart if there are influences
        if influences and sum(influences) > 0:
            wedges, texts, autotexts = ax.pie(influences, colors=colors, 
                                            autopct=lambda pct: f'{pct:.1f}%' if pct > 5 else '',
                                            startangle=90)
            # Adjust text properties
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(8)
        else:
            ax.text(0.5, 0.5, 'No significant\ninfluences', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=10)
        
        ax.set_title(f'Influences on Agent {i}', fontsize=11, fontweight='bold')
    
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
    
    plt.suptitle('Aggregated Agent Influence Analysis (Mean Pairwise Frobenius Norms)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Leave space for legend at bottom
    
    # Save plot
    pie_chart_path = os.path.join(logdir, 'aggregated_agent_influence_pie_charts.png')
    plt.savefig(pie_chart_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved aggregated influence pie charts to {pie_chart_path}")


def plot_shapley_reward_matching_barchart(mean_position_accuracies, std_position_accuracies, logdir, n_agents):
    """Plot position-wise matching accuracy between Shapley ranking and reward drop ranking"""
    plt.figure(figsize=(10, 6))
    
    positions = list(range(n_agents))
    
    bars = plt.bar(positions, mean_position_accuracies, 
                   color='green', alpha=0.7, edgecolor='black')
    
    # Add value labels on top of bars
    for i, (bar, mean_val) in enumerate(zip(bars, mean_position_accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{mean_val:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Ranking Position')
    plt.ylabel('Matching Accuracy')
    plt.title('Shapley vs Reward Drop Ranking: Position-wise Matching Accuracy')
    plt.xticks(positions, [f'Pos {i+1}' for i in range(n_agents)])
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    matching_path = os.path.join(logdir, 'shapley_reward_matching_barchart.png')
    plt.savefig(matching_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved Shapley-Reward matching bar chart to {matching_path}")


def plot_outbound_reward_matching_barchart(mean_position_accuracies, std_position_accuracies, logdir, n_agents):
    """Plot position-wise matching accuracy between outbound influence ranking and reward drop ranking"""
    plt.figure(figsize=(10, 6))
    
    positions = list(range(n_agents))
    
    bars = plt.bar(positions, mean_position_accuracies, 
                   color='orange', alpha=0.7, edgecolor='black')
    
    # Add value labels on top of bars
    for i, (bar, mean_val) in enumerate(zip(bars, mean_position_accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{mean_val:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.xlabel('Ranking Position')
    plt.ylabel('Matching Accuracy')
    plt.title('Outbound Influence vs Reward Drop Ranking: Position-wise Matching Accuracy')
    plt.xticks(positions, [f'Pos {i+1}' for i in range(n_agents)])
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    matching_path = os.path.join(logdir, 'outbound_reward_matching_barchart.png')
    plt.savefig(matching_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved Outbound-Reward matching bar chart to {matching_path}")


def plot_matching_accuracy_barchart(mean_shapley_accuracy, std_shapley_accuracy, 
                                   mean_outbound_accuracy, std_outbound_accuracy, logdir):
    """Plot overall matching accuracy comparison between different ranking methods"""
    plt.figure(figsize=(10, 6))
    
    methods = ['Shapley vs Reward Drop', 'Outbound Influence vs Reward Drop']
    accuracies = [mean_shapley_accuracy, mean_outbound_accuracy]
    std_devs = [std_shapley_accuracy, std_outbound_accuracy]
    colors = ['green', 'orange']
    
    bars = plt.bar(methods, accuracies, color=colors, alpha=0.7, 
                   edgecolor='black')
    
    # Add value labels on top of bars
    for i, (bar, mean_val) in enumerate(zip(bars, accuracies)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.ylabel('Overall Matching Accuracy')
    plt.title('Ranking Method Comparison: Overall Matching Accuracy')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    accuracy_path = os.path.join(logdir, 'matching_accuracy_comparison_barchart.png')
    plt.savefig(accuracy_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved matching accuracy comparison bar chart to {accuracy_path}")


def run_multi_seed_analysis(args):
    """
    Run analysis across multiple seeds and compute aggregated statistics.
    
    Args:
        args: Arguments containing configuration including max_iterations
    """
    max_iterations = args.max_iterations
    print(f"Running multi-seed analysis with {max_iterations} iterations...")
    
    # Storage for results across seeds
    all_shapley_values = []
    all_outbound_influence = []
    all_cascade_risk = []
    all_frob_norms = []
    all_normal_rewards = []
    all_attack_rewards = []
    all_normal_taylor_errors = []
    all_attack_taylor_errors = []
    all_shapley_vs_reward_matches = []
    all_outbound_vs_reward_matches = []
    
    # Create main log directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    main_logdir = os.path.join(
        os.getcwd(), 
        'runs', 
        f"smac_{args.map_name}_multi_seed_analysis", 
        f"{timestamp}_{max_iterations}_iterations"
    )
    os.makedirs(main_logdir, exist_ok=True)
    
    print(f"Multi-seed results will be saved to: {main_logdir}")

    # Create environment and runner for this iteration
    env = SmacWrapper.make_env(args.map_name)
    runner = Runner_MAPPO_SMAC(args, env_name=args.map_name, number=1, seed=args.seed)
    
    # Run analysis for each iteration (seed)
    for iteration in range(max_iterations):
        current_seed = iteration  # Use different seed for each iteration
        print(f"\n{'='*80}")
        print(f"ITERATION {iteration + 1}/{max_iterations} (Seed: {current_seed})")
        print(f"{'='*80}")
        
        env.seed(current_seed)
        
        # Load trained model
        runner.agent_n.load_model_from_directory(args.model_dir)
        
        # Create iteration-specific log directory
        iter_logdir = os.path.join(main_logdir, f"iteration_{iteration + 1}_seed_{current_seed}")
        
        # Set random seed for reproducibility
        random.seed(current_seed)
        np.random.seed(current_seed)
        torch.manual_seed(current_seed)
        print(f"Set random seed to {current_seed}")
        
        # Step 1: Compute Shapley values
        print("\n" + "="*50)
        print("STEP 1: Computing Shapley Values")
        print("="*50)
        shapley_values, _ = monte_carlo_shapley(env, runner, args, iter_logdir)
        all_shapley_values.append(shapley_values)
        
        # Step 2: Compute Frobenius norms
        print("\n" + "="*50)
        print("STEP 2: Computing Frobenius Norms")
        print("="*50)
        avg_frob_norms = run_frobenius_analysis(env, runner, args, iter_logdir)
        all_frob_norms.append(avg_frob_norms)
        
        # Step 3: Compute outbound influence scores
        print("\n" + "="*50)
        print("STEP 3: Computing Outbound Influence Scores")
        print("="*50)
        outbound_influence = compute_outbound_influence(avg_frob_norms)
        all_outbound_influence.append(outbound_influence)
        
        # Step 4: Compute Cascade Risk Index
        print("\n" + "="*50)
        print("STEP 4: Computing Cascade Risk Index")
        print("="*50)
        cascade_risk = compute_cascade_risk_index(shapley_values, outbound_influence)
        all_cascade_risk.append(cascade_risk)
        
        # Step 5: Run Attack Analysis with Taylor error collection
        print("\n" + "="*50)
        print("STEP 5: Attack vs No-Attack Analysis with Taylor Errors")
        print("="*50)
        normal_reward, attack_rewards, normal_taylor_errors, attack_taylor_errors = run_attack_analysis(
            env, runner, args, iter_logdir, collect_taylor_errors=True, epsilon=0.01)
        
        all_normal_rewards.append(normal_reward)
        all_attack_rewards.append(attack_rewards)
        if normal_taylor_errors:
            all_normal_taylor_errors.append(normal_taylor_errors)
        if attack_taylor_errors:
            all_attack_taylor_errors.append(attack_taylor_errors)
        
        # Step 6: Compute rankings and matching
        print("\n" + "="*50)
        print("STEP 6: Computing Rankings and Matching")
        print("="*50)
        ranking_results = compute_ranking_and_matching(shapley_values, outbound_influence, 
                                                     normal_reward, attack_rewards)
        all_shapley_vs_reward_matches.append(ranking_results['shapley_vs_reward_position_matches'])
        all_outbound_vs_reward_matches.append(ranking_results['outbound_vs_reward_position_matches'])
        
        print(f"Completed iteration {iteration + 1}/{max_iterations}")

    
    # Step 7: Compute aggregated statistics
    print(f"\n{'='*80}")
    print("COMPUTING AGGREGATED STATISTICS")
    print(f"{'='*80}")
    
    n_agents = len(all_shapley_values[0])
    
    # Aggregate Shapley values
    mean_shapley_values = np.mean(all_shapley_values, axis=0).tolist()
    std_shapley_values = np.std(all_shapley_values, axis=0).tolist()
    
    # Aggregate outbound influence
    mean_outbound_influence = np.mean(all_outbound_influence, axis=0).tolist()
    std_outbound_influence = np.std(all_outbound_influence, axis=0).tolist()
    
    # Aggregate cascade risk
    mean_cascade_risk = np.mean(all_cascade_risk, axis=0).tolist()
    std_cascade_risk = np.std(all_cascade_risk, axis=0).tolist()
    
    # Aggregate Frobenius norms
    mean_frob_norms = np.mean(all_frob_norms, axis=0)
    std_frob_norms = np.std(all_frob_norms, axis=0)
    
    # Aggregate Taylor errors
    mean_normal_taylor = None
    std_normal_taylor = None
    mean_attack_taylor = None
    std_attack_taylor = None
    
    if all_normal_taylor_errors:
        mean_normal_taylor = np.mean(all_normal_taylor_errors, axis=0).tolist()
        std_normal_taylor = np.std(all_normal_taylor_errors, axis=0).tolist()
    
    if all_attack_taylor_errors:
        # Convert to proper numpy array format: (iterations, attacked_agents, agents)
        attack_taylor_array = np.array(all_attack_taylor_errors)  # (iterations, attacked_agents, agents)
        mean_attack_taylor = np.mean(attack_taylor_array, axis=0)  # (attacked_agents, agents)
        std_attack_taylor = np.std(attack_taylor_array, axis=0)    # (attacked_agents, agents)
    
    # Aggregate matching accuracies
    mean_shapley_position_matches = np.mean(all_shapley_vs_reward_matches, axis=0).tolist()
    std_shapley_position_matches = np.std(all_shapley_vs_reward_matches, axis=0).tolist()
    
    mean_outbound_position_matches = np.mean(all_outbound_vs_reward_matches, axis=0).tolist()
    std_outbound_position_matches = np.std(all_outbound_vs_reward_matches, axis=0).tolist()
    
    # Overall matching accuracies
    mean_shapley_accuracy = np.mean([np.mean(matches) for matches in all_shapley_vs_reward_matches])
    std_shapley_accuracy = np.std([np.mean(matches) for matches in all_shapley_vs_reward_matches])
    
    mean_outbound_accuracy = np.mean([np.mean(matches) for matches in all_outbound_vs_reward_matches])
    std_outbound_accuracy = np.std([np.mean(matches) for matches in all_outbound_vs_reward_matches])
    
    # Step 8: Create aggregated visualizations
    print(f"\n{'='*80}")
    print("CREATING AGGREGATED VISUALIZATIONS")
    print(f"{'='*80}")
    
    # Aggregated plots
    plot_aggregated_shapley_barchart(mean_shapley_values, std_shapley_values, main_logdir, n_agents)
    plot_aggregated_outbound_influence_barchart(mean_outbound_influence, std_outbound_influence, main_logdir, n_agents)
    plot_aggregated_cascade_risk_barchart(mean_cascade_risk, std_cascade_risk, main_logdir, n_agents)
    create_aggregated_influence_pie_charts(mean_frob_norms, main_logdir, n_agents)
    
    # Taylor error plots
    if mean_normal_taylor is not None and mean_attack_taylor is not None:
        plot_aggregated_taylor_error_barchart(mean_normal_taylor, std_normal_taylor, 
                                            mean_attack_taylor, std_attack_taylor, main_logdir, n_agents)
    
    # Matching accuracy plots
    plot_shapley_reward_matching_barchart(mean_shapley_position_matches, std_shapley_position_matches, main_logdir, n_agents)
    plot_outbound_reward_matching_barchart(mean_outbound_position_matches, std_outbound_position_matches, main_logdir, n_agents)
    plot_matching_accuracy_barchart(mean_shapley_accuracy, std_shapley_accuracy, 
                                   mean_outbound_accuracy, std_outbound_accuracy, main_logdir)
    
    # Step 9: Save aggregated results
    print(f"\n{'='*80}")
    print("SAVING AGGREGATED RESULTS")
    print(f"{'='*80}")
    
    # Save comprehensive aggregated results
    aggregated_results = {
        'meta': {
            'map_name': args.map_name,
            'max_iterations': max_iterations,
            'base_seed': args.seed,
            'n_agents': n_agents,
            'shapley_episodes_per_seed': SHAPLEY_EPISODES,
            'frobenius_episodes_per_seed': FROBENIUS_EPISODES,
            'timestamp': timestamp
        },
        'shapley_analysis': {
            'mean_values': mean_shapley_values.tolist() if isinstance(mean_shapley_values, np.ndarray) else mean_shapley_values,
            'std_values': std_shapley_values.tolist() if isinstance(std_shapley_values, np.ndarray) else std_shapley_values,
            'all_values': [val.tolist() if isinstance(val, np.ndarray) else val for val in all_shapley_values]
        },
        'outbound_influence_analysis': {
            'mean_values': mean_outbound_influence.tolist() if isinstance(mean_outbound_influence, np.ndarray) else mean_outbound_influence,
            'std_values': std_outbound_influence.tolist() if isinstance(std_outbound_influence, np.ndarray) else std_outbound_influence,
            'all_values': [val.tolist() if isinstance(val, np.ndarray) else val for val in all_outbound_influence]
        },
        'cascade_risk_analysis': {
            'mean_values': mean_cascade_risk.tolist() if isinstance(mean_cascade_risk, np.ndarray) else mean_cascade_risk,
            'std_values': std_cascade_risk.tolist() if isinstance(std_cascade_risk, np.ndarray) else std_cascade_risk,
            'all_values': [val.tolist() if isinstance(val, np.ndarray) else val for val in all_cascade_risk]
        },
        'frobenius_analysis': {
            'mean_matrix': mean_frob_norms.tolist() if isinstance(mean_frob_norms, np.ndarray) else mean_frob_norms,
            'std_matrix': std_frob_norms.tolist() if isinstance(std_frob_norms, np.ndarray) else std_frob_norms
        },
        'attack_analysis': {
            'all_normal_rewards': [val.tolist() if isinstance(val, np.ndarray) else val for val in all_normal_rewards],
            'all_attack_rewards': [val.tolist() if isinstance(val, np.ndarray) else val for val in all_attack_rewards],
            'mean_normal_reward': float(np.mean(all_normal_rewards)),
            'std_normal_reward': float(np.std(all_normal_rewards)),
            'mean_attack_rewards': np.mean(all_attack_rewards, axis=0).tolist(),
            'std_attack_rewards': np.std(all_attack_rewards, axis=0).tolist()
        },
        'taylor_error_analysis': {
            'mean_normal_errors': mean_normal_taylor.tolist() if isinstance(mean_normal_taylor, np.ndarray) else mean_normal_taylor,
            'std_normal_errors': std_normal_taylor.tolist() if isinstance(std_normal_taylor, np.ndarray) else std_normal_taylor,
            'mean_attack_errors': mean_attack_taylor.tolist() if isinstance(mean_attack_taylor, np.ndarray) else mean_attack_taylor,
            'std_attack_errors': std_attack_taylor.tolist() if isinstance(std_attack_taylor, np.ndarray) else std_attack_taylor
        },
        'matching_analysis': {
            'shapley_vs_reward': {
                'mean_position_matches': mean_shapley_position_matches.tolist() if isinstance(mean_shapley_position_matches, np.ndarray) else mean_shapley_position_matches,
                'std_position_matches': std_shapley_position_matches.tolist() if isinstance(std_shapley_position_matches, np.ndarray) else std_shapley_position_matches,
                'mean_overall_accuracy': mean_shapley_accuracy,
                'std_overall_accuracy': std_shapley_accuracy
            },
            'outbound_vs_reward': {
                'mean_position_matches': mean_outbound_position_matches.tolist() if isinstance(mean_outbound_position_matches, np.ndarray) else mean_outbound_position_matches,
                'std_position_matches': std_outbound_position_matches.tolist() if isinstance(std_outbound_position_matches, np.ndarray) else std_outbound_position_matches,
                'mean_overall_accuracy': mean_outbound_accuracy,
                'std_overall_accuracy': std_outbound_accuracy
            }
        }
    }
    
    # Save to JSON file
    results_file = os.path.join(main_logdir, 'aggregated_multi_seed_results.json')
    with open(results_file, 'w') as f:
        json.dump(aggregated_results, f, indent=4)
    
    # Save Frobenius matrices as CSV
    frob_mean_csv = os.path.join(main_logdir, 'aggregated_frobenius_mean_matrix.csv')
    frob_std_csv = os.path.join(main_logdir, 'aggregated_frobenius_std_matrix.csv')
    np.savetxt(frob_mean_csv, mean_frob_norms, delimiter=',', fmt='%.6f')
    np.savetxt(frob_std_csv, std_frob_norms, delimiter=',', fmt='%.6f')
    
    # Step 10: Final summary
    print(f"\n{'='*80}")
    print("MULTI-SEED ANALYSIS COMPLETED!")
    print(f"{'='*80}")
    print(f"Aggregated results across {max_iterations} seeds:")
    print(f"Shapley values (mean): {[f'{m:.3f}' for m in mean_shapley_values]}")
    print(f"Outbound influence (mean): {[f'{m:.3f}' for m in mean_outbound_influence]}")
    print(f"Cascade Risk Index (mean): {[f'{m:.3f}' for m in mean_cascade_risk]}")
    print(f"Shapley vs Reward matching accuracy: {mean_shapley_accuracy:.3f}")
    print(f"Outbound vs Reward matching accuracy: {mean_outbound_accuracy:.3f}")
    print(f"\nAggregated results saved to: {main_logdir}")
    print(f"Individual iteration results saved in subdirectories")
    
    # Print risk assessment
    if max(mean_cascade_risk) > 0:
        max_risk_agent = np.argmax(mean_cascade_risk)
        max_risk_value = mean_cascade_risk[max_risk_agent]
        max_risk_std = std_cascade_risk[max_risk_agent]
        print(f"\nHighest Risk Agent: Agent {max_risk_agent} (CRI = {max_risk_value:.3f})")
    else:
        print(f"\nNo agents with significant cascade risk detected across seeds.")

    # Clean up environment
    env.close()


def main(runner: Runner_MAPPO_SMAC, env, args):
    """Main execution function"""
    
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(
        os.getcwd(), 
        'runs', 
        f"smac_{args.map_name}_shapley_frob_analysis", 
        f"{timestamp}_seed_{args.seed}"
    )
    os.makedirs(logdir, exist_ok=True)
    
    print(f"Results will be saved to: {logdir}")
    print(f"Environment: {args.map_name}")
    print(f"Number of agents: {runner.args.N}")
    print(f"Shapley episodes: {SHAPLEY_EPISODES}")
    print(f"Frobenius episodes: {FROBENIUS_EPISODES}")
    
    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"Set random seed to {args.seed}")
    
    # Step 1: Compute Shapley values
    print("\n" + "="*50)
    print("STEP 1: Computing Shapley Values")
    print("="*50)
    shapley_values, running_means_history = monte_carlo_shapley(env, runner, args, logdir)
    
    # Step 2: Compute Frobenius norms
    print("\n" + "="*50)
    print("STEP 2: Computing Frobenius Norms")
    print("="*50)
    avg_frob_norms = run_frobenius_analysis(env, runner, args, logdir)
    
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
    
    # Step 5: Run Attack Analysis with Taylor Error Analysis
    print("\n" + "="*50)
    print("STEP 5: Attack vs No-Attack Analysis with Taylor Error Analysis")
    print("="*50)
    normal_reward, attack_rewards, normal_taylor_errors, attack_taylor_errors = run_attack_analysis(
        env, runner, args, logdir, collect_taylor_errors=True, epsilon=0.01)
    attack_impacts = plot_attack_analysis_barchart(normal_reward, attack_rewards, logdir, runner.args.N)
    
    # Plot Taylor error analysis if data is available
    if normal_taylor_errors and attack_taylor_errors:
        plot_taylor_error_barchart(normal_taylor_errors, attack_taylor_errors, logdir, runner.args.N)
    
    # Step 6: Create all other visualizations
    print("\n" + "="*50)
    print("STEP 6: Creating Other Visualizations")
    print("="*50)
    
    # Shapley-related plots
    plot_convergence(running_means_history, logdir, runner.args.N)
    plot_shapley_barchart(shapley_values, logdir, runner.args.N)
    
    # Frobenius-related plots
    create_influence_heatmap(avg_frob_norms, logdir, runner.args.N)
    create_influence_pie_charts(avg_frob_norms, logdir, runner.args.N)
    
    # New feature plots
    plot_outbound_influence_barchart(outbound_influence, logdir, runner.args.N)
    plot_cascade_risk_barchart(cascade_risk, logdir, runner.args.N)
    
    # Step 7: Save results
    print("\n" + "="*50)
    print("STEP 7: Saving Results")
    print("="*50)
    save_results(shapley_values, running_means_history, avg_frob_norms, 
                outbound_influence, cascade_risk, normal_reward, attack_rewards, 
                attack_impacts, normal_taylor_errors, attack_taylor_errors, logdir, args)
    
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
        print(f"\nHighest Risk Agent: Agent {max_risk_agent} (CRI = {max_risk_value:.3f})")
    else:
        print(f"\nNo agents with significant cascade risk detected.")
    
    # Identify most vulnerable agent (highest attack impact)
    if attack_impacts and max(attack_impacts) > 0:
        max_impact_agent = np.argmax(attack_impacts)
        max_impact_value = attack_impacts[max_impact_agent]
        print(f"Most Vulnerable Agent: Agent {max_impact_agent} (Impact = {max_impact_value:.3f})")
    else:
        print(f"No significant attack impacts detected.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Seed Integrated Shapley Values, Frobenius Analysis, Taylor Error Analysis, and Attack Analysis for MAPPO in SMAC environments')
    
    # MAPPO hyperparameters (copied from existing scripts)
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help="Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=1000, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=128, help="The number of neurons in hidden layers of the rnn")
    parser.add_argument("--mlp_hidden_dim", type=int, default=128, help="The number of neurons in hidden layers of the mlp")
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
    
    # Required arguments
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save all output files")
    parser.add_argument("--map_name", type=str, required=True, help="SMAC map name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--discrete_action", type=bool, default=True, help="Whether the action space is discrete or continuous")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory to load the trained model")
    
    # Multi-seed analysis argument
    parser.add_argument("--max_iterations", type=int, default=None, 
                        help="Number of iterations for multi-seed analysis. If provided, runs multi-seed analysis instead of single-seed.")
    
    args = parser.parse_args()
    
    # Decide whether to run single-seed or multi-seed analysis
    if args.max_iterations is not None:
        print(f"Running multi-seed analysis with {args.max_iterations} iterations...")
        run_multi_seed_analysis(args)
    else:
        print(f"Running single-seed analysis with seed {args.seed}...")
        
        # Create environment and runner
        env = SmacWrapper.make_env(args.map_name)
        env.seed(args.seed)
        runner = Runner_MAPPO_SMAC(args, env_name=args.map_name, number=1, seed=args.seed)
        
        # Load trained model
        runner.agent_n.load_model_from_directory(args.model_dir)
        
        # Run analysis
        main(runner, env, args)
