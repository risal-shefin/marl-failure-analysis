#!/usr/bin/env python3
"""
Integrated Shapley Values, Frobenius Norm Analysis, and Attack Analysis for Multi-Agent RL (SMAC Version)

This script combines Monte Carlo Shapley value computation with Frobenius norm analysis
and attack vulnerability assessment to provide comprehensive agent influence and risk assessment. It includes:
1. Monte Carlo Shapley values computation (100 episodes)
2. Pairwise Frobenius norm analysis (1 episode)
3. Outbound influence score I_i^out for each agent
4. Cascade Risk Index (CRI) for each agent
5. Attack vs No-Attack analysis (episodic rewards under attack scenarios)

Features:
- Fixed episode counts: 100 for Shapley, 1 for Frobenius, 1 for each attack scenario
- Outbound influence score: sum of frob[j][i] for all j != i
- Cascade Risk Index: max(0, -shapley_i) * I_i^out
- Attack analysis: normal vs individual agent attacks (worst action attack)
- Comprehensive visualization with bar charts and attack impact analysis
- SMAC environment support
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
SHAPLEY_EPISODES = 100
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


def create_environment(config, maddpg):
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


def monte_carlo_shapley(config, maddpg, logdir):
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

    # Create environment for this rollout
    env = create_environment(config, maddpg)
    env.seed(config.seed)
    
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
    
    # Clean up environment
    env.close()
    
    return final_shapley_values, running_means_history


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


def run_frobenius_analysis(config, maddpg, logdir):
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
    
    # Create environment for this analysis
    env = create_environment(config, maddpg)
    env.seed(config.seed)
    
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
    
    # Clean up environment
    env.close()
    
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


def rollout_normal_episode(env, maddpg, seed):
    """
    Run a normal episode where all agents use their learned policies.
    
    Args:
        env: Environment instance
        maddpg: MADDPG model
        seed: Random seed for episode
        
    Returns:
        float: Total episode reward
    """
    obs, action_masks = env.reset()
    total_reward = 0.0
    step_count = 0
    
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
        for i, agent_name in enumerate(env.possible_agents):
            action = agent_actions[i].argmax()  # SMAC uses discrete actions
            actions_dict[agent_name] = action
        
        # Step environment
        obs, rewards, dones, infos, action_masks = env.step(actions_dict)
        
        # Compute total reward
        step_reward = np.sum(rewards)
        total_reward += step_reward
        step_count += 1
        
        # Check if episode is done
        if dones.all():
            break
    
    return total_reward


def rollout_attacked_episode(env, maddpg, attacked_agent_id, seed):
    """
    Run an episode where one specific agent is attacked (performs worst actions).
    
    Args:
        env: Environment instance
        maddpg: MADDPG model
        attacked_agent_id: Index of the agent to attack
        seed: Random seed for episode
        
    Returns:
        float: Total episode reward
    """
    obs, action_masks = env.reset()
    total_reward = 0.0
    step_count = 0
    
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
            if i == attacked_agent_id:
                action_logits = maddpg.get_action_logits(torch_obs, torch_masks)

                # Worst action attack
                # Apply action mask - set invalid actions to very high values so they won't be selected as minimum
                masked_logits = action_logits[attacked_agent_id].clone().squeeze()
                # masked_logits[action_masks[i] == 0] = float('inf')
                # action = torch.argmin(masked_logits).item()

                # action = torch.argmin(action_logits[attacked_agent_id]).item()

                # kth best action
                masked_logits[action_masks[i] == 0] = float('-inf') # set invalid actions to very low values so they won't be selected as maximum
                indices = torch.argsort(action_logits[attacked_agent_id].squeeze(), descending=True)
                action = indices[2].item() # 3rd best action
            else:
                # Normal agent: use learned policy
                action = agent_actions[i].argmax()
            
            actions_dict[agent_name] = action
        
        # Step environment
        obs, rewards, dones, infos, action_masks = env.step(actions_dict)
        
        # Compute total reward
        step_reward = np.sum(rewards)
        total_reward += step_reward
        step_count += 1
        
        # Check if episode is done
        if dones.all():
            break
    
    return total_reward


def run_attack_analysis(config, maddpg, logdir):
    """
    Run attack vs no-attack analysis.
    
    Args:
        config: Configuration object
        maddpg: MADDPG model
        logdir: Log directory for saving files
        
    Returns:
        tuple: (normal_reward, attack_rewards) where attack_rewards is a list of rewards
               when each agent is attacked
    """
    n_agents = maddpg.nagents
    
    print(f"Running attack vs no-attack analysis...")
    print(f"Number of agents: {n_agents}")
    
    # Create environment for analysis
    env = create_environment(config, maddpg)
    env.seed(config.seed)
    
    # Step 1: Run normal episode
    print("Running normal episode (no attacks)...")
    normal_reward = rollout_normal_episode(env, maddpg, config.seed)
    print(f"Normal episode reward: {normal_reward:.3f}")
    
    # Step 2: Run episodes with each agent attacked
    attack_rewards = []
    for agent_id in range(n_agents):
        print(f"Running episode with Agent {agent_id} attacked...")
        attacked_reward = rollout_attacked_episode(env, maddpg, agent_id, config.seed)
        attack_rewards.append(attacked_reward)
        print(f"Episode reward when Agent {agent_id} attacked: {attacked_reward:.3f}")
    
    # Clean up environment
    env.close()
    
    return normal_reward, attack_rewards


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
    
    # Step 5: Run Attack Analysis
    print("\n" + "="*50)
    print("STEP 5: Attack vs No-Attack Analysis")
    print("="*50)
    normal_reward, attack_rewards = run_attack_analysis(config, maddpg, logdir)
    attack_impacts = plot_attack_analysis_barchart(normal_reward, attack_rewards, logdir, maddpg.nagents)
    
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
    parser = argparse.ArgumentParser(description='Integrated Shapley Values, Frobenius Analysis, and Attack Analysis for Multi-Agent RL (SMAC)')
    
    parser.add_argument("map_name", help="Name of SMAC map")
    parser.add_argument("model_path", help="Path to trained MADDPG model directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    
    config = parser.parse_args()
    
    run(config)
