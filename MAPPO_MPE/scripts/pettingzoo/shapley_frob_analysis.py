#!/usr/bin/env python3
"""
Integrated Shapley Values, Frobenius Norm Analysis, and Attack Analysis for Multi-Agent RL (MAPPO - PettingZoo)

This script combines Monte Carlo Shapley value computation with Frobenius norm analysis
and attack vulnerability assessment to provide comprehensive agent influence and risk assessment. It includes:
1. Monte Carlo Shapley values computation (100 episodes)
2. Pairwise Frobenius norm analysis (1 episode)
3. Outbound influence score I_i^out for each agent
4. Cascade Risk Index (CRI) for each agent
5. Attack vs No-Attack analysis (episodic rewards under attack scenarios)
6. GIF saving functionality for coalition rollouts and attack scenarios

Features:
- Fixed episode counts: 100 for Shapley, 1 for Frobenius, 1 for each attack scenario
- Outbound influence score: sum of frob[j][i] for all j != i
- Cascade Risk Index: max(0, -shapley_i) * I_i^out
- Attack analysis: normal vs individual agent attacks (worst action)
- Comprehensive visualization with bar charts and attack impact analysis
- GIF creation for unique coalitions and attack scenarios
- PettingZoo environment support

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
import imageio
from PIL import Image
import math

from make_env_pettingzoo import make_env
from MAPPO_MPE_main import Runner_MAPPO_MPE

USE_CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if USE_CUDA else 'cpu'
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


def rollout_coalition(env, runner, coalition_mask, seed, save_gif=False, gif_path=None, agent_i=None, iteration=None):
    """
    Run a single episode rollout with the given coalition.
    Agents in the coalition use their learned policy, others use default action (0).
    
    Args:
        env: Environment instance
        runner: MAPPO runner with trained model
        coalition_mask: Boolean list indicating which agents are in the coalition
        seed: Random seed for episode
        save_gif: Whether to save frames for GIF creation
        gif_path: Path to save the GIF file
        agent_i: Target agent for Shapley computation (for info file)
        iteration: Monte Carlo iteration number (for info file)
        
    Returns:
        float: Total episode reward (shared payout)
    """
    states = env.reset(seed=seed)
    total_reward = 0.0
    step_count = 0
    frames = []
    
    while True:
        # Capture frame for GIF if requested
        if save_gif:
            frames.append(Image.fromarray(env.render()))
        
        # Get actions for all agents
        actions = []
        for i in range(runner.args.N):
            if coalition_mask[i]:
                action, _ = runner.agent_n.select_action(states[i], i, evaluate=True, return_dist=True)
                actions.append(action)
            else:
                # Agent not in coalition: use default action (0)
                actions.append(0)
        
        # Step environment
        next_states, rewards, dones, infos = env.step(actions)
        
        # Compute shared reward (sum of all agent rewards)
        step_reward = np.sum(rewards)
        total_reward += step_reward
        step_count += 1
        
        states = next_states
        
        # Check if episode is done
        if dones[0] or step_count >= runner.args.episode_limit:  # Use episode limit from args
            break
    
    # Save GIF and coalition info if requested and frames were captured
    if save_gif and frames and gif_path:
        imageio.mimsave(gif_path, frames, duration=125)
        
        # Save coalition info alongside GIF
        info_path = gif_path.replace('.gif', '_info.txt')
        with open(info_path, 'w') as f:
            f.write(f"Coalition Information:\n")
            f.write(f"Agent being evaluated: {agent_i}\n")
            f.write(f"Monte Carlo iteration: {iteration}\n")
            f.write(f"Coalition mask: {coalition_mask}\n")
            f.write(f"Agents in coalition: {[i for i, in_coalition in enumerate(coalition_mask) if in_coalition]}\n")
            f.write(f"Total reward: {total_reward:.3f}\n")
            f.write(f"Episode length: {step_count} steps\n")
        
        print(f"Saved coalition GIF to {gif_path}")
    
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
    
    # Create GIF directory if save_gifs is enabled
    gif_dir = None
    if hasattr(args, 'save_gifs') and args.save_gifs:
        gif_dir = os.path.join(logdir, 'coalition_gifs')
        os.makedirs(gif_dir, exist_ok=True)
        print(f"Coalition GIFs will be saved to: {gif_dir}")

    # Track unique coalitions for GIF saving (save only once per unique coalition)
    saved_coalition_gifs = set()
    
    # Monte Carlo iterations
    for m in tqdm(range(M), desc="Monte Carlo iterations"):
        # For each agent i
        for agent_i in range(n_agents):
            # Get other agents (all except agent_i)
            other_agents = [j for j in range(n_agents) if j != agent_i]
            # Use a random seed for this iteration to add stochasticity
            current_seed = args.seed #+ m * 1000 if args.seed is not None else None

            # Sample a random coalition
            coalition_with_i, coalition_without_i = sample_coalition(agent_i, other_agents)
            
            # Create unique identifiers for coalitions (without iteration number)
            coalition_with_str = ''.join(['1' if x else '0' for x in coalition_with_i])
            coalition_without_str = ''.join(['1' if x else '0' for x in coalition_without_i])
            
            # Generate GIF paths if saving and coalition hasn't been saved yet
            gif_path_with = None
            gif_path_without = None
            if hasattr(args, 'save_gifs') and args.save_gifs and gif_dir:
                # Only save GIFs for unique coalitions we haven't seen before
                if coalition_with_str not in saved_coalition_gifs:
                    gif_path_with = os.path.join(gif_dir, f'agent_{agent_i}_coalition_with_{coalition_with_str}.gif')
                    saved_coalition_gifs.add(coalition_with_str)
                
                if coalition_without_str not in saved_coalition_gifs:
                    gif_path_without = os.path.join(gif_dir, f'agent_{agent_i}_coalition_without_{coalition_without_str}.gif')
                    saved_coalition_gifs.add(coalition_without_str)
            
            # Rollout with coalition including agent i
            r_plus_i = rollout_coalition(env, runner, coalition_with_i, current_seed, 
                                       save_gif=(gif_path_with is not None), 
                                       gif_path=gif_path_with, 
                                       agent_i=agent_i, iteration=m)
            
            # Rollout with coalition excluding agent i  
            r_minus_i = rollout_coalition(env, runner, coalition_without_i, current_seed,
                                        save_gif=(gif_path_without is not None), 
                                        gif_path=gif_path_without, 
                                        agent_i=agent_i, iteration=m)
            
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
    
    # Print GIF saving summary
    if hasattr(args, 'save_gifs') and args.save_gifs and gif_dir:
        print(f"Saved {len(saved_coalition_gifs)} unique coalition GIFs to {gif_dir}")
    
    return final_shapley_values, running_means_history


def compute_pairwise_frob_norms(runner: Runner_MAPPO_MPE, states):
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
    
    # Create GIF directory for Frobenius analysis if save_gifs is enabled
    frob_gif_dir = None
    if hasattr(args, 'save_gifs') and args.save_gifs:
        frob_gif_dir = os.path.join(logdir, 'frobenius_gifs')
        os.makedirs(frob_gif_dir, exist_ok=True)
        print(f"Frobenius analysis GIFs will be saved to: {frob_gif_dir}")
    
    # Run episodes
    for episode in tqdm(range(num_episodes), desc="Frobenius episodes"):
        states = env.reset(seed=args.seed + episode if args.seed is not None else None)
        episode_reward = 0
        frames = []
        step_count = 0
        
        while True:
            # Capture frame for GIF if requested
            if hasattr(args, 'save_gifs') and args.save_gifs:
                frames.append(Image.fromarray(env.render()))
            
            # Get actions for all agents using learned policy
            actions = []
            for i in range(runner.args.N):
                action, _ = runner.agent_n.select_action(states[i], i, evaluate=True, return_dist=True)
                actions.append(action)
            
            # Compute pairwise Frobenius norms
            frob_norms = compute_pairwise_frob_norms(runner, states)
            
            # Accumulate Frobenius norms
            for i in range(n_agents):
                for j in range(n_agents):
                    total_frob_norms[i, j] += frob_norms[i][j]
            total_timesteps += 1
            
            # Step environment
            next_states, rewards, dones, infos = env.step(actions)
            episode_reward += np.sum(rewards)
            step_count += 1
            
            states = next_states
            
            # Check if episode is done
            if dones[0] or step_count >= runner.args.episode_limit:
                break
        
        print(f"Episode completed with total reward: {episode_reward}")
        
        # Save Frobenius analysis GIF if requested
        if hasattr(args, 'save_gifs') and args.save_gifs and frames and frob_gif_dir:
            gif_path = os.path.join(frob_gif_dir, f'frobenius_episode_{episode}.gif')
            imageio.mimsave(gif_path, frames, duration=125)
            print(f"Saved Frobenius analysis GIF to {gif_path}")
    
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
                attack_impacts, logdir, args):
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
            'env_id': args.env_id,
            'model_dir': args.model_dir,
            'seed': args.seed
        }, f, indent=2)
    
    # Save Frobenius matrix as CSV
    frob_csv_path = os.path.join(logdir, 'frobenius_norms_matrix.csv')
    np.savetxt(frob_csv_path, avg_frob_norms, delimiter=',', fmt='%.6f')
    
    print(f"Saved comprehensive results to {results_file}")
    print(f"Saved Frobenius matrix to {frob_csv_path}")


def rollout_normal_episode(env, runner, seed, save_gif=False, gif_path=None):
    """
    Run a normal episode where all agents use their learned policies.
    
    Args:
        env: Environment instance
        runner: MAPPO runner with trained model
        seed: Random seed for episode
        save_gif: Whether to save frames for GIF creation
        gif_path: Path to save the GIF file
        
    Returns:
        float: Total episode reward
    """
    states = env.reset(seed=seed)
    total_reward = 0.0
    step_count = 0
    frames = []
    
    while True:
        # Capture frame for GIF if requested
        if save_gif:
            frame = env.render()
            frames.append(Image.fromarray(frame))
        
        # Get actions from all agents using learned policies
        actions = []
        for id in range(runner.args.N):
            action = runner.agent_n.select_action(states[id], id, evaluate=True)
            actions.append(action)
        
        # Step environment
        next_states, rewards, dones, info = env.step(actions)
        
        # Compute total reward
        total_reward += sum(rewards)
        step_count += 1
        
        states = next_states
        
        # Check if episode is done
        if all(dones):
            break
    
    # Save GIF if requested and frames were captured
    if save_gif and frames and gif_path:
        imageio.mimsave(gif_path, frames, duration=125)
        print(f"Saved normal episode GIF to {gif_path}")
    
    return total_reward


def rollout_attacked_episode(env, runner, attacked_agent_id, seed, save_gif=False, gif_path=None):
    """
    Run an episode where one specific agent is attacked (performs worst actions).
    
    Args:
        env: Environment instance
        runner: MAPPO runner with trained model
        attacked_agent_id: Index of the agent to attack
        seed: Random seed for episode
        save_gif: Whether to save frames for GIF creation
        gif_path: Path to save the GIF file
        
    Returns:
        float: Total episode reward
    """
    states = env.reset(seed=seed)
    total_reward = 0.0
    step_count = 0
    frames = []
    
    while True:
        # Capture frame for GIF if requested
        if save_gif:
            frame = env.render()
            frames.append(Image.fromarray(frame))
        
        # Get actions from all agents
        actions = []
        for id in range(runner.args.N):
            if id == attacked_agent_id:
                # Worst action attack: get distribution and choose worst action
                action, dist = runner.agent_n.select_action(states[id], id, evaluate=True, return_dist=True)
                action = torch.argmin(dist.probs).item()  # Choose worst action
            else:
                # Normal agent: use learned policy
                action = runner.agent_n.select_action(states[id], id, evaluate=True)
            actions.append(action)
        
        # Step environment
        next_states, rewards, dones, info = env.step(actions)
        
        # Compute total reward
        total_reward += sum(rewards)
        step_count += 1
        
        states = next_states
        
        # Check if episode is done
        if all(dones):
            break
    
    # Save GIF if requested and frames were captured
    if save_gif and frames and gif_path:
        imageio.mimsave(gif_path, frames, duration=125)
        
        # Save attack info alongside GIF
        info_path = gif_path.replace('.gif', '_info.txt')
        with open(info_path, 'w') as f:
            f.write(f"Attack Analysis Information:\n")
            f.write(f"Attacked agent: {attacked_agent_id}\n")
            f.write(f"Total reward: {total_reward:.3f}\n")
            f.write(f"Episode length: {step_count} steps\n")
        
        print(f"Saved attacked episode GIF to {gif_path}")
    
    return total_reward


def run_attack_analysis(env, runner, args, logdir):
    """
    Run attack vs no-attack analysis.
    
    Args:
        env: Environment instance
        runner: MAPPO runner with trained model
        args: Arguments containing configuration
        logdir: Log directory for saving files
        
    Returns:
        tuple: (normal_reward, attack_rewards) where attack_rewards is a list of rewards
               when each agent is attacked
    """
    n_agents = runner.args.N
    
    print(f"Running attack vs no-attack analysis...")
    print(f"Number of agents: {n_agents}")
    
    # Create GIF directory for attack analysis if save_gifs is enabled
    attack_gif_dir = None
    if hasattr(args, 'save_gifs') and args.save_gifs:
        attack_gif_dir = os.path.join(logdir, 'attack_analysis_gifs')
        os.makedirs(attack_gif_dir, exist_ok=True)
        print(f"Attack analysis GIFs will be saved to: {attack_gif_dir}")
    
    # Step 1: Run normal episode
    print("Running normal episode (no attacks)...")
    gif_path_normal = None
    if attack_gif_dir:
        gif_path_normal = os.path.join(attack_gif_dir, 'normal_episode.gif')
    
    normal_reward = rollout_normal_episode(
        env, runner, args.seed, 
        save_gif=(gif_path_normal is not None), 
        gif_path=gif_path_normal
    )
    print(f"Normal episode reward: {normal_reward:.3f}")
    
    # Step 2: Run episodes with each agent attacked
    attack_rewards = []
    for agent_id in range(n_agents):
        print(f"Running episode with Agent {agent_id} attacked...")
        
        gif_path_attack = None
        if attack_gif_dir:
            gif_path_attack = os.path.join(attack_gif_dir, f'agent_{agent_id}_attacked.gif')
        
        attacked_reward = rollout_attacked_episode(
            env, runner, agent_id, args.seed, 
            save_gif=(gif_path_attack is not None), 
            gif_path=gif_path_attack
        )
        attack_rewards.append(attacked_reward)
        print(f"Episode reward when Agent {agent_id} attacked: {attacked_reward:.3f}")
    
    return normal_reward, attack_rewards


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
    plt.title('Attack vs No-Attack Analysis (Worst Action Attack)', fontsize=14, fontweight='bold')
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


def main(runner: Runner_MAPPO_MPE, env, args):
    """Main execution function"""
    
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(
        os.getcwd(), 
        'runs', 
        f"{args.env_id}_shapley_frob_analysis", 
        f"{timestamp}_seed_{args.seed}"
    )
    os.makedirs(logdir, exist_ok=True)
    
    print(f"Results will be saved to: {logdir}")
    print(f"Environment: {args.env_id}")
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
    
    # Step 5: Run Attack Analysis
    print("\n" + "="*50)
    print("STEP 5: Attack vs No-Attack Analysis")
    print("="*50)
    normal_reward, attack_rewards = run_attack_analysis(env, runner, args, logdir)
    attack_impacts = plot_attack_analysis_barchart(normal_reward, attack_rewards, logdir, runner.args.N)
    
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
                attack_impacts, logdir, args)
    
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
    parser = argparse.ArgumentParser(description='Integrated Shapley Values and Frobenius Analysis for MAPPO in PettingZoo environments')
    
    # MAPPO hyperparameters (copied from existing scripts)
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
    
    # Required arguments
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save all output files")
    parser.add_argument("--env_id", type=str, required=True, help="Environment ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--discrete_action", type=bool, default=True, help="Whether the action space is discrete or continuous")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory to load the trained model")
    
    # Analysis specific parameters
    parser.add_argument("--save_gifs", action="store_true", help="Save GIFs of coalition rollouts and analysis episodes")
    
    args = parser.parse_args()
    
    # Create environment and runner
    env = make_env(env_name=args.env_id, discrete=True)
    runner = Runner_MAPPO_MPE(args, env_name=args.env_id, number=1, seed=args.seed)
    
    # Load trained model
    runner.agent_n.load_model_from_directory(args.model_dir)
    
    # Run analysis
    main(runner, env, args)
