#!/usr/bin/env python3
"""
Multi-Seed Integrated Shapley Values, Frobenius Norm Analysis, Taylor Error Analysis, and Attack Analysis for Multi-Agent RL (MAPPO - PettingZoo)

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
- PettingZoo environment support

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
import imageio
from PIL import Image
import math

from make_env_pettingzoo import make_env
from MAPPO_MPE_main import Runner_MAPPO_MPE

USE_CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if USE_CUDA else 'cpu'
torch_device = torch.device("cuda" if USE_CUDA else "cpu")

# Fixed episode counts
SHAPLEY_EPISODES = 50
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


def compute_taylor_error_policy(runner: Runner_MAPPO_MPE, states, epsilon=0.01):
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


def rollout_normal_episode(env, runner, seed, save_gif=False, gif_path=None, collect_taylor_errors=False, epsilon=0.01):
    """
    Run a normal episode where all agents use their learned policies.
    
    Args:
        env: Environment instance
        runner: MAPPO runner with trained model
        seed: Random seed for episode
        save_gif: Whether to save frames for GIF creation
        gif_path: Path to save the GIF file
        collect_taylor_errors: Whether to collect Taylor error data at each timestep
        epsilon: Perturbation magnitude for Taylor error computation
        
    Returns:
        float or tuple: Total episode reward, or (total_reward, taylor_errors_per_timestep) if collect_taylor_errors=True
    """
    states = env.reset(seed=seed)
    total_reward = 0.0
    step_count = 0
    frames = []
    taylor_errors_per_timestep = [] if collect_taylor_errors else None
    
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
        
        # Collect Taylor error data if requested
        if collect_taylor_errors:
            nagents = runner.args.N
            taylor_errors = compute_taylor_error_policy(runner, states, epsilon)
            taylor_errors_per_timestep.append(taylor_errors)
        
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
    
    if collect_taylor_errors:
        return total_reward, taylor_errors_per_timestep
    else:
        return total_reward


def rollout_attacked_episode(env, runner, attacked_agent_id, seed, save_gif=False, gif_path=None, collect_taylor_errors=False, epsilon=0.01):
    """
    Run an episode where one specific agent is attacked (performs worst actions).
    
    Args:
        env: Environment instance
        runner: MAPPO runner with trained model
        attacked_agent_id: Index of the agent to attack
        seed: Random seed for episode
        save_gif: Whether to save frames for GIF creation
        gif_path: Path to save the GIF file
        collect_taylor_errors: Whether to collect Taylor error data at each timestep
        epsilon: Perturbation magnitude for Taylor error computation
        
    Returns:
        float or tuple: Total episode reward, or (total_reward, taylor_errors_per_timestep) if collect_taylor_errors=True
    """
    states = env.reset(seed=seed)
    total_reward = 0.0
    step_count = 0
    frames = []
    taylor_errors_per_timestep = [] if collect_taylor_errors else None
    
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
                # action = torch.argmin(dist.probs).item()  # Choose worst action
                
                # Get the third best action
                sorted_probs, sorted_indices = torch.sort(dist.probs.squeeze(), descending=True)
                action = sorted_indices[2].item()
            else:
                # Normal agents use their learned policy
                action = runner.agent_n.select_action(states[id], id, evaluate=True)
            actions.append(action)
        
        # Collect Taylor error data if requested
        if collect_taylor_errors:
            nagents = runner.args.N
            taylor_errors = compute_taylor_error_policy(runner, states, epsilon)
            taylor_errors_per_timestep.append(taylor_errors)
        
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
            f.write(f"Attack Information:\n")
            f.write(f"Attacked agent: {attacked_agent_id}\n")
            f.write(f"Attack type: Worst action attack\n")
            f.write(f"Total reward: {total_reward:.3f}\n")
            f.write(f"Episode length: {step_count} steps\n")
        
        print(f"Saved attacked episode GIF to {gif_path}")
    
    if collect_taylor_errors:
        return total_reward, taylor_errors_per_timestep
    else:
        return total_reward


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
        tuple: (normal_reward, attack_rewards) if collect_taylor_errors=False
               (normal_reward, attack_rewards, normal_taylor_errors, attack_taylor_errors) if collect_taylor_errors=True
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
    
    if collect_taylor_errors:
        normal_reward, normal_taylor_errors = rollout_normal_episode(
            env, runner, args.seed, 
            save_gif=(gif_path_normal is not None), 
            gif_path=gif_path_normal,
            collect_taylor_errors=True,
            epsilon=epsilon
        )
    else:
        normal_reward = rollout_normal_episode(
            env, runner, args.seed, 
            save_gif=(gif_path_normal is not None), 
            gif_path=gif_path_normal
        )
        normal_taylor_errors = None
    
    print(f"Normal episode reward: {normal_reward:.3f}")
    
    # Step 2: Run episodes with each agent attacked
    attack_rewards = []
    attack_taylor_errors = [] if collect_taylor_errors else None
    
    for agent_id in range(n_agents):
        print(f"Running episode with Agent {agent_id} attacked...")
        
        gif_path_attack = None
        if attack_gif_dir:
            gif_path_attack = os.path.join(attack_gif_dir, f'agent_{agent_id}_attacked.gif')
        
        if collect_taylor_errors:
            attacked_reward, attacked_taylor_errors_episode = rollout_attacked_episode(
                env, runner, agent_id, args.seed, 
                save_gif=(gif_path_attack is not None), 
                gif_path=gif_path_attack,
                collect_taylor_errors=True,
                epsilon=epsilon
            )
            attack_taylor_errors.append(attacked_taylor_errors_episode)
        else:
            attacked_reward = rollout_attacked_episode(
                env, runner, agent_id, args.seed, 
                save_gif=(gif_path_attack is not None), 
                gif_path=gif_path_attack
            )
        
        attack_rewards.append(attacked_reward)
        print(f"Episode reward when Agent {agent_id} attacked: {attacked_reward:.3f}")
    
    if collect_taylor_errors:
        return normal_reward, attack_rewards, normal_taylor_errors, attack_taylor_errors
    else:
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
    
    # Hide unused subplots
    for j in range(total_plots, len(axes)):
        axes[j].set_visible(False)
    
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
    
    y_min = min(all_values) - 0.1 * abs(min(all_values))
    y_max = max(all_values) + 0.1 * abs(max(all_values))
    
    # Plot 1: Normal scenario (all agents)
    ax = axes[0]
    agents = list(range(n_agents))
    colors = [agent_colors[i] for i in range(n_agents)]
    
    bars = ax.bar(agents, mean_normal_taylor,
                  color=colors, alpha=0.8, edgecolor='black', capsize=5)
    
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
                      color=colors_attack, alpha=0.8, edgecolor='black', capsize=5)
        
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
    
    # Step 5: Run Attack Analysis and Taylor Error Analysis
    print("\n" + "="*50)
    print("STEP 5: Attack vs No-Attack Analysis and Taylor Error Analysis")
    print("="*50)
    
    # Run attack analysis with Taylor error collection
    results = run_attack_analysis(env, runner, args, logdir, collect_taylor_errors=True, epsilon=0.01)
    normal_reward, attack_rewards, normal_taylor_errors, attack_taylor_errors = results
    attack_impacts = plot_attack_analysis_barchart(normal_reward, attack_rewards, logdir, runner.args.N)
    
    # Process Taylor error data for visualization
    if normal_taylor_errors and attack_taylor_errors:
        # Compute mean Taylor errors across timesteps for normal scenario
        normal_taylor_errors_array = np.array(normal_taylor_errors)  # (timesteps, agents)
        normal_mean_errors = np.mean(normal_taylor_errors_array, axis=0)  # (agents,)
        
        # Compute mean Taylor errors across timesteps for each attack scenario
        attack_mean_errors = []
        for attack_scenario in attack_taylor_errors:
            attack_scenario_array = np.array(attack_scenario)  # (timesteps, agents)
            attack_mean = np.mean(attack_scenario_array, axis=0)  # (agents,)
            attack_mean_errors.append(attack_mean)
        
        # Create Taylor error visualization
        plot_taylor_error_barchart(normal_mean_errors, attack_mean_errors, logdir, runner.args.N)
        
        print(f"Normal scenario mean Taylor errors: {[f'{val:.4f}' for val in normal_mean_errors]}")
        for i, attack_mean in enumerate(attack_mean_errors):
            print(f"Agent {i} attacked mean Taylor errors: {[f'{val:.4f}' for val in attack_mean]}")
    
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


def plot_aggregated_shapley_barchart(mean_shapley_values, std_shapley_values, logdir, n_agents):
    """Plot mean Shapley values as a bar chart"""
    plt.figure(figsize=(10, 6))
    
    agents = list(range(n_agents))
    # Get consistent color palette
    agent_colors = get_agent_colors(n_agents)
    colors = [agent_colors[i] for i in range(n_agents)]
    
    bars = plt.bar(agents, mean_shapley_values,
                   color=colors, alpha=0.8, edgecolor='black', capsize=5)
    
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
                   color=colors, alpha=0.8, edgecolor='black', capsize=5)
    
    # Add value labels on top of bars
    for bar, val in zip(bars, mean_outbound_influence):
        height = bar.get_height()
        if max(mean_outbound_influence) > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(mean_outbound_influence),
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            plt.text(bar.get_x() + bar.get_width()/2., 0.001,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Agent ID')
    plt.ylabel('Mean Outbound Influence Score (I_i^out)')
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


def plot_aggregated_cascade_risk_barchart(mean_cascade_risk, std_cascade_risk, logdir, n_agents):
    """Plot mean cascade risk index as a bar chart"""
    plt.figure(figsize=(10, 6))
    
    agents = list(range(n_agents))
    # Get consistent color palette  
    agent_colors = get_agent_colors(n_agents)
    colors = [agent_colors[i] for i in range(n_agents)]
    
    bars = plt.bar(agents, mean_cascade_risk,
                   yerr=std_cascade_risk, color=colors, alpha=0.8, 
                   edgecolor='black', capsize=5)
    
    # Add value labels on top of bars
    max_risk = max(mean_cascade_risk)
    for bar, val in zip(bars, mean_cascade_risk):
        height = bar.get_height()
        if max_risk > 0:
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max_risk,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        else:
            plt.text(bar.get_x() + bar.get_width()/2., 0.001,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Agent ID')
    plt.ylabel('Mean Cascade Risk Index (CRI_i)')
    plt.title('Mean Cascade Risk Index Across Multiple Seeds')
    plt.xticks(agents)
    plt.grid(axis='y', alpha=0.3)
    
    # Add legend
    legend_labels = [f'Agent {i}' for i in range(n_agents)]
    legend_patches = [plt.matplotlib.patches.Patch(color=colors[i], label=legend_labels[i]) 
                     for i in range(n_agents)]
    plt.legend(handles=legend_patches, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    barchart_path = os.path.join(logdir, 'mean_cascade_risk_barchart.png')
    plt.savefig(barchart_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved mean cascade risk bar chart to {barchart_path}")


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
        
        # Get influence values for agent i (how much others influence agent i)
        influences = []
        labels = []
        colors_for_agent = []
        
        for j in range(n_agents):
            influences.append(mean_frob_norms[i][j])
            # labels.append(f'Agent {j}')
            colors_for_agent.append(agent_colors[j])
        
        # Only create pie chart if there are non-zero influences
        if sum(influences) > 0:
            wedges, texts, autotexts = ax.pie(influences, colors=colors_for_agent,
                                             autopct='%1.1f%%', startangle=90)
            # Make percentage text bold
            for autotext in autotexts:
                autotext.set_fontweight('bold')
                autotext.set_fontsize(8)
        else:
            ax.text(0.5, 0.5, 'No Influence', ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title(f'Mean Influence on Agent {i}', fontweight='bold', fontsize=10)
        ax.axis('equal')
    
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
    
    plt.suptitle('Mean Agent Influence Analysis (Pairwise Frobenius Norms)', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Leave space for legend at bottom
    
    # Save plot
    pie_chart_path = os.path.join(logdir, 'mean_agent_influence_pie_charts.png')
    plt.savefig(pie_chart_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved mean influence pie charts to {pie_chart_path}")


def plot_shapley_reward_matching_barchart(mean_position_accuracies, std_position_accuracies, logdir, n_agents):
    """Plot Shapley vs Reward Drop matching accuracy for each rank/position"""
    plt.figure(figsize=(10, 6))
    
    positions = list(range(n_agents))
    position_labels = [f'Rank {i+1}' for i in range(n_agents)]
    
    # Use consistent color palette
    agent_colors = get_agent_colors(n_agents)
    colors = [agent_colors[i] for i in range(n_agents)]
    
    bars = plt.bar(positions, mean_position_accuracies,
                   color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on top of bars
    for bar, val in zip(bars, mean_position_accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.xlabel('Ranking Position', fontsize=12)
    plt.ylabel('Matching Accuracy', fontsize=12)
    plt.title('Shapley vs Reward Drop - Position-wise Matching Accuracy', fontsize=14, fontweight='bold')
    plt.xticks(positions, position_labels)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    
    plt.legend()
    
    plt.tight_layout()
    
    # Save plot
    barchart_path = os.path.join(logdir, 'shapley_reward_matching_accuracy_barchart.png')
    plt.savefig(barchart_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved Shapley-reward matching accuracy bar chart to {barchart_path}")


def plot_outbound_reward_matching_barchart(mean_position_accuracies, std_position_accuracies, logdir, n_agents):
    """Plot Outbound Influence vs Reward Drop matching accuracy for each rank/position"""
    plt.figure(figsize=(10, 6))
    
    positions = list(range(n_agents))
    position_labels = [f'Rank {i+1}' for i in range(n_agents)]
    
    # Use consistent color palette
    agent_colors = get_agent_colors(n_agents)
    colors = [agent_colors[i] for i in range(n_agents)]
    
    bars = plt.bar(positions, mean_position_accuracies,
                   color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on top of bars
    for bar, val in zip(bars, mean_position_accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.xlabel('Ranking Position', fontsize=12)
    plt.ylabel('Matching Accuracy', fontsize=12)
    plt.title('Outbound Influence vs Reward Drop - Position-wise Matching Accuracy', fontsize=14, fontweight='bold')
    plt.xticks(positions, position_labels)
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    
    plt.legend()
    
    plt.tight_layout()
    
    # Save plot
    barchart_path = os.path.join(logdir, 'outbound_reward_matching_accuracy_barchart.png')
    plt.savefig(barchart_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved outbound-reward matching accuracy bar chart to {barchart_path}")


def plot_matching_accuracy_barchart(mean_shapley_accuracy, std_shapley_accuracy, 
                                   mean_outbound_accuracy, std_outbound_accuracy, logdir):
    """Plot overall matching accuracy results as a bar chart"""
    plt.figure(figsize=(10, 6))
    
    categories = ['Shapley vs Reward Drop', 'Outbound Influence vs Reward Drop']
    accuracies = [mean_shapley_accuracy, mean_outbound_accuracy]
    errors = [std_shapley_accuracy, std_outbound_accuracy]
    
    colors = ['blue', 'orange']
    
    bars = plt.bar(categories, accuracies, color=colors, alpha=0.7, 
                   edgecolor='black')
    
    # Add value labels on top of bars
    for bar, val in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Ranking Comparison', fontsize=12)
    plt.ylabel('Mean Matching Accuracy', fontsize=12)
    plt.title('Overall Ranking Matching Accuracy Across Multiple Seeds', fontsize=14, fontweight='bold')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    
    plt.legend()
    
    plt.tight_layout()
    
    # Save plot
    barchart_path = os.path.join(logdir, 'overall_matching_accuracy_barchart.png')
    plt.savefig(barchart_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved overall matching accuracy bar chart to {barchart_path}")


def run_multi_seed_analysis(args):
    """Main execution function for multi-seed analysis"""
    
    # Create log directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(
        os.getcwd(), 
        'runs', 
        f"{args.env_id}_multi_seed_analysis", 
        f"{timestamp}_iterations_{args.max_iterations}"
    )
    os.makedirs(logdir, exist_ok=True)
    
    print(f"Results will be saved to: {logdir}")
    print(f"Environment: {args.env_id}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Shapley episodes per iteration: {SHAPLEY_EPISODES}")
    print(f"Frobenius episodes per iteration: {FROBENIUS_EPISODES}")
    
    # Initialize storage for aggregated results
    all_shapley_values = []
    all_frob_norms = []
    all_outbound_influence = []
    all_cascade_risk = []
    all_matching_results = []
    all_normal_taylor_errors = []
    all_attack_taylor_errors = []
    
    # Run analysis for each seed
    for iteration in tqdm(range(args.max_iterations), desc="Running multi-seed analysis"):
        seed = iteration
        print(f"\n" + "="*60)
        print(f"ITERATION {iteration + 1}/{args.max_iterations} (Seed: {seed})")
        print("="*60)
        
        # Create environment and runner for this iteration
        env = make_env(env_name=args.env_id, discrete=True)
        runner = Runner_MAPPO_MPE(args, env_name=args.env_id, number=1, seed=seed)
        
        # Load trained model
        runner.agent_n.load_model_from_directory(args.model_dir)
        
        # Set random seed for this iteration
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Create temporary args for this iteration
        temp_args = argparse.Namespace(**vars(args))
        temp_args.seed = seed
        temp_args.save_gifs = False  # Disable GIFs for multi-seed analysis
        
        # Get number of agents from the runner
        n_agents = runner.args.N
        
        # Run single iteration analysis (without plotting) with Taylor error collection
        shapley_values, _ = monte_carlo_shapley(env, runner, temp_args, logdir)
        avg_frob_norms = run_frobenius_analysis(env, runner, temp_args, logdir)
        outbound_influence = compute_outbound_influence(avg_frob_norms)
        
        # Run attack analysis with Taylor error collection
        results = run_attack_analysis(env, runner, temp_args, logdir, collect_taylor_errors=True, epsilon=0.01)
        normal_reward, attack_rewards, normal_taylor_errors, attack_taylor_errors = results
        
        # Compute cascade risk index
        cascade_risk = compute_cascade_risk_index(shapley_values, outbound_influence)
        
        # Process Taylor error data for this iteration
        normal_taylor_errors_array = np.array(normal_taylor_errors)  # (timesteps, agents)
        normal_mean_errors = np.mean(normal_taylor_errors_array, axis=0)  # (agents,)
        
        attack_mean_errors = []
        for attack_scenario in attack_taylor_errors:
            attack_scenario_array = np.array(attack_scenario)  # (timesteps, agents) 
            attack_mean = np.mean(attack_scenario_array, axis=0)  # (agents,)
            attack_mean_errors.append(attack_mean)
        
        # Compute rankings and matching accuracies
        matching_results = compute_ranking_and_matching(
            shapley_values, outbound_influence, normal_reward, attack_rewards
        )
        
        # Store results
        all_shapley_values.append(shapley_values)
        all_frob_norms.append(avg_frob_norms)
        all_outbound_influence.append(outbound_influence)
        all_cascade_risk.append(cascade_risk)
        all_matching_results.append(matching_results)
        all_normal_taylor_errors.append(normal_mean_errors)
        all_attack_taylor_errors.append(attack_mean_errors)
        
        print(f"Iteration {iteration + 1} completed successfully")
        print(f"  Shapley position matches: {matching_results['shapley_vs_reward_position_matches']}")
        print(f"  Outbound position matches: {matching_results['outbound_vs_reward_position_matches']}")
        
        # Clean up environment
        env.close()
    
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
    cascade_risk_array = np.array(all_cascade_risk)  # (iterations, agents)
    
    # Compute means and standard deviations
    mean_shapley = np.mean(shapley_array, axis=0)
    std_shapley = np.std(shapley_array, axis=0)
    
    mean_frob_norms = np.mean(frob_array, axis=0)
    std_frob_norms = np.std(frob_array, axis=0)
    
    mean_outbound = np.mean(outbound_array, axis=0)
    std_outbound = np.std(outbound_array, axis=0)
    
    mean_cascade_risk = np.mean(cascade_risk_array, axis=0)
    std_cascade_risk = np.std(cascade_risk_array, axis=0)
    
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
    
    # Compute Taylor error aggregated statistics
    normal_taylor_array = np.array(all_normal_taylor_errors)  # (iterations, agents)
    attack_taylor_array = np.array(all_attack_taylor_errors)  # (iterations, attacked_agents, agents)
    
    mean_normal_taylor = np.mean(normal_taylor_array, axis=0)  # (agents,)
    std_normal_taylor = np.std(normal_taylor_array, axis=0)   # (agents,)
    
    mean_attack_taylor = np.mean(attack_taylor_array, axis=0)  # (attacked_agents, agents)
    std_attack_taylor = np.std(attack_taylor_array, axis=0)    # (attacked_agents, agents)
    
    # Create visualizations
    print("\n" + "="*60)
    print("CREATING AGGREGATED VISUALIZATIONS")
    print("="*60)
    
    plot_aggregated_shapley_barchart(mean_shapley, std_shapley, logdir, n_agents)
    plot_aggregated_outbound_influence_barchart(mean_outbound, std_outbound, logdir, n_agents)
    plot_aggregated_cascade_risk_barchart(mean_cascade_risk, std_cascade_risk, logdir, n_agents)
    create_aggregated_influence_pie_charts(mean_frob_norms, logdir, n_agents)
    
    # Plot Taylor error aggregated analysis
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
                'mean_cascade_risk': mean_cascade_risk.tolist(),
                'std_cascade_risk': std_cascade_risk.tolist(),
                'mean_frob_norms': mean_frob_norms.tolist(),
                'std_frob_norms': std_frob_norms.tolist(),
                'mean_overall_shapley_accuracy': mean_overall_shapley_accuracy,
                'std_overall_shapley_accuracy': std_overall_shapley_accuracy,
                'mean_overall_outbound_accuracy': mean_overall_outbound_accuracy,
                'std_overall_outbound_accuracy': std_overall_outbound_accuracy,
                'mean_shapley_position_accuracy': mean_shapley_position_accuracy.tolist(),
                'std_shapley_position_accuracy': std_shapley_position_accuracy.tolist(),
                'mean_outbound_position_accuracy': mean_outbound_position_accuracy.tolist(),
                'std_outbound_position_accuracy': std_outbound_position_accuracy.tolist()
            },
            'raw_data': {
                'all_shapley_values': shapley_array.tolist(),
                'all_outbound_influence': outbound_array.tolist(),
                'all_cascade_risk': cascade_risk_array.tolist(),
                'all_frob_norms': frob_array.tolist(),
                'all_matching_results': all_matching_results
            },
            'configuration': {
                'max_iterations': args.max_iterations,
                'num_agents': n_agents,
                'shapley_episodes': SHAPLEY_EPISODES,
                'frobenius_episodes': FROBENIUS_EPISODES,
                'env_id': args.env_id,
                'model_dir': args.model_dir
            }
        }, f, indent=2)
    
    print(f"Saved comprehensive results to {results_file}")
    
    # Final summary
    print("\n" + "="*70)
    print("MULTI-SEED ANALYSIS COMPLETED!")
    print("="*70)
    print(f"Successfully completed {len(all_shapley_values)}/{args.max_iterations} iterations")
    print(f"Mean Shapley values: {[f'{val:.3f}±{std:.3f}' for val, std in zip(mean_shapley, std_shapley)]}")
    print(f"Mean outbound influence: {[f'{val:.3f}±{std:.3f}' for val, std in zip(mean_outbound, std_outbound)]}")
    print(f"Mean cascade risk index: {[f'{val:.3f}±{std:.3f}' for val, std in zip(mean_cascade_risk, std_cascade_risk)]}")
    print(f"Mean normal Taylor errors: {[f'{val:.4f}±{std:.4f}' for val, std in zip(mean_normal_taylor, std_normal_taylor)]}")
    print(f"Overall shapley-reward matching accuracy: {mean_overall_shapley_accuracy:.3f}±{std_overall_shapley_accuracy:.3f}")
    print(f"Overall outbound-reward matching accuracy: {mean_overall_outbound_accuracy:.3f}±{std_overall_outbound_accuracy:.3f}")
    print(f"Position-wise shapley matching: {[f'{val:.3f}±{std:.3f}' for val, std in zip(mean_shapley_position_accuracy, std_shapley_position_accuracy)]}")
    print(f"Position-wise outbound matching: {[f'{val:.3f}±{std:.3f}' for val, std in zip(mean_outbound_position_accuracy, std_outbound_position_accuracy)]}")
    print(f"\nResults saved to: {logdir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Seed Integrated Shapley Values and Frobenius Analysis for MAPPO in PettingZoo environments')
    
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
    parser.add_argument("--discrete_action", type=bool, default=True, help="Whether the action space is discrete or continuous")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory to load the trained model")
    
    # Analysis specific parameters
    parser.add_argument("--save_gifs", action="store_true", help="Save GIFs of coalition rollouts and analysis episodes")
    
    # Multi-seed analysis parameters
    parser.add_argument("--max_iterations", type=int, default=None,
                        help="Number of iterations (seeds) to run for multi-seed analysis. If not specified, runs single-seed analysis.")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for single-seed analysis (ignored if --max_iterations is specified)")
    
    args = parser.parse_args()
    
    # Decide whether to run single-seed or multi-seed analysis
    if args.max_iterations is not None:
        print("Running multi-seed analysis...")
        run_multi_seed_analysis(args)
    else:
        print("Running single-seed analysis...")
        # Create environment and runner
        env = make_env(env_name=args.env_id, discrete=True)
        runner = Runner_MAPPO_MPE(args, env_name=args.env_id, number=1, seed=args.seed)
        
        # Load trained model
        runner.agent_n.load_model_from_directory(args.model_dir)
        
        # Run single-seed analysis
        main(runner, env, args)
