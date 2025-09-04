#!/usr/bin/env python3
"""
Pairwise Frobenius Norm Analysis for Multi-Agent RL
This script runs 1000 episodes and computes the average Frobenius norm of each agent pair,
then creates pie charts showing the influences of other agents on each agent.
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
import json
import math

from utils.make_env import make_env
from algorithms.maddpg import MADDPG
from utils.pettingzoo_wrapper import PettingZooWrapper
import pettingzoo.mpe as mpe
import pettingzoo.sisl as sisl
import pettingzoo.atari as atari
import supersuit

USE_CUDA = torch.cuda.is_available()
DEVICE = 'gpu' if USE_CUDA else 'cpu'
torch_device = torch.device("cuda" if USE_CUDA else "cpu")


def preprocess_env_atari(env):
    """Preprocess Atari environment as per OpenAI baselines"""
    env = supersuit.max_observation_v0(env, 2)
    env = supersuit.frame_skip_v0(env, 4)
    env = supersuit.resize_v1(env, 84, 84)
    env = supersuit.frame_stack_v1(env, 4)
    return env


def create_environment(config, maddpg):
    """Create a fresh environment instance"""
    try:
        env_func_ref = getattr(mpe, config.env_id)
        if config.env_id == "simple_spread_v3":
            env = env_func_ref.parallel_env(
                continuous_actions=not maddpg.discrete_action, 
                render_mode='rgb_array', 
                N=maddpg.nagents
            )
        else:
            env = env_func_ref.parallel_env(
                continuous_actions=not maddpg.discrete_action, 
                render_mode='rgb_array'
            )
    except:
        try:
            env_func_ref = getattr(sisl, config.env_id)
            if config.env_id == 'waterworld_v4':
                env = env_func_ref.parallel_env(n_pursuers=5, render_mode='rgb_array')
            else:
                env = env_func_ref.parallel_env(render_mode='rgb_array')
        except:
            env_func_ref = getattr(atari, config.env_id)
            env = env_func_ref.parallel_env(render_mode='rgb_array')
            env = preprocess_env_atari(env)
    
    env = PettingZooWrapper.wrap_env(env)
    return env


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
    torch_actions = [Variable(torch.Tensor([actions[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    vf_in = torch.cat((*torch_obs, *torch_actions), dim=1)

    N = maddpg.nagents
    results = [[0.0 for _ in range(N)] for _ in range(N)]

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

    return results


def run_single_episode(env, maddpg, seed=None):
    """
    Run a single episode and collect Frobenius norms at each step.
    
    Args:
        env: Environment instance
        maddpg: MADDPG model
        seed: Random seed for episode
        
    Returns:
        list: List of N x N Frobenius norm matrices, one per timestep
    """
    obs = env.reset(seed=seed)
    episode_frob_norms = []
    
    while True:
        # Get actions for all agents
        torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=False) 
                    for i in range(maddpg.nagents)]
        
        # Get policy actions
        torch_agent_actions = maddpg.step(torch_obs, explore=False)
        agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions]
        
        # Create action dict
        actions_dict = {}
        for i, agent_name in enumerate(env.possible_agents):
            if maddpg.discrete_action:
                action = agent_actions[i].argmax()
            else:
                action = agent_actions[i][0]
            actions_dict[agent_name] = action
        
        # Convert actions to list format for Frobenius computation
        actions_list = [actions_dict[agent_name] for agent_name in env.possible_agents]
        
        # Compute pairwise Frobenius norms
        frob_norms = compute_pairwise_frob_norms(maddpg, obs, actions_list, env.action_space)
        episode_frob_norms.append(frob_norms)
        
        # Step environment
        obs, rewards, dones, _ = env.step(actions_dict)
        
        # Check if episode is done
        if dones.all():
            break
    
    return episode_frob_norms


def run_analysis(config, maddpg, logdir, num_episodes=1000):
    """
    Run multiple episodes and compute average Frobenius norms.
    
    Args:
        config: Configuration object
        maddpg: MADDPG model
        logdir: Log directory for saving files
        num_episodes: Number of episodes to run
        
    Returns:
        numpy.ndarray: N x N matrix of average Frobenius norms
    """
    n_agents = maddpg.nagents
    
    # Initialize storage for Frobenius norms
    total_frob_norms = np.zeros((n_agents, n_agents))
    total_timesteps = 0
    
    print(f"Running {num_episodes} episodes to compute pairwise Frobenius norms...")
    print(f"Number of agents: {n_agents}")
    
    # Create environment for this analysis
    env = create_environment(config, maddpg)
    
    # Run episodes
    for episode in tqdm(range(num_episodes), desc="Episodes"):
        # Use different seed for each episode
        current_seed = config.seed + episode * 1000
        
        episode_frob_norms = run_single_episode(env, maddpg, current_seed)
        
        # Accumulate Frobenius norms
        for timestep_frobs in episode_frob_norms:
            for i in range(n_agents):
                for j in range(n_agents):
                    total_frob_norms[i, j] += timestep_frobs[i][j]
            total_timesteps += 1
    
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
    
    # Color map for consistent coloring (same as test_pettingzoo_detection_localq.py)
    cmap = plt.get_cmap('tab20')
    agent_colors = {i: cmap(i % 20) for i in range(n_agents)}
    
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
                           ha="center", va="center", color="white" if avg_frob_norms[i, j] > np.max(avg_frob_norms)/2 else "black")
    
    plt.title('Agent Influence Matrix\n(Average Frobenius Norms)', fontsize=14, fontweight='bold')
    plt.xlabel('Influencing Agent (j)', fontsize=12)
    plt.ylabel('Influenced Agent (i)', fontsize=12)
    
    # Save plot
    heatmap_path = os.path.join(logdir, 'agent_influence_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved influence heatmap to {heatmap_path}")


def save_results(avg_frob_norms, logdir, config, num_episodes):
    """Save results to files for later analysis"""
    
    # Save average Frobenius norms as CSV
    frob_csv_path = os.path.join(logdir, 'average_frobenius_norms.csv')
    np.savetxt(frob_csv_path, avg_frob_norms, delimiter=',', fmt='%.6f')
    
    # Save metadata as JSON
    metadata_file = os.path.join(logdir, 'analysis_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump({
            'num_episodes': num_episodes,
            'num_agents': avg_frob_norms.shape[0],
            'env_id': config.env_id,
            'model_path': config.model_path,
            'avg_frobenius_norms': avg_frob_norms.tolist()
        }, f, indent=2)
    
    print(f"Saved results to {logdir}")
    print(f"Average Frobenius norms saved to: {frob_csv_path}")
    print(f"Metadata saved to: {metadata_file}")


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
        f"{config.env_id}_frobenius_analysis", 
        f"{timestamp}_seed_{config.seed}"
    )
    os.makedirs(logdir, exist_ok=True)
    
    print(f"Results will be saved to: {logdir}")
    print(f"Environment: {config.env_id}")
    print(f"Number of agents: {maddpg.nagents}")
    print(f"Number of episodes: {config.num_episodes}")
    
    # Set random seed for reproducibility
    if hasattr(config, 'seed') and config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        print(f"Set random seed to {config.seed}")
    
    # Run analysis
    avg_frob_norms = run_analysis(config, maddpg, logdir, num_episodes=config.num_episodes)
    
    # Create visualizations
    create_influence_pie_charts(avg_frob_norms, logdir, maddpg.nagents)
    create_influence_heatmap(avg_frob_norms, logdir, maddpg.nagents)
    
    # Save results
    save_results(avg_frob_norms, logdir, config, config.num_episodes)
    
    print("\nPairwise Frobenius norm analysis completed!")
    print(f"Average influence matrix shape: {avg_frob_norms.shape}")
    print(f"Max influence: {np.max(avg_frob_norms):.6f}")
    print(f"Min influence: {np.min(avg_frob_norms):.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pairwise Frobenius Norm Analysis for Multi-Agent RL')
    
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_path", help="Path to trained MADDPG model directory")
    parser.add_argument("--num_episodes", type=int, default=1000,
                        help="Number of episodes to run (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    
    config = parser.parse_args()
    
    run(config)
