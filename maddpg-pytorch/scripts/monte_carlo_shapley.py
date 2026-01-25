#!/usr/bin/env python3
"""
Monte Carlo Approximation of Shapley Values for Multi-Agent RL
Based on Algorithm 1: Monte Carlo approximation of Shapley values applied to a multi-agent RL context with shared payout

This script computes Shapley values using Monte Carlo sampling instead of exact computation,
making it feasible for environments with many agents.
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

# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent.parent))

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


def rollout_coalition(env, maddpg, coalition_mask, seed, save_gif=False, gif_path=None, agent_i=None, iteration=None):
    """
    Run a single episode rollout with the given coalition.
    Agents in the coalition use their learned policy, others use default action (0).
    
    Args:
        env: Environment instance
        maddpg: MADDPG model
        coalition_mask: Boolean list indicating which agents are in the coalition
        seed: Random seed for episode
        save_gif: Whether to save frames for GIF creation
        gif_path: Path to save the GIF file
        agent_i: Target agent for Shapley computation (for info file)
        iteration: Monte Carlo iteration number (for info file)
        
    Returns:
        float: Total episode reward (shared payout)
    """
    obs = env.reset(seed=seed)
    total_reward = 0.0
    step_count = 0
    frames = []
    
    while True:
        # Capture frame for GIF if requested
        if save_gif:
            frames.append(Image.fromarray(env.render()))
        
        # Get actions for all agents
        torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=False) 
                    for i in range(maddpg.nagents)]
        
        # Get policy actions
        torch_agent_actions = maddpg.step(torch_obs, explore=False)
        agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions]
        
        # Create action dict based on coalition membership
        actions_dict = {}
        for i, agent_name in enumerate(env.possible_agents):
            if coalition_mask[i]:
                # Agent in coalition: use learned policy
                if maddpg.discrete_action:
                    action = agent_actions[i].argmax()
                else:
                    action = agent_actions[i][0]
            else:
                # Agent not in coalition: use default action
                if maddpg.discrete_action:
                    action = 0
                else:
                    action = np.zeros_like(agent_actions[i][0])
            
            actions_dict[agent_name] = action
        
        # Step environment
        obs, rewards, dones, _ = env.step(actions_dict)
        
        # Compute shared reward (sum of all agent rewards)
        step_reward = sum(rewards.values()) if isinstance(rewards, dict) else np.sum(rewards)
        total_reward += step_reward
        step_count += 1
        
        # Check if episode is done
        if dones.all():
            break
    
    # Save GIF and coalition info if requested and frames were captured
    if save_gif and frames and gif_path:
        imageio.mimsave(gif_path, frames, duration=125)
        print(f"Saved coalition GIF to {gif_path}")
    
    return total_reward


def monte_carlo_shapley(config, maddpg, logdir, M=1000):
    """
    Compute Shapley values using Monte Carlo approximation.
    
    Args:
        config: Configuration object
        maddpg: MADDPG model
        logdir: Log directory for saving files
        M: Number of Monte Carlo iterations
        
    Returns:
        tuple: (final_shapley_values, running_means_history)
    """
    n_agents = maddpg.nagents
    
    # Initialize storage for marginal contributions
    marginal_contributions = [[] for _ in range(n_agents)]
    
    # Storage for running means to track convergence
    running_means_history = [[] for _ in range(n_agents)]
    
    print(f"Computing Shapley values using Monte Carlo approximation with {M} iterations...")
    print(f"Number of agents: {n_agents}")
    
    # Create GIF directory if save_gifs is enabled
    gif_dir = None
    if hasattr(config, 'save_gifs') and config.save_gifs:
        gif_dir = os.path.join(logdir, 'coalition_gifs')
        os.makedirs(gif_dir, exist_ok=True)
        print(f"Coalition GIFs will be saved to: {gif_dir}")

    # Create environment for this rollout
    env = create_environment(config, maddpg)
    
    # Track unique coalitions for GIF saving (save only once per unique coalition)
    saved_coalition_gifs = set()
    
    # Monte Carlo iterations
    for m in tqdm(range(M), desc="Monte Carlo iterations"):
        # For each agent i
        for agent_i in range(n_agents):
            # Get other agents (all except agent_i)
            other_agents = [j for j in range(n_agents) if j != agent_i]
            # Use a random seed for this iteration to add stochasticity
            current_seed = config.seed # + m*1000

            # Sample a random coalition
            coalition_with_i, coalition_without_i = sample_coalition(agent_i, other_agents)
            
            # Create unique identifiers for coalitions (without iteration number)
            coalition_with_str = ''.join(['1' if x else '0' for x in coalition_with_i])
            coalition_without_str = ''.join(['1' if x else '0' for x in coalition_without_i])
            
            # Generate GIF paths if saving and coalition hasn't been saved yet
            gif_path_with = None
            gif_path_without = None
            if hasattr(config, 'save_gifs') and config.save_gifs and gif_dir:
                coalition_with_key = f"agent{agent_i}_with_{coalition_with_str}"
                coalition_without_key = f"agent{agent_i}_without_{coalition_without_str}"
                
                # Only save GIF if we haven't seen this exact coalition before
                if coalition_with_key not in saved_coalition_gifs:
                    gif_path_with = os.path.join(gif_dir, f"{coalition_with_key}.gif")
                    saved_coalition_gifs.add(coalition_with_key)
                
                if coalition_without_key not in saved_coalition_gifs:
                    gif_path_without = os.path.join(gif_dir, f"{coalition_without_key}.gif")
                    saved_coalition_gifs.add(coalition_without_key)
            
            # Rollout with coalition including agent i
            r_plus_i = rollout_coalition(env, maddpg, coalition_with_i, current_seed, 
                                       save_gif=gif_path_with is not None, gif_path=gif_path_with,
                                       agent_i=agent_i, iteration=m)
            
            # Rollout with coalition excluding agent i
            r_minus_i = rollout_coalition(env, maddpg, coalition_without_i, current_seed,
                                        save_gif=gif_path_without is not None, gif_path=gif_path_without,
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
    if hasattr(config, 'save_gifs') and config.save_gifs and gif_dir:
        print(f"Saved {len(saved_coalition_gifs)} unique coalition GIFs to {gif_dir}")
    
    # Clean up environment
    env.close()
    
    return final_shapley_values, running_means_history


def plot_convergence(running_means_history, logdir, n_agents):
    """Plot the convergence of Shapley values over Monte Carlo iterations"""
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))
    
    for agent_i in range(n_agents):
        iterations = range(1, len(running_means_history[agent_i]) + 1)
        plt.plot(iterations, running_means_history[agent_i], 
                label=f'Agent {agent_i}', color=colors[agent_i], linewidth=2)
    
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


def plot_final_shapley_barchart(final_shapley_values, logdir, n_agents):
    """Plot final Shapley values as a bar chart"""
    plt.figure(figsize=(10, 6))
    
    agents = list(range(n_agents))
    colors = plt.cm.tab10(np.linspace(0, 1, n_agents))
    
    bars = plt.bar(agents, final_shapley_values, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on top of bars
    for bar, val in zip(bars, final_shapley_values):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01 * max(final_shapley_values),
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Agent ID')
    plt.ylabel('Shapley Value')
    plt.title('Final Shapley Values (Monte Carlo Approximation)')
    plt.xticks(agents)
    plt.grid(axis='y', alpha=0.3)
    
    # Add legend
    legend_labels = [f'Agent {i}' for i in range(n_agents)]
    legend_patches = [plt.matplotlib.patches.Patch(color=colors[i], label=legend_labels[i]) 
                     for i in range(n_agents)]
    plt.legend(handles=legend_patches, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    barchart_path = os.path.join(logdir, 'final_shapley_barchart.png')
    plt.savefig(barchart_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved final Shapley values bar chart to {barchart_path}")


def save_results(final_shapley_values, running_means_history, logdir, config, M):
    """Save results to files for later analysis"""
    
    # Save final Shapley values
    shapley_file = os.path.join(logdir, 'final_shapley_values.json')
    with open(shapley_file, 'w') as f:
        json.dump({
            'shapley_values': final_shapley_values,
            'num_agents': len(final_shapley_values),
            'monte_carlo_iterations': M,
            'env_id': config.env_id,
            'model_path': config.model_path
        }, f, indent=2)
    
    # Save convergence data
    convergence_file = os.path.join(logdir, 'convergence_data.json')
    with open(convergence_file, 'w') as f:
        json.dump({
            'running_means_history': running_means_history,
            'monte_carlo_iterations': M,
            'num_agents': len(final_shapley_values)
        }, f, indent=2)
    
    print(f"Saved results to {logdir}")


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
        f"{config.env_id}_monte_carlo_shapley", 
        f"{timestamp}_seed_{config.seed}"
    )
    os.makedirs(logdir, exist_ok=True)
    
    print(f"Results will be saved to: {logdir}")
    print(f"Environment: {config.env_id}")
    print(f"Number of agents: {maddpg.nagents}")
    print(f"Monte Carlo iterations: {config.monte_carlo_iterations}")
    
    # Set random seed for reproducibility
    if hasattr(config, 'seed') and config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        print(f"Set random seed to {config.seed}")
    
    # Compute Shapley values using Monte Carlo approximation
    final_shapley_values, running_means_history = monte_carlo_shapley(
        config, maddpg, logdir, M=config.monte_carlo_iterations
    )
    
    # Create visualizations
    plot_convergence(running_means_history, logdir, maddpg.nagents)
    plot_final_shapley_barchart(final_shapley_values, logdir, maddpg.nagents)
    
    # Save results
    save_results(final_shapley_values, running_means_history, logdir, config, config.monte_carlo_iterations)
    
    print("\nMonte Carlo Shapley value computation completed!")
    print(f"Final Shapley values: {final_shapley_values}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Monte Carlo Shapley Value Computation for Multi-Agent RL')
    
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_path", help="Path to trained MADDPG model directory")
    parser.add_argument("--monte_carlo_iterations", type=int, default=1000,
                        help="Number of Monte Carlo iterations (default: 1000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--save_gifs", action="store_true",
                        help="Save GIFs of coalition rollouts periodically")
    
    config = parser.parse_args()
    
    run(config)
