#!/usr/bin/env python3
"""
Influence Chain Mining Script

This script extracts influence chains from pairwise Frobenius norms using frequent itemset mining.
It runs multiple episodes, tracks strongest outgoing influence edges for each agent at each timestep,
builds influence chains by linking edges, and uses frequent itemset mining to discover dominant
influence pathways across episodes.

Author: GitHub Copilot
"""

import argparse
import torch
import numpy as np
import os
import csv
import math
from datetime import datetime
from pathlib import Path
from torch.autograd import Variable
from utils.make_env import make_env
from algorithms.maddpg import MADDPG
from utils.pettingzoo_wrapper import PettingZooWrapper
import pettingzoo.mpe as mpe
import pettingzoo.sisl as sisl
import pettingzoo.atari as atari
import supersuit
from collections import deque, defaultdict, Counter
from tqdm import tqdm
from itertools import combinations, chain
import pickle

# Try to import plotting libraries
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import networkx as nx
    from matplotlib.patches import Patch
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: plotting libraries not available. Plots will be skipped.")

USE_CUDA = torch.cuda.is_available()
DEVICE = 'gpu' if USE_CUDA else 'cpu'
torch_device = torch.device("cuda" if USE_CUDA else "cpu")

def preprocess_env_atari(env):
    """Preprocess Atari environments"""
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
    Returns an N x N list where entry [i][j] approximates || ∂²v_i / (∂obs_i ∂obs_j) ||_F.
    """
    # Convert discrete actions to one-hot encoding
    if maddpg.discrete_action:
        one_hot_actions = []
        for i, action in enumerate(actions):
            one_hot = np.zeros(action_spaces[i].n)
            one_hot[action] = 1.0
            one_hot_actions.append(one_hot)
        actions = one_hot_actions

    torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=True) 
                 for i in range(maddpg.nagents)]
    actions = [Variable(torch.Tensor([actions[i]]).to(torch_device), requires_grad=True) 
               for i in range(maddpg.nagents)]
    vf_in = torch.cat((*torch_obs, *actions), dim=1)

    N = maddpg.nagents
    results = [[0.0 for _ in range(N)] for _ in range(N)]

    for i in range(N):
        critic_val = maddpg.agents[i].critic(vf_in).mean()
        grad_i = torch.autograd.grad(critic_val, torch_obs[i], create_graph=True, retain_graph=True)[0]

        for j in range(N):
            hessian_matrix = []
            for k in range(grad_i.shape[1]):
                second_grad = torch.autograd.grad(
                    grad_i[0, k],
                    torch_obs[j],
                    retain_graph=True,
                    allow_unused=True
                )[0]
                if second_grad is None:
                    second_grad = torch.zeros_like(torch_obs[j])
                hessian_matrix.append(second_grad.flatten())

            H = torch.stack(hessian_matrix) if len(hessian_matrix) > 0 else torch.zeros(1, 1)
            results[i][j] = H.norm(p='fro').item()

    return results

def extract_strongest_edges(frob_matrix):
    """
    For each agent i, find the agent j (j != i) with highest influence on i.
    Returns list of (target_agent, source_agent, strength) tuples.
    """
    N = len(frob_matrix)
    edges = []
    
    for i in range(N):  # target agent
        max_influence = -1
        max_source = -1
        
        for j in range(N):  # source agent
            if i != j and frob_matrix[i][j] > max_influence:
                max_influence = frob_matrix[i][j]
                max_source = j
        
        if max_source != -1:
            edges.append((i, max_source, max_influence))
    
    return edges

def build_influence_chains(episode_edges, max_chain_length=3, max_lookahead=2, decay_factor=0.9):
    """
    Build influence chains from timestep edges by linking where target becomes source.
    
    Args:
        episode_edges: List of timestep edges, each timestep contains [(target, source, strength), ...]
        max_chain_length: Maximum length of chains to extract
        max_lookahead: Maximum timesteps ahead to look for next edge (1=immediate next, 2=next 2 timesteps)
        decay_factor: Decay factor for strength when timesteps are skipped
    
    Returns:
        List of chains, each chain is (chain_tuple, avg_strength, support_timesteps)
    """
    if not episode_edges or max_chain_length < 2:
        return []
    
    chains = []
    
    # Create a mapping from (timestep, source_node) to list of (target, strength) for fast lookup
    edge_lookup = defaultdict(list)
    for ts, edges in enumerate(episode_edges):
        for target, source, strength in edges:
            edge_lookup[(ts, source)].append((target, strength))
    
    # Start chains from each edge in each timestep
    for start_ts in range(len(episode_edges)):
        for target, source, strength in episode_edges[start_ts]:
            # Use DFS to build chains starting from this edge
            chain = _build_chain_dfs(
                current_node=target,
                current_ts=start_ts,
                chain_nodes=[source, target],
                chain_strengths=[strength],
                chain_timesteps=[start_ts],
                edge_lookup=edge_lookup,
                max_chain_length=max_chain_length,
                max_lookahead=max_lookahead,  # Updated parameter name
                decay_factor=decay_factor,
                max_ts=len(episode_edges) - 1
            )
            chains.extend(chain)
    
    return chains


def _build_chain_dfs(current_node, current_ts, chain_nodes, chain_strengths, chain_timesteps,
                     edge_lookup, max_chain_length, max_lookahead, decay_factor, max_ts):
    """
    Helper function to build chains using depth-first search.
    
    Args:
        max_lookahead: Maximum number of timesteps ahead to look for next edge
                      (1 = only next timestep, 2 = next 2 timesteps, etc.)
    
    Returns list of completed chains found from this starting point.
    """
    completed_chains = []
    
    # Add current chain if it's long enough (>= 2 nodes)
    if len(chain_nodes) >= 2:
        chain_tuple = tuple(chain_nodes)
        avg_strength = np.mean(chain_strengths)
        completed_chains.append((chain_tuple, avg_strength, chain_timesteps.copy()))
    
    # Stop if we've reached maximum chain length
    if len(chain_nodes) >= max_chain_length:
        return completed_chains
    
    # Look for continuation edges within the lookahead window
    search_end = min(max_ts + 1, current_ts + max_lookahead + 1)
    
    for next_ts in range(current_ts + 1, search_end):
        # Check if current_node has outgoing edges at next_ts
        if (next_ts, current_node) not in edge_lookup:
            continue
        for target, strength in edge_lookup[(next_ts, current_node)]:
            # Calculate how many timesteps we skipped
            timesteps_skipped = next_ts - current_ts - 1
            decayed_strength = strength * (decay_factor ** timesteps_skipped)
            
            # Recursively extend the chain
            extended_chains = _build_chain_dfs(
                current_node=target,
                current_ts=next_ts,
                chain_nodes=chain_nodes + [target],
                chain_strengths=chain_strengths + [decayed_strength],
                chain_timesteps=chain_timesteps + [next_ts],
                edge_lookup=edge_lookup,
                max_chain_length=max_chain_length,
                max_lookahead=max_lookahead,  # Updated parameter name
                decay_factor=decay_factor,
                max_ts=max_ts
            )
            completed_chains.extend(extended_chains)
    
    return completed_chains

def frequent_itemset_mining(all_chains, min_support=2, min_strength_threshold=0.001):
    """
    Apply frequent itemset mining to discover dominant influence patterns.
    
    Args:
        all_chains: List of (chain_tuple, strength, episode_id) from all episodes
        min_support: Minimum number of episodes a chain must appear in
        min_strength_threshold: Minimum average strength threshold
    
    Returns:
        Dictionary of frequent chains with their statistics
    """
    # Group chains by pattern and collect statistics
    chain_stats = defaultdict(lambda: {
        'episodes': set(),
        'strengths': [],
        'total_occurrences': 0
    })
    
    for chain_tuple, strength, episode_id in all_chains:
        if strength >= min_strength_threshold:
            chain_stats[chain_tuple]['episodes'].add(episode_id)
            chain_stats[chain_tuple]['strengths'].append(strength)
            chain_stats[chain_tuple]['total_occurrences'] += 1
    
    # Filter by minimum support and compute final statistics
    frequent_chains = {}
    for chain_tuple, stats in chain_stats.items():
        support = len(stats['episodes'])
        if support >= min_support:
            frequent_chains[chain_tuple] = {
                'support': support,
                'avg_strength': np.mean(stats['strengths']),
                'std_strength': np.std(stats['strengths']),
                'total_occurrences': stats['total_occurrences'],
                'episodes': list(stats['episodes'])
            }
    
    return frequent_chains

def run_single_episode(env, maddpg, episode_id, seed=None):
    """Run a single episode and extract influence edges"""
    obs = env.reset(seed=seed) if seed else env.reset()
    
    episode_edges = []
    timestep = 0
    
    while True:
        torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=False) 
                     for i in range(maddpg.nagents)]
        torch_agent_actions = maddpg.step(torch_obs, explore=False)
        agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions]
        
        if maddpg.discrete_action:
            actions = {agent_name: agent_actions[i].argmax() 
                      for i, agent_name in enumerate(env.possible_agents)}
        else:
            actions = {agent_name: agent_actions[i].squeeze() 
                      for i, agent_name in enumerate(env.possible_agents)}
        
        # Compute pairwise Frobenius norms
        frob_matrix = compute_pairwise_frob_norms(maddpg, obs, list(actions.values()), env.action_space)
        
        # Extract strongest edges for this timestep
        strongest_edges = extract_strongest_edges(frob_matrix)
        episode_edges.append(strongest_edges)
        
        # Step environment
        obs, rewards, dones, infos = env.step(actions)
        timestep += 1
        
        if dones.all():
            break
    
    return episode_edges

def visualize_frequent_chains(frequent_chains, logdir, top_k=20):
    """Visualize the most frequent influence chains"""
    if not PLOTTING_AVAILABLE:
        print("Plotting libraries not available. Skipping visualization.")
        return
    
    # Sort chains by support then by strength
    sorted_chains = sorted(frequent_chains.items(), 
                          key=lambda x: (x[1]['support'], x[1]['avg_strength']), 
                          reverse=True)
    
    # Take top K chains
    top_chains = sorted_chains[:top_k]
    
    if not top_chains:
        print("No frequent chains to visualize.")
        return
    
    # Plot 1: Support (frequency) - saved as separate image
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    
    chain_labels = [' → '.join(map(str, chain)) for chain, _ in top_chains]
    supports = [stats['support'] for _, stats in top_chains]
    
    # Reverse the order so highest values appear at the top
    y_positions = list(reversed(range(len(chain_labels))))
    bars1 = ax1.barh(y_positions, supports, color='skyblue', alpha=0.7)
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(chain_labels, fontsize=10)
    ax1.set_xlabel('Support (Number of Episodes)')
    ax1.set_title('Top Influence Chains by Support', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars1):
        width = bar.get_width()
        ax1.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{int(width)}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    support_plot_path = os.path.join(logdir, 'influence_chains_by_support.png')
    plt.savefig(support_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved support plot to {support_plot_path}")
    plt.close()
    
    # Plot 2: Average strength - saved as separate image
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
    
    # Sort chains by strength then by support for this plot
    sorted_chains_by_strength = sorted(frequent_chains.items(), 
                                     key=lambda x: (x[1]['avg_strength'], x[1]['support']), 
                                     reverse=True)
    
    # Take top K chains sorted by strength
    top_chains_by_strength = sorted_chains_by_strength[:top_k]
    chain_labels_strength = [' → '.join(map(str, chain)) for chain, _ in top_chains_by_strength]
    strengths = [stats['avg_strength'] for _, stats in top_chains_by_strength]
    
    y_positions_strength = list(reversed(range(len(chain_labels_strength))))
    bars2 = ax2.barh(y_positions_strength, strengths, color='lightcoral', alpha=0.7)
    ax2.set_yticks(y_positions_strength)
    ax2.set_yticklabels(chain_labels_strength, fontsize=10)
    ax2.set_xlabel('Average Strength')
    ax2.set_title('Top Influence Chains by Average Strength', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars2):
        width = bar.get_width()
        ax2.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.4f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    strength_plot_path = os.path.join(logdir, 'influence_chains_by_strength.png')
    plt.savefig(strength_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved strength plot to {strength_plot_path}")
    plt.close()

def visualize_influence_network(frequent_chains, logdir, min_support=3):
    """Create network visualization of influence relationships"""
    if not PLOTTING_AVAILABLE:
        print("NetworkX/plotting libraries not available. Skipping network visualization.")
        return
    
    try:
        # Create directed graph
        G = nx.DiGraph()
        
        # Add edges from frequent chains
        edge_weights = defaultdict(list)
        
        for chain, stats in frequent_chains.items():
            if stats['support'] >= min_support:
                # Add edges for each consecutive pair in the chain
                for i in range(len(chain) - 1):
                    source, target = chain[i], chain[i + 1]
                    weight = stats['avg_strength'] * stats['support']  # Combine strength and support
                    edge_weights[(source, target)].append(weight)
        
        # Add edges to graph with averaged weights
        for (source, target), weights in edge_weights.items():
            G.add_edge(source, target, weight=np.mean(weights))
        
        if len(G.edges()) == 0:
            print("No edges to visualize in network.")
            return
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        # Layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Draw nodes
        node_sizes = [G.degree(node) * 500 + 300 for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', 
                              alpha=0.7, linewidths=2, edgecolors='darkblue')
        
        # Draw edges with varying thickness based on weight
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1
        edge_widths = [w / max_weight * 5 + 0.5 for w in weights]
        
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, 
                              edge_color='gray', arrows=True, arrowsize=20)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')
        
        # Add edge labels for strongest connections
        edge_labels = {}
        for u, v in edges:
            weight = G[u][v]['weight']
            if weight > max_weight * 0.5:  # Only label strongest edges
                edge_labels[(u, v)] = f'{weight:.3f}'
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10)
        
        plt.title('Influence Network from Frequent Chains\n(Node size = degree, Edge width = influence strength)', 
                 fontsize=14, fontweight='bold')
        plt.axis('off')
        
        plot_path = os.path.join(logdir, 'influence_network.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved influence network to {plot_path}")
        
    except Exception as e:
        print(f"Error creating network visualization: {e}")

def save_results(frequent_chains, all_episode_chains, logdir):
    """Save results to files"""
    
    # Save frequent chains to CSV
    csv_path = os.path.join(logdir, 'frequent_influence_chains.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['chain', 'support', 'avg_strength', 'std_strength', 'total_occurrences', 'episodes']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for chain, stats in sorted(frequent_chains.items(), 
                                 key=lambda x: (x[1]['support'], x[1]['avg_strength']), 
                                 reverse=True):
            writer.writerow({
                'chain': ' → '.join(map(str, chain)),
                'support': stats['support'],
                'avg_strength': stats['avg_strength'],
                'std_strength': stats['std_strength'],
                'total_occurrences': stats['total_occurrences'],
                'episodes': ','.join(map(str, stats['episodes']))
            })
    
    print(f"Saved frequent chains to {csv_path}")
    
    # Save raw data as pickle for further analysis
    pickle_path = os.path.join(logdir, 'influence_chains_data.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump({
            'frequent_chains': frequent_chains,
            'all_episode_chains': all_episode_chains
        }, f)
    
    print(f"Saved raw data to {pickle_path}")

def print_summary_statistics(frequent_chains, all_episode_chains, num_episodes):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("INFLUENCE CHAIN MINING SUMMARY")
    print("="*60)
    
    print(f"Total episodes analyzed: {num_episodes}")
    print(f"Total chains extracted: {len(all_episode_chains)}")
    print(f"Frequent chains found: {len(frequent_chains)}")
    
    if frequent_chains:
        supports = [stats['support'] for stats in frequent_chains.values()]
        strengths = [stats['avg_strength'] for stats in frequent_chains.values()]
        
        print(f"\nSupport statistics:")
        print(f"  Mean: {np.mean(supports):.2f}")
        print(f"  Std:  {np.std(supports):.2f}")
        print(f"  Max:  {max(supports)}")
        print(f"  Min:  {min(supports)}")
        
        print(f"\nStrength statistics:")
        print(f"  Mean: {np.mean(strengths):.6f}")
        print(f"  Std:  {np.std(strengths):.6f}")
        print(f"  Max:  {max(strengths):.6f}")
        print(f"  Min:  {min(strengths):.6f}")
        
        # Show top 5 chains
        top_chains = sorted(frequent_chains.items(), 
                           key=lambda x: (x[1]['support'], x[1]['avg_strength']), 
                           reverse=True)[:5]
        
        print(f"\nTop 5 influence chains:")
        for i, (chain, stats) in enumerate(top_chains, 1):
            chain_str = ' → '.join(map(str, chain))
            print(f"  {i}. {chain_str}")
            print(f"     Support: {stats['support']}, Strength: {stats['avg_strength']:.6f}")
    
    print("="*60)

def run(config):
    """Main execution function"""
    # Load model
    maddpg = MADDPG.init_from_save(config.model_path, test_mode=True)
    
    # Create log directory
    cwd = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(cwd, 'runs', f"influence_chains_{config.env_id}", timestamp)
    os.makedirs(logdir, exist_ok=True)
    
    env = create_environment(config, maddpg)
    env.reset()
    
    maddpg.prep_training(device=DEVICE)
    
    print(f"Running influence chain mining with {config.num_episodes} episodes...")
    print(f"Environment: {config.env_id}")
    print(f"Number of agents: {maddpg.nagents}")
    print(f"Action space discrete: {maddpg.discrete_action}")
    
    # Collect chains from all episodes
    all_episode_chains = []
    
    for episode_id in tqdm(range(config.num_episodes), desc="Running episodes"):
        seed = config.seed + episode_id*1000
        
        episode_edges = run_single_episode(env, maddpg, episode_id, seed)
        
        if len(episode_edges) <= 1:  # Need at least 2 timesteps for chains
            continue

        # Build chains from this episode
        episode_chains = build_influence_chains(
            episode_edges, 
            max_chain_length=config.max_chain_length,
            max_lookahead=config.max_lookahead,  # Updated parameter name
            decay_factor=config.decay_factor
        )
        
        # Add episode ID to chains
        for chain_tuple, strength, timesteps in episode_chains:
            if len(chain_tuple) < 3:
                continue  # Skip chains with less than 3 nodes
            all_episode_chains.append((chain_tuple, strength, episode_id))
    
    # Apply frequent itemset mining
    print(f"\nApplying frequent itemset mining...")
    frequent_chains = frequent_itemset_mining(
        all_episode_chains,
        min_support=config.min_support,
        min_strength_threshold=config.min_strength_threshold
    )
    
    # Print summary
    print_summary_statistics(frequent_chains, all_episode_chains, config.num_episodes)
    
    # Save results
    save_results(frequent_chains, all_episode_chains, logdir)
    
    # Create visualizations
    if frequent_chains:
        visualize_frequent_chains(frequent_chains, logdir, top_k=config.top_k_viz)
        visualize_influence_network(frequent_chains, logdir, min_support=config.min_support)
    else:
        print("No frequent chains found. Skipping visualizations.")
    
    env.close()
    print(f"\nResults saved to: {logdir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Mine influence chains from multi-agent interactions')
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_path", help="Path to trained model directory")
    
    # Episode and mining parameters
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="Number of episodes to run (default: 50)")
    parser.add_argument("--max_chain_length", type=int, default=4,
                        help="Maximum length of influence chains (default: 4)")
    parser.add_argument("--max_lookahead", type=int, default=2,
                        help="Maximum timesteps ahead to look for next edge (default: 2)")
    parser.add_argument("--decay_factor", type=float, default=0.9,
                        help="Decay factor for strength when timesteps are skipped (default: 0.9)")
    
    # Frequent itemset mining parameters
    parser.add_argument("--min_support", type=int, default=10,
                        help="Minimum support (episodes) for frequent chains (default: 10)")
    parser.add_argument("--min_strength_threshold", type=float, default=0.001,
                        help="Minimum strength threshold for chains (default: 0.001)")
    
    # Visualization parameters
    parser.add_argument("--top_k_viz", type=int, default=20,
                        help="Number of top chains to visualize (default: 20)")
    parser.add_argument("--seed", default=42,
                        help="Use seed for reproducibility")

    config = parser.parse_args()
    
    run(config)
