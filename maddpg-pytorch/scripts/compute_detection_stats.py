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
from collections import deque
from tqdm import tqdm

# Try to import plotting libraries - if not available, plotting will be skipped
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib, seaborn, or pandas not available. Plots will be skipped.")

USE_CUDA = torch.cuda.is_available()
DEVICE = 'gpu' if USE_CUDA else 'cpu'
torch_device = torch.device("cuda" if USE_CUDA else "cpu")
ATTACK_START_TS = 5
ATTACK_STOP_TS = 15

def preprocess_env_atari(env):
    env = supersuit.max_observation_v0(env, 2)
    env = supersuit.frame_skip_v0(env, 4)
    env = supersuit.resize_v1(env, 84, 84)
    env = supersuit.frame_stack_v1(env, 4)
    return env

def compute_taylor_delta_policy(maddpg, obs, actions, action_spaces, epsilon):
    # Convert discrete actions to one-hot encoding
    if maddpg.discrete_action:
        one_hot_actions = []
        for i, action in enumerate(actions):
            one_hot = np.zeros(action_spaces[i].n)
            one_hot[action] = 1.0
            one_hot_actions.append(one_hot)
        actions = one_hot_actions

    torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    actions = [Variable(torch.Tensor([actions[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]

    delta_errors = []

    for i, agent_i in enumerate(maddpg.agents):
        if maddpg.discrete_action:
            action_logits_i = agent_i.policy(torch_obs[i])
            action_log_probs = torch.log_softmax(action_logits_i, dim=-1)
            max_action_idx = torch.argmax(action_log_probs, dim=-1)
            target_val = action_log_probs.gather(-1, max_action_idx.unsqueeze(-1)).squeeze()
        else:
            target_val = agent_i.policy(torch_obs[i]).sum()
        
        grad_i = torch.autograd.grad(target_val, torch_obs[i], create_graph=True, retain_graph=True)[0]
        eta_i = epsilon * grad_i.sign() / torch.max(grad_i.norm(p=2), torch.tensor(1e-6))
        
        # First-order Taylor approximation
        j_tilde = target_val + torch.dot(grad_i.flatten(), eta_i.flatten())

        p_torch_obs_i = torch_obs[i] + eta_i
        if maddpg.discrete_action:
            p_action_logits_i = agent_i.policy(p_torch_obs_i)
            p_action_log_probs = torch.log_softmax(p_action_logits_i, dim=-1)
            p_max_action_idx = torch.argmax(p_action_log_probs, dim=-1)
            j_perturbed = p_action_log_probs.gather(-1, p_max_action_idx.unsqueeze(-1)).squeeze()
        else:
            j_perturbed = agent_i.policy(p_torch_obs_i).sum()
            
        delta_error = abs(j_perturbed - j_tilde).item()
        delta_errors.append(delta_error)

    return delta_errors

def run_single_detection_episode(env, maddpg, ref_values, ref_stds, attacked_agent_id, k_sigma, 
                                detection_method, seed=None):
    """
    Run single episode to collect both patient 0 and attacked agent detection data
    
    Returns:
        patient_zeros: List of agent IDs detected as faulty at first detection (patient 0)
        detection_time: Timestep when patient 0s were detected (None if no detection)
        attacked_detected: Boolean indicating if attacked agent was ever detected as faulty
    """
    obs = env.reset(seed=seed) if seed else env.reset()
    
    result_deques = [deque(maxlen=5) for _ in range(maddpg.nagents)]
    prev_errors = [0 for _ in range(maddpg.nagents)]
    cnt = 0
    attacked_detected = False
    patient_zeros = []
    detection_time = None

    while True:
        torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=False) for i in range(maddpg.nagents)]
        torch_agent_actions = maddpg.step(torch_obs, explore=False)
        agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions]
        
        if maddpg.discrete_action:
            actions = {agent_name: agent_actions[i].argmax() for i, agent_name in enumerate(env.possible_agents)}
        else:
            actions = {agent_name: agent_actions[i].squeeze() for i, agent_name in enumerate(env.possible_agents)}

        # Attack logic
        if cnt >= ATTACK_START_TS and cnt <= ATTACK_STOP_TS:  # Attack window
            if maddpg.discrete_action:
                worst_action = np.random.choice(env.action_space[attacked_agent_id].n)
                actions[env.possible_agents[attacked_agent_id]] = worst_action
            else:
                actions[env.possible_agents[attacked_agent_id]] = np.random.uniform(-1, 1, size=env.action_space[attacked_agent_id].shape)

        results = compute_taylor_delta_policy(maddpg, obs, list(actions.values()), env.action_space, 0.01)

        # Check for detections
        detected_agents = []
        for i in range(maddpg.nagents):
            result_deques[i].append(results[i])
            
            if detection_method == 'mean_std':
                taylor_approx_error = np.mean(result_deques[i])
                threshold_exceeded = abs(taylor_approx_error - ref_values[i][cnt]) > k_sigma * ref_stds[i][cnt]
            elif detection_method == 'median_mad':
                taylor_approx_error = np.mean(result_deques[i])
                threshold_exceeded = abs(taylor_approx_error - ref_values[i][cnt]) > k_sigma * ref_stds[i][cnt]
            elif detection_method == 'diff':
                if cnt > 0:
                    current_diff = results[i] - prev_errors[i]
                    threshold_exceeded = abs(current_diff - ref_values[i][cnt]) > k_sigma * ref_stds[i][cnt]
                else:
                    threshold_exceeded = False
            else:
                raise ValueError(f"Unknown detection method: {detection_method}")
            
            if threshold_exceeded:
                detected_agents.append(i)
                if i == attacked_agent_id and cnt >= ATTACK_START_TS:
                    attacked_detected = True
        
        # Record patient 0 (first detection)
        if detected_agents and not patient_zeros:
            patient_zeros = detected_agents
            detection_time = cnt

        prev_errors = results.copy()
        next_obs, rewards, dones, infos = env.step(actions)
        obs = next_obs
        cnt += 1
        
        if dones.all():
            break

    return patient_zeros, detection_time, attacked_detected

def load_reference_data(ref_val_dir, nagents, method):
    """
    Load reference data based on detection method
    
    Args:
        ref_val_dir: Directory containing reference CSV files
        nagents: Number of agents
        method: Detection method ('mean_std', 'median_mad', 'diff')
    
    Returns:
        ref_values: List of reference values for each agent
        ref_stds: List of reference standard deviations for each agent
    """
    ref_values = [[] for _ in range(nagents)]
    ref_stds = [[] for _ in range(nagents)]

    for agent_id in range(nagents):
        csv_filename = f"maddpg_taylor_error_atk_free_agent_{agent_id}.csv"
        csv_path = os.path.join(ref_val_dir, csv_filename)
        
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                if method == 'mean_std':
                    # Columns: 2=mean, 4=std_dev
                    ref_values[agent_id].append(float(row[2]))
                    ref_stds[agent_id].append(float(row[4]))
                elif method == 'median_mad':
                    # Columns: 7=median, 8=mad
                    ref_values[agent_id].append(float(row[7]))
                    ref_stds[agent_id].append(float(row[8]))
                elif method == 'diff':
                    # Columns: 9=diff_mean, 10=std_dev
                    ref_values[agent_id].append(float(row[9]))
                    ref_stds[agent_id].append(float(row[10]))
                else:
                    raise ValueError(f"Unknown method: {method}")

    return ref_values, ref_stds

def plot_detection_metrics(all_results, logdir):
    """
    Create comprehensive plots of detection metrics
    """
    if not PLOTTING_AVAILABLE:
        print("Plotting libraries not available. Skipping plot generation.")
        return
    
    # Convert results to DataFrame for easier plotting
    df = pd.DataFrame(all_results)
    
    # Set up the plotting style
    plt.style.use('default')
    try:
        sns.set_palette("husl")
    except:
        pass  # fallback if seaborn not available
    
    # 1. Performance comparison plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    methods = ['mean_std', 'median_mad', 'diff']
    
    # Plot 1: Patient 0 Detection Percentages
    ax1 = axes[0]
    for method in methods:
        method_data = df[df['method'] == method]
        grouped = method_data.groupby('k_sigma').agg({
            'tp_percentage': 'mean',
            'fp_percentage': 'mean',
            'fn_percentage': 'mean'
        }).reset_index()
        
        ax1.plot(grouped['k_sigma'], grouped['tp_percentage'], 'o-', linewidth=2, markersize=6, label=f'{method.replace("_", "+")} TP')
        ax1.plot(grouped['k_sigma'], grouped['fp_percentage'], 's--', linewidth=2, markersize=6, label=f'{method.replace("_", "+")} FP')
        ax1.plot(grouped['k_sigma'], grouped['fn_percentage'], '^:', linewidth=2, markersize=6, label=f'{method.replace("_", "+")} FN')
    
    ax1.set_xlabel('K-Sigma')
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Patient 0 Detection Percentages')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Attacked Agent Detection Percentage
    ax2 = axes[1]
    for method in methods:
        method_data = df[df['method'] == method]
        grouped = method_data.groupby('k_sigma')['attacked_agent_detected_percentage'].mean()
        ax2.plot(grouped.index, grouped.values, 'o-', linewidth=2, markersize=6, label=method.replace('_', '+'))
    
    ax2.set_xlabel('K-Sigma')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Attacked Agent Detection (Any Timestep)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # Plot 3: F1-Score
    ax3 = axes[2]
    for method in methods:
        method_data = df[df['method'] == method]
        grouped = method_data.groupby('k_sigma')['f1_score'].mean()
        ax3.plot(grouped.index, grouped.values, 'o-', linewidth=2, markersize=6, label=method.replace('_', '+'))
    
    ax3.set_xlabel('K-Sigma')
    ax3.set_ylabel('F1-Score')
    ax3.set_title('F1-Score vs K-Sigma')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Percentage heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # TP Percentage heatmap
    ax1 = axes[0, 0]
    pivot_tp = df.groupby(['method', 'k_sigma'])['tp_percentage'].mean().unstack()
    if PLOTTING_AVAILABLE and 'sns' in globals():
        sns.heatmap(pivot_tp, annot=True, fmt='.1f', cmap='Greens', ax=ax1)
    else:
        im1 = ax1.imshow(pivot_tp.values, cmap='Greens', aspect='auto')
        ax1.set_xticks(range(len(pivot_tp.columns)))
        ax1.set_yticks(range(len(pivot_tp.index)))
        ax1.set_xticklabels(pivot_tp.columns)
        ax1.set_yticklabels(pivot_tp.index)
        for i in range(len(pivot_tp.index)):
            for j in range(len(pivot_tp.columns)):
                ax1.text(j, i, f'{pivot_tp.iloc[i, j]:.1f}', ha='center', va='center', color='black')
        plt.colorbar(im1, ax=ax1)
    ax1.set_title('True Positive Percentage (Patient 0)')
    ax1.set_xlabel('K-Sigma Values')
    ax1.set_ylabel('Detection Methods')
    
    # FP Percentage heatmap
    ax2 = axes[0, 1]
    pivot_fp = df.groupby(['method', 'k_sigma'])['fp_percentage'].mean().unstack()
    if PLOTTING_AVAILABLE and 'sns' in globals():
        sns.heatmap(pivot_fp, annot=True, fmt='.1f', cmap='Reds', ax=ax2)
    else:
        im2 = ax2.imshow(pivot_fp.values, cmap='Reds', aspect='auto')
        ax2.set_xticks(range(len(pivot_fp.columns)))
        ax2.set_yticks(range(len(pivot_fp.index)))
        ax2.set_xticklabels(pivot_fp.columns)
        ax2.set_yticklabels(pivot_fp.index)
        for i in range(len(pivot_fp.index)):
            for j in range(len(pivot_fp.columns)):
                ax2.text(j, i, f'{pivot_fp.iloc[i, j]:.1f}', ha='center', va='center', color='black')
        plt.colorbar(im2, ax=ax2)
    ax2.set_title('False Positive Percentage (Patient 0)')
    ax2.set_xlabel('K-Sigma Values')
    ax2.set_ylabel('Detection Methods')
    
    # Attacked Agent Detection Percentage heatmap
    ax3 = axes[1, 0]
    pivot_attacked = df.groupby(['method', 'k_sigma'])['attacked_agent_detected_percentage'].mean().unstack()
    if PLOTTING_AVAILABLE and 'sns' in globals():
        sns.heatmap(pivot_attacked, annot=True, fmt='.1f', cmap='Blues', ax=ax3)
    else:
        im3 = ax3.imshow(pivot_attacked.values, cmap='Blues', aspect='auto')
        ax3.set_xticks(range(len(pivot_attacked.columns)))
        ax3.set_yticks(range(len(pivot_attacked.index)))
        ax3.set_xticklabels(pivot_attacked.columns)
        ax3.set_yticklabels(pivot_attacked.index)
        for i in range(len(pivot_attacked.index)):
            for j in range(len(pivot_attacked.columns)):
                ax3.text(j, i, f'{pivot_attacked.iloc[i, j]:.1f}', ha='center', va='center', color='black')
        plt.colorbar(im3, ax=ax3)
    ax3.set_title('Attacked Agent Detection Percentage')
    ax3.set_xlabel('K-Sigma Values')
    ax3.set_ylabel('Detection Methods')
    
    # F1-Score heatmap
    ax4 = axes[1, 1]
    pivot_f1 = df.groupby(['method', 'k_sigma'])['f1_score'].mean().unstack()
    if PLOTTING_AVAILABLE and 'sns' in globals():
        sns.heatmap(pivot_f1, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax4)
    else:
        im4 = ax4.imshow(pivot_f1.values, cmap='RdYlBu_r', aspect='auto')
        ax4.set_xticks(range(len(pivot_f1.columns)))
        ax4.set_yticks(range(len(pivot_f1.index)))
        ax4.set_xticklabels(pivot_f1.columns)
        ax4.set_yticklabels(pivot_f1.index)
        for i in range(len(pivot_f1.index)):
            for j in range(len(pivot_f1.columns)):
                ax4.text(j, i, f'{pivot_f1.iloc[i, j]:.3f}', ha='center', va='center', color='black')
        plt.colorbar(im4, ax=ax4)
    ax4.set_title('F1-Score Heatmap')
    ax4.set_xlabel('K-Sigma Values')
    ax4.set_ylabel('Detection Methods')
    
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, 'percentage_heatmaps.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Performance metrics comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    metrics = ['tp_percentage', 'fp_percentage', 'precision', 'f1_score']
    metric_names = ['True Positive Percentage', 'False Positive Percentage', 'Precision', 'F1-Score']
    
    for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]
        
        for method in methods:
            method_data = df[df['method'] == method]
            grouped = method_data.groupby('k_sigma')[metric].mean()
            ax.plot(grouped.index, grouped.values, 'o-', linewidth=2, markersize=6, label=method.replace('_', '+'))
        
        ax.set_xlabel('K-Sigma')
        ax.set_ylabel(name)
        ax.set_title(f'{name} vs K-Sigma')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Detection accuracy by attacked agent
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        method_data = df[df['method'] == method]
        
        # Group by attacked agent and K-sigma
        pivot_agent = method_data.groupby(['attacked_agent_id', 'k_sigma'])['tp_percentage'].mean().unstack()
        
        if PLOTTING_AVAILABLE and 'sns' in globals():
            sns.heatmap(pivot_agent, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax)
        else:
            # Fallback without seaborn
            im = ax.imshow(pivot_agent.values, cmap='RdYlBu_r', aspect='auto')
            ax.set_xticks(range(len(pivot_agent.columns)))
            ax.set_yticks(range(len(pivot_agent.index)))
            ax.set_xticklabels(pivot_agent.columns)
            ax.set_yticklabels(pivot_agent.index)
            
            # Add text annotations
            for i in range(len(pivot_agent.index)):
                for j in range(len(pivot_agent.columns)):
                    ax.text(j, i, f'{pivot_agent.iloc[i, j]:.2f}', 
                           ha='center', va='center', color='black')
            
            plt.colorbar(im, ax=ax)
        
        ax.set_title(f'TP Percentage by Agent: {method.replace("_", "+")}')
        ax.set_xlabel('K-Sigma Values')
        ax.set_ylabel('Attacked Agent ID')
    
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, 'agent_specific_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Optimal K-sigma selection plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate balanced score (TP% - FP%) for each configuration
    df['balanced_score'] = df['tp_percentage'] - df['fp_percentage']
    
    for method in methods:
        method_data = df[df['method'] == method]
        grouped = method_data.groupby('k_sigma')['balanced_score'].mean()
        ax.plot(grouped.index, grouped.values, 'o-', linewidth=2, markersize=6, label=method.replace('_', '+'))
    
    ax.set_xlabel('K-Sigma')
    ax.set_ylabel('Balanced Score (TP% - FP%)')
    ax.set_title('Optimal K-Sigma Selection (Higher is Better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, 'optimal_k_sigma.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to: {logdir}")

def run_detection_analysis(config):
    """
    Run detection analysis for all configurations
    """
    maddpg = MADDPG.init_from_save(config.model_path, test_mode=True)

    # Create environment
    try:
        env_func = getattr(mpe, config.env_id)
        if config.env_id == 'simple_spread_v3':
            env = env_func.parallel_env(continuous_actions=not maddpg.discrete_action, render_mode='rgb_array', N=5)
        else:
            env = env_func.parallel_env(continuous_actions=not maddpg.discrete_action, render_mode='rgb_array')
    except:
        try:
            env_func = getattr(sisl, config.env_id)
            env = env_func.parallel_env(n_pursuers=5, render_mode='rgb_array') if config.env_id == 'waterworld_v4' else env_func.parallel_env(render_mode='rgb_array')
        except:
            env_func = getattr(atari, config.env_id)
            env = env_func.parallel_env(render_mode='rgb_array')
            env = preprocess_env_atari(env)

    env = PettingZooWrapper.wrap_env(env)
    env.reset()
    maddpg.prep_training(device=DEVICE)

    # Detection configurations
    k_sigma_values = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    methods = ['mean_std', 'median_mad', 'diff']
    
    # Create output directory
    cwd = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(cwd, 'runs', f"{config.env_id}_detection_metrics", timestamp)
    os.makedirs(logdir, exist_ok=True)

    # Results storage
    all_results = []
    all_problematic_cases = []  # Store all problematic cases for summary logging

    for method in methods:
        print(f"\n=== Processing method: {method} ===")
        
        # Load reference data for this method
        ref_values, ref_stds = load_reference_data(config.ref_val_dir, maddpg.nagents, method)
        
        for k_sigma in k_sigma_values:
            print(f"Processing K-sigma: {k_sigma}")
            
            # Test each agent as the attacked agent
            for attacked_agent_id in range(maddpg.nagents):
                
                # Run multiple episodes to get statistics
                tp_count = 0  # True Positives: Patient 0 is the attacked agent
                fp_count = 0  # False Positives: Patient 0 is not the attacked agent
                fn_count = 0  # False Negatives: No detection when there should be
                
                # Additional stat: attacked agent detected as faulty at any timestep
                attacked_agent_detected_count = 0  # Count when attacked agent is detected as faulty (regardless of patient 0 status)
                
                # Track problematic seeds for logging
                failed_detection_seeds = []  # Seeds where attacked agent was not detected
                false_positive_seeds = []   # Seeds where wrong agents were detected as patient 0
                
                num_episodes = config.num_episodes
                
                for episode in tqdm(range(num_episodes), desc=f"Method:{method}, K:{k_sigma}, Agent:{attacked_agent_id}"):
                    current_seed = 12345 + episode
                    
                    # Run single episode to get both patient 0 and attacked agent detection data
                    patient_zeros, detection_time, attacked_detected = run_single_detection_episode(
                        env, maddpg, ref_values, ref_stds, attacked_agent_id, 
                        k_sigma, method, seed=current_seed
                    )
                    
                    # Count attacked agent detection statistic
                    if attacked_detected:
                        attacked_agent_detected_count += 1
                    
                    # Count metrics for patient 0 scenario
                    if len(patient_zeros) > 0:
                        if attacked_agent_id in patient_zeros and detection_time >= ATTACK_START_TS:
                            tp_count += 1  # Correctly identified attacked agent as patient 0
                        else:
                            fp_count += 1  # Incorrectly identified wrong agent(s) as patient 0
                            false_positive_seeds.append({
                                'seed': current_seed,
                                'attacked_agent': attacked_agent_id,
                                'detected_agents': patient_zeros,
                                'detection_time': detection_time
                            })
                    else:
                        fn_count += 1  # Failed to detect any fault when attack occurred
                        failed_detection_seeds.append({
                            'seed': current_seed,
                            'attacked_agent': attacked_agent_id,
                            'reason': 'no_detection_during_attack'
                        })

                # Calculate percentages and basic metrics
                total_attack_episodes = num_episodes
                
                # Calculate percentages for patient 0 detection
                tp_percentage = (tp_count / total_attack_episodes) * 100 if total_attack_episodes > 0 else 0
                fp_percentage = (fp_count / total_attack_episodes) * 100 if total_attack_episodes > 0 else 0
                fn_percentage = (fn_count / total_attack_episodes) * 100 if total_attack_episodes > 0 else 0
                
                # Calculate percentage for attacked agent detection at any timestep
                attacked_agent_detected_percentage = (attacked_agent_detected_count / total_attack_episodes) * 100 if total_attack_episodes > 0 else 0
                
                # Calculate precision and F1-score for patient 0 detection
                precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
                recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                # Log problematic seeds
                if failed_detection_seeds or false_positive_seeds:
                    log_filename = f"problematic_seeds_{method}_k{k_sigma}_agent{attacked_agent_id}.log"
                    log_filepath = os.path.join(logdir, log_filename)
                    
                    with open(log_filepath, 'w') as logfile:
                        logfile.write(f"Problematic Seeds Report\n")
                        logfile.write(f"========================\n")
                        logfile.write(f"Method: {method}\n")
                        logfile.write(f"K-Sigma: {k_sigma}\n")
                        logfile.write(f"Attacked Agent: {attacked_agent_id}\n")
                        logfile.write(f"Total Episodes: {num_episodes}\n")
                        logfile.write(f"Performance: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1_score:.3f}\n")
                        logfile.write(f"Patient 0 Percentages: TP={tp_percentage:.1f}%, FP={fp_percentage:.1f}%, FN={fn_percentage:.1f}%\n")
                        logfile.write(f"Attacked Agent Detection: {attacked_agent_detected_percentage:.1f}%\n\n")
                        
                        if failed_detection_seeds:
                            logfile.write(f"FAILED DETECTIONS (False Negatives): {len(failed_detection_seeds)} cases\n")
                            logfile.write("-" * 60 + "\n")
                            for i, case in enumerate(failed_detection_seeds, 1):
                                logfile.write(f"{i}. Seed: {case['seed']}\n")
                                logfile.write(f"   Attacked Agent: {case['attacked_agent']}\n")
                                logfile.write(f"   Reason: {case['reason']}\n\n")
                        
                        if false_positive_seeds:
                            logfile.write(f"FALSE POSITIVES (Wrong Patient 0): {len(false_positive_seeds)} cases\n")
                            logfile.write("-" * 60 + "\n")
                            for i, case in enumerate(false_positive_seeds, 1):
                                logfile.write(f"{i}. Seed: {case['seed']}\n")
                                logfile.write(f"   Attacked Agent: {case['attacked_agent']}\n")
                                logfile.write(f"   Detected Agents: {case['detected_agents']}\n")
                                logfile.write(f"   Detection Time: {case['detection_time']}\n\n")
                        
                        logfile.write(f"Summary:\n")
                        logfile.write(f"- Failed Detections: {len(failed_detection_seeds)}\n")
                        logfile.write(f"- False Positives: {len(false_positive_seeds)}\n")
                
                # Add to global problematic cases tracking
                all_problematic_cases.extend([
                    {**case, 'method': method, 'k_sigma': k_sigma, 'type': 'failed_detection'} 
                    for case in failed_detection_seeds
                ])
                all_problematic_cases.extend([
                    {**case, 'method': method, 'k_sigma': k_sigma, 'type': 'false_positive'} 
                    for case in false_positive_seeds
                ])
                
                result = {
                    'method': method,
                    'k_sigma': k_sigma,
                    'attacked_agent_id': attacked_agent_id,
                    'num_episodes': num_episodes,
                    'tp_count': tp_count,
                    'fp_count': fp_count,
                    'fn_count': fn_count,
                    'attacked_agent_detected_count': attacked_agent_detected_count,
                    'tp_percentage': tp_percentage,
                    'fp_percentage': fp_percentage,
                    'fn_percentage': fn_percentage,
                    'attacked_agent_detected_percentage': attacked_agent_detected_percentage,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score
                }
                
                all_results.append(result)
                
                print(f"Agent {attacked_agent_id}: Patient0 TP={tp_percentage:.1f}%, FP={fp_percentage:.1f}%, FN={fn_percentage:.1f}%, AttackedAgentDetected={attacked_agent_detected_percentage:.1f}%, F1={f1_score:.3f}")

    # Save results to CSV
    output_csv = os.path.join(logdir, 'detection_metrics_results.csv')
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['method', 'k_sigma', 'attacked_agent_id', 'num_episodes', 
                     'tp_count', 'fp_count', 'fn_count', 'attacked_agent_detected_count',
                     'tp_percentage', 'fp_percentage', 'fn_percentage', 'attacked_agent_detected_percentage',
                     'precision', 'recall', 'f1_score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)

    print(f"\nResults saved to: {output_csv}")
    
    # Generate summary log of all problematic cases
    if all_problematic_cases:
        summary_log_path = os.path.join(logdir, 'all_problematic_seeds_summary.log')
        with open(summary_log_path, 'w') as logfile:
            logfile.write(f"COMPREHENSIVE PROBLEMATIC SEEDS SUMMARY\n")
            logfile.write(f"=======================================\n")
            logfile.write(f"Total Problematic Cases: {len(all_problematic_cases)}\n\n")
            
            # Group by type
            by_type = {}
            for case in all_problematic_cases:
                case_type = case['type']
                if case_type not in by_type:
                    by_type[case_type] = []
                by_type[case_type].append(case)
            
            for case_type, cases in by_type.items():
                logfile.write(f"{case_type.upper().replace('_', ' ')}: {len(cases)} cases\n")
                logfile.write("-" * 60 + "\n")
                
                for i, case in enumerate(cases, 1):
                    logfile.write(f"{i}. Method: {case['method']}, K-Sigma: {case['k_sigma']}, Seed: {case['seed']}\n")
                    if 'attacked_agent' in case:
                        logfile.write(f"   Attacked Agent: {case['attacked_agent']}\n")
                    if 'detected_agents' in case:
                        logfile.write(f"   Detected Agents: {case['detected_agents']}\n")
                    if 'detection_time' in case:
                        logfile.write(f"   Detection Time: {case['detection_time']}\n")
                    if 'reason' in case:
                        logfile.write(f"   Reason: {case['reason']}\n")
                    logfile.write("\n")
                
                logfile.write("\n")
            
            # Method-wise summary
            logfile.write("METHOD-WISE BREAKDOWN:\n")
            logfile.write("-" * 30 + "\n")
            by_method = {}
            for case in all_problematic_cases:
                method = case['method']
                if method not in by_method:
                    by_method[method] = []
                by_method[method].append(case)
            
            for method, cases in by_method.items():
                logfile.write(f"{method}: {len(cases)} problematic cases\n")
                type_counts = {}
                for case in cases:
                    case_type = case['type']
                    type_counts[case_type] = type_counts.get(case_type, 0) + 1
                for case_type, count in type_counts.items():
                    logfile.write(f"  - {case_type}: {count}\n")
                logfile.write("\n")
        
        print(f"Problematic seeds summary saved to: {summary_log_path}")
    else:
        print("No problematic cases found across all configurations!")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_detection_metrics(all_results, logdir)
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    for method in methods:
        for k_sigma in k_sigma_values:
            method_results = [r for r in all_results if r['method'] == method and r['k_sigma'] == k_sigma]
            if method_results:
                avg_tp_pct = np.mean([r['tp_percentage'] for r in method_results])
                avg_fp_pct = np.mean([r['fp_percentage'] for r in method_results])
                avg_fn_pct = np.mean([r['fn_percentage'] for r in method_results])
                avg_attacked_detected_pct = np.mean([r['attacked_agent_detected_percentage'] for r in method_results])
                avg_f1 = np.mean([r['f1_score'] for r in method_results])
                print(f"{method}, K={k_sigma}: Patient0 TP={avg_tp_pct:.1f}%, FP={avg_fp_pct:.1f}%, FN={avg_fn_pct:.1f}%, AttackedDetected={avg_attacked_detected_pct:.1f}%, F1={avg_f1:.3f}")

    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_path", help="Model directory")
    parser.add_argument("--ref_val_dir", type=str, required=True, 
                       help="Directory containing reference CSV files")
    parser.add_argument("--num_episodes", type=int, default=10,
                       help="Number of episodes to run for each configuration")

    config = parser.parse_args()
    run_detection_analysis(config)
