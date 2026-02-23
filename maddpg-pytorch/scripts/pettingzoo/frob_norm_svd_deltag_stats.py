"""
Multi-seed SVD-Based Gradient Coupling Analysis for PettingZoo Environments.

This script analyzes the relationship between Frobenius norm and gradient coupling
strength across multiple seeds using SVD-directed perturbations.

For each seed:
1. Runs a single episode
2. At each timestep, for all agent pairs (i, j):
   - Computes cross-Hessian H = ∇_{a_j} ∇_{a_i} Q_i
   - Computes Frobenius norm ||H||_F
   - Performs SVD to find v_max (right singular vector of largest singular value)
   - Perturbs a_j along v_max: a_j' = a_j + epsilon * v_max
   - Measures gradient shift: ||Δg||_2 = ||∇_{a_i} Q_i(a_j') - ∇_{a_i} Q_i(a_j)||_2
   - Uses new gradient g' to perturb a_i: a_i' = a_i + epsilon * (g' / ||g'||)
   - Measures critic value shift: ΔQ = Q_i(a_i', a_j') - Q_i(a_i, a_j)
3. Collects (seed, timestep, agent_i, agent_j, frob_norm, delta_g_norm, delta_critic) data
4. Computes statistics and generates scatter plots of:
   - ||H||_F vs ||Δg||_2 (gradient shift analysis)
   - ||H||_F vs ΔQ (critic value shift analysis)
"""
import argparse
import os
import json
import random
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from scipy import stats

from algorithms.maddpg import MADDPG
from modules.constants import DEVICE, torch_device
from modules.environment import create_environment
from modules.metrics.basic_metrics import compute_pairwise_frob_svd_coupling_analysis
from torch.autograd import Variable


class SVDCouplingAnalysisRunner:
    """
    Multi-seed experiment runner for SVD-based gradient coupling analysis.
    """
    
    def __init__(self, config):
        """
        Initialize the experiment runner.
        
        Args:
            config: Configuration object containing experiment parameters
        """
        self.config = config
        self.maddpg = None
        self.env = None
        self.logdir = None
        self.total_experiments = config.total_experiments
        self.epsilon = config.epsilon
        
        # Storage for results across all seeds
        # List of tuples: (seed, timestep, agent_i, agent_j, frob_norm, grad_norm, delta_g_norm, delta_critic1, delta_critic2)
        self.coupling_data = []

        
    def setup_experiment(self):
        """Set up the experiment environment and logging."""
        # Load MADDPG model
        self.maddpg = MADDPG.init_from_save(self.config.model_path)
        
        # Check if discrete action space (not supported for SVD perturbation)
        if self.maddpg.discrete_action:
            print("WARNING: Discrete action spaces detected.")
            print("SVD-based perturbation on one-hot vectors may not be geometrically meaningful.")
            print("Proceeding anyway, but results should be interpreted with caution.")
        
        # Create log directory
        cwd = os.getcwd()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        env_type = 'discrete' if self.maddpg.discrete_action else 'continuous'
        
        self.logdir = os.path.join(
            cwd, 'runs', 
            f"{self.config.env_id}_{env_type}_svd_coupling_analysis",
            f"{timestamp}_nagents{self.maddpg.nagents}_epsilon{self.epsilon}_seeds{self.total_experiments}"
        )
        os.makedirs(self.logdir, exist_ok=True)
        
        # Create environment
        self.env = create_environment(self.config, self.maddpg)
        
        # Prepare MADDPG for training mode to ensure all tensors are on the correct device
        device_str = 'gpu' if DEVICE == 'gpu' else 'cpu'
        self.maddpg.prep_training(device=device_str)
        
        print(f"Experiment setup complete. Log directory: {self.logdir}")
        print(f"Analyzing all {self.maddpg.nagents} x {self.maddpg.nagents} agent pairs")
        print(f"Perturbation epsilon: {self.epsilon}")
        print(f"Will run {self.total_experiments} experiments")
        
    def run_single_episode(self, seed):
        """
        Run a single episode and collect coupling analysis data at each timestep.
        
        Args:
            seed: Random seed for the episode
            
        Returns:
            List of tuples: (seed, timestep, agent_i, agent_j, frob_norm, delta_g_norm, delta_critic)
        """
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Prepare for rollout
        with torch.no_grad():
            self.maddpg.prep_rollouts(device=DEVICE)
        
        obs = self.env.reset(seed=seed)
        episode_data = []
        timestep = 0
        
        while True:
            torch_obs = [Variable(torch.tensor([obs[i]], dtype=torch.float32).to(torch_device), 
                                requires_grad=False) 
                        for i in range(self.maddpg.nagents)]
            
            # Get actions
            with torch.no_grad():
                torch_agent_actions = self.maddpg.step(torch_obs, explore=False)
                agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions]
            
            if self.maddpg.discrete_action:
                actions = {agent_name: agent_actions[i].argmax() 
                         for i, agent_name in enumerate(self.env.possible_agents)}
            else:
                actions = {agent_name: agent_actions[i].squeeze() 
                         for i, agent_name in enumerate(self.env.possible_agents)}
            
            # Compute coupling analysis for all agent pairs
            coupling_results = compute_pairwise_frob_svd_coupling_analysis(
                self.maddpg, obs, list(actions.values()), self.env.action_space, self.epsilon
            )
            
            # Store results for this timestep
            for (agent_i, agent_j), metrics in coupling_results.items():
                episode_data.append((
                    seed,
                    timestep,
                    agent_i,
                    agent_j,
                    metrics['frob_norm'],
                    metrics['grad_norm'],
                    metrics['delta_g_norm'],
                    metrics['perturbed_grad_norm'],
                    metrics['delta_critic_j_only'],
                    metrics['delta_critic1'],
                    metrics['delta_critic2']
                ))
            
            # Environment step
            next_obs, rewards, dones, infos = self.env.step(actions)
            obs = next_obs
            timestep += 1
            
            # Check if episode is done
            if dones.all():
                break
        
        return episode_data
    
    def run_all_experiments(self):
        """Run experiments for all seeds and collect coupling analysis data."""
        if self.maddpg is None:
            raise RuntimeError("MADDPG model not loaded. Call setup_experiment() first.")
        
        print(f"\nStarting multi-seed SVD coupling analysis with {self.total_experiments} seeds...")
        
        for seed in tqdm(range(self.total_experiments), desc="Running experiments"):
            episode_data = self.run_single_episode(seed)
            self.coupling_data.extend(episode_data)
        
        print(f"\nCollected {len(self.coupling_data)} data points across all experiments")
    
    def generate_plots(self, df):
        """
        Generate scatter plots of ||H||_F vs ||Δg||_2 and ||H||_F vs ΔQ for each agent pair.
        
        Args:
            df: DataFrame containing coupling data
        """
        print("\nGenerating scatter plots...")
        
        # Create subfolders for organized output
        self.plots_dir = os.path.join(self.logdir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Generate plots for different metrics
        self._generate_metric_plots(df, 'delta_g_norm', '||Δg||_2 (Gradient Shift)', 'delta_g')
        self._generate_metric_plots(df, 'delta_critic1', 'ΔQ₁ (Only Agent i Perturbed)', 'delta_critic1')
        self._generate_metric_plots(df, 'delta_critic2', 'ΔQ₂ (Both Agents Perturbed)', 'delta_critic2')
        
        # Generate adversarial assist plots
        df['delta_critic_diff'] = df['delta_critic2'] - df['delta_critic1']
        self._generate_metric_plots(df, 'delta_critic_diff', 'ΔQ₂ - ΔQ₁ (Adversarial Assist)', 'adversarial_assist')
                # Plot 1: Verify orthogonal projection — delta_critic_j_only vs ||H||_F
        # If orthogonalisation works, perturbations along d_orthogonal should cause
        # near-zero first-order change in Q_i, so points should cluster around y=0.
        self._generate_metric_plots(
            df, 'delta_critic_j_only',
            '\u0394Q (Only j Perturbed, Ortho Direction)',
            'ortho_verification'
        )
        
        # Plot 2: delta_critic2 vs perturbed gradient norm ||g'||_2
        self._generate_xy_metric_plots(
            df,
            x_col='perturbed_grad_norm',
            x_label='||g\'||_2 (Perturbed Gradient Norm)',
            y_col='delta_critic2',
            y_label='\u0394Q\u2082 (Both Agents Perturbed)',
            metric_name='dc2_vs_perturbed_grad'
        )
                # Generate 3D plots
        self._generate_3d_plots(df)
    
    def _generate_metric_plots(self, df, metric_col, metric_label, metric_name):
        """
        Generate scatter plots for a specific metric vs Frobenius norm.
        
        Args:
            df: DataFrame containing coupling data
            metric_col: Column name for the metric ('delta_g_norm' or 'delta_critic')
            metric_label: Display label for the metric
            metric_name: Short name for file naming
        """
        print(f"  Generating plots for {metric_label}...")
        
        # Get unique agent pairs
        agent_pairs = df[['agent_i', 'agent_j']].drop_duplicates().values
        
        n_pairs = len(agent_pairs)
        n_cols = min(3, int(np.ceil(np.sqrt(n_pairs))))
        n_rows = int(np.ceil(n_pairs / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_pairs == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, (agent_i, agent_j) in enumerate(agent_pairs):
            ax = axes[idx]
            
            # Filter data for this agent pair
            pair_data = df[(df['agent_i'] == agent_i) & (df['agent_j'] == agent_j)]
            
            # Remove NaN values
            pair_data = pair_data.dropna(subset=['frob_norm', metric_col])
            
            if len(pair_data) == 0:
                ax.text(0.5, 0.5, f'No valid data\nfor pair ({agent_i}, {agent_j})',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xlabel('||H||_F')
                ax.set_ylabel(metric_label)
                continue
            
            x = pair_data['frob_norm'].values
            y = pair_data[metric_col].values
            
            # Scatter plot
            ax.scatter(x, y, alpha=0.3, s=10, c='blue')
            
            # Compute linear fit and Pearson correlation
            if len(x) > 1:
                # Linear regression
                z = np.polyfit(x, y, 1)
                p = np.poly1d(z)
                x_fit = np.linspace(x.min(), x.max(), 100)
                ax.plot(x_fit, p(x_fit), "r--", linewidth=2, label=f'y={z[0]:.3f}x+{z[1]:.3f}')
                
                # Pearson correlation
                pearson_r, pearson_p = stats.pearsonr(x, y)
                ax.text(0.05, 0.95, f'r = {pearson_r:.3f}\np = {pearson_p:.3e}',
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                ax.legend()
            
            ax.set_xlabel('||H||_F (Frobenius Norm)')
            ax.set_ylabel(metric_label)
            ax.set_title(f'Agent {agent_i} → Agent {agent_j}')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_pairs, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        # Create subfolder for this metric
        metric_dir = os.path.join(self.plots_dir, metric_name)
        os.makedirs(metric_dir, exist_ok=True)
        
        # Save combined figure
        combined_path = os.path.join(metric_dir, f"coupling_analysis_{metric_name}_all_pairs.png")
        plt.savefig(combined_path, dpi=150, bbox_inches='tight')
        print(f"  Saved combined {metric_name} plot to: {combined_path}")
        plt.close()
        
        # Save individual plots for each pair
        for agent_i, agent_j in agent_pairs:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            pair_data = df[(df['agent_i'] == agent_i) & (df['agent_j'] == agent_j)]
            pair_data = pair_data.dropna(subset=['frob_norm', metric_col])
            
            if len(pair_data) == 0:
                plt.close()
                continue
            
            x = pair_data['frob_norm'].values
            y = pair_data[metric_col].values
            
            ax.scatter(x, y, alpha=0.3, s=20, c='blue')
            
            if len(x) > 1:
                try:
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    x_fit = np.linspace(x.min(), x.max(), 100)
                    ax.plot(x_fit, p(x_fit), "r--", linewidth=2, label=f'y={z[0]:.3f}x+{z[1]:.3f}')
                    
                    pearson_r, pearson_p = stats.pearsonr(x, y)
                    ax.text(0.05, 0.95, f'Pearson r = {pearson_r:.3f}\np-value = {pearson_p:.3e}',
                           transform=ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                           fontsize=12)
                    ax.legend(fontsize=11)
                except Exception:
                    pass
            
            ax.set_xlabel('||H||_F (Frobenius Norm)', fontsize=12)
            ax.set_ylabel(metric_label, fontsize=12)
            ax.set_title(f'SVD Coupling Analysis: Agent {agent_i} → Agent {agent_j}', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            individual_path = os.path.join(metric_dir, f"coupling_{metric_name}_pair_{agent_i}_to_{agent_j}.png")
            plt.savefig(individual_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"  Saved {len(agent_pairs)} individual {metric_name} plots to {metric_dir}")
    
    def _generate_xy_metric_plots(self, df, x_col, x_label, y_col, y_label, metric_name):
        """
        Generate scatter plots for an arbitrary x vs y metric pair per agent pair.

        Args:
            df: DataFrame containing coupling data
            x_col: Column name for the x-axis metric
            x_label: Display label for the x-axis
            y_col: Column name for the y-axis metric
            y_label: Display label for the y-axis
            metric_name: Short name used for file/folder naming
        """
        print(f"  Generating {x_col} vs {y_col} plots...")

        agent_pairs = df[['agent_i', 'agent_j']].drop_duplicates().values
        n_pairs = len(agent_pairs)
        n_cols = min(3, int(np.ceil(np.sqrt(n_pairs))))
        n_rows = int(np.ceil(n_pairs / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_pairs == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for idx, (agent_i, agent_j) in enumerate(agent_pairs):
            ax = axes[idx]
            pair_data = df[(df['agent_i'] == agent_i) & (df['agent_j'] == agent_j)]
            pair_data = pair_data.dropna(subset=[x_col, y_col])

            if len(pair_data) == 0:
                ax.text(0.5, 0.5, f'No valid data\nfor pair ({agent_i}, {agent_j})',
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                continue

            x = pair_data[x_col].values
            y = pair_data[y_col].values

            ax.scatter(x, y, alpha=0.3, s=10, c='steelblue')

            if len(x) > 1:
                try:
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    x_fit = np.linspace(x.min(), x.max(), 100)
                    ax.plot(x_fit, p(x_fit), 'r--', linewidth=2,
                            label=f'y={z[0]:.3f}x+{z[1]:.3f}')
                    pearson_r, pearson_p = stats.pearsonr(x, y)
                    ax.text(0.05, 0.95, f'r = {pearson_r:.3f}\np = {pearson_p:.3e}',
                            transform=ax.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    ax.legend()
                except Exception:
                    pass

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title(f'Agent {agent_i} \u2192 Agent {agent_j}')
            ax.grid(True, alpha=0.3)

        for idx in range(n_pairs, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()

        metric_dir = os.path.join(self.plots_dir, metric_name)
        os.makedirs(metric_dir, exist_ok=True)

        combined_path = os.path.join(metric_dir, f"coupling_analysis_{metric_name}_all_pairs.png")
        plt.savefig(combined_path, dpi=150, bbox_inches='tight')
        print(f"  Saved combined {metric_name} plot to: {combined_path}")
        plt.close()

        # Individual plots
        for agent_i, agent_j in agent_pairs:
            fig, ax = plt.subplots(figsize=(8, 6))
            pair_data = df[(df['agent_i'] == agent_i) & (df['agent_j'] == agent_j)]
            pair_data = pair_data.dropna(subset=[x_col, y_col])

            if len(pair_data) == 0:
                plt.close()
                continue

            x = pair_data[x_col].values
            y = pair_data[y_col].values

            ax.scatter(x, y, alpha=0.3, s=20, c='steelblue')

            if len(x) > 1:
                try:
                    z = np.polyfit(x, y, 1)
                    p = np.poly1d(z)
                    x_fit = np.linspace(x.min(), x.max(), 100)
                    ax.plot(x_fit, p(x_fit), 'r--', linewidth=2,
                            label=f'y={z[0]:.3f}x+{z[1]:.3f}')
                    pearson_r, pearson_p = stats.pearsonr(x, y)
                    ax.text(0.05, 0.95, f'Pearson r = {pearson_r:.3f}\np-value = {pearson_p:.3e}',
                            transform=ax.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                            fontsize=12)
                    ax.legend(fontsize=11)
                except Exception:
                    pass

            ax.set_xlabel(x_label, fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            ax.set_title(f'{y_label} vs {x_label}: Agent {agent_i} \u2192 Agent {agent_j}', fontsize=14)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            individual_path = os.path.join(metric_dir,
                f"coupling_{metric_name}_pair_{agent_i}_to_{agent_j}.png")
            plt.savefig(individual_path, dpi=150, bbox_inches='tight')
            plt.close()

        print(f"  Saved {len(agent_pairs)} individual {metric_name} plots to {metric_dir}")

    def _generate_3d_plots(self, df):
        """
        Generate 3D scatter plots of frob_norm vs grad_norm vs delta_critic2.
        
        Args:
            df: DataFrame containing coupling data
        """
        print(f"  Generating 3D plots...")
        
        # Import 3D plotting
        from mpl_toolkits.mplot3d import Axes3D
        
        # Create subfolder for 3D plots
        plots_3d_dir = os.path.join(self.plots_dir, '3d_plots')
        os.makedirs(plots_3d_dir, exist_ok=True)
        
        # Get unique agent pairs
        agent_pairs = df[['agent_i', 'agent_j']].drop_duplicates().values
        
        # Combined 3D plot for all pairs
        n_pairs = len(agent_pairs)
        n_cols = min(3, int(np.ceil(np.sqrt(n_pairs))))
        n_rows = int(np.ceil(n_pairs / n_cols))
        
        fig = plt.figure(figsize=(7*n_cols, 6*n_rows))
        
        for idx, (agent_i, agent_j) in enumerate(agent_pairs):
            ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')
            
            # Filter data for this agent pair
            pair_data = df[(df['agent_i'] == agent_i) & (df['agent_j'] == agent_j)]
            pair_data = pair_data.dropna(subset=['frob_norm', 'grad_norm', 'delta_critic2'])
            
            if len(pair_data) == 0:
                ax.text2D(0.5, 0.5, f'No valid data\nfor pair ({agent_i}, {agent_j})',
                         ha='center', va='center', transform=ax.transAxes)
                continue
            
            x = pair_data['frob_norm'].values
            y = pair_data['grad_norm'].values
            z = pair_data['delta_critic2'].values
            
            # Set explicit axis limits with padding to avoid illusions
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            z_min, z_max = z.min(), z.max()
            
            x_padding = (x_max - x_min) * 0.05 if x_max > x_min else 0.1
            y_padding = (y_max - y_min) * 0.05 if y_max > y_min else 0.1
            z_padding = (z_max - z_min) * 0.05 if z_max > z_min else 0.1
            
            ax.set_xlim([max(0, x_min - x_padding), x_max + x_padding])
            ax.set_ylim([max(0, y_min - y_padding), y_max + y_padding])
            ax.set_zlim([z_min - z_padding, z_max + z_padding])
            
            # Scatter plot with color gradient based on z-value
            scatter = ax.scatter(x, y, z, c=z, cmap='viridis', alpha=0.6, s=20, edgecolors='k', linewidth=0.3)
            
            # Set better viewing angle (elev=20, azim=45 reduces illusions)
            ax.view_init(elev=20, azim=45)
            
            # Use orthographic projection to eliminate perspective distortion
            ax.set_proj_type('ortho')
            
            # Add clearer grid
            ax.grid(True, alpha=0.3, linestyle='--')
            
            ax.set_xlabel('||H||_F (Frobenius Norm)', fontsize=10, labelpad=5)
            ax.set_ylabel('||g|| (Gradient Magnitude)', fontsize=10, labelpad=5)
            ax.set_zlabel('ΔQ₂ (Critic Shift)', fontsize=10, labelpad=5)
            ax.set_title(f'Agent {agent_i} → Agent {agent_j}', fontsize=11)
            
            # Add colorbar
            plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        
        plt.tight_layout()
        combined_3d_path = os.path.join(plots_3d_dir, "3d_coupling_all_pairs.png")
        plt.savefig(combined_3d_path, dpi=150, bbox_inches='tight')
        print(f"  Saved combined 3D plot to: {combined_3d_path}")
        plt.close()
        
        # Generate individual 3D plots for each pair
        for agent_i, agent_j in agent_pairs:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            pair_data = df[(df['agent_i'] == agent_i) & (df['agent_j'] == agent_j)]
            pair_data = pair_data.dropna(subset=['frob_norm', 'grad_norm', 'delta_critic2'])
            
            if len(pair_data) == 0:
                plt.close()
                continue
            
            x = pair_data['frob_norm'].values
            y = pair_data['grad_norm'].values
            z = pair_data['delta_critic2'].values
            
            # Set explicit axis limits with padding
            x_min, x_max = x.min(), x.max()
            y_min, y_max = y.min(), y.max()
            z_min, z_max = z.min(), z.max()
            
            x_padding = (x_max - x_min) * 0.05 if x_max > x_min else 0.1
            y_padding = (y_max - y_min) * 0.05 if y_max > y_min else 0.1
            z_padding = (z_max - z_min) * 0.05 if z_max > z_min else 0.1
            
            ax.set_xlim([max(0, x_min - x_padding), x_max + x_padding])
            ax.set_ylim([max(0, y_min - y_padding), y_max + y_padding])
            ax.set_zlim([z_min - z_padding, z_max + z_padding])
            
            scatter = ax.scatter(x, y, z, c=z, cmap='viridis', alpha=0.6, s=40, edgecolors='k', linewidth=0.5)
            
            # Set better viewing angle
            ax.view_init(elev=20, azim=45)
            
            # Use orthographic projection to eliminate perspective distortion
            ax.set_proj_type('ortho')
            
            # Add clearer grid
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Make panes slightly transparent
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
            ax.xaxis.pane.set_edgecolor('gray')
            ax.yaxis.pane.set_edgecolor('gray')
            ax.zaxis.pane.set_edgecolor('gray')
            ax.xaxis.pane.set_alpha(0.3)
            ax.yaxis.pane.set_alpha(0.3)
            ax.zaxis.pane.set_alpha(0.3)
            
            ax.set_xlabel('||H||_F (Frobenius Norm)', fontsize=12, labelpad=8)
            ax.set_ylabel('||g|| (Gradient Magnitude)', fontsize=12, labelpad=8)
            ax.set_zlabel('ΔQ₂ (Critic Value Shift)', fontsize=12, labelpad=8)
            ax.set_title(f'3D Coupling Analysis: Agent {agent_i} → Agent {agent_j}', fontsize=14, pad=20)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=10, pad=0.1)
            cbar.set_label('ΔQ₂', fontsize=11)
            
            plt.tight_layout()
            individual_3d_path = os.path.join(plots_3d_dir, f"3d_coupling_pair_{agent_i}_to_{agent_j}.png")
            plt.savefig(individual_3d_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"  Saved {len(agent_pairs)} individual 3D plots to {plots_3d_dir}")
    
    def save_results(self):
        """Save all results to CSV files and generate plots."""
        if len(self.coupling_data) == 0:
            print("No data collected. Skipping save.")
            return
        
        # Create DataFrame from collected data
        df = pd.DataFrame(self.coupling_data, columns=[
            'seed', 'timestep', 'agent_i', 'agent_j', 'frob_norm', 'grad_norm',
            'delta_g_norm', 'perturbed_grad_norm', 'delta_critic_j_only', 'delta_critic1', 'delta_critic2'
        ])
        
        # Compute adversarial assist metric
        df['delta_critic_diff'] = df['delta_critic2'] - df['delta_critic1']
        
        # Create CSV subfolder
        csv_dir = os.path.join(self.logdir, 'csv_data')
        os.makedirs(csv_dir, exist_ok=True)
        
        # Save raw data
        raw_csv_path = os.path.join(csv_dir, "raw_coupling_data.csv")
        df.to_csv(raw_csv_path, index=False)
        print(f"\nSaved raw coupling data to: {raw_csv_path}")
        
        # Compute statistics grouped by agent pair
        grouped = df.groupby(['agent_i', 'agent_j']).agg({
            'frob_norm': ['mean', 'std', 'count'],
            'grad_norm': ['mean', 'std', 'count'],
            'delta_g_norm': ['mean', 'std', 'count'],
            'perturbed_grad_norm': ['mean', 'std', 'count'],
            'delta_critic_j_only': ['mean', 'std', 'count'],
            'delta_critic1': ['mean', 'std', 'count'],
            'delta_critic2': ['mean', 'std', 'count'],
            'delta_critic_diff': ['mean', 'std', 'count']
        }).reset_index()
        
        # Flatten column names
        grouped.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                          for col in grouped.columns.values]
        
        # Save aggregated stats
        stats_csv_path = os.path.join(csv_dir, "mean_coupling_by_pair.csv")
        grouped.to_csv(stats_csv_path, index=False)
        print(f"Saved aggregated statistics to: {stats_csv_path}")
        
        # Save configuration
        config_dict = {
            'env_id': self.config.env_id,
            'model_path': self.config.model_path,
            'total_experiments': self.total_experiments,
            'epsilon': self.epsilon,
            'nagents': self.maddpg.nagents,
            'discrete_action': self.maddpg.discrete_action,
            'total_data_points': len(df)
        }
        
        config_path = os.path.join(self.logdir, "experiment_config.json")
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Saved experiment configuration to: {config_path}")
        
        # Generate plots
        self.generate_plots(df)
        
        # Print summary statistics
        print("\n" + "="*70)
        print("EXPERIMENT SUMMARY")
        print("="*70)
        print(f"Total seeds processed: {df['seed'].nunique()}")
        print(f"Total timesteps across all episodes: {len(df) // (self.maddpg.nagents ** 2)}")
        print(f"Total agent pairs: {self.maddpg.nagents ** 2}")
        print(f"Total data points collected: {len(df)}")
        print(f"\nPer-pair statistics:")
        print(grouped.to_string(index=False))
        
    def cleanup(self):
        """Clean up resources."""
        if self.env:
            self.env.close()
    
    def run_full_experiment(self):
        """Run the complete experiment pipeline."""
        self.setup_experiment()
        self.run_all_experiments()
        self.save_results()
        self.cleanup()
        print(f"\nExperiment completed successfully!")
        print(f"Results saved to: {self.logdir}")


def create_config_from_args():
    """Create configuration from command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-seed SVD-based gradient coupling analysis for PettingZoo environments"
    )
    parser.add_argument("env_id", help="Name of environment (e.g., 'simple_spread', 'simple_adversary')")
    parser.add_argument("model_path", help="Model directory")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="SVD perturbation magnitude (default: 0.01)")
    parser.add_argument("--total_experiments", type=int, default=100,
                        help="Total number of seed experiments to run (default: 100)")
    
    return parser.parse_args()


def main():
    """Main function to run multi-seed SVD coupling analysis experiment."""
    config = create_config_from_args()
    runner = SVDCouplingAnalysisRunner(config)
    runner.run_full_experiment()


if __name__ == '__main__':
    main()

