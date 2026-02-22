"""
Multi-seed Frobenius Norm Delta Analysis Script for PettingZoo Environments.

This script performs experiments across multiple seeds to analyze the impact
of perturbations on Frobenius norms of action influences between agent pairs.

For each seed:
1. Runs a normal episode without perturbation
2. Finds the timestep T with maximum Frobenius norm for a specified agent pair (i, j)
3. Runs a perturbed episode with agent i's observation perturbed at timesteps [T, T+4]
4. Collects delta (perturbed_frob - normal_frob) for all agent pairs at [T, T+4]
5. Computes mean delta across seeds for each timestep offset and agent pair
"""
import argparse
import os
import json
import random
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

from algorithms.maddpg import MADDPG
from modules.constants import DEVICE, torch_device
from modules.environment import create_environment
from modules.metrics import compute_pairwise_frob_norms
from torch.autograd import Variable

# Perturbation noise magnitude
PERTURBATION_SIGMA = 0.01


class FrobNormDeltaExperimentRunner:
    """
    Multi-seed experiment runner for analyzing Frobenius norm deltas from perturbations.
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
        self.target_agent_i = config.target_agent_i
        self.target_agent_j = config.target_agent_j
        
        # Storage for results across all seeds
        # Structure: {(agent_i, agent_j): {offset_k: [delta_values_across_seeds]}}
        self.delta_results = defaultdict(lambda: defaultdict(list))
        
    def setup_experiment(self):
        """Set up the experiment environment and logging."""
        # Load MADDPG model
        self.maddpg = MADDPG.init_from_save(self.config.model_path)
        
        # Create log directory
        cwd = os.getcwd()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        env_type = 'discrete' if self.maddpg.discrete_action else 'continuous'
        
        self.logdir = os.path.join(
            cwd, 'runs', 
            f"{self.config.env_id}_{env_type}_frob_norm_delta_stats",
            f"{timestamp}_nagents{self.maddpg.nagents}_target_{self.target_agent_i}_to_{self.target_agent_j}_seeds{self.total_experiments}"
        )
        os.makedirs(self.logdir, exist_ok=True)
        
        # Create environment
        self.env = create_environment(self.config, self.maddpg)
        
        # Prepare MADDPG for training mode to ensure all tensors are on the correct device
        device_str = 'gpu' if DEVICE == 'gpu' else 'cpu'
        self.maddpg.prep_training(device=device_str)
        
        print(f"Experiment setup complete. Log directory: {self.logdir}")
        print(f"Target pair: agent_{self.target_agent_i} -> agent_{self.target_agent_j}")
        print(f"Will run {self.total_experiments} experiments")
        
    def run_normal_episode(self, seed):
        """
        Run a normal episode without any perturbation.
        
        Args:
            seed: Random seed for the episode
            
        Returns:
            Dictionary containing:
                - frob_norms_history: List of Frobenius norm matrices at each timestep
                - episode_length: Total number of timesteps
        """
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        with torch.no_grad():
            self.maddpg.prep_rollouts(device=DEVICE)
        
        obs = self.env.reset(seed=seed)
        frob_norms_history = []
        
        while True:
            torch_obs = [Variable(torch.tensor([obs[i]], dtype=torch.float32).to(torch_device), 
                                requires_grad=False) 
                        for i in range(self.maddpg.nagents)]
            
            # Get actions (with no_grad for action selection to avoid gradient tracking)
            with torch.no_grad():
                torch_agent_actions = self.maddpg.step(torch_obs, explore=False)
                agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions]
            
            if self.maddpg.discrete_action:
                actions = {agent_name: agent_actions[i].argmax() 
                         for i, agent_name in enumerate(self.env.possible_agents)}
            else:
                actions = {agent_name: agent_actions[i].squeeze() 
                         for i, agent_name in enumerate(self.env.possible_agents)}
            
            # Compute Frobenius norms for all agent pairs
            frob_norms_matrix = compute_pairwise_frob_norms(
                self.maddpg, obs, list(actions.values()), self.env.action_space
            )
            frob_norms_history.append(frob_norms_matrix)
            
            # Environment step
            next_obs, rewards, dones, infos = self.env.step(actions)
            obs = next_obs
            
            # Check if episode is done
            if dones.all():
                break
        
        return {
            'frob_norms_history': frob_norms_history,
            'episode_length': len(frob_norms_history)
        }
    
    def find_max_frob_norm_timestep(self, frob_norms_history, agent_i, agent_j):
        """
        Find the timestep with maximum Frobenius norm for the specified agent pair.
        
        Args:
            frob_norms_history: List of Frobenius norm matrices
            agent_i: Influencing agent index
            agent_j: Influenced agent index
            
        Returns:
            Timestep with maximum Frobenius norm
        """
        max_frob = -float('inf')
        max_timestep = 0
        
        for t, frob_matrix in enumerate(frob_norms_history):
            frob_value = frob_matrix[agent_i][agent_j]
            if frob_value > max_frob:
                max_frob = frob_value
                max_timestep = t
        
        return max_timestep

    def compute_pairwise_frob_norm_tensor(self, torch_obs, torch_actions, agent_i, agent_j):
        """
        Compute differentiable Frobenius norm of cross-agent Hessian block for (i, j).

        Args:
            torch_obs: List of torch tensors for observations
            torch_actions: List of torch tensors for actions (requires_grad enabled)
            agent_i: Influencing agent index
            agent_j: Influenced agent index

        Returns:
            Torch scalar tensor for || ∂²v_i / (∂a_i ∂a_j) ||_F
        """
        vf_in = torch.cat((*torch_obs, *torch_actions), dim=1)
        critic_val = self.maddpg.agents[agent_i].critic(vf_in).mean()
        grad_i = torch.autograd.grad(
            critic_val,
            torch_actions[agent_i],
            create_graph=True,
            retain_graph=True
        )[0]

        hessian_matrix = []
        for k in range(grad_i.shape[1]):
            second_grad = torch.autograd.grad(
                grad_i[0, k],
                torch_actions[agent_j],
                retain_graph=True,
                allow_unused=True,
                create_graph=True
            )[0]
            hessian_matrix.append(second_grad.flatten())

        H = torch.stack(hessian_matrix)
        return H.norm(p='fro')
    
    def run_perturbed_episode(self, seed, attack_timestep):
        """
        Run an episode with observation perturbation at [T, T+4].
        
        Args:
            seed: Random seed for the episode
            attack_timestep: Starting timestep for perturbation (T)
            
        Returns:
            Dictionary containing:
                - frob_norms_history: List of Frobenius norm matrices at each timestep
                - episode_length: Total number of timesteps
        """
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        with torch.no_grad():
            self.maddpg.prep_rollouts(device=DEVICE)
        
        obs = self.env.reset(seed=seed)
        frob_norms_history = []
        timestep = 0
        perturb_window = range(attack_timestep, attack_timestep + 5)  # [T, T+4]
        
        while True:
            torch_obs = [Variable(torch.tensor([obs[i]], dtype=torch.float32).to(torch_device), 
                                requires_grad=False) 
                        for i in range(self.maddpg.nagents)]
            
            # Apply perturbation to target agent's observation if in attack window
            # if timestep in perturb_window:
                # # Perturb agent i's observation
                # perturbed_obs = obs[self.target_agent_i].copy()
                # noise = np.random.normal(0, K_SIGMA, size=perturbed_obs.shape)
                # perturbed_obs = perturbed_obs + noise
                
                # # Update the observation for the attacked agent
                # obs_list = list(obs)
                # obs_list[self.target_agent_i] = perturbed_obs
                # obs = obs_list
                
                # # Update torch_obs with perturbed observation
                # torch_obs[self.target_agent_i] = Variable(
                #     torch.tensor([perturbed_obs], dtype=torch.float32).to(torch_device),
                #     requires_grad=False
                # )
            
            # Get actions (with no_grad for action selection to avoid gradient tracking)
            if self.maddpg.discrete_action:
                with torch.no_grad():
                    torch_agent_actions = self.maddpg.step(torch_obs, explore=False)
                    agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions]

                actions = {agent_name: agent_actions[i].argmax() 
                         for i, agent_name in enumerate(self.env.possible_agents)}
                # pick the action with minimum value to simulate perturbation in discrete case
                if timestep in perturb_window:
                    actions[self.env.possible_agents[self.target_agent_i]] = agent_actions[self.target_agent_i].argmin() 
            else:
                if timestep in perturb_window:
                    torch_agent_actions = self.maddpg.step(torch_obs, explore=False)
                    for ac in torch_agent_actions:
                        ac.requires_grad_(True)

                    # Compute gradient of Frobenius norm w.r.t. target agent action
                    # objective_val = self.compute_pairwise_frob_norm_tensor(
                    #     torch_obs,
                    #     torch_agent_actions,
                    #     self.target_agent_i,
                    #     self.target_agent_j
                    # )

                    # critic val grad (maximize neg critic via grad ascent because pertubation should reduce the q value)
                    vf_in = torch.cat((*torch_obs, *torch_agent_actions), dim=1)
                    objective_val = -self.maddpg.agents[self.target_agent_i].critic(vf_in).mean()

                    grad = torch.autograd.grad(
                        objective_val,
                        torch_agent_actions[self.target_agent_i],
                        allow_unused=True
                    )[0]

                    grad_norm = torch.norm(grad, p=2)
                    p_step = PERTURBATION_SIGMA * grad / (grad_norm + 1e-8)

                    perturbed_action_torch = torch_agent_actions[self.target_agent_i] + p_step
                    action_low = torch.tensor(
                        self.env.action_space[self.target_agent_i].low,
                        device=perturbed_action_torch.device, dtype=perturbed_action_torch.dtype
                    )
                    action_high = torch.tensor(
                        self.env.action_space[self.target_agent_i].high,
                        device=perturbed_action_torch.device, dtype=perturbed_action_torch.dtype
                    )
                    perturbed_action_torch = perturbed_action_torch.clamp(action_low, action_high)

                    agent_actions = [ac.detach().cpu().numpy() for ac in torch_agent_actions]
                    agent_actions[self.target_agent_i] = perturbed_action_torch.detach().cpu().numpy()
                else:
                    with torch.no_grad():
                        torch_agent_actions = self.maddpg.step(torch_obs, explore=False)
                        agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions]
                
                actions = {agent_name: agent_actions[i].squeeze() 
                         for i, agent_name in enumerate(self.env.possible_agents)}
            
            # Compute Frobenius norms for all agent pairs
            frob_norms_matrix = compute_pairwise_frob_norms(
                self.maddpg, obs, list(actions.values()), self.env.action_space
            )
            frob_norms_history.append(frob_norms_matrix)
            
            # Environment step
            next_obs, rewards, dones, infos = self.env.step(actions)
            obs = next_obs
            timestep += 1
            
            if dones.all():
                break
        
        return {
            'frob_norms_history': frob_norms_history,
            'episode_length': len(frob_norms_history)
        }
    
    def run_single_seed_experiment(self, seed):
        """
        Run complete experiment for a single seed.
        
        Args:
            seed: Random seed for the experiment
            
        Returns:
            Dictionary containing delta results for this seed
        """
        print(f"\n{'='*50}")
        print(f"Running experiment for seed {seed}")
        print(f"{'='*50}")
        
        # Step 1: Run normal episode
        normal_episode = self.run_normal_episode(seed)
        normal_frob_history = normal_episode['frob_norms_history']
        episode_length = normal_episode['episode_length']
        
        # Step 2: Find timestep with maximum Frobenius norm for target pair
        max_frob_timestep = self.find_max_frob_norm_timestep(
            normal_frob_history, 
            self.target_agent_i, 
            self.target_agent_j
        )
        
        # Check if we have enough timesteps for the perturbation window
        if max_frob_timestep + 5 > episode_length:
            print(f"Warning: Max frob timestep {max_frob_timestep} too close to episode end ({episode_length})")
            print(f"Skipping seed {seed}")
            return None
        
        print(f"Episode length: {episode_length}")
        print(f"Max Frobenius norm timestep for pair ({self.target_agent_i}, {self.target_agent_j}): {max_frob_timestep}")
        print(f"Max Frobenius norm value: {normal_frob_history[max_frob_timestep][self.target_agent_i][self.target_agent_j]:.6f}")
        
        # Step 3: Run perturbed episode
        perturbed_episode = self.run_perturbed_episode(seed, max_frob_timestep)
        perturbed_frob_history = perturbed_episode['frob_norms_history']
        
        # Step 4: Collect deltas for all agent pairs at [T, T+4]
        seed_results = {}
        for offset_k in range(5):  # offsets 0, 1, 2, 3, 4
            timestep = max_frob_timestep + offset_k
            
            if timestep >= min(len(normal_frob_history), len(perturbed_frob_history)):
                continue
            
            for agent_i in range(self.maddpg.nagents):
                for agent_j in range(self.maddpg.nagents):
                    normal_frob = normal_frob_history[timestep][agent_i][agent_j]
                    perturbed_frob = perturbed_frob_history[timestep][agent_i][agent_j]
                    delta = perturbed_frob - normal_frob
                    
                    pair_key = (agent_i, agent_j)
                    if pair_key not in seed_results:
                        seed_results[pair_key] = {}
                    seed_results[pair_key][offset_k] = delta
        
        return {
            'seed': seed,
            'max_frob_timestep': max_frob_timestep,
            'episode_length': episode_length,
            'deltas': seed_results
        }
    
    def run_all_experiments(self):
        """Run experiments for all seeds."""
        if self.maddpg is None:
            raise RuntimeError("MADDPG model not loaded. Call setup_experiment() first.")
        
        print(f"Starting multi-seed experiments with {self.total_experiments} seeds...")
        
        successful_seeds = 0
        failed_seeds = []
        
        for seed in tqdm(range(self.total_experiments), desc="Running experiments"):
            result = self.run_single_seed_experiment(seed)
            
            if result is not None:
                # Store delta results for this seed
                for pair_key, offset_deltas in result['deltas'].items():
                    for offset_k, delta_value in offset_deltas.items():
                        self.delta_results[pair_key][offset_k].append(delta_value)
                
                successful_seeds += 1
            else:
                failed_seeds.append(seed)
        
        print(f"\nCompleted {successful_seeds} successful experiments out of {self.total_experiments}")
        if failed_seeds:
            print(f"Failed seeds: {failed_seeds}")
    
    def compute_mean_deltas(self):
        """
        Compute mean delta across seeds for each agent pair and timestep offset.
        
        Returns:
            Dictionary mapping (agent_i, agent_j, offset_k) -> mean_delta
        """
        mean_results = {}
        
        for pair_key, offset_dict in self.delta_results.items():
            agent_i, agent_j = pair_key
            for offset_k, delta_values in offset_dict.items():
                if len(delta_values) > 0:
                    mean_delta = np.mean(delta_values)
                    std_delta = np.std(delta_values)
                    n_samples = len(delta_values)
                    
                    mean_results[(agent_i, agent_j, offset_k)] = {
                        'mean': mean_delta,
                        'std': std_delta,
                        'n_samples': n_samples
                    }
        
        return mean_results
    
    def save_results(self):
        """Save all results to CSV files."""
        # Compute mean deltas
        mean_results = self.compute_mean_deltas()
        
        # Create DataFrame for mean results
        rows = []
        for (agent_i, agent_j, offset_k), stats in mean_results.items():
            rows.append({
                'agent_i': agent_i,
                'agent_j': agent_j,
                'offset_k': offset_k,
                'mean_delta': stats['mean'],
                'std_delta': stats['std'],
                'n_samples': stats['n_samples']
            })
        
        df_mean = pd.DataFrame(rows)
        df_mean = df_mean.sort_values(['agent_i', 'agent_j', 'offset_k'])
        
        # Save mean results
        mean_csv_path = os.path.join(self.logdir, "mean_frob_norm_deltas.csv")
        df_mean.to_csv(mean_csv_path, index=False)
        print(f"\nSaved mean delta results to: {mean_csv_path}")
        
        # Create DataFrame for raw delta values (all seeds)
        raw_rows = []
        for pair_key, offset_dict in self.delta_results.items():
            agent_i, agent_j = pair_key
            for offset_k, delta_values in offset_dict.items():
                for seed_idx, delta_value in enumerate(delta_values):
                    raw_rows.append({
                        'agent_i': agent_i,
                        'agent_j': agent_j,
                        'offset_k': offset_k,
                        'seed_idx': seed_idx,
                        'delta_value': delta_value
                    })
        
        df_raw = pd.DataFrame(raw_rows)
        df_raw = df_raw.sort_values(['agent_i', 'agent_j', 'offset_k', 'seed_idx'])
        
        # Save raw results
        raw_csv_path = os.path.join(self.logdir, "raw_frob_norm_deltas.csv")
        df_raw.to_csv(raw_csv_path, index=False)
        print(f"Saved raw delta results to: {raw_csv_path}")
        
        # Save configuration
        config_dict = {
            'env_id': self.config.env_id,
            'model_path': self.config.model_path,
            'total_experiments': self.total_experiments,
            'target_agent_i': self.target_agent_i,
            'target_agent_j': self.target_agent_j,
            'nagents': self.maddpg.nagents
        }
        
        config_path = os.path.join(self.logdir, "experiment_config.json")
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Saved experiment configuration to: {config_path}")
        
        # Print summary statistics
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Target pair: agent_{self.target_agent_i} -> agent_{self.target_agent_j}")
        print(f"Total agent pairs analyzed: {len(self.delta_results)}")
        print(f"Total data points: {len(raw_rows)}")
        print("\nMean deltas for target pair:")
        target_data = df_mean[(df_mean['agent_i'] == self.target_agent_i) & 
                             (df_mean['agent_j'] == self.target_agent_j)]
        if not target_data.empty:
            print(target_data.to_string(index=False))
        else:
            print("No data available for target pair")
        
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
        description="Multi-seed Frobenius norm delta analysis for PettingZoo environments"
    )
    parser.add_argument("env_id", help="Name of environment (e.g., 'simple_spread', 'simple_adversary')")
    parser.add_argument("model_path", help="Model directory")
    parser.add_argument("target_agent_i", type=int, help="Index of influencing agent in target pair")
    parser.add_argument("target_agent_j", type=int, help="Index of influenced agent in target pair")
    parser.add_argument("--total_experiments", type=int, default=100,
                        help="Total number of seed experiments to run (default: 100)")
    
    return parser.parse_args()


def main():
    """Main function to run multi-seed Frobenius norm delta experiment."""
    config = create_config_from_args()
    runner = FrobNormDeltaExperimentRunner(config)
    runner.run_full_experiment()


if __name__ == '__main__':
    main()
