"""
Multi-seed statistics experiment for analyzing action influence-based attacks.
This script performs experiments across multiple seeds to evaluate the effectiveness
of attacking at high vs low influence timesteps.
"""
import argparse
import os
import math
import csv
import random
import numpy as np
import torch
import pickle
import json
from datetime import datetime
from collections import deque
from tqdm import tqdm
from torch.autograd import Variable
import sys
import traceback

from algorithms.maddpg import MADDPG
from utils.smac_wrapper import SmacWrapper

# Import all the modular components
from modules.constants import DEVICE, K_SIGMA, torch_device
from modules.detection import get_patient_zero_detection
from modules.core_experiment import get_episode_data
from modules.metrics import (
    compute_taylor_delta_policy,
    compute_pairwise_action_influences,
    collect_agent_q_values,
    compute_pairwise_frob_norms,
    compute_pairwise_action_directional_second_derivatives
)
from modules.traceback import PatientZeroAnalyzer

# Define CUDA constants
USE_CUDA = torch.cuda.is_available()
DEVICE = 'gpu' if USE_CUDA else 'cpu'
torch_device = torch.device("cuda" if USE_CUDA else "cpu")

REF_TAYLOR_EPISODE_COUNT = 100
WATCH_WINDOW = 15  # Number of timesteps to watch after attack timestep
ATTACK_TS_FRACTION = 0.25  # Fraction of episode to consider for attack timesteps


def get_agent_fault_detection_times(fault_timeline, agent_id):
    """
    Get all fault detection times for a specific agent from the fault timeline.
    
    Args:
        fault_timeline: List of fault detection events
        agent_id: ID of the agent to get detection times for
        
    Returns:
        List of detection times for the specified agent (empty list if no detections)
    """
    if not fault_timeline:
        return []
    
    detection_times = []
    for event in fault_timeline:
        if event.get('agent') == agent_id:
            detection_times.append(event.get('t'))
    
    return sorted(detection_times)  # Sort chronologically


class MultiSeedExperimentRunner:
    """
    Multi-seed experiment runner for analyzing influence-based attacks.
    """
    
    def __init__(self, config):
        """
        Initialize the multi-seed experiment runner.
        
        Args:
            config: Configuration object containing experiment parameters
        """
        self.config = config
        self.maddpg = None
        self.logdir = None
        self.total_experiments = config.total_experiments
        self.gamma = 0.99  # Discount factor for weighted metrics
        
        # Results storage
        self.experiment_results = []
        self.failed_seeds = []
        self.cumulative_influences_data = []  # Store cumulative influence data for all seeds
        
        # Reference Taylor error cache (loaded once and reused)
        self.reference_taylor_cache = None
        
        # Patient zero analysis (will be initialized after setup)
        self.patient_zero_analyzer = None
        
    def setup_experiment(self):
        """Set up the experiment environment and logging."""
        # Load MADDPG model
        self.maddpg = MADDPG.init_from_save(self.config.model_path, test_mode=True)
        
        # Create log directory
        cwd = os.getcwd()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        env_type = 'discrete' if self.maddpg.discrete_action else 'continuous'
        self.logdir = os.path.join(cwd, 'runs', f"{self.config.map_name}_{env_type}_multi_seed_detection_stats", 
                                  f"{timestamp}_nagents{self.maddpg.nagents}_total_experiments{self.total_experiments}")
        os.makedirs(self.logdir, exist_ok=True)
        
        # Prepare MADDPG for training mode
        device_str = 'gpu' if DEVICE == 'gpu' else 'cpu'
        self.maddpg.prep_training(device=device_str)
        
        # Initialize patient zero analyzer
        self.patient_zero_analyzer = PatientZeroAnalyzer(self.maddpg.nagents)
        
        # Load reference Taylor values once if directory is provided
        if hasattr(self.config, 'taylor_ref_dir') and self.config.taylor_ref_dir is not None:
            self._load_reference_taylor_cache()
        
        print(f"Multi-seed experiment setup complete. Log directory: {self.logdir}")
        print(f"Will run {self.total_experiments} experiments")
        
    def compute_reference_taylor_error(self, seed):
        """
        Compute reference Taylor error values for a given seed.
        Based on compute_ref_taylor_error.py but returns values directly.
        
        Args:
            seed: Random seed for the experiment
            
        Returns:
            Tuple of (ref_vals, ref_std_devs) for each agent and timestep
        """
        print(f"Computing reference Taylor error for seed {seed}...")
        
        # Set random seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        total_episodes = REF_TAYLOR_EPISODE_COUNT  # Reduced from 5000 for faster computation
        result_dataset = [{} for _ in range(self.maddpg.nagents)]
        
        for episode in tqdm(range(total_episodes), desc=f"Reference episodes (seed {seed})"):
            # Reset environment (seed is set before the loop)
            env = SmacWrapper.make_env(self.config.map_name, seed=seed)
            obs, action_masks = env.reset()
            result_deques = [deque(maxlen=5) for _ in range(self.maddpg.nagents)]
            timestep = 0
            
            while True:
                noise_scale = 1e-4
                noise = [np.random.normal(loc=0, scale=noise_scale, size=obs[i].shape) for i in range(self.maddpg.nagents)]
                obs = [obs[i] + noise[i] for i in range(self.maddpg.nagents)]

                # Add noise 10% of the time
                torch_obs = [Variable(torch.tensor([obs[i]], dtype=torch.float32).to(torch_device), requires_grad=False) 
                           for i in range(self.maddpg.nagents)]
                torch_masks = [Variable(torch.tensor(action_masks[i]).to(torch_device), requires_grad=False)
                               if action_masks[i] is not None else None for i in range(self.maddpg.nagents)]
                
                # Get actions
                torch_agent_actions = self.maddpg.step(torch_obs, explore=False, action_masks=torch_masks)
                agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions]
                
                if self.maddpg.discrete_action:
                    actions = {agent_name: agent_actions[i].argmax() 
                             for i, agent_name in enumerate(env.possible_agents)}
                else:
                    actions = {agent_name: agent_actions[i].squeeze() 
                             for i, agent_name in enumerate(env.possible_agents)}
                
                # Compute Taylor delta policy
                results = compute_taylor_delta_policy(
                    self.maddpg, obs, list(actions.values()), env.action_space, 0.01
                )
                
                # Store results for each agent at this timestep
                for agent_id in range(self.maddpg.nagents):
                    result_deques[agent_id].append(results[agent_id])
                    if timestep not in result_dataset[agent_id]:
                        result_dataset[agent_id][timestep] = []
                    result_dataset[agent_id][timestep].append(np.mean(result_deques[agent_id]))
                
                # Environment step
                next_obs, rewards, dones, infos, action_masks = env.step(actions)
                obs = next_obs
                timestep += 1
                
                if dones.all():
                    break
            env.close()
        
        # Compute reference values and standard deviations
        ref_vals = [[] for _ in range(self.maddpg.nagents)]
        ref_std_devs = [[] for _ in range(self.maddpg.nagents)]
        
        for agent_id in range(self.maddpg.nagents):
            sorted_timesteps = sorted(result_dataset[agent_id].keys())
            for timestep in sorted_timesteps:
                timestep_values = result_dataset[agent_id][timestep]
                mean_val = np.mean(timestep_values)
                std_val = np.std(timestep_values)
                ref_vals[agent_id].append(mean_val)
                ref_std_devs[agent_id].append(std_val)
        
        return ref_vals, ref_std_devs
    
    def _load_reference_taylor_cache(self):
        """
        Load reference Taylor error values from disk once and cache them.
        This method is called during setup to avoid loading the file multiple times.
        """
        pickle_filename = "ref_taylor_all_seeds.pkl"
        pickle_filepath = os.path.join(self.config.taylor_ref_dir, pickle_filename)
        
        if not os.path.exists(pickle_filepath):
            raise FileNotFoundError(f"Reference Taylor values not found: {pickle_filepath}")
        
        print(f"Loading reference Taylor values cache from {pickle_filepath}...")
        
        with open(pickle_filepath, 'rb') as f:
            self.reference_taylor_cache = pickle.load(f)
        
        # Validate compatibility
        if self.reference_taylor_cache['nagents'] != self.maddpg.nagents:
            raise ValueError(f"Number of agents mismatch: saved={self.reference_taylor_cache['nagents']}, current={self.maddpg.nagents}")
        
        if self.reference_taylor_cache['env_id'] != self.config.map_name:
            print(f"Warning: Environment ID mismatch: saved={self.reference_taylor_cache['env_id']}, current={self.config.map_name}")
        
        print(f"Successfully loaded reference Taylor values cache:")
        print(f"  Episode count used: {self.reference_taylor_cache['ref_taylor_episode_count']}")
        print(f"  Number of agents: {self.reference_taylor_cache['nagents']}")
        print(f"  Total seeds in file: {self.reference_taylor_cache['total_seeds']}")
        print(f"  Available seeds: {sorted(self.reference_taylor_cache['seeds'])}")
    
    def save_all_reference_taylor_values(self, all_seeds_data, save_dir):
        """
        Save reference Taylor error values for all seeds to disk in a single file.
        
        Args:
            all_seeds_data: Dictionary mapping seed -> (ref_vals, ref_std_devs)
            save_dir: Directory to save the reference values
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Prepare data structure for all seeds
        data = {
            'seeds_data': all_seeds_data,
            'seeds': list(all_seeds_data.keys()),
            'total_seeds': len(all_seeds_data),
            'nagents': self.maddpg.nagents,
            'env_id': self.config.map_name,
            'model_path': self.config.model_path,
            'ref_taylor_episode_count': REF_TAYLOR_EPISODE_COUNT,
            'save_timestamp': datetime.now().isoformat()
        }
        
        # Save as pickle file for fast loading
        pickle_filename = "ref_taylor_all_seeds.pkl"
        pickle_filepath = os.path.join(save_dir, pickle_filename)
        
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(data, f)
        
        # Also save metadata as JSON for human readability
        json_data = {
            'seeds': list(all_seeds_data.keys()),
            'total_seeds': len(all_seeds_data),
            'nagents': self.maddpg.nagents,
            'env_id': self.config.map_name,
            'model_path': self.config.model_path,
            'ref_taylor_episode_count': REF_TAYLOR_EPISODE_COUNT,
            'save_timestamp': datetime.now().isoformat()
        }
        
        json_filename = "ref_taylor_all_seeds_metadata.json"
        json_filepath = os.path.join(save_dir, json_filename)
        
        with open(json_filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Saved reference Taylor values for {len(all_seeds_data)} seeds to:")
        print(f"  Pickle: {pickle_filepath}")
        print(f"  Metadata: {json_filepath}")
    
    def load_reference_taylor_values(self, seed, load_dir):
        """
        Load reference Taylor error values from cached data.
        
        Args:
            seed: Random seed to load values for
            load_dir: Directory containing the saved reference values (unused when cache is available)
            
        Returns:
            Tuple of (ref_vals, ref_std_devs) for each agent and timestep
        """
        ref_vals, ref_std_devs = self.reference_taylor_cache['seeds_data'][seed]
        
        print(f"Using cached reference Taylor values for seed {seed}")
        return ref_vals, ref_std_devs
    
    def precompute_and_save_taylor_values(self, seeds, save_dir):
        """
        Precompute reference Taylor values for multiple seeds and save them.
        
        Args:
            seeds: List of seeds to precompute values for
            save_dir: Directory to save the precomputed values
        """
        print(f"Precomputing reference Taylor values for {len(seeds)} seeds...")
        print(f"Save directory: {save_dir}")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Collect all seeds data in a dictionary
        all_seeds_data = {}
        
        for i, seed in enumerate(tqdm(seeds, desc="Precomputing Taylor values")):
            print(f"\nPrecomputing seed {seed} ({i+1}/{len(seeds)})...")
            ref_vals, ref_std_devs = self.compute_reference_taylor_error(seed)
            all_seeds_data[seed] = (ref_vals, ref_std_devs)
        
        # Save all seeds data to a single file
        self.save_all_reference_taylor_values(all_seeds_data, save_dir)
        
        # Save metadata about the precomputation
        metadata = {
            'seeds': seeds,
            'total_seeds': len(seeds),
            'nagents': self.maddpg.nagents,
            'env_id': self.config.map_name,
            'model_path': self.config.model_path,
            'ref_taylor_episode_count': REF_TAYLOR_EPISODE_COUNT,
            'precomputation_timestamp': datetime.now().isoformat()
        }
        
        metadata_filepath = os.path.join(save_dir, "precomputation_metadata.json")
        with open(metadata_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nPrecomputation completed!")
        print(f"Metadata saved to: {metadata_filepath}")
    
    def run_normal_episode(self, seed):
        """
        Run a normal episode without attacks to collect action influences.
        
        Args:
            seed: Random seed for the episode
            
        Returns:
            Dictionary containing episode data including action influences
        """
        # Reset directional derivatives history for this episode
        self.directional_derivatives_history = []
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        env = SmacWrapper.make_env(self.config.map_name, seed=seed)
        obs, action_masks = env.reset()
        action_influences_history = []
        q_values_history = []
        rewards_history = []  # Store rewards for each timestep
        timestep = 0
        episode_reward = 0
        
        while True:
            torch_obs = [Variable(torch.tensor([obs[i]], dtype=torch.float32).to(torch_device), requires_grad=False) 
                        for i in range(self.maddpg.nagents)]
            torch_masks = [Variable(torch.tensor(action_masks[i]).to(torch_device), requires_grad=False)
                           if action_masks[i] is not None else None for i in range(self.maddpg.nagents)]
            
            # Get actions
            torch_agent_actions = self.maddpg.step(torch_obs, explore=False, action_masks=torch_masks)
            agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions]
            
            if self.maddpg.discrete_action:
                actions = {agent_name: agent_actions[i].argmax() 
                         for i, agent_name in enumerate(env.possible_agents)}
            else:
                actions = {agent_name: agent_actions[i].squeeze() 
                         for i, agent_name in enumerate(env.possible_agents)}
            
            # Compute pairwise action influences
            pairwise_action_influences = compute_pairwise_action_influences(
                self.maddpg, obs, list(actions.values()), env.action_space
            )
            # pairwise_action_influences = compute_pairwise_frob_norms(
            #     self.maddpg, obs, list(actions.values()), env.action_space
            # )
            action_influences_history.append(pairwise_action_influences)
            
            # Compute directional second derivatives for influence analysis
            directional_second_derivatives = compute_pairwise_action_directional_second_derivatives(
                self.maddpg, obs, list(actions.values()), env.action_space
            )
            self.directional_derivatives_history.append(directional_second_derivatives)
            
            # Collect Q-values for normal episode
            q_values = collect_agent_q_values(
                self.maddpg, obs, list(actions.values()), env.action_space
            )
            q_values_history.append(q_values)
            
            # Environment step
            next_obs, rewards, dones, infos, action_masks = env.step(actions)
            
            # Store rewards for each agent at this timestep
            agent_rewards = np.array(rewards).squeeze()
            episode_reward += np.sum(agent_rewards)
            rewards_history.append(agent_rewards)
            
            obs = next_obs
            timestep += 1
            
            if dones.all():
                break
        env.close()
        
        print("Episode reward (normal):", episode_reward)
        return {
            'action_influences_history': action_influences_history,
            'directional_derivatives_history': self.directional_derivatives_history,
            'q_values_history': q_values_history,
            'rewards_history': rewards_history,
            'episode_length': timestep
        }
    
    def find_influence_timesteps(self, action_influences_history, directional_derivatives_history, agent_i, agent_j, atk_steps_limit, k_steps):
        """
        Find max and min influence timesteps of agent i on agent j in first 25% of episode.
        Uses directional second derivatives to filter timesteps:
        - High influence: positive directional second derivative + maximum action influence
        - Low influence: negative directional second derivative + minimum action influence
        
        Args:
            action_influences_history: List of action influence matrices
            directional_derivatives_history: List of directional second derivative matrices
            agent_i: Index of influencing agent
            agent_j: Index of influenced agent (where action_influences_matrix[t][j][i] = influence of i on j)
            atk_steps_limit: Last step that can be attacked
            k_steps: Number of timesteps to return (currently expecting 1)
            
        Returns:
            Tuple of (max_influence_timestep, min_influence_timestep)
        """
        positive_derivative_timesteps = []  # For high influence selection
        negative_derivative_timesteps = []  # For low influence selection
        
        for t in range(min(atk_steps_limit, len(action_influences_history), len(directional_derivatives_history))):
            # Get action influence of agent_i on agent_j at timestep t
            influence = abs(action_influences_history[t][agent_j][agent_i])
            
            # Get directional second derivative of agent_i on agent_j at timestep t
            directional_derivative = directional_derivatives_history[t][agent_j][agent_i]
            
            if directional_derivative > 0:
                positive_derivative_timesteps.append((influence, t))
            elif directional_derivative < 0:
                negative_derivative_timesteps.append((influence, t))
        
        # For high influence: among positive derivative timesteps, choose maximum action influence
        max_influences_t = []
        if positive_derivative_timesteps:
            positive_derivative_timesteps.sort(key=lambda x: x[0], reverse=True)  # Sort by influence, descending
            max_influences_t = [t for _, t in positive_derivative_timesteps[:k_steps]]
        
        # For low influence: among negative derivative timesteps, choose minimum action influence
        min_influences_t = []
        if negative_derivative_timesteps:
            negative_derivative_timesteps.sort(key=lambda x: x[0])  # Sort by influence, ascending
            min_influences_t = [t for _, t in negative_derivative_timesteps[:k_steps]]
        
        max_influences_t.sort()
        min_influences_t.sort()

        return max_influences_t, min_influences_t

    def run_attacked_episode(self, seed, attack_agent_i, attack_timesteps, ref_vals, ref_std_devs, observe_agent=None):
        """
        Run an episode with attack on specific agent at specific timesteps.
        
        Args:
            seed: Random seed for the episode
            attack_agent_i: Agent to attack (the influencing agent)
            attack_timesteps: List of timesteps to attack
            ref_vals: Reference Taylor error values
            ref_std_devs: Reference standard deviations
            observe_agent: Agent to observe impact on (the influenced agent). If None, observe attack_agent_i
            
        Returns:
            Dictionary containing attack episode results
        """
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # Determine which agent to observe impact on
        observe_agent_id = observe_agent if observe_agent is not None else attack_agent_i
        
        env = SmacWrapper.make_env(self.config.map_name, seed=seed)
        obs, action_masks = env.reset()
        episode_reward = 0
        
        # Initialize tracking
        result_deques = [deque(maxlen=5) for _ in range(self.maddpg.nagents)]
        fault_timeline = []
        fault_first_detected = {}
        q_values_history = []
        rewards_history = []  # Store rewards for attacked episode
        taylor_errors_history = []
        timestep = 0
        
        while True:
            torch_obs = [Variable(torch.tensor([obs[i]], dtype=torch.float32).to(torch_device), requires_grad=False) 
                        for i in range(self.maddpg.nagents)]
            torch_masks = [Variable(torch.tensor(action_masks[i]).to(torch_device), requires_grad=False)
                           if action_masks[i] is not None else None for i in range(self.maddpg.nagents)]
            
            # Get actions
            torch_agent_actions = self.maddpg.step(torch_obs, explore=False, action_masks=torch_masks)
            agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions]
            
            if self.maddpg.discrete_action:
                actions = {agent_name: agent_actions[i].argmax() 
                         for i, agent_name in enumerate(env.possible_agents)}
            else:
                actions = {agent_name: agent_actions[i].squeeze() 
                         for i, agent_name in enumerate(env.possible_agents)}
            
            # Apply attack if current timestep is in any attack window - attack the influencing agent (attack_agent_i)
            attack_active = False
            for attack_t in attack_timesteps:
                if attack_t <= timestep < attack_t + 5:
                    attack_active = True
                    break
            # if timestep in attack_timesteps:
            if attack_active:
                if self.maddpg.discrete_action:
                    # Worst action attack for discrete action space
                    action_logits = self.maddpg.get_action_logits(torch_obs, torch_masks)
                    
                    # Apply action mask - set invalid actions to very high values so they won't be selected as minimum
                    masked_logits = action_logits[attack_agent_i].clone().squeeze()
                    masked_logits[action_masks[attack_agent_i] == 0] = float('inf')
                    worst_action = torch.argmin(masked_logits).item()
                    actions[env.possible_agents[attack_agent_i]] = worst_action
                else:
                    # Random action for continuous space
                    actions[env.possible_agents[attack_agent_i]] = \
                        env.action_space[attack_agent_i].sample()
            
            # Compute Taylor errors
            taylor_results = compute_taylor_delta_policy(
                self.maddpg, obs, list(actions.values()), env.action_space, 0.01
            )
            taylor_errors_history.append(taylor_results)
            
            # Collect Q-values
            q_values = collect_agent_q_values(
                self.maddpg, obs, list(actions.values()), env.action_space
            )
            q_values_history.append(q_values)
            
            # Process detection for each agent
            for i in range(self.maddpg.nagents):
                result_deques[i].append(taylor_results[i])
                
                if timestep < len(ref_vals[i]) and timestep < len(ref_std_devs[i]):
                    detection_value = np.mean(result_deques[i])
                    threshold_exceeded = abs(detection_value - ref_vals[i][timestep]) > K_SIGMA * ref_std_devs[i][timestep] \
                                         and not np.isclose(detection_value, ref_vals[i][timestep], rtol=1e-5, atol=1e-5)
                    
                    if threshold_exceeded:
                        if i not in fault_first_detected:
                            fault_first_detected[i] = timestep
                        fault_timeline.append({
                            'agent': i,
                            't': timestep,
                            'contribs': {},
                            'taylor_deviation': abs(detection_value - ref_vals[i][timestep])
                        })
            
            # Environment step
            next_obs, rewards, dones, infos, action_masks = env.step(actions)

            agent_rewards = np.array(rewards).squeeze()
            rewards_history.append(agent_rewards)

            episode_reward += np.sum(agent_rewards)

            obs = next_obs
            timestep += 1
            
            if dones.all():
                break
        env.close()
        
        return {
            'fault_timeline': fault_timeline,
            'q_values_history': q_values_history,
            'rewards_history': rewards_history,
            'taylor_errors_history': taylor_errors_history,
            'episode_length': timestep,
            'episode_reward': episode_reward,
            'attack_timesteps': attack_timesteps,
            'attacked_agent': attack_agent_i,
            'observed_agent': observe_agent_id
        }
    
    def compute_attack_metrics(self, attack_results, normal_q_values, normal_rewards, ref_vals, ref_std_devs, observe_agent_j):
        """
        Compute Q-drop, reward-drop and Taylor deviation metrics for attacked episode.
        
        Args:
            attack_results: Results from attacked episode
            normal_q_values: Q values from normal episode
            normal_rewards: Rewards from normal episode
            ref_vals: Reference Taylor error values
            ref_std_devs: Reference standard deviations
            observe_agent_j: Index of agent to observe impact on (influenced agent)
            
        Returns:
            Dictionary containing computed metrics
        """
        attack_timesteps = attack_results['attack_timesteps']
        q_values_history = attack_results['q_values_history']
        rewards_history = attack_results['rewards_history']
        taylor_errors_history = attack_results['taylor_errors_history']
        episode_length = attack_results['episode_length']
        
        # Define watchable window based on all attack timesteps
        # Start from the earliest attack timestep and watch 15 timesteps from the first attack
        min_attack_timestep = min(attack_timesteps)
        max_attack_timestep = max(attack_timesteps)
        window_start = min_attack_timestep
        window_end = min(min_attack_timestep + WATCH_WINDOW, episode_length - 1)
        
        metrics = {
            'max_q_drop': 0.0,
            'weighted_q_drop_sum': 0.0,
            'max_reward_drop': 0.0,
            'weighted_reward_drop_sum': 0.0,
            'max_abs_taylor_deviation': 0.0,
            'weighted_taylor_deviation_sum': 0.0,
            'exceed_rate': 0.0,
            'window_length': window_end - window_start + 1
        }
        
        if window_start >= len(q_values_history) or window_start >= len(taylor_errors_history) or window_start >= len(rewards_history):
            return metrics
        
        if window_start >= len(normal_q_values) or window_start >= len(normal_rewards):
            return metrics
        
        # Compute metrics in watchable window for the observed agent
        exceed_count = 0
        window_steps = 0
        
        for t in range(window_start, window_end + 1):
            if t >= len(q_values_history) or t >= len(taylor_errors_history) or t >= len(normal_q_values) or \
               t >= len(rewards_history) or t >= len(normal_rewards):
                break
                
            window_steps += 1
            
            # Q-drop metrics for observed agent - difference between normal and attacked Q values
            normal_q = normal_q_values[t][observe_agent_j]
            attacked_q = q_values_history[t][observe_agent_j]
            q_drop = normal_q - attacked_q
            metrics['max_q_drop'] = max(metrics['max_q_drop'], q_drop)
            
            # Reward-drop metrics for observed agent - difference between normal and attacked rewards
            normal_reward = normal_rewards[t][observe_agent_j]
            attacked_reward = rewards_history[t][observe_agent_j]
            reward_drop = normal_reward - attacked_reward
            metrics['max_reward_drop'] = max(metrics['max_reward_drop'], reward_drop)
            
            # Weighted drops (use minimum attack timestep as reference for weight calculation)
            weight = self.gamma ** (t - min_attack_timestep)
            metrics['weighted_q_drop_sum'] += weight * q_drop
            metrics['weighted_reward_drop_sum'] += weight * reward_drop
            
            # Taylor deviation metrics for observed agent
            taylor_error = taylor_errors_history[t][observe_agent_j]
            ref_mean = ref_vals[observe_agent_j][t]
            ref_std = ref_std_devs[observe_agent_j][t]
            taylor_deviation = abs(taylor_error - ref_mean)
            
            metrics['max_abs_taylor_deviation'] = max(metrics['max_abs_taylor_deviation'], 
                                                    taylor_deviation)
            
            # Weighted Taylor deviation
            metrics['weighted_taylor_deviation_sum'] += weight * taylor_deviation
            
            # Check if exceeds threshold (mean ± K_SIGMA * std_dev)
            threshold = K_SIGMA * ref_std
            if taylor_deviation > threshold and not np.isclose(taylor_error, ref_mean, rtol=1e-5, atol=1e-5):
                exceed_count += 1
        
        # Compute exceed rate
        metrics['exceed_rate'] = exceed_count / window_steps
        
        return metrics
    
    def log_cumulative_influences(self, action_influences_history, seed, episode_length):
        """
        Store cumulative influence sums for each agent pair at each timestep.
        Data will be written to a single CSV file later.
        
        Args:
            action_influences_history: List of action influence matrices for each timestep
            seed: Random seed for this experiment
            episode_length: Length of the episode
        """
        print(f"Storing cumulative influences for seed {seed}...")
        
        # For each agent pair (i, j), compute and store cumulative influences
        for agent_i in range(self.maddpg.nagents):
            for agent_j in range(self.maddpg.nagents):
                if agent_i == agent_j:
                    continue  # Skip self-influence
                
                # Compute cumulative influence sum at each timestep
                cumulative_influence = 0.0
                timestep_cumulative_values = []
                
                for t in range(min(episode_length, len(action_influences_history))):
                    # Get influence of agent_i on agent_j at timestep t
                    # action_influences_history[t][j][i] = influence of i on j
                    influence_value = action_influences_history[t][agent_j][agent_i]
                    cumulative_influence += abs(influence_value)  # Use absolute value for cumulative sum
                    timestep_cumulative_values.append(cumulative_influence)
                
                # Store the data for this agent pair
                pair_data = {
                    'seed': seed,
                    'influencer_agent_id': agent_i,
                    'influenced_agent_id': agent_j,
                    'episode_length': episode_length,
                    'cumulative_values': timestep_cumulative_values
                }
                
                self.cumulative_influences_data.append(pair_data)
        
        print(f"Stored cumulative influences for seed {seed} ({self.maddpg.nagents * (self.maddpg.nagents - 1)} agent pairs)")
    
    def log_directional_derivatives(self, directional_derivatives_history, seed, episode_length):
        """
        Store directional second derivatives for each agent pair at each timestep.
        Data will be written to a single CSV file later.
        
        Args:
            directional_derivatives_history: List of directional second derivative matrices for each timestep
            seed: Random seed for this experiment
            episode_length: Length of the episode
        """
        if not hasattr(self, 'directional_derivatives_data'):
            self.directional_derivatives_data = []
            
        print(f"Storing directional derivatives for seed {seed}...")
        
        # For each agent pair (i, j), store directional derivatives at each timestep
        for agent_i in range(self.maddpg.nagents):
            for agent_j in range(self.maddpg.nagents):
                if agent_i == agent_j:
                    continue  # Skip self-influence
                
                # Get directional derivatives for each timestep
                timestep_derivative_values = []
                
                for t in range(min(episode_length, len(directional_derivatives_history))):
                    # Get directional second derivative of agent_i on agent_j at timestep t
                    # directional_derivatives_history[t][j][i] = g^T H g for influence of i on j
                    derivative_value = directional_derivatives_history[t][agent_j][agent_i]
                    timestep_derivative_values.append(derivative_value)
                
                # Store the data for this agent pair
                pair_data = {
                    'seed': seed,
                    'influencer_agent_id': agent_i,
                    'influenced_agent_id': agent_j,
                    'episode_length': episode_length,
                    'derivative_values': timestep_derivative_values
                }
                
                self.directional_derivatives_data.append(pair_data)
        
        print(f"Stored directional derivatives for seed {seed} ({self.maddpg.nagents * (self.maddpg.nagents - 1)} agent pairs)")
    
    def log_taylor_deviations(self, high_attack_results, low_attack_results, ref_vals, ref_std_devs, seed, agent_i, agent_j):
        """
        Store Taylor deviations from mean for both high and low influence attacks for each agent.
        Data will be written to a single CSV file later.
        
        Args:
            high_attack_results: Results from high influence attack episode
            low_attack_results: Results from low influence attack episode
            ref_vals: Reference Taylor error values
            ref_std_devs: Reference standard deviations
            seed: Random seed for this experiment
            agent_i: Influencing agent (attacked agent)
            agent_j: Influenced agent (observed agent)
        """
        # Initialize storage for Taylor deviations if not exists
        if not hasattr(self, 'taylor_deviations_data'):
            self.taylor_deviations_data = []
        
        # Process both high and low influence attacks
        for attack_type, attack_results in [('high', high_attack_results), ('low', low_attack_results)]:
            taylor_errors_history = attack_results['taylor_errors_history']
            episode_length = attack_results['episode_length']
            
            # For each agent, compute Taylor deviations from mean at each timestep
            for agent_id in range(self.maddpg.nagents):
                timestep_deviations = []
                
                for t in range(min(episode_length, len(taylor_errors_history))):
                    if t < len(ref_vals[agent_id]) and t < len(ref_std_devs[agent_id]):
                        taylor_error = taylor_errors_history[t][agent_id]
                        ref_mean = ref_vals[agent_id][t]
                        taylor_deviation = abs(taylor_error - ref_mean)
                        timestep_deviations.append(taylor_deviation)
                    else:
                        timestep_deviations.append(0.0)  # Default value if reference not available
                
                # Store the data for this agent
                agent_data = {
                    'seed': seed,
                    'attack_type': attack_type,
                    'influencer_agent_id': agent_i,
                    'influenced_agent_id': agent_j,
                    'observed_agent_id': agent_id,
                    'episode_length': episode_length,
                    'taylor_deviations': timestep_deviations
                }
                
                self.taylor_deviations_data.append(agent_data)
        
        print(f"Stored Taylor deviations for seed {seed}, pair ({agent_i} -> {agent_j}) for {self.maddpg.nagents} agents (both attack types)")
    
    def save_cumulative_influences_csv(self):
        """
        Save all cumulative influence data to a single CSV file.
        """
        if not self.cumulative_influences_data:
            print("No cumulative influence data to save.")
            return
        
        print("Saving cumulative influences to single CSV file...")
        
        # Find the maximum episode length to determine number of timestep columns
        max_episode_length = max(data['episode_length'] for data in self.cumulative_influences_data)
        
        # Create timestep column names
        timestep_columns = [f'timestep_{t}' for t in range(max_episode_length)]
        
        # Create CSV filename
        csv_filename = 'cumulative_influences_all_seeds.csv'
        csv_filepath = os.path.join(self.logdir, csv_filename)
        
        # Write to CSV file
        with open(csv_filepath, 'w', newline='') as csvfile:
            fieldnames = ['seed', 'influencer_agent_id', 'influenced_agent_id'] + timestep_columns
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for pair_data in self.cumulative_influences_data:
                row = {
                    'seed': pair_data['seed'],
                    'influencer_agent_id': pair_data['influencer_agent_id'],
                    'influenced_agent_id': pair_data['influenced_agent_id']
                }
                
                # Add cumulative values for each timestep
                cumulative_values = pair_data['cumulative_values']
                for t in range(max_episode_length):
                    if t < len(cumulative_values):
                        row[f'timestep_{t}'] = cumulative_values[t]
                    else:
                        row[f'timestep_{t}'] = ''  # Empty for timesteps beyond episode length
                
                writer.writerow(row)
        
        total_rows = len(self.cumulative_influences_data)
        print(f"Saved cumulative influences CSV: {csv_filename}")
        print(f"  Total rows: {total_rows}")
        print(f"  Max episode length: {max_episode_length}")
        print(f"  Timestep columns: timestep_0 to timestep_{max_episode_length - 1}")
    
    def save_directional_derivatives_csv(self):
        """
        Save all directional derivatives data to a single CSV file.
        """
        print("Saving directional derivatives to single CSV file...")
        
        # Find the maximum episode length to determine number of timestep columns
        max_episode_length = max(data['episode_length'] for data in self.directional_derivatives_data)
        
        # Create timestep column names
        timestep_columns = [f'timestep_{t}' for t in range(max_episode_length)]
        
        # Create CSV filename
        csv_filename = 'directional_derivatives_all_seeds.csv'
        csv_filepath = os.path.join(self.logdir, csv_filename)
        
        # Write to CSV file
        with open(csv_filepath, 'w', newline='') as csvfile:
            fieldnames = ['seed', 'influencer_agent_id', 'influenced_agent_id'] + timestep_columns
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for pair_data in self.directional_derivatives_data:
                row = {
                    'seed': pair_data['seed'],
                    'influencer_agent_id': pair_data['influencer_agent_id'],
                    'influenced_agent_id': pair_data['influenced_agent_id']
                }
                
                # Add derivative values for each timestep
                derivative_values = pair_data['derivative_values']
                for t in range(max_episode_length):
                    if t < len(derivative_values):
                        row[f'timestep_{t}'] = derivative_values[t]
                    else:
                        row[f'timestep_{t}'] = ''  # Empty for timesteps beyond episode length
                
                writer.writerow(row)
        
        total_rows = len(self.directional_derivatives_data)
        print(f"Saved directional derivatives CSV: {csv_filename}")
        print(f"  Total rows: {total_rows}")
        print(f"  Max episode length: {max_episode_length}")
        print(f"  Timestep columns: timestep_0 to timestep_{max_episode_length - 1}")
    
    def save_taylor_deviations_csv(self):
        """
        Save all Taylor deviations data to a single CSV file.
        """
        if not hasattr(self, 'taylor_deviations_data') or not self.taylor_deviations_data:
            print("No Taylor deviations data to save.")
            return
        
        print("Saving Taylor deviations to single CSV file...")
        
        # Find the maximum episode length to determine number of timestep columns
        max_episode_length = max(data['episode_length'] for data in self.taylor_deviations_data)
        
        # Create timestep column names
        timestep_columns = [f'timestep_{t}' for t in range(max_episode_length)]
        
        # Create CSV filename
        csv_filename = 'taylor_deviations_all_seeds.csv'
        csv_filepath = os.path.join(self.logdir, csv_filename)
        
        # Write to CSV file
        with open(csv_filepath, 'w', newline='') as csvfile:
            fieldnames = ['seed', 'attack_type', 'influencer_agent_id', 'influenced_agent_id', 'observed_agent_id'] + timestep_columns
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for agent_data in self.taylor_deviations_data:
                row = {
                    'seed': agent_data['seed'],
                    'attack_type': agent_data['attack_type'],
                    'influencer_agent_id': agent_data['influencer_agent_id'],
                    'influenced_agent_id': agent_data['influenced_agent_id'],
                    'observed_agent_id': agent_data['observed_agent_id']
                }
                
                # Add Taylor deviation values for each timestep
                taylor_deviations = agent_data['taylor_deviations']
                for t in range(max_episode_length):
                    if t < len(taylor_deviations):
                        row[f'timestep_{t}'] = taylor_deviations[t]
                    else:
                        row[f'timestep_{t}'] = ''  # Empty for timesteps beyond episode length
                
                writer.writerow(row)
        
        total_rows = len(self.taylor_deviations_data)
        print(f"Saved Taylor deviations CSV: {csv_filename}")
        print(f"  Total rows: {total_rows}")
        print(f"  Max episode length: {max_episode_length}")
        print(f"  Timestep columns: timestep_0 to timestep_{max_episode_length - 1}")
    
    def run_single_seed_experiment(self, seed):
        """
        Run complete experiment for a single seed.
        
        Args:
            seed: Random seed for the experiment
            
        Returns:
            Dictionary containing experiment results for all agent pairs
        """
        print(f"\n{'='*50}")
        print(f"Running experiment for seed {seed}")
        print(f"{'='*50}")
        
        # Step 1: Get reference Taylor error (either load from disk or compute)
        if hasattr(self.config, 'taylor_ref_dir') and self.config.taylor_ref_dir is not None:
            ref_vals, ref_std_devs = self.load_reference_taylor_values(seed, self.config.taylor_ref_dir)
            print(f"Successfully loaded precomputed reference Taylor values for seed {seed}")
        else:
            ref_vals, ref_std_devs = self.compute_reference_taylor_error(seed)
        
        # Step 2: Run normal episode
        normal_episode = self.run_normal_episode(seed)
        action_influences_history = normal_episode['action_influences_history']
        directional_derivatives_history = normal_episode['directional_derivatives_history']
        normal_q_values_history = normal_episode['q_values_history']
        normal_rewards_history = normal_episode['rewards_history']
        episode_length = normal_episode['episode_length']
        
        # Step 2.5: Log cumulative influences for each agent pair
        self.log_cumulative_influences(action_influences_history, seed, episode_length)
        
        # Step 2.6: Log directional derivatives for each agent pair
        self.log_directional_derivatives(directional_derivatives_history, seed, episode_length)
        
        # Step 3: Analyze all possible ordered pairs (i, j) where i influences j
        all_pair_results = []
        atk_steps_limit = math.ceil(ATTACK_TS_FRACTION * episode_length)
        
        for agent_i in range(self.maddpg.nagents):  # influencing agent
            for agent_j in range(self.maddpg.nagents):  # influenced agent
                if agent_i == agent_j:
                    continue  # Skip self
                
                print(f"\nAnalyzing pair: agent_{agent_i} influences agent_{agent_j}")
                
                # Step 4: Find max and min influence timesteps of agent_i on agent_j in first 25%
                max_influence_t, min_influence_t = self.find_influence_timesteps(
                    action_influences_history, directional_derivatives_history, agent_i, agent_j, atk_steps_limit, 1
                )
                
                # Skip this iteration if no suitable timesteps are found
                if not max_influence_t or not min_influence_t:
                    print(f"Skipping pair agent_{agent_i} -> agent_{agent_j}: No suitable timesteps found")
                    print(f"  Max influence timesteps (positive derivatives): {max_influence_t}")
                    print(f"  Min influence timesteps (negative derivatives): {min_influence_t}")
                    continue
                
                print(f"Max influence timesteps: {max_influence_t}, Min influence timesteps: {min_influence_t}")
                
                # Step 5: Run attacked episodes - attack agent_i (influencer), observe impact on agent_j (influenced)
                high_influence_attack = self.run_attacked_episode(
                    seed, agent_i, max_influence_t, ref_vals, ref_std_devs, observe_agent=agent_j
                )
                
                low_influence_attack = self.run_attacked_episode(
                    seed, agent_i, min_influence_t, ref_vals, ref_std_devs, observe_agent=agent_j 
                )
                
                # Determine patient zero for each attack
                high_patient_zero, high_patient_time = get_patient_zero_detection(high_influence_attack['fault_timeline'])
                low_patient_zero, low_patient_time = get_patient_zero_detection(low_influence_attack['fault_timeline'])
                
                # Get fault detection times for influencing and influenced agents
                high_influencer_fault_times = get_agent_fault_detection_times(high_influence_attack['fault_timeline'], agent_i)
                high_influenced_fault_times = get_agent_fault_detection_times(high_influence_attack['fault_timeline'], agent_j)
                low_influencer_fault_times = get_agent_fault_detection_times(low_influence_attack['fault_timeline'], agent_i)
                low_influenced_fault_times = get_agent_fault_detection_times(low_influence_attack['fault_timeline'], agent_j)
                
                # Compute attack metrics
                high_metrics = self.compute_attack_metrics(high_influence_attack, normal_q_values_history, normal_rewards_history, ref_vals, ref_std_devs, agent_j)
                low_metrics = self.compute_attack_metrics(low_influence_attack, normal_q_values_history, normal_rewards_history, ref_vals, ref_std_devs, agent_j)
                
                # Log Taylor deviations for both attack types
                self.log_taylor_deviations(high_influence_attack, low_influence_attack, ref_vals, ref_std_devs, seed, agent_i, agent_j)
                
                # Step 6: Analyze patient zero detection accuracy with traceback
                print(f"\n--- Patient Zero Analysis for pair agent_{agent_i} -> agent_{agent_j} ---")
                
                # Analyze high influence attack
                print(f"Analyzing HIGH influence attack:")
                high_pz_detection_analysis = self.patient_zero_analyzer.analyze_detection_accuracy(
                    high_influence_attack['fault_timeline'],
                    agent_i,  # The attacked agent is the influencer
                    high_influence_attack['attack_timesteps'],
                    directional_derivatives_history,
                    high_influence_attack['taylor_errors_history'],
                    ref_vals,
                    action_influences_history,
                    seed,
                    (agent_i, agent_j)
                )
                
                # Analyze low influence attack
                print(f"Analyzing LOW influence attack:")
                low_pz_detection_analysis = self.patient_zero_analyzer.analyze_detection_accuracy(
                    low_influence_attack['fault_timeline'],
                    agent_i,  # The attacked agent is the influencer
                    low_influence_attack['attack_timesteps'],
                    directional_derivatives_history,
                    low_influence_attack['taylor_errors_history'],
                    ref_vals,
                    action_influences_history,
                    seed,
                    (agent_i, agent_j)
                )
                
                pair_result = {
                    'agent_i': agent_i,
                    'agent_j': agent_j,
                    'max_influence_t': max_influence_t,
                    'min_influence_t': min_influence_t,
                    'high_patient_zero': high_patient_zero,
                    'high_patient_time': high_patient_time,
                    'low_patient_zero': low_patient_zero,
                    'low_patient_time': low_patient_time,
                    'high_influencer_fault_detection_times': high_influencer_fault_times,
                    'high_influenced_fault_detection_times': high_influenced_fault_times,
                    'low_influencer_fault_detection_times': low_influencer_fault_times,
                    'low_influenced_fault_detection_times': low_influenced_fault_times,
                    'high_metrics': high_metrics,
                    'low_metrics': low_metrics,
                    'high_patient_zero_analysis': high_pz_detection_analysis,
                    'low_patient_zero_analysis': low_pz_detection_analysis
                }
                
                all_pair_results.append(pair_result)
                
                # Handle multiple patient zeros for display
                high_pz_str = ', '.join(map(str, high_patient_zero)) if isinstance(high_patient_zero, list) else str(high_patient_zero)
                low_pz_str = ', '.join(map(str, low_patient_zero)) if isinstance(low_patient_zero, list) else str(low_patient_zero)
                
                print(f"High influence attack - Patient zero(s): {high_pz_str} at time {high_patient_time}")
                print(f"  Influencer (agent_{agent_i}) fault detection times: {high_influencer_fault_times}")
                print(f"  Influenced (agent_{agent_j}) fault detection times: {high_influenced_fault_times}")
                print(f"Low influence attack - Patient zero(s): {low_pz_str} at time {low_patient_time}")
                print(f"  Influencer (agent_{agent_i}) fault detection times: {low_influencer_fault_times}")
                print(f"  Influenced (agent_{agent_j}) fault detection times: {low_influenced_fault_times}")
        
        result = {
            'seed': seed,
            'episode_length': episode_length,
            'pair_results': all_pair_results,
            'total_pairs': len(all_pair_results)
        }
        
        print(f"\nCompleted analysis for {len(all_pair_results)} agent pairs")
        
        return result
    
    def run_all_experiments(self):
        """Run experiments for all seeds."""
        if self.maddpg is None:
            raise RuntimeError("MADDPG model not loaded. Call setup_experiment() first.")
            
        total_pairs_per_seed = self.maddpg.nagents * (self.maddpg.nagents - 1)
        print(f"Starting multi-seed experiments with {self.total_experiments} seeds...")
        print(f"Each seed will analyze {total_pairs_per_seed} agent pairs")
        print(f"Total pairs to analyze: {self.total_experiments * total_pairs_per_seed}")
        
        for seed in tqdm(range(self.total_experiments), desc="Running experiments"):
            result = self.run_single_seed_experiment(seed)
            self.experiment_results.append(result)
        
        total_successful_pairs = sum(result['total_pairs'] for result in self.experiment_results)
        print(f"\nCompleted {len(self.experiment_results)} successful experiments out of {self.total_experiments}")
        print(f"Total successful pairs analyzed: {total_successful_pairs}")
        print(f"Failed experiments: {len(self.failed_seeds)}")
    
    def compute_accuracies(self):
        """Compute accuracies and analyze results."""
        if not self.experiment_results:
            print("No successful experiments to analyze!")
            return
        
        print("\n" + "="*50)
        print("COMPUTING ACCURACIES")
        print("="*50)
        
        # Patient zero detection accuracy
        correct_patient_zero = 0
        total_with_detection = 0
        high_correct_patient_zero = 0
        high_total_with_detection = 0
        low_correct_patient_zero = 0
        low_total_with_detection = 0
        
        # Expectation accuracy metrics - separate accuracies for each metric
        q_drop_max_expectation_correct = 0
        q_drop_weighted_expectation_correct = 0
        reward_drop_max_expectation_correct = 0
        reward_drop_weighted_expectation_correct = 0
        taylor_max_expectation_correct = 0
        taylor_weighted_expectation_correct = 0
        exceed_rate_expectation_correct = 0
        
        # Detailed metrics
        high_metrics_list = []
        low_metrics_list = []
        
        failed_expectations = []
        total_pairs = 0
        
        # Process all pairs from all seeds
        for result in self.experiment_results:
            seed = result['seed']
            pair_results = result['pair_results']
            
            for pair_result in pair_results:
                total_pairs += 1
                
                # Patient zero analysis - we expect agent_i (attacked agent) or agent_j (influenced agent) to be detected as patient zero
                high_patient_zero = pair_result['high_patient_zero']
                low_patient_zero = pair_result['low_patient_zero']
                attacked_agent = pair_result['agent_i']  # The agent we attacked (influencer)
                
                if high_patient_zero is not None:
                    total_with_detection += 1
                    high_total_with_detection += 1
                    # Check if attacked agent or influenced agent is in the patient zero list (handles multiple patient zeros)
                    if attacked_agent in high_patient_zero:
                        correct_patient_zero += 1
                        high_correct_patient_zero += 1
                
                if low_patient_zero is not None:
                    total_with_detection += 1
                    low_total_with_detection += 1
                    if attacked_agent in low_patient_zero:
                        correct_patient_zero += 1
                        low_correct_patient_zero += 1
                
                # Expectation analysis
                high_metrics = pair_result['high_metrics']
                low_metrics = pair_result['low_metrics']
                
                high_metrics_list.append(high_metrics)
                low_metrics_list.append(low_metrics)
                
                # Check individual metric expectations (high influence should have higher impact)
                q_drop_max_better = high_metrics['max_q_drop'] >= low_metrics['max_q_drop']
                q_drop_weighted_better = high_metrics['weighted_q_drop_sum'] >= low_metrics['weighted_q_drop_sum']
                reward_drop_max_better = high_metrics['max_reward_drop'] >= low_metrics['max_reward_drop']
                reward_drop_weighted_better = high_metrics['weighted_reward_drop_sum'] >= low_metrics['weighted_reward_drop_sum']
                taylor_max_better = high_metrics['max_abs_taylor_deviation'] >= low_metrics['max_abs_taylor_deviation']
                taylor_weighted_better = high_metrics['weighted_taylor_deviation_sum'] >= low_metrics['weighted_taylor_deviation_sum']
                exceed_rate_better = high_metrics['exceed_rate'] >= low_metrics['exceed_rate']

                if q_drop_max_better:
                    q_drop_max_expectation_correct += 1
                
                if q_drop_weighted_better:
                    q_drop_weighted_expectation_correct += 1
                
                if reward_drop_max_better:
                    reward_drop_max_expectation_correct += 1
                
                if reward_drop_weighted_better:
                    reward_drop_weighted_expectation_correct += 1
                
                if taylor_max_better:
                    taylor_max_expectation_correct += 1
                
                if taylor_weighted_better:
                    taylor_weighted_expectation_correct += 1
                
                if exceed_rate_better:
                    exceed_rate_expectation_correct += 1
                
                # Log failed expectations
                if not (q_drop_max_better or q_drop_weighted_better or
                        reward_drop_max_better or reward_drop_weighted_better or
                        taylor_max_better or taylor_weighted_better or
                        exceed_rate_better):
                    failed_expectations.append({
                        'seed': seed,
                        'agent_i_influencer_attacked': pair_result['agent_i'],
                        'agent_j_influenced_observed': pair_result['agent_j'],
                        'q_drop_max_failed': not q_drop_max_better,
                        'q_drop_weighted_failed': not q_drop_weighted_better,
                        'reward_drop_max_failed': not reward_drop_max_better,
                        'reward_drop_weighted_failed': not reward_drop_weighted_better,
                        'taylor_max_failed': not taylor_max_better,
                        'taylor_weighted_failed': not taylor_weighted_better,
                        'exceed_rate_failed': not exceed_rate_better,
                        'high_q_drop_max': high_metrics['max_q_drop'],
                        'low_q_drop_max': low_metrics['max_q_drop'],
                        'high_q_drop_weighted': high_metrics['weighted_q_drop_sum'],
                        'low_q_drop_weighted': low_metrics['weighted_q_drop_sum'],
                        'high_reward_drop_max': high_metrics['max_reward_drop'],
                        'low_reward_drop_max': low_metrics['max_reward_drop'],
                        'high_reward_drop_weighted': high_metrics['weighted_reward_drop_sum'],
                        'low_reward_drop_weighted': low_metrics['weighted_reward_drop_sum'],
                        'high_taylor_max': high_metrics['max_abs_taylor_deviation'],
                        'low_taylor_max': low_metrics['max_abs_taylor_deviation'],
                        'high_taylor_weighted': high_metrics['weighted_taylor_deviation_sum'],
                        'low_taylor_weighted': low_metrics['weighted_taylor_deviation_sum'],
                        'high_exceed_rate': high_metrics['exceed_rate'],
                        'low_exceed_rate': low_metrics['exceed_rate']
                    })
        
        total_experiments = len(self.experiment_results)
        
        # Compute accuracies
        patient_zero_accuracy = correct_patient_zero / total_with_detection if total_with_detection > 0 else 0
        high_patient_zero_accuracy = high_correct_patient_zero / high_total_with_detection if high_total_with_detection > 0 else 0
        low_patient_zero_accuracy = low_correct_patient_zero / low_total_with_detection if low_total_with_detection > 0 else 0
        q_drop_max_accuracy = q_drop_max_expectation_correct / total_pairs
        q_drop_weighted_accuracy = q_drop_weighted_expectation_correct / total_pairs
        reward_drop_max_accuracy = reward_drop_max_expectation_correct / total_pairs
        reward_drop_weighted_accuracy = reward_drop_weighted_expectation_correct / total_pairs
        taylor_max_accuracy = taylor_max_expectation_correct / total_pairs
        taylor_weighted_accuracy = taylor_weighted_expectation_correct / total_pairs
        exceed_rate_accuracy = exceed_rate_expectation_correct / total_pairs
        
        # Aggregate metrics
        avg_high_q_drop_max = np.mean([m['max_q_drop'] for m in high_metrics_list])
        avg_low_q_drop_max = np.mean([m['max_q_drop'] for m in low_metrics_list])
        avg_high_q_drop_weighted = np.mean([m['weighted_q_drop_sum'] for m in high_metrics_list])
        avg_low_q_drop_weighted = np.mean([m['weighted_q_drop_sum'] for m in low_metrics_list])
        avg_high_reward_drop_max = np.mean([m['max_reward_drop'] for m in high_metrics_list])
        avg_low_reward_drop_max = np.mean([m['max_reward_drop'] for m in low_metrics_list])
        avg_high_reward_drop_weighted = np.mean([m['weighted_reward_drop_sum'] for m in high_metrics_list])
        avg_low_reward_drop_weighted = np.mean([m['weighted_reward_drop_sum'] for m in low_metrics_list])
        avg_high_taylor_max = np.mean([m['max_abs_taylor_deviation'] for m in high_metrics_list])
        avg_low_taylor_max = np.mean([m['max_abs_taylor_deviation'] for m in low_metrics_list])
        avg_high_taylor_weighted = np.mean([m['weighted_taylor_deviation_sum'] for m in high_metrics_list])
        avg_low_taylor_weighted = np.mean([m['weighted_taylor_deviation_sum'] for m in low_metrics_list])
        avg_high_exceed_rate = np.mean([m['exceed_rate'] for m in high_metrics_list])
        avg_low_exceed_rate = np.mean([m['exceed_rate'] for m in low_metrics_list])
        
        accuracy_results = {
            'total_experiments': total_experiments,
            'total_pairs': total_pairs,
            'total_with_detection': total_with_detection,
            'correct_patient_zero': correct_patient_zero,
            'patient_zero_accuracy': patient_zero_accuracy,
            'high_patient_zero_accuracy': high_patient_zero_accuracy,
            'low_patient_zero_accuracy': low_patient_zero_accuracy,
            'q_drop_max_expectation_correct': q_drop_max_expectation_correct,
            'q_drop_max_accuracy': q_drop_max_accuracy,
            'q_drop_weighted_expectation_correct': q_drop_weighted_expectation_correct,
            'q_drop_weighted_accuracy': q_drop_weighted_accuracy,
            'reward_drop_max_expectation_correct': reward_drop_max_expectation_correct,
            'reward_drop_max_accuracy': reward_drop_max_accuracy,
            'reward_drop_weighted_expectation_correct': reward_drop_weighted_expectation_correct,
            'reward_drop_weighted_accuracy': reward_drop_weighted_accuracy,
            'taylor_max_expectation_correct': taylor_max_expectation_correct,
            'taylor_max_accuracy': taylor_max_accuracy,
            'taylor_weighted_expectation_correct': taylor_weighted_expectation_correct,
            'taylor_weighted_accuracy': taylor_weighted_accuracy,
            'exceed_rate_expectation_correct': exceed_rate_expectation_correct,
            'exceed_rate_accuracy': exceed_rate_accuracy,
            'avg_high_q_drop_max': avg_high_q_drop_max,
            'avg_low_q_drop_max': avg_low_q_drop_max,
            'avg_high_q_drop_weighted': avg_high_q_drop_weighted,
            'avg_low_q_drop_weighted': avg_low_q_drop_weighted,
            'avg_high_reward_drop_max': avg_high_reward_drop_max,
            'avg_low_reward_drop_max': avg_low_reward_drop_max,
            'avg_high_reward_drop_weighted': avg_high_reward_drop_weighted,
            'avg_low_reward_drop_weighted': avg_low_reward_drop_weighted,
            'avg_high_taylor_max': avg_high_taylor_max,
            'avg_low_taylor_max': avg_low_taylor_max,
            'avg_high_taylor_weighted': avg_high_taylor_weighted,
            'avg_low_taylor_weighted': avg_low_taylor_weighted,
            'avg_high_exceed_rate': avg_high_exceed_rate,
            'avg_low_exceed_rate': avg_low_exceed_rate,
            'failed_expectations_count': len(failed_expectations)
        }
        
        print(f"Total Experiments: {total_experiments}, Total Agent Pairs: {total_pairs}")
        print(f"Patient Zero Detection Accuracy: {patient_zero_accuracy:.3f} ({correct_patient_zero}/{total_with_detection})")
        print(f"High Influence: Patient Zero Accuracy: {high_patient_zero_accuracy:.3f} ({high_correct_patient_zero}/{high_total_with_detection})")
        print(f"Low Influence: Patient Zero Accuracy: {low_patient_zero_accuracy:.3f} ({low_correct_patient_zero}/{low_total_with_detection})")
        print(f"Q-Drop Max Expectation Accuracy: {q_drop_max_accuracy:.3f} ({q_drop_max_expectation_correct}/{total_pairs})")
        print(f"Q-Drop Weighted Expectation Accuracy: {q_drop_weighted_accuracy:.3f} ({q_drop_weighted_expectation_correct}/{total_pairs})")
        print(f"Reward-Drop Max Expectation Accuracy: {reward_drop_max_accuracy:.3f} ({reward_drop_max_expectation_correct}/{total_pairs})")
        print(f"Reward-Drop Weighted Expectation Accuracy: {reward_drop_weighted_accuracy:.3f} ({reward_drop_weighted_expectation_correct}/{total_pairs})")
        print(f"Taylor Max Expectation Accuracy: {taylor_max_accuracy:.3f} ({taylor_max_expectation_correct}/{total_pairs})")
        print(f"Taylor Weighted Expectation Accuracy: {taylor_weighted_accuracy:.3f} ({taylor_weighted_expectation_correct}/{total_pairs})")
        print(f"Exceed Rate Expectation Accuracy: {exceed_rate_accuracy:.3f} ({exceed_rate_expectation_correct}/{total_pairs})")
        print(f"Average High Influence Q-Drop Max: {avg_high_q_drop_max:.6f}")
        print(f"Average Low Influence Q-Drop Max: {avg_low_q_drop_max:.6f}")
        print(f"Average High Influence Reward-Drop Max: {avg_high_reward_drop_max:.6f}")
        print(f"Average Low Influence Reward-Drop Max: {avg_low_reward_drop_max:.6f}")
        print(f"Average High Influence Taylor Max: {avg_high_taylor_max:.6f}")
        print(f"Average Low Influence Taylor Max: {avg_low_taylor_max:.6f}")
        print(f"Average High Influence Exceed Rate: {avg_high_exceed_rate:.6f}")
        print(f"Average Low Influence Exceed Rate: {avg_low_exceed_rate:.6f}")
        print(f"Failed Expectations: {len(failed_expectations)}")
        
        return accuracy_results, failed_expectations
    
    def compute_pair_specific_accuracies(self):
        """Compute accuracies and analyze results for each agent pair separately."""
        if not self.experiment_results:
            print("No successful experiments to analyze!")
            return {}
        
        print("\n" + "="*50)
        print("COMPUTING PAIR-SPECIFIC ACCURACIES")
        print("="*50)
        
        # Initialize pair-specific results storage
        pair_specific_results = {}
        
        # Get all unique pairs
        unique_pairs = set()
        for result in self.experiment_results:
            for pair_result in result['pair_results']:
                pair_key = (pair_result['agent_i'], pair_result['agent_j'])
                unique_pairs.add(pair_key)
        
        print(f"Found {len(unique_pairs)} unique agent pairs")
        
        # Process each unique pair
        for agent_i, agent_j in sorted(unique_pairs):
            pair_key = f"agent_{agent_i}_to_agent_{agent_j}"
            print(f"\nAnalyzing pair: {pair_key}")
            
            # Collect all results for this specific pair
            pair_data = []
            for result in self.experiment_results:
                for pair_result in result['pair_results']:
                    if pair_result['agent_i'] == agent_i and pair_result['agent_j'] == agent_j:
                        pair_data.append({
                            'seed': result['seed'],
                            'episode_length': result['episode_length'],
                            **pair_result
                        })
            
            if not pair_data:
                continue
            
            # Initialize counters for this pair
            correct_patient_zero = 0
            total_with_detection = 0
            high_correct_patient_zero = 0
            high_total_with_detection = 0
            low_correct_patient_zero = 0
            low_total_with_detection = 0
            
            # Expectation accuracy metrics for this pair
            q_drop_max_expectation_correct = 0
            q_drop_weighted_expectation_correct = 0
            reward_drop_max_expectation_correct = 0
            reward_drop_weighted_expectation_correct = 0
            taylor_max_expectation_correct = 0
            taylor_weighted_expectation_correct = 0
            exceed_rate_expectation_correct = 0
            
            # Detailed metrics for this pair
            high_metrics_list = []
            low_metrics_list = []
            failed_expectations = []
            
            # Process each experiment for this pair
            for data in pair_data:
                # Patient zero analysis
                high_patient_zero = data['high_patient_zero']
                low_patient_zero = data['low_patient_zero']
                attacked_agent = data['agent_i']  # The agent we attacked (influencer)
                
                if high_patient_zero is not None:
                    total_with_detection += 1
                    high_total_with_detection += 1
                    if attacked_agent in high_patient_zero:
                        correct_patient_zero += 1
                        high_correct_patient_zero += 1
                
                if low_patient_zero is not None:
                    total_with_detection += 1
                    low_total_with_detection += 1
                    if attacked_agent in low_patient_zero:
                        correct_patient_zero += 1
                        low_correct_patient_zero += 1
                
                # Expectation analysis
                high_metrics = data['high_metrics']
                low_metrics = data['low_metrics']
                
                high_metrics_list.append(high_metrics)
                low_metrics_list.append(low_metrics)
                
                # Check individual metric expectations
                q_drop_max_better = high_metrics['max_q_drop'] >= low_metrics['max_q_drop']
                q_drop_weighted_better = high_metrics['weighted_q_drop_sum'] >= low_metrics['weighted_q_drop_sum']
                reward_drop_max_better = high_metrics['max_reward_drop'] >= low_metrics['max_reward_drop']
                reward_drop_weighted_better = high_metrics['weighted_reward_drop_sum'] >= low_metrics['weighted_reward_drop_sum']
                taylor_max_better = high_metrics['max_abs_taylor_deviation'] >= low_metrics['max_abs_taylor_deviation']
                taylor_weighted_better = high_metrics['weighted_taylor_deviation_sum'] >= low_metrics['weighted_taylor_deviation_sum']
                exceed_rate_better = high_metrics['exceed_rate'] >= low_metrics['exceed_rate']

                if q_drop_max_better:
                    q_drop_max_expectation_correct += 1
                if q_drop_weighted_better:
                    q_drop_weighted_expectation_correct += 1
                if reward_drop_max_better:
                    reward_drop_max_expectation_correct += 1
                if reward_drop_weighted_better:
                    reward_drop_weighted_expectation_correct += 1
                if taylor_max_better:
                    taylor_max_expectation_correct += 1
                if taylor_weighted_better:
                    taylor_weighted_expectation_correct += 1
                if exceed_rate_better:
                    exceed_rate_expectation_correct += 1
                
                # Log failed expectations for this pair
                if not (q_drop_max_better and q_drop_weighted_better and
                        reward_drop_max_better and reward_drop_weighted_better and
                        taylor_max_better and taylor_weighted_better and
                        exceed_rate_better):
                    failed_expectations.append({
                        'seed': data['seed'],
                        'agent_i_influencer_attacked': data['agent_i'],
                        'agent_j_influenced_observed': data['agent_j'],
                        'q_drop_max_failed': not q_drop_max_better,
                        'q_drop_weighted_failed': not q_drop_weighted_better,
                        'reward_drop_max_failed': not reward_drop_max_better,
                        'reward_drop_weighted_failed': not reward_drop_weighted_better,
                        'taylor_max_failed': not taylor_max_better,
                        'taylor_weighted_failed': not taylor_weighted_better,
                        'exceed_rate_failed': not exceed_rate_better,
                        'high_q_drop_max': high_metrics['max_q_drop'],
                        'low_q_drop_max': low_metrics['max_q_drop'],
                        'high_q_drop_weighted': high_metrics['weighted_q_drop_sum'],
                        'low_q_drop_weighted': low_metrics['weighted_q_drop_sum'],
                        'high_reward_drop_max': high_metrics['max_reward_drop'],
                        'low_reward_drop_max': low_metrics['max_reward_drop'],
                        'high_reward_drop_weighted': high_metrics['weighted_reward_drop_sum'],
                        'low_reward_drop_weighted': low_metrics['weighted_reward_drop_sum'],
                        'high_taylor_max': high_metrics['max_abs_taylor_deviation'],
                        'low_taylor_max': low_metrics['max_abs_taylor_deviation'],
                        'high_taylor_weighted': high_metrics['weighted_taylor_deviation_sum'],
                        'low_taylor_weighted': low_metrics['weighted_taylor_deviation_sum'],
                        'high_exceed_rate': high_metrics['exceed_rate'],
                        'low_exceed_rate': low_metrics['exceed_rate']
                    })
            
            total_experiments_for_pair = len(pair_data)
            
            # Compute accuracies for this pair
            patient_zero_accuracy = correct_patient_zero / total_with_detection if total_with_detection > 0 else 0
            high_patient_zero_accuracy = high_correct_patient_zero / high_total_with_detection if high_total_with_detection > 0 else 0
            low_patient_zero_accuracy = low_correct_patient_zero / low_total_with_detection if low_total_with_detection > 0 else 0
            q_drop_max_accuracy = q_drop_max_expectation_correct / total_experiments_for_pair
            q_drop_weighted_accuracy = q_drop_weighted_expectation_correct / total_experiments_for_pair
            reward_drop_max_accuracy = reward_drop_max_expectation_correct / total_experiments_for_pair
            reward_drop_weighted_accuracy = reward_drop_weighted_expectation_correct / total_experiments_for_pair
            taylor_max_accuracy = taylor_max_expectation_correct / total_experiments_for_pair
            taylor_weighted_accuracy = taylor_weighted_expectation_correct / total_experiments_for_pair
            exceed_rate_accuracy = exceed_rate_expectation_correct / total_experiments_for_pair
            
            # Aggregate metrics for this pair
            avg_high_q_drop_max = np.mean([m['max_q_drop'] for m in high_metrics_list])
            avg_low_q_drop_max = np.mean([m['max_q_drop'] for m in low_metrics_list])
            avg_high_q_drop_weighted = np.mean([m['weighted_q_drop_sum'] for m in high_metrics_list])
            avg_low_q_drop_weighted = np.mean([m['weighted_q_drop_sum'] for m in low_metrics_list])
            avg_high_reward_drop_max = np.mean([m['max_reward_drop'] for m in high_metrics_list])
            avg_low_reward_drop_max = np.mean([m['max_reward_drop'] for m in low_metrics_list])
            avg_high_reward_drop_weighted = np.mean([m['weighted_reward_drop_sum'] for m in high_metrics_list])
            avg_low_reward_drop_weighted = np.mean([m['weighted_reward_drop_sum'] for m in low_metrics_list])
            avg_high_taylor_max = np.mean([m['max_abs_taylor_deviation'] for m in high_metrics_list])
            avg_low_taylor_max = np.mean([m['max_abs_taylor_deviation'] for m in low_metrics_list])
            avg_high_taylor_weighted = np.mean([m['weighted_taylor_deviation_sum'] for m in high_metrics_list])
            avg_low_taylor_weighted = np.mean([m['weighted_taylor_deviation_sum'] for m in low_metrics_list])
            avg_high_exceed_rate = np.mean([m['exceed_rate'] for m in high_metrics_list])
            avg_low_exceed_rate = np.mean([m['exceed_rate'] for m in low_metrics_list])
            
            # Calculate average delta metrics (high - low)
            avg_delta_max_q_drop = np.mean([h['max_q_drop'] - l['max_q_drop'] for h, l in zip(high_metrics_list, low_metrics_list)])
            avg_delta_weighted_q_drop_sum = np.mean([h['weighted_q_drop_sum'] - l['weighted_q_drop_sum'] for h, l in zip(high_metrics_list, low_metrics_list)])
            avg_delta_max_reward_drop = np.mean([h['max_reward_drop'] - l['max_reward_drop'] for h, l in zip(high_metrics_list, low_metrics_list)])
            avg_delta_weighted_reward_drop_sum = np.mean([h['weighted_reward_drop_sum'] - l['weighted_reward_drop_sum'] for h, l in zip(high_metrics_list, low_metrics_list)])
            avg_delta_max_abs_taylor_deviation = np.mean([h['max_abs_taylor_deviation'] - l['max_abs_taylor_deviation'] for h, l in zip(high_metrics_list, low_metrics_list)])
            avg_delta_weighted_taylor_deviation_sum = np.mean([h['weighted_taylor_deviation_sum'] - l['weighted_taylor_deviation_sum'] for h, l in zip(high_metrics_list, low_metrics_list)])
            avg_delta_exceed_rate = np.mean([h['exceed_rate'] - l['exceed_rate'] for h, l in zip(high_metrics_list, low_metrics_list)])
            
            # Store results for this pair
            pair_specific_results[pair_key] = {
                'agent_i': agent_i,
                'agent_j': agent_j,
                'total_experiments': total_experiments_for_pair,
                'total_with_detection': total_with_detection,
                'correct_patient_zero': correct_patient_zero,
                'patient_zero_accuracy': patient_zero_accuracy,
                'high_patient_zero_accuracy': high_patient_zero_accuracy,
                'low_patient_zero_accuracy': low_patient_zero_accuracy,
                'q_drop_max_expectation_correct': q_drop_max_expectation_correct,
                'q_drop_max_accuracy': q_drop_max_accuracy,
                'q_drop_weighted_expectation_correct': q_drop_weighted_expectation_correct,
                'q_drop_weighted_accuracy': q_drop_weighted_accuracy,
                'reward_drop_max_expectation_correct': reward_drop_max_expectation_correct,
                'reward_drop_max_accuracy': reward_drop_max_accuracy,
                'reward_drop_weighted_expectation_correct': reward_drop_weighted_expectation_correct,
                'reward_drop_weighted_accuracy': reward_drop_weighted_accuracy,
                'taylor_max_expectation_correct': taylor_max_expectation_correct,
                'taylor_max_accuracy': taylor_max_accuracy,
                'taylor_weighted_expectation_correct': taylor_weighted_expectation_correct,
                'taylor_weighted_accuracy': taylor_weighted_accuracy,
                'exceed_rate_expectation_correct': exceed_rate_expectation_correct,
                'exceed_rate_accuracy': exceed_rate_accuracy,
                'avg_high_q_drop_max': avg_high_q_drop_max,
                'avg_low_q_drop_max': avg_low_q_drop_max,
                'avg_high_q_drop_weighted': avg_high_q_drop_weighted,
                'avg_low_q_drop_weighted': avg_low_q_drop_weighted,
                'avg_high_reward_drop_max': avg_high_reward_drop_max,
                'avg_low_reward_drop_max': avg_low_reward_drop_max,
                'avg_high_reward_drop_weighted': avg_high_reward_drop_weighted,
                'avg_low_reward_drop_weighted': avg_low_reward_drop_weighted,
                'avg_high_taylor_max': avg_high_taylor_max,
                'avg_low_taylor_max': avg_low_taylor_max,
                'avg_high_taylor_weighted': avg_high_taylor_weighted,
                'avg_low_taylor_weighted': avg_low_taylor_weighted,
                'avg_high_exceed_rate': avg_high_exceed_rate,
                'avg_low_exceed_rate': avg_low_exceed_rate,
                'avg_delta_max_q_drop': avg_delta_max_q_drop,
                'avg_delta_weighted_q_drop_sum': avg_delta_weighted_q_drop_sum,
                'avg_delta_max_reward_drop': avg_delta_max_reward_drop,
                'avg_delta_weighted_reward_drop_sum': avg_delta_weighted_reward_drop_sum,
                'avg_delta_max_abs_taylor_deviation': avg_delta_max_abs_taylor_deviation,
                'avg_delta_weighted_taylor_deviation_sum': avg_delta_weighted_taylor_deviation_sum,
                'avg_delta_exceed_rate': avg_delta_exceed_rate,
                'failed_expectations_count': len(failed_expectations),
                'failed_expectations': failed_expectations,
                'raw_pair_data': pair_data
            }
            
            # Print summary for this pair
            print(f"  Total Experiments: {total_experiments_for_pair}")
            print(f"  Patient Zero Detection Accuracy: {patient_zero_accuracy:.3f} ({correct_patient_zero}/{total_with_detection})")
            print(f"  Q-Drop Max Expectation Accuracy: {q_drop_max_accuracy:.3f} ({q_drop_max_expectation_correct}/{total_experiments_for_pair})")
            print(f"  Q-Drop Weighted Expectation Accuracy: {q_drop_weighted_accuracy:.3f} ({q_drop_weighted_expectation_correct}/{total_experiments_for_pair})")
            print(f"  Reward-Drop Max Expectation Accuracy: {reward_drop_max_accuracy:.3f} ({reward_drop_max_expectation_correct}/{total_experiments_for_pair})")
            print(f"  Reward-Drop Weighted Expectation Accuracy: {reward_drop_weighted_accuracy:.3f} ({reward_drop_weighted_expectation_correct}/{total_experiments_for_pair})")
            print(f"  Taylor Max Expectation Accuracy: {taylor_max_accuracy:.3f} ({taylor_max_expectation_correct}/{total_experiments_for_pair})")
            print(f"  Taylor Weighted Expectation Accuracy: {taylor_weighted_accuracy:.3f} ({taylor_weighted_expectation_correct}/{total_experiments_for_pair})")
            print(f"  Exceed Rate Expectation Accuracy: {exceed_rate_accuracy:.3f} ({exceed_rate_expectation_correct}/{total_experiments_for_pair})")
            print(f"  Average Delta Q-Drop Max: {avg_delta_max_q_drop:.6f}")
            print(f"  Average Delta Reward-Drop Max: {avg_delta_max_reward_drop:.6f}")
            print(f"  Average Delta Taylor Max: {avg_delta_max_abs_taylor_deviation:.6f}")
            print(f"  Average Delta Exceed Rate: {avg_delta_exceed_rate:.6f}")
            print(f"  Failed Expectations: {len(failed_expectations)}")
        
        return pair_specific_results
    
    def print_pair_specific_summary(self, pair_specific_results):
        """Print a comprehensive summary of pair-specific results."""
        if not pair_specific_results:
            return
        
        print("\n" + "="*70)
        print("PAIR-SPECIFIC RESULTS SUMMARY")
        print("="*70)
        
        # Sort pairs by overall performance (average of all accuracy metrics)
        def calculate_overall_accuracy(results):
            accuracy_metrics = [
                results['patient_zero_accuracy'],
                results['q_drop_max_accuracy'],
                results['q_drop_weighted_accuracy'],
                results['reward_drop_max_accuracy'],
                results['reward_drop_weighted_accuracy'],
                results['taylor_max_accuracy'],
                results['taylor_weighted_accuracy'],
                results['exceed_rate_accuracy']
            ]
            return np.mean([acc for acc in accuracy_metrics if not np.isnan(acc)])
        
        sorted_pairs = sorted(pair_specific_results.items(), 
                             key=lambda x: calculate_overall_accuracy(x[1]), 
                             reverse=True)
        
        print(f"{'Pair':<20} {'PZ Acc':<8} {'Q-Max':<8} {'Q-Wei':<8} {'R-Max':<8} {'R-Wei':<8} {'T-Max':<8} {'T-Wei':<8} {'E-Rate':<8} {'Failed':<8}")
        print("-" * 90)
        
        for pair_name, results in sorted_pairs:
            print(f"{pair_name:<20} "
                  f"{results['patient_zero_accuracy']:<8.3f} "
                  f"{results['q_drop_max_accuracy']:<8.3f} "
                  f"{results['q_drop_weighted_accuracy']:<8.3f} "
                  f"{results['reward_drop_max_accuracy']:<8.3f} "
                  f"{results['reward_drop_weighted_accuracy']:<8.3f} "
                  f"{results['taylor_max_accuracy']:<8.3f} "
                  f"{results['taylor_weighted_accuracy']:<8.3f} "
                  f"{results['exceed_rate_accuracy']:<8.3f} "
                  f"{results['failed_expectations_count']:<8}")
        
        print("\nLegend:")
        print("PZ Acc  = Patient Zero Accuracy")
        print("Q-Max   = Q-Drop Max Expectation Accuracy")
        print("Q-Wei   = Q-Drop Weighted Expectation Accuracy")
        print("R-Max   = Reward-Drop Max Expectation Accuracy") 
        print("R-Wei   = Reward-Drop Weighted Expectation Accuracy")
        print("T-Max   = Taylor Max Expectation Accuracy")
        print("T-Wei   = Taylor Weighted Expectation Accuracy")
        print("E-Rate  = Exceed Rate Expectation Accuracy")
        print("Failed  = Number of Failed Expectations")
        
        # Print best and worst performing pairs
        if len(sorted_pairs) > 0:
            best_pair = sorted_pairs[0]
            worst_pair = sorted_pairs[-1]
            
            print(f"\nBest performing pair: {best_pair[0]}")
            print(f"  Overall accuracy: {calculate_overall_accuracy(best_pair[1]):.3f}")
            
            print(f"\nWorst performing pair: {worst_pair[0]}")
            print(f"  Overall accuracy: {calculate_overall_accuracy(worst_pair[1]):.3f}")
    
    def save_results(self, accuracy_results, failed_expectations, pair_specific_results=None):
        """Save all results to CSV files."""
        print("\nSaving results to CSV files...")
        
        # Save accuracy results
        accuracy_file = os.path.join(self.logdir, 'accuracy_results.csv')
        with open(accuracy_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metric', 'Value'])
            for key, value in accuracy_results.items():
                writer.writerow([key, value])
        
        # Save detailed experiment results
        detailed_file = os.path.join(self.logdir, 'detailed_results.csv')
        with open(detailed_file, 'w', newline='') as csvfile:
            fieldnames = [
                'seed', 'agent_i_influencer_attacked', 'agent_j_influenced_observed', 'max_influence_t', 'min_influence_t',
                'high_patient_zero', 'high_patient_time', 'low_patient_zero', 'low_patient_time',
                'high_influencer_fault_detection_times', 'high_influenced_fault_detection_times',
                'low_influencer_fault_detection_times', 'low_influenced_fault_detection_times',
                'high_max_q_drop', 'high_weighted_q_drop_sum', 'high_max_reward_drop', 'high_weighted_reward_drop_sum',
                'high_max_abs_taylor_deviation', 'high_weighted_taylor_deviation_sum', 'high_exceed_rate', 'high_window_length',
                'low_max_q_drop', 'low_weighted_q_drop_sum', 'low_max_reward_drop', 'low_weighted_reward_drop_sum',
                'low_max_abs_taylor_deviation', 'low_weighted_taylor_deviation_sum', 'low_exceed_rate', 'low_window_length',
                'delta_max_q_drop', 'delta_weighted_q_drop_sum', 'delta_max_reward_drop', 'delta_weighted_reward_drop_sum',
                'delta_max_abs_taylor_deviation', 'delta_weighted_taylor_deviation_sum', 'delta_exceed_rate',
                'episode_length'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Process all pairs from all seeds
            for result in self.experiment_results:
                seed = result['seed']
                episode_length = result['episode_length']
                
                for pair_result in result['pair_results']:
                    row = {
                        'seed': seed,
                        'agent_i_influencer_attacked': pair_result['agent_i'],
                        'agent_j_influenced_observed': pair_result['agent_j'],
                        'max_influence_t': pair_result['max_influence_t'],
                        'min_influence_t': pair_result['min_influence_t'],
                        'high_patient_zero': pair_result['high_patient_zero'],
                        'high_patient_time': pair_result['high_patient_time'],
                        'low_patient_zero': pair_result['low_patient_zero'],
                        'low_patient_time': pair_result['low_patient_time'],
                        'high_influencer_fault_detection_times': pair_result['high_influencer_fault_detection_times'],
                        'high_influenced_fault_detection_times': pair_result['high_influenced_fault_detection_times'],
                        'low_influencer_fault_detection_times': pair_result['low_influencer_fault_detection_times'],
                        'low_influenced_fault_detection_times': pair_result['low_influenced_fault_detection_times'],
                        'episode_length': episode_length
                    }
                    
                    # Add high influence metrics
                    for key, value in pair_result['high_metrics'].items():
                        row[f'high_{key}'] = value
                    
                    # Add low influence metrics
                    for key, value in pair_result['low_metrics'].items():
                        row[f'low_{key}'] = value
                    
                    # Add delta metrics (high - low)
                    high_metrics = pair_result['high_metrics']
                    low_metrics = pair_result['low_metrics']
                    row['delta_max_q_drop'] = high_metrics['max_q_drop'] - low_metrics['max_q_drop']
                    row['delta_weighted_q_drop_sum'] = high_metrics['weighted_q_drop_sum'] - low_metrics['weighted_q_drop_sum']
                    row['delta_max_reward_drop'] = high_metrics['max_reward_drop'] - low_metrics['max_reward_drop']
                    row['delta_weighted_reward_drop_sum'] = high_metrics['weighted_reward_drop_sum'] - low_metrics['weighted_reward_drop_sum']
                    row['delta_max_abs_taylor_deviation'] = high_metrics['max_abs_taylor_deviation'] - low_metrics['max_abs_taylor_deviation']
                    row['delta_weighted_taylor_deviation_sum'] = high_metrics['weighted_taylor_deviation_sum'] - low_metrics['weighted_taylor_deviation_sum']
                    row['delta_exceed_rate'] = high_metrics['exceed_rate'] - low_metrics['exceed_rate']
                    
                    writer.writerow(row)
        
        # Save failed expectations
        if failed_expectations:
            failed_file = os.path.join(self.logdir, 'failed_expectations.csv')
            with open(failed_file, 'w', newline='') as csvfile:
                fieldnames = [
                    'seed', 'agent_i_influencer_attacked', 'agent_j_influenced_observed', 
                    'q_drop_max_failed', 'q_drop_weighted_failed', 'reward_drop_max_failed', 'reward_drop_weighted_failed',
                    'taylor_max_failed', 'taylor_weighted_failed', 'exceed_rate_failed',
                    'high_q_drop_max', 'low_q_drop_max', 'high_q_drop_weighted', 'low_q_drop_weighted',
                    'high_reward_drop_max', 'low_reward_drop_max', 'high_reward_drop_weighted', 'low_reward_drop_weighted',
                    'high_taylor_max', 'low_taylor_max', 'high_taylor_weighted', 'low_taylor_weighted',
                    'high_exceed_rate', 'low_exceed_rate'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(failed_expectations)
        
        # Save failed seeds
        if self.failed_seeds:
            failed_seeds_file = os.path.join(self.logdir, 'failed_seeds.csv')
            with open(failed_seeds_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['seed', 'error'])
                writer.writeheader()
                writer.writerows(self.failed_seeds)
        
        # Save pair-specific results
        if pair_specific_results:
            # Create pair-specific directory
            pair_dir = os.path.join(self.logdir, 'pair_specific_results')
            os.makedirs(pair_dir, exist_ok=True)
            
            # Save overall pair-specific accuracy summary
            pair_summary_file = os.path.join(self.logdir, 'pair_specific_accuracy_summary.csv')
            with open(pair_summary_file, 'w', newline='') as csvfile:
                fieldnames = [
                    'pair_name', 'agent_i_influencer', 'agent_j_influenced', 'total_experiments',
                    'patient_zero_accuracy', 'high_patient_zero_accuracy', 'low_patient_zero_accuracy',
                    'q_drop_max_accuracy', 'q_drop_weighted_accuracy',
                    'reward_drop_max_accuracy', 'reward_drop_weighted_accuracy',
                    'taylor_max_accuracy', 'taylor_weighted_accuracy', 'exceed_rate_accuracy',
                    'avg_high_q_drop_max', 'avg_low_q_drop_max',
                    'avg_high_reward_drop_max', 'avg_low_reward_drop_max',
                    'avg_high_taylor_max', 'avg_low_taylor_max',
                    'avg_high_exceed_rate', 'avg_low_exceed_rate',
                    'avg_delta_max_q_drop', 'avg_delta_weighted_q_drop_sum',
                    'avg_delta_max_reward_drop', 'avg_delta_weighted_reward_drop_sum',
                    'avg_delta_max_abs_taylor_deviation', 'avg_delta_weighted_taylor_deviation_sum',
                    'avg_delta_exceed_rate',
                    'failed_expectations_count'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for pair_name, results in pair_specific_results.items():
                    row = {
                        'pair_name': pair_name,
                        'agent_i_influencer': results['agent_i'],
                        'agent_j_influenced': results['agent_j'],
                        'total_experiments': results['total_experiments'],
                        'patient_zero_accuracy': results['patient_zero_accuracy'],
                        'high_patient_zero_accuracy': results['high_patient_zero_accuracy'],
                        'low_patient_zero_accuracy': results['low_patient_zero_accuracy'],
                        'q_drop_max_accuracy': results['q_drop_max_accuracy'],
                        'q_drop_weighted_accuracy': results['q_drop_weighted_accuracy'],
                        'reward_drop_max_accuracy': results['reward_drop_max_accuracy'],
                        'reward_drop_weighted_accuracy': results['reward_drop_weighted_accuracy'],
                        'taylor_max_accuracy': results['taylor_max_accuracy'],
                        'taylor_weighted_accuracy': results['taylor_weighted_accuracy'],
                        'exceed_rate_accuracy': results['exceed_rate_accuracy'],
                        'avg_high_q_drop_max': results['avg_high_q_drop_max'],
                        'avg_low_q_drop_max': results['avg_low_q_drop_max'],
                        'avg_high_reward_drop_max': results['avg_high_reward_drop_max'],
                        'avg_low_reward_drop_max': results['avg_low_reward_drop_max'],
                        'avg_high_taylor_max': results['avg_high_taylor_max'],
                        'avg_low_taylor_max': results['avg_low_taylor_max'],
                        'avg_high_exceed_rate': results['avg_high_exceed_rate'],
                        'avg_low_exceed_rate': results['avg_low_exceed_rate'],
                        'avg_delta_max_q_drop': results['avg_delta_max_q_drop'],
                        'avg_delta_weighted_q_drop_sum': results['avg_delta_weighted_q_drop_sum'],
                        'avg_delta_max_reward_drop': results['avg_delta_max_reward_drop'],
                        'avg_delta_weighted_reward_drop_sum': results['avg_delta_weighted_reward_drop_sum'],
                        'avg_delta_max_abs_taylor_deviation': results['avg_delta_max_abs_taylor_deviation'],
                        'avg_delta_weighted_taylor_deviation_sum': results['avg_delta_weighted_taylor_deviation_sum'],
                        'avg_delta_exceed_rate': results['avg_delta_exceed_rate'],
                        'failed_expectations_count': results['failed_expectations_count']
                    }
                    writer.writerow(row)
            
            # Save detailed results for each pair in separate CSV files
            for pair_name, results in pair_specific_results.items():
                # Detailed experiment results for this pair
                pair_detailed_file = os.path.join(pair_dir, f'{pair_name}_detailed_results.csv')
                with open(pair_detailed_file, 'w', newline='') as csvfile:
                    fieldnames = [
                        'seed', 'agent_i_influencer_attacked', 'agent_j_influenced_observed', 'max_influence_t', 'min_influence_t',
                        'high_patient_zero', 'high_patient_time', 'low_patient_zero', 'low_patient_time',
                        'high_influencer_fault_detection_times', 'high_influenced_fault_detection_times',
                        'low_influencer_fault_detection_times', 'low_influenced_fault_detection_times',
                        'high_max_q_drop', 'high_weighted_q_drop_sum', 'high_max_reward_drop', 'high_weighted_reward_drop_sum',
                        'high_max_abs_taylor_deviation', 'high_weighted_taylor_deviation_sum', 'high_exceed_rate', 'high_window_length',
                        'low_max_q_drop', 'low_weighted_q_drop_sum', 'low_max_reward_drop', 'low_weighted_reward_drop_sum',
                        'low_max_abs_taylor_deviation', 'low_weighted_taylor_deviation_sum', 'low_exceed_rate', 'low_window_length',
                        'delta_max_q_drop', 'delta_weighted_q_drop_sum', 'delta_max_reward_drop', 'delta_weighted_reward_drop_sum',
                        'delta_max_abs_taylor_deviation', 'delta_weighted_taylor_deviation_sum', 'delta_exceed_rate',
                        'episode_length'
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for data in results['raw_pair_data']:
                        row = {
                            'seed': data['seed'],
                            'agent_i_influencer_attacked': data['agent_i'],
                            'agent_j_influenced_observed': data['agent_j'],
                            'max_influence_t': data['max_influence_t'],
                            'min_influence_t': data['min_influence_t'],
                            'high_patient_zero': data['high_patient_zero'],
                            'high_patient_time': data['high_patient_time'],
                            'low_patient_zero': data['low_patient_zero'],
                            'low_patient_time': data['low_patient_time'],
                            'high_influencer_fault_detection_times': data['high_influencer_fault_detection_times'],
                            'high_influenced_fault_detection_times': data['high_influenced_fault_detection_times'],
                            'low_influencer_fault_detection_times': data['low_influencer_fault_detection_times'],
                            'low_influenced_fault_detection_times': data['low_influenced_fault_detection_times'],
                            'episode_length': data['episode_length']
                        }
                        
                        # Add high influence metrics
                        for key, value in data['high_metrics'].items():
                            row[f'high_{key}'] = value
                        
                        # Add low influence metrics
                        for key, value in data['low_metrics'].items():
                            row[f'low_{key}'] = value
                        
                        # Add delta metrics (high - low)
                        high_metrics = data['high_metrics']
                        low_metrics = data['low_metrics']
                        row['delta_max_q_drop'] = high_metrics['max_q_drop'] - low_metrics['max_q_drop']
                        row['delta_weighted_q_drop_sum'] = high_metrics['weighted_q_drop_sum'] - low_metrics['weighted_q_drop_sum']
                        row['delta_max_reward_drop'] = high_metrics['max_reward_drop'] - low_metrics['max_reward_drop']
                        row['delta_weighted_reward_drop_sum'] = high_metrics['weighted_reward_drop_sum'] - low_metrics['weighted_reward_drop_sum']
                        row['delta_max_abs_taylor_deviation'] = high_metrics['max_abs_taylor_deviation'] - low_metrics['max_abs_taylor_deviation']
                        row['delta_weighted_taylor_deviation_sum'] = high_metrics['weighted_taylor_deviation_sum'] - low_metrics['weighted_taylor_deviation_sum']
                        row['delta_exceed_rate'] = high_metrics['exceed_rate'] - low_metrics['exceed_rate']
                        
                        writer.writerow(row)
                
                # Failed expectations for this pair
                if results['failed_expectations']:
                    pair_failed_file = os.path.join(pair_dir, f'{pair_name}_failed_expectations.csv')
                    with open(pair_failed_file, 'w', newline='') as csvfile:
                        fieldnames = [
                            'seed', 'agent_i_influencer_attacked', 'agent_j_influenced_observed', 
                            'q_drop_max_failed', 'q_drop_weighted_failed', 'reward_drop_max_failed', 'reward_drop_weighted_failed',
                            'taylor_max_failed', 'taylor_weighted_failed', 'exceed_rate_failed',
                            'high_q_drop_max', 'low_q_drop_max', 'high_q_drop_weighted', 'low_q_drop_weighted',
                            'high_reward_drop_max', 'low_reward_drop_max', 'high_reward_drop_weighted', 'low_reward_drop_weighted',
                            'high_taylor_max', 'low_taylor_max', 'high_taylor_weighted', 'low_taylor_weighted',
                            'high_exceed_rate', 'low_exceed_rate'
                        ]
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(results['failed_expectations'])
        
        # Save cumulative influences CSV
        self.save_cumulative_influences_csv()
        
        # Save directional derivatives CSV
        self.save_directional_derivatives_csv()
        
        # Save Taylor deviations CSV
        self.save_taylor_deviations_csv()
        
        print(f"Results saved to {self.logdir}")
        print(f"- Accuracy results: {accuracy_file}")
        print(f"- Detailed results: {detailed_file}")
        if failed_expectations:
            print(f"- Failed expectations: {failed_file}")
        if self.failed_seeds:
            print(f"- Failed seeds: {failed_seeds_file}")
        if pair_specific_results:
            print(f"- Pair-specific accuracy summary: {pair_summary_file}")
            print(f"- Pair-specific detailed results saved in: {pair_dir}")
            print(f"  * {len(pair_specific_results)} individual pair CSV files created")
        print(f"- Cumulative influences: cumulative_influences_all_seeds.csv")
        print(f"- Directional derivatives: directional_derivatives_all_seeds.csv")
    
    def cleanup(self):
        """Clean up resources."""
        pass
    
    def run_precomputation_mode(self):
        """Run precomputation mode to generate reference Taylor values."""
        print("Running in precomputation mode...")
        self.setup_experiment()
        
        # Generate seeds for precomputation
        seeds = list(range(self.total_experiments))
        
        # Create precomputation directory
        cwd = os.getcwd()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        env_type = 'discrete' if self.maddpg.discrete_action else 'continuous'
        precompute_dir = os.path.join(cwd, 'data', 'precomputed_taylor_values', 
                                     f"{self.config.map_name}_{env_type}_{timestamp}")
        
        # Precompute and save reference Taylor values
        self.precompute_and_save_taylor_values(seeds, precompute_dir)
        
        print(f"\nPrecomputation completed successfully!")
        print(f"Precomputed Taylor values saved to: {precompute_dir}")
        print(f"Use --taylor_ref_dir {precompute_dir} to load these values in future experiments")
        
        self.cleanup()
    
    def run_full_experiment(self):
        """Run the complete multi-seed experiment pipeline."""
        # Check if running in precomputation mode
        if hasattr(self.config, 'precompute_taylor') and self.config.precompute_taylor:
            self.run_precomputation_mode()
            return
        
        self.setup_experiment()
        self.run_all_experiments()
        accuracy_results, failed_expectations = self.compute_accuracies()
        pair_specific_results = self.compute_pair_specific_accuracies()
        self.print_pair_specific_summary(pair_specific_results)
        self.save_results(accuracy_results, failed_expectations, pair_specific_results)
        
        # Patient zero analysis summary and results saving
        self.patient_zero_analyzer.print_summary_dual()
        patient_zero_stats = self.patient_zero_analyzer.get_statistics_dual()

        # Save patient zero analysis results
        pz_analysis_file = os.path.join(self.logdir, "patient_zero_analysis_detailed.csv")
        pz_summary_file = os.path.join(self.logdir, "patient_zero_analysis_summary.json")
        
        self.patient_zero_analyzer.save_detailed_results(pz_analysis_file)
        
        # Save summary statistics as JSON
        import json
        with open(pz_summary_file, 'w') as f:
            json.dump(patient_zero_stats, f, indent=2)
        
        print(f"\nMulti-seed experiment completed successfully!")
        print(f"Results saved to: {self.logdir}")
        print(f"Pair-specific analysis completed for {len(pair_specific_results)} agent pairs")
        print(f"Patient zero analysis saved to:")
        print(f"  - Detailed results: {pz_analysis_file}")
        print(f"  - Summary statistics: {pz_summary_file}")
        self.cleanup()


def create_config_from_args():
    """Create configuration from command line arguments."""
    parser = argparse.ArgumentParser(description="Multi-seed statistics experiment")
    parser.add_argument("map_name", help="Name of SMAC map")
    parser.add_argument("model_path", help="Model directory")
    parser.add_argument("--total_experiments", type=int, default=100,
                        help="Total number of seed experiments to run")
    parser.add_argument("--precompute_taylor", action="store_true",
                        help="Precompute reference Taylor error values and save them")
    parser.add_argument("--taylor_ref_dir", type=str, default=None,
                        help="Directory containing precomputed reference Taylor error values")
    
    return parser.parse_args()


def main():
    """Main function to run multi-seed statistics experiment."""
    config = create_config_from_args()
    runner = MultiSeedExperimentRunner(config)
    runner.run_full_experiment()


if __name__ == '__main__':
    main()
