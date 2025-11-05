"""
SMAC-specific reference Taylor error computation and caching for fault detection.
"""
import os
import pickle
import json
import random
import numpy as np
import torch
from collections import deque
from tqdm import tqdm
from torch.autograd import Variable
from datetime import datetime

from utils.smac_wrapper import SmacWrapper
from modules.constants import torch_device, DEVICE
from modules.metrics import compute_taylor_delta_policy


REF_TAYLOR_EPISODE_COUNT = 1


class SmacReferenceTaylorManager:
    """
    Manager for computing, caching, and loading reference Taylor error values for SMAC environments.
    """
    
    def __init__(self, runner, map_name, config):
        """
        Initialize the SMAC reference Taylor manager.
        
        Args:
            runner: MAPPO runner instance
            map_name: Name of the SMAC map
            config: Configuration object
        """
        self.runner = runner
        self.map_name = map_name
        self.config = config
        self.nagents = runner.args.N
        self.cache = None
        
    def compute_reference_taylor_error(self, seed):
        """
        Compute reference Taylor error values for a given seed.
        
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
        
        total_episodes = REF_TAYLOR_EPISODE_COUNT
        result_dataset = [{} for _ in range(self.nagents)]
        
        for episode in tqdm(range(total_episodes), desc=f"Reference episodes (seed {seed})"):
            # Create new environment for SMAC
            env = SmacWrapper.make_env(self.map_name, seed=seed)
            obs, action_masks = env.reset()
            result_deques = [deque(maxlen=5) for _ in range(self.nagents)]
            timestep = 0
            
            while True:
                noise_scale = 1e-4
                noise = [np.random.normal(loc=0, scale=noise_scale, size=obs[i].shape) 
                        for i in range(self.nagents)]
                obs_noisy = [obs[i] + noise[i] for i in range(self.nagents)]

                # Get actions using MAPPO
                actions_list = []
                for agent_id in range(self.nagents):
                    action = self.runner.agent_n.select_action(
                        obs_noisy[agent_id], agent_id, evaluate=True, action_mask=action_masks[agent_id]
                    )
                    actions_list.append(action)
                actions = {agent_name: actions_list[i] 
                          for i, agent_name in enumerate(env.possible_agents)}
                
                # Compute Taylor delta policy
                results = compute_taylor_delta_policy(
                    self.runner, obs_noisy, list(actions.values()), env.action_space, 0.01
                )
                
                # Store results for each agent at this timestep
                for agent_id in range(self.nagents):
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
        ref_vals = [[] for _ in range(self.nagents)]
        ref_std_devs = [[] for _ in range(self.nagents)]
        
        for agent_id in range(self.nagents):
            sorted_timesteps = sorted(result_dataset[agent_id].keys())
            for timestep in sorted_timesteps:
                timestep_values = result_dataset[agent_id][timestep]
                mean_val = np.mean(timestep_values)
                std_val = np.std(timestep_values)
                ref_vals[agent_id].append(mean_val)
                ref_std_devs[agent_id].append(std_val)
        
        return ref_vals, ref_std_devs
    
    def load_cache(self):
        """
        Load reference Taylor error values from disk once and cache them.
        """
        if not hasattr(self.config, 'taylor_ref_dir') or self.config.taylor_ref_dir is None:
            return False
            
        pickle_filename = "ref_taylor_all_seeds.pkl"
        pickle_filepath = os.path.join(self.config.taylor_ref_dir, pickle_filename)
        
        if not os.path.exists(pickle_filepath):
            raise FileNotFoundError(f"Reference Taylor values not found: {pickle_filepath}")
        
        print(f"Loading reference Taylor values cache from {pickle_filepath}...")
        
        with open(pickle_filepath, 'rb') as f:
            self.cache = pickle.load(f)
        
        # Validate compatibility
        if self.cache['nagents'] != self.nagents:
            raise ValueError(
                f"Number of agents mismatch: saved={self.cache['nagents']}, "
                f"current={self.nagents}"
            )
        
        if self.cache['map_name'] != self.map_name:
            print(f"Warning: Map name mismatch: saved={self.cache['map_name']}, current={self.map_name}")
        
        print(f"Loaded reference values for {len(self.cache['seeds'])} seeds")
        return True
    
    def get_reference_values(self, seed):
        """
        Get reference Taylor error values for a seed (from cache or compute).
        
        Args:
            seed: Random seed
            
        Returns:
            Tuple of (ref_vals, ref_std_devs)
        """
        # Try to get from cache first
        if self.cache is not None and seed in self.cache['seeds']:
            seed_idx = self.cache['seeds'].index(seed)
            return self.cache['ref_vals'][seed_idx], self.cache['ref_std_devs'][seed_idx]
        
        # Otherwise compute
        return self.compute_reference_taylor_error(seed)
    
    def precompute_multiple_seeds(self, seeds, output_dir):
        """
        Precompute reference Taylor values for multiple seeds and save to disk.
        
        Args:
            seeds: List of seeds to precompute
            output_dir: Directory to save precomputed values
        """
        os.makedirs(output_dir, exist_ok=True)
        
        all_ref_vals = []
        all_ref_std_devs = []
        
        for seed in tqdm(seeds, desc="Precomputing reference Taylor values"):
            ref_vals, ref_std_devs = self.compute_reference_taylor_error(seed)
            all_ref_vals.append(ref_vals)
            all_ref_std_devs.append(ref_std_devs)
        
        # Save all results to a single pickle file
        cache_data = {
            'seeds': seeds,
            'ref_vals': all_ref_vals,
            'ref_std_devs': all_ref_std_devs,
            'nagents': self.nagents,
            'map_name': self.map_name,
            'discrete_action': True,  # SMAC is always discrete
            'timestamp': datetime.now().strftime("%Y%m%d-%H%M%S")
        }
        
        pickle_filepath = os.path.join(output_dir, "ref_taylor_all_seeds.pkl")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"Saved reference Taylor values for {len(seeds)} seeds to {pickle_filepath}")
        
        # Also save metadata as JSON
        metadata = {
            'seeds': seeds,
            'nagents': self.nagents,
            'map_name': self.map_name,
            'discrete_action': True,  # SMAC is always discrete
            'timestamp': cache_data['timestamp'],
            'ref_taylor_episode_count': REF_TAYLOR_EPISODE_COUNT
        }
        
        metadata_filepath = os.path.join(output_dir, "metadata.json")
        with open(metadata_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Saved metadata to {metadata_filepath}")
