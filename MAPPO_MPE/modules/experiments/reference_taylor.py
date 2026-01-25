"""
Reference Taylor error computation and caching for fault detection.
"""
import os
import pickle
import json
import random
import numpy as np
import torch
from collections import deque
from tqdm import tqdm
from datetime import datetime

from modules.metrics import compute_taylor_delta_policy


REF_TAYLOR_EPISODE_COUNT = 100


class ReferenceTaylorManager:
    """
    Manager for computing, caching, and loading reference Taylor error values.
    """
    
    def __init__(self, runner, env, config):
        """
        Initialize the reference Taylor manager.
        
        Args:
            runner: MAPPO runner instance
            env: Environment instance
            config: Configuration object
        """
        self.runner = runner
        self.env = env
        self.config = config
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
        nagents = self.runner.args.N
        result_dataset = [{} for _ in range(nagents)]
        epsilon = getattr(self.config, 'taylor_epsilon', 0.01)
        
        for episode in tqdm(range(total_episodes), desc=f"Reference episodes (seed {seed})"):
            # Reset environment with seed
            states, masks = self._reset_env(seed + episode)
            result_deques = [deque(maxlen=5) for _ in range(nagents)]
            timestep = 0
            
            done = np.array([False for _ in range(nagents)])
            
            while not done.all():
                # Add small noise to observations
                noise_scale = 1e-4
                noise = [np.random.normal(loc=0, scale=noise_scale, size=states[i].shape) 
                        for i in range(nagents)]
                states = [states[i] + noise[i] for i in range(nagents)]
                
                # Compute Taylor delta policy
                results = compute_taylor_delta_policy(self.runner.agent_n, states, epsilon)
                
                # Store results for each agent at this timestep
                for agent_id in range(nagents):
                    result_deques[agent_id].append(results[agent_id])
                    if timestep not in result_dataset[agent_id]:
                        result_dataset[agent_id][timestep] = []
                    result_dataset[agent_id][timestep].append(np.mean(result_deques[agent_id]))
                
                # Get actions
                actions = []
                for agent_id in range(nagents):
                    mask = masks[agent_id] if masks is not None else None
                    action, _ = self.runner.agent_n.select_action(
                        states[agent_id], agent_id, evaluate=True, return_dist=True, action_mask=mask
                    )
                    actions.append(int(action))
                
                # Environment step
                next_states, _, done, _, masks = self._step_env(actions)
                states = next_states
                timestep += 1
        
        # Compute reference values and standard deviations
        ref_vals = [[] for _ in range(nagents)]
        ref_std_devs = [[] for _ in range(nagents)]
        
        for agent_id in range(nagents):
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
        nagents = self.runner.args.N
        if self.cache['nagents'] != nagents:
            raise ValueError(
                f"Number of agents mismatch: saved={self.cache['nagents']}, "
                f"current={nagents}"
            )
        
        if self.cache['env_id'] != self.config.env_id:
            print(f"Warning: Environment ID mismatch: saved={self.cache['env_id']}, "
                  f"current={self.config.env_id}")
        
        print(f"Successfully loaded reference Taylor values cache:")
        print(f"  Episode count used: {self.cache['ref_taylor_episode_count']}")
        print(f"  Number of agents: {self.cache['nagents']}")
        print(f"  Total seeds in file: {self.cache['total_seeds']}")
        print(f"  Available seeds: {sorted(self.cache['seeds'])}")
        
        return True
    
    def get_reference_values(self, seed):
        """
        Get reference Taylor error values for a seed (from cache or compute).
        
        Args:
            seed: Random seed to get values for
            
        Returns:
            Tuple of (ref_vals, ref_std_devs) for each agent and timestep
        """
        if self.cache is not None:
            ref_vals, ref_std_devs = self.cache['seeds_data'][seed]
            print(f"Using cached reference Taylor values for seed {seed}")
            return ref_vals, ref_std_devs
        else:
            return self.compute_reference_taylor_error(seed)
    
    def save_all_seeds(self, all_seeds_data, save_dir):
        """
        Save reference Taylor error values for all seeds to disk in a single file.
        
        Args:
            all_seeds_data: Dictionary mapping seed -> (ref_vals, ref_std_devs)
            save_dir: Directory to save the reference values
        """
        os.makedirs(save_dir, exist_ok=True)
        
        nagents = self.runner.args.N
        
        # Prepare data structure for all seeds
        data = {
            'seeds_data': all_seeds_data,
            'seeds': list(all_seeds_data.keys()),
            'total_seeds': len(all_seeds_data),
            'nagents': nagents,
            'env_id': self.config.env_id,
            'model_path': getattr(self.config, 'model_path', None),
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
            'nagents': nagents,
            'env_id': self.config.env_id,
            'model_path': getattr(self.config, 'model_path', None),
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
    
    def precompute_multiple_seeds(self, seeds, save_dir):
        """
        Precompute reference Taylor values for multiple seeds and save them.
        
        Args:
            seeds: List of seeds to precompute values for
            save_dir: Directory to save the precomputed values
        """
        print(f"Precomputing reference Taylor values for {len(seeds)} seeds...")
        print(f"Save directory: {save_dir}")
        
        os.makedirs(save_dir, exist_ok=True)
        
        nagents = self.runner.args.N
        
        # Collect all seeds data in a dictionary
        all_seeds_data = {}
        
        for i, seed in enumerate(tqdm(seeds, desc="Precomputing Taylor values")):
            print(f"\nPrecomputing seed {seed} ({i+1}/{len(seeds)})...")
            ref_vals, ref_std_devs = self.compute_reference_taylor_error(seed)
            all_seeds_data[seed] = (ref_vals, ref_std_devs)
        
        # Save all seeds data to a single file
        self.save_all_seeds(all_seeds_data, save_dir)
        
        # Save metadata about the precomputation
        metadata = {
            'seeds': seeds,
            'total_seeds': len(seeds),
            'nagents': nagents,
            'env_id': self.config.env_id,
            'model_path': getattr(self.config, 'model_path', None),
            'ref_taylor_episode_count': REF_TAYLOR_EPISODE_COUNT,
            'precomputation_timestamp': datetime.now().isoformat()
        }
        
        metadata_filepath = os.path.join(save_dir, "precomputation_metadata.json")
        with open(metadata_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nPrecomputation completed!")
        print(f"Metadata saved to: {metadata_filepath}")

    # ------------------------------------------------------------------
    # Environment helpers shared with the episode runner
    # ------------------------------------------------------------------
    def _reset_env(self, seed: int):
        try:
            result = self.env.reset(seed=seed)
        except TypeError:
            if hasattr(self.env, 'seed'):
                self.env.seed(seed)
            result = self.env.reset()
        return self._extract_states_and_masks(result)

    def _step_env(self, actions):
        result = self.env.step(actions)
        if not isinstance(result, tuple):
            raise ValueError('Environment step is expected to return a tuple')
        states = result[0]
        rewards = result[1]
        dones = result[2]
        info = result[3] if len(result) > 3 else {}
        masks = None
        if len(result) > 4:
            for extra in result[4:]:
                if self._is_mask_like(extra, states):
                    masks = extra
                    break
        return states, rewards, dones, info, masks

    @staticmethod
    def _extract_states_and_masks(result):
        if isinstance(result, tuple):
            states = result[0]
            masks = None
            for extra in result[1:]:
                if ReferenceTaylorManager._is_mask_like(extra, states):
                    masks = extra
                    break
            return states, masks
        return result, None

    @staticmethod
    def _is_mask_like(candidate, states):
        if candidate is None:
            return False
        try:
            return len(candidate) == len(states)
        except TypeError:
            return False
