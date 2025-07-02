import numpy as np
from pettingzoo.mpe import simple_spread_v3
from gymnasium.spaces import Box, Discrete
import torch


class PettingZooWrapper:
    """
    Wrapper for PettingZoo environments to match the interface expected by MAPPO
    """
    def __init__(self, env_name="simple_spread_v3", continuous=False):
        if env_name == "simple_spread_v3":
            self.env = simple_spread_v3.parallel_env(N=3, max_cycles=25, continuous_actions=continuous)
        else:
            raise ValueError(f"Unsupported environment: {env_name}")
        
        obs, _ = self.env.reset(seed=42)  # Initialize with a seed for reproducibility
        self.n = len(self.env.agents)  # Number of agents
        self.agent_ids = list(self.env.agents)
        
        # Set up observation and action spaces
        self.observation_space = []
        self.action_space = []
        
        for agent in self.agent_ids:
            self.observation_space.append(self.env.observation_space(agent))
            self.action_space.append(self.env.action_space(agent))
    
    def reset(self, seed=None):
        observations, _ = self.env.reset(seed=seed) if seed is not None else self.env.reset()
        # Convert the dict to a list in the same order as agent_ids
        obs_list = [observations[agent] for agent in self.agent_ids]
        return obs_list
    
    def step(self, actions):
        # Convert list of actions to dict
        action_dict = {agent_id: action for agent_id, action in zip(self.agent_ids, actions)}
        
        # Execute actions
        observations, rewards, terminations, truncations, infos = self.env.step(action_dict)
        
        # Convert from dicts to lists
        obs_list = [observations[agent] for agent in self.agent_ids]
        reward_list = [rewards[agent] for agent in self.agent_ids]
        done_list = [terminations[agent] or truncations[agent] for agent in self.agent_ids]
        info_list = [infos[agent] if agent in infos else {} for agent in self.agent_ids]
        
        # Check if episode is done
        done = all(done_list)
        
        return obs_list, reward_list, done_list, info_list
    
    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()


def make_env(env_name="simple_spread_v3", discrete=True):
    """
    Create a wrapped PettingZoo environment.
    
    Args:
        env_name: The name of the environment
        discrete: Whether to use discrete action space
    
    Returns:
        A wrapped environment
    """
    # Convert discrete=True to continuous=False (and vice versa)
    continuous = not discrete
    return PettingZooWrapper(env_name, continuous=continuous)