import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo.mpe import simple_reference_v3, simple_speaker_listener_v4, simple_spread_v3

class PettingZooWrapper:
    """
    Wrapper for PettingZoo environments to match the interface expected by MADDPG
    """
    def __init__(self, env_name, discrete_action=False):
        # Add this line to identify as a PettingZoo environment
        self.env_type = 'pettingzoo'
        
        # Create the appropriate environment based on the name
        if env_name == "simple_reference_v3":
            self.env = simple_reference_v3.parallel_env(
                max_cycles=25, continuous_actions=not discrete_action, render_mode=None
            )
        elif env_name == "simple_speaker_listener_v4":
            self.env = simple_speaker_listener_v4.parallel_env(
                max_cycles=25, continuous_actions=not discrete_action, render_mode=None
            )
        elif env_name == "simple_spread_v3":
            self.env = simple_spread_v3.parallel_env(
                N=3, max_cycles=25, continuous_actions=not discrete_action, render_mode=None
            )
        else:
            raise ValueError(f"Unknown environment: {env_name}")
        
        # Reset the environment to get initial observations and setup
        observations, _ = self.env.reset(seed=0)
        
        # Store agent IDs in a consistent order
        self.agent_ids = list(self.env.agents)
        self.n = len(self.agent_ids)  # Number of agents
        
        # Set observation and action spaces
        self.observation_space = []
        self.action_space = []
        
        for agent in self.agent_ids:
            self.observation_space.append(self.env.observation_space(agent))
            self.action_space.append(self.env.action_space(agent))
    
    def reset(self):
        observations, _ = self.env.reset()
        # Convert dict to list of observations in agent_ids order
        obs_list = [observations[agent] for agent in self.agent_ids]
        
        # Return observations in the format expected by MADDPG: shape (n_rollout_threads, n_agents, obs_dim)
        # For each agent, we need a separate array with shape (n_rollout_threads, obs_dim)
        # Since we have only 1 environment, n_rollout_threads=1
        return [obs_list]  # This gives shape (1, n_agents, obs_dim)
    
    def step(self, actions):
        """
        Take a step in the environment with the given actions
        
        Args:
            actions: List of actions for each agent
                    For discrete actions, these should be integers or tensors with a single integer
                    For continuous actions, these should be numpy arrays or tensors
        
        Returns:
            next_obs: Observations after taking the actions
            rewards: Rewards for each agent
            dones: Done flags for each agent
            infos: Additional information
        """
        # Print debug info about actions
        # print(f"Action input type: {type(actions)}, content: {actions}")
        
        # Ensure actions are properly formatted
        if isinstance(actions, np.ndarray):
            # Extract the actions for this environment (first dimension)
            env_actions = actions[0]
        else:
            # If actions is a list of tensors or other format
            env_actions = []
            for action in actions:
                # Convert tensors to numpy arrays
                if hasattr(action, 'numpy'):
                    action = action.detach().numpy()
                # For tensors with multiple values, take the argmax for discrete actions
                if isinstance(action, np.ndarray):
                    if len(action.shape) > 1 and action.shape[1] > 1:
                        # This is for discrete actions represented as logits
                        action = np.argmax(action, axis=1)[0]
                    elif len(action.shape) == 1:
                        # Vector action, take the first element for discrete case
                        action = action[0]
                env_actions.append(action)
        
        # Create action dictionary
        action_dict = {}
        for i, agent in enumerate(self.agent_ids):
            try:
                # Get action for this agent
                if i < len(env_actions):
                    action = env_actions[i]
                    
                    # For discrete action spaces, ensure int type
                    if isinstance(self.action_space[i], Discrete):
                        # Convert to int and ensure within bounds
                        if isinstance(action, np.ndarray):
                            if action.size > 1:
                                action = np.argmax(action)  # Take most likely action
                            else:
                                action = int(action.item())
                        elif isinstance(action, (np.floating, float)):
                            action = int(action)
                        # Ensure within action space bounds
                        action = max(0, min(action, self.action_space[i].n - 1))
                    
                    # Add to action dict
                    action_dict[agent] = action
                    # print(f"Agent {i} action: {action}, type: {type(action)}")
                else:
                    # print(f"Warning: No action for agent {i}, using random action")
                    action_dict[agent] = self.action_space[i].sample()
            except Exception as e:
                # print(f"Error processing action for agent {i}: {e}")
                # Use random action as fallback
                action_dict[agent] = self.action_space[i].sample()
        
        # Clip actions for Env action spaces
        for i, agent in enumerate(self.agent_ids):
            low, high = self.action_space[i].low, self.action_space[i].high
            action_dict[agent] = np.clip(action_dict[agent], low, high)
        
        # Execute step
        observations, rewards, terminations, truncations, infos = self.env.step(action_dict)
        
        # Process observations, rewards and dones
        obs_list = [observations[agent] for agent in self.agent_ids]
        reward_list = [rewards[agent] for agent in self.agent_ids]
        done_list = [terminations[agent] or truncations[agent] for agent in self.agent_ids]
        
        # Format to match expected shapes
        # Shape [1, n_agents, obs_dim] for next_obs
        # Shape [1, n_agents] for rewards and dones
        next_obs = [obs_list]
        rewards = np.array([reward_list])
        dones = np.array([done_list])
        
        return next_obs, rewards, dones, infos
    
    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()

def make_env(env_id, discrete_action=False):
    """
    Create a PettingZoo environment that matches the interface expected by MADDPG
    
    Args:
        env_id: Name of the environment (e.g., "simple_reference_v3")
        discrete_action: Whether to use discrete actions
        
    Returns:
        A wrapped PettingZoo environment
    """
    return PettingZooWrapper(env_id, discrete_action)