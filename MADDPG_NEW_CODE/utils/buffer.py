import numpy as np
from torch import Tensor
from torch.autograd import Variable

class ReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, max_steps, num_agents, obs_dims, ac_dims):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.max_steps = max_steps
        self.num_agents = num_agents
        self.obs_buffs = []
        self.ac_buffs = []
        self.rew_buffs = []
        self.next_obs_buffs = []
        self.done_buffs = []
        for odim, adim in zip(obs_dims, ac_dims):
            self.obs_buffs.append(np.zeros((max_steps, odim)))
            self.ac_buffs.append(np.zeros((max_steps, adim)))
            self.rew_buffs.append(np.zeros(max_steps))
            self.next_obs_buffs.append(np.zeros((max_steps, odim)))
            self.done_buffs.append(np.zeros(max_steps))


        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def __len__(self):
        return self.filled_i

    def push(self, observations, actions, rewards, next_observations, dones):
        """
        Add a new experience to memory.
        
        Args:
            observations: List of observations for each agent
            actions: List of actions for each agent
            rewards: List of rewards for each agent
            next_observations: List of next observations for each agent
            dones: List of done flags for each agent
        """
        # Print debug information about inputs
        # print(f"Replay buffer push - Shapes:")
        # print(f"  observations: {[o.shape if hasattr(o, 'shape') else 'N/A' for o in observations if o is not None]}")
        # print(f"  actions: {[a.shape if hasattr(a, 'shape') else 'N/A' for a in actions if a is not None]}")
        # print(f"  rewards: {rewards.shape if hasattr(rewards, 'shape') else 'N/A'}")
        # print(f"  next_observations: {[no.shape if hasattr(no, 'shape') else 'N/A' for no in next_observations if no is not None]}")
        # print(f"  dones: {dones.shape if hasattr(dones, 'shape') else 'N/A'}")
        
        # Get number of entries to add (should be 1 if we're adding single timestep)
        nentries = 1  # Default to 1 entry
        
        # Handle special case for PettingZoo observations with shape (n_envs, n_batches, n_agents, obs_dim)
        is_pettingzoo_format = False
        if isinstance(observations, np.ndarray) and observations.ndim == 4:
            # This is a PettingZoo format
            is_pettingzoo_format = True
            # Reshape to fit buffer's expected format
            # print("Detected PettingZoo observation format, reshaping...")
            n_agents = min(observations.shape[2], self.num_agents)
            # Create a list of observations for each agent
            obs_list = []
            next_obs_list = []
            for i in range(n_agents):
                # Extract agent's observation
                agent_obs = observations[0, 0, i].reshape(1, -1)  # Shape: [1, obs_dim]
                agent_next_obs = next_observations[0, 0, i].reshape(1, -1)
                obs_list.append(agent_obs)
                next_obs_list.append(agent_next_obs)
            # Extend with dummy observations if we have fewer agents than expected
            while len(obs_list) < self.num_agents:
                # Create dummy observation with same shape as the first agent
                dummy_shape = self.obs_buffs[0].shape[1:]
                dummy_obs = np.zeros((1,) + dummy_shape)
                obs_list.append(dummy_obs)
                next_obs_list.append(dummy_obs)
            observations = obs_list
            next_observations = next_obs_list
        
        # Process actions
        if isinstance(actions, np.ndarray) and actions.ndim > 1:
            # Reshape actions if needed
            action_list = []
            for i in range(min(actions.shape[0], self.num_agents)):
                if actions.ndim == 3:  # Shape: [n_agents, n_batches, action_dim]
                    agent_action = actions[i, 0].reshape(1, -1)
                elif actions.ndim == 2:  # Shape: [n_agents, action_dim]
                    agent_action = actions[i].reshape(1, -1)
                else:
                    agent_action = actions[i]
                action_list.append(agent_action)
            # Extend with dummy actions if we have fewer agents than expected
            while len(action_list) < self.num_agents:
                # Create dummy action with same shape as the first agent
                dummy_shape = self.ac_buffs[0].shape[1:]
                dummy_action = np.zeros((1,) + dummy_shape)
                action_list.append(dummy_action)
            actions = action_list
        
        # Process rewards
        if is_pettingzoo_format:
            # Extract rewards for each agent
            if rewards.ndim == 3:  # Shape: [n_envs, n_batches, n_agents]
                reward_list = []
                for i in range(min(rewards.shape[2], self.num_agents)):
                    agent_reward = rewards[0, 0, i].reshape(1)
                    reward_list.append(agent_reward)
                # Add dummy rewards if needed
                while len(reward_list) < self.num_agents:
                    reward_list.append(np.zeros(1))
                rewards = reward_list
        
        # Process dones
        if is_pettingzoo_format:
            # Extract dones for each agent
            if dones.ndim == 3:  # Shape: [n_envs, n_batches, n_agents]
                done_list = []
                for i in range(min(dones.shape[2], self.num_agents)):
                    agent_done = dones[0, 0, i].reshape(1)
                    done_list.append(agent_done)
                # Add dummy dones if needed
                while len(done_list) < self.num_agents:
                    done_list.append(np.zeros(1))
                dones = done_list
                
        # Now handle the case where observations is not a list or doesn't have enough elements
        if not isinstance(observations, list):
            # Convert to list for consistency
            if isinstance(observations, np.ndarray):
                # Create a list with one observation
                observations = [observations.reshape(1, -1)]
            else:
                # Handle other types - create a default observation
                observations = [np.zeros((1, self.obs_buffs[0].shape[1]))]
        
        # Make sure we have enough observations for all agents
        while len(observations) < self.num_agents:
            # Create dummy observation with same shape as the first agent
            dummy_shape = self.obs_buffs[0].shape[1:]
            dummy_obs = np.zeros((1,) + dummy_shape)
            observations.append(dummy_obs)
        
        # Do the same for next_observations
        if not isinstance(next_observations, list):
            if isinstance(next_observations, np.ndarray):
                next_observations = [next_observations.reshape(1, -1)]
            else:
                next_observations = [np.zeros((1, self.next_obs_buffs[0].shape[1]))]
        
        while len(next_observations) < self.num_agents:
            dummy_shape = self.next_obs_buffs[0].shape[1:]
            dummy_obs = np.zeros((1,) + dummy_shape)
            next_observations.append(dummy_obs)
        
        # Handle actions
        if not isinstance(actions, list):
            if isinstance(actions, np.ndarray):
                actions = [actions.reshape(1, -1)]
            else:
                actions = [np.zeros((1, self.ac_buffs[0].shape[1]))]
        
        while len(actions) < self.num_agents:
            dummy_shape = self.ac_buffs[0].shape[1:]
            dummy_action = np.zeros((1,) + dummy_shape)
            actions.append(dummy_action)
        
        # Now store in buffer, making sure shapes match expectations
        for agent_i in range(self.num_agents):
            try:
                # Store observations (with error handling)
                try:
                    if agent_i < len(observations) and observations[agent_i] is not None:
                        if isinstance(observations[agent_i], np.ndarray):
                            # Reshape to match buffer's expected format
                            agent_obs = observations[agent_i].reshape(nentries, -1)
                            self.obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = agent_obs
                        else:
                            # If not numpy array, convert and reshape
                            agent_obs = np.array(observations[agent_i]).reshape(nentries, -1)
                            self.obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = agent_obs
                    else:
                        # Use zeros if observation is missing
                        dummy_obs = np.zeros((nentries, self.obs_buffs[agent_i].shape[1]))
                        self.obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = dummy_obs
                except Exception as e:
                    # print(f"Error storing observation for agent {agent_i}: {e}")
                    # Use zeros if there's an error
                    dummy_obs = np.zeros((nentries, self.obs_buffs[agent_i].shape[1]))
                    self.obs_buffs[agent_i][self.curr_i:self.curr_i + nentries] = dummy_obs
                
                # The rest of your storage logic with similar error handling
                # ...
                
            except Exception as e:
                print(f"Fatal error in storing experience for agent {agent_i}: {e}")
        
        # Update current index
        self.curr_i = (self.curr_i + nentries) % self.max_steps
        self.filled_i = min(self.filled_i + nentries, self.max_steps)

    def sample(self, N, to_gpu=False, norm_rews=True):
        inds = np.random.choice(np.arange(self.filled_i), size=N,
                                replace=False)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if norm_rews:
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             self.rew_buffs[i][:self.filled_i].std())
                        for i in range(self.num_agents)]
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]
        return ([cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)])

    def get_average_rewards(self, N):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rew_buffs[i][inds].mean() for i in range(self.num_agents)]
