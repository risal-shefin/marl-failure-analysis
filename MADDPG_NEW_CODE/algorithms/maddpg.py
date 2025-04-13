import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete
from utils.networks import MLPNetwork
from utils.misc import soft_update, hard_update, average_gradients, onehot_from_logits, gumbel_softmax
from utils.agents import DDPGAgent

MSELoss = torch.nn.MSELoss()

class MADDPG(object):
    """
    Wrapper class for DDPG-esque (Multi-Agent DDPG) agents in MARL
    """
    def __init__(self, agent_init_params, alg_types,
                 gamma=0.95, tau=0.01, lr=0.01,
                 hidden_dim=64, discrete_action=False):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether agent action is discrete
        """
        # Store init parameters for saving later
        self.init_dict = {
            'agent_init_params': agent_init_params,
            'alg_types': alg_types,
            'gamma': gamma,
            'tau': tau,
            'lr': lr,
            'hidden_dim': hidden_dim,
            'discrete_action': discrete_action
        }
        
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        self.agents = [DDPGAgent(lr=lr, hidden_dim=hidden_dim,
                                discrete_action=discrete_action,
                                **params)
                       for params in agent_init_params]
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.pol_dev = 'cpu'  # device for policies
        self.critic_dev = 'cpu'  # device for critics
        self.trgt_pol_dev = 'cpu'  # device for target policies
        self.trgt_critic_dev = 'cpu'  # device for target critics
        self.niter = 0

    @property
    def policies(self):
        return [a.policy for a in self.agents]

    @property
    def target_policies(self):
        return [a.target_policy for a in self.agents]

    def scale_noise(self, scale):
        """
        Scale noise for each agent
        Inputs:
            scale (float): scale of noise
        """
        for a in self.agents:
            a.scale_noise(scale)

    def reset_noise(self):
        for a in self.agents:
            a.reset_noise()

    def step(self, observations, explore=False):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            actions: List of actions for each agent
        """
        return [a.step(obs, explore=explore) for a, obs in zip(self.agents,
                                                                 observations)]

    def update(self, sample, agent_i, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        """
        obs, acs, rews, next_obs, dones = sample
        curr_agent = self.agents[agent_i]
        
        # Create input for target value function
        trgt_vf_in = torch.cat([o for o in next_obs] + 
                             [onehot_from_logits(pi(ob)) for pi, ob in 
                              zip(self.target_policies, next_obs)], dim=1)
        
        # print(f"trgt_vf_in shape: {trgt_vf_in.shape}")
        
        # Get device from parameter tensor instead of directly from module
        device = next(curr_agent.critic.parameters()).device
        
        # Check if we need to recreate the critics
        actual_dim = trgt_vf_in.shape[1]
        try:
            expected_dim = curr_agent.target_critic.fc1.in_features
        except AttributeError:
            expected_dim = 0
            
        # Check if dimensions don't match
        if actual_dim != expected_dim:
            print(f"EMERGENCY FIX: Critic dimension mismatch! Got {actual_dim}, expected {expected_dim}")
            print("Recreating critic networks with correct dimensions...")
            
            # Create new networks without batch normalization - use device properly
            curr_agent.critic = MLPNetwork(actual_dim, 1, 
                                         hidden_dim=64, 
                                         norm_in=False).to(device)
            curr_agent.target_critic = MLPNetwork(actual_dim, 1, 
                                               hidden_dim=64, 
                                               norm_in=False).to(device)
            
            # New optimizer with original learning rate
            try:
                original_lr = curr_agent.critic_optimizer.param_groups[0]['lr']
            except:
                original_lr = 0.01
                
            curr_agent.critic_optimizer = torch.optim.Adam(curr_agent.critic.parameters(), 
                                                        lr=original_lr)
            
            # Initialize target with main network
            hard_update(curr_agent.target_critic, curr_agent.critic)
            print("Critics successfully recreated")
        
        # Try to get target value - handle None values
        target_critic_output = curr_agent.target_critic(trgt_vf_in)
        
        # Check if we got a valid output
        if target_critic_output is None:
            print("ERROR: Target critic returned None - recreating networks")
            
            # Create new networks without batch normalization
            curr_agent.critic = MLPNetwork(actual_dim, 1, 
                                         hidden_dim=64, 
                                         norm_in=False).to(device)
            curr_agent.target_critic = MLPNetwork(actual_dim, 1, 
                                               hidden_dim=64, 
                                               norm_in=False).to(device)
            
            # Try again with the new networks
            target_critic_output = curr_agent.target_critic(trgt_vf_in)
            
            # If still None, use a zero tensor as fallback
            if target_critic_output is None:
                print("CRITICAL ERROR: Target critic still returning None - using zero tensor")
                target_critic_output = torch.zeros_like(rews[agent_i].view(-1, 1))
        
        # Now calculate target value safely
        target_value = (rews[agent_i].view(-1, 1) + self.gamma * 
                       target_critic_output *
                       (1 - dones[agent_i].view(-1, 1)))
        
        # Rest of your update method...

    def update_all_targets(self):
        """
        Update all target networks (called after normal updates have been
        performed for each agent)
        """
        for a in self.agents:
            soft_update(a.target_critic, a.critic, self.tau)
            soft_update(a.target_policy, a.policy, self.tau)
        self.niter += 1

    def prep_training(self, device='gpu'):
        for a in self.agents:
            a.policy.train()
            a.critic.train()
            a.target_policy.train()
            a.target_critic.train()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device
        if not self.critic_dev == device:
            for a in self.agents:
                a.critic = fn(a.critic)
            self.critic_dev = device
        if not self.trgt_pol_dev == device:
            for a in self.agents:
                a.target_policy = fn(a.target_policy)
            self.trgt_pol_dev = device
        if not self.trgt_critic_dev == device:
            for a in self.agents:
                a.target_critic = fn(a.target_critic)
            self.trgt_critic_dev = device

    def prep_rollouts(self, device='cpu'):
        for a in self.agents:
            a.policy.eval()
        if device == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        # only need main policy for rollouts
        if not self.pol_dev == device:
            for a in self.agents:
                a.policy = fn(a.policy)
            self.pol_dev = device

    def save(self, filename):
        """
        Save trained parameters of all agents into one file
        """
        self.prep_training(device='cpu')  # move parameters to CPU before saving
        save_dict = {
            'init_dict': self.init_dict,
            'agent_params': [agent.get_params() for agent in self.agents],
            'niter': self.niter
        }
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, agent_alg="MADDPG", adversary_alg="MADDPG",
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        alg_types = []
        
        # Calculate dimensions
        obs_dims = [obsp.shape[0] for obsp in env.observation_space]
        act_dims = []
        
        for acsp in env.action_space:
            if isinstance(acsp, Discrete):
                act_dims.append(acsp.n)
            else:
                act_dims.append(acsp.shape[0])
        
        # For Petting Zoo environments, hard-code the dimensions
        if hasattr(env, 'env_type') and env.env_type == 'pettingzoo':
            # print("Using custom MADDPG for PettingZoo with correct dimensions")
            
            # HARD-CODE: For a 2-agent environment with obs_dim=21 and act_dim=50 each
            total_obs_dim = sum(obs_dims)  # 21 + 21 = 42
            total_act_dim = sum(act_dims)  # 50 + 50 = 100
            
            # Total critic input should be all observations + all actions
            critic_input_dim = total_obs_dim + total_act_dim  # 42 + 100 = 142
            
            print(f"Hard-coded critic input dimensions: {critic_input_dim} (obs: {total_obs_dim}, act: {total_act_dim})")
        else:
            # Standard calculation for non-PettingZoo environments
            critic_input_dim = sum(obs_dims) + sum(act_dims)
        
        # Process each agent
        for i, (acsp, obsp) in enumerate(zip(env.action_space, env.observation_space)):
            num_in_pol = obsp.shape[0]  # Observation dimension for this agent's policy
            
            if isinstance(acsp, Discrete):
                discrete_action = True
                get_shape = lambda x: x.n
            else:
                discrete_action = False
                get_shape = lambda x: x.shape[0]
                
            num_out_pol = get_shape(acsp)  # Action dimension for this agent's policy
            
            # For MADDPG critic, use the hard-coded dimension
            if agent_alg == "MADDPG" or (i < getattr(env, 'n_adversaries', 0) and adversary_alg == "MADDPG"):
                num_in_critic = critic_input_dim  # Use our hard-coded value
                # print(f"Agent {i} - MADDPG critic input: {num_in_critic}")
            else:  # DDPG - only use this agent's information
                num_in_critic = obsp.shape[0] + get_shape(acsp)
                # print(f"Agent {i} - DDPG critic input: {num_in_critic}")
            
            agent_init_params.append({'num_in_pol': num_in_pol,
                                     'num_out_pol': num_out_pol,
                                     'num_in_critic': num_in_critic})
            alg_types.append(adversary_alg if i < getattr(env, 'n_adversaries', 0) else agent_alg)
        
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'hidden_dim': hidden_dim,
                     'alg_types': alg_types,
                     'agent_init_params': agent_init_params,
                     'discrete_action': discrete_action}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        save_dict = torch.load(filename)
        instance = cls(**save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance