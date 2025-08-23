from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np
from .networks import MLPNetwork, CNNNetwork, MultiAgentCriticNetwork
from .misc import hard_update, gumbel_softmax, onehot_from_logits
from .noise import OUNoise

class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=64,
                 lr=0.01, discrete_action=True, test_mode=False,
                 use_local_q=False):
        """
        Inputs:
            num_in_pol (int): number of dimensions for policy input
            num_out_pol (int): number of dimensions for policy output
            num_in_critic (int): number of dimensions for critic input
        """
        self.policy = MLPNetwork(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 discrete_action=discrete_action,
                                 test_mode=test_mode)
        self.critic = MLPNetwork(num_in_critic, 1,
                                 hidden_dim=hidden_dim,
                                 constrain_out=False,
                                 test_mode=test_mode)
        self.target_policy = MLPNetwork(num_in_pol, num_out_pol,
                                        hidden_dim=hidden_dim,
                                        constrain_out=True,
                                        discrete_action=discrete_action,
                                        test_mode=test_mode)
        self.target_critic = MLPNetwork(num_in_critic, 1,
                                        hidden_dim=hidden_dim,
                                        constrain_out=False,
                                        test_mode=test_mode)
        self.use_local_q = use_local_q
        if self.use_local_q:
            self.local_critic = MLPNetwork(num_in_pol + num_out_pol, 1,
                                           hidden_dim=hidden_dim,
                                           constrain_out=False,
                                           test_mode=test_mode)
            self.local_critic_optimizer = Adam(self.local_critic.parameters(),
                                               lr=lr)
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(obs)
        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                noise = Tensor(self.exploration.noise()).to(action.device)
                action = action + noise
            # action = action.clamp(-1, 1)
            action = action.clamp(0, 1)
        return action

    def get_params(self):
        params = {'policy': self.policy.state_dict(),
                  'critic': self.critic.state_dict(),
                  'target_policy': self.target_policy.state_dict(),
                  'target_critic': self.target_critic.state_dict(),
                  'policy_optimizer': self.policy_optimizer.state_dict(),
                  'critic_optimizer': self.critic_optimizer.state_dict()}
        if self.use_local_q:
            params.update({'local_critic': self.local_critic.state_dict(),
                           'local_critic_optimizer': self.local_critic_optimizer.state_dict()})
        return params

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])
        if self.use_local_q and 'local_critic' in params:
            self.local_critic.load_state_dict(params['local_critic'])
            self.local_critic_optimizer.load_state_dict(params['local_critic_optimizer'])


class DDPGImageAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise) for image inputs
    """
    def __init__(self, num_in_pol, num_out_pol,
                 obs_shapes_critic, total_action_dim, hidden_dim=64,
                 lr=0.01, discrete_action=True, num_in_critic=None, test_mode=False,
                 use_local_q=False):
        """
        Inputs:
            num_in_pol (tuple): Image Observation Shape
            num_out_pol (int): number of dimensions for policy output
            obs_shapes_critic (list of tuples): List of image observation shapes for each agent
            total_action_dim (int): Sum of action dimensions of all agents
            num_in_critic: Not used, kept for compatibility
        """
        n_agents = len(obs_shapes_critic)
        self.policy = CNNNetwork(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 discrete_action=discrete_action,
                                 test_mode=test_mode)
        self.critic = MultiAgentCriticNetwork(n_agents, obs_shapes_critic,
                                 total_action_dim,
                                 hidden_dim=hidden_dim,
                                 test_mode=test_mode)
        self.target_policy = CNNNetwork(num_in_pol, num_out_pol,
                                 hidden_dim=hidden_dim,
                                 constrain_out=True,
                                 discrete_action=discrete_action,
                                 test_mode=test_mode)
        self.target_critic = MultiAgentCriticNetwork(n_agents, obs_shapes_critic,
                                 total_action_dim,
                                 hidden_dim=hidden_dim,
                                 test_mode=test_mode)
        self.use_local_q = use_local_q
        if self.use_local_q:
            local_in_dim = int(np.prod(num_in_pol)) + num_out_pol
            self.local_critic = MLPNetwork(local_in_dim, 1,
                                           hidden_dim=hidden_dim,
                                           constrain_out=False,
                                           test_mode=test_mode)
            self.local_critic_optimizer = Adam(self.local_critic.parameters(),
                                               lr=lr)
        hard_update(self.target_policy, self.policy)
        hard_update(self.target_critic, self.critic)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        if not discrete_action:
            self.exploration = OUNoise(num_out_pol)
        else:
            self.exploration = 0.3  # epsilon for eps-greedy
        self.discrete_action = discrete_action

    def reset_noise(self):
        if not self.discrete_action:
            self.exploration.reset()

    def scale_noise(self, scale):
        if self.discrete_action:
            self.exploration = scale
        else:
            self.exploration.scale = scale

    def step(self, obs, explore=False):
        """
        Take a step forward in environment for a minibatch of observations
        Inputs:
            obs (PyTorch Variable): Observations for this agent
            explore (boolean): Whether or not to add exploration noise
        Outputs:
            action (PyTorch Variable): Actions for this agent
        """
        action = self.policy(obs)
        if self.discrete_action:
            if explore:
                action = gumbel_softmax(action, hard=True)
            else:
                action = onehot_from_logits(action)
        else:  # continuous action
            if explore:
                noise = Tensor(self.exploration.noise()).to(action.device)
                action = action + noise
            # action = action.clamp(-1, 1)
            action = action.clamp(0, 1)
        return action

    def get_params(self):
        params = {'policy': self.policy.state_dict(),
                  'critic': self.critic.state_dict(),
                  'target_policy': self.target_policy.state_dict(),
                  'target_critic': self.target_critic.state_dict(),
                  'policy_optimizer': self.policy_optimizer.state_dict(),
                  'critic_optimizer': self.critic_optimizer.state_dict()}
        if self.use_local_q:
            params.update({'local_critic': self.local_critic.state_dict(),
                           'local_critic_optimizer': self.local_critic_optimizer.state_dict()})
        return params

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])
        if self.use_local_q and 'local_critic' in params:
            self.local_critic.load_state_dict(params['local_critic'])
            self.local_critic_optimizer.load_state_dict(params['local_critic_optimizer'])
