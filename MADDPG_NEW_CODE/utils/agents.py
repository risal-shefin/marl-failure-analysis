from torch import Tensor
from torch.autograd import Variable
from torch.optim import Adam
from .networks import MLPNetwork
from .misc import gumbel_softmax, onehot_from_logits
from .noise import OUNoise
import torch
class DDPGAgent(object):
    """
    General class for DDPG agents (policy, critic, target policy, target
    critic, exploration noise)
    """
    def __init__(self, num_in_pol, num_out_pol, num_in_critic, hidden_dim=64,
                 lr=0.01, discrete_action=True, norm_in=True):
        """
        Inputs:
            num_in_pol (int): number of dimensions in policy input
            num_out_pol (int): number of dimensions in policy output
            num_in_critic (int): number of dimensions in critic input
        """
        print(f"Creating agent with critic input dimension: {num_in_critic}")
        
        # Define the policy and target policy networks
        self.policy = MLPNetwork(num_in_pol, num_out_pol,
                                hidden_dim=hidden_dim,
                                constrain_out=True,
                                norm_in=norm_in,
                                discrete_action=discrete_action)
        
        # Create target policy network with the exact same parameters
        self.target_policy = MLPNetwork(num_in_pol, num_out_pol,
                                      hidden_dim=hidden_dim,
                                      constrain_out=True,
                                      norm_in=norm_in,
                                      discrete_action=discrete_action)
        
        # Initialize target network with same weights as policy network
        # Use a manual parameter copy instead of hard_update
        for target_param, param in zip(self.target_policy.parameters(), self.policy.parameters()):
            target_param.data.copy_(param.data)
            
        # Print statement before creating critic networks
        print(f"Creating critic with input dimension: {num_in_critic}")
        
        # Define critic network and target critic network
        self.critic = MLPNetwork(num_in_critic, 1, hidden_dim=hidden_dim, norm_in=True)
        self.target_critic = MLPNetwork(num_in_critic, 1, hidden_dim=hidden_dim, norm_in=True)
        
        # Initialize target critic with same weights as critic
        # Use a manual parameter copy instead of hard_update
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
            
        # Create optimizers
        self.policy_optimizer = Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr)
        
        # Store action type
        self.discrete_action = discrete_action
        
        if self.discrete_action:
            # For discrete actions, we need a Categorical distribution for exploration
            self.exploration = 0.3  # exploration probability
        else:
            # For continuous actions, we use Gaussian noise for exploration
            self.exploration = OUNoise(num_out_pol)

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
        # Ensure obs has the right shape
        if obs.dim() == 1:
            # If obs is a vector with the right dimension, add batch dimension
            if obs.size(0) == self.policy.input_dim:
                obs = obs.unsqueeze(0)
            # If obs is a single scalar, expand to match input_dim
            elif obs.size(0) == 1:
                obs = obs.repeat(self.policy.input_dim).unsqueeze(0)
        elif obs.dim() > 2:
            # If obs is a higher dimensional tensor, reshape to [batch_size, input_dim]
            batch_size = obs.size(0)
            obs = obs.reshape(batch_size, -1)
            
            # Handle dimension mismatch by truncating or padding
            if obs.size(1) != self.policy.input_dim:
                if obs.size(1) > self.policy.input_dim:
                    obs = obs[:, :self.policy.input_dim]  # Truncate
                else:
                    # Pad with zeros
                    padding = torch.zeros(batch_size, self.policy.input_dim - obs.size(1))
                    obs = torch.cat([obs, padding], dim=1)
                    
        # Forward through policy
        action = self.policy(obs)
        
        # Add exploration noise if requested
        if explore:
            if self.discrete_action:
                # For discrete actions, use epsilon-greedy
                if np.random.random() < self.exploration:
                    action = torch.FloatTensor(np.random.randint(0, action.size(-1), size=action.size(0)))
            else:
                # For continuous actions, use OUNoise
                action = action + Variable(torch.Tensor(self.exploration.noise()), requires_grad=False)
                
        return action

    def get_params(self):
        return {'policy': self.policy.state_dict(),
                'critic': self.critic.state_dict(),
                'target_policy': self.target_policy.state_dict(),
                'target_critic': self.target_critic.state_dict(),
                'policy_optimizer': self.policy_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict()}

    def load_params(self, params):
        self.policy.load_state_dict(params['policy'])
        self.critic.load_state_dict(params['critic'])
        self.target_policy.load_state_dict(params['target_policy'])
        self.target_critic.load_state_dict(params['target_critic'])
        self.policy_optimizer.load_state_dict(params['policy_optimizer'])
        self.critic_optimizer.load_state_dict(params['critic_optimizer'])
