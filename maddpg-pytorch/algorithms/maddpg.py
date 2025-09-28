import torch
import torch.nn.functional as F
from gym.spaces import Box, Discrete
import gymnasium
from utils.networks import MLPNetwork
from utils.misc import soft_update, average_gradients, onehot_from_logits, gumbel_softmax
from utils.agents import DDPGAgent, DDPGImageAgent

MSELoss = torch.nn.MSELoss()

class MADDPG(object):
    """
    Wrapper class for DDPG-esque (i.e. also MADDPG) agents in multi-agent task
    """
    def __init__(self, agent_init_params, alg_types,
                 gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                 discrete_action=False, obs_shapes=None,
                 total_action_dim=None, test_mode=False,
                 local_q=False):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
                num_in_critic (int): Input dimensions to critic
            alg_types (list of str): Learning algorithm for each agent (DDPG
                                       or MADDPG)
            gamma (float): Discount factor
            tau (float): Target update rate
            lr (float): Learning rate for policy and critic
            hidden_dim (int): Number of hidden dimensions for networks
            discrete_action (bool): Whether or not to use discrete action space
        """
        self.nagents = len(alg_types)
        self.alg_types = alg_types
        if obs_shapes and any(len(obs_shape) == 3 for obs_shape in obs_shapes if isinstance(obs_shape, tuple)):
            self.agents = [DDPGImageAgent(lr=lr, discrete_action=discrete_action,
                 hidden_dim=hidden_dim,
                 obs_shapes_critic=obs_shapes,
                 total_action_dim=total_action_dim,
                 test_mode=test_mode,
                 has_local_q=local_q,
                 **params)
               for params in agent_init_params]
        else:
            self.agents = [DDPGAgent(lr=lr, discrete_action=discrete_action,
                    hidden_dim=hidden_dim,
                    test_mode=test_mode,
                    has_local_q=local_q,
                    **params)
                for params in agent_init_params]
        self.agent_init_params = agent_init_params
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.discrete_action = discrete_action
        self.local_q = local_q
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

    def step(self, observations, explore=False, action_masks=None):
        """
        Take a step forward in environment with all agents
        Inputs:
            observations: List of observations for each agent
            explore (boolean): Whether or not to add exploration noise
            action_masks: Optional list of masks for each agent
        Outputs:
            actions: List of actions for each agent
        """
        if action_masks is None:
            return [a.step(obs, explore=explore) for a, obs in zip(self.agents, observations)]
        return [a.step(obs, explore=explore, action_mask=mask)
                for a, obs, mask in zip(self.agents, observations, action_masks)]

    def get_action_logits(self, observations, action_masks=None):
        assert self.discrete_action, "get_action_logits is only available for discrete action spaces"
        if action_masks is None:
            return [a.policy(obs) for a, obs in zip(self.agents, observations)]
        logits = []
        for a, obs, mask in zip(self.agents, observations, action_masks):
            logit = a.policy(obs)
            if mask is not None:
                logit = logit.masked_fill(mask == 0, float('-1e9'))
            logits.append(logit)
        return logits


    def update(self, sample, agent_i, parallel=False, logger=None):
        """
        Update parameters of agent model based on sample from replay buffer
        Inputs:
            sample: tuple of (observations, actions, rewards, next
                    observations, and episode end masks) sampled randomly from
                    the replay buffer. Each is a list with entries
                    corresponding to each agent
            agent_i (int): index of agent to update
            parallel (bool): If true, will average gradients across threads
            logger (SummaryWriter from Tensorboard-Pytorch):
                If passed in, important quantities will be logged
        """
        obs, acs, rews, next_obs, dones, avail_actions, next_avail_actions = sample
        curr_agent = self.agents[agent_i]
        is_obs_image = len(obs[agent_i].shape) >= 3 # the first dimension can be the batch size

        curr_agent.critic_optimizer.zero_grad()
        if self.alg_types[agent_i] == 'MADDPG':
            if self.discrete_action: # one-hot encode action
                all_trgt_acs = []
                for pi, nobs, mask in zip(self.target_policies, next_obs, next_avail_actions):
                    logits = pi(nobs)
                    if mask is not None:
                        logits = logits.masked_fill(mask == 0, float('-inf'))
                    all_trgt_acs.append(onehot_from_logits(logits))
            else:
                all_trgt_acs = [pi(nobs) for pi, nobs in zip(self.target_policies,
                                                             next_obs)]
            trgt_vf_in = torch.cat((*next_obs, *all_trgt_acs), dim=1) if not is_obs_image else (next_obs, all_trgt_acs)
        else:  # DDPG
            if self.discrete_action:
                trgt_vf_in = torch.cat((next_obs[agent_i],
                                        onehot_from_logits(
                                            curr_agent.target_policy(
                                                next_obs[agent_i]))),
                                       dim=1)
            else:
                trgt_vf_in = torch.cat((next_obs[agent_i],
                                        curr_agent.target_policy(next_obs[agent_i])),
                                       dim=1)
                
        if is_obs_image:
            target_critic_val = curr_agent.target_critic(*trgt_vf_in)
        else:
            target_critic_val = curr_agent.target_critic(trgt_vf_in)
        target_value = (rews[agent_i].view(-1, 1) + self.gamma *
                        target_critic_val * (1 - dones[agent_i].view(-1, 1)))
        if self.alg_types[agent_i] == 'MADDPG':
            vf_in = torch.cat((*obs, *acs), dim=1) if not is_obs_image else (obs, acs)
        else:  # DDPG
            vf_in = torch.cat((obs[agent_i], acs[agent_i]), dim=1)
        actual_value = curr_agent.critic(*vf_in) if is_obs_image else curr_agent.critic(vf_in)
        vf_loss = MSELoss(actual_value, target_value.detach())
        vf_loss.backward()
        if parallel:
            average_gradients(curr_agent.critic)
        torch.nn.utils.clip_grad_norm(curr_agent.critic.parameters(), 0.5)
        curr_agent.critic_optimizer.step()

        if curr_agent.has_local_q:
            curr_agent.local_critic_optimizer.zero_grad()
            if is_obs_image:
                local_obs = obs[agent_i].reshape(obs[agent_i].shape[0], -1)
            else:
                local_obs = obs[agent_i]
            local_vf_in = torch.cat((local_obs, acs[agent_i]), dim=1)
            local_value = curr_agent.local_critic(local_vf_in)
            local_loss = MSELoss(local_value, target_value.detach())
            local_loss.backward()
            torch.nn.utils.clip_grad_norm(curr_agent.local_critic.parameters(), 0.5)
            curr_agent.local_critic_optimizer.step()

        curr_agent.policy_optimizer.zero_grad()

        if self.discrete_action:
            # Forward pass as if onehot (hard=True) but backprop through a differentiable
            # Gumbel-Softmax sample. The MADDPG paper uses the Gumbel-Softmax trick to backprop
            # through discrete categorical samples, but I'm not sure if that is
            # correct since it removes the assumption of a deterministic policy for
            # DDPG. Regardless, discrete policies don't seem to learn properly without it.
            curr_pol_out = curr_agent.policy(obs[agent_i])
            if avail_actions[agent_i] is not None:
                curr_pol_out = curr_pol_out.masked_fill(avail_actions[agent_i] == 0, -1e9)
            curr_pol_vf_in = gumbel_softmax(curr_pol_out, hard=True)
        else:
            curr_pol_out = curr_agent.policy(obs[agent_i])
            curr_pol_vf_in = curr_pol_out
        if self.alg_types[agent_i] == 'MADDPG':
            all_pol_acs = []
            for i, pi, ob, mask in zip(range(self.nagents), self.policies, obs, avail_actions):
                if i == agent_i:
                    all_pol_acs.append(curr_pol_vf_in)
                elif self.discrete_action:
                    logits = pi(ob)
                    if mask is not None:
                        logits = logits.masked_fill(mask == 0, float('-inf'))
                    all_pol_acs.append(onehot_from_logits(logits))
                else:
                    all_pol_acs.append(pi(ob))
            vf_in = torch.cat((*obs, *all_pol_acs), dim=1) if not is_obs_image else (obs, all_pol_acs)
        else:  # DDPG
            vf_in = torch.cat((obs[agent_i], curr_pol_vf_in),
                              dim=1)
        pol_loss = -curr_agent.critic(vf_in).mean() if not is_obs_image else -curr_agent.critic(*vf_in).mean()
        pol_loss += (curr_pol_out**2).mean() * 1e-3
        pol_loss.backward()
        if parallel:
            average_gradients(curr_agent.policy)
        torch.nn.utils.clip_grad_norm(curr_agent.policy.parameters(), 0.5)
        curr_agent.policy_optimizer.step()
        if logger is not None:
            losses = {'vf_loss': vf_loss,
                      'pol_loss': pol_loss}
            if curr_agent.has_local_q:
                losses['local_vf_loss'] = local_loss
            logger.add_scalars('agent%i/losses' % agent_i,
                               losses,
                               self.niter)

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
            if getattr(a, 'has_local_q', False):
                a.local_critic.train()
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
                if getattr(a, 'has_local_q', False):
                    a.local_critic = fn(a.local_critic)
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
        save_dict = {'init_dict': self.init_dict,
                     'agent_params': [a.get_params() for a in self.agents]}
        torch.save(save_dict, filename)

    @classmethod
    def init_from_env(cls, env, agent_alg="MADDPG", adversary_alg="MADDPG",
                      gamma=0.95, tau=0.01, lr=0.01, hidden_dim=64,
                      local_q=False):
        """
        Instantiate instance of this class from multi-agent environment
        """
        agent_init_params = []
        alg_types = [adversary_alg if atype == 'adversary' else agent_alg for
                     atype in env.agent_types]
        obs_shapes = []
        total_action_dim = 0
        for acsp, obsp, algtype in zip(env.action_space, env.observation_space,
                                       alg_types):
            obs_shapes.append(obsp.shape)
            num_in_pol = obsp.shape[0] if len(obsp.shape) == 1 else obsp.shape
            if isinstance(acsp, Box) or isinstance(acsp, gymnasium.spaces.Box):
                discrete_action = False
                get_shape = lambda x: x.shape[0]
            else:  # Discrete
                discrete_action = True
                get_shape = lambda x: x.n
            num_out_pol = get_shape(acsp)
            total_action_dim += num_out_pol
            if algtype == "MADDPG":
                num_in_critic = 0
                for oobsp in env.observation_space:
                    num_in_critic += oobsp.shape[0]
                for oacsp in env.action_space:
                    num_in_critic += get_shape(oacsp)
            else:
                num_in_critic = obsp.shape[0] + get_shape(acsp)
            agent_init_params.append({'num_in_pol': num_in_pol,
                                      'num_out_pol': num_out_pol,
                                      'num_in_critic': num_in_critic})
        init_dict = {'gamma': gamma, 'tau': tau, 'lr': lr,
                     'hidden_dim': hidden_dim,
                     'alg_types': alg_types,
                     'agent_init_params': agent_init_params,
                     'discrete_action': discrete_action,
                     'obs_shapes': obs_shapes,
                     'total_action_dim': total_action_dim,
                     'local_q': local_q}
        instance = cls(**init_dict)
        instance.init_dict = init_dict
        return instance

    @classmethod
    def init_from_save(cls, filename, test_mode=False):
        """
        Instantiate instance of this class from file created by 'save' method
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        save_dict = torch.load(filename, map_location=device)
        instance = cls(test_mode=test_mode, **save_dict['init_dict'])
        instance.init_dict = save_dict['init_dict']
        for a, params in zip(instance.agents, save_dict['agent_params']):
            a.load_params(params)
        return instance