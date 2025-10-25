"""
SMAC-specific episode execution logic for normal and attacked episodes.
"""
import random
import numpy as np
import torch
from torch.autograd import Variable
from collections import deque

from utils.smac_wrapper import SmacWrapper
from modules.constants import torch_device, K_SIGMA
from modules.metrics import (
    compute_pairwise_action_influences,
    collect_agent_q_values,
    compute_pairwise_action_directional_second_derivatives,
    compute_taylor_delta_policy
)

ATTACK_WINDOW_LENGTH = 5


class SmacEpisodeRunner:
    """
    Handles execution of normal and attacked episodes for SMAC environments.
    """
    
    def __init__(self, maddpg, map_name):
        """
        Initialize SMAC episode runner.
        
        Args:
            maddpg: MADDPG model instance
            map_name: Name of the SMAC map
        """
        self.maddpg = maddpg
        self.map_name = map_name
        self.directional_derivatives_history = []
    
    def run_normal_episode(self, seed, collect_frames=False):
        """
        Run a normal episode without attacks to collect action influences.
        
        Args:
            seed: Random seed for the episode
            collect_frames: Whether to collect RGB frames for visualization
            
        Returns:
            Dictionary containing episode data including action influences
        """
        # Reset directional derivatives history for this episode
        self.directional_derivatives_history = []
        frames = [] if collect_frames else None
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        env = SmacWrapper.make_env(self.map_name, seed=seed)
        obs, action_masks = env.reset()
        action_influences_history = []
        q_values_history = []
        rewards_history = []
        timestep = 0
        episode_reward = 0
        
        # Collect initial frame if requested
        if collect_frames:
            frame = env.render()
            frames.append(frame)
        
        while True:
            torch_obs = [torch.tensor([obs[i]], dtype=torch.float32, requires_grad=False).to(torch_device)
                        for i in range(self.maddpg.nagents)]
            torch_masks = [torch.tensor(action_masks[i], requires_grad=False).to(torch_device)
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
            
            # Collect Q-values
            q_values = collect_agent_q_values(
                self.maddpg, obs, list(actions.values()), env.action_space
            )
            q_values_history.append(q_values)
            
            # Compute pairwise action influences
            pairwise_action_influences = compute_pairwise_action_influences(
                self.maddpg, obs, list(actions.values()), env.action_space
            )
            action_influences_history.append(pairwise_action_influences)
            
            # Compute directional second derivatives for influence analysis
            directional_second_derivatives = compute_pairwise_action_directional_second_derivatives(
                self.maddpg, obs, list(actions.values()), env.action_space
            )
            self.directional_derivatives_history.append(directional_second_derivatives)
            
            # Environment step
            next_obs, rewards, dones, infos, action_masks = env.step(actions)
            
            # Store rewards for each agent at this timestep
            agent_rewards = np.array(rewards).squeeze()
            episode_reward += np.sum(agent_rewards)
            rewards_history.append(agent_rewards)
            
            obs = next_obs
            timestep += 1
            
            # Collect frame after step if requested
            if collect_frames:
                frame = env.render()
                frames.append(frame)
            
            if dones.all():
                break
        env.close()
        
        print("Episode reward (normal):", episode_reward)
        result = {
            'action_influences_history': action_influences_history,
            'directional_derivatives_history': self.directional_derivatives_history,
            'q_values_history': q_values_history,
            'rewards_history': rewards_history,
            'episode_length': timestep
        }
        
        if collect_frames and frames:
            result['frames'] = frames
        
        return result
    
    def run_attacked_episode(self, seed, attack_agent_i, attack_timesteps, 
                            ref_vals, ref_std_devs, observe_agent=None, collect_frames=False):
        """
        Run an episode with attack on specific agent at specific timesteps.
        
        Args:
            seed: Random seed for the episode
            attack_agent_i: Agent to attack (the influencing agent)
            attack_timesteps: List of timesteps to attack
            ref_vals: Reference Taylor error values
            ref_std_devs: Reference standard deviations
            observe_agent: Agent to observe impact on (the influenced agent). 
                         If None, observe attack_agent_i
            collect_frames: Whether to collect RGB frames for visualization
            
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
        
        env = SmacWrapper.make_env(self.map_name, seed=seed)
        obs, action_masks = env.reset()
        episode_reward = 0
        frames = [] if collect_frames else None
        
        # Initialize tracking
        result_deques = [deque(maxlen=5) for _ in range(self.maddpg.nagents)]
        fault_timeline = []
        fault_first_detected = {}
        q_values_history = []
        rewards_history = []
        taylor_errors_history = []
        timestep = 0
        
        # Collect initial frame if requested
        if collect_frames:
            frame = env.render()
            frames.append(frame)
        
        while True:
            noise_scale = 1e-4
            noise = [np.random.normal(loc=0, scale=noise_scale, size=obs[i].shape) 
                    for i in range(self.maddpg.nagents)]
            obs_noisy = [obs[i] + noise[i] for i in range(self.maddpg.nagents)]

            torch_obs = [Variable(torch.tensor([obs_noisy[i]], dtype=torch.float32).to(torch_device), 
                                requires_grad=False) 
                       for i in range(self.maddpg.nagents)]
            torch_masks = [torch.tensor(action_masks[i], requires_grad=False).to(torch_device)
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
            
            # Collect Q-values
            q_values = collect_agent_q_values(
                self.maddpg, obs, list(actions.values()), env.action_space
            )
            q_values_history.append(q_values)
            
            # Compute Taylor delta policy for fault detection
            taylor_results = compute_taylor_delta_policy(
                self.maddpg, obs_noisy, list(actions.values()), env.action_space, 0.01
            )
            
            # Store Taylor errors
            taylor_errors_history.append(taylor_results)
            
            # Update fault detection deques
            for agent_id in range(self.maddpg.nagents):
                result_deques[agent_id].append(taylor_results[agent_id])
            
            # Check for fault detection by threshold
            for agent_id in range(self.maddpg.nagents):
                if timestep < len(ref_vals[agent_id]) and timestep < len(ref_std_devs[agent_id]):
                    detection_value = np.mean(result_deques[agent_id])
                    threshold_exceeded = abs(detection_value - ref_vals[agent_id][timestep]) > K_SIGMA * ref_std_devs[agent_id][timestep] \
                                         and not np.isclose(detection_value, ref_vals[agent_id][timestep], rtol=1e-5, atol=1e-5)
                    
                    if threshold_exceeded:
                        if agent_id not in fault_first_detected:
                            fault_first_detected[agent_id] = timestep
                        fault_timeline.append({
                            'agent': agent_id,
                            't': timestep,
                            'contribs': {},
                            'taylor_deviation': abs(detection_value - ref_vals[agent_id][timestep])
                        })
            
            # Apply attack if current timestep is in any attack window - attack the influencing agent (attack_agent_i)
            attack_active = False
            for attack_t in attack_timesteps:
                if attack_t <= timestep < attack_t + ATTACK_WINDOW_LENGTH:
                    attack_active = True
                    break
            if attack_active:
                # Inject fault
                if self.maddpg.discrete_action:
                    # Worst action attack for discrete action space
                    action_logits = self.maddpg.get_action_logits(torch_obs, torch_masks)
                    
                    # Apply action mask - set invalid actions to very high values so they won't be selected as minimum
                    masked_logits = action_logits[attack_agent_i].clone().squeeze()
                    masked_logits[action_masks[attack_agent_i] == 0] = float('inf')
                    worst_action = torch.argmin(masked_logits).item()
                    actions[env.possible_agents[attack_agent_i]] = worst_action
                else:
                    # For continuous actions, negate the action
                    actions[env.possible_agents[attack_agent_i]] = \
                        -actions[env.possible_agents[attack_agent_i]]
            
            # Environment step
            next_obs, rewards, dones, infos, action_masks = env.step(actions)
            
            # Collect frame if requested
            if collect_frames:
                frame = env.render()
                frames.append(frame)
            
            # Store rewards for each agent
            agent_rewards = np.array(rewards).squeeze()
            rewards_history.append(agent_rewards)
            episode_reward += np.sum(agent_rewards)
            
            obs = next_obs
            timestep += 1
            
            if dones.all():
                break
        env.close()
        
        result = {
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
        
        if collect_frames and frames:
            result['frames'] = frames
        
        return result
