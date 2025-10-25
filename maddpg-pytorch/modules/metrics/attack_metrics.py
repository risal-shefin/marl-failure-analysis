"""
Attack metrics computation for fault analysis.
"""
import numpy as np
from modules.constants import K_SIGMA


WATCH_WINDOW = 15  # Number of timesteps to watch after attack timestep


class AttackMetricsComputer:
    """
    Computes various metrics to evaluate attack effectiveness.
    """
    
    def __init__(self, gamma=0.99):
        """
        Initialize attack metrics computer.
        
        Args:
            gamma: Discount factor for weighted metrics
        """
        self.gamma = gamma
    
    def compute_attack_metrics(self, attack_results, normal_q_values, normal_rewards, 
                              ref_vals, ref_std_devs, observe_agent_j):
        """
        Compute Q-drop, reward-drop and Taylor deviation metrics for attacked episode.
        
        Args:
            attack_results: Results from attacked episode
            normal_q_values: Q values from normal episode
            normal_rewards: Rewards from normal episode
            ref_vals: Reference Taylor error values
            ref_std_devs: Reference standard deviations
            observe_agent_j: Index of agent to observe impact on (influenced agent)
            
        Returns:
            Dictionary containing computed metrics
        """
        attack_timesteps = attack_results['attack_timesteps']
        q_values_history = attack_results['q_values_history']
        rewards_history = attack_results['rewards_history']
        taylor_errors_history = attack_results['taylor_errors_history']
        episode_length = attack_results['episode_length']
        
        # Define watchable window based on all attack timesteps
        # Start from the earliest attack timestep and watch 15 timesteps from the first attack
        min_attack_timestep = min(attack_timesteps)
        max_attack_timestep = max(attack_timesteps)
        window_start = min_attack_timestep
        window_end = min(min_attack_timestep + WATCH_WINDOW, episode_length - 1)
        
        metrics = {
            'max_q_drop': 0.0,
            'weighted_q_drop_sum': 0.0,
            'max_reward_drop': 0.0,
            'weighted_reward_drop_sum': 0.0,
            'max_abs_taylor_deviation': 0.0,
            'weighted_taylor_deviation_sum': 0.0,
            'exceed_rate': 0.0,
            'window_length': window_end - window_start + 1
        }
        
        if window_start >= len(q_values_history) or window_start >= len(taylor_errors_history) \
           or window_start >= len(rewards_history):
            return metrics
        
        if window_start >= len(normal_q_values) or window_start >= len(normal_rewards):
            return metrics
        
        # Compute metrics in watchable window for the observed agent
        exceed_count = 0
        window_steps = 0
        
        for t in range(window_start, window_end + 1):
            if t >= len(q_values_history) or t >= len(normal_q_values) \
               or t >= len(taylor_errors_history) or t >= len(ref_vals[observe_agent_j]) \
               or t >= len(rewards_history) or t >= len(normal_rewards):
                break
            
            # Compute discount weight for this timestep
            weight = self.gamma ** (t - window_start)
            window_steps += 1
            
            # Q-value drop (normal - attacked)
            q_drop = normal_q_values[t][observe_agent_j] - q_values_history[t][observe_agent_j]
            metrics['max_q_drop'] = max(metrics['max_q_drop'], q_drop)
            metrics['weighted_q_drop_sum'] += weight * q_drop
            
            # Reward drop (normal - attacked)
            reward_drop = normal_rewards[t][observe_agent_j] - rewards_history[t][observe_agent_j]
            metrics['max_reward_drop'] = max(metrics['max_reward_drop'], reward_drop)
            metrics['weighted_reward_drop_sum'] += weight * reward_drop
            
            # Taylor deviation from reference mean
            taylor_error = taylor_errors_history[t][observe_agent_j]
            ref_mean = ref_vals[observe_agent_j][t]
            ref_std = ref_std_devs[observe_agent_j][t]
            taylor_deviation = abs(taylor_error - ref_mean)
            
            metrics['max_abs_taylor_deviation'] = max(metrics['max_abs_taylor_deviation'], 
                                                    taylor_deviation)
            
            # Weighted Taylor deviation
            metrics['weighted_taylor_deviation_sum'] += weight * taylor_deviation
            
            # Check if exceeds threshold (mean ± K_SIGMA * std_dev)
            threshold = K_SIGMA * ref_std
            if taylor_deviation > threshold and not np.isclose(taylor_error, ref_mean, 
                                                               rtol=1e-5, atol=1e-5):
                exceed_count += 1
        
        # Compute exceed rate
        if window_steps > 0:
            metrics['exceed_rate'] = exceed_count / window_steps
        
        return metrics
