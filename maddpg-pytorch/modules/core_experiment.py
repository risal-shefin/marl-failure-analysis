"""
Core experiment execution functions.
"""
import numpy as np
import torch
import imageio
import os
from collections import deque
from torch.autograd import Variable
from PIL import Image

from .constants import torch_device, K_SIGMA
from .attacks import fgsm_attack
from .metrics import (
    compute_taylor_delta_policy,
    compute_frob_norms,
    compute_pairwise_frob_norms,
    compute_pairwise_action_influences,
    compute_second_order_action_influences,
    compute_pairwise_observation_influences,
    compute_second_order_observation_influences,
    compute_2nd_ord_dir_derivatives,
    collect_agent_q_values
)


def get_episode_data(env, maddpg, config, logdir, ref_vals, ref_std_devs, detection_method='mean_std', do_attack=False, atk_agent_id=-1, seed=None):
    """
    Collect episode data with optional attack and fault detection.
    
    Args:
        env: Environment instance
        maddpg: MADDPG agent
        config: Configuration object
        logdir: Log directory for outputs
        ref_vals: Reference values for detection
        ref_std_devs: Reference standard deviations for detection
        detection_method: Detection method ('mean_std', 'median_mad', 'diff')
        do_attack: Whether to perform attacks
        atk_agent_id: ID of agent to attack
        seed: Random seed for environment reset
        
    Returns:
        Tuple of collected metrics and data
    """
    obs = env.reset(seed=seed) if seed else env.reset()
    episode_reward = 0
    episode_rewards = [0 for _ in range(maddpg.nagents)]
    frames = []
    
    # Initialize deque buffers for last batch_size observations
    result_deques = [deque(maxlen=5) for _ in range(maddpg.nagents)]
    frob_norms_deques = [deque(maxlen=1) for _ in range(maddpg.nagents)]
    sec_dir_derivatives_deques = [deque(maxlen=1) for _ in range(maddpg.nagents)]
    
    metric_vals = []
    cnt = 0
    attacked_steps = []
    frob_norms_list = []
    sec_dir_derivatives = []
    
    # History tracking
    action_influences_matrix_history = []
    second_order_action_influences_history = []
    observation_influences_matrix_history = []
    second_order_observation_influences_history = []
    q_values_history = []
    frob_norms_matrix_history = []
    
    # Attack control
    do_start_attack = False
    attack_step_remaining = 15

    # Fault detection tracking
    fault_first_detected = {}
    fault_timeline = []
    prev_errors = [0 for i in range(maddpg.nagents)]

    while True:
        # FGSM attack (currently disabled)
        if do_attack and cnt >= config.atk_start_step and cnt <= config.atk_end_step and False:
            temp_torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=False) for i in range(maddpg.nagents)]
            temp_torch_agent_actions = maddpg.step(temp_torch_obs, explore=False)
            agent_actions = [ac.data.cpu().numpy() for ac in temp_torch_agent_actions]
            temp_actions = [agent_actions[i].squeeze() for i, agent_name in enumerate(env.possible_agents)]
            obs[atk_agent_id] = fgsm_attack(maddpg, obs, temp_actions, atk_agent_id, 0.1)
        
        # Get agent actions
        torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=False) for i in range(maddpg.nagents)]
        torch_agent_actions = maddpg.step(torch_obs, explore=False)
        agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions]
        
        if maddpg.discrete_action:
            actions = {agent_name: agent_actions[i].argmax() for i, agent_name in enumerate(env.possible_agents)}
        else:
            actions = {agent_name: agent_actions[i].squeeze() for i, agent_name in enumerate(env.possible_agents)}

        # Random attack (currently disabled)
        if do_attack and False:
            actions[env.possible_agents[atk_agent_id]] = env.action_spaces[env.possible_agents[atk_agent_id]].sample()
        
        # Action Space Attacks
        if maddpg.discrete_action:
            # Compute entropy of action logits
            action_logits = maddpg.get_action_logits(torch_obs)
            atk_agent_action_probs = torch.softmax(action_logits[atk_agent_id].squeeze(), dim=0)
            atk_agent_log_probs = torch.log(atk_agent_action_probs)
            atk_agent_entropy = -torch.sum(atk_agent_action_probs * atk_agent_log_probs)
            
            if do_attack and atk_agent_entropy < 0.5 and cnt >= 5:
                do_start_attack = True
                
            # Worst action attack for discrete action space
            if do_attack and cnt == 6:
                actions[env.possible_agents[atk_agent_id]] = torch.argmin(action_logits[atk_agent_id]).item()
                attacked_steps.append(cnt)
                attack_step_remaining -= 1
        else:
            if do_attack and cnt >= 5:
                do_start_attack = True
            # Attacks for continuous action space would go here
            if do_start_attack and attack_step_remaining > 0:
                attack_step_remaining -= 1

        if config.save_gifs:
            frames.append(Image.fromarray(env.render()))
        
        # Compute all metrics
        results = compute_taylor_delta_policy(maddpg, obs, list(actions.values()), env.action_space, 0.01)
        results_frob_norms = compute_frob_norms(maddpg, obs, list(actions.values()), env.action_space, atk_agent_id)
        
        # Pairwise metrics
        pairwise_frobs = compute_pairwise_frob_norms(maddpg, obs, list(actions.values()), env.action_space)
        frob_norms_matrix_history.append(pairwise_frobs)
        
        pairwise_action_influences = compute_pairwise_action_influences(maddpg, obs, list(actions.values()), env.action_space)
        action_influences_matrix_history.append(pairwise_action_influences)
        
        second_order_action_influences = compute_second_order_action_influences(maddpg, obs, list(actions.values()), env.action_space)
        second_order_action_influences_history.append(second_order_action_influences)
        
        pairwise_observation_influences = compute_pairwise_observation_influences(maddpg, obs, list(actions.values()), env.action_space)
        observation_influences_matrix_history.append(pairwise_observation_influences)
        
        second_order_observation_influences = compute_second_order_observation_influences(maddpg, obs, list(actions.values()), env.action_space)
        second_order_observation_influences_history.append(second_order_observation_influences)
        
        results_sec_dir_derivatives = compute_2nd_ord_dir_derivatives(maddpg, obs, list(actions.values()), env.action_space, atk_agent_id)
        q_values_history.append(collect_agent_q_values(maddpg, obs, list(actions.values()), env.action_space))

        # Process detection for each agent
        for i in range(maddpg.nagents):
            result_deques[i].append(results[i])
            
            # Apply different detection methods
            if detection_method == 'mean_std':
                detection_value = np.mean(result_deques[i])
                threshold_exceeded = abs(detection_value - ref_vals[i][cnt]) > K_SIGMA * ref_std_devs[i][cnt] and not np.isclose(detection_value, ref_vals[i][cnt], rtol=1e-5, atol=1e-5)
            elif detection_method == 'median_mad':
                detection_value = np.mean(result_deques[i])
                threshold_exceeded = abs(detection_value - ref_vals[i][cnt]) > K_SIGMA * ref_std_devs[i][cnt] and not np.isclose(detection_value, ref_vals[i][cnt], rtol=1e-5, atol=1e-5)
            elif detection_method == 'diff':
                if cnt > 0:
                    current_diff = results[i] - prev_errors[i]
                    threshold_exceeded = abs(current_diff - ref_vals[i][cnt]) > K_SIGMA * ref_std_devs[i][cnt] and not np.isclose(current_diff, ref_vals[i][cnt], rtol=1e-5, atol=1e-5)
                    detection_value = current_diff
                else:
                    threshold_exceeded = False
                    detection_value = 0.0
            else:
                raise ValueError(f"Unknown detection method: {detection_method}")
            
            # Handle fault detection
            if threshold_exceeded:
                if i not in fault_first_detected:
                    print(f" [!!!] Anomaly detected for agent {i} at timestep: {cnt}. Method: {detection_method}. Value: {detection_value:.6f}")
                    fault_first_detected[i] = cnt
                    
                    # Cascading Impact Analysis
                    prev_faults = [(f, tf) for f, tf in fault_first_detected.items() if f != i and tf < cnt]
                    contribs = {}
                    if len(prev_faults) > 0:
                        for agent in range(maddpg.nagents):
                            first_fault_ts = prev_faults[0][1]
                            values_over_time = [frob_norms_matrix_history[tau][i][agent] for tau in range(first_fault_ts, cnt + 1) if tau < len(frob_norms_matrix_history)]
                            if len(values_over_time) > 0:
                                contribs[agent] = float(np.mean(values_over_time))
                        if len(contribs) > 0:
                            ranked = sorted(contribs.items(), key=lambda x: x[1], reverse=True)
                            print(f"     >> Potential contributors to fault in agent {i} (mean ||H_{{i,f}}||_F from t_f to {cnt}): {ranked}")
                    
                    fault_timeline.append({
                        'agent': i,
                        't': cnt,
                        'contribs': contribs
                    })
            
            frob_norms_deques[i].append(results_frob_norms[i])
            sec_dir_derivatives_deques[i].append(results_sec_dir_derivatives[i])

        # Store aggregated metrics
        metric_vals.append([np.mean(result_deques[i]) for i in range(maddpg.nagents)])
        prev_errors = results
        frob_norms_list.append([np.mean(frob_norms_deques[i]) for i in range(maddpg.nagents)])
        sec_dir_derivatives.append([np.mean(sec_dir_derivatives_deques[i]) for i in range(maddpg.nagents)])

        # Environment step
        next_obs, rewards, dones, infos = env.step(actions)
        episode_reward += np.sum([rewards[:,i] if env.agent_types[i] != 'adversary' else np.zeros_like(rewards[:,i]) for i in range(maddpg.nagents)])
        episode_rewards = [episode_rewards[i] + rewards[:,i].sum() for i in range(maddpg.nagents)]

        obs = next_obs
        cnt += 1
        if dones.all():
            break

    print(f"Episode reward: {episode_reward}")
    print(f"Episode rewards: {episode_rewards}")
    if config.save_gifs:
        imageio.mimsave(os.path.join(logdir, f'{config.env_id}_episode_atk_{atk_agent_id if do_attack else "free"}.gif'), frames, duration=125)
        print(f"Saved gif of episode to {logdir}")
    print("")
    
    return (metric_vals, attacked_steps, frob_norms_list, sec_dir_derivatives, 
            frob_norms_matrix_history, fault_timeline, action_influences_matrix_history, 
            second_order_action_influences_history, observation_influences_matrix_history, 
            second_order_observation_influences_history, q_values_history)