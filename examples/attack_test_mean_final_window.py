"""Train an algorithm."""
import argparse
from collections import deque
import sys
import os
import yaml
# Add HARL to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from harl.utils.configs_tools import get_defaults_yaml_args, update_args
import numpy as np
import torch
import math
from harl.utils.trans_tools import _t2n 
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datetime import datetime
import csv
from torch.autograd import Variable

def load_taylor_history_csvs(csv_paths):
    """
    Load pre-computed Taylor history from CSV files for all agents.
    
    Args:
        csv_paths: List of paths to CSV files for each agent
        
    Returns:
        dict: {agent_id: {timestep: {'mean': float, 'std_dev': float}}}
    """
    taylor_history_data = {}
    
    for agent_id, csv_path in enumerate(csv_paths):
        agent_data = {}
        with open(csv_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                timestep = int(row['timestep'])
                mean = float(row['mean'])
                std_dev = float(row['std_dev'])
                agent_data[timestep] = {'mean': mean, 'std_dev': std_dev}
        
        taylor_history_data[agent_id] = agent_data
        print(f"Loaded Taylor history for agent {agent_id} with {len(agent_data)} timesteps")
    
    return taylor_history_data

# ---------------- PART 2 WORK -------------------
def compute_pairwise_observation_influences(maddpg, obs, actions, action_spaces):
    """
    Compute direct influence of each agent's observation on every other agent's Q-value.
    Returns an N x N list where entry [i][j] represents || ∂Q_i/∂obs_j ||_2.
    This measures how much agent j's observation directly influences agent i's Q-value.
    """
    # Get device
    torch_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Convert discrete actions to one-hot encoding
    if maddpg.discrete_action:
        one_hot_actions = []
        for i, action in enumerate(actions):
            one_hot = np.zeros(action_spaces[i].n)
            one_hot[action] = 1.0
            one_hot_actions.append(one_hot)
        actions = one_hot_actions

    torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    torch_actions = [Variable(torch.Tensor([actions[i]]).to(torch_device), requires_grad=False) for i in range(maddpg.nagents)]
    vf_in = torch.cat((*torch_obs, *torch_actions), dim=1)

    N = maddpg.nagents
    results = [[0.0 for _ in range(N)] for _ in range(N)]

    for i in range(N):
        critic_val = maddpg.agents[i].critic(vf_in).mean()
        
        for j in range(N):
            # Compute gradient of Q_i with respect to observation of agent j
            grad_qi_obsj = torch.autograd.grad(
                critic_val,
                torch_obs[j],
                retain_graph=True,
                allow_unused=True
            )[0]

            # Compute L2 norm of the gradient (direct influence magnitude)
            influence_magnitude = grad_qi_obsj.norm(p=2).item()
            results[i][j] = influence_magnitude

    return results

def compute_pairwise_observation_influences_happo(runner, obs, rnn_states_critic, masks):
    """
    HAPPO adaptation: Compute direct influence of each agent's observation on every other agent's value function.
    Returns an N x N list where entry [i][j] represents || ∂V_i/∂obs_j ||_2.
    This measures how much agent j's observation directly influences agent i's value estimation.
    
    Args:
        runner: HARL runner object containing the critic
        obs: observations for all agents, shape (n_threads, n_agents, obs_dim)
        rnn_states_critic: RNN states for critic
        masks: masks for RNN reset
        
    Returns:
        N x N matrix where entry [i][j] is the influence magnitude of agent j's obs on agent i's value
    """
    obs = torch.tensor(obs, dtype=torch.float32)
    
    # Create separate tensors for each agent's observations with gradient tracking
    agent_obs_tensors = []
    n_agents = runner.num_agents
    
    # Extract individual agent observations and make them require gradients
    for i in range(n_agents):
        agent_obs = obs[0][i].clone().detach()
        agent_obs_tensor = torch.tensor(agent_obs, dtype=torch.float32, requires_grad=True)
        agent_obs_tensors.append(agent_obs_tensor)
    
    # Reconstruct shared observations (concatenated) for critic input
    concatenated_obs = torch.cat(agent_obs_tensors, dim=0)
    share_obs = concatenated_obs.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, total_obs_dim)
    share_obs = share_obs.expand(1, n_agents, -1)  # Shape: (1, n_agents, total_obs_dim)
    
    # Get value estimates from centralized critic
    values, _ = runner.critic.get_values(
        share_obs,
        rnn_states_critic,
        masks,
    )
    values = values.squeeze()  # Shape: (n_agents,)
    # print(f"Values: {values}")
    N = n_agents
    results = [[0.0 for _ in range(N)] for _ in range(N)]
    
    # For each agent i's value function
    for i in range(N):
        # For each agent j's observation
        for j in range(N):
            # try:
            # Compute gradient of V_i with respect to observation of agent j
            grad_vi_obsj = torch.autograd.grad(
                values[i],
                agent_obs_tensors[j],
                retain_graph=True,
                allow_unused=True
            )[0]
            
            if grad_vi_obsj is not None:
                # Compute L2 norm of the gradient (direct influence magnitude)
                influence_magnitude = grad_vi_obsj.norm(p=2).item()
                results[i][j] = influence_magnitude
            else:
                results[i][j] = 0.0
                
            
    return results
def compute_second_order_observation_influences(maddpg, obs, actions, action_spaces):
    """
    Compute second-order observation influences between agents.
    Returns an N x N list where entry [i][j] represents || ∂²Q_i/(∂obs_j)² ||_F.
    This measures the curvature/sensitivity of agent i's Q-value with respect to agent j's observation.
    """
    # Get device
    torch_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Convert discrete actions to one-hot encoding
    if maddpg.discrete_action:
        one_hot_actions = []
        for i, action in enumerate(actions):
            one_hot = np.zeros(action_spaces[i].n)
            one_hot[action] = 1.0
            one_hot_actions.append(one_hot)
        actions = one_hot_actions

    torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    torch_actions = [Variable(torch.Tensor([actions[i]]).to(torch_device), requires_grad=False) for i in range(maddpg.nagents)]
    vf_in = torch.cat((*torch_obs, *torch_actions), dim=1)

    N = maddpg.nagents
    results = [[0.0 for _ in range(N)] for _ in range(N)]

    for i in range(N):
        critic_val = maddpg.agents[i].critic(vf_in).mean()
        
        for j in range(N):
            # Compute first-order gradient ∂Q_i/∂obs_j
            grad_qi_obsj = torch.autograd.grad(
                critic_val,
                torch_obs[j],
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]
            
            if grad_qi_obsj is None:
                continue
                
            # Compute second-order gradient ∂²Q_i/(∂obs_j)²
            hessian_matrix = []
            for k in range(grad_qi_obsj.shape[1]):  # iterate over observation dimensions
                second_grad = torch.autograd.grad(
                    grad_qi_obsj[0, k],
                    torch_obs[j],  # same observation variable for pure second derivative
                    retain_graph=True,
                    allow_unused=True
                )[0]
                
                if second_grad is None:
                    second_grad = torch.zeros_like(grad_qi_obsj[0])
                hessian_matrix.append(second_grad.flatten())
            
            if len(hessian_matrix) > 0:
                H = torch.stack(hessian_matrix)
                # Compute Frobenius norm of the Hessian matrix
                second_order_influence = H.norm(p='fro').item()
                results[i][j] = second_order_influence

    return results

def compute_second_order_observation_influences_happo(runner, obs, rnn_states_critic, masks):
    """
    HAPPO adaptation: Compute second-order observation influences between agents for value functions.
    Returns an N x N list where entry [i][j] represents || ∂²V_i/(∂obs_j)² ||_F.
    This measures the curvature/sensitivity of agent i's value function with respect to agent j's observation.
    
    Args:
        runner: HARL runner object containing the critic
        obs: observations for all agents, shape (n_threads, n_agents, obs_dim) 
        rnn_states_critic: RNN states for critic
        masks: masks for RNN reset
        
    Returns:
        N x N matrix where entry [i][j] is the second-order influence magnitude of agent j's obs on agent i's value
    """
    obs = torch.tensor(obs, dtype=torch.float32)
    
    # Create separate tensors for each agent's observations with gradient tracking
    agent_obs_tensors = []
    n_agents = runner.num_agents
    
    # Extract individual agent observations and make them require gradients
    for i in range(n_agents):
        agent_obs = obs[0][i].clone().detach()
        agent_obs_tensor = torch.tensor(agent_obs, dtype=torch.float32, requires_grad=True)
        agent_obs_tensors.append(agent_obs_tensor)
    
    # Reconstruct shared observations (concatenated) for critic input
    concatenated_obs = torch.cat(agent_obs_tensors, dim=0)
    share_obs = concatenated_obs.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, total_obs_dim)
    share_obs = share_obs.expand(1, n_agents, -1)  # Shape: (1, n_agents, total_obs_dim)
    
    # Get value estimates from centralized critic
    values, _ = runner.critic.get_values(
        share_obs,
        rnn_states_critic,
        masks,
    )
    values = values.squeeze()  # Shape: (n_agents,)
    
    N = n_agents
    results = [[0.0 for _ in range(N)] for _ in range(N)]
    
    # For each agent i's value function
    for i in range(N):
        # For each agent j's observation
        for j in range(N):
            try:
                # Compute first-order gradient ∂V_i/∂obs_j
                grad_vi_obsj = torch.autograd.grad(
                    values[i],
                    agent_obs_tensors[j],
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True
                )[0]
                
                if grad_vi_obsj is None:
                    results[i][j] = 0.0
                    continue
                
                # Compute second-order gradient ∂²V_i/(∂obs_j)²
                hessian_matrix = []
                for k in range(grad_vi_obsj.shape[0]):  # iterate over observation dimensions
                    second_grad = torch.autograd.grad(
                        grad_vi_obsj[k],
                        agent_obs_tensors[j],  # same observation variable for pure second derivative
                        retain_graph=True,
                        allow_unused=True
                    )[0]
                    
                    if second_grad is None:
                        second_grad = torch.zeros_like(grad_vi_obsj)
                    hessian_matrix.append(second_grad.flatten())
                
                if len(hessian_matrix) > 0:
                    H = torch.stack(hessian_matrix)
                    # Compute Frobenius norm of the Hessian matrix
                    second_order_influence = H.norm(p='fro').item()
                    results[i][j] = second_order_influence
                else:
                    results[i][j] = 0.0
                    
            except Exception:
                # Handle cases where gradient computation fails
                results[i][j] = 0.0
    
    return results


def detect_threats_faulty_in_topk(pairwise_frobs_history, fault_timeline, top_k=2, window_size=3):
    """
    Detect threats by checking if faulty agents are among top-k influencers of non-faulty agents.
    Similar to the action influence logic but using Frobenius norms.
    
    Args:
        pairwise_frobs_history: List of N x N matrices, one per timestep
        fault_timeline: List of fault events with 'agent' and 't' keys
        top_k: Number of top agents to consider (default 2)
        window_size: Number of timesteps before fault to analyze
        
    Returns:
        threat_data: Dictionary with extended threat detection results
    """
    if not pairwise_frobs_history or not fault_timeline:
        return {
            'timestamp': [],
            'target_agent': [],
            'faulty_influencers': [],
            'faulty_agent_ranks': [],
            'is_threat': [],
            'top_k_agents': [],
            'top_k_values': [],
            'threat_timesteps': set(),
            'event_type': [],
            'fault_detection_times': {},
            'analysis_windows': []
        }
    
    n_agents = len(pairwise_frobs_history[0])
    n_timesteps = len(pairwise_frobs_history)
    
    threat_data = {
        'timestamp': [],
        'target_agent': [],
        'faulty_influencers': [],
        'faulty_agent_ranks': [],
        'is_threat': [],
        'top_k_agents': [],
        'top_k_values': [],
        'threat_timesteps': set(),
        'event_type': [],
        'fault_detection_times': {},
        'analysis_windows': []
    }
    
    # Create fault detection timeline mapping
    fault_detection_times = {}
    for event in fault_timeline:
        if event['agent'] not in fault_detection_times:
            fault_detection_times[event['agent']] = event['t']
    
    threat_data['fault_detection_times'] = fault_detection_times
    
    if not fault_detection_times:
        print(f"No faults detected. No threat analysis performed.")
        return threat_data
    
    # Find the last fault detection timestep to limit analysis
    last_fault_detection_time = max(fault_detection_times.values())
    
    # Track flagged pairs to avoid duplicates
    flagged_pairs = set()  # Set of (faulty_agent_id, target_agent_id) tuples
    analysis_windows_info = []
    
    # Add original fault detection events
    for event in fault_timeline:
        faulty_agent = event['agent']
        fault_timestep = event['t']
        
        # Get Frobenius norms at the exact fault timestep
        if fault_timestep < len(pairwise_frobs_history):
            frob_matrix = pairwise_frobs_history[fault_timestep]
            
            # Get influences on the faulty agent (excluding self-influence)
            agent_influences = []
            for j in range(n_agents):
                if j != faulty_agent:  # Exclude self-influence
                    agent_influences.append((j, frob_matrix[faulty_agent][j]))
            
            # Sort by Frobenius values (highest first)
            agent_influences.sort(key=lambda x: x[1], reverse=True)
            top_k_agents = [agent_id for agent_id, _ in agent_influences[:top_k]]
            top_k_values = [value for _, value in agent_influences[:top_k]]
        else:
            top_k_agents = []
            top_k_values = []
        
        # Store fault detection event
        threat_data['timestamp'].append(fault_timestep)
        threat_data['target_agent'].append(faulty_agent)
        threat_data['faulty_influencers'].append([])  # No faulty influencers for original fault
        threat_data['faulty_agent_ranks'].append([])
        threat_data['is_threat'].append(False)  # Original fault is not a "threat" in this context
        threat_data['top_k_agents'].append(top_k_agents.copy())
        threat_data['top_k_values'].append(top_k_values.copy())
        threat_data['event_type'].append('fault_detection')
        
        analysis_windows_info.append({
            'event_type': 'fault_detection',
            'fault_agent': faulty_agent,
            'fault_time': fault_timestep,
            'window_start': fault_timestep,
            'window_end': fault_timestep,
            'window_timesteps': [fault_timestep]
        })
    
    # Find timesteps where faulty agents are top-k influencers on non-faulty agents
    # Only check timesteps up to (but not including) the last fault detection
    for t in range(min(len(pairwise_frobs_history), last_fault_detection_time)):
        frob_matrix = pairwise_frobs_history[t]
        
        # Get the set of agents that are considered faulty at timestep t
        faulty_agents_at_t = set()
        for agent_id, detection_time in fault_detection_times.items():
            if t >= detection_time:  # Only consider agent faulty from detection time onwards
                faulty_agents_at_t.add(agent_id)
        
        # Skip if no agents are faulty at this timestep
        if not faulty_agents_at_t:
            continue
        
        # For each non-faulty agent at timestep t, check if any faulty agent is in top-k influencers
        for non_faulty_agent in range(n_agents):
            # Check if this agent is faulty at timestep t
            is_faulty_at_t = non_faulty_agent in faulty_agents_at_t
            if is_faulty_at_t:
                continue  # Skip agents that are faulty at this timestep
            
            # Get Frobenius influences on this non-faulty agent and rank them (excluding self-influence)
            agent_influences = []
            for j in range(n_agents):
                if j != non_faulty_agent:  # Exclude self-influence
                    agent_influences.append((j, frob_matrix[non_faulty_agent][j]))
            
            # Sort by influence magnitude (descending)
            agent_influences.sort(key=lambda x: x[1], reverse=True)
            
            # Check if any faulty agent is in top-k
            top_k_agents = [agent_id for agent_id, _ in agent_influences[:top_k]]
            top_k_values = [value for _, value in agent_influences[:top_k]]
            faulty_in_top_k = [agent_id for agent_id in top_k_agents if agent_id in faulty_agents_at_t]
            
            if faulty_in_top_k:
                # Check if any of the faulty agents in top-k have already been flagged for this target
                new_faulty_influencers = []
                faulty_ranks = []
                
                for faulty_agent in faulty_in_top_k:
                    pair = (faulty_agent, non_faulty_agent)
                    if pair not in flagged_pairs:
                        new_faulty_influencers.append(faulty_agent)
                        flagged_pairs.add(pair)
                        
                        # Find rank of this faulty agent
                        for rank, (agent_id, _) in enumerate(agent_influences):
                            if agent_id == faulty_agent:
                                faulty_ranks.append(rank + 1)  # 1-indexed
                                break
                
                # Only create an event if there are new faulty influencers to report
                if new_faulty_influencers:
                    threat_data['timestamp'].append(t)
                    threat_data['target_agent'].append(non_faulty_agent)
                    threat_data['faulty_influencers'].append(new_faulty_influencers.copy())
                    threat_data['faulty_agent_ranks'].append(faulty_ranks.copy())
                    threat_data['is_threat'].append(True)
                    threat_data['top_k_agents'].append(top_k_agents.copy())
                    threat_data['top_k_values'].append(top_k_values.copy())
                    threat_data['threat_timesteps'].add(t)
                    threat_data['event_type'].append('top_k_influence')
                    
                    faulty_list = ', '.join(map(str, new_faulty_influencers))
                    analysis_windows_info.append({
                        'event_type': 'top_k_influence',
                        'faulty_influencers': new_faulty_influencers,
                        'target_agent': non_faulty_agent,
                        'fault_time': t,
                        'window_start': t,
                        'window_end': t,
                        'window_timesteps': [t],
                        'description': f"Faulty agent(s) {faulty_list} among top-{top_k} influencers of Agent {non_faulty_agent}"
                    })
    
    threat_data['analysis_windows'] = analysis_windows_info
    
    total_events = len(threat_data['timestamp'])
    fault_events = len([e for e in threat_data['event_type'] if e == 'fault_detection'])
    influence_events = len([e for e in threat_data['event_type'] if e == 'top_k_influence'])
    
    print(f"Faulty agents in top-{top_k} analysis completed:")
    print(f"  Total events: {total_events}")
    print(f"  Fault detection events: {fault_events}")
    print(f"  Top-{top_k} influence events: {influence_events}")
    print(f"  Unique threat timesteps: {len(threat_data['threat_timesteps'])}")
    
    return threat_data

############# THREAT DETECTION FUNCTIONS #############

def detect_threats_top_k(pairwise_frobs_history, attacked_agent_id, top_k=2, fault_timeline=None, window_size=3):
    """
    Detect threats by analyzing if the attacked agent has the HIGHEST Frobenius value (excluding self-influence)
    in the timesteps BEFORE fault events of non-attacked agents.
    Only analyzes timesteps within window_size before fault events of non-attacked agents.
    
    Args:
        pairwise_frobs_history: List of N x N matrices, one per timestep
        attacked_agent_id: ID of the attacked agent
        top_k: Number of top agents to consider
        fault_timeline: List of fault events with 'agent' and 't' keys
        window_size: Number of timesteps before fault to analyze
        
    Returns:
        threat_data: Dictionary with threat detection results
    """
    if not pairwise_frobs_history or not fault_timeline:
        return {
            'timestamp': [],
            'agent_id': [],
            'attacked_agent_rank': [],
            'is_threat': [],
            'top_k_agents': [],
            'top_k_values': [],
            'threat_timesteps': set(),
            'analysis_windows': []
        }
    
    n_agents = len(pairwise_frobs_history[0])
    n_timesteps = len(pairwise_frobs_history)
    
    threat_data = {
        'timestamp': [],
        'agent_id': [],
        'attacked_agent_rank': [],
        'is_threat': [],
        'top_k_agents': [],
        'top_k_values': [],
        'threat_timesteps': set(),
        'analysis_windows': []
    }
    
    # Find fault events for all agents (including attacked agent)
    # We want to analyze timesteps before ANY agent becomes faulty
    all_faults = fault_timeline
    
    if not all_faults:
        print(f"No faults detected. No threat analysis performed.")
        return threat_data
    
    # Create analysis windows around each fault
    analysis_timesteps = set()
    analysis_windows_info = []
    
    for fault_event in all_faults:
        fault_time = fault_event['t']
        fault_agent = fault_event['agent']
        
        # Define window BEFORE and INCLUDING fault timestep
        window_start = max(0, fault_time - window_size)
        window_end = fault_time  # Include the fault timestep itself
        
        # Only add timesteps if there are valid timesteps before the fault
        if window_end >= window_start:
            window_timesteps = list(range(window_start, window_end + 1))
            analysis_timesteps.update(window_timesteps)
            
            analysis_windows_info.append({
                'fault_agent': fault_agent,
                'fault_time': fault_time,
                'window_start': window_start,
                'window_end': window_end,
            'window_timesteps': window_timesteps
        })
    
    threat_data['analysis_windows'] = analysis_windows_info
    print(f"Analyzing {len(analysis_timesteps)} timesteps BEFORE/INCLUDING {len(all_faults)} fault events")
    
    # Analyze only the timesteps in analysis windows
    for t in sorted(analysis_timesteps):
        if t >= n_timesteps:
            continue
            
        frob_matrix = pairwise_frobs_history[t]
        
        # For each agent i (check if attacked agent influences agent i)
        for i in range(n_agents):
            # Get Frobenius values for how all agents influence agent i, EXCLUDING self-influence
            agent_i_influences = []
            for j in range(n_agents):
                if i != j:  # Exclude self-influence (diagonal elements)
                    agent_i_influences.append((j, frob_matrix[i][j]))
            
            # Sort by Frobenius values in descending order (highest first)
            agent_i_influences.sort(key=lambda x: x[1], reverse=True)
            
            # Get top-k agents and their values
            top_k_agents = [agent_id for agent_id, _ in agent_i_influences[:top_k]]
            top_k_values = [value for _, value in agent_i_influences[:top_k]]
            
            # Check if attacked agent is #1 (highest value)
            is_threat = False
            attacked_agent_rank = -1
            
            if len(agent_i_influences) > 0:
                # Check if attacked agent has the highest Frobenius value
                highest_agent = agent_i_influences[0][0]  # Agent with highest value
                if highest_agent == attacked_agent_id:
                    is_threat = True
                    attacked_agent_rank = 1
                    threat_data['threat_timesteps'].add(t)
                else:
                    # Find rank of attacked agent if it's in the list
                    for rank, (agent_id, _) in enumerate(agent_i_influences):
                        if agent_id == attacked_agent_id:
                            attacked_agent_rank = rank + 1  # 1-indexed
                            break
            
            # Store results
            threat_data['timestamp'].append(t)
            threat_data['agent_id'].append(i)
            threat_data['attacked_agent_rank'].append(attacked_agent_rank)
            threat_data['is_threat'].append(is_threat)
            threat_data['top_k_agents'].append(top_k_agents.copy())
            threat_data['top_k_values'].append(top_k_values.copy())
    
    return threat_data

def save_threat_data_csv_faulty_topk(threat_data, top_k, logdir, filename="faulty_agents_topk_threats.csv"):
    """
    Save faulty agents in top-k threat detection results to CSV file.
    
    Args:
        threat_data: Dictionary with threat detection results from detect_threats_faulty_in_topk
        top_k: Number of top agents considered
        logdir: Directory to save the file
        filename: Name of the CSV file
    """
    filepath = os.path.join(logdir, filename)
    
    # Create header
    header = [
        "timestamp", 
        "target_agent", 
        "event_type",
        "faulty_influencers",
        "faulty_agent_ranks", 
        "is_threat", 
        f"top_{top_k}_agents",
        f"top_{top_k}_values"
    ]
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        # Write data
        for i in range(len(threat_data['timestamp'])):
            row = [
                threat_data['timestamp'][i],
                threat_data['target_agent'][i],
                threat_data['event_type'][i],
                f"[{', '.join(map(str, threat_data['faulty_influencers'][i]))}]",
                f"[{', '.join(map(str, threat_data['faulty_agent_ranks'][i]))}]",
                1 if threat_data['is_threat'][i] else 0,
                f"[{', '.join(map(str, threat_data['top_k_agents'][i]))}]",
                f"[{', '.join([f'{v:.4f}' for v in threat_data['top_k_values'][i]])}]"
            ]
            writer.writerow(row)
    
    # Print summary
    total_entries = len(threat_data['timestamp'])
    threat_entries = sum(threat_data['is_threat'])
    threat_timesteps = len(threat_data['threat_timesteps'])
    fault_events = len([e for e in threat_data['event_type'] if e == 'fault_detection'])
    influence_events = len([e for e in threat_data['event_type'] if e == 'top_k_influence'])
    
    print(f"Saved faulty agents top-{top_k} threat detection results to {filepath}")
    print(f"Total entries: {total_entries}")
    print(f"  - Fault detection events: {fault_events}")
    print(f"  - Top-{top_k} influence events (threats): {influence_events}")
    print(f"Unique threat timesteps: {threat_timesteps}")
    
    # Print fault detection timeline
    if 'fault_detection_times' in threat_data and threat_data['fault_detection_times']:
        print(f"\n=== FAULT DETECTION TIMELINE ===")
        for agent_id, detection_time in threat_data['fault_detection_times'].items():
            print(f"  Agent {agent_id}: First fault detected at t={detection_time}")
    
    # Print some example influence events
    influence_examples = [(i, threat_data['timestamp'][i], threat_data['target_agent'][i], 
                          threat_data['faulty_influencers'][i], threat_data['faulty_agent_ranks'][i]) 
                         for i in range(len(threat_data['timestamp'])) 
                         if threat_data['event_type'][i] == 'top_k_influence']
    
    if influence_examples:
        print(f"\n=== SAMPLE TOP-{top_k} INFLUENCE EVENTS ===")
        for i, (idx, t, target, faulty_agents, ranks) in enumerate(influence_examples[:5]):
            faulty_info = ', '.join([f"Agent {fa} (rank #{rank})" for fa, rank in zip(faulty_agents, ranks)])
            print(f"  t={t}: {faulty_info} → Agent {target}")

def save_threat_data_csv(threat_data, attacked_agent_id, top_k, logdir, window_str):
    """
    Save threat detection results to CSV file.
    
    Args:
        threat_data: Dictionary with threat detection results
        attacked_agent_id: ID of the attacked agent
        top_k: Number of top agents considered
        logdir: Directory to save the file
        window_str: String describing the attack window
    """
    filepath = os.path.join(logdir, "raise_flag.csv")
    
    # Create header
    header = [
        "timestamp", 
        "agent_id", 
        "attacked_agent_id",
        "attacked_agent_rank", 
        "is_threat", 
        f"top_{top_k}_agents",
        f"top_{top_k}_values"
    ]
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        # Write data
        for i in range(len(threat_data['timestamp'])):
            row = [
                threat_data['timestamp'][i],
                threat_data['agent_id'][i],
                attacked_agent_id,
                threat_data['attacked_agent_rank'][i] if threat_data['is_threat'][i] else 'N/A',
                1 if threat_data['is_threat'][i] else 0,
                f"[{', '.join(map(str, threat_data['top_k_agents'][i]))}]",
                f"[{', '.join([f'{v:.4f}' for v in threat_data['top_k_values'][i]])}]"
            ]
            writer.writerow(row)
    
    # Print summary
    total_entries = len(threat_data['timestamp'])
    threat_entries = sum(threat_data['is_threat'])
    threat_timesteps = len(threat_data['threat_timesteps'])
    
    print(f"Saved threat detection results to {filepath}")
    print(f"Total entries: {total_entries}")
    print(f"Threat entries (attacked agent ranked #1): {threat_entries}")
    print(f"Unique threat timesteps: {threat_timesteps}")
    
    # Print analysis window information
    if 'analysis_windows' in threat_data and threat_data['analysis_windows']:
        print(f"\n=== FAULT-BASED ANALYSIS WINDOWS ===")
        for window_info in threat_data['analysis_windows']:
            print(f"  Fault: Agent {window_info['fault_agent']} at t={window_info['fault_time']}")
            print(f"    Analysis window: t={window_info['window_start']} to t={window_info['window_end']}")
    
    print(f"Note: Only analyzes timesteps around fault events of non-attacked agents (±3 timestep window)")

def plot_faulty_agents_topk_timeline(threat_data, pairwise_frobs_history, total_agents, top_k, logdir):
    """
    Create visualization showing fault detection timeline and faulty agents in top-k influence events.
    Similar to plot_fault_timeline_action_influences but using Frobenius norms.
    
    Args:
        threat_data: Dictionary with threat detection results from detect_threats_faulty_in_topk
        pairwise_frobs_history: List of N x N Frobenius matrices per timestep
        total_agents: Total number of agents
        top_k: Number of top agents considered
        logdir: Directory to save the plot
    """
    if not threat_data['timestamp']:
        print("No events to display in faulty agents top-k timeline.")
        return
    
    # Separate events by type
    fault_events = [(i, threat_data['timestamp'][i]) for i in range(len(threat_data['timestamp'])) 
                   if threat_data['event_type'][i] == 'fault_detection']
    influence_events = [(i, threat_data['timestamp'][i]) for i in range(len(threat_data['timestamp'])) 
                       if threat_data['event_type'][i] == 'top_k_influence']
    
    # Create combined timeline sorted by timestamp
    all_events = [(i, threat_data['timestamp'][i], threat_data['event_type'][i]) 
                 for i in range(len(threat_data['timestamp']))]
    all_events.sort(key=lambda x: x[1])  # Sort by timestamp
    
    if len(all_events) == 0:
        print("No events to display.")
        return
    
    k = len(all_events)
    fig = plt.figure(figsize=(max(8, 3*k), 6))
    gs = fig.add_gridspec(
        3, k,
        height_ratios=[1.0, 2, 0.1],
        hspace=0.15
    )

    cmap = plt.get_cmap('tab20')
    agent_colors = {i: cmap(i % 20) for i in range(total_agents)}

    # --- Timeline axis (top row) ---
    ax_timeline = fig.add_subplot(gs[0, :])
    ax_timeline.axis('off')

    arrow_y = 0.5
    ax_timeline.annotate(
        '', xy=(1, arrow_y), xytext=(0, arrow_y),
        arrowprops=dict(arrowstyle='-|>', lw=2, color='gray'),
        xycoords='axes fraction', textcoords='axes fraction'
    )

    # Milestones
    for i, (event_idx, timestamp, event_type) in enumerate(all_events):
        frac_x = (i + 0.5) / k

        # Different markers for different event types
        if event_type == 'fault_detection':
            marker_color = 'darkred'
            marker_size = 12
            target_agent = threat_data['target_agent'][event_idx]
            description = f"Fault: Agent {target_agent}"
        else:  # top_k_influence
            marker_color = 'orange'
            marker_size = 10
            target_agent = threat_data['target_agent'][event_idx]
            faulty_agents = threat_data['faulty_influencers'][event_idx]
            faulty_list = ', '.join(map(str, faulty_agents))
            description = f"Faulty {faulty_list} → Agent {target_agent}"

        # Circle marker
        ax_timeline.plot(frac_x, arrow_y, 'o', color=marker_color, markersize=marker_size, 
                        transform=ax_timeline.transAxes)

        # Event description above (with line wrapping for long descriptions)
        if len(description) > 20:  # Wrap long descriptions
            words = description.split()
            lines = []
            current_line = []
            for word in words:
                if len(' '.join(current_line + [word])) <= 20:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                    else:
                        lines.append(word)
            if current_line:
                lines.append(' '.join(current_line))
            description = '\n'.join(lines)

        ax_timeline.text(frac_x, arrow_y + 0.15,
                         description,
                         ha='center', va='bottom',
                         fontsize=9, fontweight='bold',
                         transform=ax_timeline.transAxes)

        # Timestep label below
        ax_timeline.text(frac_x, arrow_y - 0.15,
                         f"t = {timestamp}",
                         ha='center', va='top',
                         fontsize=10, color='darkblue',
                         transform=ax_timeline.transAxes)

    # --- Influence charts (middle row) ---
    for col, (event_idx, timestamp, event_type) in enumerate(all_events):
        ax = fig.add_subplot(gs[1, col])
        
        target_agent = threat_data['target_agent'][event_idx]
        top_k_agents = threat_data['top_k_agents'][event_idx]
        top_k_values = threat_data['top_k_values'][event_idx]

        if len(top_k_values) == 0:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No Data',
                    ha='center', va='center', fontsize=10, style='italic')
        else:
            # Normalize values for pie chart
            vals = np.array(top_k_values, dtype=float)
            if vals.sum() > 0:
                vals_normalized = vals / vals.sum()
            else:
                vals_normalized = vals
            
            colors = [agent_colors[a] for a in top_k_agents]
            
            # Highlight faulty agents in different color if this is an influence event
            if event_type == 'top_k_influence':
                faulty_agents_set = set(threat_data['faulty_influencers'][event_idx])
                for i, agent_id in enumerate(top_k_agents):
                    if agent_id in faulty_agents_set:
                        colors[i] = 'red'  # Highlight faulty agents in red

            wedges, _, autotexts = ax.pie(
                vals_normalized, autopct='%1.1f%%', startangle=90, colors=colors,
                wedgeprops=dict(width=0.35, edgecolor='w')
            )
            for at in autotexts:
                at.set_fontsize(8)
                at.set_fontweight('bold')
            
            title = f"Top-{top_k} Influences on Agent {target_agent}"
            ax.set_title(title, fontsize=10, pad=5)
            ax.set_aspect('equal')

    # --- Legend row (bottom row) ---
    ax_legend = fig.add_subplot(gs[2, :])
    ax_legend.axis('off')
    legend_elements = [Patch(facecolor=agent_colors[i], label=f"Agent {i}") for i in range(total_agents)]
    
    # Add legend for event types and faulty agents
    fault_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkred', 
                             markersize=10, label='Fault Detection')
    influence_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                                 markersize=10, label=f'Top-{top_k} Influence Event')
    faulty_patch = Patch(facecolor='red', label='Faulty Agent')
    legend_elements.extend([fault_marker, influence_marker, faulty_patch])
    
    ax_legend.legend(handles=legend_elements, loc='center', ncol=min(len(legend_elements), 8),
                     fontsize=9, frameon=False)

    fig.suptitle(f'Faulty Agents in Top-{top_k} Influencers Timeline (Frobenius Norm Analysis)',
                 fontsize=14, fontweight='bold', y=0.96)

    out_path = os.path.join(logdir, f'faulty_agents_topk_timeline_k{top_k}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved faulty agents top-{top_k} timeline plot to {out_path}")
    print(f"Timeline includes {len(fault_events)} fault detections and {len(influence_events)} top-{top_k} influence events")

def plot_threat_analysis_combined(fault_timeline, threat_data, pairwise_frobs_history, attacked_agent_id, total_agents, top_k, logdir):
    """
    Create combined visualization with fault timeline (top) and threat bar charts (bottom).
    Bottom row shows bar plots for only the timesteps where threats were detected.
    
    Args:
        fault_timeline: List of fault events
        threat_data: Dictionary with threat detection results
        pairwise_frobs_history: List of N x N Frobenius matrices per timestep
        attacked_agent_id: ID of the attacked agent
        total_agents: Total number of agents
        top_k: Number of top agents considered
        logdir: Directory to save the plot
    """
    if not pairwise_frobs_history:
        print("No Frobenius history data available for threat analysis plot.")
        return
    
    n_timesteps = len(pairwise_frobs_history)
    threat_timesteps = sorted(list(threat_data['threat_timesteps']))
    
    # Extract all analysis window timesteps from threat_data
    all_analysis_timesteps = set()
    if 'analysis_windows' in threat_data and threat_data['analysis_windows']:
        for window_info in threat_data['analysis_windows']:
            if 'window_timesteps' in window_info:
                all_analysis_timesteps.update(window_info['window_timesteps'])
    
    # If no analysis windows, fall back to threat timesteps
    if not all_analysis_timesteps:
        all_analysis_timesteps = set(threat_timesteps)
    
    analysis_timesteps = sorted(list(all_analysis_timesteps))
    
    if len(analysis_timesteps) == 0:
        print("No threats detected. Creating plot with fault timeline only.")
        # Create simple fault timeline plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        if len(fault_timeline) > 0:
            for i, event in enumerate(fault_timeline):
                fault_time = event['t']
                fault_agent = event['agent']
                ax.scatter(fault_time, fault_agent, color='red', s=100, marker='x', linewidth=3)
                ax.annotate(f'Fault t={fault_time}', 
                          xy=(fault_time, fault_agent),
                          xytext=(10, 10), textcoords='offset points',
                          fontsize=10, fontweight='bold', color='red')
        
        ax.set_xlim(-1, n_timesteps)
        ax.set_ylim(-0.5, total_agents - 0.5)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Agent ID')
        ax.set_title('Fault Detection Timeline - No Threats Detected')
        ax.grid(True, alpha=0.3)
        ax.set_yticks(range(total_agents))
        ax.set_yticklabels([f'Agent {i}' for i in range(total_agents)])
        
        plt.tight_layout()
        out_path = os.path.join(logdir, f'threat_analysis_combined_attack_{attacked_agent_id}_top{top_k}.png')
        plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
        plt.show()
        return
    
    # Create figure with 2 rows: fault timeline (top) and analysis bar charts (bottom)
    fig = plt.figure(figsize=(max(15, 5*len(analysis_timesteps)), 10))
    
    # Get consistent agent colors
    agent_colors = get_agent_colors(total_agents)
    
    # === TOP ROW: FAULT TIMELINE ===
    ax_fault = fig.add_subplot(2, 1, 1)
    ax_fault.set_title('Fault Detection Timeline', fontsize=14, fontweight='bold', pad=20)
    
    if len(fault_timeline) > 0:
        # Plot fault timeline
        for i, event in enumerate(fault_timeline):
            fault_time = event['t']
            fault_agent = event['agent']
            ax_fault.scatter(fault_time, fault_agent, color='red', s=100, marker='x', linewidth=3)
            ax_fault.annotate(f'Fault t={fault_time}', 
                            xy=(fault_time, fault_agent),
                            xytext=(10, 10), textcoords='offset points',
                            fontsize=10, fontweight='bold', color='red')
    
    # Mark all analysis timesteps on fault timeline  
    for t in analysis_timesteps:
        color = 'orange' if t in threat_timesteps else 'lightblue'
        alpha = 0.6 if t in threat_timesteps else 0.3
        ax_fault.axvline(x=t, color=color, alpha=alpha, linewidth=2, linestyle='-')
    
    ax_fault.set_xlim(-1, n_timesteps)
    ax_fault.set_ylim(-0.5, total_agents - 0.5)
    ax_fault.set_xlabel('Timestep')
    ax_fault.set_ylabel('Agent ID')
    ax_fault.grid(True, alpha=0.3)
    ax_fault.set_yticks(range(total_agents))
    ax_fault.set_yticklabels([f'Agent {i}' for i in range(total_agents)])
    
    # Add legend for fault timeline
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='x', color='red', markersize=10, linestyle='None', label='Fault Detected'),
        Line2D([0], [0], color='orange', alpha=0.6, linewidth=3, label='Threat Timestep (Agent Ranks #1)'),
        Line2D([0], [0], color='lightblue', alpha=0.3, linewidth=3, label='Analysis Timestep')
    ]
    ax_fault.legend(handles=legend_elements, loc='upper right')
    
    # === BOTTOM ROW: ANALYSIS BAR CHARTS ===
    # Create subplots for each analysis timestep
    n_analysis_plots = len(analysis_timesteps)
    gs_bottom = fig.add_gridspec(1, n_analysis_plots, top=0.45, bottom=0.05, hspace=0.3, wspace=0.3)
    
    for plot_idx, analysis_t in enumerate(analysis_timesteps):
        ax = fig.add_subplot(gs_bottom[0, plot_idx])
        
        # Get Frobenius matrix for this timestep
        if analysis_t < len(pairwise_frobs_history):
            frob_matrix = pairwise_frobs_history[analysis_t]
            
            # For each agent, get their top-k influencers (excluding self-influence)
            all_agent_data = []
            for target_agent in range(total_agents):
                # Get influences on this target agent (excluding self-influence)
                agent_influences = []
                for j in range(total_agents):
                    if target_agent != j:
                        agent_influences.append((j, frob_matrix[target_agent][j]))
                
                # Sort by influence magnitude (highest first)
                agent_influences.sort(key=lambda x: x[1], reverse=True)
                
                # Get top-k agents and values
                top_k_agents = [agent_id for agent_id, _ in agent_influences[:top_k]]
                top_k_values = [value for _, value in agent_influences[:top_k]]
                
                all_agent_data.append({
                    'target_agent': target_agent,
                    'top_k_agents': top_k_agents,
                    'top_k_values': top_k_values,
                    'attacked_rank': -1  # Calculate rank of attacked agent
                })
                
                # Find rank of attacked agent
                for rank, (agent_id, _) in enumerate(agent_influences):
                    if agent_id == attacked_agent_id:
                        all_agent_data[-1]['attacked_rank'] = rank + 1
                        break
            
            # Create stacked bars for each target agent
            bar_width = 0.6
            x_positions = range(len(all_agent_data))
            
            for i, agent_data in enumerate(all_agent_data):
                bottom = 0
                for j, (agent_id, frob_value) in enumerate(zip(agent_data['top_k_agents'], agent_data['top_k_values'])):
                    # Color coding
                    if agent_id == attacked_agent_id:
                        color = 'red'
                        alpha = 1.0
                    else:
                        color = agent_colors[agent_id]
                        alpha = 0.7
                    
                    bar = ax.bar(i, frob_value, bar_width, bottom=bottom, 
                               color=color, alpha=alpha)
                    
                    # Add value labels on bars
                    ax.text(i, bottom + frob_value/2, f'A{agent_id}\n{frob_value:.3f}', 
                           ha='center', va='center', fontsize=8, fontweight='bold')
                    
                    bottom += frob_value
            
            # Determine if this is a threat timestep
            is_threat = analysis_t in threat_timesteps
            
            # Title based on whether it's a threat or analysis timestep
            if is_threat:
                ax.set_title(f'THREAT t={analysis_t}\n(Agent {attacked_agent_id} Ranks #1)', 
                            fontsize=11, fontweight='bold', color='red')
                ax.set_facecolor('#fff8f8')  # Light red background
            else:
                ax.set_title(f'Analysis t={analysis_t}\n(Agent {attacked_agent_id} Not #1)', 
                            fontsize=11, fontweight='bold', color='blue')
                ax.set_facecolor('#f8f8ff')  # Light blue background
            
            ax.set_xlabel('Influenced Agent')
            ax.set_ylabel('Frobenius Norm')
            ax.set_xticks(x_positions)
            ax.set_xticklabels([f'Agent {i}' for i in range(total_agents)], rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            
        else:
            # No data available for this timestep
            ax.text(0.5, 0.5, f'No data\nfor t={analysis_t}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_title(f'Analysis t={analysis_t}')
    
    # Overall title
    fig.suptitle(f'Fault Detection and Top-{top_k} Analysis Timeline\n(Attack on Agent {attacked_agent_id} - All Analysis Windows)', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Save plot
    out_path = os.path.join(logdir, f'threat_analysis_combined_attack_{attacked_agent_id}_top{top_k}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.show()
    print(f"Saved combined analysis plot to {out_path}")
    print(f"Analysis summary: {len(analysis_timesteps)} total timesteps analyzed")
    print(f"Threat timesteps (Agent {attacked_agent_id} ranks #1): {len(threat_timesteps)}")
    print(f"Analysis timesteps: {sorted(analysis_timesteps)}")
    print(f"Threat timesteps: {sorted(threat_timesteps)}")

############# THREAT DETECTION FUNCTIONS END #############

# -------------- Ploting Starts Here -----------------
def plot_results(results, results_attacked, atk_agent_id, logdir):
        os.makedirs(logdir, exist_ok=True)
        n = len(results[0])  # number of agents
        t = len(results)     # number of time steps
        
        # Create n subplots in a row
        fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
        fig.suptitle(f'Taylor Error (Worst Action Attack | Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.95)
        
        # Ensure axes is iterable even for single agent case
        if n == 1:
            axes = [axes]
        
        for i in range(n):
            ax = axes[i]
            
            # Extract time series for agent i
            normal_series = [results[t][i] for t in range(len(results))]
            attacked_series = [results_attacked[t][i] for t in range(len(results_attacked))]
            
            # Plot the curves
            steps_normal = range(len(normal_series))
            steps_attacked = range(len(attacked_series))
            ax.plot(steps_normal, normal_series, 'b-', label='Normal', linewidth=2)
            ax.plot(steps_attacked, attacked_series, 'r-', label='Attacked', linewidth=2)
            
            ax.set_xlabel('Step')
            ax.set_ylabel('Taylor Delta Error')
            ax.set_title(f'Agent {i}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(os.path.join(logdir, f'plot_analysis_attacked_{atk_agent_id}.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved analysis plot to {logdir}")
def plot_fault_timeline(fault_timeline, total_agents, logdir):
    if len(fault_timeline) == 0:
        print("No faults detected; skipping fault timeline plot.")
        return

    k = len(fault_timeline)
    fig = plt.figure(figsize=(max(6, 3*k), 5))  # reduce height from 6 → 5
    gs = fig.add_gridspec(
        3, k,
        height_ratios=[0.8, 2, 0.1],  # smaller top & bottom rows
        hspace=0.1  # tighter vertical spacing
    )

    cmap = plt.get_cmap('tab20')
    agent_colors = {i: cmap(i % 20) for i in range(total_agents)}

    # --- Timeline axis (top row) ---
    ax_timeline = fig.add_subplot(gs[0, :])
    ax_timeline.axis('off')

    # Horizontal arrow for timeline
    arrow_y = 0.5
    ax_timeline.annotate(
        '', xy=(1, arrow_y), xytext=(0, arrow_y),
        arrowprops=dict(arrowstyle='-|>', lw=2, color='gray'),
        xycoords='axes fraction', textcoords='axes fraction'
    )

    # Milestones
    for i, event in enumerate(fault_timeline):
        frac_x = (i + 0.5) / k  # evenly spaced

        # Circle marker
        ax_timeline.plot(frac_x, arrow_y, 'o', color='darkred', markersize=10, transform=ax_timeline.transAxes)

        # Faulty agent label above
        ax_timeline.text(frac_x, arrow_y + 0.15,
                         f"Faulty agent {event['agent']}",
                         ha='center', va='bottom',
                         fontsize=11, fontweight='bold',
                         transform=ax_timeline.transAxes)

        # Timestep label below
        ax_timeline.text(frac_x, arrow_y - 0.15,
                         f"t = {event['t']}",
                         ha='center', va='top',
                         fontsize=10, color='darkblue',
                         transform=ax_timeline.transAxes)

    # --- Contributor charts (middle row) ---
    for col, event in enumerate(fault_timeline):
        ax = fig.add_subplot(gs[1, col])
        contribs = event.get('contribs', {})

        if len(contribs) == 0:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No prior faults',
                    ha='center', va='center', fontsize=10, style='italic')
        else:
            vals = np.array(list(contribs.values()), dtype=float)
            if vals.sum() > 0:
                vals /= vals.sum()
            colors = [agent_colors[a] for a in contribs.keys()]

            wedges, _, autotexts = ax.pie(
                vals, autopct='%1.1f%%', startangle=90, colors=colors,
                wedgeprops=dict(width=0.35, edgecolor='w')
            )
            for at in autotexts:
                at.set_fontsize(9)
                at.set_fontweight('bold')
            ax.set_title('Contributors', fontsize=11, pad=5)
            ax.set_aspect('equal')

    # --- Legend row (bottom row) ---
    ax_legend = fig.add_subplot(gs[2, :])
    ax_legend.axis('off')
    legend_elements = [Patch(facecolor=agent_colors[i], label=f"Agent {i}") for i in range(total_agents)]
    ax_legend.legend(handles=legend_elements, loc='center', ncol=total_agents,
                     fontsize=9, frameon=False)

    fig.suptitle('Fault Detection Timeline and Contributors',
                 fontsize=14, fontweight='bold', y=0.96)

    out_path = os.path.join(logdir, 'fault_timeline.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved fault timeline plot to {out_path}")


def plot_contributor_barchart(fault_timeline, total_agents, logdir):
    if len(fault_timeline) == 0:
        print("No faults detected; skipping contributor bar chart.")
        return

    k = len(fault_timeline)
    # Increase figure width for better spacing, especially with many events
    fig = plt.figure(figsize=(max(8, 4*k), 6))  # Increased from 3*k to 4*k width and 5 to 6 height
    gs = fig.add_gridspec(
        3, k,
        height_ratios=[0.8, 2.5, 0.2],  # Give more space to the middle row
        hspace=0.15,  # Increase vertical spacing
        wspace=0.3    # Add horizontal spacing between subplots
    )

    cmap = plt.get_cmap('tab20')
    agent_colors = {i: cmap(i % 20) for i in range(total_agents)}

    # --- Timeline axis (top row) ---
    ax_timeline = fig.add_subplot(gs[0, :])
    ax_timeline.axis('off')

    arrow_y = 0.5
    ax_timeline.annotate(
        '', xy=(1, arrow_y), xytext=(0, arrow_y),
        arrowprops=dict(arrowstyle='-|>', lw=2, color='gray'),
        xycoords='axes fraction', textcoords='axes fraction'
    )

    for i, event in enumerate(fault_timeline):
        frac_x = (i + 0.5) / k
        ax_timeline.plot(frac_x, arrow_y, 'o', color='darkred', markersize=10, transform=ax_timeline.transAxes)
        ax_timeline.text(frac_x, arrow_y + 0.15,
                         f"Faulty agent {event['agent']}",
                         ha='center', va='bottom',
                         fontsize=11, fontweight='bold',
                         transform=ax_timeline.transAxes)
        ax_timeline.text(frac_x, arrow_y - 0.15,
                         f"t = {event['t']}",
                         ha='center', va='top',
                         fontsize=10, color='darkblue',
                         transform=ax_timeline.transAxes)

    # --- Contributor bar charts (middle row) ---
    for col, event in enumerate(fault_timeline):
        ax = fig.add_subplot(gs[1, col])
        contribs = event.get('contribs', {})

        if len(contribs) == 0:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No prior faults',
                    ha='center', va='center', fontsize=10, style='italic')
        else:
            agents = list(contribs.keys())
            scores = np.array(list(contribs.values()), dtype=float)

            colors = [agent_colors[a] for a in agents]

            # Use narrower bars with proper spacing
            bar_width = 0.6  # Make bars narrower
            x_positions = range(len(agents))
            bars = ax.bar(x_positions, scores, color=colors, width=bar_width, 
                         edgecolor='black', linewidth=0.5, alpha=0.8)

            # Set appropriate x limits with padding
            if len(agents) > 1:
                ax.set_xlim(-0.8, len(agents) - 0.2)
            else:
                ax.set_xlim(-0.8, 0.8)

            # Improved label handling
            ax.set_xticks(x_positions)
            if len(agents) <= 3:
                # For few agents, use normal labels
                ax.set_xticklabels([f"Agent {i}" for i in agents], fontsize=9)
            else:
                # For many agents, use abbreviated labels with rotation
                ax.set_xticklabels([f"A{i}" for i in agents], rotation=45, ha='right', fontsize=8)

            # Add value labels on top of bars for clarity
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(scores)*0.01,
                       f'{score:.3f}', ha='center', va='bottom', fontsize=7)

            ax.set_ylabel("Contribution", fontsize=9)
            ax.set_title('Contributors', fontsize=11, pad=10)

            # Grid for readability
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Set y-axis to start from 0 for better visual comparison
            ax.set_ylim(bottom=0)

    # --- Legend row (bottom row) ---
    ax_legend = fig.add_subplot(gs[2, :])
    ax_legend.axis('off')
    legend_elements = [Patch(facecolor=agent_colors[i], label=f"Agent {i}") for i in range(total_agents)]
    ax_legend.legend(handles=legend_elements, loc='center', ncol=total_agents,
                     fontsize=9, frameon=False)

    fig.suptitle('Fault Detection Timeline and Contributor Scores',
                 fontsize=14, fontweight='bold', y=0.97)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # Adjusted margins for better label visibility

    out_path = os.path.join(logdir, 'fault_contributor_barchart.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.2)  # Added padding
    plt.show()
    print(f"Saved contributor bar chart to {out_path}")

############# PLOTING ENDS HERE #############

def plot_taylor_error_with_historical_bounds(results_attacked, taylor_history_data, attacked_steps, atk_agent_id, logdir):
    """
    Plot Taylor expansion errors with historical bounds for all agents.
    
    Args:
        results_attacked: List of timesteps, each containing Taylor errors for all agents
        taylor_history_data: Historical statistics from CSV files
        attacked_steps: List of timesteps when attacks occurred  
        atk_agent_id: ID of the attacked agent
        logdir: Directory to save the plot
    """
    n_agents = len(results_attacked[0])  # number of agents
    n_timesteps = len(results_attacked)  # number of time steps
    
    # Create 3 subplots in a row
    fig, axes = plt.subplots(1, n_agents, figsize=(5*n_agents, 5))
    fig.suptitle(f'Taylor Error vs Historical Bounds (Attack on Agent {atk_agent_id})', fontsize=16, y=0.95)
    
    # Ensure axes is iterable even for single agent case
    if n_agents == 1:
        axes = [axes]
    
    for i in range(n_agents):
        ax = axes[i]
        
        # Extract time series for agent i
        timesteps = range(n_timesteps)
        actual_errors = [results_attacked[t][i] for t in range(n_timesteps)]
        
        # Plot historical bounds if available
        if i in taylor_history_data:
            historical_data = taylor_history_data[i]
            historical_means = []
            historical_stds = []
            
            for t in timesteps:
                if t in historical_data:
                    historical_means.append(historical_data[t]['mean'])
                    historical_stds.append(historical_data[t]['std_dev'])
                else:
                    # If no historical data for this timestep, use NaN
                    historical_means.append(np.nan)
                    historical_stds.append(np.nan)
            
            historical_means = np.array(historical_means)
            historical_stds = np.array(historical_stds)
            
            # Calculate upper and lower bounds
            upper_bound = historical_means + historical_stds
            lower_bound = historical_means - historical_stds
            
            # Plot historical bounds as green filled area
            valid_mask = ~np.isnan(historical_means)
            if np.any(valid_mask):
                ax.fill_between(timesteps, lower_bound, upper_bound, 
                               color='green', alpha=0.3, label='Historical Mean ± Std')
                ax.plot(timesteps, historical_means, 'g--', alpha=0.7, linewidth=1, label='Historical Mean')
        
        # Plot actual Taylor errors during attack as red curve
        ax.plot(timesteps, actual_errors, 'r-', linewidth=2, label='Actual Taylor Error')
        
        # Mark attacked timesteps with vertical lines
        if attacked_steps:
            for attack_step in attacked_steps:
                if attack_step < n_timesteps:
                    ax.axvline(x=attack_step, color='orange', linestyle=':', alpha=0.7, linewidth=2)
            # Add legend entry for attack markers (only once)
            if attacked_steps and attacked_steps[0] < n_timesteps:
                ax.axvline(x=attacked_steps[0], color='orange', linestyle=':', alpha=0.7, 
                          linewidth=2, label='Attack Timesteps')
        
        # Highlight the attacked agent
        if i == atk_agent_id:
            ax.set_facecolor('#fff2f2')  # Light red background for attacked agent
            ax.set_title(f'Agent {i} (ATTACKED)', fontweight='bold', color='red')
        else:
            ax.set_title(f'Agent {i}')
            
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Taylor Error')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Set y-axis to start from 0 for better comparison
        ax.set_ylim(bottom=0)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(logdir, f'taylor_error_vs_historical_bounds_attack_{atk_agent_id}.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved Taylor error vs historical bounds plot to {logdir}")

def get_agent_colors(n_agents):
    """Helper function to generate consistent colors for agents."""
    cmap = plt.get_cmap('tab20')
    return {i: cmap(i % 20) for i in range(n_agents)}

def plot_pairwise_observation_influences(observation_influences_normal, observation_influences_attacked, attacked_steps, atk_agent_id, logdir):
    """
    Plot N×N grid of subplots showing pairwise observation influences.
    Each subplot (i,j) shows the influence of agent j's observation on agent i for both normal and attacked scenarios.
    """
    N = len(observation_influences_normal[0])  # number of agents
    T_normal = len(observation_influences_normal)
    T_attacked = len(observation_influences_attacked)
    
    # Create N×N subplots
    fig, axes = plt.subplots(N, N, figsize=(3*N, 3*N))
    if N == 1:
        axes = [[axes]]
    elif N > 1 and axes.ndim == 1:
        axes = axes.reshape(N, N)
    
    fig.suptitle(f'Pairwise Observation Influences: ∂V_i/∂obs_j (Attacked Agent ID: {atk_agent_id})', fontsize=14, y=0.95)
    
    for i in range(N):  # agent i (influenced)
        for j in range(N):  # agent j (influencer)
            ax = axes[i][j]
            
            # Extract influence time series for agent j's observation influence on agent i
            normal_series = [observation_influences_normal[t][i][j] for t in range(T_normal)]
            attacked_series = [observation_influences_attacked[t][i][j] for t in range(T_attacked)]
            
            # Plot time series
            timesteps_normal = range(T_normal)
            timesteps_attacked = range(T_attacked)
            
            ax.plot(timesteps_normal, normal_series, 'g-', label='Normal', linewidth=2, alpha=0.8)
            ax.plot(timesteps_attacked, attacked_series, 'r-', label='Attacked', linewidth=2, alpha=0.8)
            
            # Mark attacked region
            if attacked_steps:
                start = min(attacked_steps)
                end = max(attacked_steps)
                ax.axvspan(start, end, color='red', alpha=0.1)
            
            # Styling
            ax.set_title(f'Agent {j} → Agent {i}', fontsize=10)
            ax.set_xlabel('Timestep', fontsize=9)
            ax.set_ylabel('Observation Influence Magnitude', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            # Add legend only to the top-right subplot to avoid clutter
            if i == 0 and j == N-1:
                ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    out_path = os.path.join(logdir, f'pairwise_observation_influences_attacked_{atk_agent_id}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved pairwise observation influences plot to {out_path}")

def plot_second_order_observation_influences(second_order_observation_influences_history, attacked_steps, atk_agent_id, logdir):
    """
    Plot the time series of second-order observation influences for each agent.
    For each agent i, show how much each agent j's observation curvature influences agent i's V-value over time.
    """
    N = len(second_order_observation_influences_history[0])  # number of agents
    T = len(second_order_observation_influences_history)     # number of timesteps
    
    # Create N subplots (one for each agent showing second-order influences on it)
    max_per_row = 3
    rows = math.ceil(N / max_per_row)
    cols = min(N, max_per_row)
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows + 1))
    if N == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    fig.suptitle(f'Second-Order Observation Influences: ∂²V_i/(∂obs_j)² (Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.96)
    
    # Use consistent agent colors
    agent_colors = get_agent_colors(N)
    
    # Track legend elements to avoid duplicates
    legend_elements = []
    legend_labels = []
    attacked_region_element = None
    
    for i in range(N):
        ax = axes[i]
        
        # Plot second-order observation influences for each agent j on agent i
        for j in range(N):
            influence_series = [second_order_observation_influences_history[t][i][j] for t in range(T)]
            timesteps = range(T)
            
            # Self-influence (solid line) vs others (dashed line)
            if i == j:
                line = ax.plot(timesteps, influence_series, color=agent_colors[j], 
                                linewidth=2.0, linestyle='-', alpha=1.0)[0]
            else:
                line = ax.plot(timesteps, influence_series, color=agent_colors[j], 
                                linewidth=2.0, linestyle='--', alpha=1.0)[0]
            
            # Add to legend only for self-influence terms to avoid clutter
            if i == j:
                legend_elements.append(line)
                legend_labels.append(f'Agent {j}')
        
        # Mark attacked region
        if attacked_steps:
            start = min(attacked_steps)
            end = max(attacked_steps)
            if i == atk_agent_id:
                region = ax.axvspan(start, end, color='red', alpha=0.1)
                if attacked_region_element is None:
                    attacked_region_element = region
            else:
                ax.axvspan(start, end, color='orange', alpha=0.05)
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Second-Order Observation Influence Magnitude')
        ax.set_title(f'Second-Order Observation Influences on Agent {i}')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for j in range(N, len(axes)):
        fig.delaxes(axes[j])
    
    # Add attacked region to legend at the end if it exists
    if attacked_region_element is not None:
        legend_elements.append(attacked_region_element)
        legend_labels.append('Attacked Region')
    
    # Create single legend at the bottom
    fig.legend(legend_elements, legend_labels, loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), ncol=min(len(legend_labels), 6), 
               fontsize=9, frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout to make room for legend
    plt.tight_layout(rect=[0, 0.08, 1, 0.94])
    
    out_path = os.path.join(logdir, f'second_order_observation_influences_attacked_{atk_agent_id}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved second-order observation influences plot to {out_path}")

def plot_pairwise_frobs_comparison(pairwise_frobs_history_normal, pairwise_frobs_history_attacked, attacked_steps, atk_agent_id, logdir):
    """
    Plot comparison of pairwise Frobenius norms between normal and attacked scenarios.
    Creates a 2x3 grid: top row shows normal scenario, bottom row shows attacked scenario.
    Each column represents an agent (0, 1, 2) and shows how all other agents' observations influence that agent.
    """
    N = len(pairwise_frobs_history_normal[0])  # number of agents
    T_normal = len(pairwise_frobs_history_normal)
    T_attacked = len(pairwise_frobs_history_attacked)
    
    # Create 2x3 subplots (2 rows: normal vs attacked, 3 columns: agents 0, 1, 2)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Pairwise Frobenius Norms Comparison: Normal vs Attacked (Agent {atk_agent_id})', fontsize=16, y=0.95)
    
    # Get agent colors for consistency
    agent_colors = get_agent_colors(N)
    
    # Row 0: Normal scenario
    for agent_i in range(3):  # For agents 0, 1, 2
        ax = axes[0, agent_i]
        
        # Plot influence of each agent j on agent i
        for agent_j in range(N):
            influence_series = [pairwise_frobs_history_normal[t][agent_i][agent_j] for t in range(T_normal)]
            timesteps = range(T_normal)
            
            # Use different line styles for self vs others
            if agent_i == agent_j:
                line = ax.plot(timesteps, influence_series, color=agent_colors[agent_j], 
                              linewidth=2.5, linestyle='-', alpha=1.0, label=f'Agent {agent_j}')[0]
            else:
                line = ax.plot(timesteps, influence_series, color=agent_colors[agent_j], 
                              linewidth=2.0, linestyle='--', alpha=0.8, label=f'Agent {agent_j}')[0]
        
        ax.set_title(f'Normal: Influences on Agent {agent_i}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Frobenius Norm')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
    
    # Row 1: Attacked scenario  
    for agent_i in range(3):  # For agents 0, 1, 2
        ax = axes[1, agent_i]
        
        # Plot influence of each agent j on agent i
        for agent_j in range(N):
            influence_series = [pairwise_frobs_history_attacked[t][agent_i][agent_j] for t in range(T_attacked)]
            timesteps = range(T_attacked)
            
            # Use different line styles for self vs others
            if agent_i == agent_j:
                line = ax.plot(timesteps, influence_series, color=agent_colors[agent_j], 
                              linewidth=2.5, linestyle='-', alpha=1.0, label=f'Agent {agent_j}')[0]
            else:
                line = ax.plot(timesteps, influence_series, color=agent_colors[agent_j], 
                              linewidth=2.0, linestyle='--', alpha=0.8, label=f'Agent {agent_j}')[0]
        
        # Mark attacked region
        if attacked_steps:
            start = min(attacked_steps)
            end = max(attacked_steps)
            if agent_i == atk_agent_id:
                # Highlight attacked agent with red background
                ax.axvspan(start, end, color='red', alpha=0.15, label='Attack Period')
            else:
                # Light highlight for other agents
                ax.axvspan(start, end, color='orange', alpha=0.08)
        
        # Special styling for attacked agent
        if agent_i == atk_agent_id:
            ax.set_title(f'Attacked: Agent {agent_i} (UNDER ATTACK)', fontsize=12, fontweight='bold', color='red')
            ax.set_facecolor('#fff8f8')  # Very light red background
        else:
            ax.set_title(f'Attacked: Influences on Agent {agent_i}', fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Frobenius Norm')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    
    out_path = os.path.join(logdir, f'pairwise_frobs_comparison_attacked_{atk_agent_id}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved pairwise Frobenius norms comparison plot to {out_path}")

############# PLOTING ENDS HERE #############

# --------- COMPUTE TAYLOR POLICY ------------------
def compute_taylor_policy(runner, eval_obs, eval_available_actions, eval_rnn_states):
        # states_tensor = torch.stack([torch.tensor(state_dict[k], dtype=torch.float32, requires_grad=True) for k in state_dict.keys()])
        eval_obs = torch.tensor(eval_obs, dtype=torch.float32, requires_grad=True)
        delta_errors = []
        eval_actions_collector = []
        eval_masks = np.ones(
            (runner.algo_args["eval"]["n_eval_rollout_threads"], runner.num_agents, 1),
            dtype=np.float32,
        )

        for agent_id in range(runner.num_agents):
            cur_obs = eval_obs[:, agent_id]
            eval_actions, eval_actions_log_prob, temp_rnn_state = runner.actor[agent_id].get_actions(
                cur_obs,
                eval_rnn_states[:, agent_id],
                eval_masks[:, agent_id],
                eval_available_actions[:, agent_id]
                if eval_available_actions[0] is not None
                else None,
                deterministic=True,
            )
            # eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
            eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
            grad_i = torch.autograd.grad(
                outputs=eval_actions_log_prob,
                inputs=cur_obs,
                create_graph=True,
                retain_graph=True,
            )[0]

            eta_i = 0.01 * grad_i.sign() / torch.max(grad_i.norm(p=2), torch.tensor(1e-6))

            
            j_tilde = eval_actions_log_prob + torch.dot(grad_i.flatten(), eta_i.flatten())  #+ 0.5 * torch.dot(eta_i.flatten(), hvp.flatten())

            p_obs = cur_obs + eta_i
            _, perturb_log_prob, _ = runner.actor[agent_id].get_actions(
                p_obs,
                eval_rnn_states[:, agent_id],
                eval_masks[:, agent_id],
                eval_available_actions[:, agent_id]
                if eval_available_actions[0] is not None
                else None,
                deterministic=True,
            )
            # _, _, p_log_prob = runner.agents.choose_actions_attack(p_state, i)
            # # Actual value of perturbed point
            j_perturbed = perturb_log_prob

            delta_error = abs(j_perturbed - j_tilde).item()
            delta_errors.append(delta_error)

        return delta_errors
# ---------- Evaluation Loop --
def slice_avail(avail, agent_id):
    """Extract available actions for a specific agent"""
    if avail is None:
        return None
    first = avail[0]
    if first is None:
        return None
    return avail[:, agent_id]

def eval(runner, attack_status=False, attack_agent_id=0, seed=23, taylor_history_data=None, min_window=0, max_window=None, top_k=2):
    """Evaluate the model."""
    
    eval_episode = 0

    eval_obs, eval_share_obs, eval_available_actions = runner.eval_envs.reset(seed=seed)

    eval_rnn_states = np.zeros(
        (
            runner.algo_args["eval"]["n_eval_rollout_threads"],
            runner.num_agents,
            runner.recurrent_n,
            runner.rnn_hidden_size,
        ),
        dtype=np.float32,
    )
    eval_rnn_states_critic = np.zeros(
        (
            runner.algo_args["eval"]["n_eval_rollout_threads"],
            runner.num_agents,
            runner.recurrent_n,
            runner.rnn_hidden_size,
        ),
        dtype=np.float32,
    )
    eval_masks = np.ones(
        (runner.algo_args["eval"]["n_eval_rollout_threads"], runner.num_agents, 1),
        dtype=np.float32,
    )

    taylor_error_list = list()
    frob_norms_list = []
    sec_dir_derivatives = []
    result_deques = [deque(maxlen=5) for _ in range(runner.num_agents)]
    frob_norms_deques = [deque(maxlen=5) for _ in range(runner.num_agents)]
    sec_dir_derivatives_deques = [deque(maxlen=5) for _ in range(runner.num_agents)]

    # Additional structures to mirror get_episode_data logic
    frob_norms_matrix_history = []  # list of N x N pairwise frob matrices per timestep
    fault_first_detected = {}  # agent_id -> first detected timestep
    fault_timeline = []
    attacked_steps = []
    pairwise_frobs_history = []  # current timestep's N x N frob matrix
    pairwise_obs_influences_history = []  # list of N x N pairwise obs influence matrices per timestep
    pairwise_second_order_influences_history = []  # list of N x N pair
    taylor_history = [[] for _ in range(runner.num_agents)]
    cnt = 0
    first_faulty_timestep = None
    while True:
        eval_rnn_states_backup = np.copy(eval_rnn_states)
        eval_actions_collector = []
        for agent_id in range(runner.num_agents):
            eval_actions, temp_rnn_state = runner.actor[agent_id].act(
                eval_obs[:, agent_id],
                eval_rnn_states[:, agent_id],
                eval_masks[:, agent_id],
                eval_available_actions[:, agent_id]
                if eval_available_actions[0] is not None
                else None,
                deterministic=True,
            )
            eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
            eval_actions_collector.append(_t2n(eval_actions))

        eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
        
        if attack_status:
            # Check if current timestep is within attack window
            should_attack = (cnt >= min_window) and (max_window is None or cnt <= max_window)
            
            if should_attack:
                # mark attacked step
                # eval_actions[0][attack_agent_id] = runner.eval_envs.action_space[attack_agent_id].sample()  # Random action for attack agent
                # print(f">>>> ",eval_actions[0][attack_agent_id])
                n_actions = runner.eval_envs.action_space[attack_agent_id].n
                avail_slice = slice_avail(eval_available_actions, attack_agent_id)
                if avail_slice is not None and avail_slice[0] is not None:
                            available_actions = np.where(avail_slice[0] > 0.5)[0]
                else:
                    available_actions = list(range(n_actions))
                
                # exit("Exiting for debug")
                with torch.no_grad():
                    print(f" [!!!] Attack launched on agent {attack_agent_id} at timestep: {cnt}")
                    obs_tensor = torch.FloatTensor(eval_obs[:, attack_agent_id])
                    rnn_tensor = torch.FloatTensor(eval_rnn_states[:, attack_agent_id])
                    mask_tensor = torch.FloatTensor(eval_masks[:, attack_agent_id])
                    
                    action_log_probs, dist_entropy, action_distribution = runner.actor[attack_agent_id].evaluate_actions(
                                obs_tensor.to(runner.device),
                                rnn_tensor.to(runner.device),
                                available_actions,
                                mask_tensor.to(runner.device),
                                slice_avail(eval_available_actions, attack_agent_id),
                                None
                            )
                    q_values = action_log_probs.squeeze()
                    if q_values.numel() == 1 or len(available_actions) == 1:
                        print(f"Agent {attack_agent_id} appears to be dead or has only one action. Using available action.")
                        eval_actions[0][attack_agent_id] = available_actions[0]
                        attacked_steps.append(cnt)  # Still count as attacked even if limited actions
                    else:
                        # Get the index of the worst action within the available actions
                        worst_action_idx = torch.argmin(q_values).item()
                        # Map back to the actual action ID from available actions
                        worst_action = available_actions[worst_action_idx]
                        eval_actions[0][attack_agent_id] = worst_action
                        print(f"Agent {attack_agent_id} worst action under current policy: {worst_action}")
                        attacked_steps.append(cnt)  # Only add to attacked_steps when attack actually occurs
            else:
                # Outside attack window - agent acts normally (already handled above)
                if cnt < min_window:
                    print(f" [---] Timestep {cnt} < min_window ({min_window}), no attack")
                elif max_window is not None and cnt > max_window:
                    print(f" [---] Timestep {cnt} > max_window ({max_window}), no attack")


        # calculating taylor policy
        delta_errors = compute_taylor_policy(runner, eval_obs, eval_available_actions, eval_rnn_states_backup)
        # results_frob_norms = compute_frob_norms(runner, eval_obs, 1, eval_rnn_states_critic, eval_masks)
        # results_sec_dir_derivatives = compute_2nd_ord_dir_derivatives(runner, eval_obs, 1, eval_rnn_states_critic, eval_masks)
        pairwise_frobs = compute_pairwise_frob_norms(runner, eval_obs, eval_rnn_states_critic, eval_masks)
        pairwise_frobs_history.append(pairwise_frobs)
        # HAPPO adaptation: compute pairwise observation influences for value functions
        # These replace the MADDPG-specific compute_pairwise_observation_influences functions
        pairwise_obs_influences = compute_pairwise_observation_influences_happo(runner, eval_obs, eval_rnn_states_critic, eval_masks)
        pairwise_second_order_influences = compute_second_order_observation_influences_happo(runner, eval_obs, eval_rnn_states_critic, eval_masks)
        pairwise_obs_influences_history.append(pairwise_obs_influences)
        pairwise_second_order_influences_history.append(pairwise_second_order_influences)
        # pairwise frob matrix for cascading analysis
        pairwise_frobs = compute_pairwise_frob_norms(runner, eval_obs, eval_rnn_states_critic, eval_masks)
        frob_norms_matrix_history.append(pairwise_frobs)
        
        for i in range(runner.num_agents):
            result_deques[i].append(delta_errors[i])
            taylor_approx_error = np.mean(result_deques[i])
            taylor_history[i].append(taylor_approx_error)

            # frob_norms_deques[i].append(results_frob_norms[i])
            # sec_dir_derivatives_deques[i].append(results_sec_dir_derivatives[i])

            # Detect anomalies based on Taylor approximation error using pre-computed history
            if i not in fault_first_detected and cnt in taylor_history_data[i]:
                historical_data = taylor_history_data[i][cnt]
                historical_mean = historical_data['mean']
                historical_std = historical_data['std_dev']
                
                # Ensure minimum std deviation to avoid division by zero
                if historical_std < 1e-6:
                    historical_std = 1e-6
                
                # Check if current error is outside historical bounds (mean ± std_dev)
                lower_bound = historical_mean - historical_std
                upper_bound = historical_mean + historical_std
                
                if taylor_approx_error < lower_bound*0.5 or taylor_approx_error > upper_bound*2:
                    if first_faulty_timestep is None:
                        first_faulty_timestep = cnt
                    print(f" [!!!] Anomaly detected for agent {i} at timestep: {cnt}. Taylor Appx. Error: {taylor_approx_error}")
                    print(f"     >> Historical bounds: [{lower_bound:.6f}, {upper_bound:.6f}], Mean: {historical_mean:.6f}, Std: {historical_std:.6f}")
                    fault_first_detected[i] = cnt
                    # Cascading Impact Analysis
                    prev_faults = [(f, tf) for f, tf in fault_first_detected.items() if f != i and tf < cnt]
                    contribs = {}
                    if len(prev_faults) > 0:
                        # for f, tf in prev_faults:
                        #     values_over_time = [frob_norms_matrix_history[tau][i][f] for tau in range(tf, cnt + 1) if tau < len(frob_norms_matrix_history)]
                        #     if len(values_over_time) > 0:
                        #         contribs[f] = float(np.mean(values_over_time))
                        for agent_id in range(runner.num_agents):
                            values_over_time = [frob_norms_matrix_history[tau][i][agent_id] for tau in range(first_faulty_timestep, cnt + 1) if tau < len(frob_norms_matrix_history)]
                            if len(values_over_time) > 0:
                                contribs[agent_id] = float(np.mean(values_over_time))
                        if len(contribs) > 0:
                            ranked = sorted(contribs.items(), key=lambda x: x[1], reverse=True)
                            print(f"     >> Potential contributors to fault in agent {i} (mean ||H_{{i,f}}||_F from t_f to {cnt}): {ranked}")
                    fault_timeline.append({
                        'agent': i,
                        't': cnt,
                        'contribs': contribs
                    })

        taylor_error_list.append([np.mean(list(result_deques[j])) for j in range(runner.num_agents)])
        # frob_norms_list.append([np.mean(frob_norms_deques[i]) for i in range(runner.num_agents)])
        # sec_dir_derivatives.append([np.mean(sec_dir_derivatives_deques[i]) for i in range(runner.num_agents)])

        (
            eval_obs,
            eval_share_obs,
            eval_rewards,
            eval_dones,
            eval_infos,
            eval_available_actions,
        ) = runner.eval_envs.step(eval_actions)
        eval_data = (
            eval_obs,
            eval_share_obs,
            eval_rewards,
            eval_dones,
            eval_infos,
            eval_available_actions,
        )

        value, eval_rnn_states_critic = runner.critic.get_values(
            eval_share_obs,
            eval_rnn_states_critic,
            eval_masks,
        )

        eval_dones_env = np.all(eval_dones, axis=1)

        eval_rnn_states[
            eval_dones_env == True
        ] = np.zeros(  # if env is done, then reset rnn_state to all zero
            (
                (eval_dones_env == True).sum(),
                runner.num_agents,
                runner.recurrent_n,
                runner.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

        eval_masks = np.ones(
            (runner.algo_args["eval"]["n_eval_rollout_threads"], runner.num_agents, 1),
            dtype=np.float32,
        )
        eval_masks[eval_dones_env == True] = np.zeros(
            ((eval_dones_env == True).sum(), runner.num_agents, 1), dtype=np.float32
        )

        for eval_i in range(runner.algo_args["eval"]["n_eval_rollout_threads"]):
            if eval_dones_env[eval_i]:
                eval_episode += 1

        if eval_episode >= runner.algo_args["eval"]["eval_episodes"]:
            break

        cnt += 1
    
    # Perform threat detection analysis around fault events only
    threat_data = detect_threats_top_k(pairwise_frobs_history, attack_agent_id, top_k, fault_timeline)
    
    # NEW: Perform faulty agents in top-k analysis (general fault propagation detection)
    threat_data_faulty_topk = detect_threats_faulty_in_topk(pairwise_frobs_history, fault_timeline, top_k)
    
    # return taylor_error_list, frob_norms_list, sec_dir_derivatives, frob_norms_matrix_history, fault_timeline, attacked_steps
    return taylor_error_list, pairwise_frobs_history, pairwise_obs_influences_history, pairwise_second_order_influences_history, frob_norms_matrix_history, fault_timeline, attacked_steps, threat_data, threat_data_faulty_topk

def compute_pairwise_frob_norms(runner, eval_obs, eval_rnn_states_critic, eval_masks):
    """Compute Frobenius norms of cross-agent Hessian blocks for all (i, j).
    Returns an N x N matrix where entry [i][j] approximates || \partial^2 v_i / (\partial obs_i \partial obs_j) ||_F.
    """
    eval_obs = torch.tensor(eval_obs, dtype=torch.float32)

    agent_obs_tensors = []
    n_agents = runner.num_agents
    # assume eval_obs shape (1, n_agents, obs_dim)
    for i in range(n_agents):
        agent_obs = eval_obs[0][i].clone().detach()
        agent_obs_tensor = torch.tensor(agent_obs, dtype=torch.float32, requires_grad=True)
        agent_obs_tensors.append(agent_obs_tensor)

    concatenated_obs = torch.cat(agent_obs_tensors, dim=0)
    share_obs = concatenated_obs.unsqueeze(0).unsqueeze(0)
    share_obs = share_obs.expand(1, n_agents, -1)

    values, temp_rnn_state_critic = runner.critic.get_values(
        share_obs,
        eval_rnn_states_critic,
        eval_masks,
    )
    values = values.squeeze()

    N = n_agents
    results = [[0.0 for _ in range(N)] for _ in range(N)]

    for i in range(N):
        # gradient of v_i wrt agent i obs
        try:
            grad_i = torch.autograd.grad(values[i], agent_obs_tensors[i], create_graph=True, retain_graph=True)[0]
        except Exception:
            grad_i = torch.zeros_like(agent_obs_tensors[i])

        for j in range(N):
            hessian_matrix = []
            for k in range(grad_i.shape[0]):
                second_grad = torch.autograd.grad(
                    grad_i[k],
                    agent_obs_tensors[j],
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if second_grad is None:
                    second_grad = torch.zeros_like(agent_obs_tensors[j])
                hessian_matrix.append(second_grad.flatten())

            H = torch.stack(hessian_matrix) if len(hessian_matrix) > 0 else torch.zeros(1, 1)
            results[i][j] = H.norm(p='fro').item()
    # Normalize each row by its sum
    for i in range(N):
        row_sum = sum(results[i])
        if row_sum > 0:
            for j in range(N):
                results[i][j] /= row_sum
    return results

def compute_frob_norms(runner, eval_obs, vulnerable_agent_id, eval_rnn_states_critic, eval_masks):
    eval_obs = torch.tensor(eval_obs, dtype=torch.float32)

    # Create separate tensors for each agent's observations with gradient tracking
    agent_obs_tensors = []
    obs_dim = runner.envs.observation_space[0].shape[0]  # Assuming all agents have same obs dim
    n_agents = runner.num_agents

    for i in range(n_agents):
        # Extract agent i's observation from share_obs
        # start_idx = i * obs_dim
        # end_idx = (i + 1) * obs_dim
        agent_obs = eval_obs[0][i].clone().detach()
        agent_obs_tensor = torch.tensor(agent_obs, dtype=torch.float32, requires_grad=True)
        agent_obs_tensors.append(agent_obs_tensor)

    # Concatenate back to recreate share_obs structure with gradient tracking
    concatenated_obs = torch.cat(agent_obs_tensors, dim=0)
    share_obs = concatenated_obs.unsqueeze(0).unsqueeze(0)  # Reshape to (1, 1, obs_dim*n_agent)
    # Need to expand to match original shape (1, n_agents, obs_dim*n_agent)
    share_obs = share_obs.expand(1, n_agents, -1)
   
    values, temp_rnn_state_critic = runner.critic.get_values(
            share_obs,
            eval_rnn_states_critic,
            eval_masks,
        )
    values = values.squeeze()  # shape: (N,)

    # values = runner.agent_n.compute_value(states_tensor).squeeze(-1)  # shape: (N,)

    # Store eigenvalues for each agent pair (i, j)
    results = []

    for i in range(runner.num_agents):
        obs_dim = runner.envs.observation_space[i].shape[0]
        # Compute first-order gradient of v_i with respect to agent i's observation
        grad_i = torch.autograd.grad(values[i], agent_obs_tensors[i], create_graph=True, retain_graph=True)[0]

        # Compute cross-agent Hessian matrix for agent pair (i, j)
        # This represents ∂²v/∂obs_i∂obs_j
        hessian_matrix = []
        
        for k in range(grad_i.shape[0]):  # For each dimension of agent i's observation (has shape (obs_dim,))
            # Compute ∂²v/∂obs_i[k]∂obs_j
            second_grad = torch.autograd.grad(
                grad_i[k],
                agent_obs_tensors[vulnerable_agent_id],
                retain_graph=True,
                allow_unused=True
            )[0]
            hessian_matrix.append(second_grad.flatten())

        # Convert to tensor and compute eigenvalues
        H = torch.stack(hessian_matrix)

        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvals(H)
        real_eigenvalues = eigenvalues.real
        negative_count = (real_eigenvalues < 0).sum().item()
        total_count = len(real_eigenvalues)
        negative_ratio = negative_count / total_count if total_count > 0 else 0.0
        results.append(negative_ratio)
        continue

        # Frobenius norm of the Hessian matrix
        results.append(H.norm(p='fro').item()) 

    return results

# second order directional derivative
def compute_2nd_ord_dir_derivatives(runner, eval_obs, vulnerable_agent_id, eval_rnn_states_critic, eval_masks):
    eval_obs = torch.tensor(eval_obs, dtype=torch.float32)

    # Create separate tensors for each agent's observations with gradient tracking
    agent_obs_tensors = []
    obs_dim = runner.envs.observation_space[0].shape[0]  # Assuming all agents have same obs dim
    n_agents = runner.num_agents

    for i in range(n_agents):
        # Extract agent i's observation from share_obs
        # start_idx = i * obs_dim
        # end_idx = (i + 1) * obs_dim
        agent_obs = eval_obs[0][i].clone().detach()
        agent_obs_tensor = torch.tensor(agent_obs, dtype=torch.float32, requires_grad=True)
        agent_obs_tensors.append(agent_obs_tensor)

    # Concatenate back to recreate share_obs structure with gradient tracking
    concatenated_obs = torch.cat(agent_obs_tensors, dim=0)
    share_obs = concatenated_obs.unsqueeze(0).unsqueeze(0)  # Reshape to (1, 1, obs_dim*n_agent)
    # Need to expand to match original shape (1, n_agents, obs_dim*n_agent)
    share_obs = share_obs.expand(1, n_agents, -1)
   
    values, temp_rnn_state_critic = runner.critic.get_values(
            share_obs,
            eval_rnn_states_critic,
            eval_masks,
        )
    values = values.squeeze()  # shape: (N,)

    # values = runner.agent_n.compute_value(states_tensor).squeeze(-1)  # shape: (N,)

    # Store eigenvalues for each agent pair (i, j)
    results = []

    for i in range(runner.num_agents):
        obs_dim = runner.envs.observation_space[i].shape[0]
        # Compute first-order gradient of v_i with respect to agent i's observation
        grad_i = torch.autograd.grad(values[i], agent_obs_tensors[i], create_graph=True, retain_graph=True)[0]
        v = grad_i / torch.max(grad_i.norm(p=2), torch.tensor(1e-6))

        # Compute Hessian-vector product (HVP) of grad_i and v with respect to states_tensors[j]
        hvp = torch.autograd.grad(
            outputs=grad_i,
            inputs=agent_obs_tensors[vulnerable_agent_id],
            grad_outputs=v,
            retain_graph=True,
            allow_unused=True
        )[0]

        # Compute u^T * H * v (quadratic form)
        grad_j = torch.autograd.grad(-values[i], agent_obs_tensors[vulnerable_agent_id], create_graph=True, retain_graph=True)[0]
        u = grad_j / torch.max(grad_j.norm(p=2), torch.tensor(1e-6))
        u = -u  # negative gradient direction
        curvature_val = torch.dot(u.flatten(), hvp.flatten())
        results.append(curvature_val.item())

    return results

def plot_frobs(frobs_normal, frobs_atk, attacked_steps, atk_agent_id, logdir):
    n = len(frobs_normal[0])  # number of agents
    t = len(frobs_normal)     # number of time steps
    
    # Create n subplots in a row
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    fig.suptitle(f'Frobenius Norms (Worst Action Attack | Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.95)
    
    # Ensure axes is iterable even for single agent case
    if n == 1:
        axes = [axes]
    
    for i in range(n):
        ax = axes[i]
        
        # Extract time series for agent i
        normal_series = [frobs_normal[t][i] for t in range(len(frobs_normal))]
        attacked_series = [frobs_atk[t][i] for t in range(len(frobs_atk))]
        
        # Plot the curves
        steps = range(len(normal_series))
        ax.plot(steps, attacked_series, 'r-', label='Under Attack', linewidth=2)
        ax.plot(steps, normal_series, 'g-', label='Normal', linewidth=2)
        
        # Mark attacked timesteps with vertical lines
        if attacked_steps:
            for attack_step in attacked_steps:
                ax.axvline(x=attack_step, color='orange', linestyle=':', alpha=0.5, linewidth=1.5)
            # Add legend entry for attack markers
            ax.axvline(x=attacked_steps[0], color='orange', linestyle=':', alpha=0.5, linewidth=1.5, label='Attacked Steps')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Frobenius Norm')
        ax.set_title(f'Agent {i}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(logdir, f'plot_frobs_attacked_{atk_agent_id}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved frobenius norms plot to {logdir}")

def plot_sec_dir_derivatives(s_dir_derv_normal, s_dir_derv_atk, attacked_steps, atk_agent_id, logdir):
    n = len(s_dir_derv_normal[0])  # number of agents
    t = len(s_dir_derv_normal)     # number of time steps
    
    # Create n subplots in a row
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    fig.suptitle(f'2nd Ord. Dir. Derivatives (Worst Action Attack | Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.95)
    
    # Ensure axes is iterable even for single agent case
    if n == 1:
        axes = [axes]
    
    for i in range(n):
        ax = axes[i]
        
        # Extract time series for agent i
        normal_series = [s_dir_derv_normal[t][i] for t in range(len(s_dir_derv_normal))]
        attacked_series = [s_dir_derv_atk[t][i] for t in range(len(s_dir_derv_atk))]
        
        # Plot the curves
        steps = range(len(normal_series))
        
        ax.plot(steps, attacked_series, 'r-', label='Under Attack', linewidth=2)
        ax.plot(steps, normal_series, 'g-', label='Normal', linewidth=2)
        
        # Highlight region under y < 0 in red
        y_min = min(min(normal_series), min(attacked_series))
        if y_min < 0:
            ax.axhspan(y_min * 1.1, 0, alpha=0.2, color='red')
        
        # Mark attacked timesteps with vertical lines
        if attacked_steps:
            for attack_step in attacked_steps:
                ax.axvline(x=attack_step, color='orange', linestyle=':', alpha=0.5, linewidth=1.5)
            # Add legend entry for attack markers
            ax.axvline(x=attacked_steps[0], color='orange', linestyle=':', alpha=0.5, linewidth=1.5, label='Attacked Steps')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('2nd Ord. Dir. Derivative')
        ax.set_title(f'Agent {i}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(logdir, f'plot_sec_dir_derivatives_attacked_{atk_agent_id}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved 2nd ord. dir. derivatives plot to {logdir}")

def save_matrix_to_files(matrix, attacked_steps, attacked_agent_id, total_agents, logdir, filename):
    """
    Save matrix data to a CSV file for all timesteps.
    
    Args:
        matrix: List of timesteps, where each timestep contains n_agent data
        attacked_agent_id: ID of the attacked agent
        total_agents: Total number of agents
        logdir: Directory to save the file
    """
    filepath = os.path.join(logdir, filename)
    
    # Create header row
    # header = ["timestep", "attacked_agent"]
    header = ["timestep", "is_attacked", "attacked_agent"]
    for i in range(total_agents):
        header.append(f"agent_{i}")
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        for timestep, timestep_data in enumerate(matrix):
            is_attacked = 1 if timestep in attacked_steps else 0
            row = [timestep, is_attacked, attacked_agent_id]
            for i in range(total_agents):
                row.append(timestep_data[i])
            writer.writerow(row)
    
    print(f"Saved {len(matrix)} timestep matrices to {filepath}")

def restore(runner, restore_dir, reward):
    """Restore trained model from checkpoint"""
    for agent_id in range(runner.num_agents):
        policy_actor_state_dict = torch.load(
            os.path.join(restore_dir, f"actor_agent{agent_id}_{reward}.pt"),
            weights_only=False, map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
        runner.actor[agent_id].actor.load_state_dict(policy_actor_state_dict)

# def restore(runner,reward,episode,filepath="/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v2-discrete/happo/installtest/seed-00042-2025-08-03-20-41-48/models"):
#         """Restore model parameters."""
#         for agent_id in range(runner.num_agents):
#             policy_actor_state_dict = torch.load(
#                 str(filepath)
#                 + "/actor_agent"
#                 + str(agent_id)
#                 + "_reward_" + str(reward)
#                 + "_episode_" + str(episode)
#                 + ".pt"
#             )
#             runner.actor[agent_id].actor.load_state_dict(policy_actor_state_dict)
#         if not runner.algo_args["render"]["use_render"]:
#             policy_critic_state_dict = torch.load(
#                 str(filepath)
#                 + "/critic_agent"
#                 + "_reward_" + str(reward)
#                 + "_episode_" + str(episode)
#                 + ".pt"
#             )
#             runner.critic.critic.load_state_dict(policy_critic_state_dict)
#             if runner.value_normalizer is not None:
#                 value_normalizer_state_dict = torch.load(
#                     str(filepath)
#                     + "/value_normalizer"
#                     + "_reward_" + str(reward)
#                     + "_episode_" + str(episode)
#                     + ".pt"
#                 )
#                 runner.value_normalizer.load_state_dict(value_normalizer_state_dict)
def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="happo",
        choices=[
            "happo",
            "hatrpo",
            "haa2c",
            "haddpg",
            "hatd3",
            "hasac",
            "had3qn",
            "maddpg",
            "matd3",
            "mappo",
        ],
        help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, matd3, mappo.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="pettingzoo_mpe",
        choices=[
            "smac",
            "mamujoco",
            "pettingzoo_mpe",
            "gym",
            "football",
            "dexhands",
            "smacv2",
            "lag",
        ],
        help="Environment name. Choose from: smac, mamujoco, pettingzoo_mpe, gym, football, dexhands, smacv2, lag.",
    )
    parser.add_argument(
        "--exp_name", type=str, default="installtest", help="Experiment name."
    )
    parser.add_argument(
        "--attack_id", type=int, default=0, help="Agent ID to attack."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    parser.add_argument(
        "--reward",
        type=float,
        default=-79.879,
        help="Reward value to restore the model."
    )
    parser.add_argument(
        "--filepath",
        type=str,
        default="/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/hatrpo/Latest_3/seed-00001-2025-08-15-22-56-55/models",
        help="Filepath to restore the model from."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed."
    )
    parser.add_argument(
        "--taylor_csv_agent0", type=str, default="", help="Path to CSV file with pre-computed Taylor history for agent 0."
    )
    parser.add_argument(
        "--taylor_csv_agent1", type=str, default="", help="Path to CSV file with pre-computed Taylor history for agent 1."
    )
    parser.add_argument(
        "--taylor_csv_agent2", type=str, default="", help="Path to CSV file with pre-computed Taylor history for agent 2."
    )
    parser.add_argument(
        "--save_dir", type=str, default=None, help="Directory to save results."
    )
    parser.add_argument(
        "--min_window", type=int, default=0, help="Minimum timestep for attack window (inclusive)."
    )
    parser.add_argument(
        "--max_window", type=int, default=None, help="Maximum timestep for attack window (inclusive). If None, attacks until episode ends."
    )
    parser.add_argument(
        "--top_k", type=int, default=2, help="Top-k agents to consider for threat detection based on Frobenius values."
    )
    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict
    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:  # load config from corresponding yaml file
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])
    update_args(unparsed_dict, algo_args, env_args)  # update args from command line

    if args["env"] == "dexhands":
        import isaacgym  # isaacgym has to be imported before PyTorch

    # note: isaac gym does not support multiple instances, thus cannot eval separately
    if args["env"] == "dexhands":
        algo_args["eval"]["use_eval"] = False
        algo_args["train"]["episode_length"] = env_args["hands_episode_length"]

    # start training
    from harl.runners import RUNNER_REGISTRY
# 
    algo_args['eval']['n_eval_rollout_threads'] = 1
    algo_args['eval']['eval_episodes'] = 1
    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    restore(runner, args['filepath'],args['reward'])  # Restore the model with specific reward and episode
    runner.prep_training()
    
    attack_agent_id = args['attack_id']
    print(f"=== Evaluating with attack on agent {attack_agent_id} ===")
    print(f"=== Attack window: timesteps {args['min_window']} to {args['max_window'] if args['max_window'] is not None else 'end'} ===")
    
    # Load pre-computed Taylor history CSV files
    csv_paths = [
        args.get('taylor_csv_agent0', ''),
        args.get('taylor_csv_agent1', ''), 
        args.get('taylor_csv_agent2', '')
    ]
    taylor_history_data = load_taylor_history_csvs(csv_paths)
    
    # Run evaluation without attack
    # results_normal, frob_norms_normal, sec_dir_derivatives_normal, frob_norms_matrix_history_normal, fault_timeline_normal, attacked_steps_normal = eval(runner, False, attack_agent_id, taylor_history_data=taylor_history_data)
    # Run evaluation with attack
    results__normal, pairwise_frobs_history_normal, pairwise_obs_influences_history_normal, pairwise_second_order_influences_history_normal, frob_norms_matrix_history_atk_normal, fault_timeline_atk_normal, attacked_steps_atk_normal, threat_data_normal, threat_data_faulty_topk_normal = eval(
        runner, 
        attack_status=False, 
        attack_agent_id=attack_agent_id, 
        seed=args['seed'], 
        taylor_history_data=taylor_history_data,
        min_window=args['min_window'],
        max_window=args['max_window'],
        top_k=args['top_k']
    )
        
    # exit("Exiting")
    results_attacked, pairwise_frobs_history, pairwise_obs_influences_history, pairwise_second_order_influences_history, frob_norms_matrix_history_atk, fault_timeline_atk, attacked_steps_atk, threat_data_attacked, threat_data_faulty_topk = eval(
        runner, 
        attack_status=True, 
        attack_agent_id=attack_agent_id, 
        seed=args['seed'], 
        taylor_history_data=taylor_history_data,
        min_window=args['min_window'],
        max_window=args['max_window'],
        top_k=args['top_k']
    )

    log_dir = algo_args['attack']['log_dir']
    alg_name = algo_args['attack']['algo_name']
    date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    # Include window information in the log path
    window_str = f"window_{args['min_window']}_to_{args['max_window'] if args['max_window'] is not None else 'end'}"
    log_path = os.path.join(args['save_dir'], window_str, str(args['attack_id']), date)
    # if args['save_dir'] is not None:
    #     log_path = os.path.join(args['save_dir'], alg_name, date)
    # else:
    #     log_path = os.path.join(log_dir, alg_name, date)
    os.makedirs(log_path, exist_ok=True)

    # plot_results(results_normal, results_attacked, atk_agent_id=attack_agent_id, logdir=log_path)
    # plot_frobs(frob_norms_normal, frob_norms_atk, attacked_steps_atk, attack_agent_id, log_path)
    # plot_sec_dir_derivatives(sec_dir_derivatives_normal, sec_dir_derivatives_atk, attacked_steps_atk, attack_agent_id, log_path)
    plot_pairwise_observation_influences(pairwise_obs_influences_history_normal,pairwise_obs_influences_history, attacked_steps_atk, attack_agent_id, log_path)
    plot_second_order_observation_influences(pairwise_second_order_influences_history, attacked_steps_atk, attack_agent_id, log_path)
    plot_pairwise_frobs_comparison(pairwise_frobs_history_normal, pairwise_frobs_history, attacked_steps_atk, attack_agent_id, log_path)
    # Save matrices and include attacked steps for attacked run
    save_matrix_to_files(results_attacked, attacked_steps_atk, attack_agent_id, runner.num_agents, log_path, f'happo_taylor_error_atk_{attack_agent_id}_{window_str}.csv')
    # save_matrix_to_files(frob_norms_atk, attacked_steps_atk, attack_agent_id, runner.num_agents, log_path, f'happo_frobenius_norms_atk_{attack_agent_id}.csv')
    # save_matrix_to_files(sec_dir_derivatives_atk, attacked_steps_atk, attack_agent_id, runner.num_agents, log_path, f'happo_sec_dir_derivatives_atk_{attack_agent_id}.csv')

    # Example usage of observation influence plotting functions
    # You can uncomment and modify these to visualize observation influences
    
    # # Compute pairwise observation influences for a specific step
    # step_idx = len(attacked_steps_atk) // 2  # Use middle step as example
    # if step_idx < len(obs_attacked):
    #     obs_step = obs_attacked[step_idx]
    #     pairwise_influences = compute_pairwise_observation_influences_happo(runner, obs_step)
    #     plot_pairwise_observation_influences(pairwise_influences, runner.num_agents, 
    #                                        f'Pairwise Observation Influences - Step {attacked_steps_atk[step_idx]}',
    #                                        log_path, f'pairwise_influences_step_{attacked_steps_atk[step_idx]}.png')
    
    # # Compute and plot second-order influences
    # if step_idx < len(obs_attacked):
    #     second_order_influences = compute_second_order_observation_influences_happo(runner, obs_step)
    #     plot_second_order_observation_influences(second_order_influences, runner.num_agents,
    #                                             f'Second-Order Observation Influences - Step {attacked_steps_atk[step_idx]}',
    #                                             log_path, f'second_order_influences_step_{attacked_steps_atk[step_idx]}.png')

    # Plot fault timeline and contributor charts for attacked run
    plot_fault_timeline(fault_timeline_atk, runner.num_agents, log_path)
    plot_contributor_barchart(fault_timeline_atk, runner.num_agents, log_path)
    
    # Plot Taylor error vs historical bounds
    plot_taylor_error_with_historical_bounds(results_attacked, taylor_history_data, attacked_steps_atk, attack_agent_id, log_path)

    # === NEW THREAT DETECTION ANALYSIS ===
    # Save threat detection results to CSV
    save_threat_data_csv(threat_data_attacked, attack_agent_id, args['top_k'], log_path, window_str)
    
    # Plot combined threat analysis visualization
    plot_threat_analysis_combined(
        fault_timeline_atk, 
        threat_data_attacked, 
        pairwise_frobs_history, 
        attack_agent_id, 
        runner.num_agents, 
        args['top_k'], 
        log_path
    )
    
    # === NEW FAULTY AGENTS IN TOP-K ANALYSIS ===
    # Save faulty agents top-k analysis results to CSV
    save_threat_data_csv_faulty_topk(threat_data_faulty_topk, args['top_k'], log_path)
    
    # Plot faulty agents top-k timeline visualization
    plot_faulty_agents_topk_timeline(
        threat_data_faulty_topk, 
        pairwise_frobs_history, 
        runner.num_agents, 
        args['top_k'], 
        log_path
    )
    
    # Print threat detection summary
    if threat_data_attacked['threat_timesteps']:
        print(f"\n=== THREAT DETECTION SUMMARY ===")
        print(f"Top-{args['top_k']} Analysis: Agent {attack_agent_id} detected as potential threat")
        print(f"Threat detected at timesteps: {sorted(list(threat_data_attacked['threat_timesteps']))}")
        print(f"Total unique threat timesteps: {len(threat_data_attacked['threat_timesteps'])}")
        
        # Show some example threat instances
        threat_instances = [(i, threat_data_attacked['timestamp'][i], threat_data_attacked['agent_id'][i], threat_data_attacked['attacked_agent_rank'][i]) 
                          for i in range(len(threat_data_attacked['is_threat'])) if threat_data_attacked['is_threat'][i]]
        
        print(f"Sample threat instances (first 5):")
        for i, (idx, t, agent, rank) in enumerate(threat_instances[:5]):
            print(f"  - Timestep {t}: Agent {attack_agent_id} ranked #{rank} in top-{args['top_k']} influencers of Agent {agent}")
    else:
        print(f"\n=== THREAT DETECTION SUMMARY ===")
        print(f"Top-{args['top_k']} Analysis: No threats detected for Agent {attack_agent_id}")

    # runner.run()
    runner.close()



if __name__ == "__main__":
    main()
