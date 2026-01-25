"""
Fault detection and analysis functions.
"""
import math


def get_patient_zero_detection(fault_timeline):
    """
    Return the patient zero agent ids and detection timestep from the fault timeline.
    Handles multiple agents detected at the same earliest time.
    
    Args:
        fault_timeline: List of fault detection events
        
    Returns:
        Tuple of (agent_ids_list, timestep) for the earliest fault detection.
        agent_ids_list will contain all agents detected at the earliest time.
        Returns (None, None) if no fault timeline exists.
    """
    if not fault_timeline:
        return None, None

    # Find the earliest timestep
    earliest_time = min(
        event.get('t', float('inf')) for event in fault_timeline
    )
    
    # Find all agents detected at the earliest time
    patient_zero_agents = [
        event.get('agent') for event in fault_timeline 
        if event.get('t') == earliest_time
    ]
    
    return patient_zero_agents, earliest_time


def compute_decayed_action_influence(action_influences_matrix_history, patient_zero_time, lambda_decay):
    """
    Compute cumulative decayed pairwise influences after patient zero detection.
    
    Args:
        action_influences_matrix_history: List of N x N action influence matrices over time
        patient_zero_time: Timestep when patient zero was detected
        lambda_decay: Exponential decay factor
        
    Returns:
        3D tensor results[i][j][t] representing cumulative decayed influences
    """
    if not action_influences_matrix_history or patient_zero_time is None:
        return []

    num_timesteps = len(action_influences_matrix_history)
    num_agents = len(action_influences_matrix_history[0])

    patient_zero_index = int(patient_zero_time)
    if patient_zero_index < 0:
        patient_zero_index = 0

    if patient_zero_index >= num_timesteps:
        return [[[0.0 for _ in range(num_timesteps)] for _ in range(num_agents)] for _ in range(num_agents)]

    results = [[[0.0 for _ in range(num_timesteps)] for _ in range(num_agents)] for _ in range(num_agents)]
    cumulative_influence = [[0.0 for _ in range(num_agents)] for _ in range(num_agents)]

    for t in range(patient_zero_index, num_timesteps):
        decay_weight = math.exp(-lambda_decay * (t - patient_zero_index))

        for i in range(num_agents):
            for j in range(num_agents):
                if j == i:
                    continue

                increment = abs(action_influences_matrix_history[t][i][j]) * decay_weight
                cumulative_influence[i][j] += increment
                results[i][j][t] = cumulative_influence[i][j]

    return results