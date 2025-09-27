"""
Fault detection and analysis functions.
"""
import math


def get_patient_zero_detection(fault_timeline):
    """
    Return the patient zero agent id and detection timestep from the fault timeline.
    
    Args:
        fault_timeline: List of fault detection events
        
    Returns:
        Tuple of (agent_id, timestep) for the earliest fault detection
    """
    if not fault_timeline:
        return None, None

    earliest_event = min(
        fault_timeline,
        key=lambda event: event.get('t', float('inf'))
    )
    return earliest_event.get('agent'), earliest_event.get('t')


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