"""Fault detection utilities reused for MAPPO analysis."""
import math


def get_patient_zero_detection(fault_timeline):
    if not fault_timeline:
        return None, None

    earliest_time = min(event.get('t', math.inf) for event in fault_timeline)
    patient_zero_agents = [
        event.get('agent') for event in fault_timeline
        if event.get('t') == earliest_time
    ]
    return patient_zero_agents, earliest_time


def compute_decayed_action_influence(action_influences_matrix_history, patient_zero_time, lambda_decay):
    if not action_influences_matrix_history or patient_zero_time is None:
        return []

    num_timesteps = len(action_influences_matrix_history)
    num_agents = len(action_influences_matrix_history[0])
    patient_zero_index = max(0, int(patient_zero_time))

    results = [[[0.0 for _ in range(num_timesteps)] for _ in range(num_agents)] for _ in range(num_agents)]
    cumulative_influence = [[0.0 for _ in range(num_agents)] for _ in range(num_agents)]

    for t in range(patient_zero_index, num_timesteps):
        decay_weight = math.exp(-lambda_decay * (t - patient_zero_index))
        for i in range(num_agents):
            for j in range(num_agents):
                if i == j:
                    continue
                increment = abs(action_influences_matrix_history[t][i][j]) * decay_weight
                cumulative_influence[i][j] += increment
                results[i][j][t] = cumulative_influence[i][j]
    return results
