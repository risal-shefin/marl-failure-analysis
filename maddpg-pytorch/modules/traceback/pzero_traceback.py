"""
Patient Zero Traceback Module

This module implements the patient zero traceback algorithm to find the true source
of influence when the initially detected patient zero is incorrect.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import math


def select_agent_max_taylor_deviation(initial_patient_zero: List[int], 
                                    taylor_errors_history: List[Dict],
                                    ref_vals: List[List[float]], 
                                    detection_time: int) -> int:
    """
    Select the agent with maximum Taylor deviation from a list of candidate patient zeros.
    
    Args:
        initial_patient_zero: List of candidate patient zero agent IDs
        taylor_errors_history: History of Taylor errors for all timesteps
        ref_vals: Reference Taylor error values for each agent and timestep
        ref_std_devs: Reference standard deviations for each agent and timestep
        detection_time: Time when fault was detected
        
    Returns:
        Agent ID with maximum Taylor deviation
    """
    if len(initial_patient_zero) == 1:
        return initial_patient_zero[0]
    
    max_deviation = -math.inf
    selected_agent = initial_patient_zero[0]
    
    for agent_id in initial_patient_zero:
        current_error = taylor_errors_history[detection_time][agent_id]
        deviation = abs(current_error - ref_vals[agent_id][detection_time])
        
        if deviation > max_deviation:
            max_deviation = deviation
            selected_agent = agent_id
    
    return selected_agent


def compute_critical_rate(other_agent: int, 
                            current_agent: int, 
                            directional_derivative_history: List[np.ndarray],
                            detection_time: int,
                            window_size: int = 5) -> float:
    """
    Compute the critical rate from other_agent to current_agent
    in a window before detection time.
    
    Args:
        other_agent: ID of the potentially influencing agent
        current_agent: ID of the agent being influenced
        directional_derivative_history: History of directional derivative matrices
        detection_time: Time when fault was detected
        window_size: Size of the window to look back from detection time
        
    Returns:
        Rate of critical zone (positive D_ij) from other_agent to current_agent
    """
    
    # Define the window
    start_time = max(0, detection_time - window_size + 1)
    end_time = detection_time
    
    positive_count = 0
    total_count = 0
    
    for t in range(start_time, end_time+1):
        # directional_derivative_matrix[j][i] = directional derivative of influence of of i on j
        D_ij = directional_derivative_history[t][current_agent][other_agent]
        total_count += 1
        if D_ij > 0:
            positive_count += 1
    
    if total_count == 0:
        return 0.0
    
    return positive_count / total_count


def update_most_influential(current_most_influential: Optional[Dict],
                          candidate_agent: int,
                          critical_rate: float,
                          score: float = 0.0) -> Dict:
    """
    Update the most influential agent based on score.

    Args:
        current_most_influential: Current most influential agent info
        candidate_agent: Candidate agent ID
        dij_rate: Positive influence rate
        tie_breaking_score: Score for scoring (higher is better)
        
    Returns:
        Updated most influential agent info
    """
    candidate_info = {
        'agent': candidate_agent,
        'critical_rate': critical_rate,
        'score': score
    }
    if current_most_influential is None:
        return candidate_info

    if candidate_info['score'] > current_most_influential['score']:
        return candidate_info
    
    return current_most_influential


def trace_back_influence_chain(current_agent: int,
                             chain: List[int],
                             all_agents: List[int],
                             directional_derivative_history: List[np.ndarray],
                             detection_time: int,
                             action_influences_history: Optional[List[np.ndarray]],
                             taylor_errors_history: Optional[List[Dict]] = None,
                             ref_vals: Optional[List[List[float]]] = None,
                             use_taylor_scoring: bool = False,
                             window_size: int = 5) -> List[int]:
    """
    Recursively trace back the influence chain to find the true patient zero.
    
    Args:
        current_agent: Current agent in the chain
        chain: Current influence chain
        all_agents: List of all agent IDs
        directional_derivative_history: History of directional derivative matrices
        detection_time: Time when fault was detected
        action_influences_history: History of action influence matrices
        taylor_errors_history: History of Taylor errors (required for Taylor scoring)
        ref_vals: Reference Taylor error values (required for Taylor scoring)
        use_taylor_scoring: If True, use Taylor deviation for scoring instead of influence
        
    Returns:
        Complete influence chain from true patient zero to detected agent
    """
    
    most_influential = None
    
    for other_agent in all_agents:
        if other_agent == current_agent:
            continue  # Skip self-comparison
        
        # Compute critical influence rate from other_agent to current_agent
        critical_rate = compute_critical_rate(
            other_agent, current_agent, directional_derivative_history, detection_time
        )
        
        start_time = max(0, detection_time - window_size + 1)
        end_time = detection_time
        score = 0.0
    
        # Determine scoring score
        if use_taylor_scoring:
            # Use Taylor deviation for scoring
            for t in range(start_time, end_time+1):
                if directional_derivative_history[t][current_agent][other_agent] <= 0:
                    continue
                current_error = taylor_errors_history[detection_time][other_agent]
                score += abs(current_error - ref_vals[other_agent][detection_time])
        else:
            # Use influence score for scoring (default)
            tie_breaking_score = 0.0
            for t in range(start_time, end_time+1):
                if directional_derivative_history[t][current_agent][other_agent] <= 0:
                    continue
                score += action_influences_history[detection_time][current_agent][other_agent]
        
        # Update most influential agent
        most_influential = update_most_influential(
            most_influential, other_agent, critical_rate, score
        )
    
    # Stop if no influential agent found or cycle detected
    if (most_influential is None or
        most_influential['critical_rate'] <= 0 or
        most_influential['agent'] in chain):
        return chain
    
    # Add to influence chain and continue tracing
    chain.append(most_influential['agent'])
    return trace_back_influence_chain(
        most_influential['agent'], chain, all_agents, 
        directional_derivative_history, detection_time, action_influences_history,
        taylor_errors_history, ref_vals, use_taylor_scoring
    )


def perform_patient_zero_traceback(fault_timeline: List[Dict],
                                 directional_derivative_history: List[np.ndarray],
                                 taylor_errors_history: List[Dict],
                                 ref_vals: List[List[float]],
                                 all_agents: List[int],
                                 action_influences_history: Optional[List[np.ndarray]] = None,
                                 use_taylor_scoring: bool = False) -> Tuple[int, List[int], int]:
    """
    Main function to perform patient zero traceback.
    
    Args:
        fault_timeline: List of fault detection events
        directional_derivative_history: History of directional derivative matrices
        taylor_errors_history: History of Taylor errors
        ref_vals: Reference Taylor error values
        all_agents: List of all agent IDs
        action_influences_history: History of action influence matrices
        use_taylor_scoring: If True, use Taylor deviation for scoring instead of influence
        
    Returns:
        Tuple of (true_patient_zero, influence_chain, detection_time)
    """
    if not fault_timeline:
        return None, [], None
    
    # Step 1: Get initial patient zero detection
    from modules.detection import get_patient_zero_detection
    initial_patient_zeros, detection_time = get_patient_zero_detection(fault_timeline)
    
    if not initial_patient_zeros or detection_time is None:
        return None, [], None
    
    # Step 2: Handle multiple patient zeros by selecting one with max Taylor deviation
    if len(initial_patient_zeros) > 1:
        initial_patient_zero = select_agent_max_taylor_deviation(
            initial_patient_zeros, taylor_errors_history, ref_vals, detection_time
        )
    else:
        initial_patient_zero = initial_patient_zeros[0]
    
    # Step 3: Begin traceback from the detected agent
    influence_chain = trace_back_influence_chain(
        initial_patient_zero, [initial_patient_zero], all_agents,
        directional_derivative_history, detection_time, action_influences_history,
        taylor_errors_history, ref_vals, use_taylor_scoring
    )
    
    # Step 4: Finalize the true patient zero
    # Reverse the chain to get influence from the source to initially detected agent
    influence_chain = list(reversed(influence_chain))
    true_patient_zero = influence_chain[0] if influence_chain else initial_patient_zero
    
    return true_patient_zero, influence_chain, detection_time
