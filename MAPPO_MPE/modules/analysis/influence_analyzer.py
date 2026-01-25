"""
Influence analysis and timestep selection for attacks.
"""
import numpy as np


class InfluenceAnalyzer:
    """
    Analyzes action influences and directional derivatives to find optimal attack timesteps.
    """
    
    @staticmethod
    def find_influence_timesteps(action_influences_history, directional_derivatives_history, 
                                agent_i, agent_j, atk_steps_limit, k_steps=1):
        """
        Find max and min influence timesteps of agent i on agent j in first portion of episode.
        Uses directional second derivatives to filter timesteps:
        - High influence: positive directional second derivative + maximum action influence
        - Low influence: negative directional second derivative + minimum action influence
        
        Args:
            action_influences_history: List of action influence matrices
            directional_derivatives_history: List of directional second derivative matrices
            agent_i: Index of influencing agent
            agent_j: Index of influenced agent (where action_influences_matrix[t][j][i] = influence of i on j)
            atk_steps_limit: Last step that can be attacked
            k_steps: Number of timesteps to return (currently expecting 1)
            
        Returns:
            Tuple of (max_influence_timesteps, min_influence_timesteps)
        """
        positive_derivative_timesteps = []  # For high influence selection
        negative_derivative_timesteps = []  # For low influence selection
        
        for t in range(min(atk_steps_limit, len(action_influences_history), 
                          len(directional_derivatives_history))):
            directional_derivative = directional_derivatives_history[t][agent_j][agent_i]
            
            # Separate timesteps based on directional derivative sign
            if directional_derivative > 0:
                positive_derivative_timesteps.append(t)
            elif directional_derivative <= 0:
                negative_derivative_timesteps.append(t)
        
        # For high influence: among positive derivative timesteps, choose maximum action influence
        max_influences_t = []
        if positive_derivative_timesteps:
            max_influences_t = sorted(
                positive_derivative_timesteps,
                key=lambda t: action_influences_history[t][agent_j][agent_i],
                reverse=True
            )[:k_steps]
        
        # For low influence: among negative derivative timesteps, choose minimum action influence
        min_influences_t = []
        if negative_derivative_timesteps:
            min_influences_t = sorted(
                negative_derivative_timesteps,
                key=lambda t: action_influences_history[t][agent_j][agent_i]
            )[:k_steps]
        
        max_influences_t.sort()
        min_influences_t.sort()

        return max_influences_t, min_influences_t
    
    @staticmethod
    def get_agent_fault_detection_times(fault_timeline, agent_id):
        """
        Get all fault detection times for a specific agent from the fault timeline.
        
        Args:
            fault_timeline: List of fault detection events
            agent_id: ID of the agent to get detection times for
            
        Returns:
            List of detection times for the specified agent (empty list if no detections)
        """
        if not fault_timeline:
            return []
        
        detection_times = []
        for event in fault_timeline:
            if event.get('agent') == agent_id:
                detection_times.append(event.get('t'))
        
        return sorted(detection_times)  # Sort chronologically
