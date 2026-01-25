"""
Patient Zero Traceback Module

This module implements the patient zero detection and influence chain traceback algorithm
following the pseudocode specifications. It provides functions to:
1. Detect initial patient zero agents
2. Select agent with maximum Taylor deviation for tie-breaking
3. Trace back the influence chain to find the true patient zero
4. Log all results to CSV files with seed information

The module can be imported and used in other files for patient zero analysis.
"""

import os
import csv
import math
from typing import List, Tuple, Dict, Optional, Any
from datetime import datetime
from .detection import get_patient_zero_detection


class PatientZeroTracebackAnalyzer:
    """
    Analyzer class for patient zero detection and influence chain traceback.
    """
    
    def __init__(self, log_dir: str = None, lambda_decay: float = 0.1):
        """
        Initialize the analyzer.
        
        Args:
            log_dir: Directory to save CSV logs (if None, uses current directory)
            lambda_decay: Exponential decay factor for computing influence rates
        """
        self.log_dir = log_dir or os.getcwd()
        self.lambda_decay = lambda_decay
        self.traceback_results = []
        self.influence_chain_data = []
        
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize CSV files
        self._initialize_csv_files()
    
    def _initialize_csv_files(self):
        """Initialize CSV files with headers."""
        # Traceback results CSV
        self.traceback_csv_path = os.path.join(self.log_dir, 'patient_zero_traceback_results.csv')
        traceback_headers = [
            'seed', 'episode_length', 'num_agents', 'initial_patient_zeros', 
            'initial_detection_time', 'selected_patient_zero', 'true_patient_zero',
            'influence_chain_length', 'influence_chain', 'cycle_detected',
            'max_taylor_deviation', 'analysis_timestamp'
        ]
        
        # Influence chain details CSV
        self.influence_chain_csv_path = os.path.join(self.log_dir, 'influence_chain_details.csv')
        chain_headers = [
            'seed', 'chain_step', 'current_agent', 'most_influential_agent',
            'dij_rate', 'gij_norm', 'timestep_window_start', 'timestep_window_end',
            'analysis_timestamp'
        ]
        
        # Write headers if files don't exist
        if not os.path.exists(self.traceback_csv_path):
            with open(self.traceback_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(traceback_headers)
        
        if not os.path.exists(self.influence_chain_csv_path):
            with open(self.influence_chain_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(chain_headers)
    
    def select_agent_max_taylor_deviation(self, 
                                        initial_patient_zeros: List[int], 
                                        taylor_history: List[List[float]], 
                                        detection_time: int) -> int:
        """
        Select the agent with maximum Taylor deviation for tie-breaking.
        
        Args:
            initial_patient_zeros: List of agent IDs detected at the same time
            taylor_history: List of Taylor error histories for each agent [agent_id][timestep]
            detection_time: Timestep when patient zero was detected
            
        Returns:
            Agent ID with maximum Taylor deviation at detection time
        """
        if len(initial_patient_zeros) == 1:
            return initial_patient_zeros[0]
        
        max_deviation = -1
        selected_agent = initial_patient_zeros[0]
        
        for agent_id in initial_patient_zeros:
            if (agent_id < len(taylor_history) and 
                detection_time < len(taylor_history[agent_id])):
                
                deviation = abs(taylor_history[agent_id][detection_time])
                if deviation > max_deviation:
                    max_deviation = deviation
                    selected_agent = agent_id
        
        return selected_agent
    
    def compute_positive_dij_rate(self, 
                                other_agent: int, 
                                current_agent: int, 
                                action_influences_history: List[List[List[float]]], 
                                timestep_window: Tuple[int, int]) -> float:
        """
        Compute positive Dij rate (influence rate) from other_agent to current_agent.
        
        Args:
            other_agent: ID of the influencing agent
            current_agent: ID of the influenced agent
            action_influences_history: List of action influence matrices [timestep][influenced][influencer]
            timestep_window: Tuple of (start_timestep, end_timestep) for computation window
            
        Returns:
            Positive Dij rate value
        """
        start_t, end_t = timestep_window
        total_positive_influence = 0.0
        count = 0
        
        for t in range(start_t, min(end_t + 1, len(action_influences_history))):
            if (t < len(action_influences_history) and 
                current_agent < len(action_influences_history[t]) and
                other_agent < len(action_influences_history[t][current_agent])):
                
                # Get influence from other_agent to current_agent
                influence = action_influences_history[t][current_agent][other_agent]
                
                # Only consider positive influences and apply decay
                if influence > 0:
                    decay_weight = math.exp(-self.lambda_decay * (t - start_t))
                    total_positive_influence += influence * decay_weight
                    count += 1
        
        # Return rate (average positive influence per timestep)
        return total_positive_influence / max(count, 1)
    
    def compute_gij_norm(self, 
                        other_agent: int, 
                        current_agent: int, 
                        pairwise_frob_norms_history: List[List[List[float]]], 
                        timestep_window: Tuple[int, int]) -> float:
        """
        Compute G_ij norm (Frobenius norm) from other_agent to current_agent.
        
        Args:
            other_agent: ID of the influencing agent
            current_agent: ID of the influenced agent
            pairwise_frob_norms_history: List of pairwise Frobenius norm matrices
            timestep_window: Tuple of (start_timestep, end_timestep) for computation window
            
        Returns:
            Average G_ij norm value over the window
        """
        start_t, end_t = timestep_window
        total_norm = 0.0
        count = 0
        
        for t in range(start_t, min(end_t + 1, len(pairwise_frob_norms_history))):
            if (t < len(pairwise_frob_norms_history) and 
                current_agent < len(pairwise_frob_norms_history[t]) and
                other_agent < len(pairwise_frob_norms_history[t][current_agent])):
                
                norm_value = pairwise_frob_norms_history[t][current_agent][other_agent]
                total_norm += norm_value
                count += 1
        
        return total_norm / max(count, 1)
    
    def update_most_influential(self, 
                              current_most_influential: Optional[Dict], 
                              candidate_agent: int, 
                              dij_rate: float, 
                              gij_norm: float) -> Dict:
        """
        Update the most influential agent using Dij_rate as primary metric and G_ij norm for tie-breaking.
        
        Args:
            current_most_influential: Current most influential agent info (None if first)
            candidate_agent: Candidate agent ID
            dij_rate: Dij rate for candidate agent
            gij_norm: G_ij norm for candidate agent (tie-breaker)
            
        Returns:
            Dictionary with most influential agent information
        """
        candidate_info = {
            'agent_id': candidate_agent,
            'dij_rate': dij_rate,
            'gij_norm': gij_norm
        }
        
        if current_most_influential is None:
            return candidate_info
        
        # Primary criterion: higher Dij_rate
        if dij_rate > current_most_influential['dij_rate']:
            return candidate_info
        elif dij_rate == current_most_influential['dij_rate']:
            # Tie-breaker: higher G_ij norm
            if gij_norm > current_most_influential['gij_norm']:
                return candidate_info
        
        return current_most_influential
    
    def trace_back_influence_chain(self, 
                                 current_agent: int, 
                                 chain: List[int], 
                                 all_agents: List[int],
                                 action_influences_history: List[List[List[float]]], 
                                 pairwise_frob_norms_history: List[List[List[float]]], 
                                 detection_time: int, 
                                 seed: int,
                                 window_size: int = 5) -> List[int]:
        """
        Recursive traceback function to find the influence chain.
        
        Args:
            current_agent: Current agent being analyzed
            chain: Current influence chain
            all_agents: List of all agent IDs
            action_influences_history: Action influence matrices over time
            pairwise_frob_norms_history: Frobenius norm matrices over time
            detection_time: Time when patient zero was detected
            seed: Random seed for logging
            window_size: Size of timestep window for analysis
            
        Returns:
            Complete influence chain (will be reversed later)
        """
        most_influential = None
        
        # Define timestep window for analysis (before detection time)
        window_start = max(0, detection_time - window_size)
        window_end = detection_time
        timestep_window = (window_start, window_end)
        
        for other_agent in all_agents:
            if other_agent == current_agent:
                continue  # Skip self-comparison
            
            # Compute influence from other_agent to current_agent
            dij_rate = self.compute_positive_dij_rate(
                other_agent, current_agent, action_influences_history, timestep_window
            )
            
            # Compute G_ij norm for tie-breaking
            gij_norm = self.compute_gij_norm(
                other_agent, current_agent, pairwise_frob_norms_history, timestep_window
            )
            
            # Update most influential agent
            most_influential = self.update_most_influential(
                most_influential, other_agent, dij_rate, gij_norm
            )
            
            # Log influence chain details
            self.influence_chain_data.append({
                'seed': seed,
                'chain_step': len(chain),
                'current_agent': current_agent,
                'candidate_agent': other_agent,
                'dij_rate': dij_rate,
                'gij_norm': gij_norm,
                'timestep_window_start': window_start,
                'timestep_window_end': window_end,
                'analysis_timestamp': datetime.now().isoformat()
            })
        
        # Stop if no influential agent found or cycle detected
        if most_influential is None or most_influential['agent_id'] in chain:
            return chain
        
        # Add most influential agent to chain and continue tracing
        chain.append(most_influential['agent_id'])
        
        # Log the selection for this step
        selected_agent_data = next(
            (data for data in self.influence_chain_data 
             if data['seed'] == seed and 
                data['chain_step'] == len(chain) - 1 and
                data['candidate_agent'] == most_influential['agent_id']),
            None
        )
        if selected_agent_data:
            selected_agent_data['selected_as_most_influential'] = True
        
        return self.trace_back_influence_chain(
            most_influential['agent_id'], chain, all_agents, 
            action_influences_history, pairwise_frob_norms_history, 
            detection_time, seed, window_size
        )
    
    def analyze_patient_zero_traceback(self,
                                     fault_timeline: List[Dict],
                                     action_influences_history: List[List[List[float]]],
                                     pairwise_frob_norms_history: List[List[List[float]]],
                                     taylor_history: List[List[float]],
                                     num_agents: int,
                                     seed: int,
                                     episode_length: int) -> Dict:
        """
        Main function to analyze patient zero traceback following the pseudocode.
        
        Args:
            fault_timeline: List of fault detection events
            action_influences_history: Action influence matrices over time
            pairwise_frob_norms_history: Frobenius norm matrices over time
            taylor_history: Taylor error histories for each agent
            num_agents: Number of agents in the system
            seed: Random seed for this episode
            episode_length: Length of the episode
            
        Returns:
            Dictionary with traceback analysis results
        """
        analysis_timestamp = datetime.now().isoformat()
        
        # Step 1: Initialization
        initial_patient_zeros, detection_time = get_patient_zero_detection(fault_timeline)
        
        if initial_patient_zeros is None or detection_time is None:
            # No patient zero detected
            result = {
                'seed': seed,
                'episode_length': episode_length,
                'num_agents': num_agents,
                'initial_patient_zeros': None,
                'initial_detection_time': None,
                'selected_patient_zero': None,
                'true_patient_zero': None,
                'influence_chain_length': 0,
                'influence_chain': [],
                'cycle_detected': False,
                'max_taylor_deviation': 0.0,
                'analysis_timestamp': analysis_timestamp
            }
            self.traceback_results.append(result)
            return result
        
        # Handle tie-breaking if multiple agents detected at same time
        selected_patient_zero = initial_patient_zeros[0]
        max_taylor_deviation = 0.0
        
        if len(initial_patient_zeros) > 1:
            selected_patient_zero = self.select_agent_max_taylor_deviation(
                initial_patient_zeros, taylor_history, detection_time
            )
            
            # Get max Taylor deviation for logging
            if (selected_patient_zero < len(taylor_history) and 
                detection_time < len(taylor_history[selected_patient_zero])):
                max_taylor_deviation = abs(taylor_history[selected_patient_zero][detection_time])
        else:
            if (selected_patient_zero < len(taylor_history) and 
                detection_time < len(taylor_history[selected_patient_zero])):
                max_taylor_deviation = abs(taylor_history[selected_patient_zero][detection_time])
        
        # Step 2: Begin traceback from the detected agent
        all_agents = list(range(num_agents))
        initial_chain = [selected_patient_zero]
        
        complete_chain = self.trace_back_influence_chain(
            selected_patient_zero, initial_chain, all_agents,
            action_influences_history, pairwise_frob_norms_history,
            detection_time, seed
        )
        
        # Step 3: Finalize the True Patient Zero
        influence_chain = list(reversed(complete_chain))
        true_patient_zero = influence_chain[0] if influence_chain else selected_patient_zero
        
        # Check if cycle was detected
        cycle_detected = len(set(complete_chain)) < len(complete_chain)
        
        # Create result dictionary
        result = {
            'seed': seed,
            'episode_length': episode_length,
            'num_agents': num_agents,
            'initial_patient_zeros': initial_patient_zeros,
            'initial_detection_time': detection_time,
            'selected_patient_zero': selected_patient_zero,
            'true_patient_zero': true_patient_zero,
            'influence_chain_length': len(influence_chain),
            'influence_chain': influence_chain,
            'cycle_detected': cycle_detected,
            'max_taylor_deviation': max_taylor_deviation,
            'analysis_timestamp': analysis_timestamp
        }
        
        self.traceback_results.append(result)
        return result
    
    def save_results_to_csv(self):
        """Save all accumulated results to CSV files."""
        # Save traceback results
        if self.traceback_results:
            with open(self.traceback_csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'seed', 'episode_length', 'num_agents', 'initial_patient_zeros', 
                    'initial_detection_time', 'selected_patient_zero', 'true_patient_zero',
                    'influence_chain_length', 'influence_chain', 'cycle_detected',
                    'max_taylor_deviation', 'analysis_timestamp'
                ])
                
                for result in self.traceback_results:
                    # Convert lists to string representation for CSV
                    result_copy = result.copy()
                    result_copy['initial_patient_zeros'] = str(result['initial_patient_zeros'])
                    result_copy['influence_chain'] = str(result['influence_chain'])
                    writer.writerow(result_copy)
        
        # Save influence chain details
        if self.influence_chain_data:
            with open(self.influence_chain_csv_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'seed', 'chain_step', 'current_agent', 'candidate_agent',
                    'dij_rate', 'gij_norm', 'timestep_window_start', 'timestep_window_end',
                    'selected_as_most_influential', 'analysis_timestamp'
                ])
                
                for data in self.influence_chain_data:
                    # Add default value for selected_as_most_influential if not present
                    if 'selected_as_most_influential' not in data:
                        data['selected_as_most_influential'] = False
                    writer.writerow(data)
        
        print(f"Saved {len(self.traceback_results)} traceback results to {self.traceback_csv_path}")
        print(f"Saved {len(self.influence_chain_data)} influence chain details to {self.influence_chain_csv_path}")
        
        # Clear data after saving
        self.traceback_results = []
        self.influence_chain_data = []
    
    def get_summary_statistics(self) -> Dict:
        """Get summary statistics of all analyzed episodes."""
        if not self.traceback_results:
            return {}
        
        total_episodes = len(self.traceback_results)
        episodes_with_patient_zero = sum(1 for r in self.traceback_results if r['true_patient_zero'] is not None)
        cycles_detected = sum(1 for r in self.traceback_results if r['cycle_detected'])
        
        chain_lengths = [r['influence_chain_length'] for r in self.traceback_results if r['influence_chain_length'] > 0]
        avg_chain_length = sum(chain_lengths) / len(chain_lengths) if chain_lengths else 0
        
        return {
            'total_episodes_analyzed': total_episodes,
            'episodes_with_patient_zero': episodes_with_patient_zero,
            'detection_rate': episodes_with_patient_zero / total_episodes if total_episodes > 0 else 0,
            'cycles_detected': cycles_detected,
            'cycle_rate': cycles_detected / total_episodes if total_episodes > 0 else 0,
            'average_chain_length': avg_chain_length,
            'max_chain_length': max(chain_lengths) if chain_lengths else 0,
            'min_chain_length': min(chain_lengths) if chain_lengths else 0
        }


# Convenience functions for direct usage
def analyze_single_episode(fault_timeline: List[Dict],
                         action_influences_history: List[List[List[float]]],
                         pairwise_frob_norms_history: List[List[List[float]]],
                         taylor_history: List[List[float]],
                         num_agents: int,
                         seed: int,
                         episode_length: int,
                         log_dir: str = None) -> Dict:
    """
    Convenience function to analyze a single episode.
    
    Args:
        fault_timeline: List of fault detection events
        action_influences_history: Action influence matrices over time
        pairwise_frob_norms_history: Frobenius norm matrices over time
        taylor_history: Taylor error histories for each agent
        num_agents: Number of agents in the system
        seed: Random seed for this episode
        episode_length: Length of the episode
        log_dir: Directory to save logs (optional)
        
    Returns:
        Dictionary with traceback analysis results
    """
    analyzer = PatientZeroTracebackAnalyzer(log_dir=log_dir)
    result = analyzer.analyze_patient_zero_traceback(
        fault_timeline, action_influences_history, pairwise_frob_norms_history,
        taylor_history, num_agents, seed, episode_length
    )
    analyzer.save_results_to_csv()
    return result


def batch_analyze_episodes(episodes_data: List[Dict], log_dir: str = None) -> List[Dict]:
    """
    Convenience function to analyze multiple episodes in batch.
    
    Args:
        episodes_data: List of episode data dictionaries, each containing:
            - fault_timeline
            - action_influences_history
            - pairwise_frob_norms_history
            - taylor_history
            - num_agents
            - seed
            - episode_length
        log_dir: Directory to save logs (optional)
        
    Returns:
        List of traceback analysis results
    """
    analyzer = PatientZeroTracebackAnalyzer(log_dir=log_dir)
    results = []
    
    for episode_data in episodes_data:
        result = analyzer.analyze_patient_zero_traceback(
            episode_data['fault_timeline'],
            episode_data['action_influences_history'],
            episode_data['pairwise_frob_norms_history'],
            episode_data['taylor_history'],
            episode_data['num_agents'],
            episode_data['seed'],
            episode_data['episode_length']
        )
        results.append(result)
    
    analyzer.save_results_to_csv()
    return results