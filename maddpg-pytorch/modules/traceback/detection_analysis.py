"""
Detection Analysis Module

This module analyzes patient zero detection accuracy and provides functionality
to identify failed cases and perform traceback when necessary.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from .pzero_traceback import perform_patient_zero_traceback
from modules.detection import get_patient_zero_detection


class PatientZeroAnalyzer:
    """
    Analyzer for patient zero detection accuracy and traceback functionality.
    """
    
    def __init__(self, nagents: int):
        """
        Initialize the analyzer.
        
        Args:
            nagents: Number of agents in the system
        """
        self.nagents = nagents
        self.all_agents = list(range(nagents))
        
        # Statistics tracking
        self.detection_stats = {
            'total_cases': 0,
            'correct_detections': 0,
            'incorrect_detections': 0,
            'threshold_failures': 0,  # Detection before attack (threshold issue)
            'traceback_needed': 0,
            'traceback_successful': 0,
            'traceback_failed': 0,
            'no_detection': 0
        }
        
        # Detailed results for logging
        self.detailed_results = []
    
    def analyze_detection_accuracy(self,
                                 fault_timeline: List[Dict],
                                 attacked_agent: int,
                                 attack_timesteps: List[int],
                                 directional_derivative_history: List[np.ndarray],
                                 taylor_errors_history: List[Dict],
                                 ref_vals: List[List[float]],
                                 action_influences_history: Optional[List[np.ndarray]] = None,
                                 seed: int = None,
                                 agent_pair: Tuple[int, int] = None) -> Dict[str, Any]:
        """
        Analyze the accuracy of patient zero detection for a single case.
        
        Args:
            fault_timeline: List of fault detection events
            attacked_agent: The actual agent that was attacked
            attack_timesteps: List of timesteps when attacks occurred
            action_influences_history: History of action influence matrices
            taylor_errors_history: History of Taylor errors
            ref_vals: Reference Taylor error values
            ref_std_devs: Reference standard deviations
            frob_norms_history: History of Frobenius norms (optional)
            seed: Random seed for this experiment
            agent_pair: Tuple of (influencing_agent, influenced_agent)
            
        Returns:
            Dictionary containing analysis results
        """
        
        # Get initial detection
        detected_agents, detection_time = get_patient_zero_detection(fault_timeline)
        
        result = {
            'seed': seed,
            'agent_pair': agent_pair,
            'attacked_agent': attacked_agent,
            'attack_timesteps': attack_timesteps,
            'start_attack_timestep': min(attack_timesteps) if attack_timesteps else None,
            'detected_agents': detected_agents,
            'detection_time': detection_time,
            'is_correct': False,
            'is_threshold_failure': False,
            'traceback_performed': False,
            'traceback_result': None,
            'traceback_chain': None,
            'final_patient_zero': None,
            'final_is_correct': False
        }
        
        self.detection_stats['total_cases'] += 1
        
        # Case 1: No detection
        if not detected_agents or detection_time is None:
            self.detection_stats['no_detection'] += 1
            result['analysis'] = 'no_detection'
            self.detailed_results.append(result)
            return result
        
        # Case 2: Check if detection is correct initially
        min_attack_timestep = min(attack_timesteps) if attack_timesteps else float('inf')
        is_initially_correct = attacked_agent in detected_agents
        
        if is_initially_correct:
            self.detection_stats['correct_detections'] += 1
            result['is_correct'] = True
            result['final_patient_zero'] = attacked_agent
            result['final_is_correct'] = True
            result['analysis'] = 'correct_initial_detection'
            self.detailed_results.append(result)
            return result
        
        # Case 3: Incorrect detection - analyze the cause
        self.detection_stats['incorrect_detections'] += 1
        
        # Check if detection occurred before attack (threshold issue)
        if detection_time < min_attack_timestep:
            self.detection_stats['threshold_failures'] += 1
            result['is_threshold_failure'] = True
            result['analysis'] = 'threshold_failure_early_detection'
            result['final_patient_zero'] = detected_agents[0] if len(detected_agents) == 1 else detected_agents
            result['final_is_correct'] = False
            print(f"  Threshold failure: Detection at {detection_time} before attack at {min_attack_timestep}")
        else:
            # Need traceback
            self.detection_stats['traceback_needed'] += 1
            result['traceback_performed'] = True
            
            true_patient_zero, influence_chain, _ = perform_patient_zero_traceback(
                fault_timeline, directional_derivative_history, taylor_errors_history,
                ref_vals, self.all_agents, action_influences_history
            )
            
            result['traceback_result'] = true_patient_zero
            result['traceback_chain'] = influence_chain
            result['final_patient_zero'] = true_patient_zero
            
            if true_patient_zero == attacked_agent:
                self.detection_stats['traceback_successful'] += 1
                result['final_is_correct'] = True
                result['analysis'] = 'traceback_successful'
                print(f"  Traceback successful: Found true patient zero {true_patient_zero}")
                print(f"  Influence chain: {' -> '.join(map(str, influence_chain))}")
            else:
                self.detection_stats['traceback_failed'] += 1
                result['final_is_correct'] = False
                result['analysis'] = 'traceback_failed'
                print(f"  Traceback failed: Found {true_patient_zero}, expected {attacked_agent}")
                print(f"  Influence chain: {' -> '.join(map(str, influence_chain))}")
                    
        
        self.detailed_results.append(result)
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about detection accuracy.
        
        Returns:
            Dictionary containing statistics and percentages
        """
        stats = self.detection_stats.copy()
        total = stats['total_cases']
        
        if total == 0:
            return stats
        
        # Calculate percentages
        stats['correct_detection_rate'] = (stats['correct_detections'] / total) * 100
        stats['incorrect_detection_rate'] = (stats['incorrect_detections'] / total) * 100
        stats['threshold_failure_rate'] = (stats['threshold_failures'] / total) * 100
        stats['no_detection_rate'] = (stats['no_detection'] / total) * 100
        
        # Traceback success rate among cases that needed traceback
        if stats['traceback_needed'] > 0:
            stats['traceback_success_rate'] = (stats['traceback_successful'] / stats['traceback_needed']) * 100
        else:
            stats['traceback_success_rate'] = 0.0
        
        # Overall accuracy after traceback
        total_correct_final = stats['correct_detections'] + stats['traceback_successful']
        stats['final_accuracy_rate'] = (total_correct_final / total) * 100
        
        return stats
    
    def print_summary(self):
        """Print a summary of detection analysis results."""
        stats = self.get_statistics()
        
        print("\n" + "="*60)
        print("PATIENT ZERO DETECTION ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Total cases analyzed: {stats['total_cases']}")
        print(f"No detection: {stats['no_detection']} ({stats['no_detection_rate']:.1f}%)")
        print(f"Correct initial detections: {stats['correct_detections']} ({stats['correct_detection_rate']:.1f}%)")
        print(f"Incorrect detections: {stats['incorrect_detections']} ({stats['incorrect_detection_rate']:.1f}%)")
        
        print(f"\nIncorrect Detection Breakdown:")
        print(f"  Threshold failures (early detection): {stats['threshold_failures']} ({stats['threshold_failure_rate']:.1f}%)")
        print(f"  Cases requiring traceback: {stats['traceback_needed']}")
        
        if stats['traceback_needed'] > 0:
            print(f"\nTraceback Results:")
            print(f"  Successful traceback: {stats['traceback_successful']} ({stats['traceback_success_rate']:.1f}%)")
            print(f"  Failed traceback: {stats['traceback_failed']}")
        
        print(f"\nFinal Accuracy (after traceback): {stats['final_accuracy_rate']:.1f}%")
        print("="*60)
    
    def save_detailed_results(self, filepath: str):
        """
        Save detailed results to a CSV file.
        
        Args:
            filepath: Path to save the CSV file
        """
        
        if not self.detailed_results:
            print("No detailed results to save")
            return
        
        # Convert results to DataFrame
        df_data = []
        for result in self.detailed_results:
            row = {
                'seed': result.get('seed'),
                'agent_pair': str(result.get('agent_pair')),
                'attacked_agent': result.get('attacked_agent'),
                'min_attack_timestep': result.get('min_attack_timestep'),
                'detected_agents': str(result.get('detected_agents')),
                'detection_time': result.get('detection_time'),
                'is_correct_initial': result.get('is_correct'),
                'is_threshold_failure': result.get('is_threshold_failure'),
                'traceback_performed': result.get('traceback_performed'),
                'traceback_result': result.get('traceback_result'),
                'traceback_chain': str(result.get('traceback_chain')),
                'final_patient_zero': result.get('final_patient_zero'),
                'final_is_correct': result.get('final_is_correct'),
                'analysis': result.get('analysis')
            }
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(filepath, index=False)
        print(f"Detailed results saved to: {filepath}")
    
    def reset_statistics(self):
        """Reset all statistics and detailed results."""
        self.detection_stats = {
            'total_cases': 0,
            'correct_detections': 0,
            'incorrect_detections': 0,
            'threshold_failures': 0,
            'traceback_needed': 0,
            'traceback_successful': 0,
            'traceback_failed': 0,
            'no_detection': 0
        }
        self.detailed_results = []