"""
Detection Analysis Module

This module analyzes patient zero detection accuracy and provides functionality
to identify failed cases and perform traceback when necessary.
"""

import numpy as np
import pandas as pd
import math
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
        
        # Statistics tracking for influence tie-breaking
        self.detection_stats_influence = {
            'total_cases': 0,
            'correct_detections': 0,
            'incorrect_detections': 0,
            'threshold_failures': 0,  # Detection before attack (threshold issue)
            'traceback_needed': 0,
            'traceback_successful': 0,
            'traceback_failed': 0,
            'no_detection': 0
        }
        
        # Statistics tracking for Taylor tie-breaking
        self.detection_stats_taylor = {
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
        Analyze the accuracy of patient zero detection using both tie-breaking methods.
        
        Args:
            fault_timeline: List of fault detection events
            attacked_agent: The actual agent that was attacked
            attack_timesteps: List of timesteps when attacks occurred
            directional_derivative_history: History of directional derivative matrices
            taylor_errors_history: History of Taylor errors
            ref_vals: Reference Taylor error values
            action_influences_history: History of action influence matrices
            seed: Random seed for this experiment
            agent_pair: Tuple of (influencing_agent, influenced_agent)
            
        Returns:
            Dictionary containing analysis results for both methods
        """
        
        # Get initial detection
        detected_agents, detection_time = get_patient_zero_detection(fault_timeline)
        
        # Base result structure for both methods
        base_result = {
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
        
        # Results for both methods
        results = {
            'influence': base_result.copy(),
            'taylor': base_result.copy()
        }
        
        # Update statistics for both methods
        self.detection_stats_influence['total_cases'] += 1
        self.detection_stats_taylor['total_cases'] += 1
        
        # Case 1: No detection
        if not detected_agents or detection_time is None:
            self.detection_stats_influence['no_detection'] += 1
            self.detection_stats_taylor['no_detection'] += 1
            
            for method in ['influence', 'taylor']:
                results[method]['analysis'] = 'no_detection'
            
            print(f"  No detection for seed {seed}")
            self.detailed_results.append(results)
            return results
        
        # Case 2: Check if detection is correct initially
        start_attack_timestep = min(attack_timesteps) if attack_timesteps else float('inf')
        is_initially_correct = attacked_agent in detected_agents
        
        if is_initially_correct:
            self.detection_stats_influence['correct_detections'] += 1
            self.detection_stats_taylor['correct_detections'] += 1
            
            for method in ['influence', 'taylor']:
                results[method]['is_correct'] = True
                results[method]['final_patient_zero'] = attacked_agent
                results[method]['final_is_correct'] = True
                results[method]['analysis'] = 'correct_initial_detection'
            
            print(f"  Correct initial detection for seed {seed}: Agent {attacked_agent}")
            self.detailed_results.append(results)
            return results
        
        # Case 3: Incorrect detection - analyze the cause
        self.detection_stats_influence['incorrect_detections'] += 1
        self.detection_stats_taylor['incorrect_detections'] += 1
        
        # Check if detection occurred before attack (threshold issue)
        if detection_time < start_attack_timestep:
            self.detection_stats_influence['threshold_failures'] += 1
            self.detection_stats_taylor['threshold_failures'] += 1
            
            for method in ['influence', 'taylor']:
                results[method]['is_threshold_failure'] = True
                results[method]['analysis'] = 'threshold_failure_early_detection'
                results[method]['final_patient_zero'] = detected_agents[0] if len(detected_agents) == 1 else detected_agents
                results[method]['final_is_correct'] = False
            
            print(f"  Threshold failure for seed {seed}: Detection at {detection_time} before attack at {start_attack_timestep}")
        else:
            # Need traceback - test both methods
            self.detection_stats_influence['traceback_needed'] += 1
            self.detection_stats_taylor['traceback_needed'] += 1
            
            print(f"  Performing traceback for seed {seed} with both tie-breaking methods...")
            
            # Perform traceback with influence tie-breaking
            true_patient_zero_influence, influence_chain_influence, _ = perform_patient_zero_traceback(
                fault_timeline, directional_derivative_history, taylor_errors_history,
                ref_vals, self.all_agents, action_influences_history, use_taylor_scoring=False
            )
            
            # Perform traceback with Taylor tie-breaking
            true_patient_zero_taylor, influence_chain_taylor, _ = perform_patient_zero_traceback(
                fault_timeline, directional_derivative_history, taylor_errors_history,
                ref_vals, self.all_agents, action_influences_history, use_taylor_scoring=True
            )
            
            # Process results for influence method
            results['influence']['traceback_performed'] = True
            results['influence']['traceback_result'] = true_patient_zero_influence
            results['influence']['traceback_chain'] = influence_chain_influence
            results['influence']['final_patient_zero'] = true_patient_zero_influence
            
            if true_patient_zero_influence == attacked_agent:
                self.detection_stats_influence['traceback_successful'] += 1
                results['influence']['final_is_correct'] = True
                results['influence']['analysis'] = 'traceback_successful'
                print(f"    Influence traceback SUCCESSFUL: Found {true_patient_zero_influence} (expected {attacked_agent})")
                print(f"    Chain: {' -> '.join(map(str, influence_chain_influence))}")
            else:
                self.detection_stats_influence['traceback_failed'] += 1
                results['influence']['final_is_correct'] = False
                results['influence']['analysis'] = 'traceback_failed'
                print(f"    Influence traceback FAILED: Found {true_patient_zero_influence} (expected {attacked_agent})")
                print(f"    Chain: {' -> '.join(map(str, influence_chain_influence))}")
            
            # Process results for Taylor method
            results['taylor']['traceback_performed'] = True
            results['taylor']['traceback_result'] = true_patient_zero_taylor
            results['taylor']['traceback_chain'] = influence_chain_taylor
            results['taylor']['final_patient_zero'] = true_patient_zero_taylor
            
            if true_patient_zero_taylor == attacked_agent:
                self.detection_stats_taylor['traceback_successful'] += 1
                results['taylor']['final_is_correct'] = True
                results['taylor']['analysis'] = 'traceback_successful'
                print(f"    Taylor traceback SUCCESSFUL: Found {true_patient_zero_taylor} (expected {attacked_agent})")
                print(f"    Chain: {' -> '.join(map(str, influence_chain_taylor))}")
            else:
                self.detection_stats_taylor['traceback_failed'] += 1
                results['taylor']['final_is_correct'] = False
                results['taylor']['analysis'] = 'traceback_failed'
                print(f"    Taylor traceback FAILED: Found {true_patient_zero_taylor} (expected {attacked_agent})")
                print(f"    Chain: {' -> '.join(map(str, influence_chain_taylor))}")
        
        self.detailed_results.append(results)
        return results
    
    def get_statistics(self, method: str = 'influence') -> Dict[str, Any]:
        """
        Get comprehensive statistics about detection accuracy.
        
        Args:
            method: Which tie-breaking method statistics to return ('influence' or 'taylor')
        
        Returns:
            Dictionary containing statistics and percentages
        """
        if method == 'influence':
            stats = self.detection_stats_influence.copy()
        elif method == 'taylor':
            stats = self.detection_stats_taylor.copy()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        total = stats['total_cases']
        
        if total == 0:
            stats['correct_detection_rate'] = math.nan
            stats['incorrect_detection_rate'] = math.nan
            stats['threshold_failure_rate'] = math.nan
            stats['no_detection_rate'] = math.nan
            stats['traceback_success_rate'] = math.nan
            stats['final_accuracy_rate'] = math.nan
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
    
    def get_statistics_dual(self) -> Dict[str, Dict[str, Any]]:
        """
        Get comprehensive statistics for both tie-breaking methods.
        
        Returns:
            Dictionary containing statistics for both methods
        """
        return {
            'influence': self.get_statistics('influence'),
            'taylor': self.get_statistics('taylor')
        }
    
    def print_summary_dual(self):
        """Print a summary comparing both tie-breaking methods."""
        stats_dual = self.get_statistics_dual()
        
        print("\n" + "="*80)
        print("PATIENT ZERO DETECTION ANALYSIS SUMMARY - DUAL TIE-BREAKING COMPARISON")
        print("="*80)
        
        for method in ['influence', 'taylor']:
            stats = stats_dual[method]
            method_title = method.upper() + " TIE-BREAKING"
            
            print(f"\n{method_title}:")
            print("-" * len(method_title))
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
        
        # Comparison
        influence_accuracy = stats_dual['influence']['final_accuracy_rate']
        taylor_accuracy = stats_dual['taylor']['final_accuracy_rate']
        
        print(f"\n{'='*40}")
        print("COMPARISON:")
        print(f"{'='*40}")
        print(f"Influence tie-breaking final accuracy: {influence_accuracy:.1f}%")
        print(f"Taylor tie-breaking final accuracy: {taylor_accuracy:.1f}%")
        
        if taylor_accuracy > influence_accuracy:
            improvement = taylor_accuracy - influence_accuracy
            print(f"Taylor tie-breaking performs BETTER by {improvement:.1f} percentage points")
        elif influence_accuracy > taylor_accuracy:
            improvement = influence_accuracy - taylor_accuracy
            print(f"Influence tie-breaking performs BETTER by {improvement:.1f} percentage points")
        else:
            print("Both methods perform EQUALLY well")
        
        print("="*80)
    
    def print_summary(self, method: str = 'influence'):
        """Print a summary of detection analysis results."""
        stats = self.get_statistics(method)
        
        print("\n" + "="*60)
        print(f"PATIENT ZERO DETECTION ANALYSIS SUMMARY - {method.upper()} TIE-BREAKING")
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
        
        # Convert results to DataFrame - handle both methods
        df_data = []
        for result in self.detailed_results:
            for method in ['influence', 'taylor']:
                method_result = result[method]
                row = {
                    'method': method,
                    'seed': method_result.get('seed'),
                    'agent_pair': str(method_result.get('agent_pair')),
                    'attacked_agent': method_result.get('attacked_agent'),
                    'start_attack_timestep': method_result.get('start_attack_timestep'),
                    'detected_agents': str(method_result.get('detected_agents')),
                    'detection_time': method_result.get('detection_time'),
                    'is_correct_initial': method_result.get('is_correct'),
                    'is_threshold_failure': method_result.get('is_threshold_failure'),
                    'traceback_performed': method_result.get('traceback_performed'),
                    'traceback_result': method_result.get('traceback_result'),
                    'traceback_chain': str(method_result.get('traceback_chain')),
                    'final_patient_zero': method_result.get('final_patient_zero'),
                    'final_is_correct': method_result.get('final_is_correct'),
                    'analysis': method_result.get('analysis')
                }
                df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(filepath, index=False)
        print(f"Detailed results saved to: {filepath}")
    
    def reset_statistics(self):
        """Reset all statistics and detailed results."""
        base_stats = {
            'total_cases': 0,
            'correct_detections': 0,
            'incorrect_detections': 0,
            'threshold_failures': 0,
            'traceback_needed': 0,
            'traceback_successful': 0,
            'traceback_failed': 0,
            'no_detection': 0
        }
        
        self.detection_stats_influence = base_stats.copy()
        self.detection_stats_taylor = base_stats.copy()
        self.detection_stats = self.detection_stats_influence
        
        self.detailed_results = []