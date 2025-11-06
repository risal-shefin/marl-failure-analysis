"""
Accuracy computation for attack detection and metrics validation.
"""
import numpy as np


class AccuracyComputer:
    """
    Computes various accuracy metrics for attack detection and effectiveness.
    """
    
    @staticmethod
    def compute_experiment_accuracies(experiment_results, patient_zero_analyzer):
        """
        Compute accuracies and analyze results across all experiments.
        
        Args:
            experiment_results: List of experiment results from all seeds
            patient_zero_analyzer: PatientZeroAnalyzer instance for detection analysis
            
        Returns:
            Tuple of (accuracy_results, failed_expectations)
        """
        if not experiment_results:
            print("No experiment results to analyze.")
            return {}, []
        
        print("\n" + "="*50)
        print("COMPUTING ACCURACIES")
        print("="*50)
        
        # Patient zero detection accuracy
        correct_patient_zero = 0
        total_with_detection = 0
        high_correct_patient_zero = 0
        high_total_with_detection = 0
        low_correct_patient_zero = 0
        low_total_with_detection = 0
        
        # Expectation accuracy metrics - separate accuracies for each metric
        q_drop_max_expectation_correct = 0
        q_drop_weighted_expectation_correct = 0
        reward_drop_max_expectation_correct = 0
        reward_drop_weighted_expectation_correct = 0
        taylor_max_expectation_correct = 0
        taylor_weighted_expectation_correct = 0
        exceed_rate_expectation_correct = 0
        
        # Detailed metrics
        high_metrics_list = []
        low_metrics_list = []
        
        failed_expectations = []
        total_pairs = 0
        
        # Process all pairs from all seeds
        for result in experiment_results:
            seed = result['seed']
            
            for pair_result in result['pair_results']:
                agent_i = pair_result['agent_i']
                agent_j = pair_result['agent_j']
                
                high_metrics = pair_result['high_influence_metrics']
                low_metrics = pair_result['low_influence_metrics']
                high_patient_zero = pair_result['high_patient_zero']
                low_patient_zero = pair_result['low_patient_zero']
                high_detection_times = pair_result['high_influence_detection_times']
                low_detection_times = pair_result['low_influence_detection_times']
                high_attack_timesteps = pair_result['high_influence_attack_timesteps']
                low_attack_timesteps = pair_result['low_influence_attack_timesteps']
                
                total_pairs += 1
                
                # Store metrics for detailed analysis
                high_metrics_list.append(high_metrics)
                low_metrics_list.append(low_metrics)
                
                # Patient zero detection accuracy for high influence attacks
                if high_detection_times:
                    high_total_with_detection += 1
                    total_with_detection += 1
                    first_detection = min(high_detection_times)
                    if high_attack_timesteps and agent_i in high_patient_zero and first_detection >= min(high_attack_timesteps):
                        high_correct_patient_zero += 1
                        correct_patient_zero += 1
                
                # Patient zero detection accuracy for low influence attacks
                if low_detection_times:
                    low_total_with_detection += 1
                    total_with_detection += 1
                    first_detection = min(low_detection_times)
                    
                    if low_attack_timesteps and agent_i in low_patient_zero and first_detection >= min(low_attack_timesteps):
                        low_correct_patient_zero += 1
                        correct_patient_zero += 1
                
                # Expectation: High influence should have greater impact than low influence
                # Check each metric separately
                
                # Max Q-drop expectation
                if high_metrics['max_q_drop'] >= low_metrics['max_q_drop']:
                    q_drop_max_expectation_correct += 1
                else:
                    failed_expectations.append({
                        'seed': seed,
                        'agent_i': agent_i,
                        'agent_j': agent_j,
                        'metric': 'max_q_drop',
                        'high_value': high_metrics['max_q_drop'],
                        'low_value': low_metrics['max_q_drop']
                    })
                
                # Weighted Q-drop expectation
                if high_metrics['weighted_q_drop_sum'] >= low_metrics['weighted_q_drop_sum']:
                    q_drop_weighted_expectation_correct += 1
                else:
                    failed_expectations.append({
                        'seed': seed,
                        'agent_i': agent_i,
                        'agent_j': agent_j,
                        'metric': 'weighted_q_drop_sum',
                        'high_value': high_metrics['weighted_q_drop_sum'],
                        'low_value': low_metrics['weighted_q_drop_sum']
                    })
                
                # Max reward drop expectation
                if high_metrics['max_reward_drop'] >= low_metrics['max_reward_drop']:
                    reward_drop_max_expectation_correct += 1
                else:
                    failed_expectations.append({
                        'seed': seed,
                        'agent_i': agent_i,
                        'agent_j': agent_j,
                        'metric': 'max_reward_drop',
                        'high_value': high_metrics['max_reward_drop'],
                        'low_value': low_metrics['max_reward_drop']
                    })
                
                # Weighted reward drop expectation
                if high_metrics['weighted_reward_drop_sum'] >= low_metrics['weighted_reward_drop_sum']:
                    reward_drop_weighted_expectation_correct += 1
                else:
                    failed_expectations.append({
                        'seed': seed,
                        'agent_i': agent_i,
                        'agent_j': agent_j,
                        'metric': 'weighted_reward_drop_sum',
                        'high_value': high_metrics['weighted_reward_drop_sum'],
                        'low_value': low_metrics['weighted_reward_drop_sum']
                    })
                
                # Max Taylor deviation expectation
                if high_metrics['max_abs_taylor_deviation'] >= low_metrics['max_abs_taylor_deviation']:
                    taylor_max_expectation_correct += 1
                else:
                    failed_expectations.append({
                        'seed': seed,
                        'agent_i': agent_i,
                        'agent_j': agent_j,
                        'metric': 'max_abs_taylor_deviation',
                        'high_value': high_metrics['max_abs_taylor_deviation'],
                        'low_value': low_metrics['max_abs_taylor_deviation']
                    })
                
                # Weighted Taylor deviation expectation
                if high_metrics['weighted_taylor_deviation_sum'] >= low_metrics['weighted_taylor_deviation_sum']:
                    taylor_weighted_expectation_correct += 1
                else:
                    failed_expectations.append({
                        'seed': seed,
                        'agent_i': agent_i,
                        'agent_j': agent_j,
                        'metric': 'weighted_taylor_deviation_sum',
                        'high_value': high_metrics['weighted_taylor_deviation_sum'],
                        'low_value': low_metrics['weighted_taylor_deviation_sum']
                    })
                
                # Exceed rate expectation
                if high_metrics['exceed_rate'] >= low_metrics['exceed_rate']:
                    exceed_rate_expectation_correct += 1
                else:
                    failed_expectations.append({
                        'seed': seed,
                        'agent_i': agent_i,
                        'agent_j': agent_j,
                        'metric': 'exceed_rate',
                        'high_value': high_metrics['exceed_rate'],
                        'low_value': low_metrics['exceed_rate']
                    })
        
        # Compute accuracies
        patient_zero_accuracy = (correct_patient_zero / total_with_detection 
                                if total_with_detection > 0 else 0)
        high_patient_zero_accuracy = (high_correct_patient_zero / high_total_with_detection 
                                     if high_total_with_detection > 0 else 0)
        low_patient_zero_accuracy = (low_correct_patient_zero / low_total_with_detection 
                                    if low_total_with_detection > 0 else 0)
        
        q_drop_max_accuracy = q_drop_max_expectation_correct / total_pairs if total_pairs > 0 else 0
        q_drop_weighted_accuracy = q_drop_weighted_expectation_correct / total_pairs if total_pairs > 0 else 0
        reward_drop_max_accuracy = reward_drop_max_expectation_correct / total_pairs if total_pairs > 0 else 0
        reward_drop_weighted_accuracy = reward_drop_weighted_expectation_correct / total_pairs if total_pairs > 0 else 0
        taylor_max_accuracy = taylor_max_expectation_correct / total_pairs if total_pairs > 0 else 0
        taylor_weighted_accuracy = taylor_weighted_expectation_correct / total_pairs if total_pairs > 0 else 0
        exceed_rate_accuracy = exceed_rate_expectation_correct / total_pairs if total_pairs > 0 else 0
        
        # Compute average metrics
        avg_high_metrics = {
            'max_q_drop': np.mean([m['max_q_drop'] for m in high_metrics_list]),
            'weighted_q_drop_sum': np.mean([m['weighted_q_drop_sum'] for m in high_metrics_list]),
            'max_reward_drop': np.mean([m['max_reward_drop'] for m in high_metrics_list]),
            'weighted_reward_drop_sum': np.mean([m['weighted_reward_drop_sum'] for m in high_metrics_list]),
            'max_abs_taylor_deviation': np.mean([m['max_abs_taylor_deviation'] for m in high_metrics_list]),
            'weighted_taylor_deviation_sum': np.mean([m['weighted_taylor_deviation_sum'] for m in high_metrics_list]),
            'exceed_rate': np.mean([m['exceed_rate'] for m in high_metrics_list]),
        }
        
        avg_low_metrics = {
            'max_q_drop': np.mean([m['max_q_drop'] for m in low_metrics_list]),
            'weighted_q_drop_sum': np.mean([m['weighted_q_drop_sum'] for m in low_metrics_list]),
            'max_reward_drop': np.mean([m['max_reward_drop'] for m in low_metrics_list]),
            'weighted_reward_drop_sum': np.mean([m['weighted_reward_drop_sum'] for m in low_metrics_list]),
            'max_abs_taylor_deviation': np.mean([m['max_abs_taylor_deviation'] for m in low_metrics_list]),
            'weighted_taylor_deviation_sum': np.mean([m['weighted_taylor_deviation_sum'] for m in low_metrics_list]),
            'exceed_rate': np.mean([m['exceed_rate'] for m in low_metrics_list]),
        }
        
        # Print results
        print("\nPatient Zero Detection Accuracy:")
        print(f"  Overall: {patient_zero_accuracy:.2%} ({correct_patient_zero}/{total_with_detection})")
        print(f"  High influence: {high_patient_zero_accuracy:.2%} ({high_correct_patient_zero}/{high_total_with_detection})")
        print(f"  Low influence: {low_patient_zero_accuracy:.2%} ({low_correct_patient_zero}/{low_total_with_detection})")
        
        print("\nExpectation Accuracies (High > Low):")
        print(f"  Max Q-drop: {q_drop_max_accuracy:.2%}")
        print(f"  Weighted Q-drop: {q_drop_weighted_accuracy:.2%}")
        print(f"  Max Reward drop: {reward_drop_max_accuracy:.2%}")
        print(f"  Weighted Reward drop: {reward_drop_weighted_accuracy:.2%}")
        print(f"  Max Taylor deviation: {taylor_max_accuracy:.2%}")
        print(f"  Weighted Taylor deviation: {taylor_weighted_accuracy:.2%}")
        print(f"  Exceed rate: {exceed_rate_accuracy:.2%}")
        
        print("\nAverage High Influence Metrics:")
        for key, value in avg_high_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        print("\nAverage Low Influence Metrics:")
        for key, value in avg_low_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        print(f"\nTotal pairs analyzed: {total_pairs}")
        print(f"Total failed expectations: {len(failed_expectations)}")
        
        accuracy_results = {
            'patient_zero_accuracy': patient_zero_accuracy,
            'high_patient_zero_accuracy': high_patient_zero_accuracy,
            'low_patient_zero_accuracy': low_patient_zero_accuracy,
            'q_drop_max_accuracy': q_drop_max_accuracy,
            'q_drop_weighted_accuracy': q_drop_weighted_accuracy,
            'reward_drop_max_accuracy': reward_drop_max_accuracy,
            'reward_drop_weighted_accuracy': reward_drop_weighted_accuracy,
            'taylor_max_accuracy': taylor_max_accuracy,
            'taylor_weighted_accuracy': taylor_weighted_accuracy,
            'exceed_rate_accuracy': exceed_rate_accuracy,
            'avg_high_metrics': avg_high_metrics,
            'avg_low_metrics': avg_low_metrics,
            'total_pairs': total_pairs,
            'total_with_detection': total_with_detection,
            'high_total_with_detection': high_total_with_detection,
            'low_total_with_detection': low_total_with_detection
        }
        
        return accuracy_results, failed_expectations
    
    @staticmethod
    def compute_pair_specific_accuracies(experiment_results):
        """
        Compute accuracies for each specific agent pair across all seeds.
        
        Args:
            experiment_results: List of experiment results from all seeds
            
        Returns:
            Dictionary mapping (agent_i, agent_j) to accuracy metrics
        """
        if not experiment_results:
            return {}
        
        print("\n" + "="*50)
        print("COMPUTING PAIR-SPECIFIC ACCURACIES")
        print("="*50)
        
        # Collect data for each pair
        pair_data = {}
        
        for result in experiment_results:
            for pair_result in result['pair_results']:
                agent_i = pair_result['agent_i']
                agent_j = pair_result['agent_j']
                pair_key = (agent_i, agent_j)
                
                if pair_key not in pair_data:
                    pair_data[pair_key] = {
                        'seeds': [],
                        'high_metrics': [],
                        'low_metrics': [],
                        'q_drop_max_correct': 0,
                        'q_drop_weighted_correct': 0,
                        'reward_drop_max_correct': 0,
                        'reward_drop_weighted_correct': 0,
                        'taylor_max_correct': 0,
                        'taylor_weighted_correct': 0,
                        'exceed_rate_correct': 0,
                        'total_experiments': 0
                    }
                
                pair_data[pair_key]['seeds'].append(result['seed'])
                pair_data[pair_key]['high_metrics'].append(pair_result['high_influence_metrics'])
                pair_data[pair_key]['low_metrics'].append(pair_result['low_influence_metrics'])
                pair_data[pair_key]['total_experiments'] += 1
                
                # Check expectations
                high_m = pair_result['high_influence_metrics']
                low_m = pair_result['low_influence_metrics']
                
                if high_m['max_q_drop'] > low_m['max_q_drop']:
                    pair_data[pair_key]['q_drop_max_correct'] += 1
                if high_m['weighted_q_drop_sum'] > low_m['weighted_q_drop_sum']:
                    pair_data[pair_key]['q_drop_weighted_correct'] += 1
                if high_m['max_reward_drop'] > low_m['max_reward_drop']:
                    pair_data[pair_key]['reward_drop_max_correct'] += 1
                if high_m['weighted_reward_drop_sum'] > low_m['weighted_reward_drop_sum']:
                    pair_data[pair_key]['reward_drop_weighted_correct'] += 1
                if high_m['max_abs_taylor_deviation'] > low_m['max_abs_taylor_deviation']:
                    pair_data[pair_key]['taylor_max_correct'] += 1
                if high_m['weighted_taylor_deviation_sum'] > low_m['weighted_taylor_deviation_sum']:
                    pair_data[pair_key]['taylor_weighted_correct'] += 1
                if high_m['exceed_rate'] > low_m['exceed_rate']:
                    pair_data[pair_key]['exceed_rate_correct'] += 1
        
        # Compute accuracies for each pair
        pair_results = {}
        for pair_key, data in pair_data.items():
            total = data['total_experiments']
            pair_results[pair_key] = {
                'agent_i': pair_key[0],
                'agent_j': pair_key[1],
                'total_experiments': total,
                'q_drop_max_accuracy': data['q_drop_max_correct'] / total,
                'q_drop_weighted_accuracy': data['q_drop_weighted_correct'] / total,
                'reward_drop_max_accuracy': data['reward_drop_max_correct'] / total,
                'reward_drop_weighted_accuracy': data['reward_drop_weighted_correct'] / total,
                'taylor_max_accuracy': data['taylor_max_correct'] / total,
                'taylor_weighted_accuracy': data['taylor_weighted_correct'] / total,
                'exceed_rate_accuracy': data['exceed_rate_correct'] / total,
                'avg_high_metrics': {
                    key: np.mean([m[key] for m in data['high_metrics']])
                    for key in data['high_metrics'][0].keys()
                },
                'avg_low_metrics': {
                    key: np.mean([m[key] for m in data['low_metrics']])
                    for key in data['low_metrics'][0].keys()
                }
            }
        
        print(f"Computed pair-specific accuracies for {len(pair_results)} agent pairs")
        
        return pair_results
