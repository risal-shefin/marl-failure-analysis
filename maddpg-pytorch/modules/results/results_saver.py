"""
Results saving and reporting functionality.
"""
import os
import csv
import json


class ResultsSaver:
    """
    Saves experiment results to various file formats.
    """
    
    @staticmethod
    def save_accuracy_summary(accuracy_results, logdir):
        """
        Save accuracy summary to JSON file.
        
        Args:
            accuracy_results: Dictionary containing accuracy metrics
            logdir: Directory to save the results
        """
        summary_file = os.path.join(logdir, "accuracy_summary.json")
        
        with open(summary_file, 'w') as f:
            json.dump(accuracy_results, f, indent=2)
        
        print(f"Accuracy summary saved to: {summary_file}")
    
    @staticmethod
    def save_failed_expectations(failed_expectations, logdir):
        """
        Save failed expectations to CSV file.
        
        Args:
            failed_expectations: List of failed expectation records
            logdir: Directory to save the results
        """
        if not failed_expectations:
            return
        
        failed_file = os.path.join(logdir, "failed_expectations.csv")
        
        with open(failed_file, 'w', newline='') as csvfile:
            fieldnames = ['seed', 'agent_i', 'agent_j', 'metric', 'high_value', 'low_value']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for record in failed_expectations:
                writer.writerow(record)
        
        print(f"Failed expectations saved to: {failed_file}")
    
    @staticmethod
    def save_pair_specific_results(pair_specific_results, logdir):
        """
        Save pair-specific accuracy results to CSV files.
        
        Args:
            pair_specific_results: Dictionary mapping pairs to accuracy metrics
            logdir: Directory to save the results
        """
        if not pair_specific_results:
            return
        
        # Create subdirectory for pair-specific results
        pair_dir = os.path.join(logdir, "pair_specific_results")
        os.makedirs(pair_dir, exist_ok=True)
        
        # Save summary CSV
        summary_file = os.path.join(logdir, "pair_specific_accuracy_summary.csv")
        
        with open(summary_file, 'w', newline='') as csvfile:
            fieldnames = ['agent_i', 'agent_j', 'total_experiments',
                         'q_drop_max_accuracy', 'q_drop_weighted_accuracy',
                         'reward_drop_max_accuracy', 'reward_drop_weighted_accuracy',
                         'taylor_max_accuracy', 'taylor_weighted_accuracy',
                         'exceed_rate_accuracy']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for pair_key, results in sorted(pair_specific_results.items()):
                row = {
                    'agent_i': results['agent_i'],
                    'agent_j': results['agent_j'],
                    'total_experiments': results['total_experiments'],
                    'q_drop_max_accuracy': results['q_drop_max_accuracy'],
                    'q_drop_weighted_accuracy': results['q_drop_weighted_accuracy'],
                    'reward_drop_max_accuracy': results['reward_drop_max_accuracy'],
                    'reward_drop_weighted_accuracy': results['reward_drop_weighted_accuracy'],
                    'taylor_max_accuracy': results['taylor_max_accuracy'],
                    'taylor_weighted_accuracy': results['taylor_weighted_accuracy'],
                    'exceed_rate_accuracy': results['exceed_rate_accuracy']
                }
                writer.writerow(row)
        
        print(f"Pair-specific accuracy summary saved to: {summary_file}")
        
        # Save detailed results for each pair
        for pair_key, results in pair_specific_results.items():
            pair_file = os.path.join(pair_dir, 
                                    f"pair_{results['agent_i']}_to_{results['agent_j']}.csv")
            
            with open(pair_file, 'w', newline='') as csvfile:
                fieldnames = ['metric_type', 'metric_name', 'high_avg', 'low_avg', 'accuracy']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                # Write Q-drop metrics
                writer.writerow({
                    'metric_type': 'Q-drop',
                    'metric_name': 'max',
                    'high_avg': results['avg_high_metrics']['max_q_drop'],
                    'low_avg': results['avg_low_metrics']['max_q_drop'],
                    'accuracy': results['q_drop_max_accuracy']
                })
                writer.writerow({
                    'metric_type': 'Q-drop',
                    'metric_name': 'weighted_sum',
                    'high_avg': results['avg_high_metrics']['weighted_q_drop_sum'],
                    'low_avg': results['avg_low_metrics']['weighted_q_drop_sum'],
                    'accuracy': results['q_drop_weighted_accuracy']
                })
                
                # Write Reward-drop metrics
                writer.writerow({
                    'metric_type': 'Reward-drop',
                    'metric_name': 'max',
                    'high_avg': results['avg_high_metrics']['max_reward_drop'],
                    'low_avg': results['avg_low_metrics']['max_reward_drop'],
                    'accuracy': results['reward_drop_max_accuracy']
                })
                writer.writerow({
                    'metric_type': 'Reward-drop',
                    'metric_name': 'weighted_sum',
                    'high_avg': results['avg_high_metrics']['weighted_reward_drop_sum'],
                    'low_avg': results['avg_low_metrics']['weighted_reward_drop_sum'],
                    'accuracy': results['reward_drop_weighted_accuracy']
                })
                
                # Write Taylor deviation metrics
                writer.writerow({
                    'metric_type': 'Taylor-deviation',
                    'metric_name': 'max_abs',
                    'high_avg': results['avg_high_metrics']['max_abs_taylor_deviation'],
                    'low_avg': results['avg_low_metrics']['max_abs_taylor_deviation'],
                    'accuracy': results['taylor_max_accuracy']
                })
                writer.writerow({
                    'metric_type': 'Taylor-deviation',
                    'metric_name': 'weighted_sum',
                    'high_avg': results['avg_high_metrics']['weighted_taylor_deviation_sum'],
                    'low_avg': results['avg_low_metrics']['weighted_taylor_deviation_sum'],
                    'accuracy': results['taylor_weighted_accuracy']
                })
                
                # Write Exceed rate metric
                writer.writerow({
                    'metric_type': 'Exceed-rate',
                    'metric_name': 'rate',
                    'high_avg': results['avg_high_metrics']['exceed_rate'],
                    'low_avg': results['avg_low_metrics']['exceed_rate'],
                    'accuracy': results['exceed_rate_accuracy']
                })
        
        print(f"Pair-specific detailed results saved in: {pair_dir}")
        print(f"  {len(pair_specific_results)} individual pair CSV files created")
        
        return summary_file, pair_dir
    
    @staticmethod
    def print_pair_specific_summary(pair_specific_results):
        """
        Print summary of pair-specific results.
        
        Args:
            pair_specific_results: Dictionary mapping pairs to accuracy metrics
        """
        if not pair_specific_results:
            return
        
        print("\n" + "="*50)
        print("PAIR-SPECIFIC ACCURACY SUMMARY")
        print("="*50)
        
        for pair_key in sorted(pair_specific_results.keys()):
            results = pair_specific_results[pair_key]
            print(f"\nPair: Agent {results['agent_i']} -> Agent {results['agent_j']}")
            print(f"  Total experiments: {results['total_experiments']}")
            print(f"  Accuracies:")
            print(f"    Max Q-drop: {results['q_drop_max_accuracy']:.2%}")
            print(f"    Weighted Q-drop: {results['q_drop_weighted_accuracy']:.2%}")
            print(f"    Max Reward drop: {results['reward_drop_max_accuracy']:.2%}")
            print(f"    Weighted Reward drop: {results['reward_drop_weighted_accuracy']:.2%}")
            print(f"    Max Taylor deviation: {results['taylor_max_accuracy']:.2%}")
            print(f"    Weighted Taylor deviation: {results['taylor_weighted_accuracy']:.2%}")
            print(f"    Exceed rate: {results['exceed_rate_accuracy']:.2%}")
    
    @staticmethod
    def print_final_summary(logdir, accuracy_results, pair_specific_results=None):
        """
        Print final summary of all saved results.
        
        Args:
            logdir: Directory where results are saved
            accuracy_results: Dictionary containing accuracy metrics
            pair_specific_results: Optional dict of pair-specific results
        """
        print("\n" + "="*50)
        print("RESULTS SAVED")
        print("="*50)
        print(f"All results saved to: {logdir}")
        print("\nFiles created:")
        print("- accuracy_summary.json")
        print("- failed_expectations.csv")
        
        if pair_specific_results:
            print("- pair_specific_accuracy_summary.csv")
            print(f"- pair_specific_results/ directory with {len(pair_specific_results)} pair files")
        
        print("- cumulative_influences_all_seeds.csv")
        print("- directional_derivatives_all_seeds.csv")
        print("- taylor_deviations_all_seeds.csv")
