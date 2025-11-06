#!/usr/bin/env python3
"""Multi-seed fault analysis statistics for MAPPO agents in PettingZoo environments."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from MAPPO_MPE_main import Runner_MAPPO_MPE
from MAPPO_MPE.modules.analysis import InfluenceAnalyzer
from MAPPO_MPE.modules.constants import DEFAULT_INFLUENCE_DECAY_LAMBDA, DEVICE
from MAPPO_MPE.modules.detection import get_patient_zero_detection
from MAPPO_MPE.modules.experiments import EpisodeRunner, ExperimentDataLogger, ReferenceTaylorManager
from MAPPO_MPE.modules.metrics import AttackMetricsComputer
from MAPPO_MPE.modules.results import AccuracyComputer, ResultsSaver
from MAPPO_MPE.modules.traceback import PatientZeroAnalyzer
from MAPPO_MPE.modules.visualization import save_frames_as_gif

ATTACK_TS_FRACTION = 0.5


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("MAPPO Fault Analysis Statistics (PettingZoo)")
    parser.add_argument('--model_dir', type=str, required=True, help='Path to saved MAPPO model (.pt)')
    parser.add_argument('--env_id', type=str, required=True, help='PettingZoo environment ID (e.g., simple_spread_v3)')
    parser.add_argument('--seed', type=int, default=0, help='Base random seed')
    parser.add_argument('--total_experiments', type=int, default=5, help='Number of seeds to evaluate')
    parser.add_argument('--single_seed', action='store_true', help='Run a single-seed analysis with visualizations')
    parser.add_argument('--taylor_epsilon', type=float, default=0.01, help='Perturbation magnitude for Taylor analysis')
    parser.add_argument('--taylor_cache_dir', type=str, default=None, help='Directory to cache reference Taylor statistics')
    parser.add_argument('--precompute_taylor', action='store_true', help='Precompute reference Taylor error values and save them')
    parser.add_argument('--output_dir', type=str, default='./results', help='Base output directory')
    parser.add_argument('--discrete_action', action='store_true', default=True, help='Whether actions are discrete')
    parser.add_argument('--attack_top_k', type=int, default=1, help='Number of influential timesteps to attack')
    parser.add_argument('--lambda_decay', type=float, default=DEFAULT_INFLUENCE_DECAY_LAMBDA, help='Decay factor for influence aggregation')
    # MAPPO runner hyperparameters
    parser.add_argument('--max_train_steps', type=int, default=int(3e6))
    parser.add_argument('--episode_limit', type=int, default=25)
    parser.add_argument('--evaluate_freq', type=float, default=5000)
    parser.add_argument('--evaluate_times', type=float, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mini_batch_size', type=int, default=8)
    parser.add_argument('--rnn_hidden_dim', type=int, default=64)
    parser.add_argument('--mlp_hidden_dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lamda', type=float, default=0.95)
    parser.add_argument('--epsilon', type=float, default=0.2)
    parser.add_argument('--K_epochs', type=int, default=15)
    parser.add_argument('--use_adv_norm', type=bool, default=True)
    parser.add_argument('--use_reward_norm', type=bool, default=True)
    parser.add_argument('--use_reward_scaling', type=bool, default=False)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--use_lr_decay', type=bool, default=True)
    parser.add_argument('--use_grad_clip', type=bool, default=True)
    parser.add_argument('--use_orthogonal_init', type=bool, default=True)
    parser.add_argument('--set_adam_eps', type=bool, default=True)
    parser.add_argument('--use_relu', type=bool, default=False)
    parser.add_argument('--use_rnn', type=bool, default=False)
    parser.add_argument('--add_agent_id', type=bool, default=False)
    parser.add_argument('--use_value_clip', type=bool, default=False)
    return parser


@dataclass
class ExperimentResult:
    influencing_agent: int
    influenced_agent: int
    high_attack_metrics: Dict
    low_attack_metrics: Dict
    high_detection: Dict
    low_detection: Dict
    high_timesteps: List[int]
    low_timesteps: List[int]


class MultiSeedExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.runner: Optional[Runner_MAPPO_MPE] = None
        self.env = None
        self.logdir: Optional[str] = None
        self.total_experiments = config.total_experiments

        self.reference_manager: Optional[ReferenceTaylorManager] = None
        self.episode_runner: Optional[EpisodeRunner] = None
        self.influence_analyzer = InfluenceAnalyzer()
        self.metrics_computer: Optional[AttackMetricsComputer] = None
        self.data_logger: Optional[ExperimentDataLogger] = None
        self.patient_zero_analyzer: Optional[PatientZeroAnalyzer] = None

        self.results: List[ExperimentResult] = []
        self.experiment_results: List[Dict] = []  # Structured results for accuracy computation
        self.failed_seeds: List[int] = []

    def setup_experiment(self):
        cwd = os.getcwd()
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        mode = 'single_seed' if self.config.single_seed else 'multi_seed'
        self.logdir = os.path.join(cwd, 'runs', f"{self.config.env_id}_mappo_{mode}_fault_stats", timestamp)
        os.makedirs(self.logdir, exist_ok=True)

        print(f"Log directory: {self.logdir}")

        self.runner = Runner_MAPPO_MPE(self.config, self.config.env_id, number=1, seed=self.config.seed)
        self.runner.agent_n.load_model_from_directory(self.config.model_dir)
        self.env = self.runner.env

        self.reference_manager = ReferenceTaylorManager(self.runner, self.env, self.config)
        self.episode_runner = EpisodeRunner(self.runner, self.env, epsilon=self.config.taylor_epsilon)
        self.data_logger = ExperimentDataLogger(self.runner.args.N)
        self.patient_zero_analyzer = PatientZeroAnalyzer(self.runner.args.N)
        self.metrics_computer = AttackMetricsComputer(gamma=self.config.gamma)

        if self.config.taylor_cache_dir:
            self.reference_manager.load_cache()

        print("Experiment setup complete.")
        print(f"Running on device: {DEVICE}")

    def run_single_seed_experiment(self, seed: int):
        assert self.runner and self.env and self.reference_manager and self.episode_runner and self.data_logger and self.patient_zero_analyzer
        print(f"\n{'=' * 60}\nRunning seed {seed}\n{'=' * 60}")

        ref_vals, ref_std_devs = self.reference_manager.get_reference_values(seed)
        normal_episode = self.episode_runner.run_normal_episode(seed, collect_frames=self.config.single_seed)
        episode_length = normal_episode['episode_length']
        self.data_logger.log_cumulative_influences(normal_episode['action_influences_history'], seed, episode_length)
        self.data_logger.log_directional_derivatives(normal_episode['directional_derivatives_history'], seed, episode_length)

        if self.config.single_seed and 'frames' in normal_episode:
            gif_path = os.path.join(self.logdir, f'normal_seed{seed}.gif')
            save_frames_as_gif(normal_episode['frames'], gif_path, fps=10)
            print(f"Saved normal episode GIF: {gif_path}")

        max_steps = math.ceil(ATTACK_TS_FRACTION * episode_length)
        nagents = self.runner.args.N
        
        # Store all pair results for this seed
        all_pair_results = []

        for agent_i in range(nagents):
            for agent_j in range(nagents):
                if agent_i == agent_j:
                    continue

                print(f"\nAnalyzing influence from Agent {agent_i} to Agent {agent_j}")
                max_ts, min_ts = self.influence_analyzer.find_influence_timesteps(
                    normal_episode['action_influences_history'], 
                    normal_episode['directional_derivatives_history'],
                    agent_i, agent_j, max_steps, k_steps=self.config.attack_top_k
                )

                if not max_ts or not min_ts:
                    print(f"  Skipping pair agent_{agent_i} -> agent_{agent_j}: No suitable timesteps found")
                    print(f"    Max influence timesteps (positive derivatives): {max_ts}")
                    print(f"    Min influence timesteps (negative derivatives): {min_ts}")
                    continue
                
                print(f"  Max influence timestep: {max_ts}, Min influence timestep: {min_ts}")

                high_attack_results = self.episode_runner.run_attacked_episode(
                    seed,
                    attack_agent_i=agent_i,
                    attack_timesteps=max_ts,
                    ref_vals=ref_vals,
                    ref_std_devs=ref_std_devs,
                    observe_agent=agent_j,
                    collect_frames=self.config.single_seed,
                )

                low_attack_results = self.episode_runner.run_attacked_episode(
                    seed,
                    attack_agent_i=agent_i,
                    attack_timesteps=min_ts,
                    ref_vals=ref_vals,
                    ref_std_devs=ref_std_devs,
                    observe_agent=agent_j,
                    collect_frames=self.config.single_seed,
                )
                
                # Save low influence attack GIF in single-seed mode
                if self.config.single_seed and 'frames' in low_attack_results:
                    gif_path = os.path.join(self.logdir, f'attack_low_seed{seed}_i{agent_i}_j{agent_j}.gif')
                    save_frames_as_gif(low_attack_results['frames'], gif_path, fps=10)
                    print(f"  Saved low influence attack GIF: {gif_path}")

                normal_values = normal_episode['q_values_history']
                normal_rewards = normal_episode['rewards_history']

                high_metrics = self.metrics_computer.compute_attack_metrics(
                    high_attack_results,
                    normal_values,
                    normal_rewards,
                    ref_vals,
                    ref_std_devs,
                    agent_j,
                )
                low_metrics = self.metrics_computer.compute_attack_metrics(
                    low_attack_results,
                    normal_values,
                    normal_rewards,
                    ref_vals,
                    ref_std_devs,
                    agent_j,
                )
                
                # Log Taylor deviations
                self.data_logger.log_taylor_deviations(
                    high_attack_results, low_attack_results, ref_vals, ref_std_devs,
                    seed, agent_i, agent_j
                )
                
                # Get patient zero detection
                high_patient_zero, high_patient_time = get_patient_zero_detection(
                    high_attack_results['fault_timeline']
                )
                low_patient_zero, low_patient_time = get_patient_zero_detection(
                    low_attack_results['fault_timeline']
                )
                
                # Get fault detection times
                high_influencer_fault_times = self.influence_analyzer.get_agent_fault_detection_times(
                    high_attack_results['fault_timeline'], agent_i
                )
                high_influenced_fault_times = self.influence_analyzer.get_agent_fault_detection_times(
                    high_attack_results['fault_timeline'], agent_j
                )
                low_influencer_fault_times = self.influence_analyzer.get_agent_fault_detection_times(
                    low_attack_results['fault_timeline'], agent_i
                )
                low_influenced_fault_times = self.influence_analyzer.get_agent_fault_detection_times(
                    low_attack_results['fault_timeline'], agent_j
                )
                
                # Patient zero analysis
                print(f"\n  --- Patient Zero Analysis for pair agent_{agent_i} -> agent_{agent_j} ---")

                print(f"  Analyzing HIGH influence attack:")
                high_detection = self.patient_zero_analyzer.analyze_detection_accuracy(
                    high_attack_results['fault_timeline'],
                    attacked_agent=agent_i,
                    attack_timesteps=high_attack_results['attack_timesteps'],
                    directional_derivative_history=high_attack_results['directional_derivatives_history'],
                    taylor_errors_history=high_attack_results['taylor_errors_history'],
                    ref_vals=ref_vals,
                    action_influences_history=high_attack_results['action_influences_history'],
                    seed=seed,
                    agent_pair=(agent_i, agent_j)
                )

                print(f"  Analyzing LOW influence attack:")
                low_detection = self.patient_zero_analyzer.analyze_detection_accuracy(
                    low_attack_results['fault_timeline'],
                    attacked_agent=agent_i,
                    attack_timesteps=low_attack_results['attack_timesteps'],
                    directional_derivative_history=low_attack_results['directional_derivatives_history'],
                    taylor_errors_history=low_attack_results['taylor_errors_history'],
                    ref_vals=ref_vals,
                    action_influences_history=low_attack_results['action_influences_history'],
                    seed=seed,
                    agent_pair=(agent_i, agent_j)
                )
                
                # Display patient zero information
                high_pz_str = ', '.join(map(str, high_patient_zero)) if isinstance(high_patient_zero, list) else str(high_patient_zero)
                low_pz_str = ', '.join(map(str, low_patient_zero)) if isinstance(low_patient_zero, list) else str(low_patient_zero)
                
                print(f"  High influence attack - Patient zero(s): {high_pz_str} at time {high_patient_time}")
                print(f"    Influencer (agent_{agent_i}) fault detection times: {high_influencer_fault_times}")
                print(f"    Influenced (agent_{agent_j}) fault detection times: {high_influenced_fault_times}")
                print(f"  Low influence attack - Patient zero(s): {low_pz_str} at time {low_patient_time}")
                print(f"    Influencer (agent_{agent_i}) fault detection times: {low_influencer_fault_times}")
                print(f"    Influenced (agent_{agent_j}) fault detection times: {low_influenced_fault_times}")

                if self.config.single_seed and 'frames' in high_attack_results:
                    gif_path = os.path.join(self.logdir, f'attack_high_seed{seed}_i{agent_i}_j{agent_j}.gif')
                    save_frames_as_gif(high_attack_results['frames'], gif_path, fps=10)
                    print(f"  Saved high influence attack GIF: {gif_path}")

                # Store structured pair result
                pair_result = {
                    'agent_i': agent_i,
                    'agent_j': agent_j,
                    'high_patient_zero': high_patient_zero,
                    'low_patient_zero': low_patient_zero,
                    'high_influence_attack_timesteps': max_ts,
                    'low_influence_attack_timesteps': min_ts,
                    'high_influence_detection_times': high_influenced_fault_times,
                    'low_influence_detection_times': low_influenced_fault_times,
                    'high_influence_metrics': high_metrics,
                    'low_influence_metrics': low_metrics,
                }
                
                all_pair_results.append(pair_result)
                
                self.results.append(ExperimentResult(
                    influencing_agent=agent_i,
                    influenced_agent=agent_j,
                    high_attack_metrics=high_metrics,
                    low_attack_metrics=low_metrics,
                    high_detection=high_detection,
                    low_detection=low_detection,
                    high_timesteps=max_ts,
                    low_timesteps=min_ts,
                ))
        
        # Store seed result
        seed_result = {
            'seed': seed,
            'episode_length': episode_length,
            'pair_results': all_pair_results,
            'total_pairs': len(all_pair_results)
        }
        
        print(f"\nCompleted analysis for {len(all_pair_results)} agent pairs")
        
        return seed_result

    def run_all_experiments(self):
        """Run experiments for all seeds or single seed."""
        if self.runner is None:
            raise RuntimeError("Runner not initialized. Call setup_experiment() first.")
        
        seeds = [self.config.seed + i for i in range(self.total_experiments)]
        
        if self.config.single_seed:
            print(f"Running single seed experiment with seed {self.config.seed}...")
            result = self.run_single_seed_experiment(self.config.seed)
            self.experiment_results.append(result)
            
            print(f"\nCompleted single seed experiment")
            if self.experiment_results:
                print(f"Total pairs analyzed: {self.experiment_results[0]['total_pairs']}")
        else:
            total_pairs_per_seed = self.runner.args.N * (self.runner.args.N - 1)
            print(f"Starting multi-seed experiments with {self.total_experiments} seeds...")
            print(f"Each seed will analyze {total_pairs_per_seed} agent pairs")
            print(f"Total pairs to analyze: {self.total_experiments * total_pairs_per_seed}")
            
            for seed in tqdm(seeds, desc='Running experiments'):
                try:
                    result = self.run_single_seed_experiment(seed)
                    self.experiment_results.append(result)
                except Exception as exc:  # pylint: disable=broad-except
                    print(f"Seed {seed} failed: {exc}")
                    self.failed_seeds.append(seed)
            
            total_successful_pairs = sum(result['total_pairs'] for result in self.experiment_results)
            print(f"\nCompleted {len(self.experiment_results)} successful experiments out of {self.total_experiments}")
            print(f"Total successful pairs analyzed: {total_successful_pairs}")
            print(f"Failed experiments: {len(self.failed_seeds)}")

    def compute_accuracies(self):
        """Compute accuracies and analyze results using accuracy computer."""
        return AccuracyComputer.compute_experiment_accuracies(
            self.experiment_results,
            self.patient_zero_analyzer
        )
    
    def compute_pair_specific_accuracies(self):
        """Compute pair-specific accuracies using accuracy computer."""
        return AccuracyComputer.compute_pair_specific_accuracies(self.experiment_results)
    
    def save_results(self, accuracy_results, failed_expectations, pair_specific_results=None):
        """Save all results using results saver."""
        # Save accuracy summary
        ResultsSaver.save_accuracy_summary(accuracy_results, self.logdir)
        
        # Save failed expectations
        ResultsSaver.save_failed_expectations(failed_expectations, self.logdir)
        
        # Save pair-specific results
        if pair_specific_results:
            ResultsSaver.save_pair_specific_results(pair_specific_results, self.logdir)
        
        # Save data logger CSV files
        self.data_logger.save_cumulative_influences_csv(self.logdir)
        self.data_logger.save_directional_derivatives_csv(self.logdir)
        self.data_logger.save_taylor_deviations_csv(self.logdir)
        
        # Print final summary
        ResultsSaver.print_final_summary(self.logdir, accuracy_results, pair_specific_results)
    
    def cleanup(self):
        """Clean up resources."""
        if self.env:
            self.env.close()
    
    def run_precomputation_mode(self):
        """Run precomputation mode to generate reference Taylor values."""
        print("Running in precomputation mode...")
        self.setup_experiment()
        
        # Generate seeds for precomputation
        seeds = list(range(self.total_experiments))
        
        # Create precomputation directory
        cwd = os.getcwd()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        precompute_dir = os.path.join(cwd, 'data', 'precomputed_taylor_values',
                                     f"mappo_{self.config.env_id}_nagents{self.runner.args.N}_{timestamp}")
        
        # Precompute and save reference Taylor values using taylor manager
        self.reference_manager.precompute_multiple_seeds(seeds, precompute_dir)
        
        print(f"\nPrecomputation completed successfully!")
        print(f"Precomputed Taylor values saved to: {precompute_dir}")
        print(f"Use --taylor_cache_dir {precompute_dir} to load these values in future experiments")
        
        self.cleanup()
    
    def run_full_experiment(self):
        """Run the complete experiment pipeline (single or multi-seed)."""
        # Check if running in precomputation mode
        if hasattr(self.config, 'precompute_taylor') and self.config.precompute_taylor:
            self.run_precomputation_mode()
            return
        
        self.setup_experiment()
        self.run_all_experiments()
        accuracy_results, failed_expectations = self.compute_accuracies()
        pair_specific_results = self.compute_pair_specific_accuracies()
        ResultsSaver.print_pair_specific_summary(pair_specific_results)
        self.save_results(accuracy_results, failed_expectations, pair_specific_results)
        
        # Patient zero analysis summary and results saving
        self.patient_zero_analyzer.print_summary_dual()
        patient_zero_stats = self.patient_zero_analyzer.get_statistics_dual()
        
        # Save patient zero analysis results
        pz_analysis_file = os.path.join(self.logdir, "patient_zero_analysis_detailed.csv")
        pz_summary_file = os.path.join(self.logdir, "patient_zero_analysis_summary.json")
        
        self.patient_zero_analyzer.save_detailed_results(pz_analysis_file)
        
        # Save summary statistics as JSON
        with open(pz_summary_file, 'w') as f:
            json.dump(patient_zero_stats, f, indent=2)
        
        if self.config.single_seed:
            print(f"\nSingle seed experiment completed successfully!")
            print(f"Seed: {self.config.seed}")
        else:
            print(f"\nMulti-seed experiment completed successfully!")
        
        print(f"Results saved to: {self.logdir}")
        print(f"Pair-specific analysis completed for {len(pair_specific_results)} agent pairs")
        print(f"Patient zero analysis saved to:")
        print(f"  - Detailed results: {pz_analysis_file}")
        print(f"  - Summary statistics: {pz_summary_file}")
        self.cleanup()
    
    def finalize(self):
        """Legacy method for backward compatibility."""
        if not self.logdir:
            return
        results_saver = ResultsSaver(self.logdir)
        results_saver.save_json('experiment_results.json', {
            'results': [result.__dict__ for result in self.results],
            'failed_seeds': self.failed_seeds,
        })

        if self.patient_zero_analyzer:
            summary_path = os.path.join(self.logdir, 'patient_zero_summary.txt')
            with open(summary_path, 'w') as f:
                f.write(self.patient_zero_analyzer.get_summary_dual())
            self.patient_zero_analyzer.save_detailed_results(os.path.join(self.logdir, 'patient_zero_detailed.json'))

        if self.data_logger:
            self.data_logger.save_to_json(os.path.join(self.logdir, 'data_logger.json'))

        print(f"Results saved to {self.logdir}")


def main():
    """Main function to run multi-seed or single-seed statistics experiment."""
    parser = build_arg_parser()
    config = parser.parse_args()

    torch.set_num_threads(1)
    random.seed(config.seed)
    np.random.seed(config.seed)

    runner = MultiSeedExperimentRunner(config)
    runner.run_full_experiment()


if __name__ == '__main__':
    main()
