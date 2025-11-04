#!/usr/bin/env python3
"""Multi-seed fault analysis statistics for MAPPO agents in MultiGrid environments."""
from __future__ import annotations

import argparse
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

MULTIGRID_ENV_ROOT = PROJECT_ROOT / 'maddpg-pytorch'
if MULTIGRID_ENV_ROOT.exists() and str(MULTIGRID_ENV_ROOT) not in sys.path:
    sys.path.append(str(MULTIGRID_ENV_ROOT))

from MAPPO_Multigrid_main import Runner_MAPPO_Multigrid
from MAPPO_MPE.modules.analysis import InfluenceAnalyzer
from MAPPO_MPE.modules.constants import DEFAULT_INFLUENCE_DECAY_LAMBDA, DEVICE
from MAPPO_MPE.modules.experiments import EpisodeRunner, ExperimentDataLogger, ReferenceTaylorManager
from MAPPO_MPE.modules.results import ResultsSaver
from MAPPO_MPE.modules.traceback import PatientZeroAnalyzer
from MAPPO_MPE.modules.visualization import save_frames_as_gif

ATTACK_TS_FRACTION = 0.25


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("MAPPO Fault Analysis Statistics (MultiGrid)")
    parser.add_argument('--model_dir', type=str, required=True, help='Path to saved MAPPO model (.pt)')
    parser.add_argument('--env_id', type=str, required=True, choices=['soccer', 'collect'], help='MultiGrid environment ID')
    parser.add_argument('--flatten_obs', action='store_true', help='Flatten observations from the grid environment')
    parser.add_argument('--seed', type=int, default=42, help='Base random seed')
    parser.add_argument('--total_experiments', type=int, default=5, help='Number of seeds to evaluate')
    parser.add_argument('--single_seed', action='store_true', help='Run a single-seed analysis with visualizations')
    parser.add_argument('--ref_episodes', type=int, default=10, help='Episodes to compute reference Taylor statistics')
    parser.add_argument('--taylor_epsilon', type=float, default=0.01, help='Perturbation magnitude for Taylor analysis')
    parser.add_argument('--taylor_cache_dir', type=str, default=None, help='Directory to cache reference Taylor statistics')
    parser.add_argument('--output_dir', type=str, default='./results', help='Base output directory')
    parser.add_argument('--attack_top_k', type=int, default=1, help='Number of influential timesteps to attack')
    parser.add_argument('--lambda_decay', type=float, default=DEFAULT_INFLUENCE_DECAY_LAMBDA, help='Decay factor for influence aggregation')

    # MAPPO runner hyperparameters
    parser.add_argument('--max_train_steps', type=int, default=int(3e6))
    parser.add_argument('--episode_limit', type=int, default=120)
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
        self.runner: Optional[Runner_MAPPO_Multigrid] = None
        self.env = None
        self.logdir: Optional[str] = None
        self.total_experiments = config.total_experiments

        self.reference_manager: Optional[ReferenceTaylorManager] = None
        self.episode_runner: Optional[EpisodeRunner] = None
        self.influence_analyzer = InfluenceAnalyzer()
        self.data_logger: Optional[ExperimentDataLogger] = None
        self.patient_zero_analyzer: Optional[PatientZeroAnalyzer] = None

        self.results: List[ExperimentResult] = []
        self.failed_seeds: List[int] = []

    def setup_experiment(self):
        cwd = os.getcwd()
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        mode = 'single_seed' if self.config.single_seed else 'multi_seed'
        self.logdir = os.path.join(cwd, 'runs', f"multigrid_{self.config.env_id}_mappo_{mode}_fault_stats", timestamp)
        os.makedirs(self.logdir, exist_ok=True)

        print(f"Log directory: {self.logdir}")

        self.runner = Runner_MAPPO_Multigrid(self.config, self.config.env_id, number=1, seed=self.config.seed, flatten_obs=self.config.flatten_obs)
        self.runner.agent_n.load_model_from_directory(self.config.model_dir)
        self.env = self.runner.env

        self.reference_manager = ReferenceTaylorManager(self.runner, self.env, self.config)
        self.episode_runner = EpisodeRunner(self.runner, self.env, epsilon=self.config.taylor_epsilon)
        self.data_logger = ExperimentDataLogger(self.runner.args.N)
        self.patient_zero_analyzer = PatientZeroAnalyzer(self.runner.args.N)

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

        for agent_i in range(nagents):
            for agent_j in range(nagents):
                if agent_i == agent_j:
                    continue

                print(f"\nAnalyzing influence from Agent {agent_i} to Agent {agent_j}")
                max_ts, min_ts = self.influence_analyzer.find_influence_timesteps(
                    normal_episode['action_influences_history'], agent_i, agent_j, max_steps, k_steps=self.config.attack_top_k
                )

                if not max_ts:
                    print("  No influential timesteps identified; skipping pair.")
                    continue

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
                    attack_timesteps=min_ts if min_ts else max_ts,
                    ref_vals=ref_vals,
                    ref_std_devs=ref_std_devs,
                    observe_agent=agent_j,
                    collect_frames=False,
                )

                normal_values = normal_episode['q_values_history']
                normal_rewards = normal_episode['rewards_history']

                high_metrics = self.episode_runner.compute_attack_metrics(
                    high_attack_results,
                    normal_values,
                    normal_rewards,
                    ref_vals,
                    ref_std_devs,
                    observe_agent=agent_j,
                )
                low_metrics = self.episode_runner.compute_attack_metrics(
                    low_attack_results,
                    normal_values,
                    normal_rewards,
                    ref_vals,
                    ref_std_devs,
                    observe_agent=agent_j,
                )

                self.data_logger.log_attack_metrics(high_metrics, {
                    'seed': seed,
                    'mode': 'high',
                    'agent_i': agent_i,
                    'agent_j': agent_j,
                })
                self.data_logger.log_attack_metrics(low_metrics, {
                    'seed': seed,
                    'mode': 'low',
                    'agent_i': agent_i,
                    'agent_j': agent_j,
                })

                high_detection = self.patient_zero_analyzer.analyze_detection_accuracy(
                    high_attack_results['fault_timeline'],
                    attacked_agent=agent_i,
                    attack_timesteps=max_ts,
                    directional_derivative_history=high_attack_results['directional_derivatives_history'],
                    taylor_errors_history=high_attack_results['taylor_errors_history'],
                    ref_vals=ref_vals,
                    action_influences_history=high_attack_results['action_influences_history'],
                    seed=seed,
                    agent_pair=(agent_i, agent_j)
                )

                low_detection = self.patient_zero_analyzer.analyze_detection_accuracy(
                    low_attack_results['fault_timeline'],
                    attacked_agent=agent_i,
                    attack_timesteps=min_ts if min_ts else max_ts,
                    directional_derivative_history=low_attack_results['directional_derivatives_history'],
                    taylor_errors_history=low_attack_results['taylor_errors_history'],
                    ref_vals=ref_vals,
                    action_influences_history=low_attack_results['action_influences_history'],
                    seed=seed,
                    agent_pair=(agent_i, agent_j)
                )

                if self.config.single_seed and 'frames' in high_attack_results:
                    gif_path = os.path.join(self.logdir, f'attack_high_seed{seed}_i{agent_i}_j{agent_j}.gif')
                    save_frames_as_gif(high_attack_results['frames'], gif_path, fps=10)

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

    def run(self):
        seeds = [self.config.seed + i for i in range(self.total_experiments)]
        for seed in tqdm(seeds, desc='Seeds'):
            try:
                self.run_single_seed_experiment(seed)
            except Exception as exc:  # pylint: disable=broad-except
                print(f"Seed {seed} failed: {exc}")
                self.failed_seeds.append(seed)

    def finalize(self):
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
    parser = build_arg_parser()
    config = parser.parse_args()

    torch.set_num_threads(1)
    random.seed(config.seed)
    np.random.seed(config.seed)

    runner = MultiSeedExperimentRunner(config)
    runner.setup_experiment()
    runner.run()
    runner.finalize()


if __name__ == '__main__':
    main()
