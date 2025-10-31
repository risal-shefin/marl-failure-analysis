"""
Multi-seed statistics experiment for analyzing action influence-based attacks.
This script performs experiments across multiple seeds to evaluate the effectiveness
of attacking at high vs low influence timesteps.

Adapted for gym_multigrid_wrapper environments (soccer, collect, etc.)

REFACTORED VERSION - Uses modular components from modules/ directory.
"""
import argparse
import os
import math
import json
from datetime import datetime
from tqdm import tqdm

from algorithms.maddpg import MADDPG
from utils.gym_multigrid_wrapper import GymMultiGridWrapper

# Import all the modular components
from modules.constants import DEVICE
from modules.detection import get_patient_zero_detection
from modules.traceback import PatientZeroAnalyzer

# Import new modular components from organized subfolders
from modules.experiments import ReferenceTaylorManager, EpisodeRunner, ExperimentDataLogger
from modules.analysis import InfluenceAnalyzer
from modules.results import AccuracyComputer, ResultsSaver
from modules.visualization.utils import save_frames_as_gif
from modules.metrics import AttackMetricsComputer

ATTACK_TS_FRACTION = 0.25  # Fraction of episode to consider for attack timesteps


class MultiSeedExperimentRunner:
    """
    Multi-seed experiment runner for analyzing influence-based attacks.
    """
    
    def __init__(self, config):
        """
        Initialize the multi-seed experiment runner.
        
        Args:
            config: Configuration object containing experiment parameters
        """
        self.config = config
        self.maddpg = None
        self.env = None
        self.logdir = None
        self.total_experiments = config.total_experiments
        
        # Results storage
        self.experiment_results = []
        self.failed_seeds = []
        
        # Initialize modular components (will be set up after experiment setup)
        self.taylor_manager = None
        self.episode_runner = None
        self.influence_analyzer = None
        self.metrics_computer = None
        self.data_logger = None
        self.patient_zero_analyzer = None
        
    def setup_experiment(self):
        """Set up the experiment environment and logging."""
        # Load MADDPG model
        self.maddpg = MADDPG.init_from_save(self.config.model_path)
        
        # Create log directory
        cwd = os.getcwd()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        env_type = 'discrete' if self.maddpg.discrete_action else 'continuous'
        
        if self.config.single_seed:
            self.logdir = os.path.join(cwd, 'runs', f"{self.config.env_id}_{env_type}_single_seed_detection_stats", 
                                      f"{timestamp}_nagents{self.maddpg.nagents}_seed{self.config.seed}")
        else:
            self.logdir = os.path.join(cwd, 'runs', f"{self.config.env_id}_{env_type}_multi_seed_detection_stats", 
                                      f"{timestamp}_nagents{self.maddpg.nagents}_total_experiments{self.total_experiments}")
        os.makedirs(self.logdir, exist_ok=True)
        
        # Create environment using GymMultiGridWrapper
        self.env = GymMultiGridWrapper.make_and_wrap_env(self.config.env_id, do_flat_obs=True)
        
        # Prepare MADDPG for training mode
        device_str = 'gpu' if DEVICE == 'gpu' else 'cpu'
        self.maddpg.prep_training(device=device_str)
        
        # Initialize modular components
        self.taylor_manager = ReferenceTaylorManager(self.maddpg, self.env, self.config)
        self.episode_runner = EpisodeRunner(self.maddpg, self.env)
        self.influence_analyzer = InfluenceAnalyzer()
        self.metrics_computer = AttackMetricsComputer(gamma=0.99)
        self.data_logger = ExperimentDataLogger(self.maddpg.nagents)
        self.patient_zero_analyzer = PatientZeroAnalyzer(self.maddpg.nagents)
        
        # Load reference Taylor values cache if directory is provided
        if hasattr(self.config, 'taylor_ref_dir') and self.config.taylor_ref_dir is not None:
            self.taylor_manager.load_cache()
        
        if self.config.single_seed:
            print(f"Single seed experiment setup complete. Log directory: {self.logdir}")
            print(f"Running experiment with seed: {self.config.seed}")
        else:
            print(f"Multi-seed experiment setup complete. Log directory: {self.logdir}")
            print(f"Will run {self.total_experiments} experiments")
    
    def run_single_seed_experiment(self, seed):
        """
        Run complete experiment for a single seed.
        
        Args:
            seed: Random seed for the experiment
            
        Returns:
            Dictionary containing experiment results for all agent pairs
        """
        print(f"\n{'='*50}")
        print(f"Running experiment for seed {seed}")
        print(f"{'='*50}")
        
        # Step 1: Get reference Taylor error using taylor manager
        ref_vals, ref_std_devs = self.taylor_manager.get_reference_values(seed)
        
        # Step 2: Run normal episode using episode runner
        # Enable frame collection in single-seed mode for visualization
        collect_frames = self.config.single_seed
        normal_episode = self.episode_runner.run_normal_episode(seed, collect_frames=collect_frames)
        action_influences_history = normal_episode['action_influences_history']
        directional_derivatives_history = normal_episode['directional_derivatives_history']
        normal_q_values_history = normal_episode['q_values_history']
        normal_rewards_history = normal_episode['rewards_history']
        episode_length = normal_episode['episode_length']
        
        # Save normal episode GIF in single-seed mode
        if collect_frames and 'frames' in normal_episode:
            gif_path = os.path.join(self.logdir, f"normal_episode_seed{seed}.gif")
            save_frames_as_gif(normal_episode['frames'], gif_path, fps=10)
            print(f"Saved normal episode GIF to: {gif_path}")
        
        # Step 2.5: Log cumulative influences for each agent pair
        self.data_logger.log_cumulative_influences(action_influences_history, seed, episode_length)
        
        # Step 2.6: Log directional derivatives for each agent pair
        self.data_logger.log_directional_derivatives(directional_derivatives_history, seed, episode_length)
        
        # Step 3: Analyze all possible ordered pairs (i, j) where i influences j
        all_pair_results = []
        atk_steps_limit = math.ceil(ATTACK_TS_FRACTION * episode_length)
        
        for agent_i in range(self.maddpg.nagents):  # influencing agent
            for agent_j in range(self.maddpg.nagents):  # influenced agent
                if agent_i == agent_j:
                    continue  # Skip self
                
                print(f"\nAnalyzing pair: agent_{agent_i} influences agent_{agent_j}")
                
                # Step 4: Find max and min influence timesteps using influence analyzer
                max_influence_t, min_influence_t = self.influence_analyzer.find_influence_timesteps(
                    action_influences_history, directional_derivatives_history, 
                    agent_i, agent_j, atk_steps_limit, 1
                )
                
                # Skip this iteration if no suitable timesteps are found
                if not max_influence_t or not min_influence_t:
                    print(f"Skipping pair agent_{agent_i} -> agent_{agent_j}: No suitable timesteps found")
                    print(f"  Max influence timesteps (positive derivatives): {max_influence_t}")
                    print(f"  Min influence timesteps (negative derivatives): {min_influence_t}")
                    continue
                
                print(f"Max influence timestep: {max_influence_t}, Min influence timestep: {min_influence_t}")
                
                # Step 5: Run attacked episodes using episode runner
                high_influence_attack = self.episode_runner.run_attacked_episode(
                    seed, agent_i, max_influence_t, ref_vals, ref_std_devs, 
                    observe_agent=agent_j, collect_frames=collect_frames
                )
                
                low_influence_attack = self.episode_runner.run_attacked_episode(
                    seed, agent_i, min_influence_t, ref_vals, ref_std_devs, 
                    observe_agent=agent_j, collect_frames=collect_frames
                )
                
                # Save attacked episode GIFs in single-seed mode
                if collect_frames:
                    if 'frames' in high_influence_attack:
                        gif_path = os.path.join(self.logdir, 
                            f"high_influence_attack_seed{seed}_agent{agent_i}_to_agent{agent_j}.gif")
                        save_frames_as_gif(high_influence_attack['frames'], gif_path, fps=10)
                        print(f"Saved high influence attack GIF to: {gif_path}")
                    
                    if 'frames' in low_influence_attack:
                        gif_path = os.path.join(self.logdir, 
                            f"low_influence_attack_seed{seed}_agent{agent_i}_to_agent{agent_j}.gif")
                        save_frames_as_gif(low_influence_attack['frames'], gif_path, fps=10)
                        print(f"Saved low influence attack GIF to: {gif_path}")
                
                # Determine patient zero for each attack
                high_patient_zero, high_patient_time = get_patient_zero_detection(
                    high_influence_attack['fault_timeline']
                )
                low_patient_zero, low_patient_time = get_patient_zero_detection(
                    low_influence_attack['fault_timeline']
                )
                
                # Get fault detection times using influence analyzer
                high_influencer_fault_times = self.influence_analyzer.get_agent_fault_detection_times(
                    high_influence_attack['fault_timeline'], agent_i
                )
                high_influenced_fault_times = self.influence_analyzer.get_agent_fault_detection_times(
                    high_influence_attack['fault_timeline'], agent_j
                )
                low_influencer_fault_times = self.influence_analyzer.get_agent_fault_detection_times(
                    low_influence_attack['fault_timeline'], agent_i
                )
                low_influenced_fault_times = self.influence_analyzer.get_agent_fault_detection_times(
                    low_influence_attack['fault_timeline'], agent_j
                )
                
                # Compute attack metrics using metrics computer
                high_metrics = self.metrics_computer.compute_attack_metrics(
                    high_influence_attack, normal_q_values_history, normal_rewards_history, 
                    ref_vals, ref_std_devs, agent_j
                )
                low_metrics = self.metrics_computer.compute_attack_metrics(
                    low_influence_attack, normal_q_values_history, normal_rewards_history, 
                    ref_vals, ref_std_devs, agent_j
                )
                
                # Log Taylor deviations using data logger
                self.data_logger.log_taylor_deviations(
                    high_influence_attack, low_influence_attack, ref_vals, ref_std_devs, 
                    seed, agent_i, agent_j
                )
                
                # Step 6: Analyze patient zero detection accuracy with traceback
                print(f"\n--- Patient Zero Analysis for pair agent_{agent_i} -> agent_{agent_j} ---")
                
                # Analyze high influence attack
                print(f"Analyzing HIGH influence attack:")
                high_pz_detection_analysis = self.patient_zero_analyzer.analyze_detection_accuracy(
                    high_influence_attack['fault_timeline'],
                    agent_i,
                    high_influence_attack['attack_timesteps'],
                    directional_derivatives_history,
                    high_influence_attack['taylor_errors_history'],
                    ref_vals,
                    action_influences_history,
                    seed,
                    (agent_i, agent_j)
                )
                
                # Analyze low influence attack
                print(f"Analyzing LOW influence attack:")
                low_pz_detection_analysis = self.patient_zero_analyzer.analyze_detection_accuracy(
                    low_influence_attack['fault_timeline'],
                    agent_i,
                    low_influence_attack['attack_timesteps'],
                    directional_derivatives_history,
                    low_influence_attack['taylor_errors_history'],
                    ref_vals,
                    action_influences_history,
                    seed,
                    (agent_i, agent_j)
                )
                
                pair_result = {
                    'agent_i': agent_i,
                    'agent_j': agent_j,
                    'high_influence_attack_timesteps': max_influence_t,
                    'low_influence_attack_timesteps': min_influence_t,
                    'high_influence_detection_times': high_influenced_fault_times,
                    'low_influence_detection_times': low_influenced_fault_times,
                    'high_influence_metrics': high_metrics,
                    'low_influence_metrics': low_metrics,
                }
                
                all_pair_results.append(pair_result)
                
                # Display patient zero information
                high_pz_str = ', '.join(map(str, high_patient_zero)) if isinstance(high_patient_zero, list) else str(high_patient_zero)
                low_pz_str = ', '.join(map(str, low_patient_zero)) if isinstance(low_patient_zero, list) else str(low_patient_zero)
                
                print(f"High influence attack - Patient zero(s): {high_pz_str} at time {high_patient_time}")
                print(f"  Influencer (agent_{agent_i}) fault detection times: {high_influencer_fault_times}")
                print(f"  Influenced (agent_{agent_j}) fault detection times: {high_influenced_fault_times}")
                print(f"Low influence attack - Patient zero(s): {low_pz_str} at time {low_patient_time}")
                print(f"  Influencer (agent_{agent_i}) fault detection times: {low_influencer_fault_times}")
                print(f"  Influenced (agent_{agent_j}) fault detection times: {low_influenced_fault_times}")
        
        result = {
            'seed': seed,
            'episode_length': episode_length,
            'pair_results': all_pair_results,
            'total_pairs': len(all_pair_results)
        }
        
        print(f"\nCompleted analysis for {len(all_pair_results)} agent pairs")
        
        return result
    
    def run_all_experiments(self):
        """Run experiments for all seeds or single seed."""
        if self.maddpg is None:
            raise RuntimeError("MADDPG model not loaded. Call setup_experiment() first.")
            
        if self.config.single_seed:
            print(f"Running single seed experiment with seed {self.config.seed}...")
            result = self.run_single_seed_experiment(self.config.seed)
            self.experiment_results.append(result)
            
            print(f"\nCompleted single seed experiment")
            if self.experiment_results:
                print(f"Total pairs analyzed: {self.experiment_results[0]['total_pairs']}")
        else:
            total_pairs_per_seed = self.maddpg.nagents * (self.maddpg.nagents - 1)
            print(f"Starting multi-seed experiments with {self.total_experiments} seeds...")
            print(f"Each seed will analyze {total_pairs_per_seed} agent pairs")
            print(f"Total pairs to analyze: {self.total_experiments * total_pairs_per_seed}")
            
            for seed in tqdm(range(self.total_experiments), desc="Running experiments"):
                result = self.run_single_seed_experiment(seed)
                self.experiment_results.append(result)
            
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
        env_type = 'discrete' if self.maddpg.discrete_action else 'continuous'
        precompute_dir = os.path.join(cwd, 'data', 'precomputed_taylor_values', 
                                     f"{self.config.env_id}_{env_type}_nagents{self.maddpg.nagents}_{timestamp}")
        
        # Precompute and save reference Taylor values using taylor manager
        self.taylor_manager.precompute_multiple_seeds(seeds, precompute_dir)
        
        print(f"\nPrecomputation completed successfully!")
        print(f"Precomputed Taylor values saved to: {precompute_dir}")
        print(f"Use --taylor_ref_dir {precompute_dir} to load these values in future experiments")
        
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


def create_config_from_args():
    """Create configuration from command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-seed or single-seed statistics experiment for gym_multigrid environments"
    )
    parser.add_argument("env_id", help="Name of environment (e.g., 'soccer', 'collect')")
    parser.add_argument("model_path", help="Model directory")
    parser.add_argument("--total_experiments", type=int, default=100,
                        help="Total number of seed experiments to run (for multi-seed mode)")
    parser.add_argument("--single_seed", action="store_true",
                        help="Run single seed experiment instead of multi-seed")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for single seed experiment (only used if --single_seed is set)")
    parser.add_argument("--precompute_taylor", action="store_true",
                        help="Precompute reference Taylor error values and save them")
    parser.add_argument("--taylor_ref_dir", type=str, default=None,
                        help="Directory containing precomputed reference Taylor error values")
    
    return parser.parse_args()


def main():
    """Main function to run multi-seed or single-seed statistics experiment."""
    config = create_config_from_args()
    runner = MultiSeedExperimentRunner(config)
    runner.run_full_experiment()


if __name__ == '__main__':
    main()
