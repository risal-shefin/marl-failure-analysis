"""
Refactored test detection statistics script with modular architecture.
This script orchestrates the various modules to run fault detection experiments.
"""
import argparse
import os
import math
from datetime import datetime
from algorithms.maddpg import MADDPG

# Import all the modular components
from modules.constants import DEVICE, DEFAULT_INFLUENCE_DECAY_LAMBDA
from modules.environment import create_environment
from modules.data_processing import (
    load_reference_values,
    save_matrix_to_files,
    save_q_values_csv,
    save_q_value_drop_csv,
    save_decayed_action_influence_csv
)
from modules.detection import get_patient_zero_detection, compute_decayed_action_influence
from modules.core_experiment import get_episode_data

# Import modularized plotting functions
from modules.visualization import (
    plot_results,
    plot_frobs,
    plot_frob_norm_influences,
    plot_sec_dir_derivatives,
    plot_action_influences,
    plot_pairwise_action_influences,
    plot_second_order_action_influences,
    plot_pairwise_second_order_action_influences,
    plot_observation_influences,
    plot_pairwise_observation_influences,
    plot_second_order_observation_influences,
    plot_pairwise_second_order_observation_influences,
    plot_fault_timeline,
    plot_fault_timeline_action_influences,
    plot_fault_timeline_action_influences_stacked,
    plot_normal_scenario_action_influences_stacked,
    plot_normal_scenario_frob_norms_stacked,
    plot_attacked_scenario_frob_norms_stacked,
    plot_fault_timeline_second_order_action_influences,
    plot_fault_timeline_observation_influences,
    plot_fault_timeline_second_order_observation_influences,
    plot_contributor_barchart
)


class ExperimentRunner:
    """
    Main experiment runner class that orchestrates the fault detection analysis.
    """
    
    def __init__(self, config):
        """
        Initialize the experiment runner.
        
        Args:
            config: Configuration object containing experiment parameters
        """
        self.config = config
        self.maddpg = None
        self.env = None
        self.logdir = None
        self.ref_vals = None
        self.ref_std_devs = None
        
    def setup_experiment(self):
        """Set up the experiment environment and logging."""
        # Load MADDPG model
        self.maddpg = MADDPG.init_from_save(self.config.model_path, test_mode=True)
        
        # Create log directory
        cwd = os.getcwd()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        env_type = 'discrete' if self.maddpg.discrete_action else 'continuous'
        self.logdir = os.path.join(cwd, 'runs', f"{self.config.env_id}_{env_type}", f"{timestamp}_seed_{self.config.seed}")
        os.makedirs(self.logdir, exist_ok=True)
        
        # Create environment
        self.env = create_environment(self.config, self.maddpg)
        
        # Prepare MADDPG for training mode
        self.maddpg.prep_training(device=DEVICE)
        
        # Load reference values
        self.ref_vals, self.ref_std_devs = load_reference_values(
            self.config.ref_val_dir, 
            self.maddpg.nagents, 
            self.config.detection_method
        )
        
        print(f"Experiment setup complete. Log directory: {self.logdir}")
        
    def run_experiments(self):
        """Run both normal and attacked scenarios."""
        attacked_agent_id = self.config.attack_agent_id
        seed = self.config.seed
        
        print("Running normal scenario...")
        normal_results = get_episode_data(
            self.env, self.maddpg, self.config, self.logdir, 
            self.ref_vals, self.ref_std_devs, self.config.detection_method, 
            do_attack=False, atk_agent_id=attacked_agent_id, seed=seed
        )
        
        print("Running attacked scenario...")
        attacked_results = get_episode_data(
            self.env, self.maddpg, self.config, self.logdir, 
            self.ref_vals, self.ref_std_devs, self.config.detection_method, 
            do_attack=True, atk_agent_id=attacked_agent_id, seed=seed
        )
        
        return normal_results, attacked_results
    
    def save_data(self, normal_results, attacked_results):
        """Save all experimental data to CSV files."""
        attacked_agent_id = self.config.attack_agent_id
        
        # Unpack results
        (results_normal, _, frob_norms_normal, sec_dir_derivatives_normal, 
         frob_norms_matrix_history_normal, _, action_influences_matrix_history_normal, 
         second_order_action_influences_history_normal, observation_influences_matrix_history_normal, 
         second_order_observation_influences_history_normal, q_values_normal) = normal_results
        
        (results_attacked, attacked_steps, frob_norms_atk, sec_dir_derivatives_atk, 
         frob_norms_matrix_history, fault_timeline, action_influences_matrix_history, 
         second_order_action_influences_history, observation_influences_matrix_history, 
         second_order_observation_influences_history, q_values_attacked) = attacked_results
        
        # Save basic metrics
        save_matrix_to_files(results_attacked, attacked_steps, attacked_agent_id, 
                           self.maddpg.nagents, self.logdir, f'maddpg_taylor_error_atk_{attacked_agent_id}.csv')
        save_matrix_to_files(frob_norms_atk, attacked_steps, attacked_agent_id, 
                           self.maddpg.nagents, self.logdir, f'maddpg_frobenius_norms_atk_{attacked_agent_id}.csv')
        save_matrix_to_files(sec_dir_derivatives_atk, attacked_steps, attacked_agent_id, 
                           self.maddpg.nagents, self.logdir, f'maddpg_sec_dir_derivatives_atk_{attacked_agent_id}.csv')
        
        # Save Q-values
        save_q_values_csv(q_values_normal, self.logdir, f'maddpg_q_values_normal_{attacked_agent_id}.csv')
        save_q_values_csv(q_values_attacked, self.logdir, f'maddpg_q_values_attacked_{attacked_agent_id}.csv')
        save_q_value_drop_csv(q_values_normal, q_values_attacked, self.logdir, 
                             f'maddpg_q_value_drop_attacked_{attacked_agent_id}.csv')
        
        # Save decayed influence data
        lambda_decay = getattr(self.config, 'influence_decay_lambda', DEFAULT_INFLUENCE_DECAY_LAMBDA)
        patient_zero_agent, patient_zero_time = get_patient_zero_detection(fault_timeline)
        
        if patient_zero_time is not None and action_influences_matrix_history:
            patient_zero_index = int(patient_zero_time)
            decayed_influence_matrix = compute_decayed_action_influence(
                action_influences_matrix_history, patient_zero_index, lambda_decay
            )
            save_decayed_action_influence_csv(
                decayed_influence_matrix,
                self.logdir,
                f'maddpg_decayed_action_influence_{attacked_agent_id}',
                start_timestep=patient_zero_index
            )
            print(f"Patient zero detected: agent {patient_zero_agent} at timestep {patient_zero_index}. Decayed influence saved.")
        else:
            print("No patient zero detection found; skipping decayed action influence export.")
        
        # Save action influences matrix data
        self._save_action_influence_time_series(action_influences_matrix_history, attacked_steps, attacked_agent_id)
        
        return normal_results, attacked_results
    
    def _save_action_influence_time_series(self, action_influences_matrix_history, attacked_steps, attacked_agent_id):
        """Save action influences as time series for each agent."""
        action_influences_per_agent = []
        for i in range(self.maddpg.nagents):
            agent_i_influences = []
            for t in range(len(action_influences_matrix_history)):
                # For agent i, collect all influences from other agents at time t
                influences_at_t = [action_influences_matrix_history[t][i][j] for j in range(self.maddpg.nagents)]
                agent_i_influences.append(influences_at_t)
            action_influences_per_agent.append(agent_i_influences)
        
        # Save individual influence time series for each agent
        for i in range(self.maddpg.nagents):
            filename = f'maddpg_action_influences_on_agent_{i}_atk_{attacked_agent_id}.csv'
            save_matrix_to_files([action_influences_per_agent[i]], attacked_steps, attacked_agent_id, 
                               self.maddpg.nagents, self.logdir, filename)
    
    def generate_visualizations(self, normal_results, attacked_results):
        """Generate all visualization plots."""
        attacked_agent_id = self.config.attack_agent_id
        
        # Unpack results
        (results_normal, _, frob_norms_normal, sec_dir_derivatives_normal, 
         frob_norms_matrix_history_normal, _, action_influences_matrix_history_normal, 
         second_order_action_influences_history_normal, observation_influences_matrix_history_normal, 
         second_order_observation_influences_history_normal, q_values_normal) = normal_results
        
        (results_attacked, attacked_steps, frob_norms_atk, sec_dir_derivatives_atk, 
         frob_norms_matrix_history, fault_timeline, action_influences_matrix_history, 
         second_order_action_influences_history, observation_influences_matrix_history, 
         second_order_observation_influences_history, q_values_attacked) = attacked_results
        
        print("Generating visualizations...")
        
        # Basic metric plots
        plot_results(results_attacked, attacked_steps, attacked_agent_id, 
                    self.ref_vals, self.ref_std_devs, self.logdir, self.config.detection_method)
        plot_frobs(frob_norms_normal, frob_norms_atk, attacked_steps, attacked_agent_id, self.logdir)
        plot_sec_dir_derivatives(sec_dir_derivatives_normal, sec_dir_derivatives_atk, 
                               attacked_steps, attacked_agent_id, self.logdir)
        
        # Influence plots
        plot_frob_norm_influences(frob_norms_matrix_history_normal, frob_norms_matrix_history, 
                                attacked_steps, attacked_agent_id, self.logdir)
        plot_action_influences(action_influences_matrix_history_normal, action_influences_matrix_history, 
                             attacked_steps, attacked_agent_id, self.logdir)
        plot_pairwise_action_influences(action_influences_matrix_history_normal, action_influences_matrix_history, 
                                      attacked_steps, attacked_agent_id, self.logdir)
        plot_second_order_action_influences(second_order_action_influences_history_normal, 
                                          second_order_action_influences_history, attacked_steps, 
                                          attacked_agent_id, self.logdir)
        plot_pairwise_second_order_action_influences(second_order_action_influences_history_normal, 
                                                   second_order_action_influences_history, attacked_steps, 
                                                   attacked_agent_id, self.logdir)
        
        # Observation influence plots
        plot_observation_influences(observation_influences_matrix_history_normal, observation_influences_matrix_history, 
                                  attacked_steps, attacked_agent_id, self.logdir)
        plot_pairwise_observation_influences(observation_influences_matrix_history_normal, 
                                           observation_influences_matrix_history, attacked_steps, 
                                           attacked_agent_id, self.logdir)
        plot_second_order_observation_influences(second_order_observation_influences_history_normal, 
                                                second_order_observation_influences_history, attacked_steps, 
                                                attacked_agent_id, self.logdir)
        plot_pairwise_second_order_observation_influences(second_order_observation_influences_history_normal, 
                                                        second_order_observation_influences_history, attacked_steps, 
                                                        attacked_agent_id, self.logdir)
        
        # Fault timeline plots
        plot_fault_timeline(fault_timeline, self.maddpg.nagents, self.logdir)
        plot_fault_timeline_action_influences(fault_timeline, action_influences_matrix_history, 
                                             self.maddpg.nagents, self.logdir)
        
        # Stacked timeline plots with comparison
        timesteps, agents = plot_fault_timeline_action_influences_stacked(fault_timeline, action_influences_matrix_history, 
                                                                        self.maddpg.nagents, self.logdir)
        plot_normal_scenario_action_influences_stacked(timesteps, agents, action_influences_matrix_history_normal, 
                                                     self.maddpg.nagents, self.logdir)
        
        # Frobenius norm stacked plots
        plot_normal_scenario_frob_norms_stacked(timesteps, agents, frob_norms_matrix_history_normal, 
                                              self.maddpg.nagents, self.logdir)
        plot_attacked_scenario_frob_norms_stacked(timesteps, agents, frob_norms_matrix_history, 
                                                self.maddpg.nagents, self.logdir)
        
        # Additional timeline plots
        plot_fault_timeline_second_order_action_influences(fault_timeline, second_order_action_influences_history, 
                                                          self.maddpg.nagents, self.logdir)
        plot_fault_timeline_observation_influences(fault_timeline, observation_influences_matrix_history, 
                                                  self.maddpg.nagents, self.logdir)
        plot_fault_timeline_second_order_observation_influences(fault_timeline, second_order_observation_influences_history, 
                                                               self.maddpg.nagents, self.logdir)
        plot_contributor_barchart(fault_timeline, self.maddpg.nagents, self.logdir)
        
        print("All visualizations generated successfully!")
    
    def cleanup(self):
        """Clean up resources."""
        if self.env:
            self.env.close()
    
    def run_full_experiment(self):
        """Run the complete experiment pipeline."""
        self.setup_experiment()
        normal_results, attacked_results = self.run_experiments()
        self.save_data(normal_results, attacked_results)
        self.generate_visualizations(normal_results, attacked_results)
        print(f"Experiment completed successfully! Results saved to: {self.logdir}")
        self.cleanup()


def create_config_from_args():
    """Create configuration from command line arguments."""
    parser = argparse.ArgumentParser(description="Modular fault detection analysis")
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_path", help="Model directory")
    parser.add_argument("--save_gifs", action="store_true", 
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--ref_val_dir", type=str, default='', 
                        help="Directory containing reference values")
    parser.add_argument("--attack_agent_id", type=int, default=0, 
                        help="ID of agent to attack")
    parser.add_argument("--atk_start_step", type=int, default=-math.inf, 
                        help="Attack start step")
    parser.add_argument("--atk_end_step", type=int, default=math.inf, 
                        help="Attack end step")
    parser.add_argument("--detection_method", type=str, default='mean_std', 
                        choices=['mean_std', 'median_mad', 'diff'],
                        help="Detection method to use")
    parser.add_argument("--seed", type=int, default=23, 
                        help="Random seed")
    parser.add_argument("--influence_decay_lambda", type=float, default=DEFAULT_INFLUENCE_DECAY_LAMBDA,
                        help="Exponential decay lambda for action influences")
    
    return parser.parse_args()


def run_single_experiment(config=None):
    """
    Run a single experiment with the given configuration.
    
    Args:
        config: Configuration object. If None, will parse from command line arguments.
    """
    if config is None:
        config = create_config_from_args()
    
    runner = ExperimentRunner(config)
    runner.run_full_experiment()


def run_multiple_experiments(base_config, seeds, attack_agent_ids=None):
    """
    Run multiple experiments with different seeds and/or attack agents.
    
    Args:
        base_config: Base configuration object
        seeds: List of random seeds to use
        attack_agent_ids: Optional list of agent IDs to attack. If None, uses base_config.attack_agent_id
    """
    if attack_agent_ids is None:
        attack_agent_ids = [base_config.attack_agent_id]
    
    results = {}
    
    for seed in seeds:
        for attack_agent_id in attack_agent_ids:
            print(f"\n{'='*60}")
            print(f"Running experiment with seed={seed}, attack_agent_id={attack_agent_id}")
            print(f"{'='*60}")
            
            # Create a copy of config with updated parameters
            config_copy = create_config_from_args() if hasattr(base_config, '__dict__') else base_config
            config_copy.seed = seed
            config_copy.attack_agent_id = attack_agent_id
            
            runner = ExperimentRunner(config_copy)
            try:
                runner.setup_experiment()
                normal_results, attacked_results = runner.run_experiments()
                runner.save_data(normal_results, attacked_results)
                runner.generate_visualizations(normal_results, attacked_results)
                
                # Store results for analysis
                key = f"seed_{seed}_agent_{attack_agent_id}"
                results[key] = {
                    'normal_results': normal_results,
                    'attacked_results': attacked_results,
                    'logdir': runner.logdir
                }
                print(f"Experiment {key} completed successfully!")
                
            except Exception as e:
                print(f"Error in experiment {key}: {e}")
                results[key] = {'error': str(e)}
            finally:
                runner.cleanup()
    
    print(f"\nAll experiments completed! Results for {len(results)} configurations.")
    return results


if __name__ == '__main__':
    run_single_experiment()