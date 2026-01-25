"""
Multi-seed statistics experiment for analyzing action influence-based attacks.
This script performs experiments across multiple seeds to evaluate the effectiveness
of attacking at high vs low influence timesteps.
"""
import argparse
import os
import math
import csv
import random
import numpy as np
import torch
from datetime import datetime
from collections import deque
from tqdm import tqdm
from torch.autograd import Variable
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import traceback
from harl.utils.configs_tools import get_defaults_yaml_args, update_args
from harl.utils.trans_tools import _t2n 
import time
# Import all the modular components
from modules.constants import DEVICE, K_SIGMA, torch_device,REWARD, FILEPATH, ATTACK_ID
# from modules.environment import create_environment
from modules.detection import get_patient_zero_detection
# from modules.core_experiment import get_episode_data
# from modules.metrics import (
#     compute_taylor_delta_policy,
#     compute_pairwise_action_influences,f
#     collect_agent_q_values
# )
from harl.utils.envs_tools import (
    make_eval_env
)

from modules.traceback import PatientZeroAnalyzer

import warnings
warnings.filterwarnings("ignore")

THRESHOLD = 0.00001  # Threshold for anomaly detection in Taylor error
ATTACK_RATIO = 1.0  # Fraction of episode length to consider for high/low influence attacks
def slice_avail(avail, agent_id):
    """Extract available actions for a specific agent"""
    if avail is None:
        return None
    first = avail[0]
    if first is None:
        return None
    return avail[:, agent_id]


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
        self.runner = None
        self.env = None
        self.logdir = None
        # self.total_experiments = config.total_experiments
        self.total_experiments= self.config.total_experiments #400 # This is bascially number of seeds --> Episoded
        self.total_episodes = self.config.total_episodes #100 # This is for calculating taylor reference values
        self.K_SIGMA = self.config.K_SIGMA
        self.gamma = 0.99  # Discount factor for weighted metrics
        self.runner_args = None
        self.runner_algo_args = None
        self.runner_env_args = None
        self.folder_name = self.config.folder_name if hasattr(self.config, 'folder_name') else "test-run-K-New"
        # Results storage
        self.experiment_results = []
        self.failed_seeds = []
        self.cumulative_influences_data = []  # Store cumulative influence data for all seeds
        self.patient_zero_analyzer = None
        
    def log_directional_derivatives(self, directional_derivatives_history, seed, episode_length):
        """
        Store directional second derivatives for each agent pair at each timestep.
        Data will be written to a single CSV file later.
        
        Args:
            directional_derivatives_history: List of directional second derivative matrices for each timestep
            seed: Random seed for this experiment
            episode_length: Length of the episode
        """
        if not hasattr(self, 'directional_derivatives_data'):
            self.directional_derivatives_data = []
            
        # print(f"Storing directional derivatives for seed {seed}...")
        
        # For each agent pair (i, j), store directional derivatives at each timestep
        for agent_i in range(self.runner.num_agents):
            for agent_j in range(self.runner.num_agents):
                if agent_i == agent_j:
                    continue  # Skip self-influence
                
                # Get directional derivatives for each timestep
                timestep_derivative_values = []
                
                for t in range(min(episode_length, len(directional_derivatives_history))):
                    # Get directional second derivative of agent_i on agent_j at timestep t
                    # directional_derivatives_history[t][j][i] = g^T H g for influence of i on j
                    derivative_value = directional_derivatives_history[t][agent_j][agent_i]
                    timestep_derivative_values.append(derivative_value)
                
                # Store the data for this agent pair
                pair_data = {
                    'seed': seed,
                    'influencer_agent_id': agent_i,
                    'influenced_agent_id': agent_j,
                    'episode_length': episode_length,
                    'derivative_values': timestep_derivative_values
                }
                
                self.directional_derivatives_data.append(pair_data)
        
        # print(f"Stored directional derivatives for seed {seed} ({self.runner.num_agents * (self.runner.num_agents - 1)} agent pairs)")
    
    def log_taylor_deviations(self, high_attack_results, low_attack_results, ref_vals, ref_std_devs, seed, agent_i, agent_j):
        """
        Store Taylor deviations from mean for both high and low influence attacks for each agent.
        Data will be written to a single CSV file later.
        
        Args:
            high_attack_results: Results from high influence attack episode
            low_attack_results: Results from low influence attack episode
            ref_vals: Reference Taylor error values
            ref_std_devs: Reference standard deviations
            seed: Random seed for this experiment
            agent_i: Influencing agent (attacked agent)
            agent_j: Influenced agent (observed agent)
        """
        # Initialize storage for Taylor deviations if not exists
        if not hasattr(self, 'taylor_deviations_data'):
            self.taylor_deviations_data = []
        
        # Process both high and low influence attacks
        for attack_type, attack_results in [('high', high_attack_results), ('low', low_attack_results)]:
            taylor_errors_history = attack_results['taylor_errors_history']
            episode_length = attack_results['episode_length']
            
            # For each agent, compute Taylor deviations from mean at each timestep
            for agent_id in range(self.runner.num_agents):
                timestep_deviations = []
                
                for t in range(min(episode_length, len(taylor_errors_history))):
                    if t < len(ref_vals[agent_id]) and t < len(ref_std_devs[agent_id]):
                        taylor_error = taylor_errors_history[t][agent_id]
                        ref_mean = ref_vals[agent_id][t]
                        taylor_deviation = abs(taylor_error - ref_mean)
                        timestep_deviations.append(taylor_deviation)
                    else:
                        timestep_deviations.append(0.0)  # Default value if reference not available
                
                # Store the data for this agent
                agent_data = {
                    'seed': seed,
                    'attack_type': attack_type,
                    'influencer_agent_id': agent_i,
                    'influenced_agent_id': agent_j,
                    'observed_agent_id': agent_id,
                    'episode_length': episode_length,
                    'taylor_deviations': timestep_deviations
                }
                
                self.taylor_deviations_data.append(agent_data)

        print(f"Stored Taylor deviations for seed {seed}, pair ({agent_i} -> {agent_j}) for {self.runner.num_agents} agents (both attack types)")

    def log_cumulative_influences(self, action_influences_history, seed, episode_length):
        """
        Store cumulative influence sums for each agent pair at each timestep.
        Data will be written to a single CSV file later.
        
        Args:
            action_influences_history: List of action influence matrices for each timestep
            seed: Random seed for this experiment
            episode_length: Length of the episode
        """
        print(f"Storing cumulative influences for seed {seed}...")
        
        # For each agent pair (i, j), compute and store cumulative influences
        for agent_i in range(self.runner.num_agents):
            for agent_j in range(self.runner.num_agents):
                if agent_i == agent_j:
                    continue  # Skip self-influence
                
                # Compute cumulative influence sum at each timestep
                cumulative_influence = 0.0
                timestep_cumulative_values = []
                
                for t in range(min(episode_length, len(action_influences_history))):
                    # Get influence of agent_i on agent_j at timestep t
                    # action_influences_history[t][j][i] = influence of i on j
                    influence_value = action_influences_history[t][agent_j][agent_i]
                    cumulative_influence += influence_value  # Use absolute value for cumulative sum
                    timestep_cumulative_values.append(cumulative_influence)
                
                # Store the data for this agent pair
                pair_data = {
                    'seed': seed,
                    'influencer_agent_id': agent_i,
                    'influenced_agent_id': agent_j,
                    'episode_length': episode_length,
                    'cumulative_values': timestep_cumulative_values
                }
                
                self.cumulative_influences_data.append(pair_data)

        print(f"Stored cumulative influences for seed {seed} ({self.runner.num_agents * (self.runner.num_agents - 1)} agent pairs)")
    
    def save_directional_derivatives_csv(self):
        """
        Save all directional derivatives data to a single CSV file.
        """
        print("Saving directional derivatives to single CSV file...")
        
        # Find the maximum episode length to determine number of timestep columns
        max_episode_length = max(data['episode_length'] for data in self.directional_derivatives_data)
        
        # Create timestep column names
        timestep_columns = [f'timestep_{t}' for t in range(max_episode_length)]
        
        # Create CSV filename
        csv_filename = 'directional_derivatives_all_seeds.csv'
        csv_filepath = os.path.join(self.logdir, csv_filename)
        
        # Write to CSV file
        with open(csv_filepath, 'w', newline='') as csvfile:
            fieldnames = ['seed', 'influencer_agent_id', 'influenced_agent_id'] + timestep_columns
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for pair_data in self.directional_derivatives_data:
                row = {
                    'seed': pair_data['seed'],
                    'influencer_agent_id': pair_data['influencer_agent_id'],
                    'influenced_agent_id': pair_data['influenced_agent_id']
                }
                
                # Add derivative values for each timestep
                derivative_values = pair_data['derivative_values']
                for t in range(max_episode_length):
                    if t < len(derivative_values):
                        row[f'timestep_{t}'] = derivative_values[t]
                    else:
                        row[f'timestep_{t}'] = ''  # Empty for timesteps beyond episode length
                
                writer.writerow(row)
        
        total_rows = len(self.directional_derivatives_data)
        print(f"Saved directional derivatives CSV: {csv_filename}")
        print(f"  Total rows: {total_rows}")
        print(f"  Max episode length: {max_episode_length}")
        print(f"  Timestep columns: timestep_0 to timestep_{max_episode_length - 1}")
    
    def save_taylor_deviations_csv(self):
        """
        Save all Taylor deviations data to a single CSV file.
        """
        if not hasattr(self, 'taylor_deviations_data') or not self.taylor_deviations_data:
            print("No Taylor deviations data to save.")
            return
        
        print("Saving Taylor deviations to single CSV file...")
        
        # Find the maximum episode length to determine number of timestep columns
        max_episode_length = max(data['episode_length'] for data in self.taylor_deviations_data)
        
        # Create timestep column names
        timestep_columns = [f'timestep_{t}' for t in range(max_episode_length)]
        
        # Create CSV filename
        csv_filename = 'taylor_deviations_all_seeds.csv'
        csv_filepath = os.path.join(self.logdir, csv_filename)
        
        # Write to CSV file
        with open(csv_filepath, 'w', newline='') as csvfile:
            fieldnames = ['seed', 'attack_type', 'influencer_agent_id', 'influenced_agent_id', 'observed_agent_id'] + timestep_columns
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for agent_data in self.taylor_deviations_data:
                row = {
                    'seed': agent_data['seed'],
                    'attack_type': agent_data['attack_type'],
                    'influencer_agent_id': agent_data['influencer_agent_id'],
                    'influenced_agent_id': agent_data['influenced_agent_id'],
                    'observed_agent_id': agent_data['observed_agent_id']
                }
                
                # Add Taylor deviation values for each timestep
                taylor_deviations = agent_data['taylor_deviations']
                for t in range(max_episode_length):
                    if t < len(taylor_deviations):
                        row[f'timestep_{t}'] = taylor_deviations[t]
                    else:
                        row[f'timestep_{t}'] = ''  # Empty for timesteps beyond episode length
                
                writer.writerow(row)
        
        total_rows = len(self.taylor_deviations_data)
        print(f"Saved Taylor deviations CSV: {csv_filename}")
        print(f"  Total rows: {total_rows}")
        print(f"  Max episode length: {max_episode_length}")
        print(f"  Timestep columns: timestep_0 to timestep_{max_episode_length - 1}")
        
    def save_cumulative_influences_csv(self):
        """
        Save all cumulative influence data to a single CSV file.
        """
        if not self.cumulative_influences_data:
            print("No cumulative influence data to save.")
            return
        
        print("Saving cumulative influences to single CSV file...")
        
        # Find the maximum episode length to determine number of timestep columns
        max_episode_length = max(data['episode_length'] for data in self.cumulative_influences_data)
        
        # Create timestep column names
        timestep_columns = [f'timestep_{t}' for t in range(max_episode_length)]
        
        # Create CSV filename
        csv_filename = 'cumulative_influences_all_seeds.csv'
        csv_filepath = os.path.join(self.logdir, csv_filename)
        
        # Write to CSV file
        with open(csv_filepath, 'w', newline='') as csvfile:
            fieldnames = ['seed', 'influencer_agent_id', 'influenced_agent_id'] + timestep_columns
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for pair_data in self.cumulative_influences_data:
                row = {
                    'seed': pair_data['seed'],
                    'influencer_agent_id': pair_data['influencer_agent_id'],
                    'influenced_agent_id': pair_data['influenced_agent_id']
                }
                
                # Add cumulative values for each timestep
                cumulative_values = pair_data['cumulative_values']
                for t in range(max_episode_length):
                    if t < len(cumulative_values):
                        row[f'timestep_{t}'] = cumulative_values[t]
                    else:
                        row[f'timestep_{t}'] = ''  # Empty for timesteps beyond episode length
                
                writer.writerow(row)
        
        total_rows = len(self.cumulative_influences_data)
        print(f"Saved cumulative influences CSV: {csv_filename}")
        print(f"  Total rows: {total_rows}")
        print(f"  Max episode length: {max_episode_length}")
        print(f"  Timestep columns: timestep_0 to timestep_{max_episode_length - 1}")
    def create_environment(self, seed):
        # First cleanup any existing environment
        # self.cleanup_environment(self.runner)
        
        # Small delay to ensure cleanup is complete
        
        # time.sleep(1)
        
        # Create new environment
        self.runner.eval_envs = make_eval_env(
            self.runner_args["env"],
            seed,
            self.runner_algo_args["eval"]["n_eval_rollout_threads"],
            self.runner_env_args,
        )
        
        
    def setup_experiment(self, initial_seed=0):
        def restore(runner,reward,filepath):
            """Restore model parameters."""
            for agent_id in range(runner.num_agents):
                policy_actor_state_dict = torch.load(
                    str(filepath)
                    + "/actor_agent"
                    + str(agent_id)
                    + "_" + str(reward)
                    + ".pt"
                )
                runner.actor[agent_id].actor.load_state_dict(policy_actor_state_dict)
            
                
                runner.central_q[agent_id].restore(filepath,agent_id,reward)
            
            
            if not runner.algo_args["render"]["use_render"]:
                policy_critic_state_dict = torch.load(
                    str(filepath)
                    + "/critic_agent"
                    + "_" + str(reward)
                    + ".pt"
                )
                runner.critic.critic.load_state_dict(policy_critic_state_dict)
                if runner.value_normalizer is not None:
                    value_normalizer_state_dict = torch.load(
                        str(filepath)
                        + "/value_normalizer"
                        + "_" + str(reward)
                        + ".pt"
                    )
                    runner.value_normalizer.load_state_dict(value_normalizer_state_dict)
                    
        parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        parser.add_argument(
            "--algo",
            type=str,
            default="hatrpo",
            choices=[
                "happo",
                "hatrpo",
                "haa2c",
                "haddpg",
                "hatd3",
                "hasac",
                "had3qn",
                "runner",
                "matd3",
                "mappo",
            ],
            help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, runner, matd3, mappo.",
        )
        parser.add_argument(
            "--env",
            type=str,
            default="smac",
            choices=[
                "smac",
                "mamujoco",
                "pettingzoo_mpe",
                "gym",
                "football",
                "dexhands",
                "smacv2",
                "lag",
            ],
            help="Environment name. Choose from: smac, mamujoco, pettingzoo_mpe, gym, football, dexhands, smacv2, lag.",
        )
        parser.add_argument(
            "--exp_name", type=str, default="installtest", help="Experiment name."
        )
        parser.add_argument(
            "--attack_id", type=int, default=0, help="Agent ID to attack."
        )
        parser.add_argument(
            "--load_config",
            type=str,
            default="",
            help="If set, load existing experiment config file instead of reading from yaml config file.",
        )
        parser.add_argument(
            "--reward",
            type=float,
            default=-79.879,
            help="Reward value to restore the model."
        )
        parser.add_argument(
            "--filepath",
            type=str,
            default="",
            help="Filepath to restore the model from."
        )
        # parser.add_argument(
        #     "--seed", type=int, default=376, help="Random seed."
        # )
        # parser.add_argument(
        #     "--min_window", type=int, default=8, help="Minimum window size."
        # )
        # parser.add_argument(
        #     "--max_window", type=int, default=10, help="Maximum window size."
        # )
        # parser.add_argument(
        #     "--taylor_csv_agent0", type=str, default="/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc_new/seed-376/hatrpo/0.1/2025-09-24-20-51-51/mappo_taylor_error_atk_free_agent_0.csv", help="Path to CSV file with pre-computed Taylor history for agent 0."
        # )
        # parser.add_argument(
        #     "--taylor_csv_agent1", type=str, default="/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc_new/seed-376/hatrpo/0.1/2025-09-24-20-51-51/mappo_taylor_error_atk_free_agent_1.csv", help="Path to CSV file with pre-computed Taylor history for agent 1."
        # )
        # parser.add_argument(
        #     "--taylor_csv_agent2", type=str, default="/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc_new/seed-376/hatrpo/0.1/2025-09-24-20-51-51/mappo_taylor_error_atk_free_agent_2.csv", help="Path to CSV file with pre-computed Taylor history for agent 2."
        # )
        # parser.add_argument(
        #     "--save_dir", type=str, default='/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/Part-2/test', help="Directory to save results."
        # )
        
        args, unparsed_args = parser.parse_known_args()

        def process(arg):
            try:
                return eval(arg)
            except:
                return arg

        keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
        values = [process(v) for v in unparsed_args[1::2]]
        unparsed_dict = {k: v for k, v in zip(keys, values)}
        args = vars(args)  # convert to dict
        if args["load_config"] != "":  # load config from existing config file
            with open(args["load_config"], encoding="utf-8") as file:
                all_config = json.load(file)
            args["algo"] = all_config["main_args"]["algo"]
            args["env"] = all_config["main_args"]["env"]
            algo_args = all_config["algo_args"]
            env_args = all_config["env_args"]
        else:  # load config from corresponding yaml file
            algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])
        #! Set initial seed for the runner setup
        algo_args["seed"]["seed"]=initial_seed
        update_args(unparsed_dict, algo_args, env_args)  # update args from command line

        if args["env"] == "dexhands":
            import isaacgym  # isaacgym has to be imported before PyTorch

        # note: isaac gym does not support multiple instances, thus cannot eval separately
        if args["env"] == "dexhands":
            algo_args["eval"]["use_eval"] = False
            algo_args["train"]["episode_length"] = env_args["hands_episode_length"]

        # start training
        from harl.runners import RUNNER_REGISTRY


        algo_args['eval']['n_eval_rollout_threads'] = 1
        algo_args['eval']['eval_episodes'] = 1
        self.runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
        self.runner_args = args
        self.runner_algo_args = algo_args
        self.runner_env_args = env_args
        
        print(f"args = {self.runner_args}")
        print(f"algo_args = {self.runner_algo_args}")
        print(f"env_args = {self.runner_env_args}")
        # print(f"Checking if centralize q is set : {algo_args['algo']['use_centralized_q']}")
        # restore(self.runner,args['reward'],args['filepath'])  # Restore the model with specific reward and episode
        restore(self.runner,self.config.reward,self.config.filepath)  
        print("Model restored successfully.")
        self.runner.prep_training()
        
        # Create log directory
        cwd = os.getcwd()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        env_type = 'discrete'
        self.logdir = os.path.join(cwd, self.folder_name, str(self.total_experiments), f"{args['env']}",
                                  f"{timestamp}_multi_seed_stats_sigma_{self.K_SIGMA}")
        os.makedirs(self.logdir, exist_ok=True)
        
        # Create environment
        # self.env = create_environment(self.config, self.runner)
        
        # Prepare runner for training mode
        device_str = 'gpu' if DEVICE == 'gpu' else 'cpu'
        # self.runner.prep_training(device=device_str)
        self.patient_zero_analyzer = PatientZeroAnalyzer(self.runner.num_agents)
        print(f"Multi-seed experiment setup complete. Log directory: {self.logdir}")
        print(f"Will run {self.total_experiments} experiments")
        # for env in self.runner.envs:
        #     env.close()
        # for env in self.runner.eval_envs.envs:
        #     env.close() 
        self.runner.eval_envs.close()
        self.runner.envs.close()

    def compute_taylor_policy(self,runner, eval_obs, eval_available_actions, eval_rnn_states):
        # states_tensor = torch.stack([torch.tensor(state_dict[k], dtype=torch.float32, requires_grad=True) for k in state_dict.keys()])
        eval_obs = torch.tensor(eval_obs, dtype=torch.float32, requires_grad=True)
        delta_errors = []
        eval_actions_collector = []
        eval_masks = np.ones(
            (self.runner.algo_args["eval"]["n_eval_rollout_threads"], self.runner.num_agents, 1),
            dtype=np.float32,
        )

        for agent_id in range(self.runner.num_agents):
            cur_obs = eval_obs[:, agent_id]
            eval_actions, eval_actions_log_prob, temp_rnn_state = self.runner.actor[agent_id].get_actions(
                cur_obs,
                eval_rnn_states[:, agent_id],
                eval_masks[:, agent_id],
                eval_available_actions[:, agent_id]
                if eval_available_actions[0] is not None
                else None,
                deterministic=True,
            )
            # eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
            eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
            grad_i = torch.autograd.grad(
                outputs=eval_actions_log_prob,
                inputs=cur_obs,
                create_graph=True,
                retain_graph=True,
            )[0]

            eta_i = 0.01 * grad_i.sign() / torch.max(grad_i.norm(p=2), torch.tensor(1e-6))

            
            j_tilde = eval_actions_log_prob + torch.dot(grad_i.flatten(), eta_i.flatten())  #+ 0.5 * torch.dot(eta_i.flatten(), hvp.flatten())

            p_obs = cur_obs + eta_i
            _, perturb_log_prob, _ = self.runner.actor[agent_id].get_actions(
                p_obs,
                eval_rnn_states[:, agent_id],
                eval_masks[:, agent_id],
                eval_available_actions[:, agent_id]
                if eval_available_actions[0] is not None
                else None,
                deterministic=True,
            )
            # _, _, p_log_prob = runner.agents.choose_actions_attack(p_state, i)
            # # Actual value of perturbed point
            j_perturbed = perturb_log_prob

            delta_error = abs(j_perturbed - j_tilde).item()
            delta_errors.append(delta_error)

        return delta_errors

    def compute_reference_taylor_error(self, runner, seed, total_episodes=1000, attack_status=False, attack_agent_id=0, randomness=1.00):

        results = [{} for _ in range(runner.num_agents)]
        
        for episode in tqdm(range(total_episodes), desc="Taylor Compute episodes", total=total_episodes):
            self.create_environment(seed)
            eval_obs, eval_share_obs, eval_available_actions = runner.eval_envs.reset() 
            # print(f"eval_obs:{eval_obs}")
            result_deque = [deque(maxlen=5) for _ in range(runner.num_agents)]
            timestamp = 0

            eval_rnn_states = np.zeros(
                (
                    runner.algo_args["eval"]["n_eval_rollout_threads"],
                    runner.num_agents,
                    runner.recurrent_n,
                    runner.rnn_hidden_size,
                ),
                dtype=np.float32,
            )
            eval_masks = np.ones(
                (runner.algo_args["eval"]["n_eval_rollout_threads"], runner.num_agents, 1),
                dtype=np.float32,
            )
            while True:
                if np.random.random() < randomness:  # 25% chance to add noise to observations
                    noise = np.random.normal(loc=0.0, scale=1e-4, size=eval_obs.shape)
                    # print(f"----> Noise: {noise}")
                    eval_obs = eval_obs + noise
                eval_actions_collector = []
                eval_rnn_states_backup = np.copy(eval_rnn_states)
                for agent_id in range(runner.num_agents):
                    eval_actions, temp_rnn_state = runner.actor[agent_id].act(
                        eval_obs[:, agent_id],
                        eval_rnn_states[:, agent_id],
                        eval_masks[:, agent_id],
                        eval_available_actions[:, agent_id]
                        if eval_available_actions[0] is not None
                        else None,
                        deterministic=True,
                    )
                    eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                    eval_actions_collector.append(_t2n(eval_actions))

                eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
                
                if attack_status:
                    n_actions = runner.eval_envs.action_space[attack_agent_id].n
                    avail_slice = slice_avail(eval_available_actions, attack_agent_id)

                    if avail_slice is not None and avail_slice[0] is not None:
                        available_actions = np.where(avail_slice[0] > 0.5)[0]
                    else:
                        available_actions = list(range(n_actions))
                    obs_tensor = torch.FloatTensor(eval_obs[:, attack_agent_id])
                    rnn_tensor = torch.FloatTensor(eval_rnn_states[:, attack_agent_id])
                    mask_tensor = torch.FloatTensor(eval_masks[:, attack_agent_id])
                    with torch.no_grad():
                        # action_logits = runner.actor[attack_agent_id].actor.act.get_logits(torch.tensor(eval_obs).to(runner.device))
                        action_log_probs, dist_entropy, action_distribution = runner.actor[attack_agent_id].evaluate_actions(
                            obs_tensor.to(runner.device),
                            rnn_tensor.to(runner.device),
                            available_actions,
                            mask_tensor.to(runner.device),
                            slice_avail(eval_available_actions, attack_agent_id),
                            None
                        )
                        # Extract action probabilities and take argmin
                        ### PERFORMING WORST-CASE ACTION SELECTION
                        q_values = action_log_probs.squeeze()
                        worst_action_index = torch.argmin(q_values).item()
                    eval_actions[0][attack_agent_id] = worst_action_index


                # calculating taylor policy
                delta_errors = self.compute_taylor_policy(runner, eval_obs, eval_available_actions, eval_rnn_states_backup)
                for j in range(runner.num_agents):
                    result_deque[j].append(delta_errors[j])
                    if timestamp not in results[j]:
                        results[j][timestamp] = []
                    results[j][timestamp].append(np.mean(list(result_deque[j])))
                timestamp += 1
                (
                    eval_obs,
                    eval_share_obs,
                    eval_rewards,
                    eval_dones,
                    eval_infos,
                    eval_available_actions,
                ) = runner.eval_envs.step(eval_actions)
                eval_data = (
                    eval_obs,
                    eval_share_obs,
                    eval_rewards,
                    eval_dones,
                    eval_infos,
                    eval_available_actions,
                )
                
                # Print eval_rewards and eval_obs
                # print(f"eval_rewards: {eval_rewards}")
                # print(f"eval_obs shape: {eval_obs.shape}, eval_obs: {eval_obs}")
                # runner.logger.eval_per_step(
                #     eval_data
                # )  # logger callback at each step of evaluation

                eval_dones_env = np.all(eval_dones, axis=1)

                eval_rnn_states[
                    eval_dones_env == True
                ] = np.zeros(  # if env is done, then reset rnn_state to all zero
                    (
                        (eval_dones_env == True).sum(),
                        runner.num_agents,
                        runner.recurrent_n,
                        runner.rnn_hidden_size,
                    ),
                    dtype=np.float32,
                )

                eval_masks = np.ones(
                    (runner.algo_args["eval"]["n_eval_rollout_threads"], runner.num_agents, 1),
                    dtype=np.float32,
                )
                eval_masks[eval_dones_env == True] = np.zeros(
                    ((eval_dones_env == True).sum(), runner.num_agents, 1), dtype=np.float32
                )
                done = False
                for eval_i in range(runner.algo_args["eval"]["n_eval_rollout_threads"]):
                    if eval_dones_env[eval_i]:
                        done = True
                #         # runner.logger.eval_thread_done(
                        #     eval_i
                        # )  # logger callback when an episode is done

                if done:
                    # runner.logger.eval_log(
                    #     eval_episode
                    # )  # logger callback at the end of evaluation
                    break
                
            # for env in runner.eval_envs.envs:
            #     env.close()
            runner.eval_envs.close()
            
        # Compute reference values and standard deviations
        ref_vals = [[] for _ in range(runner.num_agents)]
        ref_std_devs = [[] for _ in range(runner.num_agents)]
        
        for agent_id in range(runner.num_agents):
            sorted_timesteps = sorted(results[agent_id].keys())
            for timestep in sorted_timesteps:
                timestep_values = results[agent_id][timestep]
                mean_val = np.mean(timestep_values)
                std_val = np.std(timestep_values)
                ref_vals[agent_id].append(mean_val)
                ref_std_devs[agent_id].append(std_val)
        
        return ref_vals, ref_std_devs

    def collect_q_values(self,runner, eval_obs, eval_actions):
        """Collect Q-values for each agent given the current observations and actions."""
        eval_obs = torch.tensor(eval_obs, dtype=torch.float32)

        agent_obs_tensors = []
        n_agents = runner.num_agents
        for i in range(n_agents):
            agent_obs = eval_obs[0][i].clone().detach()
            agent_obs_tensor = torch.tensor(agent_obs, dtype=torch.float32, requires_grad=True)
            agent_obs_tensors.append(agent_obs_tensor)

        concatenated_obs = torch.cat(agent_obs_tensors, dim=0)
        share_obs = concatenated_obs.unsqueeze(0)
        concatenated_actions = torch.cat([torch.tensor(eval_actions[0][i], dtype=torch.float32, requires_grad=True) for i in range(n_agents)], dim=0).unsqueeze(0)

        q_values = []
        for i in range(n_agents):
            q_value = runner.central_q[i].get_q_values(
                share_obs.to(runner.device),
                concatenated_actions.to(runner.device),
                gradNeed=True
            ).squeeze()
            q_values.append(q_value.item())
        return q_values

    def compute_pairwise_frob_norms_from_attack_test(self,runner, eval_obs, eval_rnn_states_critic, eval_masks):
        """
        Compute Frobenius norms of cross-agent Hessian blocks for all (i, j).
        Returns an N x N matrix where entry [i][j] approximates || \partial^2 v_i / (\partial obs_i \partial obs_j) ||_F.
        This is the version from attack_test.py
        """
        eval_obs = torch.tensor(eval_obs, dtype=torch.float32)

        agent_obs_tensors = []
        n_agents = runner.num_agents
        # assume eval_obs shape (1, n_agents, obs_dim)
        for i in range(n_agents):
            agent_obs = eval_obs[0][i].clone().detach()
            agent_obs_tensor = torch.tensor(agent_obs, dtype=torch.float32, requires_grad=True)
            agent_obs_tensors.append(agent_obs_tensor)

        concatenated_obs = torch.cat(agent_obs_tensors, dim=0)
        share_obs = concatenated_obs.unsqueeze(0).unsqueeze(0)
        share_obs = share_obs.expand(1, n_agents, -1)
        # print(f"Shape of share_obs: {share_obs.shape}")

        # exit("Exiting after one episode for edge score calculation.")
        
        values, temp_rnn_state_critic = runner.critic.get_values(
            share_obs,
            eval_rnn_states_critic,
            eval_masks,
        )
        values = values.squeeze()

        N = n_agents
        results = [[0.0 for _ in range(N)] for _ in range(N)]


        for i in range(N):
            # gradient of v_i wrt agent i obs using the individual tensor
            """
            obs tensor:
                0 agent: [agent 0, agent 1, agent 2]
                1 agent: [agent 0, agent 1, agent 2]
                2 agent: [agent 0, agent 1, agent 2]
            0 agent : values[i][0*70:(0+1)*70]
            j agent : values[i][j*70:(j+1)*70]
            """
            
            grad_i = torch.autograd.grad(values[i], agent_obs_tensors[i], create_graph=True, retain_graph=True)[0]

            for j in range(N):
                # hessian_matrix = []
                hessian_matrix = torch.autograd.grad(
                    grad_i.squeeze(),
                    agent_obs_tensors[j],
                    grad_outputs=torch.eye(grad_i.shape[0]).to(grad_i.device),
                    retain_graph=True,
                    is_grads_batched=True,
                    allow_unused=True,
                )[0]
                results[i][j] = torch.norm(hessian_matrix, p='fro').item()
       
        return results
    def compute_pairwise_action_influence(self,runner, eval_obs, eval_actions, action_spaces): #Gij
        """Compute Frobenius norms of cross-agent Hessian blocks for all (i, j).
        Returns an N x N matrix where entry [i][j] approximates || \partial^2 v_i / (\partial obs_i \partial obs_j) ||_F.
        """
        
        one_hot_actions = []
        for i, action in enumerate(eval_actions[0]):
            one_hot = np.zeros(action_spaces[i].n)
            one_hot[action] = 1.0
            one_hot_actions.append(one_hot)
        actions = one_hot_actions
        
        
        eval_obs = torch.tensor(eval_obs, dtype=torch.float32)

        agent_obs_tensors = []
        n_agents = runner.num_agents
        # assume eval_obs shape (1, n_agents, obs_dim)
        for i in range(n_agents):
            agent_obs = eval_obs[0][i].clone().detach()
            agent_obs_tensor = torch.tensor(agent_obs, dtype=torch.float32, requires_grad=True)
            agent_obs_tensors.append(agent_obs_tensor)

        concatenated_obs = torch.cat(agent_obs_tensors, dim=0)
        torch_actions = [torch.tensor(actions[i], dtype=torch.float32,requires_grad=True) for i in range(n_agents)]
        share_obs = concatenated_obs.unsqueeze(0)
        concatenated_actions = torch.cat(torch_actions, dim=0).unsqueeze(0)
        # concatenated_actions = torch.stack([torch.tensor(eval_actions[0][i], dtype=torch.float32, requires_grad=True) for i in range(n_agents)], dim=0)
        # print(f"Concatenated actions shape: {concatenated_actions.shape}")
        # print(f"Share obs shape: {share_obs.shape}")

        N = n_agents
        results = [[0.0 for _ in range(N)] for _ in range(N)]

        for i in range(N):
            # gradient of v_i wrt agent i obs
            q_value = runner.central_q[i].get_q_values_onehot(
                share_obs,
                concatenated_actions
            ).squeeze()
            # print(f">>>> Is q_value requiring grad? {q_value.requires_grad}")
            # print(torch.autograd.grad(q_value,concatenated_actions,retain_graph=True,create_graph=True))
            # print(f"Q value for agent {i}: {q_value.item()}")
            for j in range(N):
                grad_i = torch.autograd.grad(q_value, torch_actions[j], create_graph=True, retain_graph=True)[0]
                results[i][j] = grad_i.norm(p=2).item()

        return results
    
    def compute_pairwise_action_directional_second_derivatives(self,runner, eval_obs, eval_actions, action_spaces): #Gij
        """Compute Frobenius norms of cross-agent Hessian blocks for all (i, j).
        Returns an N x N matrix where entry [i][j] approximates || \partial^2 v_i / (\partial obs_i \partial obs_j) ||_F.
        """
        
        one_hot_actions = []
        for i, action in enumerate(eval_actions[0]):
            one_hot = np.zeros(action_spaces[i].n)
            one_hot[action] = 1.0
            one_hot_actions.append(one_hot)
        actions = one_hot_actions
        
        
        eval_obs = torch.tensor(eval_obs, dtype=torch.float32)

        agent_obs_tensors = []
        n_agents = runner.num_agents
        # assume eval_obs shape (1, n_agents, obs_dim)
        for i in range(n_agents):
            agent_obs = eval_obs[0][i].clone().detach()
            agent_obs_tensor = torch.tensor(agent_obs, dtype=torch.float32, requires_grad=True)
            agent_obs_tensors.append(agent_obs_tensor)

        concatenated_obs = torch.cat(agent_obs_tensors, dim=0)
        torch_actions = [torch.tensor(actions[i], dtype=torch.float32,requires_grad=True) for i in range(n_agents)]
        share_obs = concatenated_obs.unsqueeze(0)
        concatenated_actions = torch.cat(torch_actions, dim=0).unsqueeze(0)
        # concatenated_actions = torch.stack([torch.tensor(eval_actions[0][i], dtype=torch.float32, requires_grad=True) for i in range(n_agents)], dim=0)
        # print(f"Concatenated actions shape: {concatenated_actions.shape}")
        # print(f"Share obs shape: {share_obs.shape}")

        N = n_agents
        results = [[0.0 for _ in range(N)] for _ in range(N)]

        for i in range(N):
            # gradient of v_i wrt agent i obs
            q_value = - runner.central_q[i].get_q_values_onehot(
                share_obs,
                concatenated_actions
            ).squeeze()
            # print(f">>>> Is q_value requiring grad? {q_value.requires_grad}")
            # print(torch.autograd.grad(q_value,concatenated_actions,retain_graph=True,create_graph=True))
            for j in range(N):
                grad_j = torch.autograd.grad(q_value, torch_actions[j], create_graph=True, retain_graph=True)[0]
                g = grad_j.flatten()
                # hvp = torch.autograd.grad(
                #     grad_j,
                #     torch_actions[j],
                #     grad_outputs=grad_j,
                #     retain_graph=True,
                #     allow_unused=True
                # )[0]
                hvp = torch.autograd.grad(
                    grad_j.sum(),
                    torch_actions[j],
                    retain_graph=True,
                    allow_unused=True
                )[0]
                hvp_flat = hvp.flatten()
                # results[i][j] = grad_i.norm(p=2).item()
                # print(f"Grad J norm for agents ({i},{j}): {g.norm(p=2).item()}")/
                # print(f"HVP: {hvp_flat}")
                # Compute g^T H g (directional second derivative)
                directional_second_derivative = torch.dot(g, hvp_flat).item()
                results[i][j] = directional_second_derivative

        return results
    
    
    
    

    def eval(self, runner, attack_status=False, attack_agent_id=0, seed=None, ref_vals=None, ref_std_devs=None, collect_q_flag=False,min_window=8,max_window=12,observe_agent=None):
        """Evaluate the model."""
        
        eval_episode = 0
        self.create_environment(seed)
        eval_obs, eval_share_obs, eval_available_actions = runner.eval_envs.reset()
        self.directional_derivatives_history = []
        eval_rnn_states = np.zeros(
            (
                runner.algo_args["eval"]["n_eval_rollout_threads"],
                runner.num_agents,
                runner.recurrent_n,
                runner.rnn_hidden_size,
            ),
            dtype=np.float32,
        )
        eval_rnn_states_critic = np.zeros(
            (
                runner.algo_args["eval"]["n_eval_rollout_threads"],
                runner.num_agents,
                runner.recurrent_n,
                runner.rnn_hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (runner.algo_args["eval"]["n_eval_rollout_threads"], runner.num_agents, 1),
            dtype=np.float32,
        )

        taylor_error_list = list()
        frob_norms_list = []
        result_deques = [deque(maxlen=5) for _ in range(runner.num_agents)]
        observe_agent_id = observe_agent if observe_agent is not None else attack_agent_id
        # Additional structures to mirror get_episode_data logic
        frob_norms_matrix_history = []  # list of N x N pairwise frob matrices per timestep
        fault_first_detected = {}  # agent_id -> first detected timestep
        fault_all_detected = {i: [] for i in range(runner.num_agents)}  # agent_id -> list of all fault timesteps
        fault_timeline = []
        attacked_steps = []
        pairwise_action_value_influence_list = []
        pairwise_action_value_influence_history = []
        taylor_history = [[] for _ in range(runner.num_agents)]
        cnt = 0
        q_values_list = [] if collect_q_flag else None
        total_rewards = {}
        reward_ep=0
        tt=0
        while True:
            
            eval_rnn_states_backup = np.copy(eval_rnn_states)
            eval_actions_collector = []
            for agent_id in range(runner.num_agents):
                eval_actions, temp_rnn_state = runner.actor[agent_id].act(
                    eval_obs[:, agent_id],
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    eval_available_actions[:, agent_id]
                    if eval_available_actions[0] is not None
                    else None,
                    deterministic=True,
                )
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
            
            if collect_q_flag:
                q_vals = self.collect_q_values(runner, eval_obs, eval_actions)
                q_values_list.append(q_vals)
            

            if attack_status and (cnt>=min_window and cnt<max_window):
                # mark attacked step
                # eval_actions[0][attack_agent_id] = runner.eval_envs.action_space[attack_agent_id].sample()  # Random action for attack agent
                # print(f">>>> ",eval_actions[0][attack_agent_id])
                n_actions = runner.eval_envs.action_space[attack_agent_id].n
                avail_slice = slice_avail(eval_available_actions, attack_agent_id)
                if avail_slice is not None and avail_slice[0] is not None:
                            available_actions = np.where(avail_slice[0] > 0.5)[0]
                else:
                    available_actions = list(range(n_actions))
                
                # exit("Exiting for debug")
                with torch.no_grad():
                    # print(f" [!!!] Attack launched on agent {attack_agent_id} at timestep: {cnt}")
                    obs_tensor = torch.FloatTensor(eval_obs[:, attack_agent_id])
                    rnn_tensor = torch.FloatTensor(eval_rnn_states[:, attack_agent_id])
                    mask_tensor = torch.FloatTensor(eval_masks[:, attack_agent_id])
                    
                    action_log_probs, dist_entropy, action_distribution = runner.actor[attack_agent_id].evaluate_actions(
                                obs_tensor.to(runner.device),
                                rnn_tensor.to(runner.device),
                                available_actions,
                                mask_tensor.to(runner.device),
                                slice_avail(eval_available_actions, attack_agent_id),
                                None
                            )
                    q_values = action_log_probs.squeeze()
                    if q_values.numel() == 1 or len(available_actions) == 1:
                        # print(f"Agent {agent_id} appears to be dead or has only one action. Using index 0.")
                        eval_actions[0][attack_agent_id] = 0
                    else:
                        worst_action_idx = torch.argmin(q_values).item()
                        # Map back to the actual action ID from available actions
                        worst_action = available_actions[worst_action_idx]
                        eval_actions[0][attack_agent_id] = worst_action
                        # print(f"Agent {attack_agent_id} worst action under current policy: {worst_action}")
                attacked_steps.append(cnt)


            # calculating taylor policy
            
            pairwise_action_value_influence = self.compute_pairwise_action_influence(runner, eval_obs, eval_actions, runner.eval_envs.action_space)
            # pairwise_action_value_influence = self.compute_pairwise_frob_norms_from_attack_test(runner, eval_obs, eval_rnn_states_backup,eval_masks)
            # exit("Exiting for debug")
            pairwise_action_value_influence_history.append(pairwise_action_value_influence)
        
            directional_second_derivatives = self.compute_pairwise_action_directional_second_derivatives(
                self.runner, eval_obs, eval_actions,runner.eval_envs.action_space )
            
            self.directional_derivatives_history.append(directional_second_derivatives)
            
            delta_errors = self.compute_taylor_policy(runner, eval_obs, eval_available_actions, eval_rnn_states_backup)
  
            
            for i in range(runner.num_agents):
                result_deques[i].append(delta_errors[i])
                taylor_approx_error = np.mean(result_deques[i])
                taylor_history[i].append(taylor_approx_error)

                # Detect anomalies based on Taylor approximation error using pre-computed history
                # Add boundary check to prevent IndexError
                if cnt < len(ref_vals[i]) and cnt < len(ref_std_devs[i]):
                    historical_mean = ref_vals[i][cnt]  
                    historical_std = ref_std_devs[i][cnt]
                    # Ensure minimum std deviation to avoid division by zero
                    if historical_std < 1e-6:
                        historical_std = 1e-6
                    
                    # print(f"Difference : {abs(taylor_approx_error - historical_mean)}")
                    if abs(taylor_approx_error - historical_mean) > self.K_SIGMA * historical_std:
                        print(" @ Agent:", i, " @ Timestep:", cnt)
                        # print(" >>>>>>> ", taylor_approx_error, historical_mean, historical_std)
                        print(f" [!!!] Anomaly detected for agent {i} at timestep: {cnt}. Taylor Appx. Error: {taylor_approx_error}")
                        # Record all fault timesteps for this agent
                        fault_all_detected[i].append(cnt)
                        fault_timeline.append({
                                'agent': i,
                                't': cnt,
                                'contribs': {}
                            })
                    # Record first fault detection if not already recorded
                    # if i not in fault_first_detected:
                    #     fault_first_detected[i] = cnt
                        
                    #     # Cascading Impact Analysis (only for first detection)
                    #     prev_faults = [(f, tf) for f, tf in fault_first_detected.items() if f != i and tf < cnt]
                    #     contribs = {}
                    #     if len(prev_faults) > 0:
                    #         for f, tf in prev_faults:
                    #             values_over_time = [frob_norms_matrix_history[tau][i][f] for tau in range(tf, cnt + 1) if tau < len(frob_norms_matrix_history)]
                    #             if len(values_over_time) > 0:
                    #                 contribs[f] = float(np.mean(values_over_time))
                    #         if len(contribs) > 0:
                    #             ranked = sorted(contribs.items(), key=lambda x: x[1], reverse=True)
                    #             print(f"     >> Potential contributors to fault in agent {i} (mean ||H_{{i,f}}||_F from t_f to {cnt}): {ranked}")
                        
                    #     # Add to fault_timeline only for first detection (t = first detection timestep)
                    #     fault_timeline.append({
                    #         'agent': i,
                    #         't': fault_first_detected[i],  # Always the first detection timestep
                    #         't_atk': fault_all_detected[i].copy(),  # Include all fault timesteps
                    #         'contribs': contribs
                    #     })
                    # else:
                    #     # For subsequent detections, update the existing entry in fault_timeline
                    #     # Find the entry for this agent and update t_atk
                    #     for entry in fault_timeline:
                    #         if entry['agent'] == i:
                    #             entry['t_atk'] = fault_all_detected[i].copy()
                    #             break

            taylor_error_list.append([np.mean(list(result_deques[j])) for j in range(runner.num_agents)])

            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = runner.eval_envs.step(eval_actions)
            total_rewards[tt] = np.array(eval_rewards).squeeze()
            tt=tt+1
            reward_ep+=np.sum(eval_rewards)
            eval_data = (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            )

            value, eval_rnn_states_critic = runner.critic.get_values(
                eval_share_obs,
                eval_rnn_states_critic,
                eval_masks,
            )

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[
                eval_dones_env == True
            ] = np.zeros(  # if env is done, then reset rnn_state to all zero
                (
                    (eval_dones_env == True).sum(),
                    runner.num_agents,
                    runner.recurrent_n,
                    runner.rnn_hidden_size,
                ),
                dtype=np.float32,
            )

            eval_masks = np.ones(
                (runner.algo_args["eval"]["n_eval_rollout_threads"], runner.num_agents, 1),
                dtype=np.float32,
            )
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), runner.num_agents, 1), dtype=np.float32
            )

            for eval_i in range(runner.algo_args["eval"]["n_eval_rollout_threads"]):
                if eval_dones_env[eval_i]:
                    eval_episode += 1

            if eval_episode >= runner.algo_args["eval"]["eval_episodes"]:
                break

            cnt += 1
        
        for env in runner.eval_envs.envs:
            env.close()
        

        if attack_status:
            return {
            'fault_timeline': fault_timeline,
            # 'fault_all_detected': fault_all_detected,
            'q_values_history': q_values_list,
            'taylor_errors_history': taylor_error_list,
            'episode_length': cnt,
            'episode_reward': reward_ep,
            'attack_timestep': min_window,
            'attack_timesteps':[min_window],
            'attacked_agent': attack_agent_id,
            'observed_agent': observe_agent_id,
            'stepwise_rewards': total_rewards,
        }
        else:
            return {
            'action_influences_history': pairwise_action_value_influence_history,
            'directional_derivatives_history': self.directional_derivatives_history,
            'q_values_history': q_values_list,
            'episode_length': cnt,
            'stepwise_rewards': total_rewards,
        }
    
    # def find_influence_timesteps(self, action_influences_history, agent_i, agent_j, first_quarter_steps):
    #     """
    #     Find max and min influence timesteps of agent i on agent j in first 25% of episode.
        
    #     Args:
    #         action_influences_history: List of action influence matrices
    #         agent_i: Index of influencing agent
    #         agent_j: Index of influenced agent (where action_influences_matrix[t][j][i] = influence of i on j)
    #         first_quarter_steps: Number of steps in first quarter
            
    #     Returns:
    #         Tuple of (max_influence_timestep, min_influence_timestep)
    #     """
    #     influences = []
    #     for t in range(min(first_quarter_steps, len(action_influences_history))):
    #         # Correct indexing: action_influences_matrix[t][j][i] = influence of i on j
    #         influence = abs(action_influences_history[t][agent_j][agent_i])
    #         influences.append((influence, t))
        
    #     # Sort by influence magnitude
    #     influences.sort(key=lambda x: x[0])
        
    #     min_influence_t = influences[0][1]  # Lowest influence
    #     max_influence_t = influences[-1][1]  # Highest influence
        
    #     return max_influence_t, min_influence_t
    def find_influence_timesteps(self, action_influences_history, directional_derivatives_history, agent_i, agent_j, atk_steps_limit, k_steps):
        """
        Find max and min influence timesteps of agent i on agent j in first 25% of episode.
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
            Tuple of (max_influence_timestep, min_influence_timestep)
        """
        positive_derivative_timesteps = []  # For high influence selection
        negative_derivative_timesteps = []  # For low influence selection
        # print(f"Directional Derivaties History : {directional_derivatives_history}")
        for t in range(min(atk_steps_limit, len(action_influences_history), len(directional_derivatives_history))):
            # Get action influence of agent_i on agent_j at timestep t
            influence = abs(action_influences_history[t][agent_j][agent_i])
            
            # Get directional second derivative of agent_i on agent_j at timestep t
            directional_derivative = directional_derivatives_history[t][agent_j][agent_i]
            
            if directional_derivative > 0:
                positive_derivative_timesteps.append((influence, t))
            elif directional_derivative < 0:
                negative_derivative_timesteps.append((influence, t))
        
            # print(" >>> ", directional_derivative)
        print(f"--- Positive derivative timesteps (influence, t): {positive_derivative_timesteps}", flush=True)
        print(f"--- Negative derivative timesteps (influence, t): {negative_derivative_timesteps}", flush=True)
        # For high influence: among positive derivative timesteps, choose maximum action influence
        max_influences_t = []
        if positive_derivative_timesteps:
            positive_derivative_timesteps.sort(key=lambda x: x[0], reverse=True)  # Sort by influence, descending
            max_influences_t = [t for _, t in positive_derivative_timesteps[:k_steps]]
        
        # For low influence: among negative derivative timesteps, choose minimum action influence
        min_influences_t = []
        if negative_derivative_timesteps:
            negative_derivative_timesteps.sort(key=lambda x: x[0])  # Sort by influence, ascending
            min_influences_t = [t for _, t in negative_derivative_timesteps[:k_steps]]
        
        max_influences_t.sort()
        min_influences_t.sort()

        return max_influences_t, min_influences_t



    def compute_attack_metrics(self, attack_results, normal_q_values, normal_rewards_history, ref_vals, ref_std_devs, observe_agent_j):
        """
        Compute Q-drop and Taylor deviation metrics for attacked episode.
        
        Args:
            attack_results: Results from attacked episode
            normal_q_values: Q values from normal episode
            ref_vals: Reference Taylor error values
            ref_std_devs: Reference standard deviations
            observe_agent_j: Index of agent to observe impact on (influenced agent)
            
        Returns:
            Dictionary containing computed metrics
        """
        attack_timestep = attack_results['attack_timestep']
        q_values_history = attack_results['q_values_history']
        taylor_errors_history = attack_results['taylor_errors_history']
        episode_length = attack_results['episode_length']
        attack_reward_history = attack_results['stepwise_rewards'] ## adding attacked reward history
        
        # Define watchable window (attack timestep to next 15 timesteps)
        window_start = attack_timestep
        window_end = min(attack_timestep + 15, episode_length - 1)
        
        metrics = {
            'max_q_drop': 0.0,
            'weighted_q_drop_sum': 0.0,
            'max_abs_taylor_deviation': 0.0,
            'weighted_taylor_deviation_sum': 0.0,
            'exceed_rate': 0.0,
            'window_length': window_end - window_start + 1,
            'max_reward_drop': 0.0, # adding max reward drop,
            'weighted_reward_drop_sum': 0.0 # adding weighted reward drop
        }
        
        if window_start >= len(q_values_history) or window_start >= len(taylor_errors_history):
            return metrics
        
        if window_start >= len(normal_q_values):
            return metrics
        
        # Compute metrics in watchable window for the observed agent
        exceed_count = 0
        window_steps = 0
        
        for t in range(window_start, window_end + 1):
            if t >= len(q_values_history) or t >= len(taylor_errors_history) or t >= len(normal_q_values):
                break
                
            window_steps += 1
            
            # Q-drop metrics for observed agent - difference between normal and attacked Q values
            normal_q = normal_q_values[t][observe_agent_j]
            normal_reward = normal_rewards_history[t][observe_agent_j] # adding normal reward
            attacked_q = q_values_history[t][observe_agent_j]
            attack_reward = attack_reward_history[t][observe_agent_j] #adding attacked reward
            q_drop = normal_q - attacked_q
            reward_drop = normal_reward - attack_reward # adding reward drop.  ### WARNING: THIS WILL NOT WORK FOR SMAC AS THEY CAN HAVE DIFFERENT TIMESTEPS
            metrics['max_q_drop'] = max(metrics['max_q_drop'], q_drop)
            metrics['max_reward_drop'] = max(metrics['max_reward_drop'], reward_drop) # adding max reward drop
            
            # Weighted Q-drop
            weight = self.gamma ** (t - attack_timestep)
            metrics['weighted_q_drop_sum'] += weight * q_drop
            metrics['weighted_reward_drop_sum'] += weight * reward_drop # adding weighted reward drop
            
            # Taylor deviation metrics for observed agent
            taylor_error = taylor_errors_history[t][observe_agent_j]
            ref_mean = ref_vals[observe_agent_j][t]
            ref_std = ref_std_devs[observe_agent_j][t]
            taylor_deviation = abs(taylor_error - ref_mean)
            
            metrics['max_abs_taylor_deviation'] = max(metrics['max_abs_taylor_deviation'], 
                                                    taylor_deviation)
            
            # Weighted Taylor deviation
            metrics['weighted_taylor_deviation_sum'] += weight * taylor_deviation
            
            # Check if exceeds threshold (mean ± K_SIGMA * std_dev)
            threshold = self.K_SIGMA * ref_std
            if taylor_deviation > threshold:
                exceed_count += 1
        
        # Compute exceed rate
        metrics['exceed_rate'] = exceed_count / window_steps
        
        return metrics

    def run_single_seed_experiment(self, runner,seed):
        """
        Run complete experiment for a single seed.
        The seed is already set by update_seed_for_experiment() before calling this method.
        
        Returns:
            Dictionary containing experiment results for all agent pairs
        """
        current_seed =seed #runner.algo_args["seed"]["seed"]
        print(f"\n{'='*50}")
        print(f"Running experiment for seed {current_seed}")
        print(f"{'='*50}")
        
        # Step 1: Compute reference Taylor error
        ref_vals, ref_std_devs = self.compute_reference_taylor_error(runner, seed=current_seed, total_episodes=self.total_episodes, attack_status=False, attack_agent_id=0, randomness=1.00)
        
        # print(f"### Ref vals :{ref_vals} and Ref std devs : {ref_std_devs}")
        # exit("Exiting after ref vals for debug")
        # Step 2: Run normal episode
        normal_episode = self.eval(runner=self.runner, attack_status=False, seed=current_seed, ref_vals=ref_vals, ref_std_devs=ref_std_devs, collect_q_flag=True, min_window=0, max_window=0, observe_agent=None)
        action_influences_history = normal_episode['action_influences_history']
        normal_q_values_history = normal_episode['q_values_history']
        directional_derivatives_history = normal_episode['directional_derivatives_history']
        episode_length = normal_episode['episode_length']
        normal_rewards_history = normal_episode['stepwise_rewards']
        self.log_cumulative_influences(action_influences_history, seed=current_seed, episode_length=episode_length)
        
        # Step 2.6: Log directional derivatives for each agent pair
        self.log_directional_derivatives(directional_derivatives_history, current_seed, episode_length)
        
        
        # Step 3: Analyze all possible ordered pairs (i, j) where i influences j
        all_pair_results = []
        first_quarter_steps = math.ceil(ATTACK_RATIO * episode_length)

        for agent_i in range(self.runner.num_agents):  # influencing agent
            for agent_j in range(self.runner.num_agents):  # influenced agent
                if agent_i == agent_j:
                    continue  # Skip self
                # if agent_i != 0 or agent_j != 1:
                #     continue  # TEMP: Only analyze pair (1, 2) for faster testing
                
                # print(f"\nAnalyzing pair: agent_{agent_i} influences agent_{agent_j}")
                
                # Step 4: Find max and min influence timesteps of agent_i on agent_j in first 25%
                # max_influence_t, min_influence_t = self.find_influence_timesteps(
                #     action_influences_history, agent_i, agent_j, first_quarter_steps
                # )
                max_influence_t, min_influence_t = self.find_influence_timesteps(
                    action_influences_history, directional_derivatives_history, agent_i, agent_j, first_quarter_steps, k_steps=1
                )
                
                # print(f"Max influence timestep: {max_influence_t}, Min influence timestep: {min_influence_t}")
                if not max_influence_t or not min_influence_t:
                    print(f"  >> Skipping pair (agent_{agent_i}, agent_{agent_j}) due to lack of valid influence timesteps")
                    continue
                
                # Step 5: Run attacked episodes - attack agent_i (influencer), observe impact on agent_j (influenced)
                high_influence_attack = self.eval(runner=self.runner, attack_status=True, attack_agent_id=agent_i, seed=current_seed, ref_vals=ref_vals, ref_std_devs=ref_std_devs, collect_q_flag=True, min_window=max_influence_t[0], max_window=max_influence_t[0]+5, observe_agent=agent_j)
                low_influence_attack = self.eval(runner=self.runner, attack_status=True, attack_agent_id=agent_i, seed=current_seed, ref_vals=ref_vals, ref_std_devs=ref_std_devs, collect_q_flag=True, min_window=min_influence_t[0], max_window=min_influence_t[0]+5, observe_agent=agent_j)
                # Get fault detection times for influencing and influenced agents
                high_influencer_fault_times = get_agent_fault_detection_times(high_influence_attack['fault_timeline'], agent_i)
                high_influenced_fault_times = get_agent_fault_detection_times(high_influence_attack['fault_timeline'], agent_j)
                low_influencer_fault_times = get_agent_fault_detection_times(low_influence_attack['fault_timeline'], agent_i)
                low_influenced_fault_times = get_agent_fault_detection_times(low_influence_attack['fault_timeline'], agent_j)
                
                # Determine patient zero for each attack
                # print(f"High influence attack fault timeline: {high_influence_attack['fault_timeline']}")
                # print(f"High influence taylor errors history: {high_influence_attack['taylor_errors_history']}")
                # exit("Exiting for debug")
                high_patient_zero, high_patient_time = get_patient_zero_detection(high_influence_attack['fault_timeline'])
                low_patient_zero, low_patient_time = get_patient_zero_detection(low_influence_attack['fault_timeline'])
                
                # Compute attack metrics
                high_metrics = self.compute_attack_metrics(high_influence_attack, normal_q_values_history, normal_rewards_history,ref_vals, ref_std_devs, agent_j)
                low_metrics = self.compute_attack_metrics(low_influence_attack, normal_q_values_history, normal_rewards_history, ref_vals, ref_std_devs, agent_j)
                self.log_taylor_deviations(high_influence_attack, low_influence_attack, ref_vals, ref_std_devs, current_seed, agent_i, agent_j)
                print(f"\n--- Patient Zero Analysis for pair agent_{agent_i} -> agent_{agent_j} ---")
                
                # Analyze high influence attack
                print(f"Analyzing HIGH influence attack:")
                high_pz_detection_analysis = self.patient_zero_analyzer.analyze_detection_accuracy(
                    high_influence_attack['fault_timeline'],
                    agent_i,  # The attacked agent is the influencer
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
                    agent_i,  # The attacked agent is the influencer
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
                    'max_influence_t': max_influence_t,
                    'min_influence_t': min_influence_t,
                    'high_patient_zero': high_patient_zero,
                    'high_patient_time': high_patient_time,
                    'low_patient_zero': low_patient_zero,
                    'low_patient_time': low_patient_time,
                    'high_influencer_fault_detection_times': high_influencer_fault_times,
                    'high_influenced_fault_detection_times': high_influenced_fault_times,
                    'low_influencer_fault_detection_times': low_influencer_fault_times,
                    'low_influenced_fault_detection_times': low_influenced_fault_times,
                    'high_metrics': high_metrics,
                    'low_metrics': low_metrics,
                    'high_patient_zero_analysis': high_pz_detection_analysis,
                    'low_patient_zero_analysis': low_pz_detection_analysis
                }
                
                all_pair_results.append(pair_result)
                
                # Handle multiple patient zeros for display
                high_pz_str = ', '.join(map(str, high_patient_zero)) if isinstance(high_patient_zero, list) else str(high_patient_zero)
                low_pz_str = ', '.join(map(str, low_patient_zero)) if isinstance(low_patient_zero, list) else str(low_patient_zero)
                
                print(f"High influence attack - Patient zero(s): {high_pz_str} at time {high_patient_time}")
                # print(f"  Influencer (agent_{agent_i}) fault detection times: {high_influencer_fault_times}")
                # print(f"  Influenced (agent_{agent_j}) fault detection times: {high_influenced_fault_times}")
                print(f"Low influence attack - Patient zero(s): {low_pz_str} at time {low_patient_time}")
                # print(f"  Influencer (agent_{agent_i}) fault detection times: {low_influencer_fault_times}")
                # print(f"  Influenced (agent_{agent_j}) fault detection times: {low_influenced_fault_times}")
        
        result = {
            'seed': current_seed,
            'episode_length': episode_length,
            'pair_results': all_pair_results,
            'total_pairs': len(all_pair_results)
        }
        
        print(f"\nCompleted analysis for {len(all_pair_results)} agent pairs")
        
        return result

    def update_seed_for_experiment(self, runner, seed):
        """
        Update seed for current experiment iteration without recreating environment.
        
        Args:
            runner: The runner instance
            seed: New seed value to set
        """
        # Update algo_args seed - এটাই main way
        runner.algo_args["seed"]["seed"] = seed
        
        # Update random seeds for reproducibility
        import random
        import numpy as np
        import torch
        
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # # SMAC environment এর জন্য seed update করার চেষ্টা করি
        # # eval_envs এর seed update করি
        # if hasattr(runner, 'eval_envs') and runner.eval_envs is not None:
        #     if hasattr(runner.eval_envs, 'seed'):
        #         print("Hits .seed &&&&&&")
        #         runner.eval_envs.seed(seed)
        #         # IMPORTANT: Reset environment after seeding
        #         runner.eval_envs.reset()
        #     # Vectorized environment এর ক্ষেত্রে
        #     elif hasattr(runner.eval_envs, 'envs'):
        #         for env in runner.eval_envs.envs:
        #             if hasattr(env, 'seed'):
        #                 print("Hits eval_envs.seed &&&&&&")
        #                 env.seed(seed)
        #             # SMAC StarCraft2Env এর ক্ষেত্রে _seed attribute আপডেট করি
        #             if hasattr(env, '_seed'):
        #                 env._seed = seed
        #         # Reset all environments after seeding
        #         runner.eval_envs.reset()
                        
        # # # Training envs এর seed update করি
        # # if hasattr(runner, 'envs') and runner.envs is not None:
        # #     if hasattr(runner.envs, 'seed'):
        # #         runner.envs.seed(seed)
        # #         # Reset training environment after seeding
        # #         runner.envs.reset()
        # #     elif hasattr(runner.envs, 'envs'):
        # #         for env in runner.envs.envs:
        # #             if hasattr(env, 'seed'):
        # #                 env.seed(seed)
        # #             if hasattr(env, '_seed'):
        # #                 env._seed = seed
        # #         # Reset all training environments after seeding
        # #         runner.envs.reset()
                            
        
        # print(f"Updated experiment seed to: {seed}")
    
    def run_all_experiments(self, runner):
        """Run experiments for all seeds."""


        total_pairs_per_seed = runner.num_agents * (runner.num_agents - 1)
        print(f"Starting multi-seed experiments with {self.total_experiments} seeds...")
        print(f"Each seed will analyze {total_pairs_per_seed} agent pairs")
        print(f"Total pairs to analyze: {self.total_experiments * total_pairs_per_seed}")
        
        for seed in tqdm(range(self.total_experiments), desc="Running experiments"):
            # if seed != 1:
                # continue
            # প্রতিটি experiment এর জন্য seed আপডেট করি
            self.update_seed_for_experiment(runner, seed)
            result = self.run_single_seed_experiment(runner,seed)
            self.experiment_results.append(result)
        
        total_successful_pairs = sum(result['total_pairs'] for result in self.experiment_results)
        print(f"\nCompleted {len(self.experiment_results)} successful experiments out of {self.total_experiments}")
        print(f"Total successful pairs analyzed: {total_successful_pairs}")
        print(f"Failed experiments: {len(self.failed_seeds)}")
    
    def compute_accuracies(self):
        """Compute accuracies and analyze results."""
        if not self.experiment_results:
            print("No successful experiments to analyze!")
            return
        
        print("\n" + "="*50)
        print("COMPUTING ACCURACIES")
        print("="*50)
        
        # Patient zero detection accuracy
        correct_patient_zero = 0
        total_with_detection = 0
        
        # Expectation accuracy metrics - separate accuracies for each metric
        q_drop_max_expectation_correct = 0
        reward_drop_max_expectation_correct = 0
        q_drop_weighted_expectation_correct = 0
        reward_drop_weighted_expectation_correct = 0
        taylor_max_expectation_correct = 0
        taylor_weighted_expectation_correct = 0
        exceed_rate_expectation_correct = 0
        high_correct_patient_zero = 0
        low_correct_patient_zero = 0
        high_total_with_detection = 0
        low_total_with_detection = 0
        # Detailed metrics
        high_metrics_list = []
        low_metrics_list = []
        
        failed_expectations = []
        total_pairs = 0
        
        # Process all pairs from all seeds
        for result in self.experiment_results:
            seed = result['seed']
            pair_results = result['pair_results']
            
            for pair_result in pair_results:
                total_pairs += 1
                
                # Patient zero analysis - we expect agent_i (attacked agent) to be detected as patient zero
                high_patient_zero = pair_result['high_patient_zero']
                low_patient_zero = pair_result['low_patient_zero']
                attacked_agent = pair_result['agent_i']  # The agent we attacked (influencer)
                
                if high_patient_zero is not None:
                    total_with_detection += 1
                    high_total_with_detection += 1
                    if attacked_agent in high_patient_zero:
                        correct_patient_zero += 1
                        high_correct_patient_zero += 1
                
                if low_patient_zero is not None:
                    total_with_detection += 1
                    low_total_with_detection += 1   
                    if attacked_agent in low_patient_zero:
                        correct_patient_zero += 1
                        low_correct_patient_zero += 1

                # Expectation analysis
                high_metrics = pair_result['high_metrics']
                low_metrics = pair_result['low_metrics']
                
                high_metrics_list.append(high_metrics)
                low_metrics_list.append(low_metrics)
                
                # Check individual metric expectations (high influence should have higher impact)
                q_drop_max_better = high_metrics['max_q_drop'] >= low_metrics['max_q_drop']
                reward_drop_max_better = high_metrics['max_reward_drop'] >= low_metrics['max_reward_drop'] # adding reward drop comparison
                q_drop_weighted_better = high_metrics['weighted_q_drop_sum'] >= low_metrics['weighted_q_drop_sum']
                reward_drop_weighted_better = high_metrics['weighted_reward_drop_sum'] >= low_metrics['weighted_reward_drop_sum'] # adding reward drop comparison
                taylor_max_better = high_metrics['max_abs_taylor_deviation'] >= low_metrics['max_abs_taylor_deviation']
                taylor_weighted_better = high_metrics['weighted_taylor_deviation_sum'] >= low_metrics['weighted_taylor_deviation_sum']
                exceed_rate_better = high_metrics['exceed_rate'] >= low_metrics['exceed_rate']

                if q_drop_max_better:
                    q_drop_max_expectation_correct += 1
                
                if reward_drop_max_better:
                    reward_drop_max_expectation_correct += 1
                if q_drop_weighted_better:
                    q_drop_weighted_expectation_correct += 1

                if reward_drop_weighted_better:
                    reward_drop_weighted_expectation_correct += 1

                if taylor_max_better:
                    taylor_max_expectation_correct += 1
                
                if taylor_weighted_better:
                    taylor_weighted_expectation_correct += 1
                
                if exceed_rate_better:
                    exceed_rate_expectation_correct += 1
                
                # Log failed expectations
                if not (q_drop_max_better or q_drop_weighted_better or taylor_max_better or taylor_weighted_better or exceed_rate_better):
                    failed_expectations.append({
                        'seed': seed,
                        'agent_i_influencer_attacked': pair_result['agent_i'],
                        'agent_j_influenced_observed': pair_result['agent_j'],
                        'q_drop_max_failed': not q_drop_max_better,
                        'q_drop_weighted_failed': not q_drop_weighted_better,
                        'reward_drop_max_failed': not reward_drop_max_better,
                        'reward_drop_weighted_failed': not reward_drop_weighted_better,
                        'taylor_max_failed': not taylor_max_better,
                        'taylor_weighted_failed': not taylor_weighted_better,
                        'exceed_rate_failed': not exceed_rate_better,
                        'high_q_drop_max': high_metrics['max_q_drop'],
                        'low_q_drop_max': low_metrics['max_q_drop'],
                        'high_q_drop_weighted': high_metrics['weighted_q_drop_sum'],
                        'low_q_drop_weighted': low_metrics['weighted_q_drop_sum'],
                        'high_reward_drop_max': high_metrics['max_reward_drop'],
                        'low_reward_drop_max': low_metrics['max_reward_drop'],
                        'high_reward_drop_weighted': high_metrics['weighted_reward_drop_sum'],
                        'low_reward_drop_weighted': low_metrics['weighted_reward_drop_sum'],
                        'high_taylor_max': high_metrics['max_abs_taylor_deviation'],
                        'low_taylor_max': low_metrics['max_abs_taylor_deviation'],
                        'high_taylor_weighted': high_metrics['weighted_taylor_deviation_sum'],
                        'low_taylor_weighted': low_metrics['weighted_taylor_deviation_sum'],
                        'high_exceed_rate': high_metrics['exceed_rate'],
                        'low_exceed_rate': low_metrics['exceed_rate']
                    })
        
        total_experiments = len(self.experiment_results)
        if total_pairs == 0:
            print("No agent pairs were successfully analyzed across all experiments.")
            # return
        # Compute accuracies
        patient_zero_accuracy = (correct_patient_zero / total_with_detection) if total_with_detection > 0 else 0
        high_patient_zero_accuracy = (high_correct_patient_zero / high_total_with_detection) if high_total_with_detection > 0 else 0
        low_patient_zero_accuracy = (low_correct_patient_zero / low_total_with_detection) if low_total_with_detection > 0 else 0
        q_drop_max_accuracy = q_drop_max_expectation_correct / total_pairs if total_pairs > 0 else 0
        q_drop_weighted_accuracy = q_drop_weighted_expectation_correct / total_pairs if total_pairs > 0 else 0
        reward_drop_max_accuracy = reward_drop_max_expectation_correct / total_pairs if total_pairs > 0 else 0
        reward_drop_weighted_accuracy = reward_drop_weighted_expectation_correct / total_pairs if total_pairs > 0 else 0
        taylor_max_accuracy = taylor_max_expectation_correct / total_pairs  if total_pairs > 0 else 0
        taylor_weighted_accuracy = taylor_weighted_expectation_correct / total_pairs if total_pairs > 0 else 0
        exceed_rate_accuracy = exceed_rate_expectation_correct / total_pairs if total_pairs > 0 else 0
        
        # Aggregate metrics
        avg_high_q_drop_max = np.mean([m['max_q_drop'] for m in high_metrics_list])
        avg_low_q_drop_max = np.mean([m['max_q_drop'] for m in low_metrics_list])
        avg_high_q_drop_weighted = np.mean([m['weighted_q_drop_sum'] for m in high_metrics_list])
        avg_low_q_drop_weighted = np.mean([m['weighted_q_drop_sum'] for m in low_metrics_list])
        avg_high_reward_drop_max = np.mean([m['max_reward_drop'] for m in high_metrics_list])
        avg_low_reward_drop_max = np.mean([m['max_reward_drop'] for m in low_metrics_list])
        avg_high_reward_drop_weighted = np.mean([m['weighted_reward_drop_sum'] for m in high_metrics_list])
        avg_low_reward_drop_weighted = np.mean([m['weighted_reward_drop_sum'] for m in low_metrics_list])
        avg_high_taylor_max = np.mean([m['max_abs_taylor_deviation'] for m in high_metrics_list])
        avg_low_taylor_max = np.mean([m['max_abs_taylor_deviation'] for m in low_metrics_list])
        avg_high_taylor_weighted = np.mean([m['weighted_taylor_deviation_sum'] for m in high_metrics_list])
        avg_low_taylor_weighted = np.mean([m['weighted_taylor_deviation_sum'] for m in low_metrics_list])
        avg_high_exceed_rate = np.mean([m['exceed_rate'] for m in high_metrics_list])
        avg_low_exceed_rate = np.mean([m['exceed_rate'] for m in low_metrics_list])
        
        accuracy_results = {
            'total_experiments': total_experiments,
            'total_pairs': total_pairs,
            'total_with_detection': total_with_detection,
            'correct_patient_zero': correct_patient_zero,
            'patient_zero_accuracy': patient_zero_accuracy,
            'high_patient_zero_accuracy': high_patient_zero_accuracy,
            'low_patient_zero_accuracy': low_patient_zero_accuracy,
            'q_drop_max_expectation_correct': q_drop_max_expectation_correct,
            'q_drop_max_accuracy': q_drop_max_accuracy,
            'q_drop_weighted_expectation_correct': q_drop_weighted_expectation_correct,
            'q_drop_weighted_accuracy': q_drop_weighted_accuracy,
            'reward_drop_max_expectation_correct': reward_drop_max_expectation_correct,
            'reward_drop_max_accuracy': reward_drop_max_accuracy,
            'reward_drop_weighted_expectation_correct': reward_drop_weighted_expectation_correct,
            'reward_drop_weighted_accuracy': reward_drop_weighted_accuracy,
            'taylor_max_expectation_correct': taylor_max_expectation_correct,
            'taylor_max_accuracy': taylor_max_accuracy,
            'taylor_weighted_expectation_correct': taylor_weighted_expectation_correct,
            'taylor_weighted_accuracy': taylor_weighted_accuracy,
            'exceed_rate_expectation_correct': exceed_rate_expectation_correct,
            'exceed_rate_accuracy': exceed_rate_accuracy,
            'avg_high_q_drop_max': avg_high_q_drop_max,
            'avg_low_q_drop_max': avg_low_q_drop_max,
            'avg_high_q_drop_weighted': avg_high_q_drop_weighted,
            'avg_low_q_drop_weighted': avg_low_q_drop_weighted,
            'avg_high_reward_drop_max': avg_high_reward_drop_max,
            'avg_low_reward_drop_max': avg_low_reward_drop_max,
            'avg_high_reward_drop_weighted': avg_high_reward_drop_weighted,
            'avg_low_reward_drop_weighted': avg_low_reward_drop_weighted,
            'avg_high_taylor_max': avg_high_taylor_max,
            'avg_low_taylor_max': avg_low_taylor_max,
            'avg_high_taylor_weighted': avg_high_taylor_weighted,
            'avg_low_taylor_weighted': avg_low_taylor_weighted,
            'avg_high_exceed_rate': avg_high_exceed_rate,
            'avg_low_exceed_rate': avg_low_exceed_rate,
            'failed_expectations_count': len(failed_expectations)
        }
        
        print(f"Total Experiments: {total_experiments}, Total Agent Pairs: {total_pairs}")
        print(f"Patient Zero Detection Accuracy: {patient_zero_accuracy:.3f} ({correct_patient_zero}/{total_with_detection})")
        print(f"High Influence: Patient Zero Accuracy: {high_patient_zero_accuracy:.3f} ({high_correct_patient_zero}/{high_total_with_detection})")
        print(f"Low Influence: Patient Zero Accuracy: {low_patient_zero_accuracy:.3f} ({low_correct_patient_zero}/{low_total_with_detection})")
        print(f"Q-Drop Max Expectation Accuracy: {q_drop_max_accuracy:.3f} ({q_drop_max_expectation_correct}/{total_pairs})")
        print(f"Q-Drop Weighted Expectation Accuracy: {q_drop_weighted_accuracy:.3f} ({q_drop_weighted_expectation_correct}/{total_pairs})")
        print(f"Reward-Drop Max Expectation Accuracy: {reward_drop_max_accuracy:.3f} ({reward_drop_max_expectation_correct}/{total_pairs})")
        print(f"Reward-Drop Weighted Expectation Accuracy: {reward_drop_weighted_accuracy:.3f} ({reward_drop_weighted_expectation_correct}/{total_pairs})")
        print(f"Taylor Max Expectation Accuracy: {taylor_max_accuracy:.3f} ({taylor_max_expectation_correct}/{total_pairs})")
        print(f"Taylor Weighted Expectation Accuracy: {taylor_weighted_accuracy:.3f} ({taylor_weighted_expectation_correct}/{total_pairs})")
        print(f"Exceed Rate Expectation Accuracy: {exceed_rate_accuracy:.3f} ({exceed_rate_expectation_correct}/{total_pairs})")
        print(f"Average High Influence Q-Drop Max: {avg_high_q_drop_max:.6f}")
        print(f"Average Low Influence Q-Drop Max: {avg_low_q_drop_max:.6f}")
        print(f"Average High Influence Reward-Drop Max: {avg_high_reward_drop_max:.6f}")
        print(f"Average Low Influence Reward-Drop Max: {avg_low_reward_drop_max:.6f}")
        print(f"Average High Influence Taylor Max: {avg_high_taylor_max:.6f}")
        print(f"Average Low Influence Taylor Max: {avg_low_taylor_max:.6f}")
        print(f"Average High Influence Exceed Rate: {avg_high_exceed_rate:.6f}")
        print(f"Average Low Influence Exceed Rate: {avg_low_exceed_rate:.6f}")
        print(f"Failed Expectations: {len(failed_expectations)}")
        
        return accuracy_results, failed_expectations
    
    def compute_pair_specific_accuracies(self):
        """Compute accuracies and analyze results for each agent pair separately."""
        if not self.experiment_results:
            print("No successful experiments to analyze!")
            return {}
        
        print("\n" + "="*50)
        print("COMPUTING PAIR-SPECIFIC ACCURACIES")
        print("="*50)
        
        # Initialize pair-specific results storage
        pair_specific_results = {}
        
        # Get all unique pairs
        unique_pairs = set()
        for result in self.experiment_results:
            for pair_result in result['pair_results']:
                pair_key = (pair_result['agent_i'], pair_result['agent_j'])
                unique_pairs.add(pair_key)
        
        print(f"Found {len(unique_pairs)} unique agent pairs")
        
        # Process each unique pair
        for agent_i, agent_j in sorted(unique_pairs):
            pair_key = f"agent_{agent_i}_to_agent_{agent_j}"
            print(f"\nAnalyzing pair: {pair_key}")
            
            # Collect all results for this specific pair
            pair_data = []
            for result in self.experiment_results:
                for pair_result in result['pair_results']:
                    if pair_result['agent_i'] == agent_i and pair_result['agent_j'] == agent_j:
                        pair_data.append({
                            'seed': result['seed'],
                            'episode_length': result['episode_length'],
                            **pair_result
                        })
            
            if not pair_data:
                continue
            
            # Initialize counters for this pair
            correct_patient_zero = 0
            total_with_detection = 0
            high_correct_patient_zero = 0
            high_total_with_detection = 0
            low_correct_patient_zero = 0
            low_total_with_detection = 0
            
            # Expectation accuracy metrics for this pair
            q_drop_max_expectation_correct = 0
            q_drop_weighted_expectation_correct = 0
            reward_drop_max_expectation_correct = 0
            reward_drop_weighted_expectation_correct = 0
            taylor_max_expectation_correct = 0
            taylor_weighted_expectation_correct = 0
            exceed_rate_expectation_correct = 0
            
            # Detailed metrics for this pair
            high_metrics_list = []
            low_metrics_list = []
            failed_expectations = []
            
            # Process each experiment for this pair
            for data in pair_data:
                # Patient zero analysis
                high_patient_zero = data['high_patient_zero']
                low_patient_zero = data['low_patient_zero']
                attacked_agent = data['agent_i']  # The agent we attacked (influencer)
                
                if high_patient_zero is not None:
                    total_with_detection += 1
                    high_total_with_detection += 1
                    if attacked_agent in high_patient_zero:
                        correct_patient_zero += 1
                        high_correct_patient_zero += 1
                
                if low_patient_zero is not None:
                    total_with_detection += 1
                    low_total_with_detection += 1
                    if attacked_agent in low_patient_zero:
                        correct_patient_zero += 1
                        low_correct_patient_zero += 1
                
                # Expectation analysis
                high_metrics = data['high_metrics']
                low_metrics = data['low_metrics']
                
                high_metrics_list.append(high_metrics)
                low_metrics_list.append(low_metrics)
                
                # Check individual metric expectations
                q_drop_max_better = high_metrics['max_q_drop'] >= low_metrics['max_q_drop']
                q_drop_weighted_better = high_metrics['weighted_q_drop_sum'] >= low_metrics['weighted_q_drop_sum']
                reward_drop_max_better = high_metrics['max_reward_drop'] >= low_metrics['max_reward_drop']
                reward_drop_weighted_better = high_metrics['weighted_reward_drop_sum'] >= low_metrics['weighted_reward_drop_sum']
                taylor_max_better = high_metrics['max_abs_taylor_deviation'] >= low_metrics['max_abs_taylor_deviation']
                taylor_weighted_better = high_metrics['weighted_taylor_deviation_sum'] >= low_metrics['weighted_taylor_deviation_sum']
                exceed_rate_better = high_metrics['exceed_rate'] >= low_metrics['exceed_rate']

                if q_drop_max_better:
                    q_drop_max_expectation_correct += 1
                if q_drop_weighted_better:
                    q_drop_weighted_expectation_correct += 1
                if reward_drop_max_better:
                    reward_drop_max_expectation_correct += 1
                if reward_drop_weighted_better:
                    reward_drop_weighted_expectation_correct += 1
                if taylor_max_better:
                    taylor_max_expectation_correct += 1
                if taylor_weighted_better:
                    taylor_weighted_expectation_correct += 1
                if exceed_rate_better:
                    exceed_rate_expectation_correct += 1
                
                # Log failed expectations for this pair
                if not (q_drop_max_better or q_drop_weighted_better or
                        reward_drop_max_better or reward_drop_weighted_better or
                        taylor_max_better or taylor_weighted_better or
                        exceed_rate_better):
                    failed_expectations.append({
                        'seed': data['seed'],
                        'agent_i_influencer_attacked': data['agent_i'],
                        'agent_j_influenced_observed': data['agent_j'],
                        'q_drop_max_failed': not q_drop_max_better,
                        'q_drop_weighted_failed': not q_drop_weighted_better,
                        'reward_drop_max_failed': not reward_drop_max_better,
                        'reward_drop_weighted_failed': not reward_drop_weighted_better,
                        'taylor_max_failed': not taylor_max_better,
                        'taylor_weighted_failed': not taylor_weighted_better,
                        'exceed_rate_failed': not exceed_rate_better,
                        'high_q_drop_max': high_metrics['max_q_drop'],
                        'low_q_drop_max': low_metrics['max_q_drop'],
                        'high_q_drop_weighted': high_metrics['weighted_q_drop_sum'],
                        'low_q_drop_weighted': low_metrics['weighted_q_drop_sum'],
                        'high_reward_drop_max': high_metrics['max_reward_drop'],
                        'low_reward_drop_max': low_metrics['max_reward_drop'],
                        'high_reward_drop_weighted': high_metrics['weighted_reward_drop_sum'],
                        'low_reward_drop_weighted': low_metrics['weighted_reward_drop_sum'],
                        'high_taylor_max': high_metrics['max_abs_taylor_deviation'],
                        'low_taylor_max': low_metrics['max_abs_taylor_deviation'],
                        'high_taylor_weighted': high_metrics['weighted_taylor_deviation_sum'],
                        'low_taylor_weighted': low_metrics['weighted_taylor_deviation_sum'],
                        'high_exceed_rate': high_metrics['exceed_rate'],
                        'low_exceed_rate': low_metrics['exceed_rate']
                    })
            
            total_experiments_for_pair = len(pair_data)
            
            # Compute accuracies for this pair
            patient_zero_accuracy = correct_patient_zero / total_with_detection if total_with_detection > 0 else 0
            high_patient_zero_accuracy = high_correct_patient_zero / high_total_with_detection if high_total_with_detection > 0 else 0
            low_patient_zero_accuracy = low_correct_patient_zero / low_total_with_detection if low_total_with_detection > 0 else 0
            q_drop_max_accuracy = q_drop_max_expectation_correct / total_experiments_for_pair
            q_drop_weighted_accuracy = q_drop_weighted_expectation_correct / total_experiments_for_pair
            reward_drop_max_accuracy = reward_drop_max_expectation_correct / total_experiments_for_pair
            reward_drop_weighted_accuracy = reward_drop_weighted_expectation_correct / total_experiments_for_pair
            taylor_max_accuracy = taylor_max_expectation_correct / total_experiments_for_pair
            taylor_weighted_accuracy = taylor_weighted_expectation_correct / total_experiments_for_pair
            exceed_rate_accuracy = exceed_rate_expectation_correct / total_experiments_for_pair
            
            # Aggregate metrics for this pair
            avg_high_q_drop_max = np.mean([m['max_q_drop'] for m in high_metrics_list])
            avg_low_q_drop_max = np.mean([m['max_q_drop'] for m in low_metrics_list])
            avg_high_q_drop_weighted = np.mean([m['weighted_q_drop_sum'] for m in high_metrics_list])
            avg_low_q_drop_weighted = np.mean([m['weighted_q_drop_sum'] for m in low_metrics_list])
            avg_high_reward_drop_max = np.mean([m['max_reward_drop'] for m in high_metrics_list])
            avg_low_reward_drop_max = np.mean([m['max_reward_drop'] for m in low_metrics_list])
            avg_high_reward_drop_weighted = np.mean([m['weighted_reward_drop_sum'] for m in high_metrics_list])
            avg_low_reward_drop_weighted = np.mean([m['weighted_reward_drop_sum'] for m in low_metrics_list])
            avg_high_taylor_max = np.mean([m['max_abs_taylor_deviation'] for m in high_metrics_list])
            avg_low_taylor_max = np.mean([m['max_abs_taylor_deviation'] for m in low_metrics_list])
            avg_high_taylor_weighted = np.mean([m['weighted_taylor_deviation_sum'] for m in high_metrics_list])
            avg_low_taylor_weighted = np.mean([m['weighted_taylor_deviation_sum'] for m in low_metrics_list])
            avg_high_exceed_rate = np.mean([m['exceed_rate'] for m in high_metrics_list])
            avg_low_exceed_rate = np.mean([m['exceed_rate'] for m in low_metrics_list])
            
            # Calculate average delta metrics (high - low)
            avg_delta_max_q_drop = np.mean([h['max_q_drop'] - l['max_q_drop'] for h, l in zip(high_metrics_list, low_metrics_list)])
            avg_delta_weighted_q_drop_sum = np.mean([h['weighted_q_drop_sum'] - l['weighted_q_drop_sum'] for h, l in zip(high_metrics_list, low_metrics_list)])
            avg_delta_max_reward_drop = np.mean([h['max_reward_drop'] - l['max_reward_drop'] for h, l in zip(high_metrics_list, low_metrics_list)])
            avg_delta_weighted_reward_drop_sum = np.mean([h['weighted_reward_drop_sum'] - l['weighted_reward_drop_sum'] for h, l in zip(high_metrics_list, low_metrics_list)])
            avg_delta_max_abs_taylor_deviation = np.mean([h['max_abs_taylor_deviation'] - l['max_abs_taylor_deviation'] for h, l in zip(high_metrics_list, low_metrics_list)])
            avg_delta_weighted_taylor_deviation_sum = np.mean([h['weighted_taylor_deviation_sum'] - l['weighted_taylor_deviation_sum'] for h, l in zip(high_metrics_list, low_metrics_list)])
            avg_delta_exceed_rate = np.mean([h['exceed_rate'] - l['exceed_rate'] for h, l in zip(high_metrics_list, low_metrics_list)])
            
            # Store results for this pair
            pair_specific_results[pair_key] = {
                'agent_i': agent_i,
                'agent_j': agent_j,
                'total_experiments': total_experiments_for_pair,
                'total_with_detection': total_with_detection,
                'correct_patient_zero': correct_patient_zero,
                'patient_zero_accuracy': patient_zero_accuracy,
                'high_patient_zero_accuracy': high_patient_zero_accuracy,
                'low_patient_zero_accuracy': low_patient_zero_accuracy,
                'q_drop_max_expectation_correct': q_drop_max_expectation_correct,
                'q_drop_max_accuracy': q_drop_max_accuracy,
                'q_drop_weighted_expectation_correct': q_drop_weighted_expectation_correct,
                'q_drop_weighted_accuracy': q_drop_weighted_accuracy,
                'reward_drop_max_expectation_correct': reward_drop_max_expectation_correct,
                'reward_drop_max_accuracy': reward_drop_max_accuracy,
                'reward_drop_weighted_expectation_correct': reward_drop_weighted_expectation_correct,
                'reward_drop_weighted_accuracy': reward_drop_weighted_accuracy,
                'taylor_max_expectation_correct': taylor_max_expectation_correct,
                'taylor_max_accuracy': taylor_max_accuracy,
                'taylor_weighted_expectation_correct': taylor_weighted_expectation_correct,
                'taylor_weighted_accuracy': taylor_weighted_accuracy,
                'exceed_rate_expectation_correct': exceed_rate_expectation_correct,
                'exceed_rate_accuracy': exceed_rate_accuracy,
                'avg_high_q_drop_max': avg_high_q_drop_max,
                'avg_low_q_drop_max': avg_low_q_drop_max,
                'avg_high_q_drop_weighted': avg_high_q_drop_weighted,
                'avg_low_q_drop_weighted': avg_low_q_drop_weighted,
                'avg_high_reward_drop_max': avg_high_reward_drop_max,
                'avg_low_reward_drop_max': avg_low_reward_drop_max,
                'avg_high_reward_drop_weighted': avg_high_reward_drop_weighted,
                'avg_low_reward_drop_weighted': avg_low_reward_drop_weighted,
                'avg_high_taylor_max': avg_high_taylor_max,
                'avg_low_taylor_max': avg_low_taylor_max,
                'avg_high_taylor_weighted': avg_high_taylor_weighted,
                'avg_low_taylor_weighted': avg_low_taylor_weighted,
                'avg_high_exceed_rate': avg_high_exceed_rate,
                'avg_low_exceed_rate': avg_low_exceed_rate,
                'avg_delta_max_q_drop': avg_delta_max_q_drop,
                'avg_delta_weighted_q_drop_sum': avg_delta_weighted_q_drop_sum,
                'avg_delta_max_reward_drop': avg_delta_max_reward_drop,
                'avg_delta_weighted_reward_drop_sum': avg_delta_weighted_reward_drop_sum,
                'avg_delta_max_abs_taylor_deviation': avg_delta_max_abs_taylor_deviation,
                'avg_delta_weighted_taylor_deviation_sum': avg_delta_weighted_taylor_deviation_sum,
                'avg_delta_exceed_rate': avg_delta_exceed_rate,
                'failed_expectations_count': len(failed_expectations),
                'failed_expectations': failed_expectations,
                'raw_pair_data': pair_data
            }
            
            # Print summary for this pair
            print(f"  Total Experiments: {total_experiments_for_pair}")
            print(f"  Patient Zero Detection Accuracy: {patient_zero_accuracy:.3f} ({correct_patient_zero}/{total_with_detection})")
            print(f"  Q-Drop Max Expectation Accuracy: {q_drop_max_accuracy:.3f} ({q_drop_max_expectation_correct}/{total_experiments_for_pair})")
            print(f"  Q-Drop Weighted Expectation Accuracy: {q_drop_weighted_accuracy:.3f} ({q_drop_weighted_expectation_correct}/{total_experiments_for_pair})")
            print(f"  Reward-Drop Max Expectation Accuracy: {reward_drop_max_accuracy:.3f} ({reward_drop_max_expectation_correct}/{total_experiments_for_pair})")
            print(f"  Reward-Drop Weighted Expectation Accuracy: {reward_drop_weighted_accuracy:.3f} ({reward_drop_weighted_expectation_correct}/{total_experiments_for_pair})")
            print(f"  Taylor Max Expectation Accuracy: {taylor_max_accuracy:.3f} ({taylor_max_expectation_correct}/{total_experiments_for_pair})")
            print(f"  Taylor Weighted Expectation Accuracy: {taylor_weighted_accuracy:.3f} ({taylor_weighted_expectation_correct}/{total_experiments_for_pair})")
            print(f"  Exceed Rate Expectation Accuracy: {exceed_rate_accuracy:.3f} ({exceed_rate_expectation_correct}/{total_experiments_for_pair})")
            print(f"  Average Delta Q-Drop Max: {avg_delta_max_q_drop:.6f}")
            print(f"  Average Delta Reward-Drop Max: {avg_delta_max_reward_drop:.6f}")
            print(f"  Average Delta Taylor Max: {avg_delta_max_abs_taylor_deviation:.6f}")
            print(f"  Average Delta Exceed Rate: {avg_delta_exceed_rate:.6f}")
            print(f"  Failed Expectations: {len(failed_expectations)}")
        
        return pair_specific_results
    
    def print_pair_specific_summary(self, pair_specific_results):
        """Print a comprehensive summary of pair-specific results."""
        if not pair_specific_results:
            return
        
        print("\n" + "="*70)
        print("PAIR-SPECIFIC RESULTS SUMMARY")
        print("="*70)
        
        # Sort pairs by overall performance (average of all accuracy metrics)
        def calculate_overall_accuracy(results):
            accuracy_metrics = [
                results['patient_zero_accuracy'],
                results['q_drop_max_accuracy'],
                results['q_drop_weighted_accuracy'],
                results['reward_drop_max_accuracy'],
                results['reward_drop_weighted_accuracy'],
                results['taylor_max_accuracy'],
                results['taylor_weighted_accuracy'],
                results['exceed_rate_accuracy']
            ]
            return np.mean([acc for acc in accuracy_metrics if not np.isnan(acc)])
        
        sorted_pairs = sorted(pair_specific_results.items(), 
                             key=lambda x: calculate_overall_accuracy(x[1]), 
                             reverse=True)
        
        print(f"{'Pair':<20} {'PZ Acc':<8} {'Q-Max':<8} {'Q-Wei':<8} {'R-Max':<8} {'R-Wei':<8} {'T-Max':<8} {'T-Wei':<8} {'E-Rate':<8} {'Failed':<8}")
        print("-" * 90)
        
        for pair_name, results in sorted_pairs:
            print(f"{pair_name:<20} "
                  f"{results['patient_zero_accuracy']:<8.3f} "
                  f"{results['q_drop_max_accuracy']:<8.3f} "
                  f"{results['q_drop_weighted_accuracy']:<8.3f} "
                  f"{results['reward_drop_max_accuracy']:<8.3f} "
                  f"{results['reward_drop_weighted_accuracy']:<8.3f} "
                  f"{results['taylor_max_accuracy']:<8.3f} "
                  f"{results['taylor_weighted_accuracy']:<8.3f} "
                  f"{results['exceed_rate_accuracy']:<8.3f} "
                  f"{results['failed_expectations_count']:<8}")
        
        print("\nLegend:")
        print("PZ Acc  = Patient Zero Accuracy")
        print("Q-Max   = Q-Drop Max Expectation Accuracy")
        print("Q-Wei   = Q-Drop Weighted Expectation Accuracy")
        print("R-Max   = Reward-Drop Max Expectation Accuracy") 
        print("R-Wei   = Reward-Drop Weighted Expectation Accuracy")
        print("T-Max   = Taylor Max Expectation Accuracy")
        print("T-Wei   = Taylor Weighted Expectation Accuracy")
        print("E-Rate  = Exceed Rate Expectation Accuracy")
        print("Failed  = Number of Failed Expectations")
        
        # Print best and worst performing pairs
        if len(sorted_pairs) > 0:
            best_pair = sorted_pairs[0]
            worst_pair = sorted_pairs[-1]
            
            print(f"\nBest performing pair: {best_pair[0]}")
            print(f"  Overall accuracy: {calculate_overall_accuracy(best_pair[1]):.3f}")
            
            print(f"\nWorst performing pair: {worst_pair[0]}")
            print(f"  Overall accuracy: {calculate_overall_accuracy(worst_pair[1]):.3f}")
    
    def save_results(self, accuracy_results, failed_expectations, pair_specific_results=None):
        """Save all results to CSV files."""
        print("\nSaving results to CSV files...")
        
        # Save accuracy results
        accuracy_file = os.path.join(self.logdir, 'accuracy_results.csv')
        with open(accuracy_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metric', 'Value'])
            for key, value in accuracy_results.items():
                writer.writerow([key, value])
        
        # Save detailed experiment results
        detailed_file = os.path.join(self.logdir, 'detailed_results.csv')
        with open(detailed_file, 'w', newline='') as csvfile:
            fieldnames = [
                'seed', 'agent_i_influencer_attacked', 'agent_j_influenced_observed', 'max_influence_t', 'min_influence_t',
                'high_patient_zero', 'high_patient_time', 'low_patient_zero', 'low_patient_time',
                'high_influencer_fault_detection_times', 'high_influenced_fault_detection_times',
                'low_influencer_fault_detection_times', 'low_influenced_fault_detection_times',
                'high_max_q_drop', 'high_weighted_q_drop_sum', 'high_max_reward_drop', 'high_weighted_reward_drop_sum',
                'high_max_abs_taylor_deviation', 'high_weighted_taylor_deviation_sum', 'high_exceed_rate', 'high_window_length',
                'low_max_q_drop', 'low_weighted_q_drop_sum', 'low_max_reward_drop', 'low_weighted_reward_drop_sum',
                'low_max_abs_taylor_deviation', 'low_weighted_taylor_deviation_sum', 'low_exceed_rate', 'low_window_length',
                'delta_max_q_drop', 'delta_weighted_q_drop_sum', 'delta_max_reward_drop', 'delta_weighted_reward_drop_sum',
                'delta_max_abs_taylor_deviation', 'delta_weighted_taylor_deviation_sum', 'delta_exceed_rate',
                'episode_length'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Process all pairs from all seeds
            for result in self.experiment_results:
                seed = result['seed']
                episode_length = result['episode_length']
                
                for pair_result in result['pair_results']:
                    row = {
                        'seed': seed,
                        'agent_i_influencer_attacked': pair_result['agent_i'],
                        'agent_j_influenced_observed': pair_result['agent_j'],
                        'max_influence_t': pair_result['max_influence_t'],
                        'min_influence_t': pair_result['min_influence_t'],
                        'high_patient_zero': pair_result['high_patient_zero'],
                        'high_patient_time': pair_result['high_patient_time'],
                        'low_patient_zero': pair_result['low_patient_zero'],
                        'low_patient_time': pair_result['low_patient_time'],
                        'high_influencer_fault_detection_times': pair_result['high_influencer_fault_detection_times'],
                        'high_influenced_fault_detection_times': pair_result['high_influenced_fault_detection_times'],
                        'low_influencer_fault_detection_times': pair_result['low_influencer_fault_detection_times'],
                        'low_influenced_fault_detection_times': pair_result['low_influenced_fault_detection_times'],
                        'episode_length': episode_length
                    }
                    
                    # Add high influence metrics
                    for key, value in pair_result['high_metrics'].items():
                        row[f'high_{key}'] = value
                    
                    # Add low influence metrics
                    for key, value in pair_result['low_metrics'].items():
                        row[f'low_{key}'] = value
                    
                    # Add delta metrics (high - low)
                    high_metrics = pair_result['high_metrics']
                    low_metrics = pair_result['low_metrics']
                    row['delta_max_q_drop'] = high_metrics['max_q_drop'] - low_metrics['max_q_drop']
                    row['delta_weighted_q_drop_sum'] = high_metrics['weighted_q_drop_sum'] - low_metrics['weighted_q_drop_sum']
                    row['delta_max_reward_drop'] = high_metrics['max_reward_drop'] - low_metrics['max_reward_drop']
                    row['delta_weighted_reward_drop_sum'] = high_metrics['weighted_reward_drop_sum'] - low_metrics['weighted_reward_drop_sum']
                    row['delta_max_abs_taylor_deviation'] = high_metrics['max_abs_taylor_deviation'] - low_metrics['max_abs_taylor_deviation']
                    row['delta_weighted_taylor_deviation_sum'] = high_metrics['weighted_taylor_deviation_sum'] - low_metrics['weighted_taylor_deviation_sum']
                    row['delta_exceed_rate'] = high_metrics['exceed_rate'] - low_metrics['exceed_rate']
                    
                    writer.writerow(row)
        
        # Save failed expectations
        if failed_expectations:
            failed_file = os.path.join(self.logdir, 'failed_expectations.csv')
            with open(failed_file, 'w', newline='') as csvfile:
                fieldnames = [
                    'seed', 'agent_i_influencer_attacked', 'agent_j_influenced_observed', 
                    'q_drop_max_failed', 'q_drop_weighted_failed', 'reward_drop_max_failed', 'reward_drop_weighted_failed',
                    'taylor_max_failed', 'taylor_weighted_failed', 'exceed_rate_failed',
                    'high_q_drop_max', 'low_q_drop_max', 'high_q_drop_weighted', 'low_q_drop_weighted',
                    'high_reward_drop_max', 'low_reward_drop_max', 'high_reward_drop_weighted', 'low_reward_drop_weighted',
                    'high_taylor_max', 'low_taylor_max', 'high_taylor_weighted', 'low_taylor_weighted',
                    'high_exceed_rate', 'low_exceed_rate'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(failed_expectations)
        
        # Save failed seeds
        if self.failed_seeds:
            failed_seeds_file = os.path.join(self.logdir, 'failed_seeds.csv')
            with open(failed_seeds_file, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['seed', 'error'])
                writer.writeheader()
                writer.writerows(self.failed_seeds)
        
        # Save pair-specific results
        if pair_specific_results:
            # Create pair-specific directory
            pair_dir = os.path.join(self.logdir, 'pair_specific_results')
            os.makedirs(pair_dir, exist_ok=True)
            
            # Save overall pair-specific accuracy summary
            pair_summary_file = os.path.join(self.logdir, 'pair_specific_accuracy_summary.csv')
            with open(pair_summary_file, 'w', newline='') as csvfile:
                fieldnames = [
                    'pair_name', 'agent_i_influencer', 'agent_j_influenced', 'total_experiments',
                    'patient_zero_accuracy', 'high_patient_zero_accuracy', 'low_patient_zero_accuracy',
                    'q_drop_max_accuracy', 'q_drop_weighted_accuracy',
                    'reward_drop_max_accuracy', 'reward_drop_weighted_accuracy',
                    'taylor_max_accuracy', 'taylor_weighted_accuracy', 'exceed_rate_accuracy',
                    'avg_high_q_drop_max', 'avg_low_q_drop_max',
                    'avg_high_reward_drop_max', 'avg_low_reward_drop_max',
                    'avg_high_taylor_max', 'avg_low_taylor_max',
                    'avg_high_exceed_rate', 'avg_low_exceed_rate',
                    'avg_delta_max_q_drop', 'avg_delta_weighted_q_drop_sum',
                    'avg_delta_max_reward_drop', 'avg_delta_weighted_reward_drop_sum',
                    'avg_delta_max_abs_taylor_deviation', 'avg_delta_weighted_taylor_deviation_sum',
                    'avg_delta_exceed_rate',
                    'failed_expectations_count'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for pair_name, results in pair_specific_results.items():
                    row = {
                        'pair_name': pair_name,
                        'agent_i_influencer': results['agent_i'],
                        'agent_j_influenced': results['agent_j'],
                        'total_experiments': results['total_experiments'],
                        'patient_zero_accuracy': results['patient_zero_accuracy'],
                        'high_patient_zero_accuracy': results['high_patient_zero_accuracy'],
                        'low_patient_zero_accuracy': results['low_patient_zero_accuracy'],
                        'q_drop_max_accuracy': results['q_drop_max_accuracy'],
                        'q_drop_weighted_accuracy': results['q_drop_weighted_accuracy'],
                        'reward_drop_max_accuracy': results['reward_drop_max_accuracy'],
                        'reward_drop_weighted_accuracy': results['reward_drop_weighted_accuracy'],
                        'taylor_max_accuracy': results['taylor_max_accuracy'],
                        'taylor_weighted_accuracy': results['taylor_weighted_accuracy'],
                        'exceed_rate_accuracy': results['exceed_rate_accuracy'],
                        'avg_high_q_drop_max': results['avg_high_q_drop_max'],
                        'avg_low_q_drop_max': results['avg_low_q_drop_max'],
                        'avg_high_reward_drop_max': results['avg_high_reward_drop_max'],
                        'avg_low_reward_drop_max': results['avg_low_reward_drop_max'],
                        'avg_high_taylor_max': results['avg_high_taylor_max'],
                        'avg_low_taylor_max': results['avg_low_taylor_max'],
                        'avg_high_exceed_rate': results['avg_high_exceed_rate'],
                        'avg_low_exceed_rate': results['avg_low_exceed_rate'],
                        'avg_delta_max_q_drop': results['avg_delta_max_q_drop'],
                        'avg_delta_weighted_q_drop_sum': results['avg_delta_weighted_q_drop_sum'],
                        'avg_delta_max_reward_drop': results['avg_delta_max_reward_drop'],
                        'avg_delta_weighted_reward_drop_sum': results['avg_delta_weighted_reward_drop_sum'],
                        'avg_delta_max_abs_taylor_deviation': results['avg_delta_max_abs_taylor_deviation'],
                        'avg_delta_weighted_taylor_deviation_sum': results['avg_delta_weighted_taylor_deviation_sum'],
                        'avg_delta_exceed_rate': results['avg_delta_exceed_rate'],
                        'failed_expectations_count': results['failed_expectations_count']
                    }
                    writer.writerow(row)
            
            # Save detailed results for each pair in separate CSV files
            for pair_name, results in pair_specific_results.items():
                # Detailed experiment results for this pair
                pair_detailed_file = os.path.join(pair_dir, f'{pair_name}_detailed_results.csv')
                with open(pair_detailed_file, 'w', newline='') as csvfile:
                    fieldnames = [
                        'seed', 'agent_i_influencer_attacked', 'agent_j_influenced_observed', 'max_influence_t', 'min_influence_t',
                        'high_patient_zero', 'high_patient_time', 'low_patient_zero', 'low_patient_time',
                        'high_influencer_fault_detection_times', 'high_influenced_fault_detection_times',
                        'low_influencer_fault_detection_times', 'low_influenced_fault_detection_times',
                        'high_max_q_drop', 'high_weighted_q_drop_sum', 'high_max_reward_drop', 'high_weighted_reward_drop_sum',
                        'high_max_abs_taylor_deviation', 'high_weighted_taylor_deviation_sum', 'high_exceed_rate', 'high_window_length',
                        'low_max_q_drop', 'low_weighted_q_drop_sum', 'low_max_reward_drop', 'low_weighted_reward_drop_sum',
                        'low_max_abs_taylor_deviation', 'low_weighted_taylor_deviation_sum', 'low_exceed_rate', 'low_window_length',
                        'delta_max_q_drop', 'delta_weighted_q_drop_sum', 'delta_max_reward_drop', 'delta_weighted_reward_drop_sum',
                        'delta_max_abs_taylor_deviation', 'delta_weighted_taylor_deviation_sum', 'delta_exceed_rate',
                        'episode_length'
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    writer.writeheader()
                    
                    for data in results['raw_pair_data']:
                        row = {
                            'seed': data['seed'],
                            'agent_i_influencer_attacked': data['agent_i'],
                            'agent_j_influenced_observed': data['agent_j'],
                            'max_influence_t': data['max_influence_t'],
                            'min_influence_t': data['min_influence_t'],
                            'high_patient_zero': data['high_patient_zero'],
                            'high_patient_time': data['high_patient_time'],
                            'low_patient_zero': data['low_patient_zero'],
                            'low_patient_time': data['low_patient_time'],
                            'high_influencer_fault_detection_times': data['high_influencer_fault_detection_times'],
                            'high_influenced_fault_detection_times': data['high_influenced_fault_detection_times'],
                            'low_influencer_fault_detection_times': data['low_influencer_fault_detection_times'],
                            'low_influenced_fault_detection_times': data['low_influenced_fault_detection_times'],
                            'episode_length': data['episode_length']
                        }
                        
                        # Add high influence metrics
                        for key, value in data['high_metrics'].items():
                            row[f'high_{key}'] = value
                        
                        # Add low influence metrics
                        for key, value in data['low_metrics'].items():
                            row[f'low_{key}'] = value
                        
                        # Add delta metrics (high - low)
                        high_metrics = data['high_metrics']
                        low_metrics = data['low_metrics']
                        row['delta_max_q_drop'] = high_metrics['max_q_drop'] - low_metrics['max_q_drop']
                        row['delta_weighted_q_drop_sum'] = high_metrics['weighted_q_drop_sum'] - low_metrics['weighted_q_drop_sum']
                        row['delta_max_reward_drop'] = high_metrics['max_reward_drop'] - low_metrics['max_reward_drop']
                        row['delta_weighted_reward_drop_sum'] = high_metrics['weighted_reward_drop_sum'] - low_metrics['weighted_reward_drop_sum']
                        row['delta_max_abs_taylor_deviation'] = high_metrics['max_abs_taylor_deviation'] - low_metrics['max_abs_taylor_deviation']
                        row['delta_weighted_taylor_deviation_sum'] = high_metrics['weighted_taylor_deviation_sum'] - low_metrics['weighted_taylor_deviation_sum']
                        row['delta_exceed_rate'] = high_metrics['exceed_rate'] - low_metrics['exceed_rate']
                        
                        writer.writerow(row)
                
                # Failed expectations for this pair
                if results['failed_expectations']:
                    pair_failed_file = os.path.join(pair_dir, f'{pair_name}_failed_expectations.csv')
                    with open(pair_failed_file, 'w', newline='') as csvfile:
                        fieldnames = [
                            'seed', 'agent_i_influencer_attacked', 'agent_j_influenced_observed', 
                            'q_drop_max_failed', 'q_drop_weighted_failed', 'reward_drop_max_failed', 'reward_drop_weighted_failed',
                            'taylor_max_failed', 'taylor_weighted_failed', 'exceed_rate_failed',
                            'high_q_drop_max', 'low_q_drop_max', 'high_q_drop_weighted', 'low_q_drop_weighted',
                            'high_reward_drop_max', 'low_reward_drop_max', 'high_reward_drop_weighted', 'low_reward_drop_weighted',
                            'high_taylor_max', 'low_taylor_max', 'high_taylor_weighted', 'low_taylor_weighted',
                            'high_exceed_rate', 'low_exceed_rate'
                        ]
                        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(results['failed_expectations'])
        
        # Save cumulative influences CSV
        self.save_cumulative_influences_csv()
        # Save directional derivatives CSV
        self.save_directional_derivatives_csv()

        # Save Taylor deviations CSV
        self.save_taylor_deviations_csv()
        
        print(f"Results saved to {self.logdir}")
        print(f"- Accuracy results: {accuracy_file}")
        print(f"- Detailed results: {detailed_file}")
        if failed_expectations:
            print(f"- Failed expectations: {failed_file}")
        if self.failed_seeds:
            print(f"- Failed seeds: {failed_seeds_file}")
        if pair_specific_results:
            print(f"- Pair-specific accuracy summary: {pair_summary_file}")
            print(f"- Pair-specific detailed results saved in: {pair_dir}")
            print(f"  * {len(pair_specific_results)} individual pair CSV files created")
        print(f"- Cumulative influences: cumulative_influences_all_seeds.csv")
    
    def cleanup(self):
        """Clean up resources."""
        if self.env:
            self.env.close()
        if self.runner:
            self.runner.close()
    
    def run_full_experiment(self):
        """Run the complete multi-seed experiment pipeline."""
        self.setup_experiment(initial_seed=0)  # শুরুতে seed=0 দিয়ে setup করি
        self.run_all_experiments(self.runner)
        accuracy_results, failed_expectations = self.compute_accuracies()
        pair_specific_results = self.compute_pair_specific_accuracies()
        self.print_pair_specific_summary(pair_specific_results)
        self.save_results(accuracy_results, failed_expectations, pair_specific_results)
        print(f"\nMulti-seed experiment completed successfully!")
        print(f"Pair-specific analysis completed for {len(pair_specific_results)} agent pairs")
        print(f"Results saved to: {self.logdir}")
        # Patient zero analysis summary and results saving
        self.patient_zero_analyzer.print_summary_dual()
        patient_zero_stats = self.patient_zero_analyzer.get_statistics_dual()
        
        # Save patient zero analysis results
        pz_analysis_file = os.path.join(self.logdir, "patient_zero_analysis_detailed.csv")
        pz_summary_file = os.path.join(self.logdir, "patient_zero_analysis_summary.json")
        
        self.patient_zero_analyzer.save_detailed_results(pz_analysis_file)
        
        # Save summary statistics as JSON
        import json
        with open(pz_summary_file, 'w') as f:
            json.dump(patient_zero_stats, f, indent=2)
        print(f"\nMulti-seed experiment completed successfully!")
        print(f"Pair-specific analysis completed for {len(pair_specific_results)} agent pairs")
        print(f"Patient zero detection and traceback analysis completed!")
        print(f"Patient zero analysis saved to:")
        print(f"  - Detailed results: {pz_analysis_file}")
        print(f"  - Summary statistics: {pz_summary_file}")
        print(f"Results saved to: {self.logdir}")
        self.cleanup()


def create_config_from_args():
    """Create configuration from command line arguments."""
    parser = argparse.ArgumentParser(description="Multi-seed statistics experiment")
    # parser.add_argument("-N",type=int ,help="Number of agents")
    parser.add_argument("--total_experiments", type=int, default=400,
                        help="Total number of seed experiments to run")
    parser.add_argument("--total_episodes", type=int, default=100,)
    parser.add_argument("--filepath", type=str, default=None,
                        help="Path to save/load experiment data")
    parser.add_argument("--reward",type=float, default=None)
    parser.add_argument("--K_SIGMA",type=int, default=1)
    parser.add_argument("--folder_name",type=str, help="Folder name to save results")
    # parser.add_argument("model_path", help="Model directory")
    # parser.add_argument("--total_experiments", type=int, default=100,
    #                     help="Total number of seed experiments to run")
    
    return parser.parse_args()


def main():
    """Main function to run multi-seed statistics experiment."""
    config = create_config_from_args()
    runner = MultiSeedExperimentRunner(config)
    runner.run_full_experiment()


if __name__ == '__main__':
    main()