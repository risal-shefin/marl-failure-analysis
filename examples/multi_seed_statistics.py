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

# Import all the modular components
from modules.constants import DEVICE, K_SIGMA, torch_device,REWARD, FILEPATH, ATTACK_ID
# from modules.environment import create_environment
from modules.detection import get_patient_zero_detection
# from modules.core_experiment import get_episode_data
# from modules.metrics import (
#     compute_taylor_delta_policy,
#     compute_pairwise_action_influences,
#     collect_agent_q_values
# )
import warnings
warnings.filterwarnings("ignore")


def slice_avail(avail, agent_id):
    """Extract available actions for a specific agent"""
    if avail is None:
        return None
    first = avail[0]
    if first is None:
        return None
    return avail[:, agent_id]

class MultiSeedExperimentRunner:
    """
    Multi-seed experiment runner for analyzing influence-based attacks.
    """
    
    def __init__(self):
        """
        Initialize the multi-seed experiment runner.
        
        Args:
            config: Configuration object containing experiment parameters
        """
        
        self.runner = None
        self.env = None
        self.logdir = None
        # self.total_experiments = config.total_experiments
        self.total_experiments= 100
        self.total_episodes = 100
        self.gamma = 0.99  # Discount factor for weighted metrics
        
        # Results storage
        self.experiment_results = []
        self.failed_seeds = []
    
    
    def setup_experiment(self):
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
                "maddpg",
                "matd3",
                "mappo",
            ],
            help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, matd3, mappo.",
        )
        parser.add_argument(
            "--env",
            type=str,
            default="pettingzoo_mpe",
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
        # print(f"Checking if centralize q is set : {algo_args['algo']['use_centralized_q']}")
        # restore(self.runner,args['reward'],args['filepath'])  # Restore the model with specific reward and episode
        restore(self.runner,REWARD,FILEPATH)  
        print("Model restored successfully.")
        self.runner.prep_training()
        
        # Create log directory
        cwd = os.getcwd()
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        env_type = 'discrete'
        self.logdir = os.path.join(cwd, 'test-runs', f"{args['env']}", 
                                  f"{timestamp}_multi_seed_stats_{self.total_experiments}")
        os.makedirs(self.logdir, exist_ok=True)
        
        # Create environment
        # self.env = create_environment(self.config, self.maddpg)
        
        # Prepare MADDPG for training mode
        device_str = 'gpu' if DEVICE == 'gpu' else 'cpu'
        # self.maddpg.prep_training(device=device_str)
        
        print(f"Multi-seed experiment setup complete. Log directory: {self.logdir}")
        print(f"Will run {self.total_experiments} experiments")

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

    def compute_reference_taylor_error(self, runner, seed, total_episodes=1000, attack_status=False, attack_agent_id=0, randomness=0.25):

        results = [{} for _ in range(runner.num_agents)]
        for episode in tqdm(range(total_episodes), desc="Taylor Compute episodes", total=total_episodes):
            eval_obs, eval_share_obs, eval_available_actions = runner.eval_envs.reset(seed) if seed is not None else runner.eval_envs.reset()
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
                if np.random.random() < randomness:  # 10% chance to add noise to observations
                    noise = np.random.normal(loc=0.0, scale=0.01, size=eval_obs.shape)
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

    def compute_pairwise_action_influence(self,runner, eval_obs, eval_actions):
        """Compute Frobenius norms of cross-agent Hessian blocks for all (i, j).
        Returns an N x N matrix where entry [i][j] approximates || \partial^2 v_i / (\partial obs_i \partial obs_j) ||_F.
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
        share_obs = concatenated_obs.unsqueeze(0)
        concatenated_actions = torch.cat([torch.tensor(eval_actions[0][i], dtype=torch.float32,requires_grad=True) for i in range(n_agents)], dim=0).unsqueeze(0)
        # concatenated_actions = torch.stack([torch.tensor(eval_actions[0][i], dtype=torch.float32, requires_grad=True) for i in range(n_agents)], dim=0)
        # print(f"Concatenated actions shape: {concatenated_actions.shape}")
        # print(f"Share obs shape: {share_obs.shape}")

        N = n_agents
        results = [[0.0 for _ in range(N)] for _ in range(N)]

        for i in range(N):
            # gradient of v_i wrt agent i obs
            q_value = runner.central_q[i].get_q_values(
                share_obs,
                concatenated_actions,
                gradNeed=True
            ).squeeze()
            # print(f"Q value for agent {i}: {q_value.item()}")
            for j in range(N):
                grad_i = torch.autograd.grad(q_value, agent_obs_tensors[j], create_graph=True, retain_graph=True)[0]
                results[i][j] = grad_i.norm(p=2).item()

        return results

    def eval(self, runner, attack_status=False, attack_agent_id=0, seed=None, ref_vals=None, ref_std_devs=None, collect_q_flag=False,min_window=8,max_window=12,observe_agent=None):
        """Evaluate the model."""
        
        eval_episode = 0

        eval_obs, eval_share_obs, eval_available_actions = runner.eval_envs.reset(seed=seed) if seed is not None else runner.eval_envs.reset()

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

            if attack_status and (cnt>=min_window and cnt<=max_window):
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
                    print(f" [!!!] Attack launched on agent {attack_agent_id} at timestep: {cnt}")
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
                        print(f"Agent {agent_id} appears to be dead or has only one action. Using index 0.")
                        eval_actions[0][attack_agent_id] = 0
                    else:
                        worst_action = torch.argmin(q_values).item()
                        eval_actions[0][attack_agent_id] = worst_action
                        print(f"Agent {attack_agent_id} worst action under current policy: {worst_action}")
                attacked_steps.append(cnt)


            # calculating taylor policy
            pairwise_action_value_influence = self.compute_pairwise_action_influence(runner, eval_obs, eval_actions)
            # exit("Exiting for debug")
            pairwise_action_value_influence_history.append(pairwise_action_value_influence)
            delta_errors = self.compute_taylor_policy(runner, eval_obs, eval_available_actions, eval_rnn_states_backup)
  

            for i in range(runner.num_agents):
                result_deques[i].append(delta_errors[i])
                taylor_approx_error = np.mean(result_deques[i])
                taylor_history[i].append(taylor_approx_error)

 

                # Detect anomalies based on Taylor approximation error using pre-computed history
                if i not in fault_first_detected:
                    
                    historical_mean = ref_vals[i][cnt]  
                    historical_std = ref_std_devs[i][cnt]
                    # Ensure minimum std deviation to avoid division by zero
                    if historical_std < 1e-6:
                        historical_std = 1e-6
                    
                    

                    if abs(taylor_approx_error - historical_mean) > K_SIGMA * historical_std:
                        print(f" [!!!] Anomaly detected for agent {i} at timestep: {cnt}. Taylor Appx. Error: {taylor_approx_error}")
                        # print(f"     >> Historical bounds: [{lower_bound:.6f}, {upper_bound:.6f}], Mean: {historical_mean:.6f}, Std: {historical_std:.6f}")
                        fault_first_detected[i] = cnt
                        # Cascading Impact Analysis
                        prev_faults = [(f, tf) for f, tf in fault_first_detected.items() if f != i and tf < cnt]
                        contribs = {}
                        if len(prev_faults) > 0:
                            for f, tf in prev_faults:
                                values_over_time = [frob_norms_matrix_history[tau][i][f] for tau in range(tf, cnt + 1) if tau < len(frob_norms_matrix_history)]
                                if len(values_over_time) > 0:
                                    contribs[f] = float(np.mean(values_over_time))
                            if len(contribs) > 0:
                                ranked = sorted(contribs.items(), key=lambda x: x[1], reverse=True)
                                print(f"     >> Potential contributors to fault in agent {i} (mean ||H_{{i,f}}||_F from t_f to {cnt}): {ranked}")
                        fault_timeline.append({
                            'agent': i,
                            't': cnt,
                            'contribs': contribs
                        })

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
        
        

        if attack_status:
            return {
            'fault_timeline': fault_timeline,
            'q_values_history': q_values_list,
            'taylor_errors_history': taylor_error_list,
            'episode_length': cnt,
            'episode_reward': reward_ep,
            'attack_timestep': min_window,
            'attacked_agent': attack_agent_id,
            'observed_agent': observe_agent_id,
            'stepwise_rewards': total_rewards,
        }
        else:
            return {
            'action_influences_history': pairwise_action_value_influence_history,
            'q_values_history': q_values_list,
            'episode_length': cnt,
            'stepwise_rewards': total_rewards,
        }
    
    def find_influence_timesteps(self, action_influences_history, agent_i, agent_j, first_quarter_steps):
        """
        Find max and min influence timesteps of agent i on agent j in first 25% of episode.
        
        Args:
            action_influences_history: List of action influence matrices
            agent_i: Index of influencing agent
            agent_j: Index of influenced agent (where action_influences_matrix[t][j][i] = influence of i on j)
            first_quarter_steps: Number of steps in first quarter
            
        Returns:
            Tuple of (max_influence_timestep, min_influence_timestep)
        """
        influences = []
        for t in range(min(first_quarter_steps, len(action_influences_history))):
            # Correct indexing: action_influences_matrix[t][j][i] = influence of i on j
            influence = abs(action_influences_history[t][agent_j][agent_i])
            influences.append((influence, t))
        
        # Sort by influence magnitude
        influences.sort(key=lambda x: x[0])
        
        min_influence_t = influences[0][1]  # Lowest influence
        max_influence_t = influences[-1][1]  # Highest influence
        
        return max_influence_t, min_influence_t
    


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
            threshold = K_SIGMA * ref_std
            if taylor_deviation > threshold:
                exceed_count += 1
        
        # Compute exceed rate
        metrics['exceed_rate'] = exceed_count / window_steps
        
        return metrics

    def run_single_seed_experiment(self, runner, seed):
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
        
        # Step 1: Compute reference Taylor error
        ref_vals, ref_std_devs = self.compute_reference_taylor_error(runner,seed,total_episodes=self.total_episodes, attack_status=False, attack_agent_id=0, randomness=0.25)
        
        # print(f"### Ref vals :{ref_vals} and Ref std devs : {ref_std_devs}")
        # Step 2: Run normal episode
        normal_episode = self.eval(runner=self.runner, attack_status=False, seed=seed, ref_vals=ref_vals, ref_std_devs=ref_std_devs, collect_q_flag=True, min_window=0, max_window=0, observe_agent=None)
        action_influences_history = normal_episode['action_influences_history']
        normal_q_values_history = normal_episode['q_values_history']
        episode_length = normal_episode['episode_length']
        normal_rewards_history = normal_episode['stepwise_rewards']
        
        # Step 3: Analyze all possible ordered pairs (i, j) where i influences j
        all_pair_results = []
        first_quarter_steps = math.ceil(0.25 * episode_length)

        for agent_i in range(self.runner.num_agents):  # influencing agent
            for agent_j in range(self.runner.num_agents):  # influenced agent
                if agent_i == agent_j:
                    continue  # Skip self
                
                print(f"\nAnalyzing pair: agent_{agent_i} influences agent_{agent_j}")
                
                # Step 4: Find max and min influence timesteps of agent_i on agent_j in first 25%
                max_influence_t, min_influence_t = self.find_influence_timesteps(
                    action_influences_history, agent_i, agent_j, first_quarter_steps
                )
                
                print(f"Max influence timestep: {max_influence_t}, Min influence timestep: {min_influence_t}")
                
                # Step 5: Run attacked episodes - attack agent_i (influencer), observe impact on agent_j (influenced)
                high_influence_attack = self.eval(runner=self.runner, attack_status=True, attack_agent_id=agent_i, seed=seed, ref_vals=ref_vals, ref_std_devs=ref_std_devs, collect_q_flag=True, min_window=max_influence_t, max_window=max_influence_t+1, observe_agent=agent_j)
                low_influence_attack = self.eval(runner=self.runner, attack_status=True, attack_agent_id=agent_i, seed=seed, ref_vals=ref_vals, ref_std_devs=ref_std_devs, collect_q_flag=True, min_window=min_influence_t, max_window=min_influence_t+1, observe_agent=agent_j)

                # Determine patient zero for each attack
                high_patient_zero, high_patient_time = get_patient_zero_detection(high_influence_attack['fault_timeline'])
                low_patient_zero, low_patient_time = get_patient_zero_detection(low_influence_attack['fault_timeline'])
                
                # Compute attack metrics
                high_metrics = self.compute_attack_metrics(high_influence_attack, normal_q_values_history, normal_rewards_history,ref_vals, ref_std_devs, agent_j)
                low_metrics = self.compute_attack_metrics(low_influence_attack, normal_q_values_history, normal_rewards_history, ref_vals, ref_std_devs, agent_j)
                
                pair_result = {
                    'agent_i': agent_i,
                    'agent_j': agent_j,
                    'max_influence_t': max_influence_t,
                    'min_influence_t': min_influence_t,
                    'high_patient_zero': high_patient_zero,
                    'high_patient_time': high_patient_time,
                    'low_patient_zero': low_patient_zero,
                    'low_patient_time': low_patient_time,
                    'high_metrics': high_metrics,
                    'low_metrics': low_metrics
                }
                
                all_pair_results.append(pair_result)
                
                print(f"High influence attack - Patient zero: {high_patient_zero} at time {high_patient_time}")
                print(f"Low influence attack - Patient zero: {low_patient_zero} at time {low_patient_time}")
        
        result = {
            'seed': seed,
            'episode_length': episode_length,
            'pair_results': all_pair_results,
            'total_pairs': len(all_pair_results)
        }
        
        print(f"\nCompleted analysis for {len(all_pair_results)} agent pairs")
        
        return result

    def run_all_experiments(self, runner):
        """Run experiments for all seeds."""


        total_pairs_per_seed = runner.num_agents * (runner.num_agents - 1)
        print(f"Starting multi-seed experiments with {self.total_experiments} seeds...")
        print(f"Each seed will analyze {total_pairs_per_seed} agent pairs")
        print(f"Total pairs to analyze: {self.total_experiments * total_pairs_per_seed}")
        
        for seed in tqdm(range(self.total_experiments), desc="Running experiments"):
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
                    if high_patient_zero == attacked_agent:
                        correct_patient_zero += 1
                        high_correct_patient_zero += 1
                
                if low_patient_zero is not None:
                    total_with_detection += 1
                    low_total_with_detection += 1   
                    if low_patient_zero == attacked_agent:
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
                        'reward_drop_max_failed': not reward_drop_max_better, # adding reward drop failure
                        'reward_drop_weighted_failed': not reward_drop_weighted_better, # adding reward
                        'taylor_max_failed': not taylor_max_better,
                        'taylor_weighted_failed': not taylor_weighted_better,
                        'exceed_rate_failed': not exceed_rate_better,
                        'high_q_drop_max': high_metrics['max_q_drop'],
                        'low_q_drop_max': low_metrics['max_q_drop'],
                        'high_q_drop_weighted': high_metrics['weighted_q_drop_sum'],
                        'low_q_drop_weighted': low_metrics['weighted_q_drop_sum'],
                        'high_taylor_max': high_metrics['max_abs_taylor_deviation'],
                        'low_taylor_max': low_metrics['max_abs_taylor_deviation'],
                        'high_taylor_weighted': high_metrics['weighted_taylor_deviation_sum'],
                        'low_taylor_weighted': low_metrics['weighted_taylor_deviation_sum'],
                        'high_exceed_rate': high_metrics['exceed_rate'],
                        'low_exceed_rate': low_metrics['exceed_rate']
                    })
        
        total_experiments = len(self.experiment_results)
        
        # Compute accuracies
        patient_zero_accuracy = correct_patient_zero / total_with_detection if total_with_detection > 0 else 0
        high_patient_zero_accuracy = high_correct_patient_zero / high_total_with_detection if high_total_with_detection > 0 else 0
        low_patient_zero_accuracy = low_correct_patient_zero / low_total_with_detection if low_total_with_detection > 0 else 0
        q_drop_max_accuracy = q_drop_max_expectation_correct / total_pairs
        reward_drop_max_accuracy = reward_drop_max_expectation_correct / total_pairs # adding reward drop accuracy
        q_drop_weighted_accuracy = q_drop_weighted_expectation_correct / total_pairs
        reward_drop_weighted_accuracy = reward_drop_weighted_expectation_correct / total_pairs # adding reward drop accuracy
        taylor_max_accuracy = taylor_max_expectation_correct / total_pairs
        taylor_weighted_accuracy = taylor_weighted_expectation_correct / total_pairs
        exceed_rate_accuracy = exceed_rate_expectation_correct / total_pairs
        
        # Aggregate metrics
        avg_high_q_drop_max = np.mean([m['max_q_drop'] for m in high_metrics_list])
        avg_low_q_drop_max = np.mean([m['max_q_drop'] for m in low_metrics_list])
        avg_high_reward_drop_max = np.mean([m['max_reward_drop'] for m in high_metrics_list]) # adding reward drop
        avg_low_reward_drop_max = np.mean([m['max_reward_drop'] for m in low_metrics_list]) # adding reward drop
        avg_high_q_drop_weighted = np.mean([m['weighted_q_drop_sum'] for m in high_metrics_list])
        avg_low_q_drop_weighted = np.mean([m['weighted_q_drop_sum'] for m in low_metrics_list])
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
            'reward_drop_max_expectation_correct': reward_drop_max_expectation_correct, # adding reward drop
            'reward_drop_max_accuracy': reward_drop_max_accuracy, # adding reward drop
            'reward_drop_weighted_expectation_correct': reward_drop_weighted_expectation_correct, # adding reward drop
            'reward_drop_weighted_accuracy': reward_drop_weighted_accuracy, # adding reward drop
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
        print(f"High Influence Patient Zero Accuracy: {high_patient_zero_accuracy:.3f} ({high_correct_patient_zero}/{high_total_with_detection})")
        print(f"Low Influence Patient Zero Accuracy: {low_patient_zero_accuracy:.3f} ({low_correct_patient_zero}/{low_total_with_detection})")
        print(f"Q-Drop Max Expectation Accuracy: {q_drop_max_accuracy:.3f} ({q_drop_max_expectation_correct}/{total_pairs})")
        print(f"Reward Drop Max Expectation Accuracy: {reward_drop_max_accuracy:.3f} ({reward_drop_max_expectation_correct}/{total_pairs})") # adding reward drop accuracy
        print(f"Q-Drop Weighted Expectation Accuracy: {q_drop_weighted_accuracy:.3f} ({q_drop_weighted_expectation_correct}/{total_pairs})")
        print(f"Reward Drop Weighted Expectation Accuracy: {reward_drop_weighted_accuracy:.3f} ({reward_drop_weighted_expectation_correct}/{total_pairs})") # adding reward drop accuracy
        print(f"Taylor Max Expectation Accuracy: {taylor_max_accuracy:.3f} ({taylor_max_expectation_correct}/{total_pairs})")
        print(f"Taylor Weighted Expectation Accuracy: {taylor_weighted_accuracy:.3f} ({taylor_weighted_expectation_correct}/{total_pairs})")
        print(f"Exceed Rate Expectation Accuracy: {exceed_rate_accuracy:.3f} ({exceed_rate_expectation_correct}/{total_pairs})")
        print(f"Average High Influence Q-Drop Max: {avg_high_q_drop_max:.6f}")
        print(f"Average Low Influence Q-Drop Max: {avg_low_q_drop_max:.6f}")
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
                    if high_patient_zero == attacked_agent:
                        correct_patient_zero += 1
                        high_correct_patient_zero += 1
                
                if low_patient_zero is not None:
                    total_with_detection += 1
                    low_total_with_detection += 1
                    if low_patient_zero == attacked_agent:
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
                if not (q_drop_max_better and q_drop_weighted_better and
                        reward_drop_max_better and reward_drop_weighted_better and
                        taylor_max_better and taylor_weighted_better and
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
                'high_max_q_drop', 'high_weighted_q_drop_sum', 'high_max_reward_drop', 'high_weighted_reward_drop_sum',
                'high_max_abs_taylor_deviation', 'high_weighted_taylor_deviation_sum', 'high_exceed_rate', 'high_window_length',
                'low_max_q_drop', 'low_weighted_q_drop_sum', 'low_max_reward_drop', 'low_weighted_reward_drop_sum',
                'low_max_abs_taylor_deviation', 'low_weighted_taylor_deviation_sum', 'low_exceed_rate', 'low_window_length',
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
                        'episode_length': episode_length
                    }
                    
                    # Add high influence metrics
                    for key, value in pair_result['high_metrics'].items():
                        row[f'high_{key}'] = value
                    
                    # Add low influence metrics
                    for key, value in pair_result['low_metrics'].items():
                        row[f'low_{key}'] = value
                    
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
                        'high_max_q_drop', 'high_weighted_q_drop_sum', 'high_max_reward_drop', 'high_weighted_reward_drop_sum',
                        'high_max_abs_taylor_deviation', 'high_weighted_taylor_deviation_sum', 'high_exceed_rate', 'high_window_length',
                        'low_max_q_drop', 'low_weighted_q_drop_sum', 'low_max_reward_drop', 'low_weighted_reward_drop_sum',
                        'low_max_abs_taylor_deviation', 'low_weighted_taylor_deviation_sum', 'low_exceed_rate', 'low_window_length',
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
                            'episode_length': data['episode_length']
                        }
                        
                        # Add high influence metrics
                        for key, value in data['high_metrics'].items():
                            row[f'high_{key}'] = value
                        
                        # Add low influence metrics
                        for key, value in data['low_metrics'].items():
                            row[f'low_{key}'] = value
                        
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
    
    def cleanup(self):
        """Clean up resources."""
        if self.env:
            self.env.close()
        if self.runner:
            self.runner.close()
    
    def run_full_experiment(self):
        """Run the complete multi-seed experiment pipeline."""
        self.setup_experiment()
        self.run_all_experiments(self.runner)
        accuracy_results, failed_expectations = self.compute_accuracies()
        pair_specific_results = self.compute_pair_specific_accuracies()
        self.print_pair_specific_summary(pair_specific_results)
        self.save_results(accuracy_results, failed_expectations, pair_specific_results)
        print(f"\nMulti-seed experiment completed successfully!")
        print(f"Pair-specific analysis completed for {len(pair_specific_results)} agent pairs")
        print(f"Results saved to: {self.logdir}")
        
        self.cleanup()


# def create_config_from_args():
#     """Create configuration from command line arguments."""
#     parser = argparse.ArgumentParser(description="Multi-seed statistics experiment")
#     parser.add_argument("env_id", help="Name of environment")
#     parser.add_argument("model_path", help="Model directory")
#     parser.add_argument("--total_experiments", type=int, default=100,
#                         help="Total number of seed experiments to run")
    
#     return parser.parse_args()


def main():
    """Main function to run multi-seed statistics experiment."""
    # config = create_config_from_args()
    runner = MultiSeedExperimentRunner()
    runner.run_full_experiment()


if __name__ == '__main__':
    main()