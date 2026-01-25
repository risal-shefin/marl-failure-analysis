"""Train an algorithm."""
import argparse
from collections import deque
import sys
import os
import yaml
# Add HARL to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from harl.utils.configs_tools import get_defaults_yaml_args, update_args
import numpy as np
import torch
from harl.utils.trans_tools import _t2n 
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
import csv
def plot_results(results, results_attacked, atk_agent_id, logdir):
        os.makedirs(logdir, exist_ok=True)
        n = len(results[0])  # number of agents
        t = len(results)     # number of time steps
        
        # Create n subplots in a row
        fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
        fig.suptitle(f'Taylor Error (Worst Action Attack | Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.95)
        
        # Ensure axes is iterable even for single agent case
        if n == 1:
            axes = [axes]
        
        for i in range(n):
            ax = axes[i]
            
            # Extract time series for agent i
            normal_series = [results[t][i] for t in range(len(results))]
            attacked_series = [results_attacked[t][i] for t in range(len(results_attacked))]
            
            # Plot the curves
            steps_normal = range(len(normal_series))
            steps_attacked = range(len(attacked_series))
            ax.plot(steps_normal, normal_series, 'b-', label='Normal', linewidth=2)
            ax.plot(steps_attacked, attacked_series, 'r-', label='Attacked', linewidth=2)
            
            ax.set_xlabel('Step')
            ax.set_ylabel('Taylor Delta Error')
            ax.set_title(f'Agent {i}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(os.path.join(logdir, f'plot_analysis_attacked_{atk_agent_id}.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved analysis plot to {logdir}")

def save_matrix_to_files(matrix, agent_id, logdir, suffix=""):
    """
    Save matrix data to a CSV file for all timesteps.
    
    Args:
        matrix: List of timesteps, where each timestep contains n_agent data
        agent_id: ID of the agent
        logdir: Directory to save the file
    """
    filename = f"mappo_taylor_error_atk_free_agent_{suffix}.csv"
    filepath = os.path.join(logdir, filename)
    
    # Create header row
    header = ["agent", "timestep", "mean", "variance", "std_dev", "q1", "q3"]
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        for timestep, timestep_metrics in enumerate(matrix):
            row = [agent_id, timestep]
            for value in timestep_metrics:
                row.append(value)
            writer.writerow(row)
    
    print(f"Saved {len(matrix)} timestep matrices to {filepath}")

def compute_taylor_policy(runner, eval_obs, eval_available_actions, eval_rnn_states):
        # states_tensor = torch.stack([torch.tensor(state_dict[k], dtype=torch.float32, requires_grad=True) for k in state_dict.keys()])
        eval_obs = torch.tensor(eval_obs, dtype=torch.float32, requires_grad=True)
        delta_errors = []
        eval_actions_collector = []
        eval_masks = np.ones(
            (runner.algo_args["eval"]["n_eval_rollout_threads"], runner.num_agents, 1),
            dtype=np.float32,
        )

        for agent_id in range(runner.num_agents):
            cur_obs = eval_obs[:, agent_id]
            eval_actions, eval_actions_log_prob, temp_rnn_state = runner.actor[agent_id].get_actions(
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
            _, perturb_log_prob, _ = runner.actor[agent_id].get_actions(
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

def eval(runner, attack_status=False, attack_agent_id=0,seed=23,randomness=0.25):
    """Evaluate the model."""
    
    # print(f"Seeding eval with seed {seed}")
    eval_episode = 0

    eval_obs, eval_share_obs, eval_available_actions = runner.eval_envs.reset(seed=seed)

    if np.random.random() < randomness:  # 10% chance to add noise to observations
        noise = np.random.normal(loc=0.0, scale=0.1, size=eval_obs.shape)
        eval_obs = eval_obs + noise
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

    results = list()
    result_deque = [deque(maxlen=5) for _ in range(runner.num_agents)]

    while True:
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
            eval_actions[0][attack_agent_id] = runner.eval_envs.action_space[attack_agent_id].sample()  # Random action for attack agent


        # calculating taylor policy
        delta_errors = compute_taylor_policy(runner, eval_obs, eval_available_actions, eval_rnn_states_backup)
        for j in range(runner.num_agents):
            result_deque[j].append(delta_errors[j])
        results.append([np.mean(list(result_deque[j])) for j in range(runner.num_agents)])

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

        for eval_i in range(runner.algo_args["eval"]["n_eval_rollout_threads"]):
            if eval_dones_env[eval_i]:
                eval_episode += 1
                # runner.logger.eval_thread_done(
                #     eval_i
                # )  # logger callback when an episode is done

        if eval_episode >= runner.algo_args["eval"]["eval_episodes"]:
            # runner.logger.eval_log(
            #     eval_episode
            # )  # logger callback at the end of evaluation
            break
    
    return results


# def restore(runner,reward,filepath="/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v2-discrete/happo/installtest/seed-00042-2025-08-03-20-41-48/models"):
#         """Restore model parameters."""
#         for agent_id in range(runner.num_agents):
#             policy_actor_state_dict = torch.load(
#                 str(filepath)
#                 + "/actor_agent"
#                 + str(agent_id)
#                 + "_reward_" + str(reward)
        
#                 + ".pt"
#             )
#             runner.actor[agent_id].actor.load_state_dict(policy_actor_state_dict)
#         if not runner.algo_args["render"]["use_render"]:
#             policy_critic_state_dict = torch.load(
#                 str(filepath)
#                 + "/critic_agent"
#                 + "_reward_" + str(reward)
        
#                 + ".pt"
#             )
#             runner.critic.critic.load_state_dict(policy_critic_state_dict)
#             if runner.value_normalizer is not None:
#                 value_normalizer_state_dict = torch.load(
#                     str(filepath)
#                     + "/value_normalizer"
#                     + "_reward_" + str(reward)
            
#                     + ".pt"
#                 )
#                 runner.value_normalizer.load_state_dict(value_normalizer_state_dict)

def restore_model(runner, restore_dir, reward):
    """Restore trained model from checkpoint"""
    for agent_id in range(runner.num_agents):
        policy_actor_state_dict = torch.load(
            os.path.join(restore_dir, f"actor_agent{agent_id}_{reward}.pt"),
            weights_only=False
        )
        runner.actor[agent_id].actor.load_state_dict(policy_actor_state_dict)
    
    if not runner.algo_args["render"]["use_render"]:
        policy_critic_state_dict = torch.load(
            os.path.join(restore_dir, f"critic_agent_{reward}.pt"),
            weights_only=False
        )
        runner.critic.critic.load_state_dict(policy_critic_state_dict)
        
        if runner.value_normalizer is not None:
            value_normalizer_state_dict = torch.load(
                os.path.join(restore_dir, f"value_normalizer_{reward}.pt"),
                weights_only=False
            )
            runner.value_normalizer.load_state_dict(value_normalizer_state_dict)
    # if not runner.algo_args["render"]["use_render"]:
    #     policy_critic_state_dict = torch.load(
    #         os.path.join(restore_dir, f"critic_agent_{reward}.pt"),
    #         weights_only=False, map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #     )
    #     runner.critic.critic.load_state_dict(policy_critic_state_dict)
        
    #     if runner.value_normalizer is not None:
    #         value_normalizer_state_dict = torch.load(
    #             os.path.join(restore_dir, f"value_normalizer_{reward}.pt"),
    #             weights_only=False, map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #         )
    #         runner.value_normalizer.load_state_dict(value_normalizer_state_dict)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="happo",
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
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    parser.add_argument(
        "--reward",
        type=float,
        default=-5.14,
        help="Reward value to restore the model."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=376,
        help="Random seed for initialization."
    )
    parser.add_argument(
        "--total_episodes",
        type=int,
        default=5000,
        help="Total number of episodes to run."
    )
    parser.add_argument(
        "--randomness",
        type=float,
        default=0.25,
        help="Randomness factor for evaluation."
    )
    parser.add_argument(
        "--filepath",
        type=str,
        default="/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v2-discrete/happo/installtest/seed-00042-2025-08-03-20-41-48/models",
        help="Filepath to restore the model from."
    )
    parser.add_argument(
        "--save_result_dir",
        type=str,
        default=None,
        help="Directory to save the results."
    )
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
    # log_dir = algo_args['attack']['log_dir']
    alg_name = "hatrpo" #algo_args['attack']['algo_name']
    date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    randomness = args.get('randomness', 0.25)
    log_path = os.path.join(args['save_result_dir'], f"seed-{args['seed']}", alg_name, str(randomness) ,date)
    # if args['save_result_dir'] is not None:
    #     log_path = os.path.join(args['save_result_dir'], alg_name, date)
    # else:
    #     log_path = os.path.join(alg_name, date)
    os.makedirs(log_path, exist_ok=True)

    algo_args['eval']['n_eval_rollout_threads'] = 1
    algo_args['eval']['eval_episodes'] = 1
    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    restore_model(runner,args['filepath'], args['reward'])  # Restore the model with specific reward and episode
    runner.prep_training()
    total_episodes = args['total_episodes']
    print(f"Computing Taylor policy over {total_episodes} episodes...")
    # attack_agent_id = 2
    result_dataset = [{} for _ in range(runner.num_agents)]
    print(f"Randomness in eval set to {randomness}")
    for i in tqdm(range(total_episodes), desc="Processing episodes"):
        results = eval(runner,randomness=randomness,seed=args['seed'])  # Run evaluation
        for timestep in range(len(results)):
            for agent_id in range(runner.num_agents):
                if timestep not in result_dataset[agent_id]:
                    result_dataset[agent_id][timestep] = []
                result_dataset[agent_id][timestep].append(results[timestep][agent_id])
    
    for agent_id in range(runner.num_agents):
        # Compute mean and variance for each agent pair across all episodes
        print(f"\n---- Agent {agent_id}:")
        metrics_mat = []
        for timestep, timestep_values in result_dataset[agent_id].items():
            print(f"\n ---- Timestep {timestep}:")
            # Extract values for agent pair (i,j) across all episodes
            mean_val = np.mean(timestep_values)
            var_val = np.var(timestep_values)
            std_dev_val = np.std(timestep_values)
            q1, q3 = np.percentile(timestep_values, [25, 75])
            print(f"Agent {agent_id}: mean = {mean_val:.4f}, variance = {var_val:.4f}, std_dev = {std_dev_val:.4f}, IQR = {q3 - q1:.4f}")
            metrics = [mean_val, var_val, std_dev_val, q1, q3]
            metrics_mat.append(metrics)

        save_matrix_to_files(metrics_mat, agent_id, log_path, suffix=f"{agent_id}")

    print(f"\nSaved all agent metrics to {log_path}")
    # results_normal = eval(runner)  # Run evaluation
    # results_attackec = eval(runner, attack_status=True, attack_agent_id=attack_agent_id)  # Run evaluation with attack

    # log_dir = algo_args['attack']['log_dir']
    # alg_name = algo_args['attack']['algo_name']
    # date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # log_path = os.path.join(log_dir, alg_name, date)
    # os.makedirs(log_path, exist_ok=True)
    # # plot_results(results_normal, results_attackec, atk_agent_id=attack_agent_id, logdir=log_path)

    # runner.run()
    runner.close()



if __name__ == "__main__":
    main()
