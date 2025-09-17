from collections import deque
from make_env_pettingzoo import make_env
from datetime import datetime
import torch
import torch.nn as nn
import argparse
import gymnasium
import numpy as np
import os
import imageio
from PIL import Image, ImageDraw
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import csv
import math
from tqdm import tqdm

from MAPPO_MPE_main import Runner_MAPPO_MPE


def compute_taylor_error_policy(runner: Runner_MAPPO_MPE, states):
    states_tensor = torch.stack([torch.tensor(state, dtype=torch.float32, requires_grad=True) for state in states])

    delta_errors = []

    for i in range(runner.args.N):
        obs = states_tensor[i].unsqueeze(0)  # shape: (1, obs_dim)
        action, dist = runner.agent_n.compute_action(obs, i, evaluate=True, return_dist=True)
        target_val = dist.log_prob(action)
        grad_i = torch.autograd.grad(target_val, obs, create_graph=True, retain_graph=True)[0]
        # grad_i = torch.autograd.grad(critic_val, actions[i], create_graph=True, retain_graph=True)[0]

        eta_i = 0.01 * grad_i.sign() / torch.max(grad_i.norm(p=2), torch.tensor(1e-6))
        
        # First-order Taylor approximation: f(x + η) ≈ f(x) + ∇f(x)^T η
        j_tilde = target_val + torch.dot(grad_i.flatten(), eta_i.flatten())
        p_state = obs + eta_i
        p_action, p_dist = runner.agent_n.compute_action(p_state, i, evaluate=True, return_dist=True)
        j_perturbed = p_dist.log_prob(p_action)
        delta_error = abs(j_perturbed - j_tilde).item()
        delta_errors.append(delta_error)

    return delta_errors


def get_episode_data(env, runner: Runner_MAPPO_MPE):
    # Run one episode and perturb the observation of the "adversary" agent
    state = env.reset()
    done = [False for agent_id in range(runner.args.N)]
    episode_reward = {agent_id: 0.0 for agent_id in range(runner.args.N)}

    iter_count = 0

    # initialize deque buffers for last batch_size observations
    result_deques = [deque(maxlen=5) for _ in range(runner.args.N)]
    metric_vals = []

    while not all(done):
        # Get actions from the agent (in evaluation mode, training=False)
        actions = []
        
        for id in range(runner.args.N):
            action, dist = runner.agent_n.select_action(state[id], id, evaluate=True, return_dist=True)
            actions.append(action)
        
        results = compute_taylor_error_policy(runner, state)
        for i in range(runner.args.N):
            result_deques[i].append(results[i])
        metric_vals.append([np.mean(result_deques[i]) for i in range(runner.args.N)])

        next_state, reward, done, info = env.step(actions)
        
        for agent_id in range(runner.args.N):
            episode_reward[agent_id] += reward[agent_id]
        
        state = next_state
        iter_count += 1
    
    # print("Episode finished. Rewards:", episode_reward, " Steps:", iter_count)
    return metric_vals


def save_matrix_to_files(matrix, agent_id, logdir, suffix=""):
    """
    Save matrix data to a CSV file for all timesteps.
    
    Args:
        matrix: List of timesteps, where each timestep contains metrics data including:
                mean, variance, std_dev, q1, q3, median, and MAD
        agent_id: ID of the agent
        logdir: Directory to save the file
        suffix: Optional suffix for the filename
    """
    filename = f"mappo_taylor_error_atk_free_agent_{suffix}.csv"
    filepath = os.path.join(logdir, filename)
    
    # Create header row
    header = ["agent", "timestep", "mean", "variance", "std_dev", "q1", "q3", "median", "mad", "diff_mean", "diff_std"]
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        for timestep, timestep_metrics in enumerate(matrix):
            row = [agent_id, timestep]
            for value in timestep_metrics:
                row.append(value)
            writer.writerow(row)
    
    print(f"Saved {len(matrix)} timestep matrices to {filepath}")


def main(runner: Runner_MAPPO_MPE, env, args):
    cwd = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(cwd, 'runs', f"{args.env_id}_{'discrete' if args.discrete_action else 'continuous'}", f"{timestamp}_ref_taylor_error")
    os.makedirs(logdir, exist_ok=True)
    total_episodes = 10000
    print(f"Logging directory: {logdir}")

    result_dataset = [{} for _ in range(runner.args.N)]
    for i in tqdm(range(total_episodes), desc="Processing episodes"):
        results = get_episode_data(env, runner)
        for timestep in range(len(results)):
            for agent_id in range(runner.args.N):
                if timestep not in result_dataset[agent_id]:
                    result_dataset[agent_id][timestep] = []
                result_dataset[agent_id][timestep].append(results[timestep][agent_id])
    
    for agent_id in range(runner.args.N):
        # Compute mean and variance for each agent pair across all episodes
        print(f"\n---- Agent {agent_id}:")
        metrics_mat = []
        sorted_timesteps = sorted(result_dataset[agent_id].keys())
        
        for i, timestep in enumerate(sorted_timesteps):
            print(f"\n ---- Timestep {timestep}:")
            timestep_values = result_dataset[agent_id][timestep]
            # Extract values for agent pair (i,j) across all episodes
            mean_val = np.mean(timestep_values)
            var_val = np.var(timestep_values)
            std_dev_val = np.std(timestep_values)
            q1, q3 = np.percentile(timestep_values, [25, 75])
            median_val = np.median(timestep_values)
            # Compute MAD (Median Absolute Deviation)
            mad_val = np.median(np.abs(np.array(timestep_values) - median_val))
            
            # Compute difference metrics (e_i - e_{i-1})
            if i == 0:  # For timestep 0, differences are 0
                diff_mean = 0.0
                diff_std = 0.0
            else:
                prev_timestep = sorted_timesteps[i-1]
                curr_metric_vals = result_dataset[agent_id][timestep]
                prev_metric_vals = result_dataset[agent_id][prev_timestep]
                differences = [curr - prev for curr, prev in zip(curr_metric_vals, prev_metric_vals)]
                diff_mean = np.mean(differences)
                diff_std = np.std(differences)
            
            print(f"Agent {agent_id}: mean = {mean_val:.4f}, variance = {var_val:.4f}, std_dev = {std_dev_val:.4f}, IQR = {q3 - q1:.4f}, median = {median_val:.4f}, MAD = {mad_val:.4f}, diff_mean = {diff_mean:.4f}, diff_std = {diff_std:.4f}")
            metrics = [mean_val, var_val, std_dev_val, q1, q3, median_val, mad_val, diff_mean, diff_std]
            metrics_mat.append(metrics)

        save_matrix_to_files(metrics_mat, agent_id, logdir, suffix=f"{agent_id}")
    
    env.close()


if __name__ == '__main__':    
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help="Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=25, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the rnn")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the mlp")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling. Here, we do not use it.")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=float, default=False, help="Whether to use relu, if False, we will use tanh")
    parser.add_argument("--use_rnn", type=bool, default=False, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=float, default=False, help="Whether to add agent_id. Here, we do not use it.")
    parser.add_argument("--use_value_clip", type=float, default=False, help="Whether to use value clip.")
    # Add output directory argument
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save all output files")
    parser.add_argument("--env_id", type=str, required=True, help="Environment ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--discrete_action", type=bool, default=True, help="Whether the action space is discrete or continuous")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory from load the trained model")

    args = parser.parse_args()
    env = make_env(env_name=args.env_id, discrete=True)
    runner = Runner_MAPPO_MPE(args, env_name=args.env_id, number=1, seed=args.seed)

    runner.agent_n.load_model_from_directory(args.model_dir)
    main(runner, env, args)
    # runner = Runner_MAPPO_MPE(args, env_name="simple_spread_v3", number=1, seed=0)
    # runner.run()
