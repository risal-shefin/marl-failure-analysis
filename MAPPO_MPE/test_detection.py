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

from MAPPO_MPE_main import Runner_MAPPO_MPE


def perturb_random_noise(states, perturb_agent_id, noise_std=0.1):
    perturbed_states = states.copy()
    perturbed_states[perturb_agent_id] = states[perturb_agent_id] + noise_std * torch.randn_like(states[perturb_agent_id])
    return perturbed_states

def perturb_fgsm(states, perturb_agent_id, perturb_eps=0.1):
    p_states_tensor = torch.cat([torch.tensor(state, dtype=torch.float32, requires_grad=True) for state in states], dim=0)
    # Compute the gradient of the value function with respect to the observation
    values = runner.agent_n.compute_value(p_states_tensor).squeeze(-1)  # shape: (N,)
    # Compute the gradient for the perturbed agent. Taking negative of the value since fgsm is a maximization attack
    grad = torch.autograd.grad(-values[perturb_agent_id], p_states_tensor, create_graph=True)[0]
    # Perturb the observation in the direction of the gradient
    agent_state_dim = len(states[perturb_agent_id])
    return states[perturb_agent_id] + perturb_eps * np.sign(grad[perturb_agent_id*agent_state_dim : (perturb_agent_id+1)*agent_state_dim].detach().cpu().numpy())


def compute_taylor_policy(runner: Runner_MAPPO_MPE, states):
    states_tensor = torch.stack([torch.tensor(state, dtype=torch.float32, requires_grad=True) for state in states])

    delta_errors = []

    for i in range(runner.args.N):
        obs = states_tensor[i].unsqueeze(0)  # shape: (1, obs_dim)
        action, dist = runner.agent_n.compute_action(obs, i, evaluate=True, return_dist=True)
        target_val = dist.log_prob(action)
        grad_i = torch.autograd.grad(target_val, obs, create_graph=True, retain_graph=True)[0]
        # grad_i = torch.autograd.grad(critic_val, actions[i], create_graph=True, retain_graph=True)[0]

        eta_i = 0.01 * grad_i.sign() / torch.max(grad_i.norm(p=2), torch.tensor(1e-6))
        
        # Second-order Taylor expansion using Hessian-vector product (HVP)
        # Instead of computing full Hessian, compute H * eta_i directly
        # hvp = torch.autograd.grad(
        #     outputs=grad_i.flatten(), 
        #     inputs=torch_obs[i], 
        #     grad_outputs=eta_i.flatten(),
        #     retain_graph=True
        # )[0]
        
        # Second-order Taylor approximation: f(x + η) ≈ f(x) + ∇f(x)^T η + 0.5 η^T H η
        j_tilde = target_val + torch.dot(grad_i.flatten(), eta_i.flatten())# + 0.5 * torch.dot(eta_i.flatten(), hvp.flatten())
        p_state = obs + eta_i
        p_action, p_dist = runner.agent_n.compute_action(p_state, i, evaluate=True, return_dist=True)
        j_perturbed = p_dist.log_prob(p_action)
        delta_error = abs(j_perturbed - j_tilde).item()
        delta_errors.append(delta_error)

    return delta_errors

def compute_eigen_policy(runner: Runner_MAPPO_MPE, states):
    states_tensor = torch.stack([torch.tensor(state, dtype=torch.float32, requires_grad=True) for state in states])

    results = []

    for i in range(runner.args.N):
        obs = states_tensor[i].unsqueeze(0)  # shape: (1, obs_dim)
        action, dist = runner.agent_n.compute_action(obs, i, evaluate=True, return_dist=True)
        target_val = dist.log_prob(action)
        grad_i = torch.autograd.grad(target_val, obs, create_graph=True, retain_graph=True)[0]

        # Compute Hessian matrix
        hessian_flat = []
        grad_i_flat = grad_i.flatten()
        for j in range(grad_i.numel()):
            grad2 = torch.autograd.grad(
                outputs=grad_i_flat[j], 
                inputs=obs, 
                retain_graph=True,
                create_graph=False
            )[0]
            hessian_flat.append(grad2.flatten())
        
        hessian = torch.stack(hessian_flat)

        # # Compute Frobenius norm of Hessian
        # hessian_frob_norm = torch.norm(hessian, p='fro')
        # results.append(hessian_frob_norm)
        # continue
        
        # # Compute eigenvalues
        eigenvals = torch.linalg.eigvals(hessian)
        eigenval = torch.min(eigenvals.real).item() 
        results.append(eigenval)
        continue

    return results


def get_episode_data(env, runner: Runner_MAPPO_MPE, do_attack: bool, attacked_agent_id: str):

    # Run one episode and perturb the observation of the "adversary" agent
    state = env.reset(seed=runner.seed)
    done = [False for agent_id in range(runner.args.N)]
    episode_reward = {agent_id: 0.0 for agent_id in range(runner.args.N)}

    iter_count = 0
    frames = []  # List to collect frames

    # initialize deque buffers for last batch_size observations
    result_deques = [deque(maxlen=5) for _ in range(runner.args.N)]
    metric_vals = []

    while not all(done):
        # Get actions from the agent (in evaluation mode, training=False)
        actions = []
        
        for id in range(runner.args.N):
            # fgsm
            # if do_attack and id == attacked_agent_id and np.random.rand() < args.attack_rate:
            #     state[id] = perturb_fgsm(state, id, args.perturb_eps)

            action, dist = runner.agent_n.select_action(state[id], id, evaluate=True, return_dist=True)
            # action space attack
            # if do_attack and id == attacked_agent_id and np.random.rand() < args.attack_rate:
            # if do_attack and id == attacked_agent_id and dist.entropy() < 0.5:
            if do_attack and id == attacked_agent_id and runner.args.atk_step_start <= iter_count <= runner.args.atk_step_end:
                # # random action
                # action = env.action_space[attacked_agent_id].sample()
                # worst action attack
                action = torch.argmin(dist.probs).item()
                print(" >> attacked")
            
            
            actions.append(action)

        
        results = compute_taylor_policy(runner, state)
        # results = compute_eigen_policy(runner, state)
        for i in range(runner.args.N):
            result_deques[i].append(results[i])
        metric_vals.append([np.mean(result_deques[i]) for i in range(runner.args.N)])

        next_state, reward, done, info = env.step(actions)
        
        for agent_id in range(runner.args.N):
            episode_reward[agent_id] += reward[agent_id]
        
        state = next_state
        iter_count += 1
    
    print("Episode finished. Rewards:", episode_reward, " Steps:", iter_count)
    return metric_vals


def plot_results(results, results_attacked, atk_agent_id, logdir):
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

def save_matrix_to_files(matrix, attacked_agent_id, total_agents, logdir, suffix=""):
    """
    Save matrix data to a CSV file for all timesteps.
    
    Args:
        matrix: List of timesteps, where each timestep contains n_agent data
        attacked_agent_id: ID of the attacked agent
        total_agents: Total number of agents
        logdir: Directory to save the file
    """
    if attacked_agent_id is None:
        filename = f"maddpg_h_data_atk_free{suffix}.csv"
    else:
        filename = f"maddpg_h_data_atk_{attacked_agent_id}{suffix}.csv"
    filepath = os.path.join(logdir, filename)
    
    # Create header row
    # header = ["timestep", "attacked_agent"]
    header = ["num", "attacked_agent"]
    for i in range(total_agents):
        header.append(f"agent_{i}")
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        for timestep, timestep_data in enumerate(matrix):
            row = [timestep, attacked_agent_id]
            for i in range(total_agents):
                row.append(timestep_data[i])
            writer.writerow(row)
    
    print(f"Saved {len(matrix)} timestep matrices to {filepath}")


def main(runner: Runner_MAPPO_MPE, env, args):
    attacked_agent_id = args.attack_agent_id
    cwd = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(cwd, 'runs', f"{args.env_id}_{'discrete' if args.discrete_action else 'continuous'}", timestamp)
    os.makedirs(logdir, exist_ok=True)
    print(f"Logging directory: {logdir}")
    
    episode_data_unattacked = get_episode_data(env, runner, False, None)
    save_matrix_to_files(episode_data_unattacked, None, runner.args.N, logdir)

    episode_data_attacked = get_episode_data(env, runner, True, attacked_agent_id)
    save_matrix_to_files(episode_data_attacked, attacked_agent_id, runner.args.N, logdir)
    
    env.close()

    # Plot the SO-INRD values for all agents
    plot_results(episode_data_unattacked, episode_data_attacked, attacked_agent_id, logdir)


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
    parser.add_argument("--attack_rate", type=float, default=0.5, help="Attack probability when attacking (0.0-1.0)")
    parser.add_argument("--perturb_eps", type=float, default=0.1, help="Perturbation epsilon value for attacks")
    parser.add_argument("--attack_agent_id", type=int, default=0, help="Whether to add agent_id. Here, we do not use it.")
    # Add output directory argument
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save all output files")
    parser.add_argument("--env_id", type=str, default="simple_spread_v3", help="Environment ID")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--discrete_action", type=bool, default=True, help="Whether the action space is discrete or continuous")
    parser.add_argument("--atk_step_start", type=int, default=-math.inf, help="Attack start step")
    parser.add_argument("--atk_step_end", type=int, default=math.inf, help="Attack end step")

    args = parser.parse_args()
    env = make_env(env_name=args.env_id, discrete=True)
    runner = Runner_MAPPO_MPE(args, env_name=args.env_id, number=1, seed=args.seed)
    
    runner.agent_n.load_model_from_directory("/deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/MAPPO_MPE/model/MAPPO_actor_env_simple_spread_number_1_seed_0_step_1215k.pth")
    main(runner, env, args)
    # runner = Runner_MAPPO_MPE(args, env_name="simple_spread_v3", number=1, seed=0)
    # runner.run()
