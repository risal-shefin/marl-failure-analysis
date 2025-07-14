import argparse
import torch
import time
import imageio
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from utils.make_env import make_env
from algorithms.maddpg import MADDPG
import os
from datetime import datetime
from utils.pettingzoo_wrapper import PettingZooWrapper
from utils.misc import gumbel_softmax
import pettingzoo.mpe as mpe
import pettingzoo.sisl as sisl
import pettingzoo.atari as atari
import matplotlib.pyplot as plt
from PIL import Image
from collections import deque
import supersuit
import csv
import math
from tqdm import tqdm

USE_CUDA = torch.cuda.is_available()
DEVICE = 'gpu' if USE_CUDA else 'cpu'
torch_device = torch.device("cuda" if USE_CUDA else "cpu")

def preprocess_env_atari(env):
    # as per openai baseline's MaxAndSKip wrapper, maxes over the last 2 frames
    # to deal with frame flickering
    env = supersuit.max_observation_v0(env, 2)
    # skip frames for faster processing and less control
    # to be compatible with gym, use frame_skip(env, (2,5))
    env = supersuit.frame_skip_v0(env, 4)
    # downscale observation for faster processing
    env = supersuit.resize_v1(env, 84, 84)
    # allow agent to see everything on the screen despite Atari's flickering screen problem
    env = supersuit.frame_stack_v1(env, 4)
    return env


def fgsm_attack(maddpg, obs, actions, attacked_agent_id, epsilon):
    # Convert to tensors with gradient tracking
    torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    torch_actions = [Variable(torch.Tensor([actions[i]]).to(torch_device), requires_grad=False) for i in range(maddpg.nagents)]
    # Concatenate for critic input
    vf_in = torch.cat((*torch_obs, *torch_actions), dim=1)
    # Loss to maximize (degrade agent performance)
    loss = -maddpg.agents[attacked_agent_id].critic(vf_in).mean()  # Negative to maximize via gradient ascent
    # Compute gradient
    grad = torch.autograd.grad(loss, torch_obs[attacked_agent_id], retain_graph=True)[0]
    # FGSM perturbation: move in direction of gradient sign
    perturbation = epsilon * grad.sign()
    # Apply perturbation element-wise
    obs_perturbed = obs[attacked_agent_id] + perturbation.squeeze().cpu().numpy()
    return obs_perturbed


def so_inrd(maddpg, obs, actions, epsilon):
    torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    actions = [Variable(torch.Tensor([actions[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    vf_in = torch.cat((*torch_obs, *actions), dim=1)
    so_inrd_mat = [[] for _ in range(maddpg.nagents)]
    
    for i, agent_i in enumerate(maddpg.agents):
        policy_loss_i = -agent_i.critic(vf_in).mean() + (actions[i]**2).mean() * 1e-3

        for j, agent_j in enumerate(maddpg.agents):
            # The gradient with respect to obs
            grad_J = torch.autograd.grad(policy_loss_i, torch_obs[j], retain_graph=True)[0]

            # Compute η_i (adversarial perturbation direction)
            eta_i = epsilon * grad_J.sign() / torch.max(grad_J.norm(p=2), torch.tensor(1e-6))

            # Compute J tilde
            J_tilde = policy_loss_i + torch.dot(grad_J.flatten(), eta_i.flatten())

            # Perturbed state
            torch_obs_perturbed = [torch_obs[i].clone() for i in range(maddpg.nagents)]
            torch_obs_perturbed[j] = torch_obs[j] + eta_i
            vf_in_perturbed = torch.cat((*torch_obs_perturbed, *actions), dim=1)
            policy_loss_i_perturbed = -agent_i.critic(vf_in_perturbed).mean() + (actions[i]**2).mean() * 1e-3

            # Compute L
            L = policy_loss_i_perturbed - J_tilde
            so_inrd_mat[i].append(L.item())

    return so_inrd_mat


def compute_x_hessian_log_softmax(maddpg, obs, actions, action_spaces, epsilon):
    if not maddpg.discrete_action:
        raise NotImplementedError("This function is only implemented for discrete action spaces.")
    
    # Convert discrete actions to one-hot encoding
    one_hot_actions = []
    for i, action in enumerate(actions):
        one_hot = np.zeros(action_spaces[i].n)
        one_hot[action] = 1.0
        one_hot_actions.append(one_hot)
    actions = one_hot_actions

    torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    actions = [Variable(torch.Tensor([actions[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    vf_in = torch.cat((*torch_obs, *actions), dim=1)
    
    # Store eigenvalues for each agent pair (i, j)
    hess_mat = [[] for _ in range(maddpg.nagents)]
    
    for i, agent_i in enumerate(maddpg.agents):
        q_vals = []
        # Iterate through all possible actions for agent i
        for agent_action in range(action_spaces[i].n):
            if agent_action == actions[i].argmax():  # Use the original actions tensor for the actual action
                temp_actions = actions
            else:
                # Create a copy of the current actions
                temp_actions = [actions[k].clone() for k in range(maddpg.nagents)]
                # Convert agent_action to one-hot encoding
                one_hot_action = torch.zeros(action_spaces[i].n).to(torch_device)
                one_hot_action[agent_action] = 1.0
                temp_actions[i] = Variable(one_hot_action.unsqueeze(0), requires_grad=True)
            
            # Recompute vf_in with the new action
            vf_in_temp = torch.cat((*torch_obs, *temp_actions), dim=1)
            q_val = agent_i.critic(vf_in_temp).mean()
            q_vals.append(q_val)
            
        # Apply softmax to q_vals, then log, and store the value from index actions[i]
        q_vals_tensor = torch.stack(q_vals).squeeze()
        softmax_q_vals = torch.softmax(q_vals_tensor, dim=0)
        log_softmax_q_vals = torch.log(softmax_q_vals)
        log_pi = log_softmax_q_vals[actions[i].argmax()]    # actions[i] is one-hot encoded

        # Compute first-order gradient with respect to agent i's observation
        grad_i = torch.autograd.grad(log_pi, torch_obs[i], create_graph=True, retain_graph=True)[0]
        for j in range(maddpg.nagents):
            grad_j = torch.autograd.grad(grad_i.sum(), torch_obs[j], create_graph=True, retain_graph=True)[0]
            hess_mat[i].append(grad_j.sum().item())  # Store the Frobenius norm of the Hessian matrix

    return hess_mat


def compute_eigen(maddpg, obs, actions, action_spaces, epsilon):
    # if not maddpg.discrete_action:
    #     raise NotImplementedError("This function is only implemented for discrete action spaces.")
    
    # Convert discrete actions to one-hot encoding
    if maddpg.discrete_action:
        one_hot_actions = []
        for i, action in enumerate(actions):
            one_hot = np.zeros(action_spaces[i].n)
            one_hot[action] = 1.0
            one_hot_actions.append(one_hot)
        actions = one_hot_actions

    torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    actions = [Variable(torch.Tensor([actions[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    vf_in = torch.cat((*torch_obs, *actions), dim=1)
    
    # Store eigenvalues for each agent pair (i, j)
    eigen_mat = [[] for _ in range(maddpg.nagents)]
    
    for i, agent_i in enumerate(maddpg.agents):
        # q_vals = []
        # # Iterate through all possible actions for agent i
        # for agent_action in range(action_spaces[i].n):
        #     if agent_action == actions[i].argmax():  # Use the original actions tensor for the actual action
        #         temp_actions = actions
        #     else:
        #         # Create a copy of the current actions
        #         temp_actions = [actions[k].clone() for k in range(maddpg.nagents)]
        #         # Convert agent_action to one-hot encoding
        #         one_hot_action = torch.zeros(action_spaces[i].n).to(torch_device)
        #         one_hot_action[agent_action] = 1.0
        #         temp_actions[i] = Variable(one_hot_action.unsqueeze(0), requires_grad=True)
            
        #     # Recompute vf_in with the new action
        #     vf_in_temp = torch.cat((*torch_obs, *temp_actions), dim=1)
        #     q_val = agent_i.critic(vf_in_temp).mean()
        #     q_vals.append(q_val)
            
        # # Apply softmax to q_vals, then log, and store the value from index actions[i]
        # q_vals_tensor = torch.stack(q_vals).squeeze()
        # softmax_q_vals = torch.softmax(q_vals_tensor, dim=0)
        # log_softmax_q_vals = torch.log(softmax_q_vals)
        # log_pi = log_softmax_q_vals[actions[i].argmax()]    # actions[i] is one-hot encoded

        # Compute first-order gradient with respect to agent i's observation
        # grad_i = torch.autograd.grad(log_pi, torch_obs[i], create_graph=True, retain_graph=True)[0]

        critic_val = agent_i.critic(vf_in).mean()
        grad_i = torch.autograd.grad(critic_val, torch_obs[i], create_graph=True, retain_graph=True)[0]

        # so-inrd
        # eta_i = 0.1 * grad_i.sign() / torch.max(grad_i.norm(p=2), torch.tensor(1e-6))
        # j_tilde = critic_val + torch.dot(grad_i.flatten(), eta_i.flatten())
        # p_torch_obs = [torch_obs[k].clone() for k in range(maddpg.nagents)]
        # p_torch_obs[i] = torch_obs[i] + eta_i
        # j_perturbed = agent_i.critic(torch.cat((*p_torch_obs, *actions), dim=1)).mean()

        for j in range(maddpg.nagents):
            # Compute cross-agent Hessian matrix for agent pair (i, j)
            # This represents ∂²V/∂obs_i∂obs_j
            hessian_matrix = []

            # eigen_mat[i].append((j_perturbed - j_tilde).item())
            # continue
            
            for k in range(grad_i.shape[1]):  # For each dimension of agent i's observation (has shape [1, obs_dim])
                # Compute ∂²V/∂obs_i[k]∂obs_j
                second_grad = torch.autograd.grad(
                    grad_i[0, k], 
                    torch_obs[j], 
                    retain_graph=True, 
                    allow_unused=True
                )[0]
                
                hessian_matrix.append(second_grad.flatten())

            # Convert to tensor and compute eigenvalues
            H = torch.stack(hessian_matrix)

            # Frob norm
            eigen_mat[i].append(H.norm(p='fro').item())
            continue
    
            assert H.shape[0] == H.shape[1], "Hessian matrix must be square."

            # Make symmetric by averaging H and H^T for numerical stability
            H_symmetric = (H + H.T) / 2
            
            # Compute eigenvalues
            eigenvals = torch.linalg.eigvals(H_symmetric)
            
            # Get the most negative eigenvalue (real part)
            if torch.is_complex(eigenvals):
                eigenvals_real = eigenvals.real
            else:
                eigenvals_real = eigenvals
            
            min_eigenval = torch.min(eigenvals_real).item() # most negative eigenvalue
            eigen_mat[i].append(min_eigenval)

    return eigen_mat


def get_episode_data(env, maddpg, config, logdir, do_attack=False, atk_agent_id=-1, seed=None):
    # obs = env.reset()
    obs = env.reset(seed=seed) if seed else env.reset()
    # obs = env.reset(seed=12345) # better for speaker_listener_v3
    episode_reward = 0
    frames = []
    # initialize deque buffers for last batch_size observations
    result_deques = [[deque(maxlen=5) for _ in range(maddpg.nagents)] for _ in range(maddpg.nagents)]
    metric_vals = []
    cnt = 0

    while True:
        # add Gaussian noise to an agent's observation
        # noise_scale = 0.0  # adjust the standard deviation of the noise as needed
        # obs[attacked_agent] = obs[attacked_agent] + np.random.randn(*obs[attacked_agent].shape) * noise_scale

        # FGSM attack
        if do_attack and False:
            temp_torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=False) for i in range(maddpg.nagents)]
            temp_torch_agent_actions = maddpg.step(temp_torch_obs, explore=False)
            agent_actions = [ac.data.cpu().numpy() for ac in temp_torch_agent_actions]
            temp_actions = [agent_actions[i].squeeze() for i, agent_name in enumerate(env.possible_agents)]
            obs[atk_agent_id] = fgsm_attack(maddpg, obs, temp_actions, atk_agent_id, 3.0)
        
        torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=False) for i in range(maddpg.nagents)]
        torch_agent_actions = maddpg.step(torch_obs, explore=False)
        agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions]
        if maddpg.discrete_action:
            actions = {agent_name: agent_actions[i].argmax() for i, agent_name in enumerate(env.possible_agents)}
        else:
            actions = {agent_name: agent_actions[i].squeeze() for i, agent_name in enumerate(env.possible_agents)}
        action_logits = maddpg.get_action_logits(torch_obs)

        # random attack
        if do_attack and False:
            actions[env.possible_agents[atk_agent_id]] = env.action_spaces[env.possible_agents[atk_agent_id]].sample()
        
        # worst action attack
        # Compute entropy of action logits
        atk_agent_action_probs = torch.softmax(action_logits[atk_agent_id].squeeze(), dim=0)
        atk_agent_log_probs = torch.log(atk_agent_action_probs)
        atk_agent_entropy = -torch.sum(atk_agent_action_probs * atk_agent_log_probs)
        # if do_attack and np.random.rand() < 0.75:
        # if do_attack and atk_agent_entropy < 0.1:
        if do_attack and cnt >= config.atk_start_step and cnt <= config.atk_end_step:
            assert maddpg.discrete_action, "Worst action attack is only implemented for discrete action spaces."
            print(" >> attacked ")
            actions[env.possible_agents[atk_agent_id]] = torch.argmin(action_logits[atk_agent_id]).item()

        if config.save_gifs:
            frames.append(Image.fromarray(env.render()))
        
        # result_mat = so_inrd(maddpg, obs, list(actions.values()), 0.1)
        result_mat = compute_eigen(maddpg, obs, list(actions.values()), env.action_space, 0.1)
        # result_mat = compute_x_hessian_log_softmax(maddpg, obs, list(actions.values()), env.action_space, 0.1)
        for i in range(maddpg.nagents):
            for j in range(maddpg.nagents):
                result_deques[i][j].append(result_mat[i][j])
        metric_vals.append([[np.mean(result_deques[i][j]) for j in range(maddpg.nagents)] for i in range(maddpg.nagents)])

        next_obs, rewards, dones, infos = env.step(actions)
        episode_reward += np.sum([rewards[:,i] if env.agent_types[i] != 'adversary' else np.zeros_like(rewards[:,i]) for i in range(maddpg.nagents)])

        obs = next_obs
        cnt += 1
        if dones.all():
            break

    print(f"Episode reward: {episode_reward}")
    if config.save_gifs:
        imageio.mimsave(os.path.join(logdir, f'{config.env_id}_episode_atk_{atk_agent_id if do_attack else "free"}.gif'), frames, duration=125)
        print(f"Saved gif of episode to {logdir}")

    return metric_vals


def plot_results(results, results_attacked, atk_agent_id, logdir):
    n = len(results[0])  # number of agents
    t = len(results)     # number of time steps
    
    # Create n x n subplots
    fig, axes = plt.subplots(n, n, figsize=(4*n, 4*n))
    fig.suptitle(f'Most Negative Eigen (Worst Action Attack | Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.95)
    
    # Ensure axes is 2D even for single agent case
    if n == 1:
        axes = [[axes]]
    elif n == 2:
        axes = axes.reshape(n, n)
    
    for i in range(n):
        for j in range(n):
            ax = axes[i][j]
            
            # Extract time series for agent i's metric w.r.t agent j
            normal_series = [results[t][i][j] for t in range(len(results))]
            attacked_series = [results_attacked[t][i][j] for t in range(len(results_attacked))]
            
            # Plot the curves
            steps_normal = range(len(normal_series))
            steps_attacked = range(len(attacked_series))
            ax.plot(steps_normal, normal_series, 'b-', label='Normal', linewidth=2)
            ax.plot(steps_attacked, attacked_series, 'r-', label='Attacked', linewidth=2)
            
            ax.set_xlabel('Step')
            ax.set_ylabel('∂²Q/∂obs_i∂obs_j')
            ax.set_title(f'agent_{i} , agent_{j}')
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
        matrix: List of timesteps, where each timestep contains n_agent x n_agent data
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
        for j in range(total_agents):
            header.append(f"agent_{i}_{j}")
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        for timestep, timestep_data in enumerate(matrix):
            row = [timestep, attacked_agent_id]
            for i in range(total_agents):
                for j in range(total_agents):
                    row.append(timestep_data[i][j])
            writer.writerow(row)
    
    print(f"Saved {len(matrix)} timestep matrices to {filepath}")


def run(config):
    maddpg = MADDPG.init_from_save(config.model_path)

    # create a log directory under runs/<env_id>/<timestamp> using os and getcwd
    cwd = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(cwd, 'runs', f"{config.env_id}_{'discrete' if maddpg.discrete_action else 'continuous'}", timestamp)
    os.makedirs(logdir, exist_ok=True)

    try:
        env_func = getattr(mpe, config.env_id)
        env = env_func.parallel_env(continuous_actions= not maddpg.discrete_action, render_mode='rgb_array')
    except:
        try:
            env_func = getattr(sisl, config.env_id)
            env = env_func.parallel_env(n_pursuers=5, render_mode='rgb_array') if config.env_id == 'waterworld_v4' else env_func.parallel_env(render_mode='rgb_array')
        except:
            env_func = getattr(atari, config.env_id)
            env = env_func.parallel_env(render_mode='rgb_array')
            env = preprocess_env_atari(env)

    env = PettingZooWrapper.wrap_env(env)
    env.reset()

    # maddpg.prep_rollouts(device=DEVICE)
    maddpg.prep_training(device=DEVICE)


    # result_dataset = {}
    # for i in tqdm(range(500), desc="Processing episodes"):
    #     results = get_episode_data(env, maddpg, config, logdir)
    #     for timestep in range(len(results)):
    #         if timestep not in result_dataset:
    #             result_dataset[timestep] = []
    #         result_dataset[timestep].append(results[timestep])
    
    # for timestep, timestep_data in result_dataset.items():
    #     save_matrix_to_files(timestep_data, None, maddpg.nagents, logdir, suffix=f"_{timestep}")

    #     # Compute mean and variance for each agent pair across all episodes
    #     print(f"\n ---- Timestep {timestep}:")
    #     for i in range(maddpg.nagents):
    #         for j in range(maddpg.nagents):
    #             # Extract values for agent pair (i,j) across all episodes
    #             values = [agent_ij_history[i][j] for agent_ij_history in timestep_data]
    #             mean_val = np.mean(values)
    #             var_val = np.var(values)
    #             print(f"agent pair ({i}, {j}): mean = {mean_val:.4f}, variance = {var_val:.4f}")
                
    # exit()

    attacked_agent_id = config.attack_agent_id  # specify the agent to attack
    seed = 42

    results = get_episode_data(env, maddpg, config, logdir, seed=seed)
    save_matrix_to_files(results, None, maddpg.nagents, logdir)

    results_attacked = get_episode_data(env, maddpg, config, logdir, do_attack=True, atk_agent_id=attacked_agent_id, seed=seed)
    save_matrix_to_files(results_attacked, attacked_agent_id, maddpg.nagents, logdir)

    plot_results(results, results_attacked, attacked_agent_id, logdir)
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_path",
                        help="model directory")
    parser.add_argument("--save_gifs", action="store_true",
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--attack_agent_id", type=int, default=0,)
    parser.add_argument("--atk_start_step", type=int, default=-math.inf)
    parser.add_argument("--atk_end_step", type=int, default=math.inf)

    config = parser.parse_args()

    run(config)
