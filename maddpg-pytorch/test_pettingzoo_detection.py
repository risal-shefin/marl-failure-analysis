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
    loss = -(maddpg.agents[attacked_agent_id].critic(vf_in)).mean()  # Negative to maximize via gradient ascent
    # Compute gradient
    grad = torch.autograd.grad(loss, torch_obs[attacked_agent_id], retain_graph=True)[0]
    # FGSM perturbation: move in direction of gradient sign
    perturbation = epsilon * grad.sign()
    # Apply perturbation element-wise
    obs_perturbed = obs[attacked_agent_id] + perturbation.squeeze().cpu().numpy()
    return obs_perturbed


def compute_taylor_delta_policy(maddpg, obs, actions, action_spaces, epsilon):
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

    delta_errors = []

    for i, agent_i in enumerate(maddpg.agents):
        action_logits_i = agent_i.policy(torch_obs[i])
        action_log_probs = torch.log_softmax(action_logits_i, dim=-1)
        max_action_idx = torch.argmax(action_log_probs, dim=-1)
        critic_val = action_log_probs.gather(-1, max_action_idx.unsqueeze(-1)).squeeze()
        grad_i = torch.autograd.grad(critic_val, torch_obs[i], create_graph=True, retain_graph=True)[0]
        # grad_i = torch.autograd.grad(critic_val, actions[i], create_graph=True, retain_graph=True)[0]

        eta_i = epsilon * grad_i.sign() / torch.max(grad_i.norm(p=2), torch.tensor(1e-6))
        
        # Second-order Taylor approximation: f(x + η) ≈ f(x) + ∇f(x)^T η + 0.5 η^T H η
        j_tilde = critic_val + torch.dot(grad_i.flatten(), eta_i.flatten())# + 0.5 * torch.dot(eta_i.flatten(), hvp.flatten())
        p_torch_obs_i = torch_obs[i] + eta_i
        p_action_logits_i = agent_i.policy(p_torch_obs_i)
        p_action_log_probs = torch.log_softmax(p_action_logits_i, dim=-1)
        p_max_action_idx = torch.argmax(p_action_log_probs, dim=-1)
        j_perturbed = p_action_log_probs.gather(-1, p_max_action_idx.unsqueeze(-1)).squeeze()
        delta_error = abs(j_perturbed - j_tilde).item()
        delta_errors.append(delta_error)

    return delta_errors

def compute_frob_norms(maddpg, obs, actions, action_spaces, vulnerable_agent_id):
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
    
    results = []

    for i, agent_i in enumerate(maddpg.agents):
        critic_val = agent_i.critic(vf_in).mean()
        grad_i = torch.autograd.grad(critic_val, torch_obs[i], create_graph=True, retain_graph=True)[0]

        # Compute Hessian matrix
        hessian_matrix = []
        for k in range(grad_i.shape[1]):
            # Compute ∂²Q/∂obs_i[k]∂obs_j
            second_grad = torch.autograd.grad(
                grad_i[0, k], 
                torch_obs[vulnerable_agent_id], 
                retain_graph=True, 
                allow_unused=True
            )[0]
            
            hessian_matrix.append(second_grad.flatten())

        H = torch.stack(hessian_matrix)
        hessian_frob_norm = torch.norm(H, p='fro')
        results.append(hessian_frob_norm.item())

    return results


# second order directional derivative
def compute_2nd_ord_dir_derivatives(maddpg, obs, actions, action_spaces, vulnerable_agent_id):
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
    
    results = []

    for i, agent_i in enumerate(maddpg.agents):
        critic_val = agent_i.critic(vf_in).mean()
        grad_i = torch.autograd.grad(critic_val, torch_obs[i], create_graph=True, retain_graph=True)[0]
        v = grad_i / torch.max(grad_i.norm(p=2), torch.tensor(1e-6))

        # Compute Hessian-vector product (HVP) of grad_i and v with respect to torch_obs[j]
        hvp = torch.autograd.grad(
            outputs=grad_i,
            inputs=torch_obs[vulnerable_agent_id],
            grad_outputs=v,
            retain_graph=True,
            allow_unused=True
        )[0]

        # Compute u^T * H * v (quadratic form)
        grad_j = torch.autograd.grad(-critic_val, torch_obs[vulnerable_agent_id], create_graph=True, retain_graph=True)[0]
        u = grad_j / torch.max(grad_j.norm(p=2), torch.tensor(1e-6))
        curvature_val = torch.dot(u.flatten(), hvp.flatten())
        results.append(curvature_val.item())

    return results


def get_episode_data(env, maddpg, config, logdir, ref_means, ref_std_devs, do_attack=False, atk_agent_id=-1, seed=None):
    # obs = env.reset()
    obs = env.reset(seed=seed) if seed else env.reset()
    # obs = env.reset(seed=12345) # better for speaker_listener_v3
    episode_reward = 0
    episode_rewards = [0 for _ in range(maddpg.nagents)]
    frames = []
    # initialize deque buffers for last batch_size observations
    result_deques = [deque(maxlen=5) for _ in range(maddpg.nagents)]
    frob_norms_deques = [deque(maxlen=1) for _ in range(maddpg.nagents)]
    sec_dir_derivatives_deques = [deque(maxlen=1) for _ in range(maddpg.nagents)]
    metric_vals = []
    cnt = 0
    vulnerable_agent_id = None
    attacked_steps = []
    frob_norms_list = []
    sec_dir_derivatives = []
    do_start_attack = False
    attack_step_remaining = 10

    while True:
        # add Gaussian noise to an agent's observation
        # noise_scale = 0.0  # adjust the standard deviation of the noise as needed
        # obs[attacked_agent] = obs[attacked_agent] + np.random.randn(*obs[attacked_agent].shape) * noise_scale

        # FGSM attack
        if do_attack and cnt >= config.atk_start_step and cnt <= config.atk_end_step and False:
            temp_torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=False) for i in range(maddpg.nagents)]
            temp_torch_agent_actions = maddpg.step(temp_torch_obs, explore=False)
            agent_actions = [ac.data.cpu().numpy() for ac in temp_torch_agent_actions]
            temp_actions = [agent_actions[i].squeeze() for i, agent_name in enumerate(env.possible_agents)]
            obs[atk_agent_id] = fgsm_attack(maddpg, obs, temp_actions, atk_agent_id, 0.1)
        
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
        if do_attack and atk_agent_entropy < 0.05:
            do_start_attack = True
        # if do_attack and cnt >= config.atk_start_step and cnt <= config.atk_end_step:
        if do_start_attack and attack_step_remaining > 0:
            assert maddpg.discrete_action, "Worst action attack is only implemented for discrete action spaces."
            actions[env.possible_agents[atk_agent_id]] = torch.argmin(action_logits[atk_agent_id]).item()
            attacked_steps.append(cnt)
            attack_step_remaining -= 1

        if config.save_gifs:
            frames.append(Image.fromarray(env.render()))
        
        # results = compute_taylor_delta(maddpg, obs, list(actions.values()), env.action_space, 0.1)
        results = compute_taylor_delta_policy(maddpg, obs, list(actions.values()), env.action_space, 0.01)
        # results = compute_eigen(maddpg, obs, list(actions.values()), env.action_space, 0.1)
        results_frob_norms = compute_frob_norms(maddpg, obs, list(actions.values()), env.action_space, atk_agent_id)
        results_sec_dir_derivatives = compute_2nd_ord_dir_derivatives(maddpg, obs, list(actions.values()), env.action_space, atk_agent_id)

        for i in range(maddpg.nagents):
            result_deques[i].append(results[i])
            taylor_approx_error = np.mean(result_deques[i])
            if not (ref_means[i][cnt] - ref_std_devs[i][cnt] <= taylor_approx_error <= ref_means[i][cnt] + ref_std_devs[i][cnt]) and vulnerable_agent_id is None:
                print(f" [!!!] Anomaly detected for agent {i} at timestep: {cnt}. Taylor Appx. Error: {taylor_approx_error}")
                vulnerable_agent_id = i
            frob_norms_deques[i].append(results_frob_norms[i])
            sec_dir_derivatives_deques[i].append(results_sec_dir_derivatives[i])

        metric_vals.append([np.mean(result_deques[i]) for i in range(maddpg.nagents)])
        frob_norms_list.append([np.mean(frob_norms_deques[i]) for i in range(maddpg.nagents)])
        sec_dir_derivatives.append([np.mean(sec_dir_derivatives_deques[i]) for i in range(maddpg.nagents)])

        if vulnerable_agent_id is not None:
            print(f"\n --- Timestep: {cnt} ------")
            frob_norms = frob_norms_list[-1]
            # Rank frob_norm values from largest to smallest
            frob_norms_with_indices = [(i, frob_norms[i]) for i in range(len(frob_norms)) if i != vulnerable_agent_id]
            frob_norms_ranked = sorted(frob_norms_with_indices, key=lambda x: x[1], reverse=True)
            print(f" --- Frob norm ranking: {frob_norms_ranked}")

            s_dir_derivatives = sec_dir_derivatives[-1]

            for i in range(maddpg.nagents):
                if i == vulnerable_agent_id or s_dir_derivatives[i] >= 0.0:
                    continue
                print(f" >> Potential Vulnerable Position for agent {i} wrt faulty agent. 2nd Dir Derivative: {s_dir_derivatives[i]}")

        next_obs, rewards, dones, infos = env.step(actions)
        episode_reward += np.sum([rewards[:,i] if env.agent_types[i] != 'adversary' else np.zeros_like(rewards[:,i]) for i in range(maddpg.nagents)])
        episode_rewards = [episode_rewards[i] + rewards[:,i].sum() for i in range(maddpg.nagents)]

        obs = next_obs
        cnt += 1
        if dones.all():
            break

    print(f"Episode reward: {episode_reward}")
    print(f"Episode rewards: {episode_rewards}")
    if config.save_gifs:
        imageio.mimsave(os.path.join(logdir, f'{config.env_id}_episode_atk_{atk_agent_id if do_attack else "free"}.gif'), frames, duration=125)
        print(f"Saved gif of episode to {logdir}")

    return metric_vals, attacked_steps, frob_norms_list, sec_dir_derivatives


def plot_results(results_attacked, attacked_steps, atk_agent_id, ref_means, ref_std_devs, logdir):
    n = len(results_attacked[0])  # number of agents
    t = len(results_attacked)     # number of time steps
    
    # Create n subplots in a row
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    fig.suptitle(f'Taylor Error (Worst Action Attack | Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.95)
    
    # Ensure axes is iterable even for single agent case
    if n == 1:
        axes = [axes]
    
    for i in range(n):
        ax = axes[i]
        
        # Extract time series for agent i
        attacked_series = [results_attacked[t][i] for t in range(len(results_attacked))]
        
        # Plot the curves
        steps = range(len(attacked_series))
        
        # Add green region using ref_means and ref_std_devs
        ref_steps = range(len(ref_means[i]))
        ref_lower = [ref_means[i][t] - ref_std_devs[i][t] for t in range(len(ref_means[i]))]
        ref_upper = [ref_means[i][t] + ref_std_devs[i][t] for t in range(len(ref_means[i]))]
        ax.fill_between(steps, ref_lower, ref_upper, alpha=0.1, color='green')
        
        ax.plot(steps, attacked_series, 'r-', label='Under Attack', linewidth=2)
        ax.plot(steps, ref_means[i], 'g--', label='Normal (Mean)', linewidth=2)
        
        # Mark attacked timesteps with vertical lines
        if i == atk_agent_id and attacked_steps:
            for attack_step in attacked_steps:
                ax.axvline(x=attack_step, color='orange', linestyle=':', alpha=0.5, linewidth=1.5)
            # Add legend entry for attack markers
            ax.axvline(x=attacked_steps[0], color='orange', linestyle=':', alpha=0.5, linewidth=1.5, label='Attacked Steps')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Taylor Delta Error')
        ax.set_title(f'Agent {i}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(logdir, f'plot_analysis_attacked_{atk_agent_id}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved analysis plot to {logdir}")


def plot_frobs(frobs_normal, frobs_atk, attacked_steps, atk_agent_id, logdir):
    n = len(frobs_normal[0])  # number of agents
    t = len(frobs_normal)     # number of time steps
    
    # Create n subplots in a row
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    fig.suptitle(f'Frobenius Norms (Worst Action Attack | Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.95)
    
    # Ensure axes is iterable even for single agent case
    if n == 1:
        axes = [axes]
    
    for i in range(n):
        ax = axes[i]
        
        # Extract time series for agent i
        normal_series = [frobs_normal[t][i] for t in range(len(frobs_normal))]
        attacked_series = [frobs_atk[t][i] for t in range(len(frobs_atk))]
        
        # Plot the curves
        steps = range(len(normal_series))
        ax.plot(steps, attacked_series, 'r-', label='Under Attack', linewidth=2)
        ax.plot(steps, normal_series, 'g-', label='Normal', linewidth=2)
        
        # Mark attacked timesteps with vertical lines
        if attacked_steps:
            for attack_step in attacked_steps:
                ax.axvline(x=attack_step, color='orange', linestyle=':', alpha=0.5, linewidth=1.5)
            # Add legend entry for attack markers
            ax.axvline(x=attacked_steps[0], color='orange', linestyle=':', alpha=0.5, linewidth=1.5, label='Attacked Steps')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Frobenius Norm')
        ax.set_title(f'Agent {i}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(logdir, f'plot_frobs_attacked_{atk_agent_id}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved frobenius norms plot to {logdir}")


def plot_sec_dir_derivatives(s_dir_derv_normal, s_dir_derv_atk, attacked_steps, atk_agent_id, logdir):
    n = len(s_dir_derv_normal[0])  # number of agents
    t = len(s_dir_derv_normal)     # number of time steps
    
    # Create n subplots in a row
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    fig.suptitle(f'2nd Ord. Dir. Derivatives (Worst Action Attack | Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.95)
    
    # Ensure axes is iterable even for single agent case
    if n == 1:
        axes = [axes]
    
    for i in range(n):
        ax = axes[i]
        
        # Extract time series for agent i
        normal_series = [s_dir_derv_normal[t][i] for t in range(len(s_dir_derv_normal))]
        attacked_series = [s_dir_derv_atk[t][i] for t in range(len(s_dir_derv_atk))]
        
        # Plot the curves
        steps = range(len(normal_series))
        # ax.plot(steps, normal_series, 'g-', label='Normal', linewidth=2)
        ax.plot(steps, attacked_series, 'r-', label='Under Attack', linewidth=2)
        
        # Highlight region under y < 0 in red
        y_min = min(min(normal_series), min(attacked_series))
        if y_min < 0:
            ax.axhspan(y_min * 1.1, 0, alpha=0.2, color='red')
        
        # Mark attacked timesteps with vertical lines
        if attacked_steps:
            for attack_step in attacked_steps:
                ax.axvline(x=attack_step, color='orange', linestyle=':', alpha=0.5, linewidth=1.5)
            # Add legend entry for attack markers
            ax.axvline(x=attacked_steps[0], color='orange', linestyle=':', alpha=0.5, linewidth=1.5, label='Attacked Steps')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('2nd Ord. Dir. Derivative')
        ax.set_title(f'Agent {i}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(logdir, f'plot_sec_dir_derivatives_attacked_{atk_agent_id}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved 2nd ord. dir. derivatives plot to {logdir}")


def save_matrix_to_files(matrix, attacked_steps, attacked_agent_id, total_agents, logdir, filename):
    """
    Save matrix data to a CSV file for all timesteps.
    
    Args:
        matrix: List of timesteps, where each timestep contains n_agent data
        attacked_agent_id: ID of the attacked agent
        total_agents: Total number of agents
        logdir: Directory to save the file
    """
    filepath = os.path.join(logdir, filename)
    
    # Create header row
    # header = ["timestep", "attacked_agent"]
    header = ["timestep", "is_attacked", "attacked_agent"]
    for i in range(total_agents):
        header.append(f"agent_{i}")
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        for timestep, timestep_data in enumerate(matrix):
            is_attacked = 1 if timestep in attacked_steps else 0
            row = [timestep, is_attacked, attacked_agent_id]
            for i in range(total_agents):
                row.append(timestep_data[i])
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

    # Read reference values from CSV files if provided
    ref_means = [[] for _ in range(maddpg.nagents)]
    ref_std_devs = [[] for _ in range(maddpg.nagents)]

    for agent_id in range(maddpg.nagents):
        csv_filename = f"maddpg_taylor_error_atk_free_agent_{agent_id}.csv"
        csv_path = os.path.join(config.ref_val_dir, csv_filename)
        
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                ref_means[agent_id].append(float(row[2]))
                ref_std_devs[agent_id].append(float(row[4]))

    attacked_agent_id = config.attack_agent_id  # specify the agent to attack
    seed = 42

    results_normal, _, frob_norms_normal, sec_dir_derivatives_normal = get_episode_data(env, maddpg, config, logdir, ref_means, ref_std_devs, do_attack=False, atk_agent_id=attacked_agent_id, seed=seed)

    results_attacked, attacked_steps, frob_norms_atk, sec_dir_derivatives_atk = get_episode_data(env, maddpg, config, logdir, ref_means, ref_std_devs, do_attack=True, atk_agent_id=attacked_agent_id, seed=seed)
    save_matrix_to_files(results_attacked, attacked_steps, attacked_agent_id, maddpg.nagents, logdir, f'maddpg_taylor_error_atk_{attacked_agent_id}.csv')
    save_matrix_to_files(frob_norms_atk, attacked_steps, attacked_agent_id, maddpg.nagents, logdir, f'maddpg_frobenius_norms_atk_{attacked_agent_id}.csv')
    save_matrix_to_files(sec_dir_derivatives_atk, attacked_steps, attacked_agent_id, maddpg.nagents, logdir, f'maddpg_sec_dir_derivatives_atk_{attacked_agent_id}.csv')

    plot_results(results_attacked, attacked_steps, attacked_agent_id, ref_means, ref_std_devs, logdir)
    plot_frobs(frob_norms_normal, frob_norms_atk, attacked_steps, attacked_agent_id, logdir)
    plot_sec_dir_derivatives(sec_dir_derivatives_normal, sec_dir_derivatives_atk, attacked_steps, attacked_agent_id, logdir)
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_path",
                        help="model directory")
    parser.add_argument("--save_gifs", action="store_true",
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--ref_val_dir", type=str, default='',)
    parser.add_argument("--attack_agent_id", type=int, default=0,)
    parser.add_argument("--atk_start_step", type=int, default=-math.inf)
    parser.add_argument("--atk_end_step", type=int, default=math.inf)

    config = parser.parse_args()

    run(config)
