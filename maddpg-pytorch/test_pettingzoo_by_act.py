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
    torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    actions = [Variable(torch.Tensor([actions[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    vf_in = torch.cat((*torch_obs, *actions), dim=1)

    policy_loss = -maddpg.agents[attacked_agent_id].critic(vf_in).mean() + (actions[attacked_agent_id]**2).mean() * 1e-3
    grad = torch.autograd.grad(policy_loss, torch_obs[attacked_agent_id], retain_graph=True)[0]
    eta = epsilon * grad.sign()
    obs_perturbed_i = obs[attacked_agent_id] + torch.dot(grad.flatten(), eta.flatten()).cpu().numpy()
    return obs_perturbed_i

def so_inrd(maddpg, obs, actions, epsilon):
    is_obs_image = isinstance(obs[0], np.ndarray) and len(obs[0].shape) >= 3
    torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    actions = [Variable(torch.Tensor([actions[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    vf_in = torch.cat((*torch_obs, *actions), dim=1) if not is_obs_image else (torch_obs, actions)
    so_inrd_mat = [[] for _ in range(maddpg.nagents)]
    
    for i, agent_i in enumerate(maddpg.agents):
        if not is_obs_image:
            policy_loss_i = -agent_i.critic(vf_in).mean() + (actions[i]**2).mean() * 1e-3
        else:
            policy_loss_i = -agent_i.critic(*vf_in).mean() + (actions[i]**2).mean() * 1e-3

        for j, agent_j in enumerate(maddpg.agents):
            # The gradient with respect to obs
            grad_J = torch.autograd.grad(policy_loss_i, actions[j], retain_graph=True)[0]

            # Compute η_i (adversarial perturbation direction)
            eta_j = epsilon * grad_J.sign() / torch.max(grad_J.norm(p=2), torch.tensor(1e-6))

            # Compute J tilde
            J_tilde = policy_loss_i + torch.dot(grad_J.flatten(), eta_j.flatten())

            # Perturb action
            new_actions = [actions[i].clone() for i in range(maddpg.nagents)]
            new_actions[j] = actions[j] + eta_j # perturbation
            new_obs = [torch_obs[i].clone() for i in range(maddpg.nagents)]
            # new_obs[j] = torch_obs[j] + eta_j  # perturbation
            vf_in_perturbed = torch.cat((*new_obs, *new_actions), dim=1)
            policy_loss_i_perturbed = -agent_i.critic(vf_in_perturbed).mean() + (new_actions[i]**2).mean() * 1e-3

            # Compute L
            L = policy_loss_i_perturbed - J_tilde
            # so_inrd_mat[i].append(L.item())
            so_inrd_mat[i].append(grad_J.norm(p=2).detach().cpu().numpy())

    return so_inrd_mat

def hessian_wrt_action(maddpg, obs, actions, epsilon):
    is_obs_image = isinstance(obs[0], np.ndarray) and len(obs[0].shape) >= 3
    torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    actions = [Variable(torch.Tensor([actions[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    vf_in = torch.cat((*torch_obs, *actions), dim=1) if not is_obs_image else (torch_obs,actions)
    hessian_mat = [[] for _ in range(maddpg.nagents)]
    
    for i, agent_i in enumerate(maddpg.agents):
        if not is_obs_image:
            policy_loss_i = -agent_i.critic(vf_in).mean() + (actions[i]**2).mean() * 1e-3
        else:
            policy_loss_i = -agent_i.critic(*vf_in).mean() + (actions[i]**2).mean() * 1e-3

        def policy_loss_i_fn(agent_id, agent_action):
            new_actions = actions.copy()
            new_actions[agent_id] = agent_action
            new_vf_in = torch.cat((*torch_obs, *new_actions), dim=1)
            return -agent_i.critic(new_vf_in).mean() + (new_actions[i]**2).mean() * 1e-3

        for j, agent_j in enumerate(maddpg.agents):
            # The gradient with respect to action
            grad_J = torch.autograd.grad(policy_loss_i, actions[j], retain_graph=True, create_graph=True)[0]
            grad_J = grad_J.view(-1)
            # print(" >>> ", grad_J, actions[j], policy_loss_i)
            # Compute second-order gradients (Hessian diagonal elements)
            grad2_J = []
            for k in range(grad_J.shape[0]):
                grad2_scalar = torch.autograd.grad(grad_J[k], actions[j], retain_graph=True)[0]
                grad2_J.append(grad2_scalar)
            grad2_J = torch.stack(grad2_J)
            tmp = grad2_J.norm(p=1).detach().cpu().numpy()
            # print(" >>> ", i, j, grad_J.norm(p=1).detach().cpu().numpy(), tmp)
            hessian_mat[i].append(tmp)

            # grad2_J = torch.autograd.functional.hessian(lambda x: policy_loss_i_fn(j, x), actions[j])
            # grad2_J = grad2_J.squeeze()
            # hessian_mat[i].append(grad2_J.norm(p=2).cpu().numpy())

    return hessian_mat


def get_episode_data(env, maddpg, config, logdir, do_attack=False, atk_aget_id=-1):
    # obs = env.reset(seed=42)
    obs = env.reset(seed=42) # better for speaker_listener_v3
    episode_reward = 0
    frames = []
    # initialize deque buffers for last batch_size observations
    metric_deques = [[deque(maxlen=5) for _ in range(maddpg.nagents)] for _ in range(maddpg.nagents)]
    metric_vals = []

    while True:
        # add Gaussian noise to an agent's observation
        # noise_scale = 0.0  # adjust the standard deviation of the noise as needed
        # obs[attacked_agent] = obs[attacked_agent] + np.random.randn(*obs[attacked_agent].shape) * noise_scale

        # FGSM attack
        # torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=False) for i in range(maddpg.nagents)]
        # torch_agent_actions = maddpg.step(torch_obs, explore=False)
        # agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions]
        # if config.discrete_action:
        #     actions = {agent_name: agent_actions[i].argmax() for i, agent_name in enumerate(env.possible_agents)}
        # else:
        #     actions = {agent_name: agent_actions[i].squeeze() for i, agent_name in enumerate(env.possible_agents)}
        # obs[attacked_agent_id] = fgsm_attack(maddpg, obs, list(actions.values()), attacked_agent_id, 0.1)
        
        torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=False) for i in range(maddpg.nagents)]
        torch_agent_actions = maddpg.step(torch_obs, explore=False)
        agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions]
        if maddpg.discrete_action:
            actions = {agent_name: agent_actions[i].argmax() for i, agent_name in enumerate(env.possible_agents)}
        else:
            actions = {agent_name: agent_actions[i].squeeze() for i, agent_name in enumerate(env.possible_agents)}

        # random attack
        if do_attack:
            actions[env.possible_agents[atk_aget_id]] = env.action_spaces[env.possible_agents[atk_aget_id]].sample()

        if config.save_gifs:
            frames.append(Image.fromarray(env.render()))
        
        cur_actions = actions.copy()
        if maddpg.discrete_action:  # convert discrete actions to one-hot encoding
            for agent_name in actions:
                action_dim = env.action_spaces[agent_name].n
                cur_actions[agent_name] = torch.nn.functional.one_hot(torch.tensor(cur_actions[agent_name]), action_dim).float().numpy()
        result_mat = hessian_wrt_action(maddpg, obs, list(cur_actions.values()), 0.1)
        # result_mat = so_inrd(maddpg, obs, list(actions.values()), 0.1)
        for i in range(maddpg.nagents):
            for j in range(maddpg.nagents):
                metric_deques[i][j].append(result_mat[i][j])
        metric_vals.append([[sum(metric_deques[i][j]) for j in range(maddpg.nagents)] for i in range(maddpg.nagents)])

        next_obs, rewards, dones, infos = env.step(actions)
        episode_reward += np.sum([rewards[:,i] if env.agent_types[i] != 'adversary' else np.zeros_like(rewards[:,i]) for i in range(maddpg.nagents)])  # sum rewards for all agents except adversaries

        obs = next_obs
        if dones.all():
            break

    print(f"Episode reward: {episode_reward}")
    if config.save_gifs:
        imageio.mimsave(os.path.join(logdir, f'{config.env_id}_atk_{do_attack}_episode.gif'), frames, duration=125)
        print(f"Saved gif of episode to {logdir}")

    return metric_vals


def plot_results(results, results_attacked, atk_agent_id, logdir):
    n = len(results[0])  # number of agents
    t = len(results)     # number of time steps
    
    # Create n x n subplots
    fig, axes = plt.subplots(n, n, figsize=(4*n, 4*n))
    fig.suptitle(f'First Order Analysis (Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.95)
    
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
            ax.set_ylabel('First Order(L_i wrt A_j)')
            ax.set_title(f'Loss_{i} , Action_{j}')
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(logdir, f'so_inrd_analysis_attacked_{atk_agent_id}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved So-INRD analysis plot to {logdir}")


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

    attacked_agent_id = 1  # specify the agent to attack
    results = get_episode_data(env, maddpg, config, logdir)
    results_attacked = get_episode_data(env, maddpg, config, logdir, do_attack=True, atk_aget_id=attacked_agent_id)
    plot_results(results, results_attacked, attacked_agent_id, logdir)
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_path",
                        help="model directory")
    parser.add_argument("--save_gifs", action="store_true",
                        help="Saves gif of each episode into model directory")

    config = parser.parse_args()

    run(config)
