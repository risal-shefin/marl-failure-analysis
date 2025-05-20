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
import matplotlib.pyplot as plt
from PIL import Image
from collections import deque


USE_CUDA = torch.cuda.is_available()
DEVICE = 'gpu' if USE_CUDA else 'cpu'
torch_device = torch.device("cuda" if USE_CUDA else "cpu")

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


def get_episode_data(env, maddpg, config, logdir):
    # obs = env.reset(seed=42)
    obs = env.reset(seed=12345) # better for speaker_listener_v3
    episode_reward = 0
    frames = []
    # initialize deque buffers for last batch_size observations
    so_inrd_deques = [[deque(maxlen=5) for _ in range(maddpg.nagents)] for _ in range(maddpg.nagents)]
    so_inrd_vals = []

    while True:
        # add Gaussian noise to an agent's observation
        # noise_scale = 0.0  # adjust the standard deviation of the noise as needed
        attacked_agent_id = 1
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
        if config.discrete_action:
            actions = {agent_name: agent_actions[i].argmax() for i, agent_name in enumerate(env.possible_agents)}
        else:
            actions = {agent_name: agent_actions[i].squeeze() for i, agent_name in enumerate(env.possible_agents)}

        # random attack
        actions[env.possible_agents[attacked_agent_id]] = env.action_spaces[env.possible_agents[attacked_agent_id]].sample()

        if config.save_gifs:
            frames.append(Image.fromarray(env.render()))
        
        so_inrd_mat = so_inrd(maddpg, obs, list(actions.values()), 0.1)
        for i in range(maddpg.nagents):
            for j in range(maddpg.nagents):
                so_inrd_deques[i][j].append(so_inrd_mat[i][j])
        so_inrd_vals.append([[sum(so_inrd_deques[i][j]) for j in range(maddpg.nagents)] for i in range(maddpg.nagents)])

        next_obs, rewards, dones, infos = env.step(actions)
        episode_reward += np.sum([rewards[:,i] if env.agent_types[i] != 'adversary' else 0 for i in range(len(rewards))])

        obs = next_obs
        if dones.all():
            break

    print(f"Episode reward: {episode_reward}")
    if config.save_gifs:
        imageio.mimsave(os.path.join(logdir, f'{config.env_id}_episode.gif'), frames, duration=125)
        print(f"Saved gif of episode to {logdir}")

    return so_inrd_vals


def plot_so_inrd(so_inrd_list, logdir):
    # convert to array: shape (T, N_agents * N_agents)
    data = np.array(so_inrd_list)
    timesteps = np.arange(data.shape[0])

    plt.figure(figsize=(10, 6))
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            plt.plot(timesteps, data[:, i, j], label=f"{i},{j}")
    plt.xlabel("Timestep")
    plt.ylabel("SO INRD")
    plt.title("Cross SO INRD over Time for Each Agent Pair (agent 1 attacked)")
    plt.legend(loc="upper right", fontsize="small", ncol=2)
    plt.tight_layout()

    out_path = os.path.join(logdir, "cross_so.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved cross-so plot to {out_path}")

def run(config):
    maddpg = MADDPG.init_from_save(config.model_path)

    # create a log directory under runs/<env_id>/<timestamp> using os and getcwd
    cwd = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(cwd, 'runs', f"{config.env_id}_{'discrete' if maddpg.discrete_action else 'continuous'}", timestamp)
    os.makedirs(logdir, exist_ok=True)

    env_func = getattr(mpe, config.env_id)
    env = env_func.parallel_env(continuous_actions= not config.discrete_action, render_mode='rgb_array')
    env = PettingZooWrapper.wrap_env(env)
    env.reset()

    # maddpg.prep_rollouts(device=DEVICE)
    maddpg.prep_training(device=DEVICE)

    so_inrd_list = get_episode_data(env, maddpg, config, logdir)
    plot_so_inrd(so_inrd_list, logdir)
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_path",
                        help="model directory")
    parser.add_argument("--save_gifs", action="store_true",
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--discrete_action", action="store_true",
                        help="Whether the action space is discrete or continuous")

    config = parser.parse_args()

    run(config)
