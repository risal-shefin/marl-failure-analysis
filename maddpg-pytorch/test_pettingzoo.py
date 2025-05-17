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

def compute_cross_hessian_vector_product(maddpg, obs):
    # obs: list of numpy arrays of shape (batch_size, obs_dim)
    torch_obs = [Variable(torch.Tensor(obs[i]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    # compute actions under gradient for proper gradient flow
    actions = []
    for i, agent_i in enumerate(maddpg.agents):
        if maddpg.discrete_action:
            pol_out = agent_i.policy(torch_obs[i])
            ac = gumbel_softmax(pol_out, hard=True)
        else:
            ac = agent_i.policy(torch_obs[i])
        actions.append(ac)

    vf_in = torch.cat((*torch_obs, *actions), dim=1)
    hvp_ij = []
    for i, agent_i in enumerate(maddpg.agents):
        if maddpg.discrete_action:
            policy_loss = -maddpg.agents[i].critic(vf_in).mean() + (gumbel_softmax(actions[i], hard=True)**2).mean() * 1e-3
        else:
            policy_loss = -maddpg.agents[i].critic(vf_in).mean() + (actions[i]**2).mean() * 1e-3

        # compute gradient of the policy_loss w.r.t. the agent_i's policy parameters
        grads_i = torch.autograd.grad(policy_loss, agent_i.critic.parameters(), retain_graph=True, create_graph=True, allow_unused=True)
        grad_i_flat = torch.cat([g.view(-1) for g in grads_i if g is not None])
        
        for j, agent_j in enumerate(maddpg.agents):
            params_j = agent_j.policy.parameters()
            v = torch.randn_like(grad_i_flat) # HVP's vector v
            v = v / (v.norm() + 1e-8)   # normalize v to improve stability of influence measurements
            # HVP: compute (H_ij) v
            cross_hvp = torch.autograd.grad(grad_i_flat, params_j, grad_outputs=v, retain_graph=True, allow_unused=True)
            cross_hvp_flat = torch.cat([h.view(-1) for h in cross_hvp if h is not None])
            hvp_ij.append(cross_hvp_flat.norm(p=2))

    return hvp_ij


def get_episode_data(env, maddpg, config, logdir):
    # obs = env.reset(seed=42)
    obs = env.reset(seed=12345) # better for speaker_listener_v3
    episode_reward = 0
    frames = []
    cross_hvp_list = []
    # initialize deque buffers for last batch_size observations
    obs_buffer = [deque(maxlen=config.batch_size) for _ in range(maddpg.nagents)]

    while True:
        # add Gaussian noise to each agent's observation
        noise_scale = 0.0  # adjust the standard deviation of the noise as needed
        obs[1] = obs[1] + np.random.randn(*obs[1].shape) * noise_scale
        torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=False) for i in range(maddpg.nagents)]
        torch_agent_actions = maddpg.step(torch_obs, explore=False)
        agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions]
        if config.discrete_action:
            actions = {agent_name: agent_actions[i].argmax() for i, agent_name in enumerate(env.possible_agents)}
        else:
            actions = {agent_name: agent_actions[i].squeeze() for i, agent_name in enumerate(env.possible_agents)}

        if config.save_gifs:
            frames.append(Image.fromarray(env.render()))

        # append current obs to sliding window buffers
        for i in range(maddpg.nagents):
            obs_buffer[i].append(obs[i])
        # once we have batch_size observations, compute cross-HVP on last batch
        if len(obs_buffer[0]) == config.batch_size:
            batch_obs = [np.stack(obs_buffer[i], axis=0) for i in range(maddpg.nagents)]
            cross_hvp_list.append(compute_cross_hessian_vector_product(maddpg, batch_obs))

        next_obs, rewards, dones, infos = env.step(actions)
        episode_reward += np.sum([rewards[:,i] if env.agent_types[i] != 'adversary' else 0 for i in range(len(rewards))])

        obs = next_obs
        if dones.all():
                        break

    print(f"Episode reward: {episode_reward}")
    if config.save_gifs:
        imageio.mimsave(os.path.join(logdir, f'{config.env_id}_episode.gif'), frames, duration=125)
        print(f"Saved gif of episode to {logdir}")

    return cross_hvp_list


def plot_cross_hessian(cross_hessian_list, logdir):
    # convert to array: shape (T, N_agents * N_agents)
    data = np.array(cross_hessian_list)
    M = data.shape[1]
    N = int(np.sqrt(M))
    if N * N != M:
        raise ValueError(f"Expected square number of entries per timestep, got {M}")

    # reshape to (T, N, N)
    data = data.reshape(-1, N, N)
    timesteps = np.arange(data.shape[0])

    plt.figure(figsize=(10, 6))
    for i in range(N):
        for j in range(N):
            plt.plot(timesteps, data[:, i, j], label=f"{i}->{j}")
    plt.xlabel("Timestep")
    plt.ylabel("Cross Hessian")
    plt.title("Cross Hessian over Time for Each Agent Pair (agent 1 attacked)")
    plt.legend(loc="upper right", fontsize="small", ncol=2)
    plt.tight_layout()

    out_path = os.path.join(logdir, "cross_hessian.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved cross-hessian plot to {out_path}")

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

    maddpg.prep_rollouts(device=DEVICE)

    cross_hessian_list = get_episode_data(env, maddpg, config, logdir)
    plot_cross_hessian(cross_hessian_list, logdir)
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
    parser.add_argument("--batch_size", type=int, default=5,
                        help="Batch size for HVP computation")

    config = parser.parse_args()

    run(config)
