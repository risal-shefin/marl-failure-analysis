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
import pettingzoo.mpe as mpe
from PIL import Image


USE_CUDA = torch.cuda.is_available()
DEVICE = 'gpu' if USE_CUDA else 'cpu'


def get_episode_data(env, maddpg, config, logdir):
    obs = env.reset(seed=42)
    # obs = env.reset(seed=999) # better for speaker_listener_v3
    episode_reward = 0
    frames = []

    while True:
        torch_obs = [Variable(torch.Tensor([obs[i]]).to('cuda' if USE_CUDA else 'cpu'), requires_grad=False) for i in range(maddpg.nagents)]

        torch_agent_actions = maddpg.step(torch_obs, explore=False)
        agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions]
        if config.discrete_action:
            actions = {agent_name: agent_actions[i].argmax() for i, agent_name in enumerate(env.possible_agents)}
        else:
            actions = {agent_name: agent_actions[i].squeeze() for i, agent_name in enumerate(env.possible_agents)}

        if config.save_gifs:
            frames.append(Image.fromarray(env.render()))

        next_obs, rewards, dones, infos = env.step(actions)
        episode_reward += np.sum([rewards[:,i] if env.agent_types[i] != 'adversary' else 0 for i in range(len(rewards))])

        obs = next_obs
        if dones.all():
            break

    print(f"Episode reward: {episode_reward}")
    if config.save_gifs:
        imageio.mimsave(os.path.join(logdir, f'{config.env_id}_episode.gif'), frames, duration=125)
        print(f"Saved gif of episode to {logdir}")


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

    get_episode_data(env, maddpg, config, logdir)


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