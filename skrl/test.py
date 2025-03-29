from skrl.multi_agents.torch.ippo import IPPO, IPPO_DEFAULT_CONFIG
from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.memories.torch import RandomMemory
from skrl.envs.wrappers.torch import wrap_env
from skrl.models.torch import Model, CategoricalMixin, DeterministicMixin
from datetime import datetime
import pettingzoo.mpe as mpe
import torch
import torch.nn as nn
import argparse
import gymnasium
import numpy as np
import os
from train import PolicyCategorical, ValueDeterministic
import imageio
from PIL import Image, ImageDraw

# Define function to return image
def _label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)

    if np.mean(frame) < 128:
        text_color = (255, 255, 255)
    else:
        text_color = (0, 0, 0)
    drawer.text(
        (im.size[0] // 20, im.size[1] // 18), f"Episode: {episode_num+1}", fill=0
    )

    return im

def get_episode_data(env, agent: IPPO | MAPPO, do_attack: bool, attacked_agent_id: str, logdir: str):

    # Run one episode and perturb the observation of the "adversary" agent
    done = {agent_id: False for agent_id in env.agents}
    episode_reward = {agent_id: 0.0 for agent_id in env.agents}
    state, info = env.reset()
    so_inrd_vals = dict()
    episode_data = dict()
    perturb_eps = 0.1

    iter_count = 0
    frames = []  # List to collect frames

    while not all(done.values()):
        # if do_attack and iter_count > 5 and np.random.rand() < 1.0:
        #     state = perturb_obs_random_noise(state, attacked_agent_id, noise_std=perturb_eps)
        
        # Get actions from the agent (in evaluation mode, training=False)
        actions, log_prob, _ = agent.act(state, 0, 0)
        if do_attack and iter_count > 5 and np.random.rand() < 0.5:
            actions[attacked_agent_id] = env.action_space(attacked_agent_id).sample()

        # Save the frame for this step and append to frames list
        frame = env.render()
        frames.append(Image.fromarray(frame))

        next_state, reward, termination, truncation, info = env.step(actions)
        
        # Check for terminal condition per agent
        done = {agent_id: termination[agent_id] or truncation[agent_id] for agent_id in env.agents}
        
        for agent_id in env.agents:
            episode_reward[agent_id] += reward[agent_id]
        
        state = next_state
        iter_count += 1

    print("Episode finished. Rewards:", episode_reward)
    imageio.mimwrite(
        os.path.join(logdir, f"episode_vid_attack_{do_attack}.gif"), frames, duration=10
    )
    return episode_data


def main(args):
    # Dynamically import the environment from pettingzoo.mpe
    try:
        env_func = getattr(mpe, args.env_id)
    except AttributeError:
        raise ValueError(f"Environment {args.env_id} not found in pettingzoo.mpe")
    env = env_func.parallel_env(continuous_actions=False, render_mode="rgb_array")

    # wrap the environment
    env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="pettingzoo")'

    log_dir = os.path.join(os.getcwd(), "runs", f"experiments_{args.env_id}_{args.algo_name}", "exp_loss_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)     # Create the log directory if it doesn't exist

    # Agent configs
    cfg_agent = {}
    agent_kwargs = {}
    agent_class: IPPO | MAPPO = None
    if args.algo_name == 'IPPO':
        cfg_agent = IPPO_DEFAULT_CONFIG.copy()
        agent_class = IPPO
    elif args.algo_name == 'MAPPO':
        cfg_agent = MAPPO_DEFAULT_CONFIG.copy()
        agent_class = MAPPO

        shared_observation_spaces_low = []
        shared_observation_spaces_high = []
        for agent_name in env.possible_agents:
            shared_observation_spaces_low.append(env.observation_spaces[agent_name].low)
            shared_observation_spaces_high.append(env.observation_spaces[agent_name].high)
            
        shared_observation_space = gymnasium.spaces.Box(
            low=np.concatenate(shared_observation_spaces_low),
            high=np.concatenate(shared_observation_spaces_high),
            dtype=np.float32
        )
        agent_kwargs.update({"shared_observation_spaces": {agent_name: shared_observation_space for agent_name in env.possible_agents}})
    else:
        raise ValueError(f"Algorithm {args.algo_name} is not supported")
    
    # adjust some configuration if necessary
    cfg_agent.update({
        "rollouts": 128,                 # Increased rollouts for more stable updates
        "mini_batches": 8,              # Increased mini-batches for better gradient updates
        "learning_rate": 3e-4,                  # Reduced learning rate for stability
        "learning_rate_scheduler": torch.optim.lr_scheduler.LinearLR,
        "learning_rate_scheduler_kwargs": {"start_factor": 1.0, "end_factor": 0.1, "total_iters": 1e6},
        "grad_norm_clip": 1.0,              # Increased gradient clipping
        "entropy_loss_scale": 0.01,      # Added entropy loss scaling for better exploration
        "experiment": {
            "checkpoint_interval": 500,
        }
    })

    # instantiate the agent's models and memories
    models = {}
    memories = {}
    for agent_name in env.possible_agents:
        models[agent_name] = {}

        models[agent_name]["policy"] = PolicyCategorical(
            observation_space=env.observation_space(agent_name), 
            action_space=env.action_space(agent_name), 
            device=env.device)
        
        value_obs_space = env.observation_space(agent_name)
        if args.algo_name == 'MAPPO':
            value_obs_space = shared_observation_space
        models[agent_name]["value"] = ValueDeterministic(observation_space=value_obs_space, 
            action_space=env.action_space(agent_name), 
            device=env.device)
        
        memories[agent_name] = RandomMemory(memory_size=cfg_agent['rollouts'], num_envs=env.num_envs, device=env.device)


    # instantiate the agent
    # (assuming a defined environment <env> and memories <memories>)
    agent = agent_class(possible_agents=env.possible_agents,
                models=models,
                memories=memories,  # only required during training
                cfg=cfg_agent,
                observation_spaces=env.observation_spaces,
                action_spaces=env.action_spaces,
                device=env.device,
                **agent_kwargs)

    agent.load(args.model_dir) # Load the model from the specified directory

    get_episode_data(env, agent, False, None, log_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL agent on a PettingZoo MPE environment")
    parser.add_argument("--env_id", type=str,
                        help="Name of the environment from pettingzoo.mpe (for ex: simple_speaker_listener_v4)")
    parser.add_argument("--algo_name", type=str, default='IPPO',
                        help="Algorithm Name")
    parser.add_argument("--model_dir", type=str, default='',
                        help="Model Directory")
    args = parser.parse_args()
    main(args)