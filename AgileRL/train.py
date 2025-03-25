import numpy as np
import torch
import argparse
import datetime
import os
import pettingzoo.mpe as mpe
import agilerl.algorithms as algorithms
import gymnasium as gym

from tqdm import tqdm, trange
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
from agilerl.utils.algo_utils import obs_channels_to_first
from agilerl.utils.utils import create_population
from agilerl.algorithms.maddpg import MADDPG
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.training.train_multi_agent import train_multi_agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def has_image_observations(observation_spaces):
    for obs_space in observation_spaces:
        # Check if space is Box type (which images would be)
        if isinstance(obs_space, gym.spaces.Box):
            # Check if shape has 3 dimensions (typical for images: H,W,C)
            if len(obs_space.shape) == 3:
                return True
    return False

NET_IMG_CONFIG = {
    "encoder_config": {
      'hidden_size': [128, 128],  # Encoder Network head hidden size
      'channel_size': [32, 32], # CNN channel size (for image observations)
      'kernel_size': [8, 4],   # CNN kernel size   (for image observations)
      'stride_size': [4, 2],   # CNN stride size   (for image observations)
    },
    "head_config": {'hidden_size': [128, 128]}  # Network head hidden size
}

NET_MLP_CONFIG = {
    "encoder_config": {
      'hidden_size': [128, 128],  # Encoder Network head hidden size
    },
    "head_config": {'hidden_size': [128, 128]}  # Network head hidden size
}

def main(args):
    # Create log directory: logs/{env}/{timestamp}
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = os.path.join(os.getcwd(), "logs", f"{args.env_id}_{args.algo_name}", timestamp)
    os.makedirs(logdir, exist_ok=True)
    checkpoint_path = os.path.join(logdir, "checkpoint.pt")

    num_envs = 8
    # Dynamically import the environment from pettingzoo.mpe
    try:
        env_func = getattr(mpe, args.env_id)
    except AttributeError:
        raise ValueError(f"Environment {args.env_id} not found in pettingzoo.mpe")
    env = env_func.parallel_env(continuous_actions=False)
    env = AsyncPettingZooVecEnv([lambda: env for _ in range(num_envs)])

    # Configure the multi-agent algo input arguments
    observation_spaces = [env.single_observation_space(agent) for agent in env.agents]
    action_spaces = [env.single_action_space(agent) for agent in env.agents]
    net_config = NET_IMG_CONFIG if has_image_observations(observation_spaces) else NET_MLP_CONFIG

    agent_ids = env.agents
    n_agents = env.num_agents
    # Replay Buffer
    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = MultiAgentReplayBuffer(
        memory_size=1_000_000,
        field_names=field_names,
        agent_ids=agent_ids,
        device=device,
    )

    if args.algo_name == 'MADDPG':
        agent = MADDPG(
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            agent_ids=agent_ids,
            vect_noise_dim=num_envs,
            device=device,
            net_config=net_config
        )
    else:
        raise ValueError(f"Algorithm {args.algo_name} is not implemented in this train script")

    # Define training loop parameters
    max_steps = args.train_steps  # Max steps
    evo_steps = 10000
    learning_delay = 1000  # Steps before starting learning
    total_steps = 0
    progress_bar = tqdm(total=max_steps, desc="Training Progress")
    last_step_count = agent.steps[-1]
    best_score = -np.inf

    while agent.steps[-1] < max_steps:
        state, info  = env.reset() # Reset environment at start of episode
        scores = np.zeros(num_envs)
        completed_episode_scores = []

        steps = 0
        for idx_step in range(evo_steps // num_envs):
            # Get next action from agent
            cont_actions, discrete_action = agent.get_action(
                obs=state,
                training=True,
                infos=info,
            )
            action = discrete_action if agent.discrete_actions else cont_actions

            # Act in environment
            next_state, reward, termination, truncation, info = env.step(action)

            scores += np.sum(np.array(list(reward.values())).transpose(), axis=-1)
            total_steps += num_envs
            steps += num_envs

            # Save experiences to replay buffer
            done = termination or truncation
            memory.save_to_memory(state, cont_actions, reward, next_state, done, is_vectorised=True)

            # Learn according to learning frequency
            # Handle learn steps > num_envs
            if agent.learn_step > num_envs:
                learn_step = agent.learn_step // num_envs
                if (
                    idx_step % learn_step == 0
                    and len(memory) >= agent.batch_size
                    and memory.counter > learning_delay
                ):
                    # Sample replay buffer
                    experiences = memory.sample(agent.batch_size)
                    # Learn according to agent's RL algorithm
                    agent.learn(experiences)
            # Handle num_envs > learn step; learn multiple times per step in env
            elif (
                len(memory) >= agent.batch_size and memory.counter > learning_delay
            ):
                for _ in range(num_envs // agent.learn_step):
                    # Sample replay buffer
                    experiences = memory.sample(agent.batch_size)
                    # Learn according to agent's RL algorithm
                    agent.learn(experiences)

            # Update the state
            state = next_state

            # Calculate scores and reset noise for finished episodes
            reset_noise_indices = []
            term_array = np.array(list(termination.values())).transpose()
            trunc_array = np.array(list(truncation.values())).transpose()
            for idx, (d, t) in enumerate(zip(term_array, trunc_array)):
                if np.any(d) or np.any(t):
                    completed_episode_scores.append(scores[idx])
                    agent.scores.append(scores[idx])
                    scores[idx] = 0
                    reset_noise_indices.append(idx)
            agent.reset_action_noise(reset_noise_indices)

        agent.steps[-1] += steps
        progress_bar.update(agent.steps[-1] - last_step_count)
        last_step_count = agent.steps[-1]

        # Episodic Score Summary
        mean_episode_score = np.mean(completed_episode_scores)
        print(f"Mean score: {mean_episode_score}", flush=True)

        # Save best agent's checkpoint
        if mean_episode_score >= best_score:
            best_score = mean_episode_score
            agent.save_checkpoint(checkpoint_path)
            print(f"Saved checkpoint with score: {best_score}", flush=True)

    progress_bar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL agent on a PettingZoo MPE environment")
    parser.add_argument("--env_id", type=str,
                        help="Name of the environment from pettingzoo.mpe (for ex: simple_speaker_listener_v4)")
    parser.add_argument("--algo_name", type=str, default='MADDPG',
                        help="Algorithm Name")
    parser.add_argument("--train_steps", type=int, default=10_000_000,
                        help="Total Train Steps")
    args = parser.parse_args()
    main(args)