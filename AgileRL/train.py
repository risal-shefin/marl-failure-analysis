import numpy as np
import torch
from pettingzoo.mpe import simple_speaker_listener_v4
from tqdm import trange

from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
from agilerl.utils.algo_utils import obs_channels_to_first
from agilerl.algorithms.maddpg import MADDPG
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_envs = 8
env = simple_speaker_listener_v4.parallel_env(max_cycles=25, continuous_actions=True)
env = AsyncPettingZooVecEnv([lambda: env for _ in range(num_envs)])
env.reset()

# Configure the multi-agent algo input arguments
observation_spaces = [env.single_observation_space(agent) for agent in env.agents]
action_spaces = [env.single_action_space(agent) for agent in env.agents]

channels_last = False  # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
n_agents = env.num_agents
agent_ids = [agent_id for agent_id in env.agents]
field_names = ["state", "action", "reward", "next_state", "done"]
memory = MultiAgentReplayBuffer(
    memory_size=1_000_000,
    field_names=field_names,
    agent_ids=agent_ids,
    device=device,
)

agent = MADDPG(
    observation_spaces=observation_spaces,
    action_spaces=action_spaces,
    agent_ids=agent_ids,
    vect_noise_dim=num_envs,
    device=device,
)

# Define training loop parameters
max_steps = int(5e7)  # Max steps
total_steps = 0
progress_bar = tqdm(total=max_steps, desc="Training Progress")
initial_step = agent.steps[-1]
best_score = -np.inf
while agent.steps[-1] < max_steps:
    state, info  = env.reset() # Reset environment at start of episode
    scores = np.zeros(num_envs)
    completed_episode_scores = []
    if channels_last:
        state = {agent_id: obs_channels_to_first(s) for agent_id, s in state.items()}

    steps = 0
    for _ in range(1000):

        # Get next action from agent
        cont_actions, discrete_action = agent.get_action(
            obs=state,
            training=True,
            infos=info,
        )
        if agent.discrete_actions:
            action = discrete_action
        else:
            action = cont_actions

        # Act in environment
        next_state, reward, termination, truncation, info = env.step(action)

        scores += np.sum(np.array(list(reward.values())).transpose(), axis=-1)
        total_steps += num_envs
        steps += num_envs

        # Save experiences to replay buffer
        if channels_last:
            next_state = {
                agent_id: obs_channels_to_first(ns)
                for agent_id, ns in next_state.items()
            }
        done = termination or truncation
        memory.save_to_memory(state, cont_actions, reward, next_state, done, is_vectorised=True)

        # Learn according to learning frequency
        if len(memory) >= agent.batch_size:
            for _ in range(num_envs // agent.learn_step):
                experiences = memory.sample(agent.batch_size) # Sample replay buffer
                agent.learn(experiences) # Learn according to agent's RL algorithm

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

    mean_episode_score = np.mean(completed_episode_scores)
    print(f"Mean score: {mean_episode_score}", flush=True)
    agent.steps[-1] += steps
    progress_bar.update(agent.steps[-1] - initial_step)
    initial_step = agent.steps[-1]
    if mean_episode_score > best_score:
        best_score = mean_episode_score
        agent.save_checkpoint('./simple_speaker_listener_v4_checkpoint.pt')
        print(f"Saved checkpoint with score: {best_score}", flush=True)
progress_bar.close()