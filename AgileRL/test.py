import torch
import numpy as np
from pettingzoo.mpe import simple_speaker_listener_v4
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
from agilerl.algorithms.maddpg import MADDPG
from agilerl.utils.algo_utils import obs_channels_to_first

# Define a perturbation function to add Gaussian noise to a specific agent's observation.
def perturb_observation(obs, perturb_agent, noise_std=0.1):
    """
    Perturbs the observation of the specified agent by adding Gaussian noise.
    Arguments:
        obs (dict): Mapping from agent id to its observation.
        perturb_agent (str): The agent whose observation is perturbed.
        noise_std (float): Standard deviation of the Gaussian noise.
    Returns:
        dict: Perturbed observation dictionary.
    """
    perturbed_obs = {}
    for agent_id, ob in obs.items():
        if agent_id == perturb_agent:
            # Assuming the observation is a NumPy array; if it's a tensor, use torch.randn_like
            perturbed_obs[agent_id] = ob + noise_std * np.random.randn(*ob.shape)
        else:
            perturbed_obs[agent_id] = ob
    return perturbed_obs

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up a single test environment
num_envs = 1
base_env = simple_speaker_listener_v4.parallel_env(max_cycles=25, continuous_actions=True)
env = AsyncPettingZooVecEnv([lambda: base_env for _ in range(num_envs)])
state, info = env.reset(seed=42)

# Configure agent parameters similar to training
agent_ids = env.agents
observation_spaces = [env.single_observation_space(a) for a in env.agents]
action_spaces = [env.single_action_space(a) for a in env.agents]

# Create the MADDPG agent
agent = MADDPG(
    observation_spaces=observation_spaces,
    action_spaces=action_spaces,
    agent_ids=agent_ids,
    vect_noise_dim=num_envs,
    device=device,
)

# Load the trained checkpoint (.pt extension is acceptable by torch)
agent.load_checkpoint('./simple_speaker_listener_v4_checkpoint.pt')

# Run one episode and perturb the observation of the "adversary" agent
done = {agent_id: False for agent_id in env.agents}
episode_reward = {agent_id: 0.0 for agent_id in env.agents}

while not all(done.values()):
    # Perturb observations for the targeted agent. ["listener_0", "speaker_0"]
    perturbed_state = perturb_observation(state, "listener_0", noise_std=0.1)
    
    # Get actions from the agent (in evaluation mode, training=False)
    cont_actions, discrete_action = agent.get_action(
        obs=state,
        training=False,
        infos=info
    )
    # Choose continuous actions if available
    action = cont_actions if not agent.discrete_actions else discrete_action
    
    next_state, reward, termination, truncation, info = env.step(action)
    
    # Check for terminal condition per agent
    done = {agent_id: termination[agent_id] or truncation[agent_id] for agent_id in env.agents}
    
    # Accumulate rewards
    for agent_id in env.agents:
        episode_reward[agent_id] += reward[agent_id]
    
    state = next_state

print("Episode finished. Rewards:", episode_reward)
env.close()