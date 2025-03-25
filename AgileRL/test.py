from collections import deque
import torch
import numpy as np
import argparse

from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
from agilerl.algorithms.maddpg import MADDPG
from agilerl.utils.algo_utils import obs_channels_to_first
from datetime import datetime
import os
import pettingzoo.mpe as mpe
import matplotlib.pyplot as plt
import random
import imageio
from PIL import Image, ImageDraw

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define function to return image
def _label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)

    if np.mean(frame) < 128:
        text_color = (255, 255, 255)
    else:
        text_color = (0, 0, 0)
    drawer.text(
        (im.size[0] / 20, im.size[1] / 18), f"Episode: {episode_num+1}", fill=text_color
    )

    return im


# Define a perturbation function to add Gaussian noise to a specific agent's observation.
def perturb_obs_random_noise(obs, perturb_agent, noise_std=0.1):
    perturbed_obs = {}
    for agent_id, ob in obs.items():
        if agent_id == perturb_agent:
            # Assuming the observation is a NumPy array; if it's a tensor, use torch.randn_like
            perturbed_obs[agent_id] = ob + noise_std * np.random.randn(*ob.shape)
        else:
            perturbed_obs[agent_id] = ob
    return perturbed_obs


def compute_q_of_all_actions(agent, agent_id, agent_index, stacked_states, actions_dist):
    """
    # Compute one hot distribution of all possible actions of the agent with the given agent_id.
    # The remaining agents' actions are kept the same as the original action_dist.
    # Note: the critic expects a distribution representing actions."
    """
    one_hot_all_actions = []
    # Calculate total number of elements
    total_elements = np.prod(actions_dist[agent_id].shape)
    # Generate each one-hot array
    for i in range(total_elements):
        new_actions_dist = actions_dist.copy()
        # Create a flattened array of zeros
        flat_array = np.zeros(total_elements)
        # Set the i-th element to 1
        flat_array[i] = 1
        # Reshape to the desired shape and add to the list
        new_actions_dist[agent_id] = flat_array.reshape(actions_dist[agent_id].shape)
        one_hot_all_actions.append(new_actions_dist)

    # Compute Q values
    q_vals = []
    for acts in one_hot_all_actions:
        processed_actions = {
            agent_id: torch.tensor(act_dist, device=device, dtype=torch.float32)
            for agent_id, act_dist in acts.items()
        }
        stacked_actions = torch.cat(list(processed_actions.values()), dim=1)
        q_val = agent.critics[agent_index](stacked_states, stacked_actions)
        q_vals.append(q_val)

    return q_vals

def compute_log_prob_action(agent, agent_id, agent_index, stacked_states, actions, actions_dist):
    # Compute Q values for all possible actions of the agent with agent_id
    q_vals = compute_q_of_all_actions(agent, agent_id, agent_index, stacked_states, actions_dist)
    # Convert list of q_vals to a tensor
    q_tensor = torch.cat(q_vals, dim=0)

    # apply softmax to get probability distribution
    prob_dist = torch.softmax(q_tensor, dim=0)
    return -torch.log(prob_dist[actions[agent_id]]) # log probability of the action taken

def compute_so_inrd(agent: MADDPG, agent_id, agent_index, obs, actions, actions_dist, epsilon):

    # Stack states
    stacked_states = agent.stack_critic_observations(agent.preprocess_observation(obs)).detach().clone().requires_grad_(True)

    loss = compute_log_prob_action(agent, agent_id, agent_index, stacked_states, actions, actions_dist)
    
    # Compute the gradient of q_value with respect to stacked_states using autograd.grad
    grad_J = torch.autograd.grad(loss, stacked_states, create_graph=False)[0]

    # Compute η_i (adversarial perturbation direction)
    eta_i = epsilon * grad_J.sign() / torch.max(grad_J.norm(p=2), torch.tensor(1e-6))

    # Compute J tilde
    J_tilde = loss + torch.dot(grad_J.flatten(), eta_i.flatten())

    # Perturbed state towards adversarial direction
    perturbed_stacked_states = stacked_states + eta_i
    perturbed_loss = compute_log_prob_action(agent, agent_id, agent_index, perturbed_stacked_states, actions, actions_dist)
    
    # Compute L
    L = perturbed_loss - J_tilde
    return L.item()


def get_episode_data(env, agent, do_attack: bool, attacked_agent_id: str, logdir: str):

    # Run one episode and perturb the observation of the "adversary" agent
    done = {agent_id: False for agent_id in env.agents}
    episode_reward = {agent_id: 0.0 for agent_id in env.agents}
    state, info = env.reset(seed=25)
    so_inrd_vals = dict()
    episode_data = dict()
    perturb_eps = 0.1

    for agent_id in env.agents:
        episode_data[agent_id] = {'so_inrd_l': []}
        so_inrd_vals[agent_id] = deque(maxlen=5) # maintain a window of k. we are considering a subtrajectory of the last k states.

    iter_count = 0
    frames = []  # List to collect frames

    while not all(done.values()):
        # if do_attack and iter_count > 5 and np.random.rand() < 1.0:
        #     state = perturb_obs_random_noise(state, attacked_agent_id, noise_std=perturb_eps)
        
        # Get actions from the agent (in evaluation mode, training=False)
        cont_actions, discrete_action = agent.get_action(
            obs=state,
            training=False,
            infos=info
        )
        if do_attack and iter_count > 5 and np.random.rand() < 0.5:
            discrete_action[attacked_agent_id] = env.action_space(attacked_agent_id).sample()
        # Choose discreate action if available
        action = discrete_action if agent.discrete_actions else cont_actions

        # Save the frame for this step and append to frames list
        frame = env.render()[0]
        frames.append(_label_with_episode_number(frame, 0))

        next_state, reward, termination, truncation, info = env.step(action)
        
        # Check for terminal condition per agent
        done = {agent_id: termination[agent_id] or truncation[agent_id] for agent_id in env.agents}
        
        for idx, agent_id in enumerate(env.agents):
            so_inrd_l = compute_so_inrd(agent, agent_id, idx, state, action, cont_actions, perturb_eps)
            so_inrd_vals[agent_id].append(so_inrd_l)
            episode_data[agent_id]['so_inrd_l'].append(sum(so_inrd_vals[agent_id]))
        
        for agent_id in env.agents:
            episode_reward[agent_id] += reward[agent_id]
        
        state = next_state
        iter_count += 1

    print("Episode finished. Rewards:", episode_reward)
    imageio.mimwrite(
        os.path.join(logdir, f"episode_vid_attack_{do_attack}.gif"), frames, duration=10
    )
    return episode_data


def plot(episode_data_unattacked, episode_data_attacked, attacked_agent_id, log_dir: str):    
    # Get the agent IDs from the first dictionary
    agent_ids = list(episode_data_unattacked.keys())
    num_agents = len(agent_ids)
    
    # Calculate grid dimensions
    rows = int(np.ceil(np.sqrt(num_agents)))
    cols = int(np.ceil(num_agents / rows))
    
    # Create a single figure with subplots for all agents
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    fig.suptitle(f'SO-INRD Values for All Agents, Attacked Agent: {attacked_agent_id}, Attack Mode: Random Noise after Step 5 at 50% Rate', fontsize=16)
    
    # Make axes iterable even for a single subplot
    if num_agents == 1:
        axes = np.array([axes])
    
    # Flatten axes array for easy iteration
    axes = axes.flatten()
    
    # Plot data for each agent in its own subplot
    for i, agent_id in enumerate(agent_ids):
        # Extract the so_inrd_l values for both unattacked and attacked scenarios
        unattacked_values = episode_data_unattacked[agent_id]['so_inrd_l']
        attacked_values = episode_data_attacked[agent_id]['so_inrd_l']
        
        # Plot the values
        axes[i].plot(range(len(unattacked_values)), unattacked_values, label='Unattacked', color='blue')
        axes[i].plot(range(len(attacked_values)), attacked_values, label='Attacked', color='red')
        
        # Add labels and title for each subplot
        axes[i].set_xlabel('Step')
        axes[i].set_ylabel('SO-INRD L Value')
        axes[i].set_title(f'Agent: {agent_id}')
        axes[i].legend()
        axes[i].grid(True)
    
    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle
    
    # Save the figure with all subplots
    plot_path = os.path.join(log_dir, 'so_inrd_all_agents_eps_0.1.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Plot for all agents saved to {plot_path}")


def main(args):
    # Set up a single test environment
    num_envs = 1
    # Dynamically import the environment from pettingzoo.mpe
    try:
        env_func = getattr(mpe, args.env_id)
    except AttributeError:
        raise ValueError(f"Environment {args.env_id} not found in pettingzoo.mpe")
    env = env_func.parallel_env(continuous_actions=False, render_mode='rgb_array')
    env = AsyncPettingZooVecEnv([lambda: env for _ in range(num_envs)])

    log_dir = os.path.join(os.getcwd(), "logs", f"{args.env_id}_{args.algo_name}", "exp_loss_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)     # Create the log directory if it doesn't exist

    # Print all available discrete actions for each agent
    for agent_id in env.agents:
        action_space = env.single_action_space(agent_id)
        if hasattr(action_space, 'n'):  # For Discrete action spaces
            print(f"Agent {agent_id} has {action_space.n} discrete actions (0 to {action_space.n-1})")
        elif hasattr(action_space, 'nvec'):  # For MultiDiscrete action spaces
            print(f"Agent {agent_id} has MultiDiscrete action space with dimensions: {action_space.nvec}")
        else:
            print(f"Agent {agent_id} has continuous action space: {action_space}")

    # Configure agent parameters similar to training
    agent_ids = env.agents
    observation_spaces = [env.single_observation_space(a) for a in env.agents]
    action_spaces = [env.single_action_space(a) for a in env.agents]

    # Create the MADDPG agent
    if args.algo_name == 'MADDPG':
        agent = MADDPG.load(args.model_dir, device=device)
        print("\n--Loaded MADDPG agent from", args.model_dir)
    else:
        raise ValueError(f"Algorithm {args.algo_name} is not implemented in this test script")

    episode_data_unattacked = get_episode_data(env, agent, False, None, log_dir)

    # Agent Ids of simple_speaker_listener_v4 env: [listener_0, speaker_0]
    # Agent Ids of simple_speaker_listener_v4 env: [agent_0, agent_1, agent_2]
    attacked_agent_id = "agent_0"
    episode_data_attacked = get_episode_data(env, agent, True, attacked_agent_id, log_dir)

    env.close()

    # Plot the SO-INRD values for all agents
    plot(episode_data_unattacked, episode_data_attacked, attacked_agent_id, log_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL agent on a PettingZoo MPE environment")
    parser.add_argument("--env_id", type=str,
                        help="Name of the environment from pettingzoo.mpe (for ex: simple_speaker_listener_v4)")
    parser.add_argument("--algo_name", type=str, default='MADDPG',
                        help="Algorithm Name")
    parser.add_argument("--model_dir", type=str,
                        help="checkpoint path")
    args = parser.parse_args()
    main(args)