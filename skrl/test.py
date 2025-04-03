from collections import deque
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
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from skrl.utils import set_seed


def plot(episode_data_unattacked, episode_data_attacked, attacked_agent_id, log_dir: str):    
    # Get the agent IDs from the first dictionary
    agent_ids = list(episode_data_unattacked.keys())
    num_agents = len(agent_ids)
    
    # Calculate grid dimensions
    rows = int(np.ceil(np.sqrt(num_agents)))
    cols = int(np.ceil(num_agents / rows))
    
    # Create a single figure with subplots for all agents
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    fig.suptitle(f'SO-INRD Values for All Agents, Attacked Agent: {attacked_agent_id}, Attack Mode: Random Attack after Step 5 at 50% Rate', fontsize=16)
    
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


def perturb_random_noise(states, perturb_agent_id, noise_std=0.1):
    perturbed_states = states.copy()
    perturbed_states[perturb_agent_id] = states[perturb_agent_id] + noise_std * torch.randn_like(states[perturb_agent_id])
    return perturbed_states

def compute_log_prob(agent: IPPO | MAPPO, agent_id, state, action):
    policy_model = agent.models[agent_id]["policy"]
    net_output, _ = policy_model.compute({"states": agent._state_preprocessor[agent_id](state)}, role="policy")

    # In some cases, got 1.00 in dist.probs when using Categorical(logits=net_output) which produced some issues. 
    # Precision errors in internal computations most likely. Directly using torch.softmax solves the problem.
    # dist = Categorical(logits=net_output) if policy_model._c_unnormalized_log_prob else Categorical(probs=net_output)
    
    probs = torch.softmax(net_output, dim=-1) if policy_model._c_unnormalized_log_prob else net_output
    dist = Categorical(probs=probs)
    log_prob = dist.log_prob(action)
    return log_prob

def compute_so_inrd(agent: IPPO | MAPPO, agent_id, states, epsilon):
    actions, _, _ = agent.act(states, 0, 0)

    state = states[agent_id].detach().clone().requires_grad_(True)
    # Transfer action to agent.device before computing log probability
    loss = compute_log_prob(agent, agent_id, state, actions[agent_id])
    
    # Compute the gradient of q_value with respect to stacked_states using autograd.grad
    grad_J = torch.autograd.grad(loss, state, create_graph=False)[0]

    # Compute η_i (adversarial perturbation direction)
    eta_i = epsilon * grad_J.sign() / torch.max(grad_J.norm(p=2), torch.tensor(1e-6))

    # Compute J tilde
    J_tilde = loss + torch.dot(grad_J.flatten(), eta_i.flatten())

    # Perturbed state towards adversarial direction
    perturbed_state = states[agent_id] + eta_i
    perturbed_loss = compute_log_prob(agent, agent_id, perturbed_state, actions[agent_id])
    
    # Compute L
    L = perturbed_loss - J_tilde
    return L.item()


def get_episode_data(env, agent: IPPO | MAPPO, do_attack: bool, attacked_agent_id: str, logdir: str):

    # Run one episode and perturb the observation of the "adversary" agent
    state, info = env.reset(seed=42)
    done = {agent_id: False for agent_id in env.agents}
    episode_reward = {agent_id: 0.0 for agent_id in env.agents}
    so_inrd_vals = dict()
    episode_data = dict()
    perturb_eps = 0.1

    iter_count = 0
    frames = []  # List to collect frames

    for agent_id in env.agents:
        episode_data[agent_id] = {'so_inrd_l': []}
        so_inrd_vals[agent_id] = deque(maxlen=5) # maintain a window of k. we are considering a subtrajectory of the last k states.

    while not all(done.values()):
        # if do_attack and iter_count > 5 and np.random.rand() < 1.0:
        #     state = perturb_random_noise(state, attacked_agent_id, noise_std=perturb_eps)
        
        # Get actions from the agent (in evaluation mode, training=False)
        actions, log_prob, _ = agent.act(state, 0, 0)
        if do_attack and iter_count > 5 and np.random.rand() < 0.5:
            actions[attacked_agent_id] = torch.tensor([[env.action_space(attacked_agent_id).sample()]])

        # Save the frame for this step and append to frames list
        frame = env.render()
        frames.append(Image.fromarray(frame))

        next_state, reward, termination, truncation, info = env.step(actions)
        
        # Check for terminal condition per agent
        done = {agent_id: termination[agent_id] or truncation[agent_id] for agent_id in env.agents}

        for idx, agent_id in enumerate(env.agents):
            so_inrd_l = compute_so_inrd(agent, agent_id, state, perturb_eps)
            so_inrd_vals[agent_id].append(so_inrd_l)
            episode_data[agent_id]['so_inrd_l'].append(sum(so_inrd_vals[agent_id]))
        
        for agent_id in env.agents:
            episode_reward[agent_id] += reward[agent_id].item()
        
        state = next_state
        iter_count += 1

    print("Episode finished. Rewards:", episode_reward)
    imageio.mimwrite(
        os.path.join(logdir, f"episode_vid_attack_{do_attack}.gif"), frames, duration=125
    )
    return episode_data


def main(args):
    set_seed(42)
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
            device=env.device,
            eval_mode=True)  # Set eval_mode to True to get deterministic actions
        
        value_obs_space = env.observation_space(agent_name)
        if args.algo_name == 'MAPPO':
            value_obs_space = shared_observation_space
        models[agent_name]["value"] = ValueDeterministic(observation_space=value_obs_space, 
            action_space=env.action_space(agent_name), 
            device=env.device)
        
        memories[agent_name] = RandomMemory(memory_size=cfg_agent['rollouts'], num_envs=env.num_envs, device=env.device)


    # instantiate the agent
    # (assuming a defined environment <env> and memories <memories>)
    agent: IPPO | MAPPO = agent_class(possible_agents=env.possible_agents,
                models=models,
                memories=memories,  # only required during training
                cfg=cfg_agent,
                observation_spaces=env.observation_spaces,
                action_spaces=env.action_spaces,
                device=env.device,
                **agent_kwargs)

    agent.load(args.model_dir) # Load the model from the specified directory

    episode_data_unattacked = get_episode_data(env, agent, False, None, log_dir)
    attacked_agent_id = "agent_0"
    episode_data_attacked = get_episode_data(env, agent, True, attacked_agent_id, log_dir)

    env.close()

    # Plot the SO-INRD values for all agents
    plot(episode_data_unattacked, episode_data_attacked, attacked_agent_id, log_dir)


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