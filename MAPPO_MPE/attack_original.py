from collections import deque
from make_env_pettingzoo import make_env
from datetime import datetime
import torch
import torch.nn as nn
import argparse
import gymnasium
import numpy as np
import os
import imageio
from PIL import Image, ImageDraw
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import csv

from MAPPO_MPE_main import Runner_MAPPO_MPE

def plot(episode_data_unattacked, episode_data_attacked, attacked_agent_id, log_dir: str, filename: str,args):    
    # Get the agent IDs from the first dictionary
    agent_ids = list(episode_data_unattacked.keys())
    num_agents = len(agent_ids)
    
    # Calculate grid dimensions
    rows = int(np.ceil(np.sqrt(num_agents)))
    cols = int(np.ceil(num_agents / rows))
    
    # Create a single figure with subplots for all agents
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
    fig.suptitle(f'SO-INRD Values for All Agents, Attacked Agent: {attacked_agent_id}, Attack Mode: Attack after Step 5 at {args.attack_rate*100}% Rate on Eps :{args.perturb_eps}', fontsize=16)
    
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
    plot_path = os.path.join(log_dir, f'{filename}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Plot for all agents saved to {plot_path}")


def perturb_random_noise(states, perturb_agent_id, noise_std=0.1):
    perturbed_states = states.copy()
    perturbed_states[perturb_agent_id] = states[perturb_agent_id] + noise_std * torch.randn_like(states[perturb_agent_id])
    return perturbed_states


def get_action_log_prob(runner, agent_id, state, action):

    obs = torch.tensor(obs, dtype=torch.float).unsqueeze(0)  # shape: (1, obs_dim)
            
    # Add agent ID if needed
    if runner.agent_n.add_agent_id:
        agent_id_one_hot = torch.zeros(1, runner.agent_n.N)
        agent_id_one_hot[0, agent_id] = 1.0
        actor_input = torch.cat([state, agent_id_one_hot], dim=-1)
    else:
        actor_input = state

    # Reset RNN hidden state if using RNN
    if runner.use_rnn:
        batch_size = actor_input.size(0)  # Should be 1
        self.actor.rnn_hidden = torch.zeros(batch_size, self.rnn_hidden_dim, 
                                            device=actor_input.device)
    
    # Get the actor network output (either probabilities or logits)
    actor_output = agent.policy_n[agent_id].actor(state)
    
    # Create a distribution based on the output
    dist = Categorical(logits=actor_output)
    
    # Calculate log probability of the given action
    log_prob = dist.log_prob(action)
        
    return log_prob

def compute_so_inrd(agent, agent_id, states, epsilon):

    state = torch.tensor(states[agent_id]).requires_grad_(True)
    action = torch.tensor(runner.agent_n.select_action(state, agent_id,evaluate=True))
    loss = runner.agent_n.compute_log_prob(state, agent_id, action)
    # print(f"Log Prob: {loss}")

    # Compute the gradient of q_value with respect to stacked_states using autograd.grad
    grad_J = torch.autograd.grad(loss, state, create_graph=False)[0]

    # Compute η_i (adversarial perturbation direction)
    eta_i = epsilon * grad_J.sign() / torch.max(grad_J.norm(p=2), torch.tensor(1e-6))

    # Compute J tilde
    J_tilde = loss + torch.dot(grad_J.flatten(), eta_i.flatten())

    # Perturbed state towards adversarial direction
    perturbed_state = state + eta_i
    perturbed_loss= runner.agent_n.compute_log_prob(perturbed_state, agent_id, action)
    # Compute L
    L = perturbed_loss - J_tilde
    return L.item()


def get_episode_data(env, agent, do_attack: bool, attacked_agent_id: str, logdir: str,args):

    # Run one episode and perturb the observation of the "adversary" agent
    state = env.reset()
    done = [False for agent_id in range(runner.args.N)]
    episode_reward = {agent_id: 0.0 for agent_id in range(runner.args.N)}
    so_inrd_vals = dict()
    episode_data = dict()
    perturb_eps = 0.001
    
    # Add actions tracking
    action_data = []

    iter_count = 0
    frames = []  # List to collect frames

    for agent_id in range(runner.args.N):
        episode_data[agent_id] = {'so_inrd_l': []}
        so_inrd_vals[agent_id] = deque(maxlen=5) # maintain a window of k. we are considering a subtrajectory of the last k states.

    while not all(done):
        # Get actions from the agent (in evaluation mode, training=False)
        actions = []
        original_actions = []
        
        for id in range(runner.args.N):
            action = runner.agent_n.select_action(state[id], id, evaluate=True)
            original_actions.append(action)
            
            # If attacking and it's the targeted agent, potentially modify action
            if do_attack and id == attacked_agent_id and iter_count > 5 and np.random.rand() < args.attack_rate:
                action = env.action_space[attacked_agent_id].sample()
            
            actions.append(action)
        
        # Record action for attacked agent
        if id == attacked_agent_id:
            # Store step number, original action, and actual action used
            action_data.append({
                'step': iter_count,
                'original_action': original_actions[attacked_agent_id],
                'used_action': actions[attacked_agent_id],
                'was_attacked': do_attack and iter_count > 5 and original_actions[attacked_agent_id] != actions[attacked_agent_id]
            })
        
        next_state, reward, done, info = env.step(actions)

        for agent_id in range(runner.args.N):
            so_inrd_l = compute_so_inrd(runner, agent_id, state, args.perturb_eps)
            so_inrd_vals[agent_id].append(so_inrd_l)
            episode_data[agent_id]['so_inrd_l'].append(sum(so_inrd_vals[agent_id]))
        
        for agent_id in range(runner.args.N):
            episode_reward[agent_id] += reward[agent_id]
        
        state = next_state
        iter_count += 1

    print("Episode finished. Rewards:", episode_reward)
    
    # Add the action data to the episode data
    episode_data['action_data'] = action_data
    
    return episode_data


def main(runner, env, args):
    attacked_agent_id = args.attacked_agent_id
    log_dir = os.path.join(os.getcwd(), f"Attacked_Agent_{attacked_agent_id}", "attack_runs", 
                          f"experiments_{runner.env_name}_MAPPO", 
                          "exp_loss_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate filename if not provided
    if args.filename is None:
        args.filename = f"Agent_{attacked_agent_id}_{int(args.attack_rate*100)}%_attack_eps_{args.perturb_eps}"
    
    episode_data_unattacked = get_episode_data(env, runner, False, None, log_dir,args)
    episode_data_attacked = get_episode_data(env, runner, True, attacked_agent_id, log_dir,args)

    # Save action data to CSV
    csv_path = os.path.join(log_dir, f"action_comparison_{args.filename}.csv")
    save_action_comparison_to_csv(
        episode_data_unattacked.get('action_data', []),
        episode_data_attacked.get('action_data', []),
        attacked_agent_id,
        csv_path
    )
    
    env.close()

    # Plot the SO-INRD values for all agents
    plot(episode_data_unattacked, episode_data_attacked, attacked_agent_id, log_dir,args.filename,args)


def save_action_comparison_to_csv(unattacked_action_data, attacked_action_data, agent_id, csv_path):
    """
    Save action comparison data to a CSV file.
    
    Args:
        unattacked_action_data: List of action data dictionaries from unattacked episode
        attacked_action_data: List of action data dictionaries from attacked episode
        agent_id: ID of the agent being attacked
        csv_path: Path to save the CSV file
    """
    # Create a combined dataset where we can compare actions at each step
    combined_data = {}
    
    # Process unattacked data
    for entry in unattacked_action_data:
        step = entry['step']
        if step not in combined_data:
            combined_data[step] = {'step': step}
        combined_data[step]['unattacked_action'] = entry['used_action']
    
    # Process attacked data
    for entry in attacked_action_data:
        step = entry['step']
        if step not in combined_data:
            combined_data[step] = {'step': step}
        combined_data[step]['attacked_action'] = entry['used_action']
        combined_data[step]['was_attacked'] = entry['was_attacked']
        combined_data[step]['original_action'] = entry['original_action']
    
    # Sort by step
    sorted_data = [combined_data[step] for step in sorted(combined_data.keys())]
    
    # Write to CSV
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['step', 'unattacked_action', 'original_action', 'attacked_action', 'was_attacked']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in sorted_data:
            # Fill in any missing fields with None/empty values
            for field in fieldnames:
                if field not in row:
                    row[field] = ""
            writer.writerow(row)
    
    print(f"Action comparison saved to {csv_path}")

if __name__ == '__main__':
    env = make_env(env_name="simple_spread_v3", discrete=True)
    
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help="Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=25, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the rnn")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the mlp")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling. Here, we do not use it.")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=float, default=False, help="Whether to use relu, if False, we will use tanh")
    parser.add_argument("--use_rnn", type=bool, default=False, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=float, default=False, help="Whether to add agent_id. Here, we do not use it.")
    parser.add_argument("--use_value_clip", type=float, default=False, help="Whether to use value clip.")
    parser.add_argument("--attack_rate", type=float, default=0.5, help="Attack probability when attacking (0.0-1.0)")
    parser.add_argument("--perturb_eps", type=float, default=0.1, help="Perturbation epsilon value for attacks")
    parser.add_argument("--filename", type=str, default=None, help="Custom filename for plots (if None, auto-generated)")
    parser.add_argument("--attacked_agent_id", type=int, default=0, help="Whether to add agent_id. Here, we do not use it.")
    # Add output directory argument
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save all output files")

    args = parser.parse_args()
    runner = Runner_MAPPO_MPE(args, env_name="simple_spread_v3", number=1, seed=0)
    
    runner.agent_n.load_model_from_directory("/deac/csc/vanbastelaerGrp/guptd23/RL_Project/MARL-code-pytorch/MAPPO_MPE/model/MAPPO_actor_env_simple_spread_number_1_seed_0_step_1215k.pth")
    main(runner, env,args)
    # runner = Runner_MAPPO_MPE(args, env_name="simple_spread_v3", number=1, seed=0)
    # runner.run()
