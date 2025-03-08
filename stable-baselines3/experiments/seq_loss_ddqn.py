from collections import deque
import warnings
import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import Tensor
from torch.nn import functional as F
from stable_baselines3.dqn.ddqn import DoubleDQN
from datetime import datetime
import os
import argparse

def compute_cross_entropy_loss(agent, obs_list, actions):
    loss = torch.tensor(0.0)
    for i in range(len(obs_list)):  
        q_values = agent.q_net(agent.policy.obs_to_tensor(obs_list[i])[0])[0]
        action = actions[i]
        policy_probs = torch.softmax(q_values, dim=-1)  # Convert Q-values to probabilities
        loss += torch.log(policy_probs[action])  # Cross-entropy loss
    return loss

def compute_loss(agent, obs, action, reward, next_obs, done, step_size):
    with torch.no_grad():
        # Compute the next Q-values using the target network
        next_q_values = agent.q_net_target(next_obs)

        # Decouple action selection from value estimation
        # Compute q-values for the next observation using the online q net
        next_q_values_online = agent.q_net(next_obs)
        # Select action with online network
        next_actions_online = torch.argmax(next_q_values_online, dim=1)

        # Estimate the q-values for the selected actions using target q network
        next_q_values = torch.gather(next_q_values, dim=1, index=next_actions_online.unsqueeze(-1))
        # Avoid potential broadcast issue
        next_q_values = next_q_values.reshape(-1, 1)
        # k-step TD target
        target_q_values = reward + (1 - done) * (agent.gamma ** step_size) * next_q_values

    # Get current Q-values estimates
    current_q_values = agent.q_net(obs)

    # Retrieve the q-values for the actions from the replay buffer
    current_q_values = torch.gather(current_q_values, dim=1, index=action.unsqueeze(0))

    # Compute Huber loss (less sensitive to outliers)
    # loss = F.smooth_l1_loss(current_q_values, target_q_values)
    loss = current_q_values - target_q_values

    return loss

def perturb_obs_fgsm(agent: DoubleDQN, env, obs, epsilon):
    """ Compute perturbed obs using FGSM attack """
    obs_tensor = agent.policy.obs_to_tensor(obs)[0]
    obs_tensor = obs_tensor.float().requires_grad_(True) # Clone the obs tensor and set requires_grad=True

    saved_env_state = env.unwrapped.clone_state(include_rng=True) # Save the Current State Before Rollout
    action, _ = agent.predict(obs, deterministic=True)
    next_obs, reward, terminated, truncated, _ = env.step(action)
    next_obs_tensor = agent.policy.obs_to_tensor(next_obs)[0]
    env.unwrapped.restore_state(saved_env_state)  # Restore the Environment Back to the Saved State

    loss = compute_loss(agent, obs_tensor, torch.tensor(action).unsqueeze(0), torch.tensor(reward).unsqueeze(0), 
                        next_obs_tensor, torch.tensor(int(terminated or truncated)).unsqueeze(0))

    # action, _ = agent.predict(obs, deterministic=True)
    # q_values = agent.q_net(obs_tensor)
    # target_probs = F.one_hot(torch.tensor(action), num_classes=q_values.shape[-1]).float()
    # loss = compute_loss(agent, obs_tensor, target_probs)

    # The gradient with respect to obs
    grad_J = torch.autograd.grad(loss, obs_tensor)[0]

    # Compute η_i (adversarial perturbation direction)
    eta_i = epsilon * grad_J.sign()

    # Perturbed state
    # eta_i's shape (1, 3, 210, 160) to (210, 160, 3)
    perturbed_obs = obs + eta_i.permute(0, 2, 3, 1).squeeze(0).detach().numpy()
    return perturbed_obs

def perturb_obs_random_noise(obs, epsilon):
    # Normalize to [0,1]
    obs = obs / 255.0
    # Add Gaussian noise with mean=0, std=0.1
    noise = np.random.randn(*obs.shape) * epsilon
    # Add noise & Clip values to valid range [0,1]
    perturbed_obs = np.clip(obs + noise, 0, 1)
    # Rescale back to [0,255] and convert to NumPy uint8
    return (perturbed_obs * 255.0).astype(np.uint8)

def so_inrd(agent: DoubleDQN, episode_data, loss_window_size, epsilon):
    """ Second Order Identification of Non-Robust Directions (SO-INRD) """
    history_length = len(episode_data['states'])
    history_start_idx = max(0, history_length - loss_window_size)
    obs = episode_data['states'][history_start_idx]

    obs_tensor = agent.policy.obs_to_tensor(obs)[0]
    obs_tensor = obs_tensor.float().requires_grad_(True) # Clone the obs tensor and set requires_grad=True

    last_obs = episode_data['next_states'][-1]
    last_obs_tensor = agent.policy.obs_to_tensor(last_obs)[0]

    action = episode_data['actions'][history_start_idx]
    subtraj_reward = sum(episode_data['rewards'][history_start_idx:])
    last_done = int(episode_data['dones'][-1])

    # n-step td error
    # loss = compute_loss(agent, obs_tensor, torch.tensor(action).unsqueeze(0), torch.tensor(subtraj_reward).unsqueeze(0),
    #                     last_obs_tensor, torch.tensor(last_done).unsqueeze(0), loss_window_size)

    # q_values = agent.q_net(obs_tensor)
    # target_probs = F.one_hot(torch.tensor(action), num_classes=q_values.shape[-1]).float()
    # loss = compute_loss(agent, obs_tensor, target_probs)
    loss = compute_cross_entropy_loss(agent, episode_data['states'][history_start_idx:], episode_data['actions'][history_start_idx:])
    return loss

    # The gradient with respect to obs
    grad_J = torch.autograd.grad(loss, obs_tensor)[0]
    # print(" >>", grad_J.norm(p=2))
    
    # Compute η_i (adversarial perturbation direction)
    eta_i = epsilon * grad_J.sign() / torch.max(grad_J.norm(p=2), torch.tensor(1.0))

    # Compute J tilde
    J_tilde = loss + torch.dot(grad_J.flatten(), eta_i.flatten())

    # Perturbed state
    perturbed_obs_tensor = obs_tensor + eta_i
    perturbed_loss = compute_loss(agent, perturbed_obs_tensor, torch.tensor(action).unsqueeze(0), torch.tensor(reward).unsqueeze(0), 
                        next_obs_tensor, torch.tensor(int(done)).unsqueeze(0))
    # perturbed_loss = compute_loss(agent, perturbed_obs_tensor, target_probs)

    # Compute L
    L = perturbed_loss - J_tilde
    return L


def get_episode_data(model_dir, env_id, do_attack: bool, log_dir: str):
    env = gym.make(env_id, render_mode="rgb_array")
    obs, info = env.reset(seed=42)
    agent = DoubleDQN.load(model_dir, env=env)
    done = False
    cumulative_rew = 0
    loss_window_size = 10

    episode_data = {'states': [], 'actions': [], 'rewards': [], 'dones': [], 
                    'next_states': [], 'so_inrd_l': [], 'attack_flag': []}

    perutrb_eps = 0.1
    step_counter = 0
    while not done:
        step_counter += 1
        
        is_attacked = False
        original_obs = obs
        original_action, _ = agent.predict(obs, deterministic=True)
        original_q_values = agent.q_net(agent.policy.obs_to_tensor(obs)[0])
        original_act_prob =  torch.softmax(original_q_values, dim=-1)[0][original_action]
        # if do_attack and agent.get_values(torch_obs) > 0.7:
        if do_attack and step_counter > 200 and np.random.rand() < 1.0:
            # obs = perturb_obs_fgsm(agent, env, obs, perutrb_eps)
            obs = perturb_obs_random_noise(obs, perutrb_eps)
            is_attacked = True

            atk_action, _ = agent.predict(obs, deterministic=True)
            atk_q_values = agent.q_net(agent.policy.obs_to_tensor(obs)[0])
            atk_act_prob =  torch.softmax(atk_q_values, dim=-1)[0][original_action]
            if original_action != atk_action and False:
                from experiments.utils import plot_two_frames
                # plot_two_frames(original_obs, obs, os.path.join(log_dir, f'fgsm_attack_{step_counter}.png'))
                print(f"\n-----Original Action: {original_action}, Perturbed Obs Action: {atk_action}")
                print(f"Original Q-Values: {original_q_values}")
                print(f"Perturbed Obs Q-Values: {atk_q_values}")
                print(f"Original Action Prob: {original_act_prob}, Original Action Prob in Perturbed Obs: {atk_act_prob}, Perturbed Obs Action Prob: {torch.softmax(atk_q_values, dim=-1)[0][atk_action]}")

        action, _ = agent.predict(obs, deterministic=True)
        # Single environment step
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        cumulative_rew += reward

        episode_data['states'].append(obs)
        episode_data['actions'].append(action)
        episode_data['rewards'].append(reward)
        episode_data['dones'].append(done)
        episode_data['next_states'].append(next_obs)
        episode_data['attack_flag'].append(is_attacked)
        episode_data['so_inrd_l'].append(so_inrd(agent, episode_data, loss_window_size, perutrb_eps).item())

        obs = next_obs  # Set the next state as the current state
    
    print(f"Episode Reward = {cumulative_rew}")
    episode_data['sum_reward'] = cumulative_rew
    env.close()
    return episode_data
    # return cumulative_rew


def plot(episode_data, episode_data_attacked, log_dir: str):
    fig = plt.figure()
    # print(" >>", max(episode_data['so_inrd_l']))
    plt.plot(np.arange(len(episode_data['so_inrd_l'])), episode_data['so_inrd_l'], 
             label="Unattacked")
    plt.plot(np.arange(len(episode_data_attacked['so_inrd_l'])), episode_data_attacked['so_inrd_l'], 
             label="Attacked")
    # Add a dashed vertical line at step_x
    attack_flags = episode_data_attacked['attack_flag']
    attacked_steps = np.arange(len(episode_data_attacked['so_inrd_l']))[attack_flags]
    attacked_values = np.zeros(len(episode_data_attacked['so_inrd_l']))[attack_flags]
    # plt.scatter(attacked_steps, attacked_values, color='r', marker='x', label="Attacked Step")
    # plt.axvline(x=index, color='r', linestyle='--', label=f'Attacked Step')
    plt.xlabel("Steps")
    # plt.ylabel("Q(s1,a1) - (r_1+..+r_{k-1} + Q(s_k,a_k))")
    plt.ylabel("log(p(a1))+log(p(a2))+...+log(p(ak))")
    # plt.yscale('log')
    plt.title("Env: Boxing, Noise Attack 100% After 200 Steps")
    plt.legend()
    plt.savefig(os.path.join(log_dir, 'so_inrd_fgsm_1.0_200_eps_10.0.png'), dpi=300, format='png',bbox_inches='tight')
    plt.close(fig)


def main(args):
    cur_dir = os.getcwd()
    log_dir = os.path.join(cur_dir, "logs", args.env_id, "exp_loss_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    episode_data = get_episode_data(args.model_dir, args.env_id, False, log_dir)
    episode_data_attacked = get_episode_data(args.model_dir, args.env_id, True, log_dir)
    plot(episode_data, episode_data_attacked, log_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDQN Testing Arguments")
    parser.add_argument("--env_id", type=str, required=True, help="Name of the Gymnasium environment (e.g., ALE/Boxing-v5)")
    parser.add_argument("--model_dir", type=str, required=True, help="model.zip directory")
    args = parser.parse_args()
    main(args)
