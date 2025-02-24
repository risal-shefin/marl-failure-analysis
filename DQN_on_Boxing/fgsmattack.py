import gymnasium as gym
from stable_baselines3 import DQN
import ale_py
import numpy as np
import torch
from torch.nn import functional as F
from stable_baselines3.common.torch_layers import NatureCNN
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.ticker as ticker
import csv
# Create the Boxing environment
env = gym.make("BoxingNoFrameskip-v4")

def compute_policy_loss(model, state, next_state, rewards, done):
    next_q_values = model.q_net_target(next_state)
    # Greedy: pick the highest Q-value for each sample
    next_q_values, _ = next_q_values.max(dim=1)
    next_q_values = next_q_values.reshape(-1, 1)
    target_q_values = rewards + (1 - done) * model.gamma * next_q_values

    current_q_values = model.q_net(state)
    # Assuming 18 actions; adjust if needed.
    current_q_values = torch.gather(
        current_q_values, 
        dim=1, 
        index=torch.tensor(np.arange(18)).unsqueeze(0).to(current_q_values.device)
    )
    loss = F.smooth_l1_loss(current_q_values, target_q_values)
    return loss

def fgsm_attack(model, state, next_state, reward, done, epsilon=0.01):
    
    state = state.float()
    next_state = next_state.float()
    # Clone the input state and allow gradient computation
    state_adv = state.clone().detach().requires_grad_(True)
    loss = compute_policy_loss(model, state_adv, next_state, reward, done)
    model.q_net.zero_grad()
    loss.backward()
    # Compute perturbation: epsilon * sign(gradient)
    perturbation = epsilon * state_adv.grad.sign()
    perturbed_state = state_adv + perturbation
    return perturbed_state.detach()

def simulate_normal_trajectory(model, env):
    
    trajectory = []
    obs, info = env.reset()
    done = False
    while not done:
        # Save the current internal state of the environment
        env_state = env.unwrapped.ale.cloneState()
        action, _ = model.predict(obs, deterministic=True)
        state_tensor = model.policy.obs_to_tensor(obs)[0]
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state_tensor = model.policy.obs_to_tensor(next_obs)[0]
        loss = compute_policy_loss(model, state_tensor, next_state_tensor, reward, done)
        trajectory.append({
            'obs': obs,
            'action': action,
            'reward': reward,
            'done': done,
            'env_state': env_state,
            'loss': loss.item(),
            'next_obs': next_obs
        })
        obs = next_obs
    return trajectory

def simulate_adversarial_trajectory(model, env, trajectory, attack_idx, epsilon=0.01):

    # Copy the losses for steps before the attack
    adv_losses = [step['loss'] for step in trajectory[:attack_idx]]
    
    # At the attack point, restore the environment state
    attack_step = trajectory[attack_idx]
    env.unwrapped.ale.restoreState(attack_step['env_state'])
    
    # Prepare the attacked observation:
    # (we use the observation and next_obs from the normal trajectory at this step)
    attack_obs = attack_step['obs']
    attack_reward = attack_step['reward']
    attack_done = attack_step['done']
    attack_next_obs = attack_step['next_obs']
    
    attack_state_tensor = model.policy.obs_to_tensor(attack_obs)[0]
    attack_next_state_tensor = model.policy.obs_to_tensor(attack_next_obs)[0]
    
    # Compute the FGSM adversarial perturbation on the chosen state
    perturbed_state_tensor = fgsm_attack(model, attack_state_tensor, attack_next_state_tensor, attack_reward, attack_done, epsilon)
    
    # Convert the perturbed tensor back to the observation format.
    # Depending on your obs_to_tensor, you may need to adjust this.
    # Here we assume a batch dimension exists that we remove.
    perturbed_obs = perturbed_state_tensor.squeeze(0).cpu().numpy()
    
    # Use the perturbed observation as the starting point for the adversarial simulation.
    current_obs = perturbed_obs
    done = attack_done  # typically this will be False at the attack point
    while not done:
        action, _ = model.predict(current_obs, deterministic=True)
        state_tensor = model.policy.obs_to_tensor(current_obs)[0]
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        next_state_tensor = model.policy.obs_to_tensor(next_obs)[0]
        loss = compute_policy_loss(model, state_tensor, next_state_tensor, reward, done)
        adv_losses.append(loss.item())
        current_obs = next_obs
    return adv_losses

def plot_loss_trajectories(normal_losses, adv_losses, attack_idx,epsilon,result_dir):
    """Plot both the normal and adversarial loss trajectories along the steps."""
    plt.figure(figsize=(10, 6))
    x_normal = list(range(len(normal_losses)))
    x_adv = list(range(len(adv_losses)))
    plt.plot(x_normal, normal_losses, label="Normal Loss Trajectory", marker="o")
    plt.plot(x_adv, adv_losses, label="Adversarial Loss Trajectory", marker="x")
    plt.axvline(x=attack_idx, color='r', linestyle='--', label="Attack Point")
    plt.xlabel("Step Index")
    plt.ylabel("Loss")
    plt.title(f"Normal vs Adversarial Loss Trajectories [Deterministic] [Attack State : {attack_idx}]")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.savefig(f"{result_dir}/Krishan_First_Run_{attack_idx}_{epsilon}.jpeg")
    # plt.show()

# def plot_post_attack_loss(normal_losses, adv_losses, attack_idx,epsilon):
#     """
#     Plot the loss trajectories after the attack point on a log scale.

#     normal_losses: list of losses from the normal (attack-free) trajectory.
#     adv_losses: list of losses from the adversarial trajectory.
#     attack_idx: index at which the attack was performed.
#     """
#     # Extract losses from the attack point onward.
#     normal_post = normal_losses[attack_idx:]
#     adv_post = adv_losses[attack_idx:]
#     # Create a step index starting from the attack point.
#     steps = list(range(attack_idx, attack_idx + len(normal_post)))
    
#     plt.figure(figsize=(10, 6))
#     # Plot using semilogy for logarithmic scaling on the y-axis.
#     plt.semilogy(steps, normal_post, label="Normal Loss (Post-Attack)", linestyle="-", marker="o")
#     plt.semilogy(steps, adv_post, label="Adversarial Loss (Post-Attack)", linestyle="-", marker="x")
    
#     plt.xlabel("Step Index (Post-Attack)")
#     plt.ylabel("Loss (log scale)")
#     plt.title("Post-Attack Loss Trajectories (Log Scale)")
#     # Position legend below the x-axis.
#     plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
#     plt.tight_layout()
#     plt.savefig(f"Krishan_First_Run_post_scene_{attack_idx}_{epsilon}.jpeg")
#     # plt.show()


def plot_post_attack_loss(normal_losses, adv_losses, attack_idx, epsilon,result_dir):
    
    # Extract losses from the attack point onward.
    normal_post = normal_losses[attack_idx:]
    adv_post = adv_losses[attack_idx:]
    # Create a step index starting from the attack point.
    steps = list(range(attack_idx, attack_idx + len(normal_post)))
    
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Top subplot: Normal loss trajectory.
    axs[0].semilogy(steps, normal_post, label="Normal Loss (Post-Attack)", linestyle="-", marker="o", markersize=3)
    axs[0].set_ylabel("Loss (log scale)")
    axs[0].set_title("Normal Loss Trajectory (Post-Attack) {Deterministic}")
    axs[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.1))
    axs[0].grid(True)
    
    # Bottom subplot: Adversarial loss trajectory.
    axs[1].semilogy(steps, adv_post, label="Adversarial Loss (Post-Attack)", linestyle="-", marker="x", markersize=3)
    axs[1].set_xlabel("Episode (Step Index)")
    axs[1].set_ylabel("Loss (log scale)")
    axs[1].set_title("Adversarial Loss Trajectory (Post-Attack) {Deterministic}")
    axs[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.1))
    axs[1].grid(True)
    
    # To avoid overlapping of x-axis tick labels (for ~7000 episodes), limit the number of ticks.
    for ax in axs:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
        for label in ax.get_xticklabels():
            label.set_rotation(45)
    
    plt.tight_layout()
    plt.savefig(f"{result_dir}/Krishan_First_Run_post_scene_{attack_idx}_{epsilon}.jpeg")


def set_random_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Gymnasium reset now supports seed argument.
    env.reset(seed=seed)
    env.action_space.seed(seed)

def main():
    # seed taken so far --> 42,
    seed = 42
    set_random_seed(seed)
    result_dir="/deac/csc/vanbastelaerGrp/guptd23/RL_Project/AdversaryLoss/DQN_on_Boxing/Krishan_Deterministic_3336"
    # Load your pretrained DQN model
    model = DQN.load("/deac/csc/vanbastelaerGrp/guptd23/RL_Project/AdversaryLoss/DQN_on_Boxing/boxing_v4_dqn.zip")
    epsilons=[0.1,0.01,0.001,0.0001]
    normal_traj = simulate_normal_trajectory(model, env)
    normal_losses = [step['loss'] for step in normal_traj]
    total_normal_loss = sum(normal_losses)
    
    # Prepare a list to store the CSV rows.
    results = []
    results.append({"Status": "Normal", "Epsilon": 0, "Total Los": total_normal_loss})
    
    print("Normal Trajectory Done ...")
    # Choose a random step (except the very last) at which to perform the attack.
    attack_idx = np.random.randint(500, len(normal_traj) - 1)
    print(f"Attacking at step index: {attack_idx}")
    for epsilon in tqdm(epsilons):
        # First, simulate an attack-free (normal) episode
        
        # Now simulate the adversarial trajectory that forks at the attack index.
        adv_losses = simulate_adversarial_trajectory(model, env, normal_traj, attack_idx, epsilon=epsilon)
        total_adv_loss = sum(adv_losses)
        results.append({"Status": "FGSM", "Epsilon": epsilon, "Total Los": total_adv_loss})
        print("Adversarial Trajectory Done ...")
        # Plot the normal and adversarial loss trajectories.
        plot_loss_trajectories(normal_losses, adv_losses, attack_idx,epsilon,result_dir)
        # Plot only the post-attack losses on a log scale for better visualization.
        plot_post_attack_loss(normal_losses, adv_losses, attack_idx,epsilon,result_dir)
    # Save the results in CSV format.
    csv_filename = f"{result_dir}/loss_results_{attack_idx}.csv"
    with open(csv_filename, mode="w", newline="") as csv_file:
        fieldnames = ["Status", "Epsilon", "Total Los"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
if __name__ == "__main__":
    main()
