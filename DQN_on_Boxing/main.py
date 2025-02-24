import gymnasium as gym
from stable_baselines3 import DQN
import ale_py
import numpy as np
import torch
from torch.nn import functional as F
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    CombinedExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from gymnasium import spaces

#  Got from here https://www.gymlibrary.dev/environments/atari/boxing/
env = gym.make("BoxingNoFrameskip-v4")


# model = DQN("CnnPolicy", env, verbose=1)
# model.learn(total_timesteps=1e6, log_interval=4)
# model.save("boxing_v4_dqn")

def compute_policy_loss(model,state,next_state,rewards,done):
    next_q_values = model.q_net_target(next_state)
    # Follow greedy policy: use the one with the highest value
    next_q_values, _ = next_q_values.max(dim=1)
    # Avoid potential broadcast issue
    next_q_values = next_q_values.reshape(-1, 1)
    # 1-step TD target
    target_q_values = rewards + (1 - done) * model.gamma * next_q_values

    # Get current Q-values estimates
    current_q_values = model.q_net(state)
    # print(">>>>",current_q_values.shape)
    # print("===",action.shape)
    # Retrieve the q-values for the actions from the replay buffer
    current_q_values = torch.gather(current_q_values, dim=1, index=torch.tensor(np.arange(18)).unsqueeze(0))

    # Compute Huber loss (less sensitive to outliers)
    loss = F.smooth_l1_loss(current_q_values, target_q_values)
    return loss
    # return reward+model.gamma*model.q_net_target(state) - model.q_net(state)


# source: https://arxiv.org/pdf/2306.05873
def fo_inrd(model, state, next_state, reward,done, epsilon=0.01):
    # Sample a small random perturbation
    perturbation = torch.randn_like(state.float()) * epsilon
    
    # Perturb the state
    perturbed_state = state + perturbation
    
    # Compute the cost for the original state
    cost_original = compute_policy_loss(model, state, next_state,reward,done)
    
    # Compute the cost for the perturbed state
    cost_perturbed = compute_policy_loss(model, perturbed_state, next_state,reward, done)
    
    # Compute the difference in cost
    return cost_perturbed
    delta_cost = cost_perturbed - cost_original
    return delta_cost

def main():
    model = DQN.load("/deac/csc/vanbastelaerGrp/guptd23/RL_Project/AdversaryLoss/DQN_on_Boxing/boxing_v4_dqn.zip")
    done=False
    cum_reward=0
    obs, info = env.reset()
    # print(model.q_net(torch.tensor(obs)))
    # exit()
    # print(obs,info)
    while not done:
        action, _states = model.predict(obs, deterministic=False)
        # print(_states)
        next_obs, reward, terminated, truncated, info = env.step(action)
        # print(model.q_net(obs_tensor))
        # print(model.q_net_target(obs_tensor))
        # print(f"Compute Policy : {}")
        # exit()
        state = model.policy.obs_to_tensor(obs)[0]
        next_state = model.policy.obs_to_tensor(next_obs)[0]
        loss = compute_policy_loss(model,state, next_state, reward, terminated or truncated)
        print(loss)

        print("---",fo_inrd(model, state, next_state, reward, terminated or truncated,epsilon=10))
        cum_reward+=reward
        if terminated or truncated:
            done = True
        obs = next_obs
        


    print(f"Cum Reward: {cum_reward}")
    print(obs,info)

if __name__=="__main__":
    main()