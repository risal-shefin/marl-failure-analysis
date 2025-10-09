import argparse
import torch
import time
import imageio
import numpy as np
from pathlib import Path
from torch.autograd import Variable
from utils.make_env import make_env
from algorithms.maddpg import MADDPG
import os
from datetime import datetime
from utils.smac_wrapper import SmacWrapper
from utils.misc import gumbel_softmax
import pettingzoo.mpe as mpe
import pettingzoo.sisl as sisl
import pettingzoo.atari as atari
import matplotlib.pyplot as plt
from PIL import Image
from collections import deque
import supersuit
import csv
import math
from tqdm import tqdm
from matplotlib.patches import Patch

USE_CUDA = torch.cuda.is_available()
DEVICE = 'gpu' if USE_CUDA else 'cpu'
torch_device = torch.device("cuda" if USE_CUDA else "cpu")
K_SIGMA = 1.4

def create_environment(config):
    """Create a fresh SMAC environment instance"""
    env = SmacWrapper.make_env(config.map_name, seed=config.seed)
    return env

def fgsm_attack(maddpg, obs, actions, attacked_agent_id, epsilon):
    # Convert to tensors with gradient tracking
    torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    torch_actions = [Variable(torch.Tensor([actions[i]]).to(torch_device), requires_grad=False) for i in range(maddpg.nagents)]
    # Concatenate for critic input
    vf_in = torch.cat((*torch_obs, *torch_actions), dim=1)
    # Loss to maximize (degrade agent performance)
    loss = -(maddpg.agents[attacked_agent_id].critic(vf_in)).mean()  # Negative to maximize via gradient ascent
    # Compute gradient
    grad = torch.autograd.grad(loss, torch_obs[attacked_agent_id], retain_graph=True)[0]
    # FGSM perturbation: move in direction of gradient sign
    perturbation = epsilon * grad.sign()
    # Apply perturbation element-wise
    obs_perturbed = obs[attacked_agent_id] + perturbation.squeeze().cpu().numpy()
    return obs_perturbed

def get_agent_colors(n_agents):
    """
    Get consistent color palette for agents across all plots.
    
    Args:
        n_agents: Number of agents
        
    Returns:
        dict: Dictionary mapping agent index to color
    """
    # cmap = plt.get_cmap('tab20')
    cmap = plt.get_cmap('Set1')
    agent_colors = {i: cmap(i % 20) for i in range(n_agents)}
    return agent_colors

def compute_taylor_delta_policy(maddpg, obs, actions, action_spaces, epsilon):
    # Convert discrete actions to one-hot encoding
    if maddpg.discrete_action:
        one_hot_actions = []
        for i, action in enumerate(actions):
            one_hot = np.zeros(action_spaces[i].n)
            one_hot[action] = 1.0
            one_hot_actions.append(one_hot)
        actions = one_hot_actions

    torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    actions = [Variable(torch.Tensor([actions[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    vf_in = torch.cat((*torch_obs, *actions), dim=1)

    delta_errors = []

    for i, agent_i in enumerate(maddpg.agents):
        action_logits_i = agent_i.policy(torch_obs[i])
        action_log_probs = torch.log_softmax(action_logits_i, dim=-1)
        max_action_idx = torch.argmax(action_log_probs, dim=-1)
        critic_val = action_log_probs.gather(-1, max_action_idx.unsqueeze(-1)).squeeze()
        grad_i = torch.autograd.grad(critic_val, torch_obs[i], create_graph=True, retain_graph=True)[0]
        # grad_i = torch.autograd.grad(critic_val, actions[i], create_graph=True, retain_graph=True)[0]

        eta_i = epsilon * grad_i.sign() / torch.max(grad_i.norm(p=2), torch.tensor(1e-6))
        
        # Second-order Taylor approximation: f(x + η) ≈ f(x) + ∇f(x)^T η + 0.5 η^T H η
        j_tilde = critic_val + torch.dot(grad_i.flatten(), eta_i.flatten())# + 0.5 * torch.dot(eta_i.flatten(), hvp.flatten())
        p_torch_obs_i = torch_obs[i] + eta_i
        p_action_logits_i = agent_i.policy(p_torch_obs_i)
        p_action_log_probs = torch.log_softmax(p_action_logits_i, dim=-1)
        p_max_action_idx = torch.argmax(p_action_log_probs, dim=-1)
        j_perturbed = p_action_log_probs.gather(-1, p_max_action_idx.unsqueeze(-1)).squeeze()
        delta_error = abs(j_perturbed - j_tilde).item()
        delta_errors.append(delta_error)

    return delta_errors

def compute_frob_norms(maddpg, obs, actions, action_spaces, vulnerable_agent_id):
    # if not maddpg.discrete_action:
    #     raise NotImplementedError("This function is only implemented for discrete action spaces.")
    
    # Convert discrete actions to one-hot encoding
    if maddpg.discrete_action:
        one_hot_actions = []
        for i, action in enumerate(actions):
            one_hot = np.zeros(action_spaces[i].n)
            one_hot[action] = 1.0
            one_hot_actions.append(one_hot)
        actions = one_hot_actions

    torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    actions = [Variable(torch.Tensor([actions[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    vf_in = torch.cat((*torch_obs, *actions), dim=1)
    
    results = []

    for i, agent_i in enumerate(maddpg.agents):
        critic_val = agent_i.critic(vf_in).mean()
        grad_i = torch.autograd.grad(critic_val, torch_obs[i], create_graph=True, retain_graph=True)[0]

        # Compute Hessian matrix
        hessian_matrix = []
        for k in range(grad_i.shape[1]):
            # Compute ∂²Q/∂obs_i[k]∂obs_j
            second_grad = torch.autograd.grad(
                grad_i[0, k], 
                torch_obs[vulnerable_agent_id], 
                retain_graph=True, 
                allow_unused=True
            )[0]
            
            hessian_matrix.append(second_grad.flatten())

        H = torch.stack(hessian_matrix)
        hessian_frob_norm = torch.norm(H, p='fro')
        results.append(hessian_frob_norm.item())

    return results


def compute_pairwise_frob_norms(maddpg, obs, actions, action_spaces):
    """
    Compute Frobenius norms of cross-agent Hessian blocks for all (i, j).
    Returns an N x N list where entry [i][j] approximates || \partial^2 v_i / (\partial obs_i \partial obs_j) ||_F.
    """
    # Convert discrete actions to one-hot encoding
    if maddpg.discrete_action:
        one_hot_actions = []
        for i, action in enumerate(actions):
            one_hot = np.zeros(action_spaces[i].n)
            one_hot[action] = 1.0
            one_hot_actions.append(one_hot)
        actions = one_hot_actions

    torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    actions = [Variable(torch.Tensor([actions[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    vf_in = torch.cat((*torch_obs, *actions), dim=1)

    N = maddpg.nagents
    results = [[0.0 for _ in range(N)] for _ in range(N)]

    for i in range(N):
        critic_val = maddpg.agents[i].critic(vf_in).mean()
        grad_i = torch.autograd.grad(critic_val, torch_obs[i], create_graph=True, retain_graph=True)[0]
        # grad_i = torch.autograd.grad(critic_val, actions[i], create_graph=True, retain_graph=True)[0]

        for j in range(N):
            hessian_matrix = []
            for k in range(grad_i.shape[1]):
                second_grad = torch.autograd.grad(
                    grad_i[0, k],
                    torch_obs[j],
                    # actions[j],  # Change to actions[j] to compute cross-agent action Hessian
                    retain_graph=True,
                    allow_unused=True
                )[0]
                if second_grad is None:
                    second_grad = torch.zeros_like(torch_obs[j])
                hessian_matrix.append(second_grad.flatten())

            H = torch.stack(hessian_matrix) if len(hessian_matrix) > 0 else torch.zeros(1, 1)
            results[i][j] = H.norm(p='fro').item()

    return results


def compute_pairwise_action_influences(maddpg, obs, actions, action_spaces):
    """
    Compute direct influence of each agent's action on every other agent's Q-value.
    Returns an N x N list where entry [i][j] represents || ∂Q_i/∂a_j ||_2.
    This measures how much agent j's action directly influences agent i's Q-value.
    """
    # Convert discrete actions to one-hot encoding
    if maddpg.discrete_action:
        one_hot_actions = []
        for i, action in enumerate(actions):
            one_hot = np.zeros(action_spaces[i].n)
            one_hot[action] = 1.0
            one_hot_actions.append(one_hot)
        actions = one_hot_actions

    torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=False) for i in range(maddpg.nagents)]
    torch_actions = [Variable(torch.Tensor([actions[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    vf_in = torch.cat((*torch_obs, *torch_actions), dim=1)

    N = maddpg.nagents
    results = [[0.0 for _ in range(N)] for _ in range(N)]

    for i in range(N):
        critic_val = maddpg.agents[i].critic(vf_in).mean()
        
        for j in range(N):
            # Compute gradient of Q_i with respect to action of agent j
            grad_qi_aj = torch.autograd.grad(
                critic_val,
                torch_actions[j],
                retain_graph=True,
                allow_unused=True
            )[0]

            # Compute L2 norm of the gradient (direct influence magnitude)
            influence_magnitude = grad_qi_aj.norm(p=2).item()
            results[i][j] = influence_magnitude

    return results


def compute_second_order_action_influences(maddpg, obs, actions, action_spaces):
    """
    Compute second-order action influences between agents.
    Returns an N x N list where entry [i][j] represents || ∂²Q_i/(∂a_j)² ||_F.
    This measures the curvature/sensitivity of agent i's Q-value with respect to agent j's action.
    """
    # Convert discrete actions to one-hot encoding
    if maddpg.discrete_action:
        one_hot_actions = []
        for i, action in enumerate(actions):
            one_hot = np.zeros(action_spaces[i].n)
            one_hot[action] = 1.0
            one_hot_actions.append(one_hot)
        actions = one_hot_actions

    torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=False) for i in range(maddpg.nagents)]
    torch_actions = [Variable(torch.Tensor([actions[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    vf_in = torch.cat((*torch_obs, *torch_actions), dim=1)

    N = maddpg.nagents
    results = [[0.0 for _ in range(N)] for _ in range(N)]

    for i in range(N):
        critic_val = maddpg.agents[i].critic(vf_in).mean()
        
        for j in range(N):
            # Compute first-order gradient ∂Q_i/∂a_j
            grad_qi_aj = torch.autograd.grad(
                critic_val,
                torch_actions[j],
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]
            
            if grad_qi_aj is None:
                continue
                
            # Compute second-order gradient ∂²Q_i/(∂a_j)²
            hessian_matrix = []
            for k in range(grad_qi_aj.shape[1]):  # iterate over action dimensions
                second_grad = torch.autograd.grad(
                    grad_qi_aj[0, k],
                    torch_actions[j],  # Same action variable j
                    retain_graph=True,
                    allow_unused=True
                )[0]
                
                if second_grad is None:
                    second_grad = torch.zeros_like(torch_actions[j])
                hessian_matrix.append(second_grad.flatten())
            
            if len(hessian_matrix) > 0:
                H = torch.stack(hessian_matrix)
                # Compute Frobenius norm of the Hessian matrix
                second_order_influence = H.norm(p='fro').item()
                results[i][j] = second_order_influence

    return results


def compute_pairwise_observation_influences(maddpg, obs, actions, action_spaces):
    """
    Compute direct influence of each agent's observation on every other agent's Q-value.
    Returns an N x N list where entry [i][j] represents || ∂Q_i/∂obs_j ||_2.
    This measures how much agent j's observation directly influences agent i's Q-value.
    """
    # Convert discrete actions to one-hot encoding
    if maddpg.discrete_action:
        one_hot_actions = []
        for i, action in enumerate(actions):
            one_hot = np.zeros(action_spaces[i].n)
            one_hot[action] = 1.0
            one_hot_actions.append(one_hot)
        actions = one_hot_actions

    torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    torch_actions = [Variable(torch.Tensor([actions[i]]).to(torch_device), requires_grad=False) for i in range(maddpg.nagents)]
    vf_in = torch.cat((*torch_obs, *torch_actions), dim=1)

    N = maddpg.nagents
    results = [[0.0 for _ in range(N)] for _ in range(N)]

    for i in range(N):
        critic_val = maddpg.agents[i].critic(vf_in).mean()
        
        for j in range(N):
            # Compute gradient of Q_i with respect to observation of agent j
            grad_qi_obsj = torch.autograd.grad(
                critic_val,
                torch_obs[j],
                retain_graph=True,
                allow_unused=True
            )[0]

            # Compute L2 norm of the gradient (direct influence magnitude)
            influence_magnitude = grad_qi_obsj.norm(p=2).item()
            results[i][j] = influence_magnitude

    return results


def compute_second_order_observation_influences(maddpg, obs, actions, action_spaces):
    """
    Compute second-order observation influences between agents.
    Returns an N x N list where entry [i][j] represents || ∂²Q_i/(∂obs_j)² ||_F.
    This measures the curvature/sensitivity of agent i's Q-value with respect to agent j's observation.
    """
    # Convert discrete actions to one-hot encoding
    if maddpg.discrete_action:
        one_hot_actions = []
        for i, action in enumerate(actions):
            one_hot = np.zeros(action_spaces[i].n)
            one_hot[action] = 1.0
            one_hot_actions.append(one_hot)
        actions = one_hot_actions

    torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    torch_actions = [Variable(torch.Tensor([actions[i]]).to(torch_device), requires_grad=False) for i in range(maddpg.nagents)]
    vf_in = torch.cat((*torch_obs, *torch_actions), dim=1)

    N = maddpg.nagents
    results = [[0.0 for _ in range(N)] for _ in range(N)]

    for i in range(N):
        critic_val = maddpg.agents[i].critic(vf_in).mean()
        
        for j in range(N):
            # Compute first-order gradient ∂Q_i/∂obs_j
            grad_qi_obsj = torch.autograd.grad(
                critic_val,
                torch_obs[j],
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]
            
            if grad_qi_obsj is None:
                continue
                
            # Compute second-order gradient ∂²Q_i/(∂obs_j)²
            hessian_matrix = []
            for k in range(grad_qi_obsj.shape[1]):  # iterate over observation dimensions
                second_grad = torch.autograd.grad(
                    grad_qi_obsj[0, k],
                    torch_obs[j],  # same observation variable for pure second derivative
                    retain_graph=True,
                    allow_unused=True
                )[0]
                
                if second_grad is None:
                    second_grad = torch.zeros_like(grad_qi_obsj[0])
                hessian_matrix.append(second_grad.flatten())
            
            if len(hessian_matrix) > 0:
                H = torch.stack(hessian_matrix)
                # Compute Frobenius norm of the Hessian matrix
                second_order_influence = H.norm(p='fro').item()
                results[i][j] = second_order_influence

    return results


# second order directional derivative
def compute_2nd_ord_dir_derivatives(maddpg, obs, actions, action_spaces, vulnerable_agent_id):
    # if not maddpg.discrete_action:
    #     raise NotImplementedError("This function is only implemented for discrete action spaces.")
    
    # Convert discrete actions to one-hot encoding
    if maddpg.discrete_action:
        one_hot_actions = []
        for i, action in enumerate(actions):
            one_hot = np.zeros(action_spaces[i].n)
            one_hot[action] = 1.0
            one_hot_actions.append(one_hot)
        actions = one_hot_actions

    torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    actions = [Variable(torch.Tensor([actions[i]]).to(torch_device), requires_grad=True) for i in range(maddpg.nagents)]
    vf_in = torch.cat((*torch_obs, *actions), dim=1)
    
    results = []

    for i, agent_i in enumerate(maddpg.agents):
        critic_val = agent_i.critic(vf_in).mean()
        grad_i = torch.autograd.grad(critic_val, torch_obs[i], create_graph=True, retain_graph=True)[0]
        v = grad_i / torch.max(grad_i.norm(p=2), torch.tensor(1e-6))

        # Compute Hessian-vector product (HVP) of grad_i and v with respect to torch_obs[j]
        hvp = torch.autograd.grad(
            outputs=grad_i,
            inputs=torch_obs[vulnerable_agent_id],
            grad_outputs=v,
            retain_graph=True,
            allow_unused=True
        )[0]

        # Compute u^T * H * v (quadratic form)
        grad_j = torch.autograd.grad(-critic_val, torch_obs[vulnerable_agent_id], create_graph=True, retain_graph=True)[0]
        u = grad_j / torch.max(grad_j.norm(p=2), torch.tensor(1e-6))
        curvature_val = torch.dot(u.flatten(), hvp.flatten())
        results.append(curvature_val.item())

    return results


def get_episode_data(env, maddpg, config, logdir, ref_vals, ref_std_devs, detection_method='mean_std', do_attack=False, atk_agent_id=-1, seed=None):
    obs, action_masks = env.reset()
    # obs = env.reset(seed=12345) # better for speaker_listener_v3
    episode_reward = 0
    episode_rewards = [0 for _ in range(maddpg.nagents)]
    frames = []
    # initialize deque buffers for last batch_size observations
    result_deques = [deque(maxlen=5) for _ in range(maddpg.nagents)]
    frob_norms_deques = [deque(maxlen=1) for _ in range(maddpg.nagents)]
    sec_dir_derivatives_deques = [deque(maxlen=1) for _ in range(maddpg.nagents)]
    metric_vals = []
    cnt = 0
    vulnerable_agent_id = None
    attacked_steps = []
    frob_norms_list = []
    sec_dir_derivatives = []
    action_influences_matrix_history = []  # list over timesteps of N x N action influence matrices
    second_order_action_influences_history = []  # list over timesteps of N x N second-order action influence matrices (∂²Q_i/(∂a_j)²)
    observation_influences_matrix_history = []  # list over timesteps of N x N observation influence matrices
    second_order_observation_influences_history = []  # list over timesteps of N x N second-order observation influence matrices (∂²Q_i/(∂obs_j)²)
    do_start_attack = False
    attack_step_remaining = 5

    # Fault detection tracking
    fault_first_detected = {}  # agent_id -> timestep first detected
    fault_timeline = []  # list of dicts: {agent, t, contribs: {f: c}}
    frob_norms_matrix_history = []  # list over timesteps of N x N frob norm matrices

    prev_errors = [0 for i in range(maddpg.nagents)]

    while True:
        # add Gaussian noise to an agent's observation
        # noise_scale = 0.0  # adjust the standard deviation of the noise as needed
        # obs[attacked_agent] = obs[attacked_agent] + np.random.randn(*obs[attacked_agent].shape) * noise_scale

        # FGSM attack
        if do_attack and cnt >= config.atk_start_step and cnt <= config.atk_end_step and False:
            temp_torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=False) for i in range(maddpg.nagents)]
            temp_torch_agent_actions = maddpg.step(temp_torch_obs, explore=False, action_masks=torch_masks)
            agent_actions = [ac.data.cpu().numpy() for ac in temp_torch_agent_actions]
            temp_actions = [agent_actions[i].squeeze() for i, agent_name in enumerate(env.possible_agents)]
            obs[atk_agent_id] = fgsm_attack(maddpg, obs, temp_actions, atk_agent_id, 0.1)
        
        torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=False) for i in range(maddpg.nagents)]
        torch_masks = [Variable(torch.Tensor(action_masks[i]).to(torch_device), requires_grad=False)
                       if action_masks[i] is not None else None for i in range(maddpg.nagents)]
        torch_agent_actions = maddpg.step(torch_obs, explore=False, action_masks=torch_masks)
        agent_actions = [ac.data.cpu().numpy() for ac in torch_agent_actions]
        if maddpg.discrete_action:
            actions = {agent_name: agent_actions[i].argmax() for i, agent_name in enumerate(env.possible_agents)}
        else:
            actions = {agent_name: agent_actions[i].squeeze() for i, agent_name in enumerate(env.possible_agents)}

        # random attack
        if do_attack and False:
            actions[env.possible_agents[atk_agent_id]] = env.action_spaces[env.possible_agents[atk_agent_id]].sample()
        
        # Action Space Attacks
        if maddpg.discrete_action:
            # Compute entropy of action logits
            action_logits = maddpg.get_action_logits(torch_obs, torch_masks)
            atk_agent_action_probs = torch.softmax(action_logits[atk_agent_id].squeeze(), dim=0)
            atk_agent_log_probs = torch.log(atk_agent_action_probs)
            atk_agent_entropy = -torch.sum(atk_agent_action_probs * atk_agent_log_probs)
            if do_attack and cnt >= 20:
                do_start_attack = True
            # worst action attack for discrete action space
            # if do_attack and np.random.rand() < 0.75:
            # if do_attack and cnt >= config.atk_start_step and cnt <= config.atk_end_step:
            if do_start_attack and attack_step_remaining > 0:
                action_logits = maddpg.get_action_logits(torch_obs, torch_masks)

                # Worst action attack
                # Apply action mask - set invalid actions to very high values so they won't be selected as minimum
                masked_logits = action_logits[atk_agent_id].clone().squeeze()
                masked_logits[action_masks[atk_agent_id] == 0] = float('inf')
                actions[env.possible_agents[atk_agent_id]] = torch.argmin(masked_logits).item()
                attacked_steps.append(cnt)
                attack_step_remaining -= 1
        else:
            if do_attack and cnt >= 5:
                do_start_attack = True
            # random action attack
            if do_start_attack and attack_step_remaining > 0:
                # actions[env.possible_agents[atk_agent_id]] = env.action_spaces[env.possible_agents[atk_agent_id]].sample()
                # attacked_steps.append(cnt)
                attack_step_remaining -= 1

        if config.save_gifs:
            frames.append(Image.fromarray(env.render()))
        
        # results = compute_taylor_delta(maddpg, obs, list(actions.values()), env.action_space, 0.1)
        results = compute_taylor_delta_policy(maddpg, obs, list(actions.values()), env.action_space, 0.01)
        # results = compute_eigen(maddpg, obs, list(actions.values()), env.action_space, 0.1)
        results_frob_norms = compute_frob_norms(maddpg, obs, list(actions.values()), env.action_space, atk_agent_id)
        # Pairwise Frobenius norms across all agent pairs for cascading impact analysis
        pairwise_frobs = compute_pairwise_frob_norms(maddpg, obs, list(actions.values()), env.action_space)
        frob_norms_matrix_history.append(pairwise_frobs)
        # Pairwise action influences: direct impact of each agent's action on others' Q-values
        pairwise_action_influences = compute_pairwise_action_influences(maddpg, obs, list(actions.values()), env.action_space)
        action_influences_matrix_history.append(pairwise_action_influences)
        # Second-order action influences: how action influences change with respect to other actions
        second_order_action_influences = compute_second_order_action_influences(maddpg, obs, list(actions.values()), env.action_space)
        second_order_action_influences_history.append(second_order_action_influences)
        # Pairwise observation influences: direct impact of each agent's observation on others' Q-values
        pairwise_observation_influences = compute_pairwise_observation_influences(maddpg, obs, list(actions.values()), env.action_space)
        observation_influences_matrix_history.append(pairwise_observation_influences)
        # Second-order observation influences: how observation influences change with respect to other observations
        second_order_observation_influences = compute_second_order_observation_influences(maddpg, obs, list(actions.values()), env.action_space)
        second_order_observation_influences_history.append(second_order_observation_influences)
        results_sec_dir_derivatives = compute_2nd_ord_dir_derivatives(maddpg, obs, list(actions.values()), env.action_space, atk_agent_id)

        for i in range(maddpg.nagents):
            if cnt >= len(ref_vals[i]):
                continue  # skip detection if beyond reference data length
            
            result_deques[i].append(results[i])
            
            # Apply different detection methods
            if detection_method == 'mean_std':
                detection_value = np.mean(result_deques[i])
                threshold_exceeded = abs(detection_value - ref_vals[i][cnt]) > K_SIGMA * ref_std_devs[i][cnt]
                # threshold_exceeded = abs(detection_value - ref_vals[i][cnt]) > 0.01
            elif detection_method == 'median_mad':
                detection_value = np.mean(result_deques[i])
                threshold_exceeded = abs(detection_value - ref_vals[i][cnt]) > K_SIGMA * ref_std_devs[i][cnt]
            elif detection_method == 'diff':
                if cnt > 0:
                    current_diff = results[i] - prev_errors[i]
                    threshold_exceeded = abs(current_diff - ref_vals[i][cnt]) > K_SIGMA * ref_std_devs[i][cnt]
                    detection_value = current_diff
                else:
                    threshold_exceeded = False
                    detection_value = 0.0
            else:
                raise ValueError(f"Unknown detection method: {detection_method}")
            
            if threshold_exceeded:
                if i not in fault_first_detected:
                    print(f" [!!!] Anomaly detected for agent {i} at timestep: {cnt}. Method: {detection_method}. Value: {detection_value:.6f}")
                    fault_first_detected[i] = cnt
                    # Cascading Impact Analysis
                    prev_faults = [(f, tf) for f, tf in fault_first_detected.items() if f != i and tf < cnt]
                    contribs = {}
                    if len(prev_faults) > 0:
                        for f, tf in prev_faults:
                            # Mean Frobenius norm from t_f to current t for H_{i,f}
                            # values_over_time = [frob_norms_matrix_history[tau][i][f] for tau in range(tf, cnt + 1) if tau < len(frob_norms_matrix_history)]
                            # if len(values_over_time) > 0:
                            #     contribs[f] = float(np.mean(values_over_time))
                            for agent in range(maddpg.nagents):
                                first_fault_ts = prev_faults[0][1]
                                values_over_time = [frob_norms_matrix_history[tau][i][agent] for tau in range(first_fault_ts, cnt + 1) if tau < len(frob_norms_matrix_history)]
                                if len(values_over_time) > 0:
                                    contribs[agent] = float(np.mean(values_over_time))
                        if len(contribs) > 0:
                            ranked = sorted(contribs.items(), key=lambda x: x[1], reverse=True)
                            print(f"     >> Potential contributors to fault in agent {i} (mean ||H_{{i,f}}||_F from t_f to {cnt}): {ranked}")
                    fault_timeline.append({
                        'agent': i,
                        't': cnt,
                        'contribs': contribs
                    })
            frob_norms_deques[i].append(results_frob_norms[i])
            sec_dir_derivatives_deques[i].append(results_sec_dir_derivatives[i])

        metric_vals.append([np.mean(result_deques[i]) for i in range(maddpg.nagents)])
        prev_errors = results
        frob_norms_list.append([np.mean(frob_norms_deques[i]) for i in range(maddpg.nagents)])
        sec_dir_derivatives.append([np.mean(sec_dir_derivatives_deques[i]) for i in range(maddpg.nagents)])

        next_obs, rewards, dones, infos, action_masks = env.step(actions)
        episode_reward += np.sum([rewards[:,i] if env.agent_types[i] != 'adversary' else np.zeros_like(rewards[:,i]) for i in range(maddpg.nagents)])
        episode_rewards = [episode_rewards[i] + rewards[:,i].sum() for i in range(maddpg.nagents)]

        obs = next_obs
        cnt += 1
        if dones.all():
            break

    print(f"Episode reward: {episode_reward}")
    print(f"Episode rewards: {episode_rewards}")
    if config.save_gifs:
        imageio.mimsave(os.path.join(logdir, f'{config.map_name}_episode_atk_{atk_agent_id if do_attack else "free"}.gif'), frames, duration=125)
        print(f"Saved gif of episode to {logdir}")
    print("")
    return metric_vals, attacked_steps, frob_norms_list, sec_dir_derivatives, frob_norms_matrix_history, fault_timeline, action_influences_matrix_history, second_order_action_influences_history, observation_influences_matrix_history, second_order_observation_influences_history


def plot_results(results_attacked, attacked_steps, atk_agent_id, ref_vals, ref_std_devs, logdir, detection_method='mean_std'):
    n = len(results_attacked[0])  # number of agents
    t = len(results_attacked)     # number of time steps
    
    # Create n subplots in a row
    max_per_row = 3
    rows = math.ceil(n / max_per_row)
    cols = min(n, max_per_row)
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten()  # so you can index axes[i] easily
    fig.suptitle(f'Taylor Error ({detection_method.upper().replace("_", "+")} | Worst Action Attack | Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.95)
    
    # Ensure axes is iterable even for single agent case
    if n == 1:
        axes = [axes]
    
    for i in range(n):
        ax = axes[i]
        
        # Extract time series for agent i
        attacked_series = [results_attacked[t][i] for t in range(len(results_attacked))]
        
        # For 'diff' detection method, plot the differences instead of raw values
        if detection_method == 'diff':
            # Calculate differences for plotting (skip first timestep as it has no previous value)
            diff_series = []
            for t in range(1, len(attacked_series)):
                diff = attacked_series[t] - attacked_series[t-1]
                diff_series.append(diff)
            
            # Update series to plot differences
            attacked_series = diff_series
            steps_length = min(len(attacked_series), len(ref_vals[i]))
            steps = range(1, steps_length + 1)  # Start from timestep 1
        else:
            # Plot the curves normally
            steps_length = min(len(attacked_series), len(ref_vals[i]))
            steps = range(steps_length)
        ref_vals[i] = ref_vals[i][:steps_length]
        ref_std_devs[i] = ref_std_devs[i][:steps_length]
        attacked_series = attacked_series[:steps_length]

        # Add green region using ref_vals and ref_std_devs
        ref_lower = [ref_vals[i][t] - K_SIGMA*ref_std_devs[i][t] for t in range(len(ref_vals[i]))]
        ref_upper = [ref_vals[i][t] + K_SIGMA*ref_std_devs[i][t] for t in range(len(ref_vals[i]))]
        ax.fill_between(steps, ref_lower, ref_upper, alpha=0.1, color='green')
        
        ax.plot(steps, attacked_series, 'r-', label='Observed', linewidth=2)
        ax.plot(steps, ref_vals[i], 'g--', label='Reference', linewidth=2)
        
        # Mark attacked timesteps with vertical lines
        if i == atk_agent_id and attacked_steps:
            # for attack_step in attacked_steps:
            #     ax.axvline(x=attack_step, color='orange', linestyle=':', alpha=0.5, linewidth=1.5)
            # # Add legend entry for attack markers
            # ax.axvline(x=attacked_steps[0], color='orange', linestyle=':', alpha=0.5, linewidth=1.5, label='Attacked Steps')
            start = min(attacked_steps)
            end = max(attacked_steps)
            ax.axvspan(start, end, color='red', alpha=0.1, label='Attacked Region')
        
        ax.set_xlabel('Step')
        if detection_method == 'diff':
            ax.set_ylabel('Taylor Error Difference')
        else:
            ax.set_ylabel('Taylor Delta Error')
        ax.set_title(f'Agent {i}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # hide the unused axes
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(logdir, f'plot_analysis_{detection_method}_attacked_{atk_agent_id}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved analysis plot to {logdir}")


def plot_frobs(frobs_normal, frobs_atk, attacked_steps, atk_agent_id, logdir):
    n = len(frobs_normal[0])  # number of agents
    t = len(frobs_normal)     # number of time steps
    
    # Create n subplots in a row
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    fig.suptitle(f'Frobenius Norms (Worst Action Attack | Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.95)
    
    # Ensure axes is iterable even for single agent case
    if n == 1:
        axes = [axes]
    
    for i in range(n):
        ax = axes[i]
        
        # Extract time series for agent i
        normal_series = [frobs_normal[t][i] for t in range(len(frobs_normal))]
        attacked_series = [frobs_atk[t][i] for t in range(len(frobs_atk))]
        
        # Plot the curves
        normal_steps = range(len(normal_series))
        attacked_steps = range(len(attacked_series))
        ax.plot(attacked_steps, attacked_series, 'r-', label='Under Attack', linewidth=2)
        ax.plot(normal_steps, normal_series, 'g-', label='Normal', linewidth=2)

        # Mark attacked timesteps with vertical lines
        if attacked_steps:
            for attack_step in attacked_steps:
                ax.axvline(x=attack_step, color='orange', linestyle=':', alpha=0.5, linewidth=1.5)
            # Add legend entry for attack markers
            ax.axvline(x=attacked_steps[0], color='orange', linestyle=':', alpha=0.5, linewidth=1.5, label='Attacked Steps')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Frobenius Norm')
        ax.set_title(f'Agent {i}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(logdir, f'plot_frobs_attacked_{atk_agent_id}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved frobenius norms plot to {logdir}")


def plot_sec_dir_derivatives(s_dir_derv_normal, s_dir_derv_atk, attacked_steps, atk_agent_id, logdir):
    n = len(s_dir_derv_normal[0])  # number of agents
    t = len(s_dir_derv_normal)     # number of time steps
    
    # Create n subplots in a row
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    fig.suptitle(f'2nd Ord. Dir. Derivatives (Worst Action Attack | Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.95)
    
    # Ensure axes is iterable even for single agent case
    if n == 1:
        axes = [axes]
    
    for i in range(n):
        ax = axes[i]
        
        # Extract time series for agent i
        normal_series = [s_dir_derv_normal[t][i] for t in range(len(s_dir_derv_normal))]
        attacked_series = [s_dir_derv_atk[t][i] for t in range(len(s_dir_derv_atk))]
        
        # Plot the curves
        normal_steps = range(len(normal_series))
        attacked_steps = range(len(attacked_series))

        ax.plot(normal_steps, normal_series, 'g-', label='Normal', linewidth=2)
        ax.plot(attacked_steps, attacked_series, 'r-', label='Under Attack', linewidth=2)
        
        # Highlight region under y < 0 in red
        y_min = min(min(normal_series), min(attacked_series))
        if y_min < 0:
            ax.axhspan(y_min * 1.1, 0, alpha=0.2, color='red')
        
        # Mark attacked timesteps with vertical lines
        if attacked_steps:
            for attack_step in attacked_steps:
                ax.axvline(x=attack_step, color='orange', linestyle=':', alpha=0.5, linewidth=1.5)
            # Add legend entry for attack markers
            ax.axvline(x=attacked_steps[0], color='orange', linestyle=':', alpha=0.5, linewidth=1.5, label='Attacked Steps')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('2nd Ord. Dir. Derivative')
        ax.set_title(f'Agent {i}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(logdir, f'plot_sec_dir_derivatives_attacked_{atk_agent_id}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved 2nd ord. dir. derivatives plot to {logdir}")


def plot_fault_timeline(fault_timeline, total_agents, logdir):
    if len(fault_timeline) == 0:
        print("No faults detected; skipping fault timeline plot.")
        return

    k = len(fault_timeline)
    fig = plt.figure(figsize=(max(6, 3*k), 5))  # reduce height from 6 → 5
    gs = fig.add_gridspec(
        3, k,
        height_ratios=[0.8, 2, 0.1],  # smaller top & bottom rows
        hspace=0.1  # tighter vertical spacing
    )

    cmap = plt.get_cmap('tab20')
    agent_colors = {i: cmap(i % 20) for i in range(total_agents)}

    # --- Timeline axis (top row) ---
    ax_timeline = fig.add_subplot(gs[0, :])
    ax_timeline.axis('off')

    # Horizontal arrow for timeline
    arrow_y = 0.5
    ax_timeline.annotate(
        '', xy=(1, arrow_y), xytext=(0, arrow_y),
        arrowprops=dict(arrowstyle='-|>', lw=2, color='gray'),
        xycoords='axes fraction', textcoords='axes fraction'
    )

    # Milestones
    for i, event in enumerate(fault_timeline):
        frac_x = (i + 0.5) / k  # evenly spaced

        # Circle marker
        ax_timeline.plot(frac_x, arrow_y, 'o', color='darkred', markersize=10, transform=ax_timeline.transAxes)

        # Faulty agent label above
        ax_timeline.text(frac_x, arrow_y + 0.15,
                         f"Faulty agent {event['agent']}",
                         ha='center', va='bottom',
                         fontsize=11, fontweight='bold',
                         transform=ax_timeline.transAxes)

        # Timestep label below
        ax_timeline.text(frac_x, arrow_y - 0.15,
                         f"t = {event['t']}",
                         ha='center', va='top',
                         fontsize=10, color='darkblue',
                         transform=ax_timeline.transAxes)

    # --- Contributor charts (middle row) ---
    for col, event in enumerate(fault_timeline):
        ax = fig.add_subplot(gs[1, col])
        contribs = event.get('contribs', {})

        if len(contribs) == 0:
            ax.axis('off')
            ax.text(0.5, 0.5, 'Patient Zero',
                    ha='center', va='center', fontsize=10, style='italic')
        else:
            vals = np.array(list(contribs.values()), dtype=float)
            if vals.sum() > 0:
                vals /= vals.sum()
            colors = [agent_colors[a] for a in contribs.keys()]

            wedges, _, autotexts = ax.pie(
                vals, autopct='%1.1f%%', startangle=90, colors=colors,
                wedgeprops=dict(width=0.35, edgecolor='w')
            )
            for at in autotexts:
                at.set_fontsize(9)
                at.set_fontweight('bold')
            ax.set_title('Contributors', fontsize=11, pad=5)
            ax.set_aspect('equal')

    # --- Legend row (bottom row) ---
    ax_legend = fig.add_subplot(gs[2, :])
    ax_legend.axis('off')
    legend_elements = [Patch(facecolor=agent_colors[i], label=f"Agent {i}") for i in range(total_agents)]
    ax_legend.legend(handles=legend_elements, loc='center', ncol=total_agents,
                     fontsize=9, frameon=False)

    fig.suptitle('Fault Detection Timeline and Contributors',
                 fontsize=14, fontweight='bold', y=0.96)

    out_path = os.path.join(logdir, 'fault_timeline.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved fault timeline plot to {out_path}")


def plot_fault_timeline_action_influences(fault_timeline, action_influences_matrix_history, total_agents, logdir):
    """
    Plot fault timeline with action influence contributors instead of Frobenius norm influences.
    Each fault event shows the action influences from other agents as contributors.
    Additionally flags timesteps where faulty agents are among top-k influencers on non-faulty agents.
    """
    if len(fault_timeline) == 0:
        print("No faults detected; skipping action influences fault timeline plot.")
        return

    # Parameters for top-k influence detection
    k_top = 2  # Look for faulty agents in top-2 influencers
    
    # Create a mapping of when each agent was first detected as faulty
    fault_detection_times = {}  # agent_id -> timestep when first detected as faulty
    for event in fault_timeline:
        if event['agent'] not in fault_detection_times:
            fault_detection_times[event['agent']] = event['t']
    
    # Find the last fault detection timestep to stop flagging after this point
    last_fault_detection_time = max(fault_detection_times.values()) if fault_detection_times else -1
    
    # Find the first fault detection timestep for mean calculation
    first_fault_detection_time = min(fault_detection_times.values()) if fault_detection_times else -1
    
    # Find the first faulty agent (patient zero) - the one detected earliest
    first_faulty_agent = None
    if fault_detection_times:
        first_faulty_agent = min(fault_detection_times.keys(), key=lambda agent: fault_detection_times[agent])
    
    # Create extended timeline with additional flagged timesteps
    extended_timeline = []
    
    # Track already flagged (faulty_agent, target_agent) pairs to avoid duplicates
    flagged_pairs = set()  # Set of (faulty_agent_id, target_agent_id) tuples
    
    # Add original fault detection events with exact timestep action influences (not mean)
    for event in fault_timeline:
        # Use exact timestep action influences for fault detection events (like original version)
        faulty_agent = event['agent']
        fault_timestep = event['t']
        
        # Get action influences at the exact fault timestep
        if fault_timestep < len(action_influences_matrix_history):
            action_influences = action_influences_matrix_history[fault_timestep][faulty_agent]
            
            # Create contributors dict from action influences (include all agents including self)
            contribs = {}
            for j in range(total_agents):
                contribs[j] = abs(action_influences[j])  # Use absolute value of influence
        else:
            contribs = {}
        
        extended_timeline.append({
            'type': 'fault_detection',
            'agent': event['agent'],
            't': event['t'],
            'contribs': contribs,  # Use exact timestep influences, not mean
            'description': f"Faulty agent {event['agent']}"
        })
    
    # Find additional timesteps where faulty agents are top-k influencers on non-faulty agents
    # Only check timesteps up to (but not including) the last fault detection
    for t in range(min(len(action_influences_matrix_history), last_fault_detection_time)):
        influences_at_t = action_influences_matrix_history[t]
        
        # Get the set of agents that are considered faulty at timestep t
        faulty_agents_at_t = set()
        for agent_id, detection_time in fault_detection_times.items():
            if t >= detection_time:  # Only consider agent faulty from detection time onwards
                faulty_agents_at_t.add(agent_id)
        
        # Skip if no agents are faulty at this timestep
        if not faulty_agents_at_t:
            continue
        
        # For each non-faulty agent at timestep t, check if any faulty agent is in top-k influencers
        for non_faulty_agent in range(total_agents):
            # Check if this agent is faulty at timestep t
            is_faulty_at_t = non_faulty_agent in faulty_agents_at_t
            if is_faulty_at_t:
                continue  # Skip agents that are faulty at this timestep
                
            # Get influences on this non-faulty agent and rank them
            agent_influences = [(j, abs(influences_at_t[non_faulty_agent][j])) for j in range(total_agents)]
            # Sort by influence magnitude (descending)
            ranked_influences = sorted(agent_influences, key=lambda x: x[1], reverse=True)
            
            # Check if any faulty agent is in top-k
            top_k_agents = [agent_id for agent_id, _ in ranked_influences[:k_top]]
            faulty_in_top_k = [agent_id for agent_id in top_k_agents if agent_id in faulty_agents_at_t]
            
            if faulty_in_top_k:
                # Check if any of the faulty agents in top-k have already been flagged for this target
                new_faulty_influencers = []
                for faulty_agent in faulty_in_top_k:
                    pair = (faulty_agent, non_faulty_agent)
                    if pair not in flagged_pairs:
                        new_faulty_influencers.append(faulty_agent)
                        flagged_pairs.add(pair)  # Mark this pair as flagged
                
                # Only create an event if there are new faulty influencers to report
                if new_faulty_influencers:
                    # Check if this exact timestep+target combination is already in timeline
                    already_exists = any(event['t'] == t and event.get('target_agent') == non_faulty_agent 
                                       for event in extended_timeline)
                    if not already_exists:
                        # Use exact timestep action influences for top-k influence events (same as fault detection)
                        if t < len(action_influences_matrix_history):
                            action_influences = action_influences_matrix_history[t][non_faulty_agent]
                            
                            # Create contributors dict from action influences (include all agents including self)
                            contribs = {}
                            for j in range(total_agents):
                                contribs[j] = abs(action_influences[j])  # Use absolute value of influence
                        else:
                            contribs = {}
                        
                        faulty_list = ', '.join(map(str, new_faulty_influencers))
                        extended_timeline.append({
                            'type': 'top_k_influence',
                            'agent': non_faulty_agent,  # The affected agent
                            'faulty_influencers': new_faulty_influencers,
                            't': t,
                            'contribs': contribs,  # Use exact timestep influences, same as fault detection
                            'target_agent': non_faulty_agent,
                            "description": f"Faulty agent {faulty_list} is among the top-{k_top} influencers of Agent {non_faulty_agent}"
                        })
    
    # Sort extended timeline by timestep
    extended_timeline.sort(key=lambda x: x['t'])
    
    if len(extended_timeline) == 0:
        print("No events to display in action influences fault timeline.")
        return
    
    k = len(extended_timeline)
    fig = plt.figure(figsize=(max(8, 3*k), 6))  # Increased height for better visibility
    gs = fig.add_gridspec(
        3, k,
        height_ratios=[1.0, 2, 0.1],
        hspace=0.15
    )

    cmap = plt.get_cmap('tab20')
    agent_colors = {i: cmap(i % 20) for i in range(total_agents)}

    # --- Timeline axis (top row) ---
    ax_timeline = fig.add_subplot(gs[0, :])
    ax_timeline.axis('off')

    arrow_y = 0.5
    ax_timeline.annotate(
        '', xy=(1, arrow_y), xytext=(0, arrow_y),
        arrowprops=dict(arrowstyle='-|>', lw=2, color='gray'),
        xycoords='axes fraction', textcoords='axes fraction'
    )

    # Milestones
    for i, event in enumerate(extended_timeline):
        frac_x = (i + 0.5) / k

        # Different markers for different event types
        if event['type'] == 'fault_detection':
            marker_color = 'darkred'
            marker_size = 12
        else:  # top_k_influence
            marker_color = 'orange'
            marker_size = 10

        # Circle marker
        ax_timeline.plot(frac_x, arrow_y, 'o', color=marker_color, markersize=marker_size, 
                        transform=ax_timeline.transAxes)

        # Event description above (with line wrapping for long descriptions)
        description = event['description']
        if len(description) > 25:  # Wrap long descriptions
            words = description.split()
            lines = []
            current_line = []
            for word in words:
                if len(' '.join(current_line + [word])) <= 25:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                    else:
                        lines.append(word)
            if current_line:
                lines.append(' '.join(current_line))
            description = '\n'.join(lines)

        ax_timeline.text(frac_x, arrow_y + 0.15,
                         description,
                         ha='center', va='bottom',
                         fontsize=9, fontweight='bold',
                         transform=ax_timeline.transAxes)

        # Timestep label below
        ax_timeline.text(frac_x, arrow_y - 0.15,
                         f"t = {event['t']}",
                         ha='center', va='top',
                         fontsize=10, color='darkblue',
                         transform=ax_timeline.transAxes)

    # --- Contributor charts (middle row) ---
    for col, event in enumerate(extended_timeline):
        ax = fig.add_subplot(gs[1, col])
        
        contribs = event.get('contribs', {})

        # Check if this is the first faulty agent (patient zero) and a fault detection event
        if (event['type'] == 'fault_detection' and 
            event['agent'] == first_faulty_agent):
            ax.axis('off')
            ax.text(0.5, 0.5, 'Patient Zero',
                    ha='center', va='center', fontsize=12, fontweight='bold', 
                    style='italic', color='darkred')
        elif len(contribs) == 0:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No Data',
                    ha='center', va='center', fontsize=10, style='italic')
        else:
            vals = np.array(list(contribs.values()), dtype=float)
            if vals.sum() > 0:
                vals /= vals.sum()  # Normalize to sum to 1
            colors = [agent_colors[a] for a in contribs.keys()]

            wedges, _, autotexts = ax.pie(
                vals, autopct='%1.1f%%', startangle=90, colors=colors,
                wedgeprops=dict(width=0.35, edgecolor='w')
            )
            for at in autotexts:
                at.set_fontsize(8)
                at.set_fontweight('bold')
            
            # Different title based on event type
            title = f"Influences on Agent {event['agent']}"
            # if event['type'] == 'fault_detection':
            #     title = 'Contributors to Fault'
            # else:
            #     title = f"Influences on Agent {event['agent']}"
            
            ax.set_title(title, fontsize=10, pad=5)
            ax.set_aspect('equal')

    # --- Legend row (bottom row) ---
    ax_legend = fig.add_subplot(gs[2, :])
    ax_legend.axis('off')
    legend_elements = [Patch(facecolor=agent_colors[i], label=f"Agent {i}") for i in range(total_agents)]
    
    # Add legend for event types
    fault_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkred', 
                             markersize=10, label='Fault Detection')
    influence_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                                 markersize=10, label=f'Vulnerable Top-{k_top} Influence')
    legend_elements.extend([fault_marker, influence_marker])
    
    ax_legend.legend(handles=legend_elements, loc='center', ncol=min(len(legend_elements), 8),
                     fontsize=9, frameon=False)

    fig.suptitle('Fault Detection Timeline with Action Influence Analysis',
                 fontsize=14, fontweight='bold', y=0.96)

    out_path = os.path.join(logdir, 'fault_timeline_action_influences.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved enhanced action influences fault timeline plot to {out_path}")
    print(f"Timeline includes {len([e for e in extended_timeline if e['type'] == 'fault_detection'])} fault detections and {len([e for e in extended_timeline if e['type'] == 'top_k_influence'])} top-{k_top} influence events")


def plot_action_influences(action_influences_matrix_history_normal, action_influences_matrix_history, attacked_steps, atk_agent_id, logdir):
    """
    Plot the time series of action influences for each agent.
    Shows normal scenario in first row and attacked scenario in second row.
    For each agent i, show how much each agent j influences agent i's Q-value over time.
    """
    N = len(action_influences_matrix_history[0])  # number of agents
    T_attacked = len(action_influences_matrix_history)     # number of timesteps in attacked scenario
    T_normal = len(action_influences_matrix_history_normal)  # number of timesteps in normal scenario
    
    # Create 2*N subplots (two rows: normal and attacked scenarios)
    max_per_row = 3
    cols = min(N, max_per_row)
    rows = 2  # Two rows: normal and attacked scenarios
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows + 1))
    if N == 1:
        axes = axes.reshape(rows, 1)
    
    fig.suptitle(f'Action Influences: ∂Q_i/∂a_j (Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.96)
    
    # Use consistent agent colors
    agent_colors = get_agent_colors(N)
    
    # Track legend elements to avoid duplicates
    legend_elements = []
    legend_labels = []
    attacked_region_element = None
    
    scenarios = [
        (action_influences_matrix_history_normal, T_normal, "Normal"),
        (action_influences_matrix_history, T_attacked, "Attacked")
    ]
    
    # Calculate global max and min for consistent y-axis scaling
    global_max = 0
    global_min = 0
    for influences_data, T, _ in scenarios:
        for t in range(T):
            for i in range(N):
                for j in range(N):
                    value = influences_data[t][i][j]
                    global_max = max(global_max, value)
                    global_min = min(global_min, value)
    
    # Add row labels for scenarios
    for scenario_idx, (influences_data, T, scenario_name) in enumerate(scenarios):
        # Add scenario label on the left side
        if scenario_idx == 0:
            fig.text(0.02, 0.75, 'Normal\nScenario', fontsize=14, fontweight='bold', 
                    ha='center', va='center', rotation=90)
        else:
            fig.text(0.02, 0.25, 'Attacked\nScenario', fontsize=14, fontweight='bold', 
                    ha='center', va='center', rotation=90)
        
        for i in range(min(N, cols)):  # Only plot up to max_per_row agents
            ax = axes[scenario_idx, i]
            
            # Plot action influences for each agent j on agent i
            for j in range(N):
                influence_series = [influences_data[t][i][j] for t in range(T)]
                timesteps = range(T)
                
                # Self-influence (solid line)
                if i == j:
                    line = ax.plot(timesteps, influence_series, color=agent_colors[j], 
                                    linewidth=2.0, linestyle='-', alpha=1.0)[0]
                else:
                    # Influence from others (dashed line)
                    line = ax.plot(timesteps, influence_series, color=agent_colors[j], 
                                    linewidth=2.0, linestyle='--', alpha=1.0)[0]
                
                # avoid duplicate legend entries (only add for first subplot)
                if scenario_idx == 0 and i == 0:
                    legend_elements.append(line)
                    legend_labels.append(f'Agent {j}')
            
            # Mark attacked region (only for attacked scenario)
            if scenario_idx == 1 and attacked_steps:
                start = min(attacked_steps)
                end = max(attacked_steps)
                if i == atk_agent_id:
                    region = ax.axvspan(start, end, color='red', alpha=0.1)
                    if attacked_region_element is None:  # Store for later addition to legend
                        attacked_region_element = region
                else:
                    ax.axvspan(start, end, color='orange', alpha=0.05)
            
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Action Influence Magnitude')
            ax.set_title(f'Influences on Agent {i}')
            ax.grid(True, alpha=0.3)
            
            # Set consistent y-axis limits across all plots
            y_margin = (global_max - global_min) * 0.1
            ax.set_ylim(global_min - y_margin, global_max + y_margin)
    
    # Hide unused subplots
    for scenario_idx in range(rows):
        for j in range(N, cols):
            if j < cols:
                axes[scenario_idx, j].set_visible(False)
    
    # Add attacked region to legend at the end if it exists
    if attacked_region_element is not None:
        legend_elements.append(attacked_region_element)
        legend_labels.append('Attacked Region')
    
    # Create single legend at the bottom
    fig.legend(legend_elements, legend_labels, loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), ncol=min(len(legend_labels), 6), 
               fontsize=9, frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout to make room for legend and row labels
    plt.tight_layout(rect=[0.04, 0.08, 1, 0.94])
    
    out_path = os.path.join(logdir, f'action_influences_attacked_{atk_agent_id}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved action influences plot to {out_path}")


def plot_fault_timeline_second_order_action_influences(fault_timeline, second_order_action_influences_history, total_agents, logdir):
    """
    Plot fault timeline with second-order action influence contributors.
    Each fault event shows the second-order action influences from other agents as contributors.
    """
    if len(fault_timeline) == 0:
        print("No faults detected; skipping second-order action influences fault timeline plot.")
        return

    k = len(fault_timeline)
    fig = plt.figure(figsize=(max(6, 3*k), 5))
    gs = fig.add_gridspec(
        3, k,
        height_ratios=[0.8, 2, 0.1],
        hspace=0.1
    )

    cmap = plt.get_cmap('tab20')
    agent_colors = {i: cmap(i % 20) for i in range(total_agents)}

    # --- Timeline axis (top row) ---
    ax_timeline = fig.add_subplot(gs[0, :])
    ax_timeline.axis('off')

    arrow_y = 0.5
    ax_timeline.annotate(
        '', xy=(1, arrow_y), xytext=(0, arrow_y),
        arrowprops=dict(arrowstyle='-|>', lw=2, color='gray'),
        xycoords='axes fraction', textcoords='axes fraction'
    )

    # Milestones
    for i, event in enumerate(fault_timeline):
        frac_x = (i + 0.5) / k

        # Circle marker
        ax_timeline.plot(frac_x, arrow_y, 'o', color='darkred', markersize=10, transform=ax_timeline.transAxes)

        # Faulty agent label above
        ax_timeline.text(frac_x, arrow_y + 0.15,
                         f"Faulty agent {event['agent']}",
                         ha='center', va='bottom',
                         fontsize=11, fontweight='bold',
                         transform=ax_timeline.transAxes)

        # Timestep label below
        ax_timeline.text(frac_x, arrow_y - 0.15,
                         f"t = {event['t']}",
                         ha='center', va='top',
                         fontsize=10, color='darkblue',
                         transform=ax_timeline.transAxes)

    # --- Contributor charts (middle row) ---
    for col, event in enumerate(fault_timeline):
        ax = fig.add_subplot(gs[1, col])
        
        # For the first fault, don't show contributors
        if col == 0:
            contribs = {}
        else:
            # Calculate second-order action influence contributors for this fault event
            faulty_agent = event['agent']
            fault_timestep = event['t']
            
            # Get second-order action influences at the fault timestep
            if fault_timestep < len(second_order_action_influences_history):
                second_order_influences = second_order_action_influences_history[fault_timestep][faulty_agent]
                
                # Create contributors dict from second-order action influences (include all agents including self)
                contribs = {}
                for j in range(total_agents):
                    contribs[j] = abs(second_order_influences[j])  # Use absolute value of influence
            else:
                contribs = {}

        if len(contribs) == 0:
            ax.axis('off')
            if col == 0:
                ax.text(0.5, 0.5, 'Patient Zero',
                        ha='center', va='center', fontsize=10, style='italic')
            else:
                ax.text(0.5, 0.5, 'No 2nd-order action influences',
                        ha='center', va='center', fontsize=10, style='italic')
        else:
            vals = np.array(list(contribs.values()), dtype=float)
            if vals.sum() > 0:
                vals /= vals.sum()  # Normalize to sum to 1
            colors = [agent_colors[a] for a in contribs.keys()]

            wedges, _, autotexts = ax.pie(
                vals, autopct='%1.1f%%', startangle=90, colors=colors,
                wedgeprops=dict(width=0.35, edgecolor='w')
            )
            for at in autotexts:
                at.set_fontsize(9)
                at.set_fontweight('bold')
            ax.set_title('Contributors', fontsize=11, pad=5)
            ax.set_aspect('equal')

    # --- Legend row (bottom row) ---
    ax_legend = fig.add_subplot(gs[2, :])
    ax_legend.axis('off')
    legend_elements = [Patch(facecolor=agent_colors[i], label=f"Agent {i}") for i in range(total_agents)]
    ax_legend.legend(handles=legend_elements, loc='center', ncol=total_agents,
                     fontsize=9, frameon=False)

    fig.suptitle('Fault Detection Timeline and Second-Order Action Influence Contributors',
                 fontsize=14, fontweight='bold', y=0.96)

    out_path = os.path.join(logdir, 'fault_timeline_second_order_action_influences.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved second-order action influences fault timeline plot to {out_path}")


def plot_frob_norm_influences(frob_norms_matrix_history_normal, frob_norms_matrix_history, attacked_steps, atk_agent_id, logdir):
    """
    Plot the time series of Frobenius norm influences for each agent.
    Shows normal scenario in first row and attacked scenario in second row.
    For each agent i, show how much each agent j's Frobenius norm influences agent i over time.
    """
    N = len(frob_norms_matrix_history[0])  # number of agents
    T_attacked = len(frob_norms_matrix_history)     # number of timesteps in attacked scenario
    T_normal = len(frob_norms_matrix_history_normal)  # number of timesteps in normal scenario
    
    # Create 2*N subplots (two rows: normal and attacked scenarios)
    max_per_row = 3
    cols = min(N, max_per_row)
    rows = 2  # Two rows: normal and attacked scenarios
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows + 1))
    if N == 1:
        axes = axes.reshape(rows, 1)
    
    fig.suptitle(f'Frobenius Norm Influences: ||H_{{i,j}}||_F (Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.96)
    
    # Use consistent agent colors
    agent_colors = get_agent_colors(N)
    
    # Track legend elements to avoid duplicates
    legend_elements = []
    legend_labels = []
    attacked_region_element = None
    
    scenarios = [
        (frob_norms_matrix_history_normal, T_normal, "Normal"),
        (frob_norms_matrix_history, T_attacked, "Attacked")
    ]
    
    # Calculate global max and min for consistent y-axis scaling
    global_max = 0
    global_min = 0
    for influences_data, T, _ in scenarios:
        for t in range(T):
            for i in range(N):
                for j in range(N):
                    value = influences_data[t][i][j]
                    global_max = max(global_max, value)
                    global_min = min(global_min, value)
    
    # Add row labels for scenarios
    for scenario_idx, (influences_data, T, scenario_name) in enumerate(scenarios):
        # Add scenario label on the left side
        if scenario_idx == 0:
            fig.text(0.02, 0.75, 'Normal\nScenario', fontsize=14, fontweight='bold', 
                    ha='center', va='center', rotation=90)
        else:
            fig.text(0.02, 0.25, 'Attacked\nScenario', fontsize=14, fontweight='bold', 
                    ha='center', va='center', rotation=90)
        
        for i in range(min(N, cols)):  # Only plot up to max_per_row agents
            ax = axes[scenario_idx, i]
            
            # Plot Frobenius norm influences for each agent j on agent i
            for j in range(N):
                influence_series = [influences_data[t][i][j] for t in range(T)]
                timesteps = range(T)
                
                # Self-influence (solid line)
                if i == j:
                    line = ax.plot(timesteps, influence_series, color=agent_colors[j], 
                                    linewidth=2.0, linestyle='-', alpha=1.0)[0]
                else:
                    # Influence from others (dashed line)
                    line = ax.plot(timesteps, influence_series, color=agent_colors[j], 
                                    linewidth=2.0, linestyle='--', alpha=1.0)[0]
                
                # avoid duplicate legend entries (only add for first subplot)
                if scenario_idx == 0 and i == 0:
                    legend_elements.append(line)
                    legend_labels.append(f'Agent {j}')
            
            # Mark attacked region (only for attacked scenario)
            if scenario_idx == 1 and attacked_steps:
                start = min(attacked_steps)
                end = max(attacked_steps)
                if i == atk_agent_id:
                    region = ax.axvspan(start, end, color='red', alpha=0.1)
                    if attacked_region_element is None:  # Store for later addition to legend
                        attacked_region_element = region
                else:
                    ax.axvspan(start, end, color='orange', alpha=0.05)
            
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Frobenius Norm Influence Magnitude')
            ax.set_title(f'Frob Norm Influences on Agent {i}')
            ax.grid(True, alpha=0.3)
            
            # Set consistent y-axis limits across all plots
            y_margin = (global_max - global_min) * 0.1
            ax.set_ylim(global_min - y_margin, global_max + y_margin)
    
    # Hide unused subplots
    for scenario_idx in range(rows):
        for j in range(N, cols):
            if j < cols:
                axes[scenario_idx, j].set_visible(False)
    
    # Add attacked region to legend at the end if it exists
    if attacked_region_element is not None:
        legend_elements.append(attacked_region_element)
        legend_labels.append('Attacked Region')
    
    # Create single legend at the bottom
    fig.legend(legend_elements, legend_labels, loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), ncol=min(len(legend_labels), 6), 
               fontsize=9, frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout to make room for legend and row labels
    plt.tight_layout(rect=[0.04, 0.08, 1, 0.94])
    
    out_path = os.path.join(logdir, f'frob_norm_influences_attacked_{atk_agent_id}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved Frobenius norm influences plot to {out_path}")


def plot_pairwise_action_influences(action_influences_normal, action_influences_attacked, attacked_steps, atk_agent_id, logdir):
    """
    Plot N×N grid of subplots showing pairwise action influences.
    Each subplot (i,j) shows the influence of agent j on agent i for both normal and attacked scenarios.
    """
    N = len(action_influences_normal[0])  # number of agents
    T_normal = len(action_influences_normal)
    T_attacked = len(action_influences_attacked)
    
    # Create N×N subplots
    fig, axes = plt.subplots(N, N, figsize=(3*N, 3*N))
    if N == 1:
        axes = [[axes]]
    elif N > 1 and axes.ndim == 1:
        axes = axes.reshape(N, N)
    
    fig.suptitle(f'Pairwise Action Influences: ∂Q_i/∂a_j (Attacked Agent ID: {atk_agent_id})', fontsize=14, y=0.95)
    
    for i in range(N):  # agent i (influenced)
        for j in range(N):  # agent j (influencer)
            ax = axes[i][j]
            
            # Extract influence time series for agent j's influence on agent i
            normal_series = [action_influences_normal[t][i][j] for t in range(T_normal)]
            attacked_series = [action_influences_attacked[t][i][j] for t in range(T_attacked)]
            
            # Plot time series
            timesteps_normal = range(T_normal)
            timesteps_attacked = range(T_attacked)
            
            ax.plot(timesteps_normal, normal_series, 'g-', label='Normal', linewidth=2, alpha=0.8)
            ax.plot(timesteps_attacked, attacked_series, 'r-', label='Attacked', linewidth=2, alpha=0.8)
            
            # Mark attacked region
            if attacked_steps:
                start = min(attacked_steps)
                end = max(attacked_steps)
                ax.axvspan(start, end, color='red', alpha=0.1)
            
            # Styling
            ax.set_title(f'Agent {j} → Agent {i}', fontsize=10)
            ax.set_xlabel('Timestep', fontsize=9)
            ax.set_ylabel('Influence Magnitude', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            # Add legend only to the top-right subplot to avoid clutter
            if i == 0 and j == N-1:
                ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    out_path = os.path.join(logdir, f'pairwise_action_influences_attacked_{atk_agent_id}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved pairwise action influences plot to {out_path}")


def plot_second_order_action_influences(second_order_action_influences_history_normal, second_order_action_influences_history, attacked_steps, atk_agent_id, logdir):
    """
    Plot the time series of second-order action influences for each agent.
    Shows normal scenario in first row and attacked scenario in second row.
    For each agent i, show how much each agent j's action curvature influences agent i's Q-value over time.
    """
    N = len(second_order_action_influences_history[0])  # number of agents
    T_attacked = len(second_order_action_influences_history)     # number of timesteps in attacked scenario
    T_normal = len(second_order_action_influences_history_normal)  # number of timesteps in normal scenario
    
    # Create 2*N subplots (two rows: normal and attacked scenarios)
    max_per_row = 3
    cols = min(N, max_per_row)
    rows = 2  # Two rows: normal and attacked scenarios
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows + 1))
    if N == 1:
        axes = axes.reshape(rows, 1)
    
    fig.suptitle(f'Second-Order Action Influences: ∂²Q_i/(∂a_j)² (Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.96)
    
    # Use consistent agent colors
    agent_colors = get_agent_colors(N)
    
    # Track legend elements to avoid duplicates
    legend_elements = []
    legend_labels = []
    attacked_region_element = None
    
    scenarios = [
        (second_order_action_influences_history_normal, T_normal, "Normal"),
        (second_order_action_influences_history, T_attacked, "Attacked")
    ]
    
    # Calculate global max and min for consistent y-axis scaling
    global_max = 0
    global_min = 0
    for influences_data, T, _ in scenarios:
        for t in range(T):
            for i in range(N):
                for j in range(N):
                    value = influences_data[t][i][j]
                    global_max = max(global_max, value)
                    global_min = min(global_min, value)
    
    # Add row labels for scenarios
    for scenario_idx, (influences_data, T, scenario_name) in enumerate(scenarios):
        # Add scenario label on the left side
        if scenario_idx == 0:
            fig.text(0.02, 0.75, 'Normal\nScenario', fontsize=14, fontweight='bold', 
                    ha='center', va='center', rotation=90)
        else:
            fig.text(0.02, 0.25, 'Attacked\nScenario', fontsize=14, fontweight='bold', 
                    ha='center', va='center', rotation=90)
        
        for i in range(min(N, cols)):  # Only plot up to max_per_row agents
            ax = axes[scenario_idx, i]
            
            # Plot second-order influences for each agent j on agent i
            for j in range(N):
                influence_series = [influences_data[t][i][j] for t in range(T)]
                timesteps = range(T)
                
                # Self-influence (solid line) vs others (dashed line)
                if i == j:
                    line = ax.plot(timesteps, influence_series, color=agent_colors[j], 
                                    linewidth=2.5, linestyle='-', alpha=1.0)[0]
                else:
                    line = ax.plot(timesteps, influence_series, color=agent_colors[j], 
                                    linewidth=2.0, linestyle='--', alpha=0.8)[0]
                
                # avoid duplicate legend entries (only add for first subplot)  
                if scenario_idx == 0 and i == 0:
                    legend_elements.append(line)
                    legend_labels.append(f'Agent {j}')
            
            # Mark attacked region (only for attacked scenario)
            if scenario_idx == 1 and attacked_steps:
                start = min(attacked_steps)
                end = max(attacked_steps)
                if i == atk_agent_id:
                    region = ax.axvspan(start, end, color='red', alpha=0.1)
                    if attacked_region_element is None:  # Store for later addition to legend
                        attacked_region_element = region
                else:
                    ax.axvspan(start, end, color='orange', alpha=0.05)
            
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Second-Order Influence Magnitude')
            ax.set_title(f'Second-Order Influences on Agent {i}')
            ax.grid(True, alpha=0.3)
            
            # Set consistent y-axis limits across all plots
            y_margin = (global_max - global_min) * 0.1
            ax.set_ylim(global_min - y_margin, global_max + y_margin)
    
    # Hide unused subplots
    for scenario_idx in range(rows):
        for j in range(N, cols):
            if j < cols:
                axes[scenario_idx, j].set_visible(False)
    
    # Add attacked region to legend at the end if it exists
    if attacked_region_element is not None:
        legend_elements.append(attacked_region_element)
        legend_labels.append('Attacked Region')
    
    # Create single legend at the bottom
    fig.legend(legend_elements, legend_labels, loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), ncol=min(len(legend_labels), 6), 
               fontsize=9, frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout to make room for legend and row labels
    plt.tight_layout(rect=[0.04, 0.08, 1, 0.94])
    
    out_path = os.path.join(logdir, f'second_order_action_influences_attacked_{atk_agent_id}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved second-order action influences plot to {out_path}")


def plot_pairwise_second_order_action_influences(second_order_normal, second_order_attacked, attacked_steps, atk_agent_id, logdir):
    """
    Plot N×N grid of subplots showing pairwise second-order action influences.
    Each subplot (i,j) shows ∂²Q_i/(∂a_j)² for both normal and attacked scenarios.
    """
    N = len(second_order_normal[0])  # number of agents
    T_normal = len(second_order_normal)
    T_attacked = len(second_order_attacked)
    
    # Create N×N subplots
    fig, axes = plt.subplots(N, N, figsize=(3*N, 3*N))
    if N == 1:
        axes = [[axes]]
    elif N > 1 and axes.ndim == 1:
        axes = axes.reshape(N, N)
    
    fig.suptitle(f'Pairwise Second-Order Action Influences: ∂²Q_i/(∂a_j)² (Attacked Agent ID: {atk_agent_id})', fontsize=14, y=0.95)
    
    for i in range(N):  # agent i (influenced)
        for j in range(N):  # agent j (influencer)
            ax = axes[i][j]
            
            # Extract second-order influence time series for agent j's action curvature on agent i
            normal_series = [second_order_normal[t][i][j] for t in range(T_normal)]
            attacked_series = [second_order_attacked[t][i][j] for t in range(T_attacked)]
            
            # Plot time series
            timesteps_normal = range(T_normal)
            timesteps_attacked = range(T_attacked)
            
            ax.plot(timesteps_normal, normal_series, 'g-', label='Normal', linewidth=2, alpha=0.8)
            ax.plot(timesteps_attacked, attacked_series, 'r-', label='Attacked', linewidth=2, alpha=0.8)
            
            # Mark attacked region
            if attacked_steps:
                start = min(attacked_steps)
                end = max(attacked_steps)
                ax.axvspan(start, end, color='red', alpha=0.1)
            
            # Styling
            ax.set_title(f'Agent {j} → Agent {i}', fontsize=10)
            ax.set_xlabel('Timestep', fontsize=9)
            ax.set_ylabel('2nd-Order Influence', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            # Add legend only to the top-right subplot to avoid clutter
            if i == 0 and j == N-1:
                ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    out_path = os.path.join(logdir, f'pairwise_second_order_action_influences_attacked_{atk_agent_id}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved pairwise second-order action influences plot to {out_path}")


def plot_observation_influences(observation_influences_matrix_history_normal, observation_influences_matrix_history, attacked_steps, atk_agent_id, logdir):
    """
    Plot the time series of observation influences for each agent.
    Shows normal scenario in first row and attacked scenario in second row.
    For each agent i, show how much each agent j's observation influences agent i's Q-value over time.
    """
    N = len(observation_influences_matrix_history[0])  # number of agents
    T_attacked = len(observation_influences_matrix_history)     # number of timesteps in attacked scenario
    T_normal = len(observation_influences_matrix_history_normal)  # number of timesteps in normal scenario
    
    # Create 2*N subplots (two rows: normal and attacked scenarios)
    max_per_row = 3
    cols = min(N, max_per_row)
    rows = 2  # Two rows: normal and attacked scenarios
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows + 1))
    if N == 1:
        axes = axes.reshape(rows, 1)
    
    fig.suptitle(f'Observation Influences: ∂Q_i/∂obs_j (Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.96)
    
    # Use consistent agent colors
    agent_colors = get_agent_colors(N)
    
    # Track legend elements to avoid duplicates
    legend_elements = []
    legend_labels = []
    attacked_region_element = None
    
    scenarios = [
        (observation_influences_matrix_history_normal, T_normal, "Normal"),
        (observation_influences_matrix_history, T_attacked, "Attacked")
    ]
    
    # Calculate global max and min for consistent y-axis scaling
    global_max = 0
    global_min = 0
    for influences_data, T, _ in scenarios:
        for t in range(T):
            for i in range(N):
                for j in range(N):
                    value = influences_data[t][i][j]
                    global_max = max(global_max, value)
                    global_min = min(global_min, value)
    
    # Add row labels for scenarios
    for scenario_idx, (influences_data, T, scenario_name) in enumerate(scenarios):
        # Add scenario label on the left side
        if scenario_idx == 0:
            fig.text(0.02, 0.75, 'Normal\nScenario', fontsize=14, fontweight='bold', 
                    ha='center', va='center', rotation=90)
        else:
            fig.text(0.02, 0.25, 'Attacked\nScenario', fontsize=14, fontweight='bold', 
                    ha='center', va='center', rotation=90)
        
        for i in range(min(N, cols)):  # Only plot up to max_per_row agents
            ax = axes[scenario_idx, i]
            
            # Plot observation influences for each agent j on agent i
            for j in range(N):
                influence_series = [influences_data[t][i][j] for t in range(T)]
                timesteps = range(T)
                
                # Self-influence (solid line)
                if i == j:
                    line = ax.plot(timesteps, influence_series, color=agent_colors[j], 
                                    linewidth=2.0, linestyle='-', alpha=1.0)[0]
                else:
                    # Influence from others (dashed line)
                    line = ax.plot(timesteps, influence_series, color=agent_colors[j], 
                                    linewidth=2.0, linestyle='--', alpha=1.0)[0]
                
                # avoid duplicate legend entries (only add for first subplot)
                if scenario_idx == 0 and i == 0:
                    legend_elements.append(line)
                    legend_labels.append(f'Agent {j}')
            
            # Mark attacked region (only for attacked scenario)
            if scenario_idx == 1 and attacked_steps:
                start = min(attacked_steps)
                end = max(attacked_steps)
                if i == atk_agent_id:
                    region = ax.axvspan(start, end, color='red', alpha=0.1)
                    if attacked_region_element is None:
                        attacked_region_element = region
                else:
                    ax.axvspan(start, end, color='orange', alpha=0.05)
            
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Observation Influence Magnitude')
            ax.set_title(f'Observation Influences on Agent {i}')
            ax.grid(True, alpha=0.3)
            
            # Set consistent y-axis limits across all plots
            y_margin = (global_max - global_min) * 0.1
            ax.set_ylim(global_min - y_margin, global_max + y_margin)
    
    # Hide unused subplots
    for scenario_idx in range(rows):
        for j in range(N, cols):
            if j < cols:
                axes[scenario_idx, j].set_visible(False)
    
    # Add attacked region to legend at the end if it exists
    if attacked_region_element is not None:
        legend_elements.append(attacked_region_element)
        legend_labels.append('Attacked Region')
    
    # Create single legend at the bottom
    fig.legend(legend_elements, legend_labels, loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), ncol=min(len(legend_labels), 6), 
               fontsize=9, frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout to make room for legend and row labels
    plt.tight_layout(rect=[0.04, 0.08, 1, 0.94])
    
    out_path = os.path.join(logdir, f'observation_influences_attacked_{atk_agent_id}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved observation influences plot to {out_path}")


def plot_fault_timeline_observation_influences(fault_timeline, observation_influences_matrix_history, total_agents, logdir):
    """
    Plot fault timeline with observation influence contributors.
    Each fault event shows the observation influences from other agents as contributors.
    """
    if len(fault_timeline) == 0:
        print("No faults detected; skipping observation influences fault timeline plot.")
        return

    k = len(fault_timeline)
    fig = plt.figure(figsize=(max(6, 3*k), 5))
    gs = fig.add_gridspec(
        3, k,
        height_ratios=[0.8, 2, 0.1],
        hspace=0.1
    )

    cmap = plt.get_cmap('tab20')
    agent_colors = {i: cmap(i % 20) for i in range(total_agents)}

    # --- Timeline axis (top row) ---
    ax_timeline = fig.add_subplot(gs[0, :])
    ax_timeline.axis('off')

    arrow_y = 0.5
    ax_timeline.annotate(
        '', xy=(1, arrow_y), xytext=(0, arrow_y),
        arrowprops=dict(arrowstyle='-|>', lw=2, color='gray'),
        xycoords='axes fraction', textcoords='axes fraction'
    )

    # Milestones
    for i, event in enumerate(fault_timeline):
        frac_x = (i + 0.5) / k

        # Circle marker
        ax_timeline.plot(frac_x, arrow_y, 'o', color='darkred', markersize=10, transform=ax_timeline.transAxes)

        # Faulty agent label above
        ax_timeline.text(frac_x, arrow_y + 0.15,
                         f"Faulty agent {event['agent']}",
                         ha='center', va='bottom',
                         fontsize=11, fontweight='bold',
                         transform=ax_timeline.transAxes)

        # Timestep label below
        ax_timeline.text(frac_x, arrow_y - 0.15,
                         f"t = {event['t']}",
                         ha='center', va='top',
                         fontsize=10, color='darkblue',
                         transform=ax_timeline.transAxes)

    # --- Contributor charts (middle row) ---
    for col, event in enumerate(fault_timeline):
        ax = fig.add_subplot(gs[1, col])
        
        # For the first fault, don't show contributors
        if col == 0:
            contribs = {}
        else:
            # Calculate observation influence contributors for this fault event
            faulty_agent = event['agent']
            fault_timestep = event['t']
            
            # Get observation influences at the fault timestep
            if fault_timestep < len(observation_influences_matrix_history):
                observation_influences = observation_influences_matrix_history[fault_timestep][faulty_agent]
                
                # Create contributors dict from observation influences (include all agents including self)
                contribs = {}
                for j in range(total_agents):
                    contribs[j] = abs(observation_influences[j])  # Use absolute value of influence
            else:
                contribs = {}

        if len(contribs) == 0:
            ax.axis('off')
            if col == 0:
                ax.text(0.5, 0.5, 'Patient Zero',
                        ha='center', va='center', fontsize=10, style='italic')
            else:
                ax.text(0.5, 0.5, 'No observation influences',
                        ha='center', va='center', fontsize=10, style='italic')
        else:
            vals = np.array(list(contribs.values()), dtype=float)
            if vals.sum() > 0:
                vals /= vals.sum()  # Normalize to sum to 1
            colors = [agent_colors[a] for a in contribs.keys()]

            wedges, _, autotexts = ax.pie(
                vals, autopct='%1.1f%%', startangle=90, colors=colors,
                wedgeprops=dict(width=0.35, edgecolor='w')
            )
            for at in autotexts:
                at.set_fontsize(9)
                at.set_fontweight('bold')
            ax.set_title('Contributors', fontsize=11, pad=5)
            ax.set_aspect('equal')

    # --- Legend row (bottom row) ---
    ax_legend = fig.add_subplot(gs[2, :])
    ax_legend.axis('off')
    legend_elements = [Patch(facecolor=agent_colors[i], label=f"Agent {i}") for i in range(total_agents)]
    ax_legend.legend(handles=legend_elements, loc='center', ncol=total_agents,
                     fontsize=9, frameon=False)

    fig.suptitle('Fault Detection Timeline and Observation Influence Contributors',
                 fontsize=14, fontweight='bold', y=0.96)

    out_path = os.path.join(logdir, 'fault_timeline_observation_influences.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved observation influences fault timeline plot to {out_path}")


def plot_pairwise_observation_influences(observation_influences_normal, observation_influences_attacked, attacked_steps, atk_agent_id, logdir):
    """
    Plot N×N grid of subplots showing pairwise observation influences.
    Each subplot (i,j) shows the influence of agent j's observation on agent i for both normal and attacked scenarios.
    """
    N = len(observation_influences_normal[0])  # number of agents
    T_normal = len(observation_influences_normal)
    T_attacked = len(observation_influences_attacked)
    
    # Create N×N subplots
    fig, axes = plt.subplots(N, N, figsize=(3*N, 3*N))
    if N == 1:
        axes = [[axes]]
    elif N > 1 and axes.ndim == 1:
        axes = axes.reshape(N, N)
    
    fig.suptitle(f'Pairwise Observation Influences: ∂Q_i/∂obs_j (Attacked Agent ID: {atk_agent_id})', fontsize=14, y=0.95)
    
    for i in range(N):  # agent i (influenced)
        for j in range(N):  # agent j (influencer)
            ax = axes[i][j]
            
            # Extract influence time series for agent j's observation influence on agent i
            normal_series = [observation_influences_normal[t][i][j] for t in range(T_normal)]
            attacked_series = [observation_influences_attacked[t][i][j] for t in range(T_attacked)]
            
            # Plot time series
            timesteps_normal = range(T_normal)
            timesteps_attacked = range(T_attacked)
            
            ax.plot(timesteps_normal, normal_series, 'g-', label='Normal', linewidth=2, alpha=0.8)
            ax.plot(timesteps_attacked, attacked_series, 'r-', label='Attacked', linewidth=2, alpha=0.8)
            
            # Mark attacked region
            if attacked_steps:
                start = min(attacked_steps)
                end = max(attacked_steps)
                ax.axvspan(start, end, color='red', alpha=0.1)
            
            # Styling
            ax.set_title(f'Agent {j} → Agent {i}', fontsize=10)
            ax.set_xlabel('Timestep', fontsize=9)
            ax.set_ylabel('Observation Influence Magnitude', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            # Add legend only to the top-right subplot to avoid clutter
            if i == 0 and j == N-1:
                ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    out_path = os.path.join(logdir, f'pairwise_observation_influences_attacked_{atk_agent_id}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved pairwise observation influences plot to {out_path}")


def plot_second_order_observation_influences(second_order_observation_influences_history_normal, second_order_observation_influences_history, attacked_steps, atk_agent_id, logdir):
    """
    Plot the time series of second-order observation influences for each agent.
    Shows normal scenario in first row and attacked scenario in second row.
    For each agent i, show how much each agent j's observation curvature influences agent i's Q-value over time.
    """
    N = len(second_order_observation_influences_history[0])  # number of agents
    T_attacked = len(second_order_observation_influences_history)     # number of timesteps in attacked scenario
    T_normal = len(second_order_observation_influences_history_normal)  # number of timesteps in normal scenario
    
    # Create 2*N subplots (two rows: normal and attacked scenarios)
    max_per_row = 3
    cols = min(N, max_per_row)
    rows = 2  # Two rows: normal and attacked scenarios
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows + 1))
    if N == 1:
        axes = axes.reshape(rows, 1)
    
    fig.suptitle(f'Second-Order Observation Influences: ∂²Q_i/(∂obs_j)² (Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.96)
    
    # Use consistent agent colors
    agent_colors = get_agent_colors(N)
    
    # Track legend elements to avoid duplicates
    legend_elements = []
    legend_labels = []
    attacked_region_element = None
    
    scenarios = [
        (second_order_observation_influences_history_normal, T_normal, "Normal"),
        (second_order_observation_influences_history, T_attacked, "Attacked")
    ]
    
    # Calculate global max and min for consistent y-axis scaling
    global_max = 0
    global_min = 0
    for influences_data, T, _ in scenarios:
        for t in range(T):
            for i in range(N):
                for j in range(N):
                    value = influences_data[t][i][j]
                    global_max = max(global_max, value)
                    global_min = min(global_min, value)
    
    # Add row labels for scenarios
    for scenario_idx, (influences_data, T, scenario_name) in enumerate(scenarios):
        # Add scenario label on the left side
        if scenario_idx == 0:
            fig.text(0.02, 0.75, 'Normal\nScenario', fontsize=14, fontweight='bold', 
                    ha='center', va='center', rotation=90)
        else:
            fig.text(0.02, 0.25, 'Attacked\nScenario', fontsize=14, fontweight='bold', 
                    ha='center', va='center', rotation=90)
        
        for i in range(min(N, cols)):  # Only plot up to max_per_row agents
            ax = axes[scenario_idx, i]
            
            # Plot second-order observation influences for each agent j on agent i
            for j in range(N):
                influence_series = [influences_data[t][i][j] for t in range(T)]
                timesteps = range(T)
                
                # Self-influence (solid line) vs others (dashed line)
                if i == j:
                    line = ax.plot(timesteps, influence_series, color=agent_colors[j], 
                                    linewidth=2.0, linestyle='-', alpha=1.0)[0]
                else:
                    line = ax.plot(timesteps, influence_series, color=agent_colors[j], 
                                    linewidth=2.0, linestyle='--', alpha=1.0)[0]
                
                # avoid duplicate legend entries (only add for first subplot)
                if scenario_idx == 0 and i == 0:
                    legend_elements.append(line)
                    legend_labels.append(f'Agent {j}')
            
            # Mark attacked region (only for attacked scenario)
            if scenario_idx == 1 and attacked_steps:
                start = min(attacked_steps)
                end = max(attacked_steps)
                if i == atk_agent_id:
                    region = ax.axvspan(start, end, color='red', alpha=0.1)
                    if attacked_region_element is None:
                        attacked_region_element = region
                else:
                    ax.axvspan(start, end, color='orange', alpha=0.05)
            
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Second-Order Observation Influence Magnitude')
            ax.set_title(f'Second-Order Observation Influences on Agent {i}')
            ax.grid(True, alpha=0.3)
            
            # Set consistent y-axis limits across all plots
            y_margin = (global_max - global_min) * 0.1
            ax.set_ylim(global_min - y_margin, global_max + y_margin)
    
    # Hide unused subplots
    for scenario_idx in range(rows):
        for j in range(N, cols):
            if j < cols:
                axes[scenario_idx, j].set_visible(False)
    
    # Add attacked region to legend at the end if it exists
    if attacked_region_element is not None:
        legend_elements.append(attacked_region_element)
        legend_labels.append('Attacked Region')
    
    # Create single legend at the bottom
    fig.legend(legend_elements, legend_labels, loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), ncol=min(len(legend_labels), 6), 
               fontsize=9, frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout to make room for legend and row labels
    plt.tight_layout(rect=[0.04, 0.08, 1, 0.94])
    
    out_path = os.path.join(logdir, f'second_order_observation_influences_attacked_{atk_agent_id}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved second-order observation influences plot to {out_path}")


def plot_fault_timeline_second_order_observation_influences(fault_timeline, second_order_observation_influences_history, total_agents, logdir):
    """
    Plot fault timeline with second-order observation influence contributors.
    Each fault event shows the second-order observation influences from other agents as contributors.
    """
    if len(fault_timeline) == 0:
        print("No faults detected; skipping second-order observation influences fault timeline plot.")
        return

    k = len(fault_timeline)
    fig = plt.figure(figsize=(max(6, 3*k), 5))
    gs = fig.add_gridspec(
        3, k,
        height_ratios=[0.8, 2, 0.1],
        hspace=0.1
    )

    cmap = plt.get_cmap('tab20')
    agent_colors = {i: cmap(i % 20) for i in range(total_agents)}

    # --- Timeline axis (top row) ---
    ax_timeline = fig.add_subplot(gs[0, :])
    ax_timeline.axis('off')

    arrow_y = 0.5
    ax_timeline.annotate(
        '', xy=(1, arrow_y), xytext=(0, arrow_y),
        arrowprops=dict(arrowstyle='-|>', lw=2, color='gray'),
        xycoords='axes fraction', textcoords='axes fraction'
    )

    # Milestones
    for i, event in enumerate(fault_timeline):
        frac_x = (i + 0.5) / k

        # Circle marker
        ax_timeline.plot(frac_x, arrow_y, 'o', color='darkred', markersize=10, transform=ax_timeline.transAxes)

        # Faulty agent label above
        ax_timeline.text(frac_x, arrow_y + 0.15,
                         f"Faulty agent {event['agent']}",
                         ha='center', va='bottom',
                         fontsize=11, fontweight='bold',
                         transform=ax_timeline.transAxes)

        # Timestep label below
        ax_timeline.text(frac_x, arrow_y - 0.15,
                         f"t = {event['t']}",
                         ha='center', va='top',
                         fontsize=10, color='darkblue',
                         transform=ax_timeline.transAxes)

    # --- Contributor charts (middle row) ---
    for col, event in enumerate(fault_timeline):
        ax = fig.add_subplot(gs[1, col])
        
        # For the first fault, don't show contributors
        if col == 0:
            contribs = {}
        else:
            # Calculate second-order observation influence contributors for this fault event
            faulty_agent = event['agent']
            fault_timestep = event['t']
            
            # Get second-order observation influences at the fault timestep
            if fault_timestep < len(second_order_observation_influences_history):
                second_order_obs_influences = second_order_observation_influences_history[fault_timestep][faulty_agent]
                
                # Create contributors dict from second-order observation influences (include all agents including self)
                contribs = {}
                for j in range(total_agents):
                    contribs[j] = abs(second_order_obs_influences[j])  # Use absolute value of influence
            else:
                contribs = {}

        if len(contribs) == 0:
            ax.axis('off')
            if col == 0:
                ax.text(0.5, 0.5, 'Patient Zero',
                        ha='center', va='center', fontsize=10, style='italic')
            else:
                ax.text(0.5, 0.5, 'No 2nd-order obs influences',
                        ha='center', va='center', fontsize=10, style='italic')
        else:
            vals = np.array(list(contribs.values()), dtype=float)
            if vals.sum() > 0:
                vals /= vals.sum()  # Normalize to sum to 1
            colors = [agent_colors[a] for a in contribs.keys()]

            wedges, _, autotexts = ax.pie(
                vals, autopct='%1.1f%%', startangle=90, colors=colors,
                wedgeprops=dict(width=0.35, edgecolor='w')
            )
            for at in autotexts:
                at.set_fontsize(9)
                at.set_fontweight('bold')
            ax.set_title('Contributors', fontsize=11, pad=5)
            ax.set_aspect('equal')

    # --- Legend row (bottom row) ---
    ax_legend = fig.add_subplot(gs[2, :])
    ax_legend.axis('off')
    legend_elements = [Patch(facecolor=agent_colors[i], label=f"Agent {i}") for i in range(total_agents)]
    ax_legend.legend(handles=legend_elements, loc='center', ncol=total_agents,
                     fontsize=9, frameon=False)

    fig.suptitle('Fault Detection Timeline and Second-Order Observation Influence Contributors',
                 fontsize=14, fontweight='bold', y=0.96)

    out_path = os.path.join(logdir, 'fault_timeline_second_order_observation_influences.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved second-order observation influences fault timeline plot to {out_path}")


def plot_pairwise_second_order_observation_influences(second_order_obs_normal, second_order_obs_attacked, attacked_steps, atk_agent_id, logdir):
    """
    Plot N×N grid of subplots showing pairwise second-order observation influences.
    Each subplot (i,j) shows ∂²Q_i/(∂obs_j)² for both normal and attacked scenarios.
    """
    N = len(second_order_obs_normal[0])  # number of agents
    T_normal = len(second_order_obs_normal)
    T_attacked = len(second_order_obs_attacked)
    
    # Create N×N subplots
    fig, axes = plt.subplots(N, N, figsize=(3*N, 3*N))
    if N == 1:
        axes = [[axes]]
    elif N > 1 and axes.ndim == 1:
        axes = axes.reshape(N, N)
    
    fig.suptitle(f'Pairwise Second-Order Observation Influences: ∂²Q_i/(∂obs_j)² (Attacked Agent ID: {atk_agent_id})', fontsize=14, y=0.95)
    
    for i in range(N):  # agent i (influenced)
        for j in range(N):  # agent j (influencer)
            ax = axes[i][j]
            
            # Extract second-order observation influence time series for agent j's observation curvature on agent i
            normal_series = [second_order_obs_normal[t][i][j] for t in range(T_normal)]
            attacked_series = [second_order_obs_attacked[t][i][j] for t in range(T_attacked)]
            
            # Plot time series
            timesteps_normal = range(T_normal)
            timesteps_attacked = range(T_attacked)
            
            ax.plot(timesteps_normal, normal_series, 'g-', label='Normal', linewidth=2, alpha=0.8)
            ax.plot(timesteps_attacked, attacked_series, 'r-', label='Attacked', linewidth=2, alpha=0.8)
            
            # Mark attacked region
            if attacked_steps:
                start = min(attacked_steps)
                end = max(attacked_steps)
                ax.axvspan(start, end, color='red', alpha=0.1)
            
            # Styling
            ax.set_title(f'Agent {j} → Agent {i}', fontsize=10)
            ax.set_xlabel('Timestep', fontsize=9)
            ax.set_ylabel('2nd-Order Obs Influence', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            # Add legend only to the top-right subplot to avoid clutter
            if i == 0 and j == N-1:
                ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    out_path = os.path.join(logdir, f'pairwise_second_order_observation_influences_attacked_{atk_agent_id}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved pairwise second-order observation influences plot to {out_path}")


def plot_contributor_barchart(fault_timeline, total_agents, logdir):
    if len(fault_timeline) == 0:
        print("No faults detected; skipping contributor bar chart.")
        return

    k = len(fault_timeline)
    # Increase figure width for better spacing, especially with many events
    fig = plt.figure(figsize=(max(8, 4*k), 6))  # Increased from 3*k to 4*k width and 5 to 6 height
    gs = fig.add_gridspec(
        3, k,
        height_ratios=[0.8, 2.5, 0.2],  # Give more space to the middle row
        hspace=0.15,  # Increase vertical spacing
        wspace=0.3    # Add horizontal spacing between subplots
    )

    cmap = plt.get_cmap('tab20')
    agent_colors = {i: cmap(i % 20) for i in range(total_agents)}

    # --- Timeline axis (top row) ---
    ax_timeline = fig.add_subplot(gs[0, :])
    ax_timeline.axis('off')

    arrow_y = 0.5
    ax_timeline.annotate(
        '', xy=(1, arrow_y), xytext=(0, arrow_y),
        arrowprops=dict(arrowstyle='-|>', lw=2, color='gray'),
        xycoords='axes fraction', textcoords='axes fraction'
    )

    for i, event in enumerate(fault_timeline):
        frac_x = (i + 0.5) / k
        ax_timeline.plot(frac_x, arrow_y, 'o', color='darkred', markersize=10, transform=ax_timeline.transAxes)
        ax_timeline.text(frac_x, arrow_y + 0.15,
                         f"Faulty agent {event['agent']}",
                         ha='center', va='bottom',
                         fontsize=11, fontweight='bold',
                         transform=ax_timeline.transAxes)
        ax_timeline.text(frac_x, arrow_y - 0.15,
                         f"t = {event['t']}",
                         ha='center', va='top',
                         fontsize=10, color='darkblue',
                         transform=ax_timeline.transAxes)

    # --- Contributor bar charts (middle row) ---
    for col, event in enumerate(fault_timeline):
        ax = fig.add_subplot(gs[1, col])
        contribs = event.get('contribs', {})

        if len(contribs) == 0:
            ax.axis('off')
            ax.text(0.5, 0.5, 'Patient Zero',
                    ha='center', va='center', fontsize=10, style='italic')
        else:
            agents = list(contribs.keys())
            scores = np.array(list(contribs.values()), dtype=float)

            colors = [agent_colors[a] for a in agents]

            # Use narrower bars with proper spacing
            bar_width = 0.6  # Make bars narrower
            x_positions = range(len(agents))
            bars = ax.bar(x_positions, scores, color=colors, width=bar_width, 
                         edgecolor='black', linewidth=0.5, alpha=0.8)

            # Set appropriate x limits with padding
            if len(agents) > 1:
                ax.set_xlim(-0.8, len(agents) - 0.2)
            else:
                ax.set_xlim(-0.8, 0.8)

            # Improved label handling
            ax.set_xticks(x_positions)
            if len(agents) <= 3:
                # For few agents, use normal labels
                ax.set_xticklabels([f"Agent {i}" for i in agents], fontsize=9)
            else:
                # For many agents, use abbreviated labels with rotation
                ax.set_xticklabels([f"A{i}" for i in agents], rotation=45, ha='right', fontsize=8)

            # Add value labels on top of bars for clarity
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(scores)*0.01,
                       f'{score:.3f}', ha='center', va='bottom', fontsize=7)

            ax.set_ylabel("Contribution", fontsize=9)
            ax.set_title('Contributors', fontsize=11, pad=10)

            # Grid for readability
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Set y-axis to start from 0 for better visual comparison
            ax.set_ylim(bottom=0)

    # --- Legend row (bottom row) ---
    ax_legend = fig.add_subplot(gs[2, :])
    ax_legend.axis('off')
    legend_elements = [Patch(facecolor=agent_colors[i], label=f"Agent {i}") for i in range(total_agents)]
    ax_legend.legend(handles=legend_elements, loc='center', ncol=total_agents,
                     fontsize=9, frameon=False)

    fig.suptitle('Fault Detection Timeline and Contributor Scores',
                 fontsize=14, fontweight='bold', y=0.97)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # Adjusted margins for better label visibility

    out_path = os.path.join(logdir, 'fault_contributor_barchart.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.2)  # Added padding
    plt.show()
    print(f"Saved contributor bar chart to {out_path}")


def save_matrix_to_files(matrix, attacked_steps, attacked_agent_id, total_agents, logdir, filename):
    """
    Save matrix data to a CSV file for all timesteps.
    
    Args:
        matrix: List of timesteps, where each timestep contains n_agent data
        attacked_agent_id: ID of the attacked agent
        total_agents: Total number of agents
        logdir: Directory to save the file
    """
    filepath = os.path.join(logdir, filename)
    
    # Create header row
    # header = ["timestep", "attacked_agent"]
    header = ["timestep", "is_attacked", "attacked_agent"]
    for i in range(total_agents):
        header.append(f"agent_{i}")
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        for timestep, timestep_data in enumerate(matrix):
            is_attacked = 1 if timestep in attacked_steps else 0
            row = [timestep, is_attacked, attacked_agent_id]
            for i in range(total_agents):
                row.append(timestep_data[i])
            writer.writerow(row)
    
    print(f"Saved {len(matrix)} timestep matrices to {filepath}")


def run(config):
    maddpg = MADDPG.init_from_save(config.model_path, test_mode=True)

    # create a log directory under runs/<env_id>/<timestamp> using os and getcwd
    cwd = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join(cwd, 'runs', f"{config.map_name}_{'discrete' if maddpg.discrete_action else 'continuous'}", f"{timestamp}_seed_{config.seed}")
    os.makedirs(logdir, exist_ok=True)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # maddpg.prep_rollouts(device=DEVICE)
    maddpg.prep_training(device=DEVICE)

    # Read reference values from CSV files if provided
    ref_vals = [[] for _ in range(maddpg.nagents)]
    ref_std_devs = [[] for _ in range(maddpg.nagents)]

    for agent_id in range(maddpg.nagents):
        csv_filename = f"maddpg_taylor_error_atk_free_agent_{agent_id}.csv"
        csv_path = os.path.join(config.ref_val_dir, csv_filename)
        
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                if config.detection_method == 'mean_std':
                    # Use mean and std_dev columns
                    ref_vals[agent_id].append(float(row[2]))  # mean
                    ref_std_devs[agent_id].append(float(row[4]))  # std_dev
                elif config.detection_method == 'median_mad':
                    # Use median and MAD columns
                    ref_vals[agent_id].append(float(row[7]))  # median
                    ref_std_devs[agent_id].append(float(row[8]))  # MAD
                elif config.detection_method == 'diff':
                    # Use diff_mean and diff_std columns
                    ref_vals[agent_id].append(float(row[9]))  # diff_mean
                    ref_std_devs[agent_id].append(float(row[10]))  # diff_std
                else:
                    raise ValueError(f"Unknown detection method: {config.detection_method}")

    attacked_agent_id = config.attack_agent_id  # specify the agent to attack
    seed = config.seed

    env = create_environment(config)
    results_normal, _, frob_norms_normal, sec_dir_derivatives_normal, frob_norms_matrix_history_normal, _, action_influences_matrix_history_normal, second_order_action_influences_history_normal, observation_influences_matrix_history_normal, second_order_observation_influences_history_normal = get_episode_data(env, maddpg, config, logdir, ref_vals, ref_std_devs, config.detection_method, do_attack=False, atk_agent_id=attacked_agent_id, seed=seed)
    env.close()

    env = create_environment(config)
    results_attacked, attacked_steps, frob_norms_atk, sec_dir_derivatives_atk, frob_norms_matrix_history, fault_timeline, action_influences_matrix_history, second_order_action_influences_history, observation_influences_matrix_history, second_order_observation_influences_history = get_episode_data(env, maddpg, config, logdir, ref_vals, ref_std_devs, config.detection_method, do_attack=True, atk_agent_id=attacked_agent_id, seed=seed)
    env.close()
    save_matrix_to_files(results_attacked, attacked_steps, attacked_agent_id, maddpg.nagents, logdir, f'maddpg_taylor_error_atk_{attacked_agent_id}.csv')
    save_matrix_to_files(frob_norms_atk, attacked_steps, attacked_agent_id, maddpg.nagents, logdir, f'maddpg_frobenius_norms_atk_{attacked_agent_id}.csv')
    save_matrix_to_files(sec_dir_derivatives_atk, attacked_steps, attacked_agent_id, maddpg.nagents, logdir, f'maddpg_sec_dir_derivatives_atk_{attacked_agent_id}.csv')
    
    # Save action influences matrix data - need to convert from N×N matrices per timestep to per-agent time series
    action_influences_per_agent = []
    for i in range(maddpg.nagents):
        agent_i_influences = []
        for t in range(len(action_influences_matrix_history)):
            # For agent i, collect all influences from other agents at time t
            influences_at_t = [action_influences_matrix_history[t][i][j] for j in range(maddpg.nagents)]
            agent_i_influences.append(influences_at_t)
        action_influences_per_agent.append(agent_i_influences)
    
    # Save individual influence time series for each agent
    for i in range(maddpg.nagents):
        filename = f'maddpg_action_influences_on_agent_{i}_atk_{attacked_agent_id}.csv'
        save_matrix_to_files([action_influences_per_agent[i]], attacked_steps, attacked_agent_id, maddpg.nagents, logdir, filename)

    plot_results(results_attacked, attacked_steps, attacked_agent_id, ref_vals, ref_std_devs, logdir, config.detection_method)
    plot_frobs(frob_norms_normal, frob_norms_atk, attacked_steps, attacked_agent_id, logdir)
    plot_frob_norm_influences(frob_norms_matrix_history_normal, frob_norms_matrix_history, attacked_steps, attacked_agent_id, logdir)
    plot_sec_dir_derivatives(sec_dir_derivatives_normal, sec_dir_derivatives_atk, attacked_steps, attacked_agent_id, logdir)
    plot_action_influences(action_influences_matrix_history_normal, action_influences_matrix_history, attacked_steps, attacked_agent_id, logdir)
    plot_pairwise_action_influences(action_influences_matrix_history_normal, action_influences_matrix_history, attacked_steps, attacked_agent_id, logdir)
    plot_second_order_action_influences(second_order_action_influences_history_normal, second_order_action_influences_history, attacked_steps, attacked_agent_id, logdir)
    plot_pairwise_second_order_action_influences(second_order_action_influences_history_normal, second_order_action_influences_history, attacked_steps, attacked_agent_id, logdir)
    plot_observation_influences(observation_influences_matrix_history_normal, observation_influences_matrix_history, attacked_steps, attacked_agent_id, logdir)
    plot_pairwise_observation_influences(observation_influences_matrix_history_normal, observation_influences_matrix_history, attacked_steps, attacked_agent_id, logdir)
    plot_second_order_observation_influences(second_order_observation_influences_history_normal, second_order_observation_influences_history, attacked_steps, attacked_agent_id, logdir)
    plot_pairwise_second_order_observation_influences(second_order_observation_influences_history_normal, second_order_observation_influences_history, attacked_steps, attacked_agent_id, logdir)
    plot_fault_timeline(fault_timeline, maddpg.nagents, logdir)
    plot_fault_timeline_action_influences(fault_timeline, action_influences_matrix_history, maddpg.nagents, logdir)
    plot_fault_timeline_second_order_action_influences(fault_timeline, second_order_action_influences_history, maddpg.nagents, logdir)
    plot_fault_timeline_observation_influences(fault_timeline, observation_influences_matrix_history, maddpg.nagents, logdir)
    plot_fault_timeline_second_order_observation_influences(fault_timeline, second_order_observation_influences_history, maddpg.nagents, logdir)
    plot_contributor_barchart(fault_timeline, maddpg.nagents, logdir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("map_name", help="Name of environment")
    parser.add_argument("model_path",
                        help="model directory")
    parser.add_argument("--save_gifs", action="store_true",
                        help="Saves gif of each episode into model directory")
    parser.add_argument("--ref_val_dir", type=str, default='',)
    parser.add_argument("--attack_agent_id", type=int, default=0,)
    parser.add_argument("--atk_start_step", type=int, default=-math.inf)
    parser.add_argument("--atk_end_step", type=int, default=math.inf)
    parser.add_argument("--detection_method", type=str, default='mean_std', 
                        choices=['mean_std', 'median_mad', 'diff'],
                        help="Detection method to use: 'mean_std', 'median_mad', or 'diff'")
    parser.add_argument("--seed", type=int, default=23)

    config = parser.parse_args()

    run(config)
