"""
Agent influence computation functions for analyzing multi-agent interactions.
"""
import torch
import numpy as np
from torch.autograd import Variable
from ..constants import torch_device


def compute_pairwise_action_influences(maddpg, obs, actions, action_spaces):
    """
    Compute direct influence of each agent's action on every other agent's Q-value.
    
    Args:
        maddpg: MADDPG agent
        obs: List of observations
        actions: List of actions
        action_spaces: List of action spaces
        
    Returns:
        N x N list where entry [i][j] represents || ∂Q_i/∂a_j ||_2
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
    
    Args:
        maddpg: MADDPG agent
        obs: List of observations
        actions: List of actions
        action_spaces: List of action spaces
        
    Returns:
        N x N list where entry [i][j] represents || ∂²Q_i/(∂a_j)² ||_F
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
    
    Args:
        maddpg: MADDPG agent
        obs: List of observations
        actions: List of actions
        action_spaces: List of action spaces
        
    Returns:
        N x N list where entry [i][j] represents || ∂Q_i/∂obs_j ||_2
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
    
    Args:
        maddpg: MADDPG agent
        obs: List of observations
        actions: List of actions
        action_spaces: List of action spaces
        
    Returns:
        N x N list where entry [i][j] represents || ∂²Q_i/(∂obs_j)² ||_F
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


def collect_agent_q_values(maddpg, obs, actions, action_spaces):
    """
    Return the critic output for each agent given observations and actions at a timestep.
    
    Args:
        maddpg: MADDPG agent
        obs: List of observations
        actions: List of actions
        action_spaces: List of action spaces
        
    Returns:
        List of Q-values for each agent
    """
    if maddpg.discrete_action:
        one_hot_actions = []
        for i, action in enumerate(actions):
            one_hot = np.zeros(action_spaces[i].n)
            one_hot[action] = 1.0
            one_hot_actions.append(one_hot)
        actions = one_hot_actions

    torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=False) for i in range(maddpg.nagents)]
    torch_actions = [Variable(torch.Tensor([actions[i]]).to(torch_device), requires_grad=False) for i in range(maddpg.nagents)]
    vf_in = torch.cat((*torch_obs, *torch_actions), dim=1)

    q_values = []
    for agent in maddpg.agents:
        critic_val = agent.critic(vf_in).mean().item()
        q_values.append(critic_val)

    return q_values