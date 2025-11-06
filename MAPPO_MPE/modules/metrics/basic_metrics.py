"""
Metric computation functions for analyzing agent behavior and interactions.
"""
import torch
import numpy as np
from torch.autograd import Variable
from ..constants import torch_device


def compute_taylor_delta_policy(mappo, obs, epsilon=0.01):
    """
    Compute Taylor delta policy approximation errors for all agents using MAPPO.
    
    Args:
        mappo: MAPPO agent
        obs: List of observations for each agent
        epsilon: Perturbation magnitude (default: 0.01)
        
    Returns:
        List of delta errors for each agent
    """
    delta_errors = []

    for i in range(mappo.N):
        # Convert observation to tensor with gradient tracking
        obs_tensor = torch.tensor([obs[i]], dtype=torch.float32, requires_grad=True)
        
        # Get action and distribution from policy
        action, dist = mappo.compute_action(obs_tensor, i, evaluate=True, return_dist=True)
        
        # Use log probability as the target value
        target_val = dist.log_prob(action)
        
        # Compute gradient with respect to observation
        grad_i = torch.autograd.grad(target_val, obs_tensor, create_graph=True, retain_graph=True)[0]

        # Compute perturbation direction
        eta_i = epsilon * grad_i.sign() / torch.max(grad_i.norm(p=2), torch.tensor(1e-6))
        
        # First-order Taylor approximation: f(x + η) ≈ f(x) + ∇f(x)^T η
        j_tilde = target_val + torch.dot(grad_i.flatten(), eta_i.flatten())
        
        # Compute perturbed observation
        p_obs = obs_tensor + eta_i
        p_action, p_dist = mappo.compute_action(p_obs, i, evaluate=True, return_dist=True)
        j_perturbed = p_dist.log_prob(p_action)
        
        # Calculate approximation error
        delta_error = abs(j_perturbed - j_tilde).item()
        delta_errors.append(delta_error)

    return delta_errors


def compute_frob_norms(mappo, state, obs, actions, action_spaces, vulnerable_agent_id):
    """
    Compute Frobenius norms of Hessian matrices for all agents using MAPPO's centralized Q.
    
    Args:
        mappo: MAPPO agent
        state: Global state (for centralized value function)
        obs: List of observations for each agent
        actions: List of actions
        action_spaces: List of action spaces (kept for API compatibility)
        vulnerable_agent_id: ID of the vulnerable agent
        
    Returns:
        List of Frobenius norms for each agent
    """
    if not mappo.use_central_q:
        raise RuntimeError("Central Q network is required for compute_frob_norms. Enable it by setting use_central_q to True.")
    
    # Convert observations and state to tensors
    torch_obs = [torch.tensor([obs[i]], dtype=torch.float32, requires_grad=True) for i in range(mappo.N)]
    state_tensor = torch.tensor([state], dtype=torch.float32, requires_grad=True)
    
    # Create action tensors with gradient tracking
    actions_tensor = [torch.tensor([actions[i]], dtype=torch.long, requires_grad=False) for i in range(mappo.N)]
    
    results = []

    for i in range(mappo.N):
        # Prepare inputs for centralized Q network
        critic_inputs = []
        s = state_tensor.repeat(mappo.N, 1)  # (N, state_dim)
        critic_inputs.append(s)
        
        if mappo.add_agent_id:
            critic_inputs.append(torch.eye(mappo.N))
        
        critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)
        
        # Prepare joint action (one-hot encoded)
        action_one_hot = torch.nn.functional.one_hot(torch.cat(actions_tensor), num_classes=mappo.action_dim).float()
        joint_action = action_one_hot.reshape(1, mappo.N * mappo.action_dim).repeat(mappo.N, 1)
        q_inputs = torch.cat([critic_inputs, joint_action], dim=-1)
        
        # Get Q-value for agent i
        q_values = mappo.central_q(q_inputs)
        critic_val = q_values[i].mean()
        
        # Compute gradient with respect to the vulnerable agent's observation
        grad_i = torch.autograd.grad(critic_val, torch_obs[vulnerable_agent_id], create_graph=True, retain_graph=True)[0]

        # Compute Hessian matrix
        hessian_matrix = []
        for k in range(grad_i.shape[1]):
            # Compute ∂²Q/∂obs_vulnerable[k]∂obs_vulnerable
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


def compute_pairwise_frob_norms(mappo, state, obs, actions, action_spaces):
    """
    Compute Frobenius norms of cross-agent Hessian blocks for all (i, j) using MAPPO's centralized Q.
    
    Args:
        mappo: MAPPO agent
        state: Global state (for centralized value function)
        obs: List of observations for each agent
        actions: List of actions
        action_spaces: List of action spaces (kept for API compatibility)
        
    Returns:
        N x N list where entry [i][j] approximates || ∂²Q_i / (∂a_i ∂a_j) ||_F
    """
    if not mappo.use_central_q:
        raise RuntimeError("Central Q network is required for compute_pairwise_frob_norms. Enable it by setting use_central_q to True.")
    
    # Convert observations and state to tensors
    torch_obs = [torch.tensor([obs[i]], dtype=torch.float32, requires_grad=False) for i in range(mappo.N)]
    state_tensor = torch.tensor([state], dtype=torch.float32, requires_grad=False)
    
    # Create action tensors with gradient tracking - need to convert to one-hot for gradients
    actions_one_hot = []
    for i in range(mappo.N):
        action_oh = torch.nn.functional.one_hot(torch.tensor([actions[i]]), num_classes=mappo.action_dim).float()
        action_oh.requires_grad = True
        actions_one_hot.append(action_oh)

    N = mappo.N
    results = [[0.0 for _ in range(N)] for _ in range(N)]

    for i in range(N):
        # Prepare inputs for centralized Q network for agent i
        critic_inputs = []
        s = state_tensor.repeat(N, 1)  # (N, state_dim)
        critic_inputs.append(s)
        
        if mappo.add_agent_id:
            critic_inputs.append(torch.eye(N))
        
        critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)
        
        # Prepare joint action
        joint_action = torch.cat([a for a in actions_one_hot], dim=-1).repeat(N, 1)
        q_inputs = torch.cat([critic_inputs, joint_action], dim=-1)
        
        # Get Q-value for agent i
        q_values = mappo.central_q(q_inputs)
        critic_val = q_values[i].mean()
        
        # Compute gradient with respect to agent i's action
        grad_i = torch.autograd.grad(critic_val, actions_one_hot[i], create_graph=True, retain_graph=True)[0]

        for j in range(N):
            hessian_matrix = []
            for k in range(grad_i.shape[1]):
                second_grad = torch.autograd.grad(
                    grad_i[0, k],
                    actions_one_hot[j],  # Compute cross-agent action Hessian
                    retain_graph=True,
                    allow_unused=True
                )[0]
                hessian_matrix.append(second_grad.flatten())

            H = torch.stack(hessian_matrix) if len(hessian_matrix) > 0 else torch.zeros(1, 1)
            results[i][j] = H.norm(p='fro').item()

    return results


def compute_2nd_ord_dir_derivatives(mappo, state, obs, actions, action_spaces, vulnerable_agent_id):
    """
    Compute second order directional derivatives using MAPPO's centralized Q.
    
    Args:
        mappo: MAPPO agent
        state: Global state (for centralized value function)
        obs: List of observations for each agent
        actions: List of actions
        action_spaces: List of action spaces (kept for API compatibility)
        vulnerable_agent_id: ID of the vulnerable agent
        
    Returns:
        List of second order directional derivatives for each agent
    """
    if not mappo.use_central_q:
        raise RuntimeError("Central Q network is required for compute_2nd_ord_dir_derivatives. Enable it by setting use_central_q to True.")
    
    # Convert observations and state to tensors
    torch_obs = [torch.tensor([obs[i]], dtype=torch.float32, requires_grad=True) for i in range(mappo.N)]
    state_tensor = torch.tensor([state], dtype=torch.float32, requires_grad=True)
    
    # Create action tensors
    actions_tensor = [torch.tensor([actions[i]], dtype=torch.long, requires_grad=False) for i in range(mappo.N)]
    
    results = []

    for i in range(mappo.N):
        # Prepare inputs for centralized Q network
        critic_inputs = []
        s = state_tensor.repeat(mappo.N, 1)  # (N, state_dim)
        critic_inputs.append(s)
        
        if mappo.add_agent_id:
            critic_inputs.append(torch.eye(mappo.N))
        
        critic_inputs = torch.cat([x for x in critic_inputs], dim=-1)
        
        # Prepare joint action (one-hot encoded)
        action_one_hot = torch.nn.functional.one_hot(torch.cat(actions_tensor), num_classes=mappo.action_dim).float()
        joint_action = action_one_hot.reshape(1, mappo.N * mappo.action_dim).repeat(mappo.N, 1)
        q_inputs = torch.cat([critic_inputs, joint_action], dim=-1)
        
        # Get Q-value for agent i
        q_values = mappo.central_q(q_inputs)
        critic_val = q_values[i].mean()
        
        # Compute gradient with respect to agent i's observation
        grad_i = torch.autograd.grad(critic_val, torch_obs[i], create_graph=True, retain_graph=True)[0]
        
        # Compute directional derivative direction
        direction = grad_i.sign() / torch.max(grad_i.norm(p=2), torch.tensor(1e-6))
        
        # Compute directional derivative
        directional_derivative = torch.sum(grad_i * direction)
        
        # Compute second directional derivative with respect to vulnerable agent's observation
        second_directional_derivative = torch.autograd.grad(
            directional_derivative, torch_obs[vulnerable_agent_id], retain_graph=True
        )[0]
        
        result = torch.norm(second_directional_derivative).item()
        results.append(result)

    return results

def collect_agent_q_values(mappo, state, obs, actions, action_spaces):
    """
    Return the centralized Q-value for each agent given state, observations and actions at a timestep.
    
    Args:
        mappo: MAPPO agent
        state: Global state (for centralized value function)
        obs: List of observations for each agent
        actions: List of actions
        action_spaces: List of action spaces (kept for API compatibility)
        
    Returns:
        List of Q-values for each agent
    """
    if not mappo.use_central_q:
        raise RuntimeError("Central Q network is required for collect_agent_q_values. Enable it by setting use_central_q to True.")
    
    # Use centralized Q-value function
    q_values = mappo.get_central_q(state, actions)
    return q_values.tolist()


def collect_agent_q_value(mappo, agent_id, state, obs, actions, action_spaces):
    """
    Return the centralized Q-value for the selected agent given state, observations and actions at a timestep.
    
    Args:
        mappo: MAPPO agent
        agent_id: ID of the agent to get Q-value for
        state: Global state (for centralized value function)
        obs: List of observations for each agent
        actions: List of actions
        action_spaces: List of action spaces (kept for API compatibility)
        
    Returns:
        Q-value for the specified agent
    """
    if not mappo.use_central_q:
        # Fall back to value function if central Q is not available
        return mappo.get_value(state)[agent_id]
    
    # Use centralized Q-value function
    q_values = mappo.get_central_q(state, actions)
    return q_values[agent_id]
