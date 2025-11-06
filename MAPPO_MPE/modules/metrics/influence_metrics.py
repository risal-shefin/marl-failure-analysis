"""
Agent influence computation functions for analyzing multi-agent interactions.
"""
import torch
import numpy as np
from torch.autograd import Variable
from ..constants import torch_device


def compute_pairwise_action_influences(mappo, state, obs, actions, action_spaces):
    """
    Compute direct influence of each agent's action on every other agent's Q-value using MAPPO's centralized Q.
    
    Args:
        mappo: MAPPO agent
        state: Global state (for centralized value function)
        obs: List of observations for each agent
        actions: List of actions
        action_spaces: List of action spaces (kept for API compatibility)
        
    Returns:
        N x N list where entry [i][j] represents || ∂Q_i/∂a_j ||_2
    """
    if not mappo.use_central_q:
        raise RuntimeError("Central Q network is required for compute_pairwise_action_influences. Enable it by setting use_central_q to True.")
    
    # Convert observations and state to tensors
    torch_obs = [torch.tensor([obs[i]], dtype=torch.float32, requires_grad=False) for i in range(mappo.N)]
    state_tensor = torch.tensor([state], dtype=torch.float32, requires_grad=False)
    
    # Create action tensors with gradient tracking - use one-hot encoding for gradients
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
        
        for j in range(N):
            # Compute gradient of Q_i with respect to action of agent j
            grad_qi_aj = torch.autograd.grad(
                critic_val,
                actions_one_hot[j],
                retain_graph=True,
                allow_unused=True
            )[0]

            # Compute L2 norm of the gradient (direct influence magnitude)
            influence_magnitude = grad_qi_aj.norm(p=2).item()
            results[i][j] = influence_magnitude

    return results


def compute_pairwise_action_directional_second_derivatives(mappo, state, obs, actions, action_spaces):
    """
    Compute second-order directional derivative g^T H g for each agent's action influence using MAPPO's centralized Q.
    
    This computes the directional second derivative along the gradient direction,
    where g = ∂Q_i/∂a_j and H = ∂²Q_i/(∂a_j)².
    
    Args:
        mappo: MAPPO agent
        state: Global state (for centralized value function)
        obs: List of observations for each agent
        actions: List of actions
        action_spaces: List of action spaces (kept for API compatibility)
        
    Returns:
        N x N list where entry [i][j] represents g^T H g for the influence of 
        agent j's action on agent i's Q-value
    """
    if not mappo.use_central_q:
        raise RuntimeError("Central Q network is required for compute_pairwise_action_directional_second_derivatives. Enable it by setting use_central_q to True.")
    
    # Convert observations and state to tensors
    torch_obs = [torch.tensor([obs[i]], dtype=torch.float32, requires_grad=False) for i in range(mappo.N)]
    state_tensor = torch.tensor([state], dtype=torch.float32, requires_grad=False)
    
    # Create action tensors with gradient tracking - use one-hot encoding for gradients
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
        
        # Get Q-value for agent i (negate for attacker's perspective)
        q_values = mappo.central_q(q_inputs)
        critic_val = -q_values[i].mean()
        
        for j in range(N):
            # Compute first-order gradient g = ∂Q_i/∂a_j
            grad_qi_aj = torch.autograd.grad(
                critic_val,
                actions_one_hot[j],
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]
                
            # Flatten the gradient for easier manipulation
            g = grad_qi_aj.flatten()
            
            # Compute Hessian-vector product: H * g
            # This is more efficient than computing the full Hessian matrix
            hvp = torch.autograd.grad(
                grad_qi_aj,
                actions_one_hot[j],
                grad_outputs=grad_qi_aj,
                retain_graph=True,
                allow_unused=True
            )[0]
                
            # Flatten the Hessian-vector product
            hvp_flat = hvp.flatten()
            
            # Compute g^T H g (directional second derivative)
            directional_second_derivative = torch.dot(g, hvp_flat).item()
            results[i][j] = directional_second_derivative

    return results


def compute_second_order_action_influences(mappo, state, obs, actions, action_spaces):
    """
    Compute second-order action influences between agents using MAPPO's centralized Q.
    
    Args:
        mappo: MAPPO agent
        state: Global state (for centralized value function)
        obs: List of observations for each agent
        actions: List of actions
        action_spaces: List of action spaces (kept for API compatibility)
        
    Returns:
        N x N list where entry [i][j] represents || ∂²Q_i/(∂a_j)² ||_F
    """
    if not mappo.use_central_q:
        raise RuntimeError("Central Q network is required for compute_second_order_action_influences. Enable it by setting use_central_q to True.")
    
    # Convert observations and state to tensors
    torch_obs = [torch.tensor([obs[i]], dtype=torch.float32, requires_grad=False) for i in range(mappo.N)]
    state_tensor = torch.tensor([state], dtype=torch.float32, requires_grad=False)
    
    # Create action tensors with gradient tracking - use one-hot encoding for gradients
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
        
        for j in range(N):
            # Compute first-order gradient ∂Q_i/∂a_j
            grad_qi_aj = torch.autograd.grad(
                critic_val,
                actions_one_hot[j],
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]
                
            # Compute second-order gradient ∂²Q_i/(∂a_j)²
            hessian_matrix = []
            for k in range(grad_qi_aj.shape[1]):  # iterate over action dimensions
                second_grad = torch.autograd.grad(
                    grad_qi_aj[0, k],
                    actions_one_hot[j],  # Same action variable j
                    retain_graph=True,
                    allow_unused=True
                )[0]
                
                hessian_matrix.append(second_grad.flatten())
            
            H = torch.stack(hessian_matrix)
            # Compute Frobenius norm of the Hessian matrix
            second_order_influence = H.norm(p='fro').item()
            results[i][j] = second_order_influence

    return results


def compute_pairwise_observation_influences(mappo, state, obs, actions, action_spaces):
    """
    Compute direct influence of each agent's observation on every other agent's Q-value using MAPPO's centralized Q.
    
    Args:
        mappo: MAPPO agent
        state: Global state (for centralized value function)
        obs: List of observations for each agent
        actions: List of actions
        action_spaces: List of action spaces (kept for API compatibility)
        
    Returns:
        N x N list where entry [i][j] represents || ∂Q_i/∂obs_j ||_2
    """
    if not mappo.use_central_q:
        raise RuntimeError("Central Q network is required for compute_pairwise_observation_influences. Enable it by setting use_central_q to True.")
    
    # Convert observations and state to tensors with gradient tracking
    torch_obs = [torch.tensor([obs[i]], dtype=torch.float32, requires_grad=True) for i in range(mappo.N)]
    state_tensor = torch.tensor([state], dtype=torch.float32, requires_grad=False)
    
    # Create action tensors
    actions_tensor = [torch.tensor([actions[i]], dtype=torch.long, requires_grad=False) for i in range(mappo.N)]

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
        
        # Prepare joint action (one-hot encoded)
        action_one_hot = torch.nn.functional.one_hot(torch.cat(actions_tensor), num_classes=mappo.action_dim).float()
        joint_action = action_one_hot.reshape(1, mappo.N * mappo.action_dim).repeat(N, 1)
        q_inputs = torch.cat([critic_inputs, joint_action], dim=-1)
        
        # Get Q-value for agent i
        q_values = mappo.central_q(q_inputs)
        critic_val = q_values[i].mean()
        
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


def compute_second_order_observation_influences(mappo, state, obs, actions, action_spaces):
    """
    Compute second-order observation influences between agents using MAPPO's centralized Q.
    
    Args:
        mappo: MAPPO agent
        state: Global state (for centralized value function)
        obs: List of observations for each agent
        actions: List of actions
        action_spaces: List of action spaces (kept for API compatibility)
        
    Returns:
        N x N list where entry [i][j] represents || ∂²Q_i/(∂obs_j)² ||_F
    """
    if not mappo.use_central_q:
        raise RuntimeError("Central Q network is required for compute_second_order_observation_influences. Enable it by setting use_central_q to True.")
    
    # Convert observations and state to tensors with gradient tracking
    torch_obs = [torch.tensor([obs[i]], dtype=torch.float32, requires_grad=True) for i in range(mappo.N)]
    state_tensor = torch.tensor([state], dtype=torch.float32, requires_grad=False)
    
    # Create action tensors
    actions_tensor = [torch.tensor([actions[i]], dtype=torch.long, requires_grad=False) for i in range(mappo.N)]

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
        
        # Prepare joint action (one-hot encoded)
        action_one_hot = torch.nn.functional.one_hot(torch.cat(actions_tensor), num_classes=mappo.action_dim).float()
        joint_action = action_one_hot.reshape(1, mappo.N * mappo.action_dim).repeat(N, 1)
        q_inputs = torch.cat([critic_inputs, joint_action], dim=-1)
        
        # Get Q-value for agent i
        q_values = mappo.central_q(q_inputs)
        critic_val = q_values[i].mean()
        
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
