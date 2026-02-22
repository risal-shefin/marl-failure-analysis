"""
Metric computation functions for analyzing agent behavior and interactions.
"""
import torch
import numpy as np
from torch.autograd import Variable
from ..constants import torch_device


def compute_taylor_delta_policy(maddpg, obs, actions, action_spaces, epsilon):
    """
    Compute Taylor delta policy approximation errors for all agents.
    
    Args:
        maddpg: MADDPG agent
        obs: List of observations
        actions: List of actions
        action_spaces: List of action spaces
        epsilon: Perturbation magnitude
        
    Returns:
        List of delta errors for each agent
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

    delta_errors = []

    for i, agent_i in enumerate(maddpg.agents):
        action_logits_i = agent_i.policy(torch_obs[i])
        action_log_probs = torch.log_softmax(action_logits_i, dim=-1)
        max_action_idx = torch.argmax(action_log_probs, dim=-1)
        critic_val = action_log_probs.gather(-1, max_action_idx.unsqueeze(-1)).squeeze()
        grad_i = torch.autograd.grad(critic_val, torch_obs[i], create_graph=True, retain_graph=True)[0]

        eta_i = epsilon * grad_i.sign() / torch.max(grad_i.norm(p=2), torch.tensor(1e-6))
        
        # Second-order Taylor approximation: f(x + η) ≈ f(x) + ∇f(x)^T η + 0.5 η^T H η
        j_tilde = critic_val + torch.dot(grad_i.flatten(), eta_i.flatten())
        p_torch_obs_i = torch_obs[i] + eta_i
        p_action_logits_i = agent_i.policy(p_torch_obs_i)
        p_action_log_probs = torch.log_softmax(p_action_logits_i, dim=-1)
        p_max_action_idx = torch.argmax(p_action_log_probs, dim=-1)
        j_perturbed = p_action_log_probs.gather(-1, p_max_action_idx.unsqueeze(-1)).squeeze()
        delta_error = abs(j_perturbed - j_tilde).item()
        delta_errors.append(delta_error)

    return delta_errors


def compute_frob_norms(maddpg, obs, actions, action_spaces, vulnerable_agent_id):
    """
    Compute Frobenius norms of Hessian matrices for all agents.
    
    Args:
        maddpg: MADDPG agent
        obs: List of observations
        actions: List of actions
        action_spaces: List of action spaces
        vulnerable_agent_id: ID of the vulnerable agent
        
    Returns:
        List of Frobenius norms for each agent
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
    
    Args:
        maddpg: MADDPG agent
        obs: List of observations
        actions: List of actions
        action_spaces: List of action spaces
        
    Returns:
        N x N list where entry [i][j] approximates || ∂²v_i / (∂obs_i ∂obs_j) ||_F
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
        grad_i = torch.autograd.grad(critic_val, actions[i], create_graph=True, retain_graph=True)[0]

        for j in range(N):
            hessian_matrix = []
            for k in range(grad_i.shape[1]):
                second_grad = torch.autograd.grad(
                    grad_i[0, k],
                    actions[j],  # Change to actions[j] to compute cross-agent action Hessian
                    retain_graph=True,
                    allow_unused=True
                )[0]
                if second_grad is None:
                    second_grad = torch.zeros_like(torch_obs[j])
                hessian_matrix.append(second_grad.flatten())

            H = torch.stack(hessian_matrix) if len(hessian_matrix) > 0 else torch.zeros(1, 1)
            results[i][j] = H.norm(p='fro').item()

    return results


def compute_2nd_ord_dir_derivatives(maddpg, obs, actions, action_spaces, vulnerable_agent_id):
    """
    Compute second order directional derivatives.
    
    Args:
        maddpg: MADDPG agent
        obs: List of observations
        actions: List of actions
        action_spaces: List of action spaces
        vulnerable_agent_id: ID of the vulnerable agent
        
    Returns:
        List of second order directional derivatives for each agent
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
    
    results = []

    for i, agent_i in enumerate(maddpg.agents):
        critic_val = agent_i.critic(vf_in).mean()
        grad_i = torch.autograd.grad(critic_val, torch_obs[i], create_graph=True, retain_graph=True)[0]
        
        direction = grad_i.sign() / torch.max(grad_i.norm(p=2), torch.tensor(1e-6))
        
        directional_derivative = torch.sum(grad_i * direction)
        second_directional_derivative = torch.autograd.grad(
            directional_derivative, torch_obs[vulnerable_agent_id], retain_graph=True
        )[0]
        
        result = torch.norm(second_directional_derivative).item()
        results.append(result)

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

def collect_agent_q_value(maddpg, agent_id, obs, actions, action_spaces):
    """
    Return the critic output for the selected agent given observations and actions at a timestep.
    
    Args:
        maddpg: MADDPG agent
        agent_id: ID of the agent to get Q-value for
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
    return maddpg.agents[agent_id].critic(vf_in).mean().item()


def compute_pairwise_frob_svd_coupling_analysis(maddpg, obs, actions, action_spaces, epsilon=0.01, lam=1.0):
    """
    Compute SVD-based gradient coupling analysis for all agent pairs.

    Uses a composite perturbation direction to avoid the objective misalignment
    (damping effect) that arises when the gradient-boost direction for agent i
    conflicts with the direct gradient direction for agent j.  The composite
    direction is:

        d_optimal = ĝ_{v2} + λ * (H^T g / ||H^T g||)

    where ĝ_{v2} = ∇_{a_j} Q_i / ||∇_{a_j} Q_i|| is the normalised direct
    gradient of Q_i w.r.t. a_j and H^T g is the assist direction that boosts
    ∇_{a_i} Q_i.  The final perturbation is:

        a_j' = a_j + ε * d_optimal / ||d_optimal||

    For each pair (i, j):
    1. Compute cross-Hessian H = ∇_{a_j} ∇_{a_i} Q_i
    2. Compute Frobenius norm ||H||_F
    3. Compute composite direction: d_optimal = ĝ_{v2} + λ * (H^T g / ||H^T g||)
    4. Perturb a_j along d_optimal: a_j' = a_j + ε * d_optimal / ||d_optimal||
    5. Compute gradient shift: Δg = ||∇_{a_i} Q_i(a_j') - ∇_{a_i} Q_i(a_j)||_2
    6. Perturb a_i along sign of original gradient: a_i' = a_i - ε * sign(g)
    7. Compute critic value shift: ΔQ = Q_i(a_i', a_j') - Q_i(a_i, a_j)

    Args:
        maddpg: MADDPG agent
        obs: List of observations
        actions: List of actions
        action_spaces: List of action spaces
        epsilon: Perturbation magnitude (default: 0.01)
        lam: Trade-off coefficient λ between the direct function-increase term
             (ĝ_{v2}) and the adversarial gradient-assist term (H^T g direction).
             lam=0 reduces to a pure FGSM step on a_j; lam→∞ recovers the
             original sign(H^T g) boost direction. (default: 1.0)

    Returns:
        Dictionary mapping (agent_i, agent_j) -> {'frob_norm': float,
            'grad_norm': float, 'delta_g_norm': float,
            'delta_critic1': float, 'delta_critic2': float}
    """
    # Convert discrete actions to one-hot encoding
    if maddpg.discrete_action:
        one_hot_actions = []
        for i, action in enumerate(actions):
            one_hot = np.zeros(action_spaces[i].n)
            one_hot[action] = 1.0
            one_hot_actions.append(one_hot)
        actions = one_hot_actions

    torch_obs = [Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=True) 
                 for i in range(maddpg.nagents)]
    torch_actions = [Variable(torch.Tensor([actions[i]]).to(torch_device), requires_grad=True) 
                     for i in range(maddpg.nagents)]
    
    N = maddpg.nagents
    results = {}
    
    for i in range(N):
        # Compute base critic value and first gradient for agent i
        vf_in = torch.cat((*torch_obs, *torch_actions), dim=1)
        critic_val_i = maddpg.agents[i].critic(vf_in).mean() # Negate so that gradient ascent minimizes Q_i
        
        # First gradient: g = ∇_{a_i} Q_i
        grad_i = torch.autograd.grad(
            critic_val_i, 
            torch_actions[i], 
            create_graph=True, 
            retain_graph=True
        )[0]
        
        # Compute gradient magnitude (L2 norm)
        grad_norm = grad_i.norm(p=2).item()
        
        for j in range(N):
            # Compute cross-Hessian H = ∇_{a_j} ∇_{a_i} Q_i
            hessian_matrix = []
            for k in range(grad_i.shape[1]):
                second_grad = torch.autograd.grad(
                    grad_i[0, k],
                    torch_actions[j],
                    retain_graph=True,
                    allow_unused=True,
                    create_graph=True
                )[0]
                hessian_matrix.append(second_grad.flatten())
            
            H = torch.stack(hessian_matrix)
            
            # Compute Frobenius norm
            frob_norm = H.norm(p='fro').item()
            
            # # Perform SVD to get v_max (right singular vector of largest singular value)
            # U, S, Vt = torch.linalg.svd(H, full_matrices=False)
            # # v_max is the first column of V (first row of Vt)
            # v_max = Vt[0, :]  # Shape: [action_dim_j]
            
            # # Reshape v_max to match torch_actions[j] shape [1, action_dim_j]
            # v_max = v_max.unsqueeze(0)
            
            # # Perturb action j: a_j' = a_j + epsilon * v_max
            # perturbed_action_j = torch_actions[j] + epsilon * v_max

            # Compute H^T g (assist direction): shape [action_dim_j]
            H_T_g = H.T @ grad_i.T  # Shape: [action_dim_j, 1]
            H_T_g = H_T_g.squeeze()  # Shape: [action_dim_j]

            # --- Composite Objective (Issue 2: Objective Misalignment fix) ---
            # Compute the direct gradient g_{v2} = ∇_{a_j} Q_i.  When the pure
            # boost direction sign(H^T g) is negatively correlated with ∇_{a_j} Q_i
            # it steepens the hill for agent i while pushing the system down with
            # respect to agent j (damping effect).  The composite direction balances
            # the two objectives:
            #   d_optimal = ĝ_{v2} + λ * (H^T g / ||H^T g||)
            grad_j = torch.autograd.grad(
                critic_val_i,
                torch_actions[j],
                retain_graph=True,
                allow_unused=True
            )[0]

            # Normalise the direct gradient: ĝ_{v2}
            g_v2 = grad_j.flatten()
            g_v2_hat = g_v2 / torch.clamp(g_v2.norm(p=2), min=1e-8)

            # Normalise the assist direction: H^T g / ||H^T g||
            H_T_g_hat = H_T_g / torch.clamp(H_T_g.norm(p=2), min=1e-8)

            # Composite direction and unit-normalise before applying perturbation
            d_optimal = -g_v2_hat + lam * H_T_g_hat
            d_optimal_unit = d_optimal / torch.clamp(d_optimal.norm(p=2), min=1e-8)
            perturbed_action_j = torch_actions[j] + epsilon * d_optimal_unit.unsqueeze(0)
            # Clamp to action space bounds if continuous
            if not maddpg.discrete_action:
                action_low = torch.tensor(
                    action_spaces[j].low,
                    device=perturbed_action_j.device,
                    dtype=perturbed_action_j.dtype
                )
                action_high = torch.tensor(
                    action_spaces[j].high,
                    device=perturbed_action_j.device,
                    dtype=perturbed_action_j.dtype
                )
                perturbed_action_j = perturbed_action_j.clamp(action_low, action_high)
            
            # Compute gradient with perturbed action j
            torch_actions_perturbed = [
                torch_actions[idx].clone().detach().requires_grad_(True) if idx != j 
                else perturbed_action_j.detach().requires_grad_(True)
                for idx in range(N)
            ]
            
            vf_in_perturbed = torch.cat((*torch_obs, *torch_actions_perturbed), dim=1)
            critic_val_perturbed = maddpg.agents[i].critic(vf_in_perturbed).mean()
            
            grad_i_perturbed = torch.autograd.grad(
                critic_val_perturbed,
                torch_actions_perturbed[i],
                retain_graph=True
            )[0]

            # Compute ||Δg||_2
            delta_g = grad_i_perturbed - grad_i.detach()
            delta_g_norm = delta_g.norm(p=2).item()
            
            # FGSM-style perturbation for action_i using the original gradient sign.
            # Since sign(H^T g) only boosts grad_i without changing its sign direction,
            # sign(g') == sign(g) and recomputing the gradient is unnecessary.
            perturbed_action_i = torch_actions[i].detach() - epsilon * grad_i.detach().sign()
            
            # Clamp to action space bounds if continuous
            if not maddpg.discrete_action:
                action_low_i = torch.tensor(
                    action_spaces[i].low,
                    device=perturbed_action_i.device,
                    dtype=perturbed_action_i.dtype
                )
                action_high_i = torch.tensor(
                    action_spaces[i].high,
                    device=perturbed_action_i.device,
                    dtype=perturbed_action_i.dtype
                )
                perturbed_action_i = perturbed_action_i.clamp(action_low_i, action_high_i)
            
            # Compute critic values for delta_critic calculation
            # Original critic value: Q_i(a_i, a_j)
            with torch.no_grad():
                vf_in_original = torch.cat((*torch_obs, *[a.detach() for a in torch_actions]), dim=1)
                critic_original = maddpg.agents[i].critic(vf_in_original).mean().item()
                
                # delta_critic1: Only agent i perturbed with ORIGINAL gradient (no j perturbation)
                torch_actions_i_only_perturbed = [
                    perturbed_action_i if idx == i 
                    else torch_actions[idx].detach()
                    for idx in range(N)
                ]
                vf_in_i_only = torch.cat((*torch_obs, *torch_actions_i_only_perturbed), dim=1)
                critic_i_only = maddpg.agents[i].critic(vf_in_i_only).mean().item()
                delta_critic1 = critic_i_only - critic_original
                
                # delta_critic2: Both i and j perturbed
                torch_actions_fully_perturbed = [
                    perturbed_action_i if idx == i 
                    else perturbed_action_j.detach() if idx == j 
                    else torch_actions[idx].detach()
                    for idx in range(N)
                ]
                vf_in_fully_perturbed = torch.cat((*torch_obs, *torch_actions_fully_perturbed), dim=1)
                critic_perturbed = maddpg.agents[i].critic(vf_in_fully_perturbed).mean().item()
                delta_critic2 = critic_perturbed - critic_original
            
            results[(i, j)] = {
                'frob_norm': frob_norm,
                'grad_norm': grad_norm,
                'delta_g_norm': delta_g_norm,
                'delta_critic1': delta_critic1,
                'delta_critic2': delta_critic2
            }
    
    return results
