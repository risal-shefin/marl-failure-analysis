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
        N x N list where entry [i][j] approximates || ∂²v_i / (∂a_i ∂a_j) ||_F
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


def compute_pairwise_frob_svd_coupling_analysis(maddpg, obs, actions, action_spaces, epsilon=0.01):
    """
    Compute SVD-based gradient coupling analysis for all agent pairs.

    Uses an orthogonally-projected perturbation direction to isolate the
    pure second-order (cross-Hessian) effect of a_j on ∇_{a_i} Q_i, while
    eliminating any first-order influence of a_j on Q_i.

    The assist direction induced by the cross-Hessian is:

        d_assist = H^T (∇_{a_i} Q_i)

    where H = ∇_{a_j} ∇_{a_i} Q_i.  To discard the direct first-order
    effect of a_j on Q_i, d_assist is projected onto the subspace orthogonal
    to g_{v2} = ∇_{a_j} Q_i:

        d_orthogonal = d_assist - (d_assist^T g_{v2} / ||g_{v2}||^2) * g_{v2}

    By construction d_orthogonal^T g_{v2} = 0, so perturbations along
    d_orthogonal cause no first-order change in Q_i via a_j, and any
    observed effect on ∇_{a_i} Q_i is attributable solely to the
    cross-Hessian coupling.  The final perturbation is:

        a_j' = a_j + ε * d_orthogonal / ||d_orthogonal||

    For each pair (i, j):
    1. Compute cross-Hessian H = ∇_{a_j} ∇_{a_i} Q_i
    2. Compute Frobenius norm ||H||_F
    3. Compute assist direction: d_assist = H^T g,  g = ∇_{a_i} Q_i
    4. Project out the first-order component: d_orthogonal = d_assist - proj_{g_{v2}} d_assist
    5. Perturb a_j along d_orthogonal: a_j' = a_j + ε * d_orthogonal / ||d_orthogonal||
    6. Compute gradient shift: Δg = ||∇_{a_i} Q_i(a_j') - ∇_{a_i} Q_i(a_j)||_2
    7. Perturb a_i along sign of original gradient: a_i' = a_i - ε * sign(g)
    8. Compute critic value shift: ΔQ = Q_i(a_i', a_j') - Q_i(a_i, a_j)

    Args:
        maddpg: MADDPG agent
        obs: List of observations
        actions: List of actions
        action_spaces: List of action spaces
        epsilon: Perturbation magnitude (default: 0.01)

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

            # --- Orthogonal Projection (discard first-order effect of a_j) ---
            # Compute the direct gradient g_{v2} = ∇_{a_j} Q_i and project
            # d_assist = H^T g onto the subspace orthogonal to g_{v2}.  This
            # ensures that perturbations along the resulting direction cause no
            # first-order change in Q_i via a_j, so any observed shift in
            # ∇_{a_i} Q_i is attributable solely to the cross-Hessian coupling.
            grad_j = torch.autograd.grad(
                critic_val_i,
                torch_actions[j],
                retain_graph=True,
                allow_unused=True
            )[0]

            # d_assist = H^T g  (already computed as H_T_g)
            d_assist = H_T_g  # Shape: [action_dim_j]

            # g_{v2} = ∇_{a_j} Q_i
            g_v2 = grad_j.flatten()  # Shape: [action_dim_j]

            # Project d_assist onto subspace orthogonal to g_{v2}:
            #   d_orthogonal = d_assist - (d_assist · g_{v2} / ||g_{v2}||^2) * g_{v2}
            g_v2_norm_sq = torch.clamp((g_v2 * g_v2).sum(), min=1e-16)
            projection_coeff = (d_assist * g_v2).sum() / g_v2_norm_sq
            d_orthogonal = d_assist - projection_coeff * g_v2

            # Unit-normalise and apply perturbation
            d_orthogonal_unit = d_orthogonal / torch.clamp(d_orthogonal.norm(p=2), min=1e-8)
            perturbed_action_j = torch_actions[j] + epsilon * d_orthogonal_unit.unsqueeze(0)
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

            # Compute ||g'||_2 (perturbed gradient norm)
            perturbed_grad_norm = grad_i_perturbed.norm(p=2).item()
            
            # FGSM-style perturbation for action_i using the original gradient sign.
            # Since sign(H^T g) only boosts grad_i without changing its sign direction,
            # sign(g') == sign(g) and recomputing the gradient is unnecessary.
            # perturbed_action_i = torch_actions[i].detach() - epsilon * grad_i.detach().sign()
            perturbed_action_i = torch_actions[i].detach() - epsilon * grad_i.detach()
            perturbed_action_i2 = torch_actions[i].detach() - epsilon * grad_i_perturbed.detach()
            
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
                perturbed_action_i2 = perturbed_action_i2.clamp(action_low_i, action_high_i)
            
            # Compute critic values for delta_critic calculation
            # Original critic value: Q_i(a_i, a_j)
            with torch.no_grad():
                vf_in_original = torch.cat((*torch_obs, *[a.detach() for a in torch_actions]), dim=1)
                critic_original = maddpg.agents[i].critic(vf_in_original).mean().item()
                
                # delta_critic_j_only: Only agent j perturbed along orthogonal direction (verify ortho projection)
                torch_actions_j_only_perturbed = [
                    perturbed_action_j.detach() if idx == j
                    else torch_actions[idx].detach()
                    for idx in range(N)
                ]
                vf_in_j_only = torch.cat((*torch_obs, *torch_actions_j_only_perturbed), dim=1)
                critic_j_only = maddpg.agents[i].critic(vf_in_j_only).mean().item()
                delta_critic_j_only = critic_j_only - critic_original

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
                    perturbed_action_i2 if idx == i 
                    else perturbed_action_j.detach() if idx == j 
                    else torch_actions[idx].detach()
                    for idx in range(N)
                ]
                vf_in_fully_perturbed = torch.cat((*torch_obs, *torch_actions_fully_perturbed), dim=1)
                critic_perturbed = maddpg.agents[i].critic(vf_in_fully_perturbed).mean().item()
                delta_critic2 = critic_perturbed - critic_original

                # if delta_critic2 >= delta_critic1 and delta_g_norm > 0.1 and i != j:
                #     print(" Delta grad norm ", delta_g_norm)
                #     print(" Perturb Grad Norm: ", perturbed_grad_norm)
                #     print("\n Grad I: ", grad_i.detach().cpu().numpy())
                #     print(" Grad I Perturbed: ", grad_i_perturbed.detach().cpu().numpy())
                #     print("\n Torch Action I: ", torch_actions[i].detach().cpu().numpy())
                #     print(" Perturbed Torch Action I: ", perturbed_action_i.detach().cpu().numpy())
                #     print("\n Torch Action J: ", torch_actions[j].detach().cpu().numpy())
                #     print(" Perturbed Torch Action J: ", perturbed_action_j.detach().cpu().numpy())
                #     print("\n Delta Critic 1 ", delta_critic1)
                #     print(" Delta Critic 2 ", delta_critic2)
                #     print(" Delta Critic J only ", delta_critic_j_only)

                #     cos_sim = torch.nn.functional.cosine_similarity(
                #         grad_i.flatten(), grad_i_perturbed.flatten(), dim=0
                #     ).item()
                #     print("cos_sim(g, g_perturbed):", cos_sim)
                #     print("\n")
            
            results[(i, j)] = {
                'frob_norm': frob_norm,
                'grad_norm': grad_norm,
                'delta_g_norm': delta_g_norm,
                'perturbed_grad_norm': perturbed_grad_norm,
                'delta_critic_j_only': delta_critic_j_only,
                'delta_critic1': delta_critic1,
                'delta_critic2': delta_critic2
            }
    
    return results


def compute_pairwise_svd_gradient_shift(maddpg, obs, actions, epsilon=0.01):
    """
    Compute cross-Hessian Frobenius norm and SVD-directed gradient shift for
    every agent pair (i, j).

    For each pair:
      1. Build cross-Hessian  H = ∇_{a_j} ∇_{a_i} Q_i  (shape: dim_i × dim_j)
      2. Compute Frobenius norm  ||H||_F
      3. Take top right singular vector v_max of H — the direction in a_j space
         that maximally rotates ∇_{a_i} Q_i
      4. Perturb  a_j' = a_j + ε * v_max
      5. Measure  ||Δg||_2 = ||∇_{a_i} Q_i(a_j') - ∇_{a_i} Q_i(a_j)||_2

    Continuous action spaces only.

    Args:
        maddpg:   MADDPG agent
        obs:      list of per-agent observations
        actions:  list of per-agent actions (numpy arrays)
        epsilon:  perturbation magnitude along v_max (default: 0.01)

    Returns:
        dict mapping (agent_i, agent_j) -> {'frob_norm': float, 'delta_g_norm': float}
    """
    N = maddpg.nagents

    torch_obs = [
        Variable(torch.Tensor([obs[i]]).to(torch_device), requires_grad=True)
        for i in range(N)
    ]
    torch_actions = [
        Variable(torch.Tensor([actions[i]]).to(torch_device), requires_grad=True)
        for i in range(N)
    ]

    results = {}

    for i in range(N):
        vf_in = torch.cat((*torch_obs, *torch_actions), dim=1)
        critic_val_i = maddpg.agents[i].critic(vf_in).mean()

        # Base gradient  g = ∇_{a_i} Q_i
        grad_i = torch.autograd.grad(
            critic_val_i, torch_actions[i],
            create_graph=True, retain_graph=True
        )[0]  # shape: [1, dim_i]

        for j in range(N):
            # Cross-Hessian  H[k, :] = ∂(grad_i[k]) / ∂a_j
            rows = []
            for k in range(grad_i.shape[1]):
                row = torch.autograd.grad(
                    grad_i[0, k], torch_actions[j],
                    retain_graph=True, allow_unused=True, create_graph=False
                )[0]
                rows.append(row.flatten())

            H = torch.stack(rows)  # [dim_i, dim_j]
            frob_norm = H.norm(p='fro').item()

            # Top right singular vector: direction in a_j space that most shifts g_i
            _, _, Vt = torch.linalg.svd(H.detach(), full_matrices=False)
            v_max = Vt[0].unsqueeze(0)  # [1, dim_j]

            # Perturbed action j
            perturbed_aj = (torch_actions[j].detach() + epsilon * v_max).requires_grad_(True)

            torch_actions_p = [
                torch_actions[idx].clone().detach().requires_grad_(True) if idx != j
                else perturbed_aj
                for idx in range(N)
            ]

            vf_in_p = torch.cat((*torch_obs, *torch_actions_p), dim=1)
            grad_i_p = torch.autograd.grad(
                maddpg.agents[i].critic(vf_in_p).mean(),
                torch_actions_p[i],
                retain_graph=False
            )[0]

            delta_g_norm = (grad_i_p - grad_i.detach()).norm(p=2).item()

            results[(i, j)] = {
                'frob_norm': frob_norm,
                'delta_g_norm': delta_g_norm,
                # numpy array shape [1, dim_j] — usable to replay perturbed episodes
                'perturbed_action_j': perturbed_aj.detach().cpu().numpy(),
            }

    return results
