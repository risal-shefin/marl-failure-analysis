"""Policy-related metrics for MAPPO analysis."""
from __future__ import annotations

from typing import Iterable, List

import torch

from MAPPO_MPE_main import Runner_MAPPO_MPE

from ..constants import torch_device


def _ensure_state_tensors(states: Iterable) -> List[torch.Tensor]:
    tensors = []
    for state in states:
        if isinstance(state, torch.Tensor):
            tensor = state.detach().clone().requires_grad_(True)
        else:
            tensor = torch.tensor(state, dtype=torch.float32, requires_grad=True)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        tensors.append(tensor)
    return tensors


def compute_pairwise_frob_norms(runner: Runner_MAPPO_MPE, states: Iterable) -> List[List[float]]:
    """Approximate pairwise influence between agents via Frobenius norms."""
    states_tensors = _ensure_state_tensors(states)
    # Flatten into global state expected by the critic
    global_state = torch.cat([tensor for tensor in states_tensors], dim=-1).squeeze(0)
    global_state = global_state.to(torch_device)

    values = runner.agent_n.compute_value(global_state).squeeze(-1)
    N = runner.args.N
    results: List[List[float]] = [[0.0 for _ in range(N)] for _ in range(N)]

    for i in range(N):
        grad_i = torch.autograd.grad(values[i], states_tensors[i].to(torch_device),
                                     create_graph=True, retain_graph=True)[0]
        for j in range(N):
            hessian_rows = []
            for k in range(grad_i.shape[-1]):
                second_grad = torch.autograd.grad(grad_i[..., k], states_tensors[j].to(torch_device),
                                                  retain_graph=True, allow_unused=True)[0]
                if second_grad is None:
                    second_grad = torch.zeros_like(states_tensors[j])
                hessian_rows.append(second_grad.reshape(-1))
            H = torch.stack(hessian_rows)
            results[i][j] = torch.norm(H, p='fro').item()

    # Normalize rows for stability
    for i in range(N):
        row_sum = sum(results[i]) + 1e-10
        results[i] = [val / row_sum for val in results[i]]

    return results


def compute_taylor_error_policy(runner: Runner_MAPPO_MPE, states: Iterable, epsilon: float = 0.01) -> List[float]:
    """Compute Taylor approximation error of MAPPO policy for each agent."""
    states_tensors = _ensure_state_tensors(states)
    delta_errors: List[float] = []

    for agent_id in range(runner.args.N):
        obs = states_tensors[agent_id].to(torch_device)
        action, dist = runner.agent_n.compute_action(obs, agent_id, evaluate=True, return_dist=True)
        target_val = dist.log_prob(action)
        grad_i = torch.autograd.grad(target_val, obs, create_graph=True, retain_graph=True)[0]
        eta_i = epsilon * grad_i.sign() / torch.max(grad_i.norm(p=2), torch.tensor(1e-6, device=grad_i.device))
        j_tilde = target_val + torch.dot(grad_i.flatten(), eta_i.flatten())
        perturbed_obs = obs + eta_i
        p_action, p_dist = runner.agent_n.compute_action(perturbed_obs, agent_id, evaluate=True, return_dist=True)
        j_perturbed = p_dist.log_prob(p_action)
        delta_errors.append(abs(j_perturbed - j_tilde).item())

    return delta_errors
