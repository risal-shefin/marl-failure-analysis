"""Value-based helpers for MAPPO analysis."""
from __future__ import annotations

from typing import Iterable, List

import torch

from MAPPO_MPE_main import Runner_MAPPO_MPE

from ..constants import torch_device


def _flatten_states(states: Iterable) -> torch.Tensor:
    tensors = []
    for state in states:
        if isinstance(state, torch.Tensor):
            tensor = state.detach().to(torch_device)
        else:
            tensor = torch.tensor(state, dtype=torch.float32, device=torch_device)
        if tensor.dim() > 1:
            tensor = tensor.view(-1)
        tensors.append(tensor)
    return torch.cat(tensors, dim=0)


def collect_agent_values(runner: Runner_MAPPO_MPE, states: Iterable) -> List[float]:
    """Collect critic values per agent for the given joint state."""
    global_state = _flatten_states(states)
    values = runner.agent_n.compute_value(global_state).detach().cpu().numpy().squeeze(-1)
    return values.tolist()


def collect_agent_value(runner: Runner_MAPPO_MPE, agent_id: int, states: Iterable) -> float:
    """Collect critic value for a specific agent."""
    return collect_agent_values(runner, states)[agent_id]
