"""Centralized Q network implementation."""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from harl.models.base.cnn import CNNBase
from harl.models.base.mlp import MLPBase
from harl.utils.envs_tools import check, get_shape_from_obs_space
from harl.utils.models_tools import get_active_func, get_init_method, init


class CentralizedQNet(nn.Module):
    """Centralized Q network that conditions on global state and agent action."""

    def __init__(self, args, cent_obs_space, act_space, device=torch.device("cpu")):
        super().__init__()

        self.hidden_sizes = args["hidden_sizes"]
        self.initialization_method = args["initialization_method"]
        self.activation_func = args["activation_func"]
        self.tpdv = dict(dtype=torch.float32, device=device)

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base_cls = CNNBase if len(cent_obs_shape) == 3 else MLPBase
        self.base = base_cls(args, cent_obs_shape)

        if isinstance(act_space, (list, tuple)):
            self.action_spaces: List = list(act_space)
        else:
            self.action_spaces = [act_space]

        self.action_info = []
        self.total_raw_action_dim = 0
        self.total_encoded_action_dim = 0
        for space in self.action_spaces:
            action_type = space.__class__.__name__
            if action_type == "Box":
                dim = space.shape[0]
                encoded_dim = dim
                self.action_info.append(
                    {
                        "type": action_type,
                        "raw_dim": dim,
                        "encoded_dim": encoded_dim,
                        "n": None,
                    }
                )
            elif action_type == "Discrete":
                self.action_info.append(
                    {
                        "type": action_type,
                        "raw_dim": 1,
                        "encoded_dim": space.n,
                        "n": space.n,
                    }
                )
            else:
                raise NotImplementedError(
                    f"Action space type {action_type} is not supported for centralized Q training."
                )

            self.total_raw_action_dim += self.action_info[-1]["raw_dim"]
            self.total_encoded_action_dim += self.action_info[-1]["encoded_dim"]

        init_method = get_init_method(self.initialization_method)

        def init_(module):
            return init(module, init_method, lambda x: nn.init.constant_(x, 0))

        self.activation = get_active_func(self.activation_func)
        self.q_fc = init_(
            nn.Linear(
                self.hidden_sizes[-1] + self.total_encoded_action_dim,
                self.hidden_sizes[-1],
            )
        )
        self.q_out = init_(nn.Linear(self.hidden_sizes[-1], 1))

        self.to(device)

    def forward(self, cent_obs, actions):
        """Forward pass of the centralized Q network."""

        cent_obs = check(cent_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)

        if actions.shape[-1] != self.total_raw_action_dim:
            raise ValueError(
                "Joint action tensor has incompatible shape: "
                f"expected last dimension {self.total_raw_action_dim}, "
                f"got {actions.shape[-1]}"
            )

        features = self.base(cent_obs)
        action_features = self._process_actions(actions)

        concat = torch.cat([features, action_features], dim=-1)
        hidden = self.activation(self.q_fc(concat))
        q_values = self.q_out(hidden)

        return q_values

    def _process_actions(self, actions):
        """Convert raw actions into network-compatible representations."""

        processed: List[torch.Tensor] = []
        start = 0
        for info in self.action_info:
            end = start + info["raw_dim"]
            raw_action = actions[..., start:end]
            if info["type"] == "Box":
                processed.append(raw_action.float())
            else:  # Discrete
                if raw_action.shape[-1] != 1:
                    raise ValueError(
                        "Expected discrete actions to have a single dimension per agent."
                    )
                discrete_action = raw_action.squeeze(-1).long()
                processed.append(
                    F.one_hot(discrete_action, num_classes=info["n"]).float()
                )
            start = end

        return torch.cat(processed, dim=-1)
