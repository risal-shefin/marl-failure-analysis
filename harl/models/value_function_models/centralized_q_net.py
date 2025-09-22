"""Centralized Q network implementation."""

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

        self.action_type = act_space.__class__.__name__
        if self.action_type == "Box":
            self.action_dim = act_space.shape[0]
        elif self.action_type == "Discrete":
            self.action_dim = act_space.n
        else:
            raise NotImplementedError(
                f"Action space type {self.action_type} is not supported for centralized Q training."
            )

        init_method = get_init_method(self.initialization_method)

        def init_(module):
            return init(module, init_method, lambda x: nn.init.constant_(x, 0))

        self.activation = get_active_func(self.activation_func)
        self.q_fc = init_(
            nn.Linear(self.hidden_sizes[-1] + self.action_dim, self.hidden_sizes[-1])
        )
        self.q_out = init_(nn.Linear(self.hidden_sizes[-1], 1))

        self.to(device)

    def forward(self, cent_obs, actions):
        """Forward pass of the centralized Q network."""

        cent_obs = check(cent_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)

        features = self.base(cent_obs)
        action_features = self._process_actions(actions)

        concat = torch.cat([features, action_features], dim=-1)
        hidden = self.activation(self.q_fc(concat))
        q_values = self.q_out(hidden)

        return q_values

    def _process_actions(self, actions):
        """Convert raw actions into network-compatible representations."""

        if self.action_type == "Box":
            return actions.float()

        if actions.dim() == 2 and actions.shape[-1] == 1:
            actions = actions.squeeze(-1)
        actions = actions.long()
        return F.one_hot(actions, num_classes=self.action_dim).float()
