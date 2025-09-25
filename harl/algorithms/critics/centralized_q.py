"""Centralized Q function trainer."""

import numpy as np
import torch
import torch.nn as nn

from harl.models.value_function_models.centralized_q_net import CentralizedQNet
from harl.utils.envs_tools import check
from harl.utils.models_tools import (
    get_grad_norm,
    huber_loss,
    mse_loss,
    update_linear_schedule,
)


class CentralizedQFunction:
    """Train a centralized Q function conditioned on state and action."""

    def __init__(self, args, cent_obs_space, act_space, device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.critic_epoch = args["critic_epoch"]
        self.critic_num_mini_batch = args["critic_num_mini_batch"]
        self.value_loss_coef = args["value_loss_coef"]
        self.max_grad_norm = args["max_grad_norm"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.use_huber_loss = args["use_huber_loss"]
        self.huber_delta = args["huber_delta"]

        self.lr = args["critic_lr"]
        self.opti_eps = args["opti_eps"]
        self.weight_decay = args["weight_decay"]
        self.q_net = CentralizedQNet(args, cent_obs_space, act_space, device)

        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(),
            lr=self.lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def lr_decay(self, episode, episodes):
        """Linearly decay the learning rate."""

        update_linear_schedule(self.optimizer, episode, episodes, self.lr)

    def prep_training(self):
        """Set the network into training mode."""

        self.q_net.train()

    def prep_rollout(self):
        """Set the network into evaluation mode."""

        self.q_net.eval()

    def get_q_values(self, cent_obs, actions, gradNeed=False):
        """Evaluate Q values."""
        if gradNeed:
            return self.q_net(cent_obs, actions)
        with torch.no_grad():
            return self.q_net(cent_obs, actions)

    def _mini_batch_generator(self, share_obs, actions, returns, active_masks):
        batch_size = share_obs.shape[0]
        if batch_size == 0:
            return []
        num_mini_batch = min(self.critic_num_mini_batch, batch_size)
        mini_batch_size = max(batch_size // num_mini_batch, 1)
        indices = np.random.permutation(batch_size)

        batches = []
        for start in range(0, batch_size, mini_batch_size):
            end = min(start + mini_batch_size, batch_size)
            batch_indices = indices[start:end]
            batches.append(
                (
                    share_obs[batch_indices],
                    actions[batch_indices],
                    returns[batch_indices],
                    None if active_masks is None else active_masks[batch_indices],
                )
            )
        return batches

    def _update(self, sample, value_normalizer=None):
        share_obs_batch, actions_batch, return_batch, active_masks_batch = sample

        share_obs_batch = check(share_obs_batch).to(**self.tpdv)
        actions_batch = check(actions_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)

        if active_masks_batch is not None:
            active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        q_pred = self.q_net(share_obs_batch, actions_batch)

        if value_normalizer is not None:
            target = value_normalizer.normalize(return_batch)
        else:
            target = return_batch

        if self.use_huber_loss:
            q_loss = huber_loss(target - q_pred, self.huber_delta)
        else:
            q_loss = mse_loss(target - q_pred)

        if active_masks_batch is not None:
            mask_sum = active_masks_batch.sum()
            if mask_sum.item() > 0:
                q_loss = (q_loss * active_masks_batch).sum() / mask_sum
            else:
                q_loss = q_loss.mean()
        else:
            q_loss = q_loss.mean()

        self.optimizer.zero_grad()
        (q_loss * self.value_loss_coef).backward()

        if self.use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(
                self.q_net.parameters(), self.max_grad_norm
            )
        else:
            grad_norm = get_grad_norm(self.q_net.parameters())

        self.optimizer.step()

        return q_loss.detach().cpu().item(), float(grad_norm)

    def train(
        self,
        share_obs,
        actions,
        returns,
        active_masks=None,
        value_normalizer=None,
    ):
        """Train the centralized Q function using rollout data."""

        train_info = {
            "central_q_loss": 0.0,
            "central_q_grad_norm": 0.0,
        }

        num_updates = 0
        for _ in range(self.critic_epoch):
            batches = self._mini_batch_generator(
                share_obs, actions, returns, active_masks
            )
            if not batches:
                break
            for batch in batches:
                loss, grad_norm = self._update(batch, value_normalizer=value_normalizer)
                train_info["central_q_loss"] += loss
                train_info["central_q_grad_norm"] += grad_norm
                num_updates += 1

        if num_updates > 0:
            train_info["central_q_loss"] /= num_updates
            train_info["central_q_grad_norm"] /= num_updates

        return train_info

    def save(self, save_dir, agent_id, suffix=""):
        """Save the Q network parameters."""

        torch.save(
            self.q_net.state_dict(),
            str(save_dir) + f"/central_q_agent{agent_id}{suffix}.pt",
        )

    def restore(self, model_dir, agent_id, suffix=""):
        """Restore the Q network parameters if present."""
        # print(f"Model dir: {model_dir}")
        state_dict = torch.load(
            str(model_dir) + f"/central_q_agent{agent_id}_{suffix}.pt"
        )
        # print(f"State dict:{state_dict}")
        first_key = list(state_dict.keys())[0]
        # print(f"First key: {first_key}, shape: {state_dict[first_key].shape}")
        self.q_net.load_state_dict(state_dict)
