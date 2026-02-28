"""Centralized Q Critic (per agent)."""
import numpy as np
import torch
import torch.nn as nn
import gym
from harl.utils.models_tools import (
    get_grad_norm,
    huber_loss,
    mse_loss,
    update_linear_schedule,
)
from harl.utils.envs_tools import check, get_shape_from_obs_space
from harl.models.value_function_models.v_net import VNet


class CentralizedQCritic:
    """Centralized Q Critic (per agent).

    A per-agent centralized critic that learns a Q-function conditioned on the
    global (shared) observation and all agents' concatenated actions.  Each
    agent maintains its own network with independent parameters.

    Internally the Q-network is implemented as a VNet whose input observation
    space is artificially extended to include the total action dimension, so
    the same MLP/RNN backbone is reused without additional model files.

    Training and loss computation follow the same PPO-style value-function
    procedure used by VCritic (clipped value loss, optional Huber loss,
    optional value normalisation).
    """

    def __init__(self, args, cent_obs_space, total_act_dim, device=torch.device("cpu")):
        """Initialize CentralizedQCritic.
        Args:
            args: (dict) algorithm and model arguments.
            cent_obs_space: (gym.Space) centralized observation space.
            total_act_dim: (int) sum of all agents' action dimensions.
            device: (torch.device) device to use for tensor operations.
        """
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.clip_param = args["clip_param"]
        self.critic_epoch = args["critic_epoch"]
        self.critic_num_mini_batch = args["critic_num_mini_batch"]
        self.data_chunk_length = args["data_chunk_length"]
        self.value_loss_coef = args["value_loss_coef"]
        self.max_grad_norm = args["max_grad_norm"]
        self.huber_delta = args["huber_delta"]

        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.use_naive_recurrent_policy = args["use_naive_recurrent_policy"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.use_clipped_value_loss = args["use_clipped_value_loss"]
        self.use_huber_loss = args["use_huber_loss"]
        self.use_policy_active_masks = args["use_policy_active_masks"]

        self.critic_lr = args["critic_lr"]
        self.opti_eps = args["opti_eps"]
        self.weight_decay = args["weight_decay"]

        # Build an extended obs space: [share_obs, all_agents_actions]
        obs_shape = get_shape_from_obs_space(cent_obs_space)
        extended_dim = obs_shape[0] + total_act_dim
        extended_obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(extended_dim,), dtype=np.float32
        )

        self.critic = VNet(args, extended_obs_space, device)

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    def lr_decay(self, episode, episodes):
        """Decay the critic learning rate.
        Args:
            episode: (int) current training episode.
            episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_values(self, cent_obs, all_actions, rnn_states_critic, masks):
        """Get Q-value predictions.
        Args:
            cent_obs: (np.ndarray) centralized (global) observations.
            all_actions: (np.ndarray) all agents' actions concatenated, shape (..., total_act_dim).
            rnn_states_critic: (np.ndarray) RNN hidden states for the critic.
            masks: (np.ndarray) RNN reset masks.
        Returns:
            q_values: (torch.Tensor) Q-value predictions.
            rnn_states_critic: (torch.Tensor) updated RNN hidden states.
        """
        cent_obs = check(cent_obs).to(**self.tpdv)
        all_actions = check(all_actions).to(**self.tpdv)
        combined = torch.cat([cent_obs, all_actions], dim=-1)
        q_values, rnn_states_critic = self.critic(combined, rnn_states_critic, masks)
        return q_values, rnn_states_critic

    def cal_value_loss(
        self, values, value_preds_batch, return_batch, value_normalizer=None
    ):
        """Calculate value function loss (identical to VCritic).
        Args:
            values: (torch.Tensor) Q-value predictions.
            value_preds_batch: (torch.Tensor) old value predictions from data batch.
            return_batch: (torch.Tensor) reward-to-go returns.
            value_normalizer: (ValueNorm) optional value normalizer.
        Returns:
            value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
            -self.clip_param, self.clip_param
        )
        if value_normalizer is not None:
            value_normalizer.update(return_batch)
            error_clipped = value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self.use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self.use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        return value_loss.mean()

    def update(self, sample, value_normalizer=None):
        """Update the Q critic network.
        Args:
            sample: (Tuple) data batch from OnPolicyCentralQBuffer generator.
                (share_obs, all_actions, rnn_states_critic, value_preds, returns, masks)
            value_normalizer: (ValueNorm) optional value normalizer.
        Returns:
            value_loss: (torch.Tensor) critic loss.
            critic_grad_norm: (torch.Tensor) gradient norm.
        """
        (
            share_obs_batch,
            all_actions_batch,
            rnn_states_critic_batch,
            value_preds_batch,
            return_batch,
            masks_batch,
        ) = sample

        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)

        q_values, _ = self.get_values(
            share_obs_batch, all_actions_batch, rnn_states_critic_batch, masks_batch
        )

        value_loss = self.cal_value_loss(
            q_values, value_preds_batch, return_batch, value_normalizer=value_normalizer
        )

        self.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()

        if self.use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.max_grad_norm
            )
        else:
            critic_grad_norm = get_grad_norm(self.critic.parameters())

        self.critic_optimizer.step()

        return value_loss, critic_grad_norm

    def train(self, critic_buffer, value_normalizer=None):
        """Perform a training update using minibatch GD.
        Args:
            critic_buffer: (OnPolicyCentralQBuffer) buffer containing training data.
            value_normalizer: (ValueNorm) optional value normalizer.
        Returns:
            train_info: (dict) training statistics (loss, grad norm).
        """
        train_info = {"value_loss": 0, "critic_grad_norm": 0}

        for _ in range(self.critic_epoch):
            if self.use_recurrent_policy:
                data_generator = critic_buffer.recurrent_generator_critic(
                    self.critic_num_mini_batch, self.data_chunk_length
                )
            elif self.use_naive_recurrent_policy:
                data_generator = critic_buffer.naive_recurrent_generator_critic(
                    self.critic_num_mini_batch
                )
            else:
                data_generator = critic_buffer.feed_forward_generator_critic(
                    self.critic_num_mini_batch
                )

            for sample in data_generator:
                value_loss, critic_grad_norm = self.update(
                    sample, value_normalizer=value_normalizer
                )
                train_info["value_loss"] += value_loss.item()
                train_info["critic_grad_norm"] += critic_grad_norm

        num_updates = self.critic_epoch * self.critic_num_mini_batch
        for k in train_info:
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        """Prepare for training."""
        self.critic.train()

    def prep_rollout(self):
        """Prepare for rollout."""
        self.critic.eval()
