"""On-policy buffer for centralized Q critic."""
import torch
import numpy as np
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.utils.trans_tools import _flatten, _sa_cast


class OnPolicyCentralQBuffer(OnPolicyCriticBufferEP):
    """On-policy buffer for centralized Q critic.

    Extends OnPolicyCriticBufferEP by additionally storing all agents'
    concatenated actions so that the Q critic can condition on them.
    """

    def __init__(self, args, share_obs_space, total_act_dim):
        """Initialize on-policy central Q buffer.
        Args:
            args: (dict) arguments
            share_obs_space: (gym.Space or list) share observation space
            total_act_dim: (int) total action dimension across all agents
        """
        super().__init__(args, share_obs_space)

        # Buffer for all agents' concatenated actions at each timestep
        self.all_actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, total_act_dim),
            dtype=np.float32,
        )

    def insert(
        self,
        share_obs,
        all_actions,
        rnn_states_critic,
        value_preds,
        rewards,
        masks,
        bad_masks,
    ):
        """Insert data into buffer."""
        self.all_actions[self.step] = all_actions.copy()
        super().insert(share_obs, rnn_states_critic, value_preds, rewards, masks, bad_masks)

    def feed_forward_generator_critic(
        self, critic_num_mini_batch=None, mini_batch_size=None
    ):
        """Training data generator for critic that uses MLP network.
        Yields (share_obs, all_actions, rnn_states_critic, value_preds, returns, masks).
        """
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        if mini_batch_size is None:
            assert batch_size >= critic_num_mini_batch
            mini_batch_size = batch_size // critic_num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(critic_num_mini_batch)
        ]

        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        all_actions = self.all_actions.reshape(-1, self.all_actions.shape[-1])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(
            -1, *self.rnn_states_critic.shape[2:]
        )
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)

        for indices in sampler:
            yield (
                share_obs[indices],
                all_actions[indices],
                rnn_states_critic[indices],
                value_preds[indices],
                returns[indices],
                masks[indices],
            )

    def naive_recurrent_generator_critic(self, critic_num_mini_batch):
        """Training data generator for critic that uses RNN network (naive).
        Yields (share_obs, all_actions, rnn_states_critic, value_preds, returns, masks).
        """
        n_rollout_threads = self.rewards.shape[1]
        assert n_rollout_threads >= critic_num_mini_batch
        num_envs_per_batch = n_rollout_threads // critic_num_mini_batch

        perm = torch.randperm(n_rollout_threads).numpy()
        T, N = self.episode_length, num_envs_per_batch

        for batch_id in range(critic_num_mini_batch):
            start_id = batch_id * num_envs_per_batch
            ids = perm[start_id : start_id + num_envs_per_batch]
            share_obs_batch = _flatten(T, N, self.share_obs[:-1, ids])
            all_actions_batch = _flatten(T, N, self.all_actions[:, ids])
            value_preds_batch = _flatten(T, N, self.value_preds[:-1, ids])
            return_batch = _flatten(T, N, self.returns[:-1, ids])
            masks_batch = _flatten(T, N, self.masks[:-1, ids])
            rnn_states_critic_batch = self.rnn_states_critic[0, ids]

            yield (
                share_obs_batch,
                all_actions_batch,
                rnn_states_critic_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
            )

    def recurrent_generator_critic(self, critic_num_mini_batch, data_chunk_length):
        """Training data generator for critic that uses RNN network (chunked).
        Yields (share_obs, all_actions, rnn_states_critic, value_preds, returns, masks).
        """
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length
        mini_batch_size = data_chunks // critic_num_mini_batch

        assert episode_length % data_chunk_length == 0
        assert data_chunks >= 2

        rand = torch.randperm(data_chunks).numpy()
        sampler = [
            rand[i * mini_batch_size : (i + 1) * mini_batch_size]
            for i in range(critic_num_mini_batch)
        ]

        if len(self.share_obs.shape) > 3:
            share_obs = (
                self.share_obs[:-1]
                .transpose(1, 0, 2, 3, 4)
                .reshape(-1, *self.share_obs.shape[2:])
            )
        else:
            share_obs = _sa_cast(self.share_obs[:-1])
        all_actions = _sa_cast(self.all_actions)
        value_preds = _sa_cast(self.value_preds[:-1])
        returns = _sa_cast(self.returns[:-1])
        masks = _sa_cast(self.masks[:-1])
        rnn_states_critic = (
            self.rnn_states_critic[:-1]
            .transpose(1, 0, 2, 3)
            .reshape(-1, *self.rnn_states_critic.shape[2:])
        )

        for indices in sampler:
            share_obs_batch = []
            all_actions_batch = []
            rnn_states_critic_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []

            for index in indices:
                ind = index * data_chunk_length
                share_obs_batch.append(share_obs[ind : ind + data_chunk_length])
                all_actions_batch.append(all_actions[ind : ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind : ind + data_chunk_length])
                return_batch.append(returns[ind : ind + data_chunk_length])
                masks_batch.append(masks[ind : ind + data_chunk_length])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

            L, N = data_chunk_length, mini_batch_size
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            all_actions_batch = np.stack(all_actions_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(
                N, *self.rnn_states_critic.shape[2:]
            )

            share_obs_batch = _flatten(L, N, share_obs_batch)
            all_actions_batch = _flatten(L, N, all_actions_batch)
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)

            yield (
                share_obs_batch,
                all_actions_batch,
                rnn_states_critic_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
            )
