"""Runner for on-policy MA algorithms."""
import numpy as np
import torch
from harl.runners.on_policy_base_runner import OnPolicyBaseRunner


class OnPolicyMARunner(OnPolicyBaseRunner):
    """Runner for on-policy MA algorithms."""

    def train(self):
        """Training procedure for MAPPO."""
        actor_train_infos = []

        # compute advantages
        if not self.enable_heterogeneous_agents:
            if self.value_normalizer is not None:
                advantages = self.critic_buffer.returns[
                    :-1
                ] - self.value_normalizer.denormalize(self.critic_buffer.value_preds[:-1])
            else:
                advantages = (
                    self.critic_buffer.returns[:-1] - self.critic_buffer.value_preds[:-1]
                )
        else:
            advantages = np.zeros(
                (
                    self.algo_args["train"]["episode_length"],
                    self.algo_args["train"]["n_rollout_threads"],
                    self.num_agents,
                    1,
                ),
                dtype=np.float32,
            )
            for agent_type in self.type_order:
                critic_buffer = self.critic_buffers_by_type[agent_type]
                value_normalizer = self.value_normalizers_by_type[agent_type]
                if value_normalizer is not None:
                    type_adv = critic_buffer.returns[:-1] - value_normalizer.denormalize(
                        critic_buffer.value_preds[:-1]
                    )
                else:
                    type_adv = critic_buffer.returns[:-1] - critic_buffer.value_preds[:-1]
                type_agent_ids = self.type_to_agent_ids[agent_type]
                if self.state_type == "EP":
                    for aid in type_agent_ids:
                        advantages[:, :, aid, :] = type_adv
                elif self.state_type == "FP":
                    advantages[:, :, type_agent_ids, :] = type_adv

        # normalize advantages for FP
        if self.state_type == "FP":
            active_masks_collector = [
                self.actor_buffer[i].active_masks for i in range(self.num_agents)
            ]
            active_masks_array = np.stack(active_masks_collector, axis=2)
            advantages_copy = advantages.copy()
            advantages_copy[active_masks_array[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        # update actors
        if self.share_param:
            actor_train_info = self.actor[0].share_param_train(
                self.actor_buffer, advantages.copy(), self.num_agents, self.state_type
            )
            for _ in torch.randperm(self.num_agents):
                actor_train_infos.append(actor_train_info)
        else:
            for agent_id in range(self.num_agents):
                if self.state_type == "EP":
                    actor_train_info = self.actor[agent_id].train(
                        self.actor_buffer[agent_id], advantages.copy(), "EP"
                    )
                elif self.state_type == "FP":
                    actor_train_info = self.actor[agent_id].train(
                        self.actor_buffer[agent_id],
                        advantages[:, :, agent_id].copy(),
                        "FP",
                    )
                actor_train_infos.append(actor_train_info)

        # update critic
        if not self.enable_heterogeneous_agents:
            critic_train_info = self.critic.train(self.critic_buffer, self.value_normalizer)
        else:
            critic_train_info = {}
            for agent_type in self.type_order:
                type_info = self.critics_by_type[agent_type].train(
                    self.critic_buffers_by_type[agent_type],
                    self.value_normalizers_by_type[agent_type],
                )
                for k, v in type_info.items():
                    critic_train_info[f"type_{agent_type}/{k}"] = v

        if self.enable_central_q:
            for agent_id in range(self.num_agents):
                cent_train_info = self.centralized_critics[agent_id].train(
                    self.centralized_critic_buffers[agent_id],
                    self.centralized_value_normalizers[agent_id],
                )
                for k, v in cent_train_info.items():
                    critic_train_info[f"centralized_agent{agent_id}/{k}"] = v

        return actor_train_infos, critic_train_info
