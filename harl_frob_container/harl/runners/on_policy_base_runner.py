"""Base runner for on-policy algorithms."""

import time
import numpy as np
import torch
import setproctitle
from harl.common.valuenorm import ValueNorm
from harl.common.buffers.on_policy_actor_buffer import OnPolicyActorBuffer
from harl.common.buffers.on_policy_critic_buffer_ep import OnPolicyCriticBufferEP
from harl.common.buffers.on_policy_critic_buffer_fp import OnPolicyCriticBufferFP
from harl.algorithms.actors import ALGO_REGISTRY
from harl.algorithms.critics.v_critic import VCritic
from harl.algorithms.critics.centralized_q_critic import CentralizedQCritic
from harl.common.buffers.on_policy_central_q_buffer import OnPolicyCentralQBuffer
from harl.utils.trans_tools import _t2n
from harl.utils.envs_tools import (
    make_eval_env,
    make_train_env,
    make_render_env,
    set_seed,
    get_num_agents,
)
from harl.utils.models_tools import init_device, find_checkpoint
from harl.utils.configs_tools import init_dir, save_config
from harl.envs import LOGGER_REGISTRY

class _MeanRewardProxy:
    """Adapter exposing get_mean_rewards() for logger.episode_log()."""

    def __init__(self, mean_reward):
        self._mean_reward = mean_reward

    def get_mean_rewards(self):
        return self._mean_reward


class OnPolicyBaseRunner:
    """Base runner for on-policy algorithms."""

    def __init__(self, args, algo_args, env_args):
        """Initialize the OnPolicyBaseRunner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args

        self.hidden_sizes = algo_args["model"]["hidden_sizes"]
        self.rnn_hidden_size = self.hidden_sizes[-1]
        self.recurrent_n = algo_args["model"]["recurrent_n"]
        self.action_aggregation = algo_args["algo"]["action_aggregation"]
        self.state_type = env_args.get("state_type", "EP")
        self.share_param = algo_args["algo"]["share_param"]
        self.fixed_order = algo_args["algo"]["fixed_order"]
        set_seed(algo_args["seed"])
        self.device = init_device(algo_args["device"])
        if not self.algo_args["render"]["use_render"]:  # train, not render
            self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
                args["env"],
                env_args,
                args["algo"],
                args["exp_name"],
                algo_args["seed"]["seed"],
                logger_path=algo_args["logger"]["log_dir"],
            )
            save_config(args, algo_args, env_args, self.run_dir)
        # set the title of the process
        setproctitle.setproctitle(
            str(args["algo"]) + "-" + str(args["env"]) + "-" + str(args["exp_name"])
        )

        # set the config of env
        if self.algo_args["render"]["use_render"]:  # make envs for rendering
            (
                self.envs,
                self.manual_render,
                self.manual_expand_dims,
                self.manual_delay,
                self.env_num,
            ) = make_render_env(args["env"], algo_args["seed"]["seed"], env_args)
        else:  # make envs for training and evaluation
            self.envs = make_train_env(
                args["env"],
                algo_args["seed"]["seed"],
                algo_args["train"]["n_rollout_threads"],
                env_args,
            )
            self.eval_envs = (
                make_eval_env(
                    args["env"],
                    algo_args["seed"]["seed"],
                    algo_args["eval"]["n_eval_rollout_threads"],
                    env_args,
                )
                if algo_args["eval"]["use_eval"]
                else None
            )
        self.num_agents = get_num_agents(args["env"], env_args, self.envs)

        self.enable_heterogeneous_agents = (
            args["env"] == "pettingzoo_mpe"
            and env_args.get("enable_heterogeneous_agents", False)
        )
        self.eval_reward_mode = self.algo_args["eval"].get("eval_reward_mode", "team")
        self.agent_types = getattr(self.envs, "agent_types", None)
        self.type_to_agent_ids = getattr(self.envs, "type_to_agent_ids", None)
        self.agent_id_to_type = getattr(self.envs, "agent_id_to_type", None)

        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space)

        if self.enable_heterogeneous_agents:
            if (
                self.agent_types is None
                or self.type_to_agent_ids is None
                or self.agent_id_to_type is None
            ):
                raise ValueError(
                    "Heterogeneous mode requires environment metadata: "
                    "agent_types/type_to_agent_ids/agent_id_to_type"
                )
            if self.share_param:
                raise ValueError(
                    "share_param=True is not supported with enable_heterogeneous_agents=True; "
                    "set share_param=False to avoid cross-type parameter sharing."
                )
            self.type_order = list(self.type_to_agent_ids.keys())
            if self.eval_reward_mode not in ["team", "competitive"]:
                raise ValueError(
                    f"Unknown eval_reward_mode={self.eval_reward_mode}; expected 'team' or 'competitive'."
                )
            self.type_to_critic_index = {t: i for i, t in enumerate(self.type_order)}
            if (
                self.state_type == "EP"
                and env_args.get("reward_mode", "global_sum_shared") == "individual"
                and any(len(agent_ids) > 1 for agent_ids in self.type_to_agent_ids.values())
            ):
                raise ValueError(
                    "state_type='EP' with reward_mode='individual' and multiple agents per type "
                    "is unsupported in heterogeneous mode; use state_type='FP' or a shared/team reward mode."
                )
        else:
            self.type_order = None
            self.type_to_critic_index = None

        # actor
        if self.share_param:
            self.actor = []
            agent = ALGO_REGISTRY[args["algo"]](
                {**algo_args["model"], **algo_args["algo"]},
                self.envs.observation_space[0],
                self.envs.action_space[0],
                device=self.device,
            )
            self.actor.append(agent)
            for agent_id in range(1, self.num_agents):
                assert (
                    self.envs.observation_space[agent_id]
                    == self.envs.observation_space[0]
                ), "Agents have heterogeneous observation spaces, parameter sharing is not valid."
                assert (
                    self.envs.action_space[agent_id] == self.envs.action_space[0]
                ), "Agents have heterogeneous action spaces, parameter sharing is not valid."
                self.actor.append(self.actor[0])
        else:
            self.actor = []
            for agent_id in range(self.num_agents):
                agent = ALGO_REGISTRY[args["algo"]](
                    {**algo_args["model"], **algo_args["algo"]},
                    self.envs.observation_space[agent_id],
                    self.envs.action_space[agent_id],
                    device=self.device,
                )
                self.actor.append(agent)

        self.enable_central_q = algo_args["algo"].get("enable_central_q", False)
        self.eval_start_episode = self.algo_args["eval"].get("eval_start_episode", 1)

        if self.algo_args["render"]["use_render"] is False:  # train, not render
            self.actor_buffer = []
            for agent_id in range(self.num_agents):
                ac_bu = OnPolicyActorBuffer(
                    {**algo_args["train"], **algo_args["model"]},
                    self.envs.observation_space[agent_id],
                    self.envs.action_space[agent_id],
                )
                self.actor_buffer.append(ac_bu)

            share_observation_space = self.envs.share_observation_space[0]
            if not self.enable_heterogeneous_agents:
                self.critic = VCritic(
                    {**algo_args["model"], **algo_args["algo"]},
                    share_observation_space,
                    device=self.device,
                )
                if self.state_type == "EP":
                    # EP stands for Environment Provided, as phrased by MAPPO paper.
                    # In EP, the global states for all agents are the same.
                    self.critic_buffer = OnPolicyCriticBufferEP(
                        {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                        share_observation_space,
                    )
                elif self.state_type == "FP":
                    # FP stands for Feature Pruned, as phrased by MAPPO paper.
                    # In FP, the global states for all agents are different, and thus needs the dimension of the number of agents.
                    self.critic_buffer = OnPolicyCriticBufferFP(
                        {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                        share_observation_space,
                        self.num_agents,
                    )
                else:
                    raise NotImplementedError

                if self.algo_args["train"]["use_valuenorm"] is True:
                    self.value_normalizer = ValueNorm(1, device=self.device)
                else:
                    self.value_normalizer = None
            else:
                self.critics_by_type = {}
                self.critic_buffers_by_type = {}
                self.value_normalizers_by_type = {}
                for agent_type in self.type_order:
                    n_type_agents = len(self.type_to_agent_ids[agent_type])
                    self.critics_by_type[agent_type] = VCritic(
                        {**algo_args["model"], **algo_args["algo"]},
                        share_observation_space,
                        device=self.device,
                    )
                    if self.state_type == "EP":
                        self.critic_buffers_by_type[agent_type] = OnPolicyCriticBufferEP(
                            {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                            share_observation_space,
                        )
                    elif self.state_type == "FP":
                        self.critic_buffers_by_type[agent_type] = OnPolicyCriticBufferFP(
                            {**algo_args["train"], **algo_args["model"], **algo_args["algo"]},
                            share_observation_space,
                            n_type_agents,
                        )
                    else:
                        raise NotImplementedError

                    if self.algo_args["train"]["use_valuenorm"] is True:
                        self.value_normalizers_by_type[agent_type] = ValueNorm(1, device=self.device)
                    else:
                        self.value_normalizers_by_type[agent_type] = None
                # legacy attributes retained for logger compatibility
                self.critic = self.critics_by_type[self.type_order[0]]
                self.critic_buffer = self.critic_buffers_by_type[self.type_order[0]]
                self.value_normalizer = self.value_normalizers_by_type[self.type_order[0]]

            # per-agent centralized Q critics (optional)
            if self.enable_central_q:
                from harl.utils.envs_tools import get_shape_from_act_space

                total_act_dim = sum(
                    get_shape_from_act_space(self.envs.action_space[i])
                    for i in range(self.num_agents)
                )
                self.centralized_critics = []
                self.centralized_critic_buffers = []
                for agent_id in range(self.num_agents):
                    cent_critic = CentralizedQCritic(
                        {**algo_args["model"], **algo_args["algo"]},
                        share_observation_space,
                        total_act_dim,
                        device=self.device,
                    )
                    self.centralized_critics.append(cent_critic)
                    cent_buffer = OnPolicyCentralQBuffer(
                        {
                            **algo_args["train"],
                            **algo_args["model"],
                            **algo_args["algo"],
                        },
                        share_observation_space,
                        total_act_dim,
                    )
                    self.centralized_critic_buffers.append(cent_buffer)
                if algo_args["train"]["use_valuenorm"] is True:
                    self.centralized_value_normalizers = [
                        ValueNorm(1, device=self.device)
                        for _ in range(self.num_agents)
                    ]
                else:
                    self.centralized_value_normalizers = [None] * self.num_agents

            self.logger = LOGGER_REGISTRY[args["env"]](
                args, algo_args, env_args, self.num_agents, self.writter, self.run_dir
            )
            self.best_eval_reward = -np.inf
            self.best_eval_reward_by_type = (
                {agent_type: -np.inf for agent_type in self.type_order}
                if self.enable_heterogeneous_agents
                else None
            )
        if self.algo_args["train"]["model_dir"] is not None:  # restore model
            self.restore()

    def _compute_type_eval_metrics(self, eval_rewards_array):
        """Compute per-type eval metrics according to eval_reward_mode."""
        team_metrics = {
            agent_type: float(np.mean(eval_rewards_array[:, agent_ids, :]))
            for agent_type, agent_ids in self.type_to_agent_ids.items()
        }
        if self.eval_reward_mode == "team":
            return team_metrics

        competitive_metrics = {}
        for agent_type in self.type_order:
            other_types = [t for t in self.type_order if t != agent_type]
            if len(other_types) == 0:
                competitive_metrics[agent_type] = team_metrics[agent_type]
            else:
                other_mean = float(np.mean([team_metrics[t] for t in other_types]))
                competitive_metrics[agent_type] = team_metrics[agent_type] - other_mean
        return competitive_metrics

    def run(self):
        """Run the training (or rendering) pipeline."""
        if self.algo_args["render"]["use_render"] is True:
            self.render()
            return
        print("start running")
        self.warmup()

        episodes = (
            int(self.algo_args["train"]["num_env_steps"])
            // self.algo_args["train"]["episode_length"]
            // self.algo_args["train"]["n_rollout_threads"]
        )

        self.logger.init(episodes)  # logger callback at the beginning of training

        for episode in range(1, episodes + 1):
            if self.algo_args["train"][
                "use_linear_lr_decay"
            ]:  # linear decay of learning rate
                if self.share_param:
                    self.actor[0].lr_decay(episode, episodes)
                else:
                    for agent_id in range(self.num_agents):
                        self.actor[agent_id].lr_decay(episode, episodes)
                if self.enable_heterogeneous_agents:
                    for agent_type in self.type_order:
                        self.critics_by_type[agent_type].lr_decay(episode, episodes)
                else:
                    self.critic.lr_decay(episode, episodes)
                if self.enable_central_q:
                    for agent_id in range(self.num_agents):
                        self.centralized_critics[agent_id].lr_decay(episode, episodes)

            self.logger.episode_init(
                episode
            )  # logger callback at the beginning of each episode

            self.prep_rollout()  # change to eval mode
            for step in range(self.algo_args["train"]["episode_length"]):
                # Sample actions from actors and values from critics
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                ) = self.collect(step)
                # actions: (n_threads, n_agents, action_dim)
                (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                ) = self.envs.step(actions)
                # obs: (n_threads, n_agents, obs_dim)
                # share_obs: (n_threads, n_agents, share_obs_dim)
                # rewards: (n_threads, n_agents, 1)
                # dones: (n_threads, n_agents)
                # infos: (n_threads)
                # available_actions: (n_threads, ) of None or (n_threads, n_agents, action_number)
                data = (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                self.logger.per_step(data)  # logger callback at each step

                self.insert(data)  # insert data into buffer

            # compute return and update network
            self.compute()
            self.prep_training()  # change to train mode

            actor_train_infos, critic_train_info = self.train()

            # log information
            if episode % self.algo_args["train"]["log_interval"] == 0:
                if not self.enable_heterogeneous_agents:
                    logging_critic_buffer = self.critic_buffer
                else:
                    weighted_mean_rewards = []
                    weights = []
                    for agent_type in self.type_order:
                        weighted_mean_rewards.append(
                            self.critic_buffers_by_type[agent_type].get_mean_rewards()
                        )
                        weights.append(len(self.type_to_agent_ids[agent_type]))

                    logging_critic_buffer = _MeanRewardProxy(
                        float(np.average(weighted_mean_rewards, weights=weights))
                    )

                self.logger.episode_log(
                    actor_train_infos,
                    critic_train_info,
                    self.actor_buffer,
                    logging_critic_buffer,
                )

            # eval
            if episode >= self.eval_start_episode and episode % self.algo_args["train"]["eval_interval"] == 0:
                if self.algo_args["eval"]["use_eval"]:
                    self.prep_rollout()
                    eval_reward = self.eval()
                    if self.enable_heterogeneous_agents:
                        for agent_type, type_eval_reward in eval_reward["by_type"].items():
                            if type_eval_reward >= self.best_eval_reward_by_type[agent_type]:
                                self.best_eval_reward_by_type[agent_type] = type_eval_reward
                                self.save_by_agent_type(agent_type, type_eval_reward)
                    else:
                        if eval_reward >= self.best_eval_reward:
                            self.best_eval_reward = eval_reward
                            self.save(eval_reward)
                else:
                    self.save()

            self.after_update()

    def warmup(self):
        """Warm up the replay buffer."""
        # reset env
        obs, share_obs, available_actions = self.envs.reset()
        # replay buffer
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].obs[0] = obs[:, agent_id].copy()
            if self.actor_buffer[agent_id].available_actions is not None:
                self.actor_buffer[agent_id].available_actions[0] = available_actions[
                    :, agent_id
                ].copy()
        if not self.enable_heterogeneous_agents:
            if self.state_type == "EP":
                self.critic_buffer.share_obs[0] = share_obs[:, 0].copy()
            elif self.state_type == "FP":
                self.critic_buffer.share_obs[0] = share_obs.copy()
        else:
            if self.state_type == "EP":
                for agent_type in self.type_order:
                    self.critic_buffers_by_type[agent_type].share_obs[0] = share_obs[:, 0].copy()
            elif self.state_type == "FP":
                for agent_type in self.type_order:
                    type_agent_ids = self.type_to_agent_ids[agent_type]
                    self.critic_buffers_by_type[agent_type].share_obs[0] = share_obs[:, type_agent_ids].copy()
        if self.enable_central_q:
            for agent_id in range(self.num_agents):
                if self.state_type == "EP":
                    self.centralized_critic_buffers[agent_id].share_obs[0] = (
                        share_obs[:, 0].copy()
                    )
                elif self.state_type == "FP":
                    self.centralized_critic_buffers[agent_id].share_obs[0] = (
                        share_obs[:, agent_id].copy()
                    )

    @torch.no_grad()
    def collect(self, step):
        """Collect actions and values from actors and critics.
        Args:
            step: step in the episode.
        Returns:
            values, actions, action_log_probs, rnn_states, rnn_states_critic
        """
        # collect actions, action_log_probs, rnn_states from n actors
        action_collector = []
        action_log_prob_collector = []
        rnn_state_collector = []
        for agent_id in range(self.num_agents):
            action, action_log_prob, rnn_state = self.actor[agent_id].get_actions(
                self.actor_buffer[agent_id].obs[step],
                self.actor_buffer[agent_id].rnn_states[step],
                self.actor_buffer[agent_id].masks[step],
                self.actor_buffer[agent_id].available_actions[step]
                if self.actor_buffer[agent_id].available_actions is not None
                else None,
            )
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
        # (n_agents, n_threads, dim) -> (n_threads, n_agents, dim)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)

        # collect values, rnn_states_critic from critic(s)
        if not self.enable_heterogeneous_agents:
            if self.state_type == "EP":
                value, rnn_state_critic = self.critic.get_values(
                    self.critic_buffer.share_obs[step],
                    self.critic_buffer.rnn_states_critic[step],
                    self.critic_buffer.masks[step],
                )
                # (n_threads, dim)
                values = _t2n(value)
                rnn_states_critic = _t2n(rnn_state_critic)
            elif self.state_type == "FP":
                value, rnn_state_critic = self.critic.get_values(
                    np.concatenate(self.critic_buffer.share_obs[step]),
                    np.concatenate(self.critic_buffer.rnn_states_critic[step]),
                    np.concatenate(self.critic_buffer.masks[step]),
                )  # concatenate (n_threads, n_agents, dim) into (n_threads * n_agents, dim)
                # split (n_threads * n_agents, dim) into (n_threads, n_agents, dim)
                values = np.array(
                    np.split(_t2n(value), self.algo_args["train"]["n_rollout_threads"])
                )
                rnn_states_critic = np.array(
                    np.split(
                        _t2n(rnn_state_critic), self.algo_args["train"]["n_rollout_threads"]
                    )
                )
        else:
            values = np.zeros(
                (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1),
                dtype=np.float32,
            )
            rnn_states_critic = np.zeros(
                (
                    self.algo_args["train"]["n_rollout_threads"],
                    self.num_agents,
                    self.recurrent_n,
                    self.rnn_hidden_size,
                ),
                dtype=np.float32,
            )
            for agent_type in self.type_order:
                type_agent_ids = self.type_to_agent_ids[agent_type]
                critic_buffer = self.critic_buffers_by_type[agent_type]
                critic = self.critics_by_type[agent_type]
                if self.state_type == "EP":
                    value, rnn_state_critic = critic.get_values(
                        critic_buffer.share_obs[step],
                        critic_buffer.rnn_states_critic[step],
                        critic_buffer.masks[step],
                    )
                    value_np = _t2n(value)
                    rnn_np = _t2n(rnn_state_critic)
                    values[:, type_agent_ids, :] = value_np[:, None, :]
                    rnn_states_critic[:, type_agent_ids, :, :] = rnn_np[:, None, :, :]
                elif self.state_type == "FP":
                    value, rnn_state_critic = critic.get_values(
                        np.concatenate(critic_buffer.share_obs[step]),
                        np.concatenate(critic_buffer.rnn_states_critic[step]),
                        np.concatenate(critic_buffer.masks[step]),
                    )
                    value_np = np.array(
                        np.split(_t2n(value), self.algo_args["train"]["n_rollout_threads"])
                    )
                    rnn_np = np.array(
                        np.split(
                            _t2n(rnn_state_critic),
                            self.algo_args["train"]["n_rollout_threads"],
                        )
                    )
                    values[:, type_agent_ids, :] = value_np
                    rnn_states_critic[:, type_agent_ids, :, :] = rnn_np

        if self.enable_central_q:
            self._collect_central_q_values(step, actions)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def _collect_central_q_values(self, step, actions):
        """Collect Q-values from per-agent centralized Q critics.
        Args:
            step: (int) step in the episode.
            actions: (np.ndarray) all agents' actions, shape (n_threads, n_agents, act_dim).
        Stores results in self._centralized_q_values and self._centralized_rnn_states.
        """
        # Concatenate all agents' actions: (n_threads, n_agents * act_dim)
        all_actions = actions.reshape(actions.shape[0], -1)
        centralized_q_values = []
        centralized_rnn_states = []
        for agent_id in range(self.num_agents):
            cent_q, cent_rnn_state = self.centralized_critics[agent_id].get_values(
                self.centralized_critic_buffers[agent_id].share_obs[step],
                all_actions,
                self.centralized_critic_buffers[agent_id].rnn_states_critic[step],
                self.centralized_critic_buffers[agent_id].masks[step],
            )
            centralized_q_values.append(_t2n(cent_q))
            centralized_rnn_states.append(_t2n(cent_rnn_state))
        self._centralized_q_values = centralized_q_values
        self._centralized_rnn_states = centralized_rnn_states

    def insert(self, data):
        """Insert data into buffer."""
        (
            obs,  # (n_threads, n_agents, obs_dim)
            share_obs,  # (n_threads, n_agents, share_obs_dim)
            rewards,  # (n_threads, n_agents, 1)
            dones,  # (n_threads, n_agents)
            infos,  # type: list, shape: (n_threads, n_agents)
            available_actions,  # (n_threads, ) of None or (n_threads, n_agents, action_number)
            values,  # EP: (n_threads, dim), FP: (n_threads, n_agents, dim)
            actions,  # (n_threads, n_agents, action_dim)
            action_log_probs,  # (n_threads, n_agents, action_dim)
            rnn_states,  # (n_threads, n_agents, dim)
            rnn_states_critic,  # EP: (n_threads, dim), FP: (n_threads, n_agents, dim)
        ) = data

        dones_env = np.all(dones, axis=1)  # if all agents are done, then env is done
        rnn_states[
            dones_env == True
        ] = np.zeros(  # if env is done, then reset rnn_state to all zero
            (
                (dones_env == True).sum(),
                self.num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

        # If env is done, then reset rnn_state_critic to all zero
        if not self.enable_heterogeneous_agents:
            if self.state_type == "EP":
                rnn_states_critic[dones_env == True] = np.zeros(
                    ((dones_env == True).sum(), self.recurrent_n, self.rnn_hidden_size),
                    dtype=np.float32,
                )
            elif self.state_type == "FP":
                rnn_states_critic[dones_env == True] = np.zeros(
                    (
                        (dones_env == True).sum(),
                        self.num_agents,
                        self.recurrent_n,
                        self.rnn_hidden_size,
                    ),
                    dtype=np.float32,
                )
        else:
            rnn_states_critic[dones_env == True] = np.zeros(
                (
                    (dones_env == True).sum(),
                    self.num_agents,
                    self.recurrent_n,
                    self.rnn_hidden_size,
                ),
                dtype=np.float32,
            )

        # masks use 0 to mask out threads that just finish.
        # this is used for denoting at which point should rnn state be reset
        masks = np.ones(
            (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1),
            dtype=np.float32,
        )
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        # active_masks use 0 to mask out agents that have died
        active_masks = np.ones(
            (self.algo_args["train"]["n_rollout_threads"], self.num_agents, 1),
            dtype=np.float32,
        )
        active_masks[dones == True] = np.zeros(
            ((dones == True).sum(), 1), dtype=np.float32
        )
        active_masks[dones_env == True] = np.ones(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        # bad_masks use 0 to denote truncation and 1 to denote termination
        if self.state_type == "EP":
            bad_masks = np.array(
                [
                    [0.0]
                    if "bad_transition" in info[0].keys()
                    and info[0]["bad_transition"] == True
                    else [1.0]
                    for info in infos
                ]
            )
        elif self.state_type == "FP":
            bad_masks = np.array(
                [
                    [
                        [0.0]
                        if "bad_transition" in info[agent_id].keys()
                        and info[agent_id]["bad_transition"] == True
                        else [1.0]
                        for agent_id in range(self.num_agents)
                    ]
                    for info in infos
                ]
            )

        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].insert(
                obs[:, agent_id],
                rnn_states[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                masks[:, agent_id],
                active_masks[:, agent_id],
                available_actions[:, agent_id]
                if available_actions[0] is not None
                else None,
            )

        if not self.enable_heterogeneous_agents:
            if self.state_type == "EP":
                self.critic_buffer.insert(
                    share_obs[:, 0],
                    rnn_states_critic,
                    values,
                    rewards[:, 0],
                    masks[:, 0],
                    bad_masks,
                )
            elif self.state_type == "FP":
                self.critic_buffer.insert(
                    share_obs, rnn_states_critic, values, rewards, masks, bad_masks
                )
        else:
            for agent_type in self.type_order:
                type_agent_ids = self.type_to_agent_ids[agent_type]
                if self.state_type == "EP":
                    type_rnn_states_critic = np.mean(
                        rnn_states_critic[:, type_agent_ids], axis=1
                    )
                    type_values = np.mean(values[:, type_agent_ids], axis=1)
                    type_rewards = np.mean(rewards[:, type_agent_ids], axis=1)
                    type_masks = np.min(masks[:, type_agent_ids], axis=1)
                    self.critic_buffers_by_type[agent_type].insert(
                        share_obs[:, 0],
                        type_rnn_states_critic,
                        type_values,
                        type_rewards,
                        type_masks,
                        bad_masks,
                    )
                elif self.state_type == "FP":
                    self.critic_buffers_by_type[agent_type].insert(
                        share_obs[:, type_agent_ids],
                        rnn_states_critic[:, type_agent_ids],
                        values[:, type_agent_ids],
                        rewards[:, type_agent_ids],
                        masks[:, type_agent_ids],
                        bad_masks[:, type_agent_ids],
                    )

        if self.enable_central_q:
            all_actions = actions.reshape(actions.shape[0], -1)
            for agent_id in range(self.num_agents):
                cent_rnn_states = self._centralized_rnn_states[agent_id].copy()
                cent_rnn_states[dones_env == True] = np.zeros(
                    (
                        (dones_env == True).sum(),
                        self.recurrent_n,
                        self.rnn_hidden_size,
                    ),
                    dtype=np.float32,
                )
                if self.state_type == "EP":
                    self.centralized_critic_buffers[agent_id].insert(
                        share_obs[:, 0],
                        all_actions,
                        cent_rnn_states,
                        self._centralized_q_values[agent_id],
                        rewards[:, 0],
                        masks[:, 0],
                        bad_masks,
                    )
                elif self.state_type == "FP":
                    self.centralized_critic_buffers[agent_id].insert(
                        share_obs[:, agent_id],
                        all_actions,
                        cent_rnn_states,
                        self._centralized_q_values[agent_id],
                        rewards[:, agent_id],
                        masks[:, agent_id],
                        bad_masks[:, agent_id],
                    )

    @torch.no_grad()
    def compute(self):
        """Compute returns and advantages.
        Compute critic evaluation of the last state,
        and then let buffer compute returns, which will be used during training.
        """
        if not self.enable_heterogeneous_agents:
            if self.state_type == "EP":
                next_value, _ = self.critic.get_values(
                    self.critic_buffer.share_obs[-1],
                    self.critic_buffer.rnn_states_critic[-1],
                    self.critic_buffer.masks[-1],
                )
                next_value = _t2n(next_value)
            elif self.state_type == "FP":
                next_value, _ = self.critic.get_values(
                    np.concatenate(self.critic_buffer.share_obs[-1]),
                    np.concatenate(self.critic_buffer.rnn_states_critic[-1]),
                    np.concatenate(self.critic_buffer.masks[-1]),
                )
                next_value = np.array(
                    np.split(_t2n(next_value), self.algo_args["train"]["n_rollout_threads"])
                )
            self.critic_buffer.compute_returns(next_value, self.value_normalizer)
        else:
            for agent_type in self.type_order:
                critic = self.critics_by_type[agent_type]
                critic_buffer = self.critic_buffers_by_type[agent_type]
                value_normalizer = self.value_normalizers_by_type[agent_type]
                if self.state_type == "EP":
                    next_value, _ = critic.get_values(
                        critic_buffer.share_obs[-1],
                        critic_buffer.rnn_states_critic[-1],
                        critic_buffer.masks[-1],
                    )
                    next_value = _t2n(next_value)
                elif self.state_type == "FP":
                    next_value, _ = critic.get_values(
                        np.concatenate(critic_buffer.share_obs[-1]),
                        np.concatenate(critic_buffer.rnn_states_critic[-1]),
                        np.concatenate(critic_buffer.masks[-1]),
                    )
                    next_value = np.array(
                        np.split(_t2n(next_value), self.algo_args["train"]["n_rollout_threads"])
                    )
                critic_buffer.compute_returns(next_value, value_normalizer)

        if self.enable_central_q:
            # Run actors on the terminal observation to obtain next actions,
            # then bootstrap Q(s_T, a_T) for return computation.
            next_actions_collector = []
            for agent_id in range(self.num_agents):
                next_action, _, _ = self.actor[agent_id].get_actions(
                    self.actor_buffer[agent_id].obs[-1],
                    self.actor_buffer[agent_id].rnn_states[-1],
                    self.actor_buffer[agent_id].masks[-1],
                    self.actor_buffer[agent_id].available_actions[-1]
                    if self.actor_buffer[agent_id].available_actions is not None
                    else None,
                )
                next_actions_collector.append(_t2n(next_action))
            # Concatenate per-agent actions: list of (n_threads, act_dim) -> (n_threads, n_agents * act_dim)
            next_all_actions = np.concatenate(next_actions_collector, axis=-1)

            for agent_id in range(self.num_agents):
                next_cent_q, _ = self.centralized_critics[agent_id].get_values(
                    self.centralized_critic_buffers[agent_id].share_obs[-1],
                    next_all_actions,
                    self.centralized_critic_buffers[agent_id].rnn_states_critic[-1],
                    self.centralized_critic_buffers[agent_id].masks[-1],
                )
                next_cent_q = _t2n(next_cent_q)
                self.centralized_critic_buffers[agent_id].compute_returns(
                    next_cent_q, self.centralized_value_normalizers[agent_id]
                )

    def train(self):
        """Train the model."""
        raise NotImplementedError

    def after_update(self):
        """Do the necessary data operations after an update.
        After an update, copy the data at the last step to the first position of the buffer.
        This will be used for then generating new actions.
        """
        for agent_id in range(self.num_agents):
            self.actor_buffer[agent_id].after_update()
        if self.enable_heterogeneous_agents:
            for agent_type in self.type_order:
                self.critic_buffers_by_type[agent_type].after_update()
        else:
            self.critic_buffer.after_update()
        if self.enable_central_q:
            for agent_id in range(self.num_agents):
                self.centralized_critic_buffers[agent_id].after_update()

    @torch.no_grad()
    def eval(self):
        """Evaluate the model."""
        self.logger.eval_init()  # logger callback at the beginning of evaluation
        eval_episode = 0

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (
                self.algo_args["eval"]["n_eval_rollout_threads"],
                self.num_agents,
                self.recurrent_n,
                self.rnn_hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (self.algo_args["eval"]["n_eval_rollout_threads"], self.num_agents, 1),
            dtype=np.float32,
        )

        while True:
            eval_actions_collector = []
            for agent_id in range(self.num_agents):
                eval_actions, temp_rnn_state = self.actor[agent_id].act(
                    eval_obs[:, agent_id],
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    eval_available_actions[:, agent_id]
                    if eval_available_actions[0] is not None
                    else None,
                    deterministic=True,
                )
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)
            eval_data = (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            )
            self.logger.eval_per_step(
                eval_data
            )  # logger callback at each step of evaluation

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[
                eval_dones_env == True
            ] = np.zeros(  # if env is done, then reset rnn_state to all zero
                (
                    (eval_dones_env == True).sum(),
                    self.num_agents,
                    self.recurrent_n,
                    self.rnn_hidden_size,
                ),
                dtype=np.float32,
            )

            eval_masks = np.ones(
                (self.algo_args["eval"]["n_eval_rollout_threads"], self.num_agents, 1),
                dtype=np.float32,
            )
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
            )

            for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    self.logger.eval_thread_done(
                        eval_i
                    )  # logger callback when an episode is done

            if eval_episode >= self.algo_args["eval"]["eval_episodes"]:
                eval_avg_rew = self.logger.eval_log(
                    eval_episode
                )  # logger callback at the end of evaluation
                if self.enable_heterogeneous_agents:
                    eval_rewards_array = np.array(self.logger.eval_episode_rewards)
                    type_metrics = self._compute_type_eval_metrics(eval_rewards_array)
                    return {"overall": float(eval_avg_rew), "by_type": type_metrics}
                return eval_avg_rew

    @torch.no_grad()
    def render(self):
        """Render the model."""
        print("start rendering")
        if self.manual_expand_dims:
            # this env needs manual expansion of the num_of_parallel_envs dimension
            for _ in range(self.algo_args["render"]["render_episodes"]):
                eval_obs, _, eval_available_actions = self.envs.reset()
                eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                eval_available_actions = (
                    np.expand_dims(np.array(eval_available_actions), axis=0)
                    if eval_available_actions is not None
                    else None
                )
                eval_rnn_states = np.zeros(
                    (
                        self.env_num,
                        self.num_agents,
                        self.recurrent_n,
                        self.rnn_hidden_size,
                    ),
                    dtype=np.float32,
                )
                eval_masks = np.ones(
                    (self.env_num, self.num_agents, 1), dtype=np.float32
                )
                rewards = 0
                while True:
                    eval_actions_collector = []
                    for agent_id in range(self.num_agents):
                        eval_actions, temp_rnn_state = self.actor[agent_id].act(
                            eval_obs[:, agent_id],
                            eval_rnn_states[:, agent_id],
                            eval_masks[:, agent_id],
                            eval_available_actions[:, agent_id]
                            if eval_available_actions is not None
                            else None,
                            deterministic=True,
                        )
                        eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                        eval_actions_collector.append(_t2n(eval_actions))
                    eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
                    (
                        eval_obs,
                        _,
                        eval_rewards,
                        eval_dones,
                        _,
                        eval_available_actions,
                    ) = self.envs.step(eval_actions[0])
                    rewards += eval_rewards[0][0]
                    eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                    eval_available_actions = (
                        np.expand_dims(np.array(eval_available_actions), axis=0)
                        if eval_available_actions is not None
                        else None
                    )
                    if self.manual_render:
                        self.envs.render()
                    if self.manual_delay:
                        time.sleep(0.1)
                    if eval_dones[0]:
                        print(f"total reward of this episode: {rewards}")
                        break
        else:
            # this env does not need manual expansion of the num_of_parallel_envs dimension
            # such as dexhands, which instantiates a parallel env of 64 pair of hands
            for _ in range(self.algo_args["render"]["render_episodes"]):
                eval_obs, _, eval_available_actions = self.envs.reset()
                eval_rnn_states = np.zeros(
                    (
                        self.env_num,
                        self.num_agents,
                        self.recurrent_n,
                        self.rnn_hidden_size,
                    ),
                    dtype=np.float32,
                )
                eval_masks = np.ones(
                    (self.env_num, self.num_agents, 1), dtype=np.float32
                )
                rewards = 0
                while True:
                    eval_actions_collector = []
                    for agent_id in range(self.num_agents):
                        eval_actions, temp_rnn_state = self.actor[agent_id].act(
                            eval_obs[:, agent_id],
                            eval_rnn_states[:, agent_id],
                            eval_masks[:, agent_id],
                            eval_available_actions[:, agent_id]
                            if eval_available_actions[0] is not None
                            else None,
                            deterministic=True,
                        )
                        eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                        eval_actions_collector.append(_t2n(eval_actions))
                    eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
                    (
                        eval_obs,
                        _,
                        eval_rewards,
                        eval_dones,
                        _,
                        eval_available_actions,
                    ) = self.envs.step(eval_actions)
                    rewards += eval_rewards[0][0][0]
                    if self.manual_render:
                        self.envs.render()
                    if self.manual_delay:
                        time.sleep(0.1)
                    if eval_dones[0][0]:
                        print(f"total reward of this episode: {rewards}")
                        break
        if "smac" in self.args["env"]:  # replay for smac, no rendering
            if "v2" in self.args["env"]:
                self.envs.env.save_replay()
            else:
                self.envs.save_replay()

    def prep_rollout(self):
        """Prepare for rollout."""
        for agent_id in range(self.num_agents):
            self.actor[agent_id].prep_rollout()
        if self.enable_heterogeneous_agents:
            for agent_type in self.type_order:
                self.critics_by_type[agent_type].prep_rollout()
        else:
            self.critic.prep_rollout()
        if self.enable_central_q:
            for agent_id in range(self.num_agents):
                self.centralized_critics[agent_id].prep_rollout()

    def prep_training(self):
        """Prepare for training."""
        for agent_id in range(self.num_agents):
            self.actor[agent_id].prep_training()
        if self.enable_heterogeneous_agents:
            for agent_type in self.type_order:
                self.critics_by_type[agent_type].prep_training()
        else:
            self.critic.prep_training()
        if self.enable_central_q:
            for agent_id in range(self.num_agents):
                self.centralized_critics[agent_id].prep_training()

    def save(self, mean_reward=None):
        """Save model parameters."""
        suffix = f"_rew{mean_reward:.4f}" if mean_reward is not None else ""
        for agent_id in range(self.num_agents):
            policy_actor = self.actor[agent_id].actor
            torch.save(
                policy_actor.state_dict(),
                str(self.save_dir) + f"/actor_agent{agent_id}{suffix}.pt",
            )
        if not self.enable_heterogeneous_agents:
            policy_critic = self.critic.critic
            torch.save(
                policy_critic.state_dict(),
                str(self.save_dir) + f"/critic_agent{suffix}.pt",
            )
            if self.value_normalizer is not None:
                torch.save(
                    self.value_normalizer.state_dict(),
                    str(self.save_dir) + f"/value_normalizer{suffix}.pt",
                )
        else:
            for agent_type in self.type_order:
                torch.save(
                    self.critics_by_type[agent_type].critic.state_dict(),
                    str(self.save_dir) + f"/critic_type_{agent_type}{suffix}.pt",
                )
                if self.value_normalizers_by_type[agent_type] is not None:
                    torch.save(
                        self.value_normalizers_by_type[agent_type].state_dict(),
                        str(self.save_dir) + f"/value_normalizer_type_{agent_type}{suffix}.pt",
                    )
        if self.enable_central_q:
            for agent_id in range(self.num_agents):
                torch.save(
                    self.centralized_critics[agent_id].critic.state_dict(),
                    str(self.save_dir)
                    + f"/central_q_critic_agent{agent_id}{suffix}.pt",
                )
                if self.centralized_value_normalizers[agent_id] is not None:
                    torch.save(
                        self.centralized_value_normalizers[agent_id].state_dict(),
                        str(self.save_dir)
                        + f"/central_q_value_normalizer_agent{agent_id}{suffix}.pt",
                    )

    def save_by_agent_type(self, agent_type, mean_reward):
        """Save checkpoints for a specific agent type based on its own eval metric."""
        suffix = f"_type_{agent_type}_rew{mean_reward:.4f}"
        for agent_id in self.type_to_agent_ids[agent_type]:
            policy_actor = self.actor[agent_id].actor
            torch.save(
                policy_actor.state_dict(),
                str(self.save_dir) + f"/actor_agent{agent_id}{suffix}.pt",
            )

        if self.enable_heterogeneous_agents:
            torch.save(
                self.critics_by_type[agent_type].critic.state_dict(),
                str(self.save_dir) + f"/critic_type_{agent_type}{suffix}.pt",
            )
            if self.value_normalizers_by_type[agent_type] is not None:
                torch.save(
                    self.value_normalizers_by_type[agent_type].state_dict(),
                    str(self.save_dir)
                    + f"/value_normalizer_type_{agent_type}{suffix}.pt",
                )

        if self.enable_central_q:
            for agent_id in self.type_to_agent_ids[agent_type]:
                torch.save(
                    self.centralized_critics[agent_id].critic.state_dict(),
                    str(self.save_dir)
                    + f"/central_q_critic_agent{agent_id}{suffix}.pt",
                )
                if self.centralized_value_normalizers[agent_id] is not None:
                    torch.save(
                        self.centralized_value_normalizers[agent_id].state_dict(),
                        str(self.save_dir)
                        + f"/central_q_value_normalizer_agent{agent_id}{suffix}.pt",
                    )


    def restore(self):
        """Restore model parameters."""
        model_dir = self.algo_args["train"]["model_dir"]
        for agent_id in range(self.num_agents):
            path = find_checkpoint(model_dir, f"actor_agent{agent_id}")
            self.actor[agent_id].actor.load_state_dict(torch.load(path, map_location=self.device))
        if not self.algo_args["render"]["use_render"]:
            if not self.enable_heterogeneous_agents:
                self.critic.critic.load_state_dict(
                    torch.load(find_checkpoint(model_dir, "critic_agent"), map_location=self.device)
                )
                if self.value_normalizer is not None:
                    self.value_normalizer.load_state_dict(
                        torch.load(find_checkpoint(model_dir, "value_normalizer"), map_location=self.device)
                    )
            else:
                for agent_type in self.type_order:
                    self.critics_by_type[agent_type].critic.load_state_dict(
                        torch.load(find_checkpoint(model_dir, f"critic_type_{agent_type}"), map_location=self.device)
                    )
                    if self.value_normalizers_by_type[agent_type] is not None:
                        self.value_normalizers_by_type[agent_type].load_state_dict(
                            torch.load(
                                find_checkpoint(
                                    model_dir, f"value_normalizer_type_{agent_type}"
                                ),
                                map_location=self.device,
                            )
                        )
            if self.enable_central_q:
                for agent_id in range(self.num_agents):
                    self.centralized_critics[agent_id].critic.load_state_dict(
                        torch.load(
                            find_checkpoint(
                                model_dir, f"central_q_critic_agent{agent_id}"
                            ),
                            map_location=self.device,
                        )
                    )
                    if self.centralized_value_normalizers[agent_id] is not None:
                        self.centralized_value_normalizers[agent_id].load_state_dict(
                            torch.load(
                                find_checkpoint(
                                    model_dir,
                                    f"central_q_value_normalizer_agent{agent_id}",
                                ),
                                map_location=self.device,
                            )
                        )

    def close(self):
        """Close environment, writter, and logger."""
        if self.algo_args["render"]["use_render"]:
            self.envs.close()
        else:
            self.envs.close()
            if self.algo_args["eval"]["use_eval"] and self.eval_envs is not self.envs:
                self.eval_envs.close()
            self.writter.export_scalars_to_json(str(self.log_dir + "/summary.json"))
            self.writter.close()
            self.logger.close()
