import copy
import importlib
import logging
import numpy as np
import supersuit as ss

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)


class PettingZooMPEEnv:
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.scenario = args["scenario"]
        del self.args["scenario"]
        self.enable_heterogeneous_agents = self.args.pop(
            "enable_heterogeneous_agents", False
        )
        self.reward_mode = self.args.pop("reward_mode", "global_sum_shared")
        self._validate_reward_mode()

        self.discrete = True
        if (
            "continuous_actions" in self.args
            and self.args["continuous_actions"] == True
        ):
            self.discrete = False
        if "max_cycles" in self.args:
            self.max_cycles = self.args["max_cycles"]
            self.args["max_cycles"] += 1
        else:
            self.max_cycles = 25
            self.args["max_cycles"] = 26
        self.cur_step = 0
        self.module = importlib.import_module("pettingzoo.mpe." + self.scenario)
        self.env = ss.pad_action_space_v0(
            ss.pad_observations_v0(self.module.parallel_env(**self.args))
        )
        self.env.reset()
        self.n_agents = self.env.num_agents
        self.agents = self.env.agents
        self.agent_types = [self._agent_type_from_name(agent) for agent in self.agents]
        self.agent_id_to_type = {
            agent_id: self.agent_types[agent_id] for agent_id in range(self.n_agents)
        }
        self.type_to_agent_ids = {}
        for agent_id, agent_type in enumerate(self.agent_types):
            self.type_to_agent_ids.setdefault(agent_type, []).append(agent_id)
        self.share_observation_space = self.repeat(self.env.state_space)
        self.observation_space = self.unwrap(self.env.observation_spaces)
        self.action_space = self.unwrap(self.env.action_spaces)
        self._seed = 0

    def _validate_reward_mode(self):
        allowed_modes = {"global_sum_shared", "team_by_type", "individual"}
        if self.reward_mode not in allowed_modes:
            raise ValueError(
                f"Unsupported reward_mode={self.reward_mode}. "
                f"Expected one of {sorted(allowed_modes)}"
            )
        if (not self.enable_heterogeneous_agents) and self.reward_mode != "global_sum_shared":
            raise ValueError(
                "Non-legacy reward_mode is only allowed when "
                "enable_heterogeneous_agents=True"
            )

    @staticmethod
    def _agent_type_from_name(agent_name):
        return agent_name.split("_", 1)[0]

    def _aggregate_rewards(self, rew):
        if self.reward_mode == "individual":
            return [[rew[agent]] for agent in self.agents]

        if self.reward_mode == "team_by_type":
            team_rewards = {}
            for agent_id, agent in enumerate(self.agents):
                agent_type = self.agent_id_to_type[agent_id]
                team_rewards.setdefault(agent_type, 0.0)
                team_rewards[agent_type] += rew[agent]
            return [
                [team_rewards[self.agent_id_to_type[agent_id]]]
                for agent_id in range(self.n_agents)
            ]

        total_reward = sum([rew[agent] for agent in self.agents])
        return [[total_reward]] * self.n_agents

    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """
        if self.discrete:
            obs, rew, term, trunc, info = self.env.step(self.wrap(actions.flatten()))
        else:
            obs, rew, term, trunc, info = self.env.step(self.wrap(actions))
        self.cur_step += 1
        if self.cur_step == self.max_cycles:
            trunc = {agent: True for agent in self.agents}
            for agent in self.agents:
                info[agent]["bad_transition"] = True
        dones = {agent: term[agent] or trunc[agent] for agent in self.agents}
        s_obs = self.repeat(self.env.state())
        rewards = self._aggregate_rewards(rew)
        return (
            self.unwrap(obs),
            s_obs,
            rewards,
            self.unwrap(dones),
            self.unwrap(info),
            self.get_avail_actions(),
        )

    def reset(self):
        """Returns initial observations and states"""
        self._seed += 1
        self.cur_step = 0
        reset_result = self.env.reset(seed=self._seed)
        # Newer PettingZoo returns (obs, infos) tuple; older returns just obs dict
        obs_dict = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        obs = self.unwrap(obs_dict)
        s_obs = self.repeat(self.env.state())
        return obs, s_obs, self.get_avail_actions()

    def get_avail_actions(self):
        if self.discrete:
            avail_actions = []
            for agent_id in range(self.n_agents):
                avail_agent = self.get_avail_agent_actions(agent_id)
                avail_actions.append(avail_agent)
            return avail_actions
        else:
            return None

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return [1] * self.action_space[agent_id].n

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed):
        self._seed = seed

    def wrap(self, l):
        d = {}
        for i, agent in enumerate(self.agents):
            d[agent] = l[i]
        return d

    def unwrap(self, d):
        l = []
        for agent in self.agents:
            l.append(d[agent])
        return l

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]
