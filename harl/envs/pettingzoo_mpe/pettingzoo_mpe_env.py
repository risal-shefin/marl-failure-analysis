# import copy
# import importlib
# import logging
# import numpy as np
# import supersuit as ss
# from gym import spaces as gym_spaces

# logging.basicConfig()
# logging.getLogger().setLevel(logging.ERROR)


# class PettingZooMPEEnv:
#     def __init__(self, args):
#         self.args = copy.deepcopy(args)
#         self.scenario = args["scenario"]
#         del self.args["scenario"]
#         # default; can be auto-detected after env creation as well
#         self.discrete = True
#         if (
#             "continuous_actions" in self.args
#             and self.args["continuous_actions"] == True
#         ):
#             self.discrete = False
#         if "max_cycles" in self.args:
#             self.max_cycles = self.args["max_cycles"]
#             self.args["max_cycles"] += 1
#         else:
#             self.max_cycles = 25
#             self.args["max_cycles"] = 26
#         # self.args["N"] = 5 # for simple spread
#         self.cur_step = 0

#         # Deterministic domain routing without try/except
#         self._domain = "sisl" if ("multiwalker" in self.scenario) else "mpe"
#         module_path = "pettingzoo." + self._domain + "." + self.scenario
#         module_args = dict(self.args)
#         if self._domain == "sisl":
#             # Do not pass HARL's discrete/continuous flag to SISL env constructors
#             module_args.pop("continuous_actions", None)
#         self.module = importlib.import_module(module_path)

#         self.env = ss.pad_action_space_v0(
#             ss.pad_observations_v0(self.module.parallel_env(**module_args))
#         )
#         # initial reset (no seed) to initialize spaces
#         self.env.reset()
#         self.n_agents = self.env.num_agents
#         self.agents = self.env.agents
#         self.observation_space = self.unwrap(self.env.observation_spaces)
#         self.action_space = self.unwrap(self.env.action_spaces)

#         # Auto-detect discrete/continuous from action spaces (no gymnasium import)
#         self.discrete = all(hasattr(sp, "n") for sp in self.action_space)

#         # Determine if env provides a global state/state_space
#         self._has_state = hasattr(self.env, "state_space")

#         # Share observation space: prefer env.state_space when available, else build by concatenating obs spaces
#         if self._has_state:
#             self.share_observation_space = self.repeat(self.env.state_space)
#         else:
#             # compute flat dim across agents
#             dims = []
#             for sp in self.observation_space:
#                 if hasattr(sp, "shape") and sp.shape is not None:
#                     dims.append(int(np.prod(sp.shape)))
#                 else:
#                     # fallback: cannot infer, assume zero
#                     dims.append(0)
#             share_dim = int(sum(dims))
#             share_box = gym_spaces.Box(low=-np.inf, high=np.inf, shape=(share_dim,), dtype=np.float32)
#             self.share_observation_space = self.repeat(share_box)

#         self._seed = 0

#     def step(self, actions):
#         """
#         return local_obs, global_state, rewards, dones, infos, available_actions
#         """
#         if self.discrete:
#             obs_dict, rew, term, trunc, info = self.env.step(self.wrap(actions.flatten()))
#         else:
#             obs_dict, rew, term, trunc, info = self.env.step(self.wrap(actions))
#         self.cur_step += 1
#         if self.cur_step == self.max_cycles:
#             trunc = {agent: True for agent in self.agents}
#             for agent in self.agents:
#                 info[agent]["bad_transition"] = True
#         dones = {agent: term[agent] or trunc[agent] for agent in self.agents}
#         # share/global obs
#         if self._has_state:
#             s_vec = self.env.state()
#         else:
#             s_vec = np.concatenate([obs_dict[agent] for agent in self.agents], axis=0)
#         s_obs = self.repeat(s_vec)
#         total_reward = sum([rew[agent] for agent in self.agents])
#         rewards = [[total_reward]] * self.n_agents
#         return (
#             self.unwrap(obs_dict),
#             s_obs,
#             rewards,
#             self.unwrap(dones),
#             self.unwrap(info),
#             self.get_avail_actions(),
#         )

#     def reset(self, seed=None):
#         """Returns initial observations and states"""
#         print("%%%% SEED is ", seed)
#         self._seed += 1
#         self.cur_step = 0
#         obs_dict = self.env.reset(seed=seed if seed is not None else self._seed)
#         if self._has_state:
#             s_vec = self.env.state()
#         else:
#             s_vec = np.concatenate([obs_dict[agent] for agent in self.agents], axis=0)
#         s_obs = self.repeat(s_vec)
#         return self.unwrap(obs_dict), s_obs, self.get_avail_actions()

#     def get_avail_actions(self):
#         if self.discrete:
#             avail_actions = []
#             for agent_id in range(self.n_agents):
#                 avail_agent = self.get_avail_agent_actions(agent_id)
#                 avail_actions.append(avail_agent)
#             return avail_actions
#         else:
#             return None

#     def get_avail_agent_actions(self, agent_id):
#         """Returns the available actions for agent_id"""
#         return [1] * self.action_space[agent_id].n if self.discrete else None

#     def render(self):
#         self.env.render()

#     def close(self):
#         self.env.close()

#     def seed(self, seed):
#         self._seed = seed

#     def wrap(self, l):
#         d = {}
#         for i, agent in enumerate(self.agents):
#             d[agent] = l[i]
#         return d

#     def unwrap(self, d):
#         l = []
#         for agent in self.agents:
#             l.append(d[agent])
#         return l

#     def repeat(self, a):
#         return [a for _ in range(self.n_agents)]


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
        self.args["N"] = 5 # for simple spread
        self.cur_step = 0
        self.module = importlib.import_module("pettingzoo.mpe." + self.scenario)
        self.env = ss.pad_action_space_v0(
            ss.pad_observations_v0(self.module.parallel_env(**self.args))
        )
        self.env.reset()
        self.n_agents = self.env.num_agents
        self.agents = self.env.agents
        self.share_observation_space = self.repeat(self.env.state_space)
        self.observation_space = self.unwrap(self.env.observation_spaces)
        self.action_space = self.unwrap(self.env.action_spaces)
        self._seed = 0

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
        total_reward = sum([rew[agent] for agent in self.agents])
        rewards = [[total_reward]] * self.n_agents
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
        obs = self.unwrap(self.env.reset(seed=self._seed))
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
        self.env.render()

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