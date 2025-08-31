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
        self.args["render_mode"] = "rgb_array"
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
        # self.args["N"] = 5 # for simple spread
        self.cur_step = 0
        # Normalize scenario to available module without try/except
        from importlib.util import find_spec

        mod_name = "pettingzoo.mpe." + self.scenario
        if self.scenario.endswith("_v3") and find_spec(mod_name) is None:
            print(f"##### Scenario {self.scenario} not found, falling back to v2 #####",flush=True)
            self.scenario = self.scenario[:-3] + "_v2"
            mod_name = "pettingzoo.mpe." + self.scenario
            print(f"## MODULE NAME: {mod_name} ##",flush=True)
        self.module = importlib.import_module(mod_name)
        self.env = ss.pad_action_space_v0(
            ss.pad_observations_v0(self.module.parallel_env(**self.args))
        )
        # Initial reset to populate agents/spaces; handle tuple/dict return
        out = self.env.reset(seed=None)
        if isinstance(out, tuple):
            obs_init, _ = out
        else:
            obs_init = out
        self.n_agents = self.env.num_agents
        self.agents = self.env.agents
        # Share/global observation space (prefer env.state_space if available)
        if hasattr(self.env, "state_space"):
            self.share_observation_space = self.repeat(self.env.state_space)
        else:
            # Fallback: build a flat state by concatenating per-agent obs shapes
            # (shape-only proxy; actual state built at runtime)
            dims = []
            for agent in self.agents:
                sp = (
                    self.env.observation_spaces[agent]
                    if hasattr(self.env, "observation_spaces")
                    else self.env.observation_space(agent)
                )
                dims.append(int(np.prod(sp.shape)))
            share_dim = int(sum(dims))
            from gym import spaces as gym_spaces  # local import to avoid hard dependency at top

            self.share_observation_space = self.repeat(
                gym_spaces.Box(low=-np.inf, high=np.inf, shape=(share_dim,), dtype=np.float32)
            )
        # Local obs/action spaces (support both dict properties and callable accessors)
        if hasattr(self.env, "observation_spaces"):
            self.observation_space = [self.env.observation_spaces[a] for a in self.agents]
        else:
            self.observation_space = [self.env.observation_space(a) for a in self.agents]
        if hasattr(self.env, "action_spaces"):
            self.action_space = [self.env.action_spaces[a] for a in self.agents]
        else:
            self.action_space = [self.env.action_space(a) for a in self.agents]
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
        # Build shared/global observation
        if hasattr(self.env, "state"):
            s_vec = self.env.state()
        else:
            s_vec = np.concatenate([obs[a] for a in self.agents], axis=0)
        s_obs = self.repeat(s_vec)
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

    def reset(self, seed=None):
        """Returns initial observations and states. Handles both legacy and gymnasium-style APIs."""
        # If a seed is provided externally, honor it; otherwise advance internal seed
        if seed is not None:
            self._seed = seed
        else:
            self._seed += 1
        self.cur_step = 0
        out = self.env.reset(seed=self._seed)
        # PettingZoo/Gymnasium: reset may return (obs_dict, infos)
        if isinstance(out, tuple):
            obs_dict, _ = out
        else:
            obs_dict = out
        # Build shared/global observation
        if hasattr(self.env, "state"):
            s_vec = self.env.state()
        else:
            s_vec = np.concatenate([obs_dict[a] for a in self.agents], axis=0)
        s_obs = self.repeat(s_vec)
        obs = self.unwrap(obs_dict)
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