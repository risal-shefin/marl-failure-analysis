import numpy as np

from smac.env.pettingzoo import StarCraft2PZEnv


class SmacWrapper:
    """Wrap StarCraft2 PettingZoo env to provide list-based obs and masks."""

    def __init__(self, env):
        self.env = env
        self.action_space = [env.action_space(agent) for agent in env.possible_agents]
        self.observation_space = [env.observation_space(agent)['observation'] for agent in env.possible_agents]
        self.action_mask_space = [env.observation_space(agent)['action_mask'] for agent in env.possible_agents]
        self.agent_types = ["agent" for _ in env.possible_agents]

    @staticmethod
    def wrap_env(env):
        return SmacWrapper(env)

    @staticmethod
    def make_env(map_name: str):
        env = StarCraft2PZEnv.parallel_env(map_name=map_name)
        return SmacWrapper.wrap_env(env)

    def reset(self):
        obs_info = self.env.reset()
        obs = []
        masks = []
        for agent in self.env.possible_agents:
            info = obs_info[agent]
            obs.append(info["observation"])
            masks.append(info["action_mask"])
        return obs, masks
    
    def seed(self, seed):
        self.env.seed(seed)

    def step(self, actions):
        obs_info, rewards, terminations, truncations, infos = self.env.step(actions)
        states = []
        rewards_list = []
        dones = []
        masks = []
        for i, agent in enumerate(self.env.possible_agents):
            if agent in self.env.agents:
                states.append(obs_info[agent]["observation"])
                rewards_list.append(rewards[agent])
                done = terminations[agent] or truncations[agent]
                dones.append(done)
                masks.append(obs_info[agent]["action_mask"])
            else:
                states.append(np.zeros(self.observation_space[i].shape))
                rewards_list.append(0.0)
                dones.append(True)
                action_mask = np.zeros(self.action_mask_space[i].shape)
                action_mask[0] = 1 # dead agent can only take noop action
                masks.append(action_mask)
        rewards_arr = np.array([rewards_list])
        dones_arr = np.array([dones])
        return states, rewards_arr, dones_arr, infos, masks

    def close(self):
        return self.env.close()
    
    def render(self):
        return self.env.env.render(mode="rgb_array")

    def __getattr__(self, name):
        return getattr(self.env, name)
