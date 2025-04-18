import numpy as np

class PettingZooWrapper:
    """Wraps a PettingZoo parallel env to provide list-based obs/action and vectorized step/reset."""
    def __init__(self, env):
        self.env = env
        self.action_space = list(env.action_spaces.values())
        self.observation_space = list(env.observation_spaces.values())
        self.agent_types = ["adversary" if name.find("adversary") != -1 else "agent" for name in env.possible_agents]

    @staticmethod
    def wrap_env(env):
        return PettingZooWrapper(env)

    def reset(self, seed=None):
        obs_dict, _ = self.env.reset() if seed is None else self.env.reset(seed=seed)
        return list(obs_dict.values())

    def step(self, actions):
        obs_dict, rewards, termination, truncation, infos = self.env.step(actions)
        rewards = np.array([list(rewards.values())])
        infos = np.array([list(infos.values())])
        dones = np.array([[termination[a] or truncation[a] for a in self.env.possible_agents]])
        return list(obs_dict.values()), rewards, dones, infos

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        # Delegate attribute access to the underlying env
        return getattr(self.env, name)
