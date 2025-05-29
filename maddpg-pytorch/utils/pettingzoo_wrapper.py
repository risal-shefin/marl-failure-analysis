import numpy as np

class PettingZooWrapper:
    """Wraps a PettingZoo parallel env to provide list-based obs/action and vectorized step/reset."""
    def __init__(self, env):
        self.env = env
        self.action_space = [env.action_space(agent) for agent in env.possible_agents]
        self.observation_space = [env.observation_space(agent) for agent in env.possible_agents]
        self.agent_types = ["adversary" if name.find("adversary") != -1 else "agent" for name in env.possible_agents]

    @staticmethod
    def wrap_env(env):
        return PettingZooWrapper(env)

    def reset(self, seed=None):
        obs_dict, _ = self.env.reset() if seed is None else self.env.reset(seed=seed)
        return list(obs_dict.values())

    def step(self, actions):
        obs_dict, rewards_dict, termination, truncation, infos_dict = self.env.step(actions)
        states = []
        rewards = []
        infos = []
        dones = []
        for agent in self.env.possible_agents:
            if agent in self.env.agents:    # Check if agent is still in the game
                states.append(obs_dict[agent])
                rewards.append(rewards_dict[agent])
                infos.append(infos_dict[agent])
                dones.append(termination[agent] or truncation[agent])
            else:
                states.append(np.zeros(self.env.observation_space(agent).shape))  # Placeholder for dead agents
                rewards.append(0.0)
                infos.append({})
                dones.append(True)
        rewards = np.array([rewards])
        dones = np.array([dones])
        infos = np.array([infos])
        return states, rewards, dones, infos

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        # Delegate attribute access to the underlying env
        return getattr(self.env, name)
