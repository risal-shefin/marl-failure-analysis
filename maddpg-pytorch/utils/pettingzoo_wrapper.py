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
        obs_dict, infos = self.env.reset() if seed is None else self.env.reset(seed=seed)
        masks = []
        if hasattr(self.env, 'action_masks'):
            mask_dict = self.env.action_masks()
            for agent in self.env.possible_agents:
                masks.append(mask_dict.get(agent))
        elif isinstance(infos, dict):
            for agent in self.env.possible_agents:
                info = infos.get(agent)
                mask = None
                if info:
                    mask = info.get('action_mask')
                    if mask is None:
                        mask = info.get('avail_actions')
                masks.append(mask)
        else:
            masks = [None] * len(self.env.possible_agents)
        obs = [obs_dict[agent] for agent in self.env.possible_agents]
        return obs, masks

    def step(self, actions):
        obs_dict, rewards_dict, termination, truncation, infos_dict = self.env.step(actions)
        states = []
        rewards = []
        infos = []
        dones = []
        masks = []
        for agent in self.env.possible_agents:
            if agent in self.env.agents:  # Check if agent is still in the game
                states.append(obs_dict[agent])
                rewards.append(rewards_dict[agent])
                info = infos_dict[agent]
                infos.append(info)
                dones.append(termination[agent] or truncation[agent])
                mask = info.get('action_mask')
                if mask is None:
                    mask = info.get('avail_actions')
                masks.append(mask)
            else:
                states.append(np.zeros(self.env.observation_space(agent).shape))  # Placeholder for dead agents
                rewards.append(0.0)
                infos.append({})
                dones.append(True)
                masks.append(None)
        rewards = np.array([rewards])
        dones = np.array([dones])
        infos = np.array([infos])
        return states, rewards, dones, infos, masks

    def close(self):
        return self.env.close()

    def __getattr__(self, name):
        # Delegate attribute access to the underlying env
        return getattr(self.env, name)
