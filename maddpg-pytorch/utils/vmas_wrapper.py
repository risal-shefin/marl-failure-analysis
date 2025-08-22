import numpy as np
import gym
import vmas

class VmasWrapper:
    def __init__(self, env):
        self.env = env
        self.nagents = len(env.agents)
        self.agent_types = ["agent" for _ in range(self.nagents)]

    @staticmethod
    def wrap_env(env):
        return VmasWrapper(env)

    @staticmethod
    def make_and_wrap_env(env_id, device='cpu', num_envs=1, n_agents=5, seed=None, max_steps=100, is_discrete_action=False):
        if device == 'gpu':
            device = 'cuda'
        env = vmas.make_env(
            scenario=env_id, # can be scenario name or BaseScenario class
            num_envs=num_envs,
            device=device,
            continuous_actions=not is_discrete_action,
            max_steps=max_steps, # Defines the horizon. None is infinite horizon.
            seed=seed, # Seed of the environment
            n_agents=n_agents  # Additional arguments you want to pass to the scenario
        )
        return VmasWrapper.wrap_env(env)

    def step(self, actions):
        obs, rewards, dones, info = self.env.step(actions)
        rewards = np.array(rewards).T   # transfom shape (nagents, batch) to (batch, nagents)
        dones = np.array([dones for _ in range(self.nagents)]).T
        info = np.array([info])
        masks = [None for _ in range(self.nagents)]
        return obs, rewards, dones, info, masks

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        masks = [None for _ in range(self.nagents)]
        return obs, masks

    def __getattr__(self, name):
        # Delegate attribute access to the underlying env
        return getattr(self.env, name)
