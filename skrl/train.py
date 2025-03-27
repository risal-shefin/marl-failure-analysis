from skrl.multi_agents.torch.ippo import IPPO, IPPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.memories.torch import RandomMemory
from skrl.envs.wrappers.torch import wrap_env
from skrl.models.torch import Model, CategoricalMixin, DeterministicMixin
from pettingzoo.mpe import simple_spread_v3
import torch
import torch.nn as nn


# define the model
# CategoricalMixin for Discrete Actions
class MLP(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU(),
                                 nn.Linear(64, 32),
                                 nn.ELU(),
                                 nn.Linear(32, self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


class Value(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU(),
                                 nn.Linear(64, 32),
                                 nn.ELU(),
                                 nn.Linear(32, 1))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}



# load the environment
env = simple_spread_v3.parallel_env(continuous_actions=False)

# wrap the environment
env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="pettingzoo")'

# adjust some configuration if necessary
cfg_agent = IPPO_DEFAULT_CONFIG.copy()
cfg_agent.update({
    "rollouts": 128,                 # Increased rollouts for more stable updates
    "mini_batches": 8,              # Increased mini-batches for better gradient updates
    "learning_rate": 3e-4,                  # Reduced learning rate for stability
    "learning_rate_scheduler": torch.optim.lr_scheduler.LinearLR,
    "learning_rate_scheduler_kwargs": {"start_factor": 1.0, "end_factor": 0.1, "total_iters": 1e6},
    "grad_norm_clip": 1.0,              # Increased gradient clipping
    "entropy_loss_scale": 0.01,      # Added entropy loss scaling for better exploration
    "experiment": {
        "checkpoint_interval": 500,
    }
})

# instantiate the agent's models
models = {}
memories = {}
for agent_name in env.possible_agents:
    models[agent_name] = {}
    models[agent_name]["policy"] = MLP(
        observation_space=env.observation_space(agent_name), 
        action_space=env.action_space(agent_name), 
        device=env.device)
    models[agent_name]["value"] = Value(observation_space=env.observation_space(agent_name), 
        action_space=env.action_space(agent_name), 
        device=env.device)
    
    memories[agent_name] = RandomMemory(memory_size=cfg_agent['rollouts'], num_envs=env.num_envs, device=env.device)


# instantiate the agent
# (assuming a defined environment <env> and memories <memories>)
agent = IPPO(possible_agents=env.possible_agents,
             models=models,
             memories=memories,  # only required during training
             cfg=cfg_agent,
             observation_spaces=env.observation_spaces,
             action_spaces=env.action_spaces,
             device=env.device)



# create a sequential trainer
cfg = {"timesteps": 10000000, "headless": False}
trainer = SequentialTrainer(env=env, agents=agent, cfg=cfg)

# train the agent(s)
trainer.train()

# evaluate the agent(s)
trainer.eval()
