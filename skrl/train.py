from skrl.multi_agents.torch.ippo import IPPO, IPPO_DEFAULT_CONFIG
from skrl.multi_agents.torch.mappo import MAPPO, MAPPO_DEFAULT_CONFIG
from skrl.trainers.torch import SequentialTrainer
from skrl.memories.torch import RandomMemory
from skrl.envs.wrappers.torch import wrap_env
from skrl.models.torch import Model, CategoricalMixin, DeterministicMixin
import pettingzoo.mpe as mpe
import torch
import torch.nn as nn
import argparse
import gymnasium
import numpy as np


# define the model
# CategoricalMixin for Discrete Actions
class PolicyCategorical(CategoricalMixin, Model):
    def __init__(self, observation_space, action_space, device, unnormalized_log_prob=True, eval_mode=False):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)
        self.eval_mode = eval_mode

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_actions)
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


class ValueDeterministic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


def main(args):
    # Dynamically import the environment from pettingzoo.mpe
    try:
        env_func = getattr(mpe, args.env_id)
    except AttributeError:
        raise ValueError(f"Environment {args.env_id} not found in pettingzoo.mpe")
    env = env_func.parallel_env(continuous_actions=False)

    # wrap the environment
    env = wrap_env(env)  # or 'env = wrap_env(env, wrapper="pettingzoo")'

    # Agent configs
    cfg_agent = {}
    agent_kwargs = {}
    agent_class: IPPO | MAPPO = None
    if args.algo_name == 'IPPO':
        cfg_agent = IPPO_DEFAULT_CONFIG.copy()
        agent_class = IPPO
    elif args.algo_name == 'MAPPO':
        cfg_agent = MAPPO_DEFAULT_CONFIG.copy()
        agent_class = MAPPO

        shared_observation_spaces_low = []
        shared_observation_spaces_high = []
        for agent_name in env.possible_agents:
            shared_observation_spaces_low.append(env.observation_spaces[agent_name].low)
            shared_observation_spaces_high.append(env.observation_spaces[agent_name].high)
        shared_observation_space = gymnasium.spaces.Box(
            low=np.concatenate(shared_observation_spaces_low),
            high=np.concatenate(shared_observation_spaces_high),
            dtype=np.float32
        )
        agent_kwargs.update({"shared_observation_spaces": {agent_name: shared_observation_space for agent_name in env.possible_agents}})
    else:
        raise ValueError(f"Algorithm {args.algo_name} is not supported")
    
    # adjust some configuration if necessary
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

    # instantiate the agent's models and memories
    models = {}
    memories = {}
    for agent_name in env.possible_agents:
        models[agent_name] = {}

        models[agent_name]["policy"] = PolicyCategorical(
            observation_space=env.observation_space(agent_name), 
            action_space=env.action_space(agent_name), 
            device=env.device)
        
        value_obs_space = env.observation_space(agent_name)
        if args.algo_name == 'MAPPO':
            value_obs_space = shared_observation_space
        models[agent_name]["value"] = ValueDeterministic(observation_space=value_obs_space, 
            action_space=env.action_space(agent_name), 
            device=env.device)
        
        memories[agent_name] = RandomMemory(memory_size=cfg_agent['rollouts'], num_envs=env.num_envs, device=env.device)


    # instantiate the agent
    # (assuming a defined environment <env> and memories <memories>)
    agent = agent_class(possible_agents=env.possible_agents,
                models=models,
                memories=memories,  # only required during training
                cfg=cfg_agent,
                observation_spaces=env.observation_spaces,
                action_spaces=env.action_spaces,
                device=env.device,
                **agent_kwargs)


    # create a sequential trainer
    cfg = {"timesteps": args.train_steps, "headless": False}
    trainer = SequentialTrainer(env=env, agents=agent, cfg=cfg)

    # train the agent(s)
    trainer.train()

    # evaluate the agent(s)
    trainer.eval()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL agent on a PettingZoo MPE environment")
    parser.add_argument("--env_id", type=str,
                        help="Name of the environment from pettingzoo.mpe (for ex: simple_speaker_listener_v4)")
    parser.add_argument("--algo_name", type=str, default='IPPO',
                        help="Algorithm Name")
    parser.add_argument("--train_steps", type=int, default=10_000_000,
                        help="Total Train Steps")
    args = parser.parse_args()
    main(args)