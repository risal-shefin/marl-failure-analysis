import numpy as np
import torch
import argparse
import datetime
import os
import pettingzoo.mpe as mpe
from tqdm import tqdm, trange

from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
from agilerl.utils.algo_utils import obs_channels_to_first
from agilerl.utils.utils import create_population
from agilerl.algorithms.maddpg import MADDPG
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.training.train_multi_agent import train_multi_agent

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_network_config():
    # Define the network configuration
    NET_CONFIG = {
        "head_config": {"hidden_size": [32, 32]}  # Actor head hidden size
    }
    return NET_CONFIG

def get_initial_hp(algo_name):
    # Define the initial hyperparameters
    INIT_HP = {
        # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        "ALGO": algo_name,
        "CHANNELS_LAST": False,
        "BATCH_SIZE": 32,  # Batch size
        "O_U_NOISE": True,  # Ornstein Uhlenbeck action noise
        "EXPL_NOISE": 0.1,  # Action noise scale
        "MEAN_NOISE": 0.0,  # Mean action noise
        "THETA": 0.15,  # Rate of mean reversion in OU noise
        "DT": 0.01,  # Timestep for OU noise
        "LR_ACTOR": 0.001,  # Actor learning rate
        "LR_CRITIC": 0.001,  # Critic learning rate
        "GAMMA": 0.95,  # Discount factor
        "MEMORY_SIZE": 1000000,  # Max memory buffer size
        "LEARN_STEP": 100,  # Learning frequency
        "TAU": 0.01,  # For soft update of target parameters
        "POLICY_FREQ": 2,  # Policy frequnecy
        "POP_SIZE": 4,  # Population size
    }
    return INIT_HP

def get_mutation_config():
    # Mutation config for RL hyperparameters
    hp_config = HyperparameterConfig(
        lr_actor = RLParameter(min=1e-4, max=1e-2),
        lr_critic = RLParameter(min=1e-4, max=1e-2),
        batch_size = RLParameter(min=8, max=512, dtype=int),
        learn_step = RLParameter(
            min=20, max=200, dtype=int, grow_factor=1.5, shrink_factor=0.75
            )
    )
    return hp_config

def main(args):
    NET_CONFIG = get_network_config()
    INIT_HP = get_initial_hp(args.algo_name)
    hp_config = get_mutation_config()

    # Create log directory: logs/{env}/{timestamp}
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logdir = os.path.join(os.getcwd(), "logs", f"{args.env_id} {args.algo_name}", timestamp)
    os.makedirs(logdir, exist_ok=True)

    num_envs = 8
    # Dynamically import the environment from pettingzoo.mpe
    try:
        env_func = getattr(mpe, args.env_id)
    except AttributeError:
        raise ValueError(f"Environment {args.env_id} not found in pettingzoo.mpe")
    env = env_func.parallel_env(continuous_actions=False)
    env = AsyncPettingZooVecEnv([lambda: env for _ in range(num_envs)])
    env.reset(seed=25)

    # Configure the multi-agent algo input arguments
    observation_spaces = [env.single_observation_space(agent) for agent in env.agents]
    action_spaces = [env.single_action_space(agent) for agent in env.agents]
    if INIT_HP["CHANNELS_LAST"]:   # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        observation_spaces = [obs_channels_to_first(obs) for obs in observation_spaces]

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    INIT_HP["AGENT_IDS"] = env.agents

    # Create a population ready for evolutionary hyper-parameter optimisation
    agent_pop = create_population(
        INIT_HP["ALGO"],
        observation_spaces,
        action_spaces,
        NET_CONFIG,
        INIT_HP,
        hp_config,
        population_size=INIT_HP["POP_SIZE"],
        num_envs=num_envs,
        device=device,
    )

    n_agents = env.num_agents
    # Replay Buffer
    field_names = ["state", "action", "reward", "next_state", "done"]
    memory = MultiAgentReplayBuffer(
        INIT_HP["MEMORY_SIZE"],
        field_names=field_names,
        agent_ids=INIT_HP["AGENT_IDS"],
        device=device,
    )

    # Instantiate a tournament selection object (used for HPO)
    tournament = TournamentSelection(
        tournament_size=2,  # Tournament selection size
        elitism=True,  # Elitism in tournament selection
        population_size=INIT_HP["POP_SIZE"],  # Population size
        eval_loop=1,  # Evaluate using last N fitness scores
    )

    # Instantiate a mutations object (used for HPO)
    mutations = Mutations(
        no_mutation=0.2,  # Probability of no mutation
        architecture=0.2,  # Probability of architecture mutation
        new_layer_prob=0.2,  # Probability of new layer mutation
        parameters=0.2,  # Probability of parameter mutation
        activation=0,  # Probability of activation function mutation
        rl_hp=0.2,  # Probability of RL hyperparameter mutation
        mutation_sd=0.1,  # Mutation strength
        rand_seed=1,
        device=device,
    )

    trained_population, population_fitnesses = train_multi_agent(
        env=env,  # Pettingzoo-style environment
        env_name=args.env_id,  # Environment name
        algo=INIT_HP["ALGO"],  # Algorithm
        pop=agent_pop,  # Population of agents
        memory=memory,  # Replay buffer
        INIT_HP=INIT_HP,  # IINIT_HP dictionary
        swap_channels=INIT_HP['CHANNELS_LAST'],  # Swap image channel from last to first
        max_steps=args.train_steps,  # Max number of training steps
        evo_steps=10000,  # Evolution frequency
        eval_steps=None,  # Number of steps in evaluation episode
        eval_loop=1,  # Number of evaluation episodes
        learning_delay=1000,  # Steps before starting learning
        target=200.,  # Target score for early stopping
        tournament=tournament,  # Tournament selection object
        mutation=mutations,  # Mutations object
        wb=False,  # Weights and Biases tracking,
        checkpoint=1000,  # Checkpoint frequency
        checkpoint_path=os.path.join(logdir, "checkpoints"),  # Checkpoint path,
        overwrite_checkpoints=True,  # Overwrite checkpoints
        save_elite=True,  # Save elite agent
        elite_path=os.path.join(logdir, "elite"),  # Elite agent path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an RL agent on a PettingZoo MPE environment")
    parser.add_argument("--env_id", type=str,
                        help="Name of the environment from pettingzoo.mpe (for ex: simple_speaker_listener_v4)")
    parser.add_argument("--algo_name", type=str, default='MADDPG',
                        help="Algorithm Name")
    parser.add_argument("--train_steps", type=int, default=10_000_000,
                        help="Total Train Steps")
    args = parser.parse_args()
    main(args)