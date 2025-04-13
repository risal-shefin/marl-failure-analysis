import argparse
import torch
import time
import os
import numpy as np
import csv
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
# Use PettingZoo environments if specified
from utils.make_env_pettingzoo import make_env as make_env_pettingzoo
# Also keep the original make_env for backward compatibility
from utils.make_env import make_env as make_env_original
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.maddpg import MADDPG
from tqdm import tqdm
USE_CUDA = False  # torch.cuda.is_available()

def make_parallel_env(env_id, n_rollout_threads, seed, discrete_action):
    def get_env_fn(rank):
        def init_env():
            # Check if this is a PettingZoo environment
            if any(env_name in env_id for env_name in ['simple_reference_v3', 'simple_speaker_listener_v4', 'simple_spread_v3']):
                env = make_env_pettingzoo(env_id, discrete_action=discrete_action)
            else:
                env = make_env_original(env_id, discrete_action=discrete_action)
            # env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])

def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        curr_run = 'run1'
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            curr_run = 'run1'
        else:
            curr_run = 'run%i' % (max(exst_run_nums) + 1)
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    # Setup CSV for reward tracking
    os.makedirs(f"./results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    csv_filename = f"rewards_{config.model_name}_{timestamp}.csv"
    csv_path = os.path.join(f"./results", csv_filename)
    
    # Check if this is a PettingZoo environment
    # is_pettingzoo = any(env_name in config.env_id for env_name in 
    #                    ['simple_reference_v3', 'simple_speaker_listener_v4', 'simple_spread_v3'])
    is_pettingzoo = True
    
    # print(f"Is PettingZoo environment: {is_pettingzoo}")
    
    # Initialize environment
    env = make_parallel_env(config.env_id, config.n_rollout_threads, config.seed,
                            config.discrete_action)
    
    # Print information about observation and action spaces
    # print("Observation and action spaces:")
    # for i, (obs_space, act_space) in enumerate(zip(env.observation_space, env.action_space)):
    #     # print(f"Agent {i} observation space: {obs_space}")
    #     # print(f"Agent {i} action space: {act_space}, type: {type(act_space)}")
        # if hasattr(act_space, 'n'):
        #     # print(f"  Discrete with {act_space.n} actions")
        # if hasattr(act_space, 'shape'):
            # print(f"  Continuous with shape {act_space.shape}")
    
    # Create a completely new maddpg implementation specifically for PettingZoo
    if is_pettingzoo:
        # For PettingZoo, use a custom MADDPG initialization with no batch normalization
        # print("Using custom MADDPG for PettingZoo (no batch normalization)")
        agent_init_params = []
        alg_types = []
        
        for i, (acsp, obsp) in enumerate(zip(env.action_space, env.observation_space)):
            # Handle different action space types
            if isinstance(acsp, Discrete):
                # Discrete action space
                act_dim = acsp.n
                discrete_action = True
                # print(f"Agent {i}: Discrete action space with {act_dim} actions")
            else:
                # Continuous action space or other
                # Check if it has shape attribute with non-zero length
                if hasattr(acsp, 'shape') and len(acsp.shape) > 0:
                    act_dim = acsp.shape[0]
                elif hasattr(acsp, 'n'):
                    # Some spaces have 'n' attribute but are not instances of Discrete
                    act_dim = acsp.n
                else:
                    # Fallback: Assume scalar continuous action
                    act_dim = 1
                discrete_action = False
                # print(f"Agent {i}: Continuous action space with dimension {act_dim}")
            
            # Get the actual observation dimension for this agent
            obs_dim = obsp.shape[0]  # This should be 21 for your environment
            
            # print(f"Agent {i}: obs_dim={obs_dim}, act_dim={act_dim}")
            
            agent_init_params.append({
                'num_in_pol': obs_dim,  # Use the individual agent's observation dimension
                'num_out_pol': act_dim,
                'num_in_critic': sum(obs_space.shape[0] for obs_space in env.observation_space),
                'norm_in': False  # Explicitly disable batch normalization
            })
            
            # Use MADDPG for all agents (simplified)
            alg_types.append(config.agent_alg)
        
        # Default values for parameters that might not be in config
        gamma = getattr(config, 'gamma', 0.95)  # Default gamma value
        tau = getattr(config, 'tau', 0.01)      # Default tau value
        lr = getattr(config, 'lr', 0.01)        # Default learning rate
        hidden_dim = getattr(config, 'hidden_dim', 64)  # Default hidden dimension
        
        init_dict = {
            'gamma': gamma,
            'tau': tau,
            'lr': lr,
            'hidden_dim': hidden_dim,
            'alg_types': alg_types,
            'agent_init_params': agent_init_params,
            'discrete_action': discrete_action
        }
        
        maddpg = MADDPG(**init_dict)
    else:
        # For original environments, use the standard initialization
        maddpg = MADDPG.init_from_env(env, agent_alg=config.agent_alg,
                                     adversary_alg=config.adversary_alg,
                                     tau=config.tau,
                                     lr=config.lr,
                                     hidden_dim=config.hidden_dim)
    
    # Initialize CSV with headers including each agent
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        headers = ['Episode', 'Step', 'Total_Reward', 'Mean_Reward_Per_Agent']
        # Add headers for individual agent rewards
        for i in range(maddpg.nagents):
            headers.append(f'Agent_{i}_Reward')
        writer.writerow(headers)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)
    replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if hasattr(acsp, 'shape') and len(acsp.shape) > 0 
                                  else acsp.n if hasattr(acsp, 'n')
                                  else 1  # Default fallback
                                  for acsp in env.action_space])
    t = 0
    
    # For storing episode statistics
    episode_stats = []
    best_mean_reward = -np.inf
    for ep_i in tqdm(range(0, config.n_episodes, config.n_rollout_threads)):
        # print("Episodes %i-%i of %i" % (ep_i + 1,
        #                                 ep_i + 1 + config.n_rollout_threads,
        #                                 config.n_episodes))
        obs = env.reset()
        # print(f"Reset observations type: {type(obs)}, shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}")
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        maddpg.prep_rollouts(device='cpu')

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()

        # Track episode rewards for each agent
        episode_rewards = [0] * maddpg.nagents
        for et_i in range(config.episode_length):
            # Debug print for observations
            # print(f"Step {et_i} observations shape: {obs.shape if hasattr(obs, 'shape') else 'N/A'}")
            
            # rearrange observations to be per agent, and convert to torch Variable
            torch_obs = []
            
            # Special handling for PettingZoo observations
            if is_pettingzoo and isinstance(obs, np.ndarray) and obs.ndim == 4:
                # Shape is (n_envs, n_batches, n_agents, obs_dim)
                # print(f"Processing PettingZoo observations with shape: {obs.shape}")
                # Extract each agent's observation separately
                for i in range(maddpg.nagents):
                    if i < obs.shape[2]:  # Make sure agent index is within bounds
                        agent_obs = obs[0, 0, i]  # Take the observation for this agent
                        # print(f"Agent {i} observation shape: {agent_obs.shape}")
                        # Add batch dimension if needed
                        if len(agent_obs.shape) == 1:
                            agent_obs = agent_obs.reshape(1, -1)
                        torch_obs.append(torch.Tensor(agent_obs))
                    else:
                        # print(f"Warning: No observation for agent {i}, using zeros")
                        # Create a dummy observation with correct dimension
                        dummy_obs = torch.zeros(1, maddpg.agents[i].policy.fc1.in_features)
                        torch_obs.append(dummy_obs)
            else:
                # Original observation handling
                for i in range(maddpg.nagents):
                    try:
                        if isinstance(obs, np.ndarray):
                            if obs.ndim == 3:  # Format [n_threads, n_agents, obs_dim]
                                agent_obs = obs[:, i]
                            else:
                                # Try other formats
                                agent_obs = obs[i].reshape(1, -1)  # Ensure 2D
                        elif isinstance(obs, list):
                            agent_obs = np.array([row[i] for row in obs])
                        else:
                            agent_obs = np.vstack(obs[:, i])
                        
                        # Ensure agent_obs is the right shape for network input
                        if len(agent_obs.shape) == 1:
                            agent_obs = agent_obs.reshape(1, -1)
                            
                        torch_obs.append(torch.Tensor(agent_obs))
                    except Exception as e:
                        print(f"Error processing obs for agent {i}: {e}")
                        # Create a dummy observation with correct dimension
                        dummy_obs = torch.zeros(1, maddpg.agents[i].policy.fc1.in_features)
                        torch_obs.append(dummy_obs)
            
            # Get actions
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # Convert actions to numpy arrays
            actions = []
            for i, action in enumerate(torch_agent_actions):
                # For discrete actions, convert to the expected format
                if maddpg.discrete_action:
                    # Get the action with highest probability
                    if action.dim() > 1:
                        action = action.max(dim=1)[1]
                    action = action.data.cpu().numpy()
                    # print(f"Agent {i} discrete action: {action}")
                else:
                    # For continuous actions
                    action = action.data.cpu().numpy()
                    # print(f"Agent {i} continuous action: {action}")
                actions.append(action)
            
            actions_array = [np.array([ac]) for ac in actions]

            # Execute environment step
            next_obs, rewards, dones, infos = env.step(actions_array)
            
            # Print debug info about rewards
            # print(f"Rewards shape: {rewards.shape if hasattr(rewards, 'shape') else 'N/A'}, content: {rewards}")
            
            # Handle rewards based on environment type
            if is_pettingzoo:
                # For PettingZoo environments, rewards may be formatted differently
                for i in range(maddpg.nagents):
                    if isinstance(rewards, np.ndarray):
                        if rewards.ndim == 1:
                            # If rewards is a 1D array [n_agents]
                            if i < rewards.size:
                                episode_rewards[i] += rewards[i]
                        elif rewards.ndim == 2:
                            # If rewards is a 2D array [n_batches, n_agents]
                            if i < rewards.shape[1]:
                                episode_rewards[i] += rewards[0, i]
                        elif rewards.ndim == 3:
                            # If rewards is a 3D array [n_envs, n_batches, n_agents]
                            if i < rewards.shape[2]:
                                episode_rewards[i] += rewards[0, 0, i]
                    else:
                        # If rewards is a list or other type, try to access by index
                        try:
                            episode_rewards[i] += rewards[i]
                        except (IndexError, TypeError):
                            print(f"Warning: Could not update reward for agent {i}")
            else:
                # For original environments
                for i in range(maddpg.nagents):
                    episode_rewards[i] += rewards[0, i]  # Original indexing
            
            replay_buffer.push(obs.copy(), actions, rewards, next_obs.copy(), dones)
            obs = next_obs
            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if USE_CUDA:
                    maddpg.prep_training(device='gpu')
                else:
                    maddpg.prep_training(device='cpu')
                for u_i in range(config.n_rollout_threads):
                    for a_i in range(maddpg.nagents):
                        sample = replay_buffer.sample(config.batch_size,
                                                      to_gpu=USE_CUDA)
                        maddpg.update(sample, a_i, logger=logger)
                    maddpg.update_all_targets()
                maddpg.prep_rollouts(device='cpu')
                
        # End of episode - log rewards
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)
        
        # Calculate and log total reward and mean reward
        total_reward = sum(episode_rewards)
        mean_reward = total_reward / maddpg.nagents
        
        # Store stats
        episode_stats.append({
            'episode': ep_i + 1,
            'total_reward': total_reward,
            'mean_reward': mean_reward,
            'agent_rewards': episode_rewards.copy()
        })
        
        # Log to CSV with individual agent rewards
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            row = [ep_i + 1, t, total_reward, mean_reward]
            # Add individual agent rewards to the row
            row.extend(episode_rewards)
            writer.writerow(row)
            
        # Print reward information
        print(f"Episode {ep_i+1} - Total reward: {total_reward:.4f}, Mean reward: {mean_reward:.4f}")
        # for a_i, a_reward in enumerate(episode_rewards):
        #     print(f"  Agent {a_i}: {a_reward:.4f}")

        # if ep_i % config.save_interval < config.n_rollout_threads:
            # os.makedirs(run_dir / 'incremental', exist_ok=True)
            # maddpg.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
        if mean_reward >= best_mean_reward:
            print(f"New best mean reward: {mean_reward:.4f} (previous: {best_mean_reward:.4f}) Episode {ep_i + 1}")
            best_mean_reward = mean_reward
            maddpg.save(run_dir / 'model.pt')

    # maddpg.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()
    
    # Print final training statistics
    # print("\n" + "="*50)
    # print(f"Training completed for {config.env_id}")
    # print(f"Total episodes: {len(episode_stats)}")
    
    # Calculate average reward across all episodes
    total_mean = sum(stat['mean_reward'] for stat in episode_stats) / len(episode_stats)
    print(f"Average reward per episode: {total_mean:.4f}")
    print(f"Rewards saved to: {csv_path}")
    print("="*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", help="Name of environment")
    parser.add_argument("--model_name",
                        default='maddpg',
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--seed",
                        default=42, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=int(1e7), type=int)
    parser.add_argument("--episode_length", default=25, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=int(1e5), type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--hidden_dim", default=64, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action",
                        action='store_true')

    config = parser.parse_args()
    print("#"*50)
    print(f"Model name: {config.model_name}")
    print(f"Environment ID: {config.env_id}")
    print(f"Agent algorithm: {config.agent_alg}")
    print(f"Discrete action: {config.discrete_action}")
    print(f"Number of episodes to train: {config.n_episodes}")
    print(f"Episode length: {config.episode_length}")
    print("#"*50)

    run(config)
