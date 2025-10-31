import argparse
import torch
import time
import os
import numpy as np
from gym.spaces import Box, Discrete
import gymnasium
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.buffer import ReplayBuffer
from utils.gym_multigrid_wrapper import GymMultiGridWrapper
from algorithms.maddpg import MADDPG

USE_CUDA = torch.cuda.is_available()
DEVICE = 'gpu' if USE_CUDA else 'cpu'

def eval(env, is_discrete_action, maddpg, n_episodes, use_terminal_reward_only=False):
    total_reward = 0
    success_rate = 0

    for ep_i in range(n_episodes):
        obs = env.reset()
        episode_reward = 0
        with torch.no_grad():
            maddpg.prep_rollouts(device=DEVICE)

        while True:
            torch_obs = [Variable(torch.Tensor([obs[i]]).to('cuda' if USE_CUDA else 'cpu'), requires_grad=False) for i in range(maddpg.nagents)]
            with torch.no_grad():
                torch_agent_actions = maddpg.step(torch_obs, explore=False)
            actions = [ac.data.cpu().numpy().argmax() for ac in torch_agent_actions]
            next_obs, rewards, dones, infos = env.step(actions)
            agent_rewards = np.array(rewards).squeeze()
            if not use_terminal_reward_only:
                episode_reward += np.sum(agent_rewards)
            obs = next_obs
            if dones.all():
                if use_terminal_reward_only:
                    episode_reward = np.sum(agent_rewards)  # only terminal reward
                if infos[0][0]['success']:
                    success_rate += 1
                break
        total_reward += episode_reward
    
    success_rate /= n_episodes
    avg_reward = total_reward / n_episodes
    return avg_reward, success_rate


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

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if not USE_CUDA:
        torch.set_num_threads(config.n_training_threads)

    env = GymMultiGridWrapper.make_and_wrap_env(config.env_id, True)
    eval_env = GymMultiGridWrapper.make_and_wrap_env(config.env_id, True)
    env.reset()

    maddpg = MADDPG.init_from_env(env, agent_alg=config.agent_alg,
                                  adversary_alg=config.adversary_alg,
                                  tau=config.tau,
                                  lr=config.lr,
                                  hidden_dim=config.hidden_dim,
                                  local_q=config.local_q)
    replay_buffer = ReplayBuffer(config.buffer_length, maddpg.nagents,
                                 [obsp.shape[0] if len(obsp.shape) == 1 else obsp.shape for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) or isinstance(acsp, gymnasium.spaces.Box) else acsp.n
                                  for acsp in env.action_space])
    t = 0
    best_eval_reward = -np.inf
    best_eval_success_rate = 0.0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        obs = env.reset()
        # obs.shape = (n_rollout_threads, nagent)(nobs), nobs differs per agent so not tensor
        maddpg.prep_rollouts(device=DEVICE)

        explr_pct_remaining = max(0, config.n_exploration_eps - ep_i) / config.n_exploration_eps
        maddpg.scale_noise(config.final_noise_scale + (config.init_noise_scale - config.final_noise_scale) * explr_pct_remaining)
        maddpg.reset_noise()

        episode_length = 0
        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable
            # torch_obs = [Variable(torch.Tensor(np.vstack(obs[:, i])),
            #                       requires_grad=False)
            #              for i in range(maddpg.nagents)]
            torch_obs = [Variable(torch.Tensor([obs[i]]).to('cuda' if USE_CUDA else 'cpu'), requires_grad=False) for i in range(maddpg.nagents)]
            # get actions as torch Variables
            torch_agent_actions = maddpg.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            actions = [ac.data.cpu().numpy().argmax() for ac in torch_agent_actions]
            next_obs, rewards, dones, infos = env.step(actions)
            replay_buffer.push(obs, actions, rewards, next_obs, dones)
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
                maddpg.prep_rollouts(device=DEVICE)
            
            episode_length += 1
            if dones.all():
                break

        ep_rews = replay_buffer.get_average_rewards(
            episode_length * config.n_rollout_threads)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i, a_ep_rew, ep_i)

        print(f"Ep#{ep_i+1},rew:{np.mean(ep_rews):.3f}", flush=True)

        if ep_i % config.save_interval < config.n_rollout_threads:
            eval_reward, eval_success_rate = eval(eval_env, config.discrete_action, maddpg, n_episodes=10)
            if eval_success_rate > best_eval_success_rate or (eval_success_rate == best_eval_success_rate and eval_reward >= best_eval_reward):
                maddpg.save(run_dir / f'model_sr{eval_success_rate}_r{eval_reward}.pt')
                best_eval_reward = eval_reward
                best_eval_success_rate = eval_success_rate
                print(f"Model saved with success rate: {eval_success_rate:.3f}, eval reward: {eval_reward:.3f}", flush=True)

    maddpg.save(run_dir / 'model_end.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--seed",
                        default=1, type=int,
                        help="Random seed")
    parser.add_argument("--n_rollout_threads", default=1, type=int)
    parser.add_argument("--n_training_threads", default=6, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=10000000, type=int)
    parser.add_argument("--episode_length", default=100, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for model training")
    parser.add_argument("--n_exploration_eps", default=100000, type=int)
    parser.add_argument("--init_noise_scale", default=0.3, type=float)
    parser.add_argument("--final_noise_scale", default=0.0, type=float)
    parser.add_argument("--save_interval", default=100, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--agent_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--adversary_alg",
                        default="MADDPG", type=str,
                        choices=['MADDPG', 'DDPG'])
    parser.add_argument("--discrete_action",
                        default=False,
                        help="Use discrete action space")
    parser.add_argument("--local_q", action='store_true',
                        help="Train additional decentralized Q functions")

    config = parser.parse_args()
    print("\n-- Configs: --")
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("--------------")
    run(config)
