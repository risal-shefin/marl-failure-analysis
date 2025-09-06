import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from mappo import MAPPO
from make_env_pettingzoo import make_env
from gym.spaces import Box, Discrete
from datetime import datetime


class Runner_MAPPO_MPE:
    def __init__(self, args, env_name, number, seed):
        self.args = args
        self.env_name = env_name
        self.number = number
        self.seed = seed
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        # Create env
        self.env = make_env(env_name, discrete=True)  # Discrete action space
        self.args.N = self.env.n  # The number of agents
        
        # Handle observation dimensions for each agent
        self.args.obs_dim_n = []
        for obs_space in self.env.observation_space:
            # Properly handle Box observation spaces from gymnasium
            if isinstance(obs_space, Box) or str(type(obs_space)) == "<class 'gymnasium.spaces.box.Box'>":
                self.args.obs_dim_n.append(obs_space.shape[0])
            else:
                print(f"Unexpected observation space type: {type(obs_space)}", flush=True)
                self.args.obs_dim_n.append(obs_space.shape[0])  # Try to use shape attribute anyway
        
        # Handle action dimensions for each agent
        self.args.action_dim_n = []
        for act_space in self.env.action_space:
            # Handle both gym and gymnasium Discrete spaces
            if isinstance(act_space, Discrete) or str(type(act_space)) == "<class 'gymnasium.spaces.discrete.Discrete'>":
                self.args.action_dim_n.append(act_space.n)
            else:
                print(f"Unexpected action space type: {type(act_space)}", flush=True)
                if hasattr(act_space, 'n'):
                    self.args.action_dim_n.append(act_space.n)
                else:
                    self.args.action_dim_n.append(act_space.shape[0])  # For continuous actions
        
        # Only for homogenous agents environments like Spread in MPE, all agents have the same dimension of observation space and action space
        self.args.obs_dim = self.args.obs_dim_n[0]  # The dimensions of an agent's observation space
        self.args.action_dim = self.args.action_dim_n[0]  # The dimensions of an agent's action space
        self.args.state_dim = np.sum(self.args.obs_dim_n)  # The dimensions of global state space (Sum of the dimensions of the local observation space of all agents)
        
        print("observation_space=", self.env.observation_space, flush=True)
        print("obs_dim_n={}".format(self.args.obs_dim_n), flush=True)
        print("action_space=", self.env.action_space, flush=True)
        print("action_dim_n={}".format(self.args.action_dim_n), flush=True)

        # Setup output directory structure
        self.output_dir = os.path.join(args.output_dir, f"train_{env_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
        self.data_dir = os.path.join(self.output_dir, 'data')
        self.model_dir = os.path.join(self.output_dir, 'models')
        self.tensorboard_dir = os.path.join(self.output_dir, 'tensorboard')
        
        # Create directories if they don't exist
        for directory in [self.output_dir, self.data_dir, self.model_dir, self.tensorboard_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}", flush=True)

        # Create N agents
        self.agent_n = MAPPO(self.args)
        self.replay_buffer = ReplayBuffer(self.args)

        # Create a tensorboard with the new path
        self.writer = SummaryWriter(log_dir=os.path.join(
            self.tensorboard_dir, 'MAPPO_env_{}_number_{}_seed_{}'.format(self.env_name, self.number, self.seed)))

        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0
        self.best_eval_reward = float('-inf')  # Track the best evaluation reward
        
        if self.args.use_reward_norm:
            print("------use reward norm------", flush=True)
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------", flush=True)
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)

    def run(self, ):
        evaluate_num = -1  # Record the number of evaluations
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()  # Evaluate the policy every 'evaluate_freq' steps
                evaluate_num += 1

            _, episode_steps = self.run_episode_mpe(evaluate=False)  # Run an episode
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)  # Training
                self.replay_buffer.reset_buffer()

        self.evaluate_policy()
        self.env.close()

    def evaluate_policy(self):
        evaluate_reward = 0
        for _ in range(self.args.evaluate_times):
            episode_reward, _ = self.run_episode_mpe(evaluate=True)
            evaluate_reward += episode_reward

        evaluate_reward = evaluate_reward / self.args.evaluate_times
        self.evaluate_rewards.append(evaluate_reward)
        print("total_steps:{} \t evaluate_reward:{}".format(self.total_steps, evaluate_reward), flush=True)
        self.writer.add_scalar('evaluate_step_rewards_{}'.format(self.env_name), evaluate_reward, global_step=self.total_steps)
        
        # Save the rewards data to the specified directory
        rewards_file = os.path.join(self.data_dir, 'MAPPO_env_{}_number_{}_seed_{}.npy'.format(
            self.env_name, self.number, self.seed))
        try:
            np.save(rewards_file, np.array(self.evaluate_rewards))
        except Exception as e:
            print(f"Error saving rewards data: {e}", flush=True)
            
        # Only save model if the current reward is better than the best so far
        if evaluate_reward > self.best_eval_reward:
            old_best = self.best_eval_reward
            self.best_eval_reward = evaluate_reward
            improvement = self.best_eval_reward - old_best
            
            print(f"New best reward: {self.best_eval_reward:.2f} (improved by {improvement:.2f})! Saving model...", flush=True)
            
            try:
                self.save_best_model(evaluate_reward)
            except Exception as e:
                print(f"Error saving model: {e}", flush=True)

    def save_best_model(self, score):
        """Save the model with the best performance."""
        model_path = os.path.join(self.model_dir, 
                                  f'MAPPO_seed_{self.seed}_score_{score:.2f}.pt')
        
        model_data = {
            'actor_state_dict': self.agent_n.actor.state_dict(),
            'critic_state_dict': self.agent_n.critic.state_dict(),
            'steps': self.total_steps,
            'reward': self.best_eval_reward
        }
        
        torch.save(model_data, model_path)
        print(f"Best model successfully saved at step {self.total_steps} at {model_path}", flush=True)

    def run_episode_mpe(self, evaluate=False):
        total_reward = 0
        reset_result = self.env.reset()  # Reset the environment
        if isinstance(reset_result, tuple):
            obs_n, action_masks = reset_result
        else:
            obs_n = reset_result
            action_masks = [np.ones(self.args.action_dim) for _ in range(self.args.N)]
        episode_steps = 0
        
        while True:
            # Get actions for all agents using the policy
            actions = []
            action_logprobs = []
            
            # Collect state for critic
            s = np.concatenate(obs_n)  # This assumes global state is concatenation of observations
            
            # Get values for the current state
            v = self.agent_n.get_value(s)
            
            for agent_id in range(self.args.N):
                obs = obs_n[agent_id]
                mask = action_masks[agent_id]
                if evaluate:
                    action = self.agent_n.select_action(obs, agent_id, evaluate=True, action_mask=mask)
                    actions.append(action)
                else:
                    # When training, we need both actions and their log probabilities
                    action, action_logprob = self.agent_n.choose_action([obs], False, [mask])
                    actions.append(action[0])  # choose_action returns a numpy array
                    action_logprobs.append(action_logprob[0])  # choose_action returns a numpy array

            # Take a step in the environment
            step_result = self.env.step(actions)
            if isinstance(step_result, tuple) and len(step_result) == 5:
                next_obs_n, reward_n, done_n, info_n, next_masks = step_result
            else:
                next_obs_n, reward_n, done_n, info_n = step_result
                next_masks = [np.ones(self.args.action_dim) for _ in range(self.args.N)]

            # Store transitions in the replay buffer if not evaluating
            if not evaluate:
                self.replay_buffer.store_transition(
                    episode_step=episode_steps,
                    obs_n=obs_n,
                    s=s,
                    v_n=v,
                    a_n=np.array(actions),
                    a_logprob_n=np.array(action_logprobs),
                    r_n=np.array(reward_n),
                    done_n=np.array(done_n),
                    action_mask_n=np.array(action_masks)
                )
            
            # Calculate the total reward
            reward = sum(reward_n)
            total_reward += reward
            
            # Update the observations
            obs_n = next_obs_n
            action_masks = next_masks
            
            episode_steps += 1
            done = all(done_n) or episode_steps >= self.args.episode_limit
            
            # End the episode if done
            if done:
                # If not evaluating, store the last value
                if not evaluate:
                    # Get the value of the last state, or zero if terminal
                    if all(done_n):  # If truly done (not just hitting episode limit)
                        v_last = np.zeros_like(v)
                    else:
                        # Calculate the next state for the critic and get its value
                        next_s = np.concatenate(next_obs_n)
                        v_last = self.agent_n.get_value(next_s)
                    
                    # Store the last value and increment episode counter
                    self.replay_buffer.store_last_value(episode_steps, v_last)
                
                break
        
        return total_reward, episode_steps


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in MPE environment")
    parser.add_argument("--max_train_steps", type=int, default=int(3e6), help="Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=25, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=int, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=int, default=3, help="Evaluate times")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the rnn")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the mlp")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling. Here, we do not use it.")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=float, default=False, help="Whether to use relu, if False, we will use tanh")
    parser.add_argument("--use_rnn", type=bool, default=False, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=float, default=False, help="Whether to add agent_id. Here, we do not use it.")
    parser.add_argument("--use_value_clip", type=float, default=False, help="Whether to use value clip.")
    parser.add_argument("--env_id", type=str, required=True, help="The name of the environment to run")
    
    # Add output directory argument
    parser.add_argument("--output_dir", type=str, default="./runs", help="Directory to save all output files")

    args = parser.parse_args()
    runner = Runner_MAPPO_MPE(args, env_name=args.env_id, number=1, seed=42)
    runner.run()
