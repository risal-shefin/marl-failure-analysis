import gymnasium as gym
import os
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import weakref

from stable_baselines3.dqn.ddqn import DoubleDQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import plot_results
from experiments.callbacks.best_model_callback import SaveOnBestTrainingRewardCallback

def main(args):
    # Create log dir
    cur_dir = os.getcwd()
    log_dir = os.path.join(cur_dir, "logs", args.env_id, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    env = gym.make(args.env_id, render_mode="rgb_array")
    env = Monitor(env, log_dir)

    double_dqn_model = DoubleDQN(
        "CnnPolicy",
        env,
        verbose=1,
        buffer_size=args.buffer_size,
        policy_kwargs=dict(net_arch=[256, 256, 256]),
        seed=42,
    )

    # Create the callback: check every 1000 steps
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir, model_loader_fn=weakref.ref(double_dqn_model))

    # Train the agent and display a progress bar
    double_dqn_model.learn(total_timesteps=args.train_timesteps, progress_bar=True, callback=callback, tb_log_name="DDQN")
    
    # Save the last agent
    double_dqn_model.save(os.path.join(log_dir, "last_model"))

    # plot train rewards
    plot_results([log_dir], args.train_timesteps, results_plotter.X_TIMESTEPS, f"DDQN {args.env_id}")
    plt.savefig(os.path.join(log_dir, f"DDQN_{args.env_id}_train_rewards.png"), dpi=300, format='png',bbox_inches='tight')

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(double_dqn_model, double_dqn_model.get_env(), n_eval_episodes=10)
    print(f"Last Model's Evaluation: mean reward={mean_reward}, std reward={std_reward}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDQN Training Arguments")

    parser.add_argument("--env_id", type=str, required=True,
                        help="Name of the Gymnasium environment (e.g., ALE/Boxing-v5)")
    
    parser.add_argument("--train_timesteps", type=int, default=10_000_000,
                        help="Total number of training timesteps (default: 10M)")
    
    parser.add_argument("--buffer_size", type=int, default=1_000_000,
                        help="Replay buffer size (default: 1M)")

    args = parser.parse_args()
    main(args)
