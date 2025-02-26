import gymnasium as gym
import os
from datetime import datetime
import argparse
import matplotlib.pyplot as plt

from stable_baselines3.dqn.ddqn import DoubleDQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import plot_results
from experiments.callbacks.best_model_callback import SaveOnBestTrainingRewardCallback

def main(args):
    env = gym.make(args.env_id, render_mode="rgb_array")
    # Load the agent
    double_dqn_model = DoubleDQN.load(args.model_dir, env=env)
    
    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(double_dqn_model, double_dqn_model.get_env(), n_eval_episodes=10)
    print(f"Test Model's Evaluation: mean reward={mean_reward}, std reward={std_reward}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDQN Testing Arguments")

    parser.add_argument("--env_id", type=str, required=True,
                        help="Name of the Gymnasium environment (e.g., ALE/Boxing-v5)")
    
    parser.add_argument("--model_dir", type=str, required=True,
                        help="model.zip directory")


    args = parser.parse_args()
    main(args)
