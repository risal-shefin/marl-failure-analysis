#!/usr/bin/env python3
"""
SMAC Video Rendering Implementation
Supports both SC2 replay files and direct frame capture for video creation
"""

import os
import sys
import json
import numpy as np
import torch
from PIL import Image
import imageio
import cv2
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harl.utils.configs_tools import get_defaults_yaml_args, update_args
from harl.utils.trans_tools import _t2n
from harl.runners import RUNNER_REGISTRY

def save_replay_method(runner, num_episodes=5, replay_dir="replays"):
    """
    Method 1: Use StarCraft II's built-in replay system
    This creates .SC2Replay files that can be played in StarCraft II
    """
    os.makedirs(replay_dir, exist_ok=True)
    
    # Enable replay saving in the SMAC environment
    base_env = None
    if hasattr(runner.eval_envs, 'env'):
        base_env = runner.eval_envs.env
    elif hasattr(runner.eval_envs, 'envs') and len(runner.eval_envs.envs) > 0:
        base_env = runner.eval_envs.envs[0]
    else:
        base_env = runner.eval_envs
    
    # Navigate to the SMAC environment through potential wrapper layers
    smac_env = base_env
    while hasattr(smac_env, 'env') and hasattr(smac_env, '__class__'):
        if 'StarCraft2Env' in str(smac_env.__class__):
            break
        smac_env = smac_env.env
    
    print(f"Found environment class: {smac_env.__class__}")
    
    # Set replay directory and prefix
    if hasattr(smac_env, 'replay_dir'):
        smac_env.replay_dir = replay_dir
        print(f"Set replay directory: {replay_dir}")
    if hasattr(smac_env, 'replay_prefix'):
        smac_env.replay_prefix = f"harl_gameplay_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Set replay prefix: {smac_env.replay_prefix}")
    
    replay_paths = []
    
    for episode in range(num_episodes):
        print(f"Recording episode {episode + 1}/{num_episodes}")
        
        # Reset environment
        eval_obs, eval_share_obs, eval_available_actions = runner.eval_envs.reset()
        
        # Initialize RNN states
        eval_rnn_states = np.zeros(
            (runner.algo_args["eval"]["n_eval_rollout_threads"], runner.num_agents,
             runner.recurrent_n, runner.rnn_hidden_size), dtype=np.float32
        )
        eval_masks = np.ones(
            (runner.algo_args["eval"]["n_eval_rollout_threads"], runner.num_agents, 1),
            dtype=np.float32
        )
        
        step_count = 0
        while True:
            # Get actions from trained agents
            eval_actions_collector = []
            for agent_id in range(runner.num_agents):
                eval_actions, temp_rnn_state = runner.actor[agent_id].act(
                    eval_obs[:, agent_id],
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    eval_available_actions[:, agent_id] if eval_available_actions[0] is not None else None,
                    deterministic=True,
                )
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))
            
            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
            
            # Step environment
            (eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, 
             eval_available_actions) = runner.eval_envs.step(eval_actions)
            
            step_count += 1
            
            # Update masks
            eval_dones_env = np.all(eval_dones, axis=1)
            eval_masks = np.ones(
                (runner.algo_args["eval"]["n_eval_rollout_threads"], runner.num_agents, 1),
                dtype=np.float32
            )
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), runner.num_agents, 1), dtype=np.float32
            )
            
            if eval_dones_env[0]:
                break
        
        print(f"Episode {episode + 1} completed in {step_count} steps")
        
        # Save replay if the environment supports it
        if hasattr(smac_env, 'save_replay'):
            replay_path = smac_env.save_replay()
            if replay_path:
                replay_paths.append(replay_path)
                print(f"Replay saved: {replay_path}")
            else:
                print(f"save_replay() returned None for episode {episode + 1}")
        else:
            print(f"Environment does not have save_replay method")
    
    return replay_paths

def capture_frames_method(runner, num_episodes=1, output_dir="videos", fps=10):
    """
    Method 2: Capture frames directly and create video files
    Works by trying multiple rendering approaches
    """
    os.makedirs(output_dir, exist_ok=True)
    
    video_paths = []
    
    for episode in range(num_episodes):
        print(f"Capturing frames for episode {episode + 1}/{num_episodes}")
        
        # Reset environment
        eval_obs, eval_share_obs, eval_available_actions = runner.eval_envs.reset()
        
        # Initialize RNN states
        eval_rnn_states = np.zeros(
            (runner.algo_args["eval"]["n_eval_rollout_threads"], runner.num_agents,
             runner.recurrent_n, runner.rnn_hidden_size), dtype=np.float32
        )
        eval_masks = np.ones(
            (runner.algo_args["eval"]["n_eval_rollout_threads"], runner.num_agents, 1),
            dtype=np.float32
        )
        
        frames = []
        step_count = 0
        
        while True:
            # Capture frame before taking action
            frame = capture_frame(runner.eval_envs)
            if frame is not None:
                frames.append(frame)
            
            # Get actions from trained agents
            eval_actions_collector = []
            for agent_id in range(runner.num_agents):
                eval_actions, temp_rnn_state = runner.actor[agent_id].act(
                    eval_obs[:, agent_id],
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    eval_available_actions[:, agent_id] if eval_available_actions[0] is not None else None,
                    deterministic=True,
                )
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))
            
            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
            
            # Step environment
            (eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, 
             eval_available_actions) = runner.eval_envs.step(eval_actions)
            
            step_count += 1
            
            # Update masks
            eval_dones_env = np.all(eval_dones, axis=1)
            eval_masks = np.ones(
                (runner.algo_args["eval"]["n_eval_rollout_threads"], runner.num_agents, 1),
                dtype=np.float32
            )
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), runner.num_agents, 1), dtype=np.float32
            )
            
            if eval_dones_env[0]:
                break
        
        # Save frames as video
        if frames:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save as MP4
            mp4_path = os.path.join(output_dir, f"gameplay_episode_{episode+1}_{timestamp}.mp4")
            save_frames_as_mp4(frames, mp4_path, fps=fps)
            video_paths.append(mp4_path)
            
            # Save as GIF
            gif_path = os.path.join(output_dir, f"gameplay_episode_{episode+1}_{timestamp}.gif")
            save_frames_as_gif(frames, gif_path, duration=int(1000/fps))
            video_paths.append(gif_path)
            
            print(f"Episode {episode + 1} saved: {len(frames)} frames")
        else:
            print(f"Warning: No frames captured for episode {episode + 1}")
    
    return video_paths

def capture_frame(env):
    """
    Try multiple methods to capture a frame from the environment
    """
    frame = None
    
    # Method 1: Direct rendering
    if hasattr(env, 'render'):
        if hasattr(env.render, '__call__'):
            frame = env.render(mode='rgb_array') if 'mode' in env.render.__code__.co_varnames else env.render()
    
    # Method 2: Access wrapped environment
    if frame is None and hasattr(env, 'env'):
        if hasattr(env.env, 'render'):
            if hasattr(env.env.render, '__call__'):
                frame = env.env.render(mode='rgb_array') if 'mode' in env.env.render.__code__.co_varnames else env.env.render()
    
    # Method 3: Navigate through wrapper layers
    if frame is None:
        base_env = env
        while hasattr(base_env, 'env') and base_env.env is not base_env:
            base_env = base_env.env
        
        if hasattr(base_env, 'render'):
            if hasattr(base_env.render, '__call__'):
                frame = base_env.render(mode='rgb_array') if 'mode' in base_env.render.__code__.co_varnames else base_env.render()
    
    # Method 4: For StarCraft II, try to get screen capture
    if frame is None:
        # Navigate to StarCraft2Env
        base_env = env
        while hasattr(base_env, 'env'):
            if 'StarCraft2Env' in str(base_env.__class__):
                break
            base_env = base_env.env
        
        # Try to capture from SC2 controller
        if hasattr(base_env, '_controller') and hasattr(base_env._controller, 'observe'):
            obs = base_env._controller.observe()
            if hasattr(obs, 'observation') and hasattr(obs.observation, 'render_data'):
                render_data = obs.observation.render_data
                if hasattr(render_data, 'map') and render_data.map:
                    # Convert SC2 map data to image
                    map_data = render_data.map
                    if hasattr(map_data, 'data'):
                        # This is a simplified example - actual implementation would need
                        # proper conversion from SC2 map format to RGB image
                        pass
    
    # Validate frame
    if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
        # Ensure frame is in correct format
        if len(frame.shape) == 4 and frame.shape[0] == 1:
            frame = frame.squeeze(0)  # Remove batch dimension
        
        return frame
    
    return None

def save_frames_as_mp4(frames, output_path, fps=10):
    """Save frames as MP4 video using OpenCV"""
    if not frames:
        print("No frames to save")
        return False
    
    # Get frame dimensions
    height, width = frames[0].shape[:2]
    
    # Define the codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(bgr_frame)
    
    out.release()
    print(f"Video saved: {output_path}")
    return True

def save_frames_as_gif(frames, output_path, duration=100):
    """Save frames as animated GIF"""
    if not frames:
        print("No frames to save")
        return False
    
    # Convert frames to PIL Images
    pil_frames = []
    for frame in frames:
        # Ensure frame is uint8
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        # Handle different shapes
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            pil_frames.append(Image.fromarray(frame))
        elif len(frame.shape) == 3 and frame.shape[2] == 4:
            # RGBA to RGB
            rgb_frame = frame[:, :, :3]
            pil_frames.append(Image.fromarray(rgb_frame))
        elif len(frame.shape) == 2:
            # Grayscale to RGB
            rgb_frame = np.stack([frame] * 3, axis=2)
            pil_frames.append(Image.fromarray(rgb_frame))
    
    if pil_frames:
        pil_frames[0].save(
            output_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0
        )
        print(f"GIF saved: {output_path}")
        return True
    
    return False

def setup_runner_for_rendering(algo="happo", env="smac", map_name="3s_vs_3z", 
                              restore_dir=None, restore_reward=None):
    """
    Setup HARL runner with trained model for rendering
    """
    # Get default configurations
    algo_args, env_args = get_defaults_yaml_args(algo, env)
    main_args = {"algo": algo, "env": env, "exp_name": "rendering"}
    
    # Configure for evaluation
    algo_args["eval"]["n_eval_rollout_threads"] = 1
    algo_args["eval"]["eval_episodes"] = 1
    algo_args["render"]["use_render"] = True  # Enable rendering mode
    
    # Set map name for SMAC
    if env == "smac":
        env_args["map_name"] = map_name
    
    # Create runner
    runner = RUNNER_REGISTRY[main_args["algo"]](main_args, algo_args, env_args)
    
    # Debug: Print runner attributes
    print(f"Runner type: {type(runner)}")
    print(f"Runner has critic: {hasattr(runner, 'critic')}")
    print(f"Runner has value_function: {hasattr(runner, 'value_function')}")
    if hasattr(runner, 'algo_args'):
        print(f"Render mode: {runner.algo_args.get('render', {}).get('use_render', 'unknown')}")
    
    # Load trained model if provided
    if restore_dir and restore_reward:
        print(f"Loading model from {restore_dir}")
        for agent_id in range(runner.num_agents):
            model_path = os.path.join(restore_dir, f"actor_agent{agent_id}_{restore_reward}.pt")
            if os.path.exists(model_path):
                policy_actor_state_dict = torch.load(model_path, weights_only=False)
                runner.actor[agent_id].actor.load_state_dict(policy_actor_state_dict)
                print(f"Loaded actor for agent {agent_id}")
            else:
                print(f"Warning: Model file not found: {model_path}")
        
        # Load critic if available (handle different algorithm structures)
        critic_path = os.path.join(restore_dir, f"critic_agent_{restore_reward}.pt")
        if os.path.exists(critic_path):
            policy_critic_state_dict = torch.load(critic_path, weights_only=False)
            # Try different critic access patterns for different algorithms
            if hasattr(runner, 'critic'):
                if hasattr(runner.critic, 'critic'):
                    runner.critic.critic.load_state_dict(policy_critic_state_dict)
                else:
                    runner.critic.load_state_dict(policy_critic_state_dict)
                print("Loaded critic")
            elif hasattr(runner, 'value_function'):
                runner.value_function.load_state_dict(policy_critic_state_dict)
                print("Loaded value function")
            else:
                print("Warning: Could not find critic to load")
        else:
            print("Critic file not found - continuing without critic")
    
    runner.prep_training()
    return runner

def play_sc2_replay(replay_path):
    """
    Play a StarCraft II replay file using the SC2 client
    """
    from pysc2 import run_configs
    
    if not os.path.exists(replay_path):
        print(f"Replay file not found: {replay_path}")
        return False
    
    print(f"Playing replay: {replay_path}")
    
    # Get run configuration
    run_config = run_configs.get()
    
    # Start SC2 and play replay
    with run_config.start() as controller:
        replay_data = run_config.replay_data(replay_path)
        controller.start_replay(replay_data=replay_data, observed_player=0)
        
        print("Replay started in StarCraft II")
        print("Press Ctrl+C to stop monitoring...")
        
        while True:
            controller.step()
    
    return True

def main():
    """
    Main function demonstrating both rendering methods
    """
    print("SMAC Gameplay Rendering Demo")
    print("="*50)
    
    # Configuration
    ALGO = "happo"
    ENV = "smac" 
    MAP_NAME = "3s_vs_3z"  # Change to your desired map
    RESTORE_DIR = "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/happo/smac_3s_vs_3z/seed-00001-2025-08-29-08-50-05/models"  # Update this path
    RESTORE_REWARD = "60.492"  # Update this
    
    # Setup runner
    runner = setup_runner_for_rendering(
        algo=ALGO, 
        env=ENV, 
        map_name=MAP_NAME,
        restore_dir=RESTORE_DIR,
        restore_reward=RESTORE_REWARD
    )
    print(f"✓ Runner setup complete for {MAP_NAME}")
    
    # Method 1: Save SC2 Replays
    print("\n" + "="*50)
    print("METHOD 1: StarCraft II Replay Files")
    print("="*50)
    
    replay_paths = save_replay_method(runner, num_episodes=3, replay_dir="sc2_replays")
    
    if replay_paths:
        print(f"✓ Created {len(replay_paths)} replay files:")
        for path in replay_paths:
            print(f"  - {path}")
        
        # Offer to play the first replay
        if replay_paths:
            response = input(f"\nPlay first replay in SC2? (y/n): ").lower()
            if response == 'y':
                play_sc2_replay(replay_paths[0])
    else:
        print("✗ No replay files were created")
    
    # Method 2: Capture Frames
    print("\n" + "="*50)
    print("METHOD 2: Frame Capture (MP4/GIF)")
    print("="*50)
    
    video_paths = capture_frames_method(runner, num_episodes=2, output_dir="captured_videos", fps=15)
    
    if video_paths:
        print(f"✓ Created {len(video_paths)} video files:")
        for path in video_paths:
            print(f"  - {path}")
    else:
        print("⚠ No video files were created (SMAC may not support direct frame capture)")
        print("  Consider using the SC2 replay method instead")
    
    print("\n" + "="*50)
    print("RENDERING COMPLETE")
    print("="*50)
    print("Note: SMAC environments work best with SC2 replay files (.SC2Replay)")
    print("These can be played in the StarCraft II client for full visual experience.")
    
    # Cleanup
    runner.close()

if __name__ == "__main__":
    main()