#!/usr/bin/env python3
"""
Test script to verify GIF creation from PettingZoo MPE environment
"""

import os
import sys
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from harl.envs.pettingzoo_mpe.pettingzoo_mpe_env import PettingZooMPEEnv

def test_environment_rendering():
    """Test if the environment can render frames"""
    
    # Configuration for simple_spread_v3 environment
    env_args = {
        "scenario": "simple_spread_v3",
        "max_cycles": 25,
        "continuous_actions": False,
    }
    
    print("Creating PettingZoo MPE environment...")
    env = PettingZooMPEEnv(env_args)
    
    print(f"Environment created with {env.n_agents} agents")
    print(f"Render mode is set to: {env.args.get('render_mode', 'not set')}")
    
    # Reset environment
    obs, share_obs, avail_actions = env.reset(seed=42)
    
    print("Testing frame capture...")
    frames = []
    
    for step in range(10):
        # Get a frame
        try:
            frame = env.render()
            print(f"Step {step}: Got frame of type {type(frame)}")
            if isinstance(frame, np.ndarray):
                print(f"  Frame shape: {frame.shape}, dtype: {frame.dtype}")
                if frame.size > 0:
                    frames.append(frame.copy())
            else:
                print(f"  Frame is not a numpy array: {frame}")
        except Exception as e:
            print(f"Step {step}: Error capturing frame: {e}")
        
        # Take random actions
        actions = []
        for agent_id in range(env.n_agents):
            n_actions = env.action_space[agent_id].n
            random_action = np.random.randint(0, n_actions)
            actions.append(random_action)
        
        actions = np.array(actions).reshape(1, -1, 1)  # Shape for env.step
        
        obs, share_obs, rewards, dones, infos, avail_actions = env.step(actions)
        
        if np.all(dones):
            break
    
    print(f"\nCaptured {len(frames)} frames")
    
    # Test GIF creation
    if frames:
        from examples.shapley_monte_carlo import save_gif_from_frames
        
        gif_path = "test_output.gif"
        print(f"Saving test GIF to: {gif_path}")
        
        save_gif_from_frames(frames, gif_path, duration=500)  # 500ms per frame for easier viewing
        
        if os.path.exists(gif_path):
            print(f"SUCCESS: GIF created successfully at {gif_path}")
        else:
            print("ERROR: GIF file was not created")
    else:
        print("ERROR: No frames captured - cannot create GIF")
    
    env.close()

def create_test_frames():
    """Create some test frames to verify GIF creation works"""
    print("Creating test frames for GIF verification...")
    
    # Create some simple test frames
    frames = []
    for i in range(10):
        # Create a 100x100 RGB frame with changing colors
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        
        # Create a simple animation - moving colored square
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][i % 3]
        x_pos = 10 + (i * 8)
        
        frame[20:40, x_pos:x_pos+20] = color
        frames.append(frame)
    
    from examples.shapley_monte_carlo import save_gif_from_frames
    
    gif_path = "test_animation.gif"
    print(f"Saving test animation GIF to: {gif_path}")
    
    save_gif_from_frames(frames, gif_path, duration=300)
    
    if os.path.exists(gif_path):
        print(f"SUCCESS: Test animation GIF created at {gif_path}")
    else:
        print("ERROR: Test animation GIF was not created")

if __name__ == "__main__":
    print("="*50)
    print("Testing GIF Creation for PettingZoo MPE")
    print("="*50)
    
    # First test with synthetic frames
    create_test_frames()
    
    print("\n" + "-"*50)
    
    # Then test with actual environment
    try:
        test_environment_rendering()
    except Exception as e:
        print(f"Error testing environment rendering: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTest completed!")
