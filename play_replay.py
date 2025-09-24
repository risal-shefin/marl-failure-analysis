#!/usr/bin/env python3
"""
Script to play StarCraft II replay files.
This script will launch StarCraft II and play the specified replay.
"""

import os
import sys
from pysc2 import run_configs
from pysc2.lib import replay


def play_sc2_replay(replay_path):
    """
    Play a StarCraft II replay file.
    
    Args:
        replay_path (str): Path to the .SC2Replay file
    """
    if not os.path.exists(replay_path):
        print(f"Error: Replay file not found at {replay_path}")
        return False
    
    if not replay_path.endswith('.SC2Replay'):
        print("Error: File must be a .SC2Replay file")
        return False
    
    print(f"Loading replay: {replay_path}")
    
    try:
        # Get the run configuration
        run_config = run_configs.get()
        
        # Start the SC2 process
        print("Starting StarCraft II...")
        with run_config.start() as controller:
            # Load and start the replay
            print("Loading replay...")
            replay_data = run_config.replay_data(replay_path)
            
            start_replay = controller.start_replay(
                replay_data=replay_data,
                options=None,
                observed_player=0  # Observe all players
            )
            
            print("Replay started successfully!")
            print("The replay should now be playing in StarCraft II.")
            print("You can control playback speed and view using the SC2 interface.")
            print("Press Ctrl+C to stop the script (this won't stop the replay).")
            
            # Keep the connection alive while replay is playing
            try:
                while True:
                    controller.step()
            except KeyboardInterrupt:
                print("\nScript interrupted. StarCraft II will continue running.")
                
    except Exception as e:
        print(f"Error playing replay: {e}")
        return False
    
    return True


def main():
    # Your replay path
    replay_path = "/home/guptd23/StarCraftII/Replays/3s_vs_3z_2025-09-03-00-41-04.SC2Replay"
    
    print("StarCraft II Replay Player")
    print("=" * 40)
    print(f"Replay file: {replay_path}")
    print()
    
    # Check if StarCraft II is installed
    try:
        run_config = run_configs.get()
        print(f"StarCraft II found at: {run_config.exec_path}")
    except Exception as e:
        print(f"Error: StarCraft II not found or not properly configured: {e}")
        print("Make sure StarCraft II is installed and PYSC2 is properly configured.")
        return
    
    # Play the replay
    success = play_sc2_replay(replay_path)
    
    if success:
        print("Replay playback initiated successfully!")
    else:
        print("Failed to start replay playback.")


if __name__ == "__main__":
    main()
