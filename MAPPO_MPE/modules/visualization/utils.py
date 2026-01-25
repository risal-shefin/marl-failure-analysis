"""
Core visualization utilities.
"""
import matplotlib.pyplot as plt
import imageio
import os


def get_agent_colors(n_agents):
    """
    Get consistent color palette for agents across all plots.
    
    Args:
        n_agents: Number of agents
        
    Returns:
        dict: Dictionary mapping agent index to color
    """
    cmap = plt.get_cmap('tab20')
    agent_colors = {i: cmap(i % 20) for i in range(n_agents)}
    return agent_colors


def save_frames_as_gif(frames, filepath, fps=10):
    """
    Save a list of RGB frames as a GIF file.
    
    Args:
        frames: List of numpy arrays representing RGB frames
        filepath: Path where to save the GIF file
        fps: Frames per second for the GIF (default: 10)
    """
    if not frames:
        print(f"Warning: No frames to save for {filepath}")
        return
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Calculate duration per frame in seconds
    duration = 1.0 / fps
    
    # Save frames as GIF
    imageio.mimsave(filepath, frames, duration=duration)
    print(f"Saved GIF with {len(frames)} frames to: {filepath}")