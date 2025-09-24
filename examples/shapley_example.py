#!/usr/bin/env python3
"""
Example usage of Monte Carlo Shapley Values computation
This script demonstrates how to run the Shapley value computation
"""

import os
import sys

# Add HARL to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_shapley_example():
    """
    Example of running Monte Carlo Shapley values computation
    """
    
    # Example 1: Basic usage with default parameters
    print("=" * 60)
    print("Example 1: Basic Monte Carlo Shapley Values")
    print("=" * 60)
    
    cmd1 = """
    python shapley_monte_carlo.py \\
        --algo happo \\
        --env pettingzoo_mpe \\
        --exp_name shapley_example_basic \\
        --M 500 \\
        --seed 42
    """
    print("Command to run:")
    print(cmd1.strip())
    print("\nThis will:")
    print("- Use HAPPO algorithm with PettingZoo MPE environment")
    print("- Run 500 Monte Carlo samples")
    print("- Use random seed 42 for reproducibility")
    print("- Save results to shapley_results/[timestamp]/")
    
    # Example 2: With model restoration
    print("\n" + "=" * 60)
    print("Example 2: With Pre-trained Model")
    print("=" * 60)
    
    cmd2 = """
    python shapley_monte_carlo.py \\
        --algo happo \\
        --env pettingzoo_mpe \\
        --exp_name shapley_pretrained \\
        --M 1000 \\
        --restore_dir /path/to/model/checkpoints \\
        --restore_reward 100 \\
        --restore_episode 1000 \\
        --seed 42
    """
    print("Command to run:")
    print(cmd2.strip())
    print("\nThis will:")
    print("- Load a pre-trained model from specified directory")
    print("- Use the restored model for Shapley value computation")
    print("- Run 1000 Monte Carlo samples for better accuracy")
    
    # Example 3: With exact computation for small number of agents
    print("\n" + "=" * 60)
    print("Example 3: Monte Carlo + Exact Computation")
    print("=" * 60)
    
    cmd3 = """
    python shapley_monte_carlo.py \\
        --algo happo \\
        --env pettingzoo_mpe \\
        --exp_name shapley_comparison \\
        --M 2000 \\
        --exact \\
        --seed 42
    """
    print("Command to run:")
    print(cmd3.strip())
    print("\nThis will:")
    print("- Compute both Monte Carlo and exact Shapley values")
    print("- Compare the two methods for accuracy assessment")
    print("- Only works if number of agents <= 6 (computational complexity)")
    
    # Example 4: With GIF Creation
    print("\n" + "=" * 60)
    print("Example 4: Save GIFs of Game Episodes")
    print("=" * 60)
    
    cmd4 = """
    python shapley_monte_carlo.py \\
        --algo happo \\
        --env pettingzoo_mpe \\
        --exp_name shapley_with_gifs \\
        --M 500 \\
        --save_gifs \\
        --seed 42
    """
    print("Command to run:")
    print(cmd4.strip())
    print("\nThis will:")
    print("- Save GIFs of selected game episodes during computation")
    print("- Show how different coalitions perform visually")
    print("- Save GIFs in results/[timestamp]/gifs/ directory")
    print("- Include episodes with different coalition configurations")
    
    # Example 5: Different environment
    print("\n" + "=" * 60)
    print("Example 5: Different Environment (SMAC)")
    print("=" * 60)
    
    cmd5 = """
    python shapley_monte_carlo.py \\
        --algo happo \\
        --env smac \\
        --exp_name shapley_smac \\
        --M 1500 \\
        --seed 123
    """
    print("Command to run:")
    print(cmd5.strip())
    print("\nThis will:")
    print("- Use SMAC (StarCraft Multi-Agent Challenge) environment")
    print("- Run with 1500 Monte Carlo samples")
    
    # Results explanation
    print("\n" + "=" * 60)
    print("Understanding the Results")
    print("=" * 60)
    
    print("""
    The script will output:
    
    1. Console Output:
       - Monte Carlo Shapley values for each agent
       - Total Shapley value (should equal total system reward)
       - If exact computation is enabled: comparison with exact values
       - If GIFs enabled: number of GIFs created
    
    2. Files Created:
       - shapley_monte_carlo.csv: Shapley values in CSV format
       - shapley_monte_carlo.png: Bar chart visualization
       - shapley_exact.csv: Exact Shapley values (if computed)
       - shapley_exact.png: Exact values visualization
       - config_and_results.json: Complete configuration and results
       - gifs/ directory: Game episode GIFs (if --save_gifs flag used)
    
    3. GIF Files (if enabled):
       - episode_XXXX_coalition_Y.gif: Episodes with different coalitions
       - Shows visual performance of different agent combinations
       - Helps understand why certain agents have higher Shapley values
    
    4. Interpretation:
       - Positive Shapley value: Agent contributes positively to team performance
       - Negative Shapley value: Agent hurts team performance
       - Larger absolute value: Larger impact on team performance
       - Sum of all Shapley values equals the total team reward
       - GIFs show the actual gameplay corresponding to different coalitions
    """)
    
    # Implementation notes
    print("\n" + "=" * 60)
    print("Implementation Notes")
    print("=" * 60)
    
    print("""
    Key Features of this Implementation:
    
    1. Follows Algorithm 1 exactly:
       - Monte Carlo sampling of coalitions
       - Marginal contribution computation
       - Proper averaging over samples
    
    2. Normal Scenario Only:
       - No attacks or adversarial behavior
       - Pure cooperative Shapley value computation
       - All agents use their trained policies when in coalition
    
    3. Handles Different Action Spaces:
       - Discrete actions (e.g., movement, attack)
       - Continuous actions (e.g., force, velocity)
       - Available action masks
    
    4. Configurable:
       - Number of Monte Carlo samples (M)
       - Random seeds for reproducibility
       - Different algorithms and environments
       - Model restoration from checkpoints
       - GIF creation for visual analysis
    
    5. Validation:
       - Optional exact computation for small agent numbers
       - Efficiency property: sum of Shapley values = total reward
       - Comparison between Monte Carlo and exact methods
    
    6. Visualization:
       - Bar charts of Shapley values
       - GIF animations of game episodes (optional)
       - Shows actual gameplay for different coalitions
       - Helps understand agent contributions visually
    """)


if __name__ == "__main__":
    run_shapley_example()
