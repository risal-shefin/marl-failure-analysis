"""
Influence-based visualization functions for action and observation influences.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from .utils import get_agent_colors


def plot_action_influences(action_influences_matrix_history_normal, action_influences_matrix_history, attacked_steps, atk_agent_id, logdir):
    """
    Plot the time series of action influences for each agent.
    Shows normal scenario in first row and attacked scenario in second row.
    For each agent i, show how much each agent j influences agent i's Q-value over time.
    """
    N = len(action_influences_matrix_history[0])  # number of agents
    T_attacked = len(action_influences_matrix_history)     # number of timesteps in attacked scenario
    T_normal = len(action_influences_matrix_history_normal)  # number of timesteps in normal scenario
    
    # Create 2*N subplots (two rows: normal and attacked scenarios)
    max_per_row = 3
    cols = min(N, max_per_row)
    rows = 2  # Two rows: normal and attacked scenarios
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows + 1))
    if N == 1:
        axes = axes.reshape(rows, 1)
    
    fig.suptitle(f'Action Influences: ∂Q_i/∂a_j (Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.96)
    
    # Use consistent agent colors
    agent_colors = get_agent_colors(N)
    
    # Track legend elements to avoid duplicates
    legend_elements = []
    legend_labels = []
    attacked_region_element = None
    
    scenarios = [
        (action_influences_matrix_history_normal, T_normal, "Normal"),
        (action_influences_matrix_history, T_attacked, "Attacked")
    ]
    
    # Calculate global max and min for consistent y-axis scaling
    global_max = 0
    global_min = 0
    for influences_data, T, _ in scenarios:
        for t in range(T):
            for i in range(N):
                for j in range(N):
                    value = influences_data[t][i][j]
                    global_max = max(global_max, value)
                    global_min = min(global_min, value)
    
    # Add row labels for scenarios
    for scenario_idx, (influences_data, T, scenario_name) in enumerate(scenarios):
        # Add scenario label on the left side
        if scenario_idx == 0:
            fig.text(0.02, 0.75, 'Normal\nScenario', fontsize=14, fontweight='bold', 
                    ha='center', va='center', rotation=90)
        else:
            fig.text(0.02, 0.25, 'Attacked\nScenario', fontsize=14, fontweight='bold', 
                    ha='center', va='center', rotation=90)
        
        for i in range(min(N, cols)):  # Only plot up to max_per_row agents
            ax = axes[scenario_idx, i]
            
            # Plot action influences for each agent j on agent i
            for j in range(N):
                influence_series = [influences_data[t][i][j] for t in range(T)]
                timesteps = range(T)
                
                # Self-influence (solid line)
                if i == j:
                    line = ax.plot(timesteps, influence_series, color=agent_colors[j], 
                                    linewidth=2.0, linestyle='-', alpha=1.0)[0]
                else:
                    # Influence from others (dashed line)
                    line = ax.plot(timesteps, influence_series, color=agent_colors[j], 
                                    linewidth=2.0, linestyle='--', alpha=1.0)[0]
                
                # avoid duplicate legend entries (only add for first subplot)
                if scenario_idx == 0 and i == 0:
                    legend_elements.append(line)
                    legend_labels.append(f'Agent {j}')
            
            # Mark attacked region (only for attacked scenario)
            if scenario_idx == 1 and attacked_steps:
                start = min(attacked_steps)
                end = max(attacked_steps)
                if i == atk_agent_id:
                    region = ax.axvspan(start, end, color='red', alpha=0.1)
                    if attacked_region_element is None:  # Store for later addition to legend
                        attacked_region_element = region
                else:
                    ax.axvspan(start, end, color='orange', alpha=0.05)
            
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Action Influence Magnitude')
            ax.set_title(f'Influences on Agent {i}')
            ax.grid(True, alpha=0.3)
            
            # Set consistent y-axis limits across all plots
            y_margin = (global_max - global_min) * 0.1
            ax.set_ylim(global_min - y_margin, global_max + y_margin)
    
    # Hide unused subplots
    for scenario_idx in range(rows):
        for j in range(N, cols):
            if j < cols:
                axes[scenario_idx, j].set_visible(False)
    
    # Add attacked region to legend at the end if it exists
    if attacked_region_element is not None:
        legend_elements.append(attacked_region_element)
        legend_labels.append('Attacked Region')
    
    # Create single legend at the bottom
    fig.legend(legend_elements, legend_labels, loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), ncol=min(len(legend_labels), 6), 
               fontsize=9, frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout to make room for legend and row labels
    plt.tight_layout(rect=[0.04, 0.08, 1, 0.94])
    
    out_path = os.path.join(logdir, f'action_influences_attacked_{atk_agent_id}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved action influences plot to {out_path}")


def plot_pairwise_action_influences(action_influences_normal, action_influences_attacked, attacked_steps, atk_agent_id, logdir):
    """
    Plot N×N grid of subplots showing pairwise action influences.
    Each subplot (i,j) shows the influence of agent j on agent i for both normal and attacked scenarios.
    """
    N = len(action_influences_normal[0])  # number of agents
    T_normal = len(action_influences_normal)
    T_attacked = len(action_influences_attacked)
    
    # Create N×N subplots
    fig, axes = plt.subplots(N, N, figsize=(3*N, 3*N))
    if N == 1:
        axes = [[axes]]
    elif N > 1 and axes.ndim == 1:
        axes = axes.reshape(N, N)
    
    fig.suptitle(f'Pairwise Action Influences: ∂Q_i/∂a_j (Attacked Agent ID: {atk_agent_id})', fontsize=14, y=0.95)
    
    for i in range(N):  # agent i (influenced)
        for j in range(N):  # agent j (influencer)
            ax = axes[i][j]
            
            # Extract influence time series for agent j's influence on agent i
            normal_series = [action_influences_normal[t][i][j] for t in range(T_normal)]
            attacked_series = [action_influences_attacked[t][i][j] for t in range(T_attacked)]
            
            # Plot time series
            timesteps_normal = range(T_normal)
            timesteps_attacked = range(T_attacked)
            
            ax.plot(timesteps_normal, normal_series, 'g-', label='Normal', linewidth=2, alpha=0.8)
            ax.plot(timesteps_attacked, attacked_series, 'r-', label='Attacked', linewidth=2, alpha=0.8)
            
            # Mark attacked region
            if attacked_steps:
                start = min(attacked_steps)
                end = max(attacked_steps)
                ax.axvspan(start, end, color='red', alpha=0.1)
            
            # Styling
            ax.set_title(f'Agent {j} → Agent {i}', fontsize=10)
            ax.set_xlabel('Timestep', fontsize=9)
            ax.set_ylabel('Influence Magnitude', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            # Add legend only to the top-right subplot to avoid clutter
            if i == 0 and j == N-1:
                ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    out_path = os.path.join(logdir, f'pairwise_action_influences_attacked_{atk_agent_id}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved pairwise action influences plot to {out_path}")


def plot_second_order_action_influences(second_order_action_influences_history_normal, second_order_action_influences_history, attacked_steps, atk_agent_id, logdir):
    """
    Plot the time series of second-order action influences for each agent.
    Shows normal scenario in first row and attacked scenario in second row.
    For each agent i, show how much each agent j's action curvature influences agent i's Q-value over time.
    """
    N = len(second_order_action_influences_history[0])  # number of agents
    T_attacked = len(second_order_action_influences_history)     # number of timesteps in attacked scenario
    T_normal = len(second_order_action_influences_history_normal)  # number of timesteps in normal scenario
    
    # Create 2*N subplots (two rows: normal and attacked scenarios)
    max_per_row = 3
    cols = min(N, max_per_row)
    rows = 2  # Two rows: normal and attacked scenarios
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows + 1))
    if N == 1:
        axes = axes.reshape(rows, 1)
    
    fig.suptitle(f'Second-Order Action Influences: ∂²Q_i/(∂a_j)² (Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.96)
    
    # Use consistent agent colors
    agent_colors = get_agent_colors(N)
    
    # Track legend elements to avoid duplicates
    legend_elements = []
    legend_labels = []
    attacked_region_element = None
    
    scenarios = [
        (second_order_action_influences_history_normal, T_normal, "Normal"),
        (second_order_action_influences_history, T_attacked, "Attacked")
    ]
    
    # Calculate global max and min for consistent y-axis scaling
    global_max = 0
    global_min = 0
    for influences_data, T, _ in scenarios:
        for t in range(T):
            for i in range(N):
                for j in range(N):
                    value = influences_data[t][i][j]
                    global_max = max(global_max, value)
                    global_min = min(global_min, value)
    
    # Add row labels for scenarios
    for scenario_idx, (influences_data, T, scenario_name) in enumerate(scenarios):
        # Add scenario label on the left side
        if scenario_idx == 0:
            fig.text(0.02, 0.75, 'Normal\nScenario', fontsize=14, fontweight='bold', 
                    ha='center', va='center', rotation=90)
        else:
            fig.text(0.02, 0.25, 'Attacked\nScenario', fontsize=14, fontweight='bold', 
                    ha='center', va='center', rotation=90)
        
        for i in range(min(N, cols)):  # Only plot up to max_per_row agents
            ax = axes[scenario_idx, i]
            
            # Plot second-order influences for each agent j on agent i
            for j in range(N):
                influence_series = [influences_data[t][i][j] for t in range(T)]
                timesteps = range(T)
                
                # Self-influence (solid line) vs others (dashed line)
                if i == j:
                    line = ax.plot(timesteps, influence_series, color=agent_colors[j], 
                                    linewidth=2.5, linestyle='-', alpha=1.0)[0]
                else:
                    line = ax.plot(timesteps, influence_series, color=agent_colors[j], 
                                    linewidth=2.0, linestyle='--', alpha=0.8)[0]
                
                # avoid duplicate legend entries (only add for first subplot)  
                if scenario_idx == 0 and i == 0:
                    legend_elements.append(line)
                    legend_labels.append(f'Agent {j}')
            
            # Mark attacked region (only for attacked scenario)
            if scenario_idx == 1 and attacked_steps:
                start = min(attacked_steps)
                end = max(attacked_steps)
                if i == atk_agent_id:
                    region = ax.axvspan(start, end, color='red', alpha=0.1)
                    if attacked_region_element is None:  # Store for later addition to legend
                        attacked_region_element = region
                else:
                    ax.axvspan(start, end, color='orange', alpha=0.05)
            
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Second-Order Influence Magnitude')
            ax.set_title(f'Second-Order Influences on Agent {i}')
            ax.grid(True, alpha=0.3)
            
            # Set consistent y-axis limits across all plots
            y_margin = (global_max - global_min) * 0.1
            ax.set_ylim(global_min - y_margin, global_max + y_margin)
    
    # Hide unused subplots
    for scenario_idx in range(rows):
        for j in range(N, cols):
            if j < cols:
                axes[scenario_idx, j].set_visible(False)
    
    # Add attacked region to legend at the end if it exists
    if attacked_region_element is not None:
        legend_elements.append(attacked_region_element)
        legend_labels.append('Attacked Region')
    
    # Create single legend at the bottom
    fig.legend(legend_elements, legend_labels, loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), ncol=min(len(legend_labels), 6), 
               fontsize=9, frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout to make room for legend and row labels
    plt.tight_layout(rect=[0.04, 0.08, 1, 0.94])
    
    out_path = os.path.join(logdir, f'second_order_action_influences_attacked_{atk_agent_id}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved second-order action influences plot to {out_path}")


def plot_pairwise_second_order_action_influences(second_order_normal, second_order_attacked, attacked_steps, atk_agent_id, logdir):
    """
    Plot N×N grid of subplots showing pairwise second-order action influences.
    Each subplot (i,j) shows ∂²Q_i/(∂a_j)² for both normal and attacked scenarios.
    """
    N = len(second_order_normal[0])  # number of agents
    T_normal = len(second_order_normal)
    T_attacked = len(second_order_attacked)
    
    # Create N×N subplots
    fig, axes = plt.subplots(N, N, figsize=(3*N, 3*N))
    if N == 1:
        axes = [[axes]]
    elif N > 1 and axes.ndim == 1:
        axes = axes.reshape(N, N)
    
    fig.suptitle(f'Pairwise Second-Order Action Influences: ∂²Q_i/(∂a_j)² (Attacked Agent ID: {atk_agent_id})', fontsize=14, y=0.95)
    
    for i in range(N):  # agent i (influenced)
        for j in range(N):  # agent j (influencer)
            ax = axes[i][j]
            
            # Extract second-order influence time series for agent j's action curvature on agent i
            normal_series = [second_order_normal[t][i][j] for t in range(T_normal)]
            attacked_series = [second_order_attacked[t][i][j] for t in range(T_attacked)]
            
            # Plot time series
            timesteps_normal = range(T_normal)
            timesteps_attacked = range(T_attacked)
            
            ax.plot(timesteps_normal, normal_series, 'g-', label='Normal', linewidth=2, alpha=0.8)
            ax.plot(timesteps_attacked, attacked_series, 'r-', label='Attacked', linewidth=2, alpha=0.8)
            
            # Mark attacked region
            if attacked_steps:
                start = min(attacked_steps)
                end = max(attacked_steps)
                ax.axvspan(start, end, color='red', alpha=0.1)
            
            # Styling
            ax.set_title(f'Agent {j} → Agent {i}', fontsize=10)
            ax.set_xlabel('Timestep', fontsize=9)
            ax.set_ylabel('2nd-Order Influence', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            # Add legend only to the top-right subplot to avoid clutter
            if i == 0 and j == N-1:
                ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    out_path = os.path.join(logdir, f'pairwise_second_order_action_influences_attacked_{atk_agent_id}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved pairwise second-order action influences plot to {out_path}")


def plot_observation_influences(observation_influences_matrix_history_normal, observation_influences_matrix_history, attacked_steps, atk_agent_id, logdir):
    """
    Plot the time series of observation influences for each agent.
    Shows normal scenario in first row and attacked scenario in second row.
    For each agent i, show how much each agent j's observation influences agent i's Q-value over time.
    """
    N = len(observation_influences_matrix_history[0])  # number of agents
    T_attacked = len(observation_influences_matrix_history)     # number of timesteps in attacked scenario
    T_normal = len(observation_influences_matrix_history_normal)  # number of timesteps in normal scenario
    
    # Create 2*N subplots (two rows: normal and attacked scenarios)
    max_per_row = 3
    cols = min(N, max_per_row)
    rows = 2  # Two rows: normal and attacked scenarios
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows + 1))
    if N == 1:
        axes = axes.reshape(rows, 1)
    
    fig.suptitle(f'Observation Influences: ∂Q_i/∂obs_j (Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.96)
    
    # Use consistent agent colors
    agent_colors = get_agent_colors(N)
    
    # Track legend elements to avoid duplicates
    legend_elements = []
    legend_labels = []
    attacked_region_element = None
    
    scenarios = [
        (observation_influences_matrix_history_normal, T_normal, "Normal"),
        (observation_influences_matrix_history, T_attacked, "Attacked")
    ]
    
    # Calculate global max and min for consistent y-axis scaling
    global_max = 0
    global_min = 0
    for influences_data, T, _ in scenarios:
        for t in range(T):
            for i in range(N):
                for j in range(N):
                    value = influences_data[t][i][j]
                    global_max = max(global_max, value)
                    global_min = min(global_min, value)
    
    # Add row labels for scenarios
    for scenario_idx, (influences_data, T, scenario_name) in enumerate(scenarios):
        # Add scenario label on the left side
        if scenario_idx == 0:
            fig.text(0.02, 0.75, 'Normal\nScenario', fontsize=14, fontweight='bold', 
                    ha='center', va='center', rotation=90)
        else:
            fig.text(0.02, 0.25, 'Attacked\nScenario', fontsize=14, fontweight='bold', 
                    ha='center', va='center', rotation=90)
        
        for i in range(min(N, cols)):  # Only plot up to max_per_row agents
            ax = axes[scenario_idx, i]
            
            # Plot observation influences for each agent j on agent i
            for j in range(N):
                influence_series = [influences_data[t][i][j] for t in range(T)]
                timesteps = range(T)
                
                # Self-influence (solid line)
                if i == j:
                    line = ax.plot(timesteps, influence_series, color=agent_colors[j], 
                                    linewidth=2.0, linestyle='-', alpha=1.0)[0]
                else:
                    # Influence from others (dashed line)
                    line = ax.plot(timesteps, influence_series, color=agent_colors[j], 
                                    linewidth=2.0, linestyle='--', alpha=1.0)[0]
                
                # avoid duplicate legend entries (only add for first subplot)
                if scenario_idx == 0 and i == 0:
                    legend_elements.append(line)
                    legend_labels.append(f'Agent {j}')
            
            # Mark attacked region (only for attacked scenario)
            if scenario_idx == 1 and attacked_steps:
                start = min(attacked_steps)
                end = max(attacked_steps)
                if i == atk_agent_id:
                    region = ax.axvspan(start, end, color='red', alpha=0.1)
                    if attacked_region_element is None:
                        attacked_region_element = region
                else:
                    ax.axvspan(start, end, color='orange', alpha=0.05)
            
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Observation Influence Magnitude')
            ax.set_title(f'Observation Influences on Agent {i}')
            ax.grid(True, alpha=0.3)
            
            # Set consistent y-axis limits across all plots
            y_margin = (global_max - global_min) * 0.1
            ax.set_ylim(global_min - y_margin, global_max + y_margin)
    
    # Hide unused subplots
    for scenario_idx in range(rows):
        for j in range(N, cols):
            if j < cols:
                axes[scenario_idx, j].set_visible(False)
    
    # Add attacked region to legend at the end if it exists
    if attacked_region_element is not None:
        legend_elements.append(attacked_region_element)
        legend_labels.append('Attacked Region')
    
    # Create single legend at the bottom
    fig.legend(legend_elements, legend_labels, loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), ncol=min(len(legend_labels), 6), 
               fontsize=9, frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout to make room for legend and row labels
    plt.tight_layout(rect=[0.04, 0.08, 1, 0.94])
    
    out_path = os.path.join(logdir, f'observation_influences_attacked_{atk_agent_id}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved observation influences plot to {out_path}")


def plot_pairwise_observation_influences(observation_influences_normal, observation_influences_attacked, attacked_steps, atk_agent_id, logdir):
    """
    Plot N×N grid of subplots showing pairwise observation influences.
    Each subplot (i,j) shows the influence of agent j's observation on agent i for both normal and attacked scenarios.
    """
    N = len(observation_influences_normal[0])  # number of agents
    T_normal = len(observation_influences_normal)
    T_attacked = len(observation_influences_attacked)
    
    # Create N×N subplots
    fig, axes = plt.subplots(N, N, figsize=(3*N, 3*N))
    if N == 1:
        axes = [[axes]]
    elif N > 1 and axes.ndim == 1:
        axes = axes.reshape(N, N)
    
    fig.suptitle(f'Pairwise Observation Influences: ∂Q_i/∂obs_j (Attacked Agent ID: {atk_agent_id})', fontsize=14, y=0.95)
    
    for i in range(N):  # agent i (influenced)
        for j in range(N):  # agent j (influencer)
            ax = axes[i][j]
            
            # Extract influence time series for agent j's observation influence on agent i
            normal_series = [observation_influences_normal[t][i][j] for t in range(T_normal)]
            attacked_series = [observation_influences_attacked[t][i][j] for t in range(T_attacked)]
            
            # Plot time series
            timesteps_normal = range(T_normal)
            timesteps_attacked = range(T_attacked)
            
            ax.plot(timesteps_normal, normal_series, 'g-', label='Normal', linewidth=2, alpha=0.8)
            ax.plot(timesteps_attacked, attacked_series, 'r-', label='Attacked', linewidth=2, alpha=0.8)
            
            # Mark attacked region
            if attacked_steps:
                start = min(attacked_steps)
                end = max(attacked_steps)
                ax.axvspan(start, end, color='red', alpha=0.1)
            
            # Styling
            ax.set_title(f'Agent {j} → Agent {i}', fontsize=10)
            ax.set_xlabel('Timestep', fontsize=9)
            ax.set_ylabel('Observation Influence Magnitude', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            # Add legend only to the top-right subplot to avoid clutter
            if i == 0 and j == N-1:
                ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    out_path = os.path.join(logdir, f'pairwise_observation_influences_attacked_{atk_agent_id}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved pairwise observation influences plot to {out_path}")


def plot_second_order_observation_influences(second_order_observation_influences_history_normal, second_order_observation_influences_history, attacked_steps, atk_agent_id, logdir):
    """
    Plot the time series of second-order observation influences for each agent.
    Shows normal scenario in first row and attacked scenario in second row.
    For each agent i, show how much each agent j's observation curvature influences agent i's Q-value over time.
    """
    N = len(second_order_observation_influences_history[0])  # number of agents
    T_attacked = len(second_order_observation_influences_history)     # number of timesteps in attacked scenario
    T_normal = len(second_order_observation_influences_history_normal)  # number of timesteps in normal scenario
    
    # Create 2*N subplots (two rows: normal and attacked scenarios)
    max_per_row = 3
    cols = min(N, max_per_row)
    rows = 2  # Two rows: normal and attacked scenarios
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows + 1))
    if N == 1:
        axes = axes.reshape(rows, 1)
    
    fig.suptitle(f'Second-Order Observation Influences: ∂²Q_i/(∂obs_j)² (Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.96)
    
    # Use consistent agent colors
    agent_colors = get_agent_colors(N)
    
    # Track legend elements to avoid duplicates
    legend_elements = []
    legend_labels = []
    attacked_region_element = None
    
    scenarios = [
        (second_order_observation_influences_history_normal, T_normal, "Normal"),
        (second_order_observation_influences_history, T_attacked, "Attacked")
    ]
    
    # Calculate global max and min for consistent y-axis scaling
    global_max = 0
    global_min = 0
    for influences_data, T, _ in scenarios:
        for t in range(T):
            for i in range(N):
                for j in range(N):
                    value = influences_data[t][i][j]
                    global_max = max(global_max, value)
                    global_min = min(global_min, value)
    
    # Add row labels for scenarios
    for scenario_idx, (influences_data, T, scenario_name) in enumerate(scenarios):
        # Add scenario label on the left side
        if scenario_idx == 0:
            fig.text(0.02, 0.75, 'Normal\nScenario', fontsize=14, fontweight='bold', 
                    ha='center', va='center', rotation=90)
        else:
            fig.text(0.02, 0.25, 'Attacked\nScenario', fontsize=14, fontweight='bold', 
                    ha='center', va='center', rotation=90)
        
        for i in range(min(N, cols)):  # Only plot up to max_per_row agents
            ax = axes[scenario_idx, i]
            
            # Plot second-order observation influences for each agent j on agent i
            for j in range(N):
                influence_series = [influences_data[t][i][j] for t in range(T)]
                timesteps = range(T)
                
                # Self-influence (solid line) vs others (dashed line)
                if i == j:
                    line = ax.plot(timesteps, influence_series, color=agent_colors[j], 
                                    linewidth=2.0, linestyle='-', alpha=1.0)[0]
                else:
                    line = ax.plot(timesteps, influence_series, color=agent_colors[j], 
                                    linewidth=2.0, linestyle='--', alpha=1.0)[0]
                
                # avoid duplicate legend entries (only add for first subplot)
                if scenario_idx == 0 and i == 0:
                    legend_elements.append(line)
                    legend_labels.append(f'Agent {j}')
            
            # Mark attacked region (only for attacked scenario)
            if scenario_idx == 1 and attacked_steps:
                start = min(attacked_steps)
                end = max(attacked_steps)
                if i == atk_agent_id:
                    region = ax.axvspan(start, end, color='red', alpha=0.1)
                    if attacked_region_element is None:
                        attacked_region_element = region
                else:
                    ax.axvspan(start, end, color='orange', alpha=0.05)
            
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Second-Order Observation Influence Magnitude')
            ax.set_title(f'Second-Order Observation Influences on Agent {i}')
            ax.grid(True, alpha=0.3)
            
            # Set consistent y-axis limits across all plots
            y_margin = (global_max - global_min) * 0.1
            ax.set_ylim(global_min - y_margin, global_max + y_margin)
    
    # Hide unused subplots
    for scenario_idx in range(rows):
        for j in range(N, cols):
            if j < cols:
                axes[scenario_idx, j].set_visible(False)
    
    # Add attacked region to legend at the end if it exists
    if attacked_region_element is not None:
        legend_elements.append(attacked_region_element)
        legend_labels.append('Attacked Region')
    
    # Create single legend at the bottom
    fig.legend(legend_elements, legend_labels, loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), ncol=min(len(legend_labels), 6), 
               fontsize=9, frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout to make room for legend and row labels
    plt.tight_layout(rect=[0.04, 0.08, 1, 0.94])
    
    out_path = os.path.join(logdir, f'second_order_observation_influences_attacked_{atk_agent_id}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved second-order observation influences plot to {out_path}")


def plot_pairwise_second_order_observation_influences(second_order_obs_normal, second_order_obs_attacked, attacked_steps, atk_agent_id, logdir):
    """
    Plot N×N grid of subplots showing pairwise second-order observation influences.
    Each subplot (i,j) shows ∂²Q_i/(∂obs_j)² for both normal and attacked scenarios.
    """
    N = len(second_order_obs_normal[0])  # number of agents
    T_normal = len(second_order_obs_normal)
    T_attacked = len(second_order_obs_attacked)
    
    # Create N×N subplots
    fig, axes = plt.subplots(N, N, figsize=(3*N, 3*N))
    if N == 1:
        axes = [[axes]]
    elif N > 1 and axes.ndim == 1:
        axes = axes.reshape(N, N)
    
    fig.suptitle(f'Pairwise Second-Order Observation Influences: ∂²Q_i/(∂obs_j)² (Attacked Agent ID: {atk_agent_id})', fontsize=14, y=0.95)
    
    for i in range(N):  # agent i (influenced)
        for j in range(N):  # agent j (influencer)
            ax = axes[i][j]
            
            # Extract second-order observation influence time series for agent j's observation curvature on agent i
            normal_series = [second_order_obs_normal[t][i][j] for t in range(T_normal)]
            attacked_series = [second_order_obs_attacked[t][i][j] for t in range(T_attacked)]
            
            # Plot time series
            timesteps_normal = range(T_normal)
            timesteps_attacked = range(T_attacked)
            
            ax.plot(timesteps_normal, normal_series, 'g-', label='Normal', linewidth=2, alpha=0.8)
            ax.plot(timesteps_attacked, attacked_series, 'r-', label='Attacked', linewidth=2, alpha=0.8)
            
            # Mark attacked region
            if attacked_steps:
                start = min(attacked_steps)
                end = max(attacked_steps)
                ax.axvspan(start, end, color='red', alpha=0.1)
            
            # Styling
            ax.set_title(f'Agent {j} → Agent {i}', fontsize=10)
            ax.set_xlabel('Timestep', fontsize=9)
            ax.set_ylabel('2nd-Order Obs Influence', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            # Add legend only to the top-right subplot to avoid clutter
            if i == 0 and j == N-1:
                ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    out_path = os.path.join(logdir, f'pairwise_second_order_observation_influences_attacked_{atk_agent_id}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved pairwise second-order observation influences plot to {out_path}")


def plot_frob_norm_influences(frob_norms_matrix_history_normal, frob_norms_matrix_history, attacked_steps, atk_agent_id, logdir):
    """
    Plot the time series of Frobenius norm influences for each agent.
    Shows normal scenario in first row and attacked scenario in second row.
    For each agent i, show how much each agent j's Frobenius norm influences agent i over time.
    """
    N = len(frob_norms_matrix_history[0])  # number of agents
    T_attacked = len(frob_norms_matrix_history)     # number of timesteps in attacked scenario
    T_normal = len(frob_norms_matrix_history_normal)  # number of timesteps in normal scenario
    
    # Create 2*N subplots (two rows: normal and attacked scenarios)
    max_per_row = 3
    cols = min(N, max_per_row)
    rows = 2  # Two rows: normal and attacked scenarios
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows + 1))
    if N == 1:
        axes = axes.reshape(rows, 1)
    
    fig.suptitle(f'Frobenius Norm Influences: ||H_{{i,j}}||_F (Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.96)
    
    # Use consistent agent colors
    agent_colors = get_agent_colors(N)
    
    # Track legend elements to avoid duplicates
    legend_elements = []
    legend_labels = []
    attacked_region_element = None
    
    scenarios = [
        (frob_norms_matrix_history_normal, T_normal, "Normal"),
        (frob_norms_matrix_history, T_attacked, "Attacked")
    ]
    
    # Calculate global max and min for consistent y-axis scaling
    global_max = 0
    global_min = 0
    for influences_data, T, _ in scenarios:
        for t in range(T):
            for i in range(N):
                for j in range(N):
                    value = influences_data[t][i][j]
                    global_max = max(global_max, value)
                    global_min = min(global_min, value)
    
    # Add row labels for scenarios
    for scenario_idx, (influences_data, T, scenario_name) in enumerate(scenarios):
        # Add scenario label on the left side
        if scenario_idx == 0:
            fig.text(0.02, 0.75, 'Normal\nScenario', fontsize=14, fontweight='bold', 
                    ha='center', va='center', rotation=90)
        else:
            fig.text(0.02, 0.25, 'Attacked\nScenario', fontsize=14, fontweight='bold', 
                    ha='center', va='center', rotation=90)
        
        for i in range(min(N, cols)):  # Only plot up to max_per_row agents
            ax = axes[scenario_idx, i]
            
            # Plot Frobenius norm influences for each agent j on agent i
            for j in range(N):
                influence_series = [influences_data[t][i][j] for t in range(T)]
                timesteps = range(T)
                
                # Self-influence (solid line)
                if i == j:
                    line = ax.plot(timesteps, influence_series, color=agent_colors[j], 
                                    linewidth=2.0, linestyle='-', alpha=1.0)[0]
                else:
                    # Influence from others (dashed line)
                    line = ax.plot(timesteps, influence_series, color=agent_colors[j], 
                                    linewidth=2.0, linestyle='--', alpha=1.0)[0]
                
                # avoid duplicate legend entries (only add for first subplot)
                if scenario_idx == 0 and i == 0:
                    legend_elements.append(line)
                    legend_labels.append(f'Agent {j}')
            
            # Mark attacked region (only for attacked scenario)
            if scenario_idx == 1 and attacked_steps:
                start = min(attacked_steps)
                end = max(attacked_steps)
                if i == atk_agent_id:
                    region = ax.axvspan(start, end, color='red', alpha=0.1)
                    if attacked_region_element is None:  # Store for later addition to legend
                        attacked_region_element = region
                else:
                    ax.axvspan(start, end, color='orange', alpha=0.05)
            
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Frobenius Norm Influence Magnitude')
            ax.set_title(f'Frob Norm Influences on Agent {i}')
            ax.grid(True, alpha=0.3)
            
            # Set consistent y-axis limits across all plots
            y_margin = (global_max - global_min) * 0.1
            ax.set_ylim(global_min - y_margin, global_max + y_margin)
    
    # Hide unused subplots
    for scenario_idx in range(rows):
        for j in range(N, cols):
            if j < cols:
                axes[scenario_idx, j].set_visible(False)
    
    # Add attacked region to legend at the end if it exists
    if attacked_region_element is not None:
        legend_elements.append(attacked_region_element)
        legend_labels.append('Attacked Region')
    
    # Create single legend at the bottom
    fig.legend(legend_elements, legend_labels, loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), ncol=min(len(legend_labels), 6), 
               fontsize=9, frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout to make room for legend and row labels
    plt.tight_layout(rect=[0.04, 0.08, 1, 0.94])
    
    out_path = os.path.join(logdir, f'frob_norm_influences_attacked_{atk_agent_id}.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved Frobenius norm influences plot to {out_path}")
