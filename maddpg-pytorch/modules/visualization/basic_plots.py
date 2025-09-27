"""
Basic plotting functions for detection statistics visualization.
"""
import os
import math
import matplotlib.pyplot as plt
from .utils import get_agent_colors
from modules.constants import K_SIGMA


def plot_results(results_attacked, attacked_steps, atk_agent_id, ref_vals, ref_std_devs, logdir, detection_method='mean_std'):
    """Plot Taylor error results for attacked scenario."""
    n = len(results_attacked[0])  # number of agents
    t = len(results_attacked)     # number of time steps
    
    # Create n subplots in a row
    max_per_row = 3
    rows = math.ceil(n / max_per_row)
    cols = min(n, max_per_row)
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten()  # so you can index axes[i] easily
    fig.suptitle(f'Taylor Error ({detection_method.upper().replace("_", "+")} | Worst Action Attack | Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.95)
    
    # Ensure axes is iterable even for single agent case
    if n == 1:
        axes = [axes]
    
    for i in range(n):
        ax = axes[i]
        
        # Extract time series for agent i
        attacked_series = [results_attacked[t][i] for t in range(len(results_attacked))]
        
        # For 'diff' detection method, plot the differences instead of raw values
        if detection_method == 'diff':
            # Calculate differences for plotting (skip first timestep as it has no previous value)
            diff_series = []
            for t in range(1, len(attacked_series)):
                diff = attacked_series[t] - attacked_series[t-1]
                diff_series.append(diff)
            
            # Update series to plot differences
            attacked_series = diff_series
            steps_length = len(attacked_series)
            steps = range(1, steps_length + 1)  # Start from timestep 1
        else:
            # Plot the curves normally
            steps_length = len(attacked_series)
            steps = range(steps_length)
        ref_vals[i] = ref_vals[i][:steps_length]
        ref_std_devs[i] = ref_std_devs[i][:steps_length]

        # Add green region using ref_vals and ref_std_devs
        ref_lower = [ref_vals[i][t] - K_SIGMA*ref_std_devs[i][t] for t in range(len(ref_vals[i]))]
        ref_upper = [ref_vals[i][t] + K_SIGMA*ref_std_devs[i][t] for t in range(len(ref_vals[i]))]
        ax.fill_between(steps, ref_lower, ref_upper, alpha=0.1, color='green')
        
        ax.plot(steps, attacked_series, 'r-', label='Observed', linewidth=2)
        ax.plot(steps, ref_vals[i], 'g--', label='Reference', linewidth=2)
        
        # Mark attacked timesteps with vertical lines
        if i == atk_agent_id and attacked_steps:
            # for attack_step in attacked_steps:
            #     ax.axvline(x=attack_step, color='orange', linestyle=':', alpha=0.5, linewidth=1.5)
            # # Add legend entry for attack markers
            # ax.axvline(x=attacked_steps[0], color='orange', linestyle=':', alpha=0.5, linewidth=1.5, label='Attacked Steps')
            start = min(attacked_steps)
            end = max(attacked_steps)
            ax.axvspan(start, end, color='red', alpha=0.1, label='Attacked Region')
        
        ax.set_xlabel('Step')
        if detection_method == 'diff':
            ax.set_ylabel('Taylor Error Difference')
        else:
            ax.set_ylabel('Taylor Delta Error')
        ax.set_title(f'Agent {i}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # hide the unused axes
    for j in range(n, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(logdir, f'plot_analysis_{detection_method}_attacked_{atk_agent_id}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved analysis plot to {logdir}")


def plot_frobs(frobs_normal, frobs_atk, attacked_steps, atk_agent_id, logdir):
    """Plot Frobenius norms comparison between normal and attacked scenarios."""
    n = len(frobs_normal[0])  # number of agents
    t = len(frobs_normal)     # number of time steps
    
    # Create n subplots in a row
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    fig.suptitle(f'Frobenius Norms (Worst Action Attack | Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.95)
    
    # Ensure axes is iterable even for single agent case
    if n == 1:
        axes = [axes]
    
    for i in range(n):
        ax = axes[i]
        
        # Extract time series for agent i
        normal_series = [frobs_normal[t][i] for t in range(len(frobs_normal))]
        attacked_series = [frobs_atk[t][i] for t in range(len(frobs_atk))]
        
        # Plot the curves
        steps = range(len(normal_series))
        ax.plot(steps, attacked_series, 'r-', label='Under Attack', linewidth=2)
        ax.plot(steps, normal_series, 'g-', label='Normal', linewidth=2)
        
        # Mark attacked timesteps with vertical lines
        if attacked_steps:
            for attack_step in attacked_steps:
                ax.axvline(x=attack_step, color='orange', linestyle=':', alpha=0.5, linewidth=1.5)
            # Add legend entry for attack markers
            ax.axvline(x=attacked_steps[0], color='orange', linestyle=':', alpha=0.5, linewidth=1.5, label='Attacked Steps')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Frobenius Norm')
        ax.set_title(f'Agent {i}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(logdir, f'plot_frobs_attacked_{atk_agent_id}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved frobenius norms plot to {logdir}")


def plot_sec_dir_derivatives(s_dir_derv_normal, s_dir_derv_atk, attacked_steps, atk_agent_id, logdir):
    n = len(s_dir_derv_normal[0])  # number of agents
    t = len(s_dir_derv_normal)     # number of time steps
    
    # Create n subplots in a row
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    fig.suptitle(f'2nd Ord. Dir. Derivatives (Worst Action Attack | Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.95)
    
    # Ensure axes is iterable even for single agent case
    if n == 1:
        axes = [axes]
    
    for i in range(n):
        ax = axes[i]
        
        # Extract time series for agent i
        normal_series = [s_dir_derv_normal[t][i] for t in range(len(s_dir_derv_normal))]
        attacked_series = [s_dir_derv_atk[t][i] for t in range(len(s_dir_derv_atk))]
        
        # Plot the curves
        normal_steps = range(len(normal_series))
        attacked_steps = range(len(attacked_series))

        ax.plot(normal_steps, normal_series, 'g-', label='Normal', linewidth=2)
        ax.plot(attacked_steps, attacked_series, 'r-', label='Under Attack', linewidth=2)
        
        # Highlight region under y < 0 in red
        y_min = min(min(normal_series), min(attacked_series))
        if y_min < 0:
            ax.axhspan(y_min * 1.1, 0, alpha=0.2, color='red')
        
        # Mark attacked timesteps with vertical lines
        if attacked_steps:
            for attack_step in attacked_steps:
                ax.axvline(x=attack_step, color='orange', linestyle=':', alpha=0.5, linewidth=1.5)
            # Add legend entry for attack markers
            ax.axvline(x=attacked_steps[0], color='orange', linestyle=':', alpha=0.5, linewidth=1.5, label='Attacked Steps')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('2nd Ord. Dir. Derivative')
        ax.set_title(f'Agent {i}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(logdir, f'plot_sec_dir_derivatives_attacked_{atk_agent_id}.png'), dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved 2nd ord. dir. derivatives plot to {logdir}")