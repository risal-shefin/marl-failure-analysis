"""
Stacked and complex timeline visualization functions.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from .utils import get_agent_colors
from matplotlib.patches import Patch


def plot_fault_timeline_action_influences_stacked(fault_timeline, action_influences_matrix_history, 
                                                 total_agents, logdir):
    """
    Plot fault timeline with stacked bar charts for action influence contributors.
    Each fault event shows the action influences from other agents as contributors in a stacked bar format.
    Additionally flags ALL timesteps where faulty agents are among top-k influencers on non-faulty agents.
    """
    if len(fault_timeline) == 0:
        print("No faults detected; skipping stacked action influences fault timeline plot.")
        return

    # Parameters for top-k influence detection
    k_top = 2  # Look for faulty agents in top-2 influencers
    
    # Create a mapping of when each agent was first detected as faulty
    fault_detection_times = {}  # agent_id -> timestep when first detected as faulty
    for event in fault_timeline:
        if event['agent'] not in fault_detection_times:
            fault_detection_times[event['agent']] = event['t']
    
    # Find the last fault detection timestep to stop flagging after this point
    last_fault_detection_time = max(fault_detection_times.values()) if fault_detection_times else -1
    
    # Find the first fault detection timestep for mean calculation
    first_fault_detection_time = min(fault_detection_times.values()) if fault_detection_times else -1
    
    # Find the first faulty agent (patient zero) - the one detected earliest
    first_faulty_agent = None
    if fault_detection_times:
        first_faulty_agent = min(fault_detection_times.keys(), key=lambda agent: fault_detection_times[agent])
    
    # Create extended timeline with additional flagged timesteps
    extended_timeline = []
    
    # Add original fault detection events with exact timestep action influences
    for event in fault_timeline:
        faulty_agent = event['agent']
        fault_timestep = event['t']
        
        # Get action influences at the exact fault timestep
        if fault_timestep < len(action_influences_matrix_history):
            action_influences = action_influences_matrix_history[fault_timestep][faulty_agent]
            
            # Create contributors dict from action influences (include all agents including self)
            contribs = {}
            for j in range(total_agents):
                contribs[j] = abs(action_influences[j])  # Use absolute value of influence
        else:
            contribs = {}
        
        extended_timeline.append({
            'type': 'fault_detection',
            'agent': event['agent'],
            't': event['t'],
            'contribs': contribs,
            'description': f"Faulty agent {event['agent']}"
        })
    
    # Get timesteps that already have fault detection events
    fault_detection_timesteps = set(event['t'] for event in fault_timeline)
    
    # Find ALL timesteps where faulty agents are top-k influencers on non-faulty agents
    # Only check timesteps up to (but not including) the last fault detection
    # Skip timesteps that already have fault detection events
    for t in range(min(len(action_influences_matrix_history), last_fault_detection_time)):
        # Skip if this timestep already has a fault detection event
        if t in fault_detection_timesteps:
            continue
            
        influences_at_t = action_influences_matrix_history[t]
        
        # Get the set of agents that are considered faulty at timestep t
        faulty_agents_at_t = set()
        for agent_id, detection_time in fault_detection_times.items():
            if t >= detection_time:  # Only consider agent faulty from detection time onwards
                faulty_agents_at_t.add(agent_id)
        
        # Skip if no agents are faulty at this timestep
        if not faulty_agents_at_t:
            continue
        
        # For each non-faulty agent at timestep t, check if any faulty agent is in top-k influencers
        for non_faulty_agent in range(total_agents):
            # Check if this agent is faulty at timestep t
            is_faulty_at_t = non_faulty_agent in faulty_agents_at_t
            if is_faulty_at_t:
                continue  # Skip agents that are faulty already
                
            # Get influences on this non-faulty agent and rank them
            agent_influences = [(j, abs(influences_at_t[non_faulty_agent][j])) for j in range(total_agents)]
            # Sort by influence magnitude (descending)
            ranked_influences = sorted(agent_influences, key=lambda x: x[1], reverse=True)
            
            # Check if any faulty agent is in top-k
            top_k_agents = [agent_id for agent_id, _ in ranked_influences[:k_top]]
            faulty_in_top_k = [agent_id for agent_id in top_k_agents if agent_id in faulty_agents_at_t]
            
            if faulty_in_top_k:
                # Include ALL such timesteps, not just first occurrence per (faulty_agent, target_agent) pair
                # Use exact timestep action influences for top-k influence events
                if t < len(action_influences_matrix_history):
                    action_influences = action_influences_matrix_history[t][non_faulty_agent]
                    
                    # Create contributors dict from action influences (include all agents including self)
                    contribs = {}
                    for j in range(total_agents):
                        contribs[j] = abs(action_influences[j])  # Use absolute value of influence
                else:
                    contribs = {}
                
                faulty_list = ', '.join(map(str, faulty_in_top_k))
                description_lines = [
                    "Faulty",
                    "Influence", 
                    f"A{faulty_list}→A{non_faulty_agent}"
                ]
                extended_timeline.append({
                    'type': 'top_k_influence',
                    'agent': non_faulty_agent,  # The affected agent
                    'faulty_influencers': faulty_in_top_k,
                    't': t,
                    'contribs': contribs,
                    'target_agent': non_faulty_agent,
                    "description": "\n".join(description_lines)
                })
    
    # Sort extended timeline by timestep
    extended_timeline.sort(key=lambda x: x['t'])
    
    if len(extended_timeline) == 0:
        print("No events to display in stacked action influences fault timeline.")
        return
    
    k = len(extended_timeline)
    fig = plt.figure(figsize=(max(12, 0.7 * k + 4), 6))
    
    # Create a grid layout with tighter spacing
    gs = fig.add_gridspec(2, 1, height_ratios=[0.8, 2.5], hspace=0.1)  # Reduced hspace from default
    ax_timeline = fig.add_subplot(gs[0, 0])  # Timeline on top
    ax_main = fig.add_subplot(gs[1, 0])  # Main stacked bar chart

    agent_colors = get_agent_colors(total_agents)

    # --- Timeline axis (top) ---
    ax_timeline.axis('off')

    if k > 0:
        # Get timesteps for x-axis positioning
        timesteps = [event['t'] for event in extended_timeline]
        min_t, max_t = min(timesteps), max(timesteps)
        
        # Draw timeline arrow
        arrow_y = 0.5
        ax_timeline.annotate(
            '', xy=(0.95, arrow_y), xytext=(0.05, arrow_y),
            arrowprops=dict(arrowstyle='-|>', lw=2, color='gray'),
            xycoords='axes fraction', textcoords='axes fraction'
        )

        # Milestones
        for i, event in enumerate(extended_timeline):
            # Map event index to x position (0.1 to 0.9 range) to handle multiple events at same timestep
            frac_x = (i + 0.5) / k

            # Different markers for different event types
            if event['type'] == 'fault_detection':
                marker_color = 'darkred'
                marker_size = 12
            else:  # top_k_influence
                marker_color = 'orange'
                marker_size = 10

            # Circle marker
            ax_timeline.plot(frac_x, arrow_y, 'o', color=marker_color, markersize=marker_size, 
                            transform=ax_timeline.transAxes)

            # Event description above (with better line wrapping)
            description = event['description']
            # Check if description already has newlines (properly formatted)
            if '\n' in description:
                # Keep the existing newlines - don't override them
                pass
            elif len(description) > 20:  # Only wrap if no newlines and too long
                words = description.split()
                lines = []
                current_line = []
                for word in words:
                    if len(' '.join(current_line + [word])) <= 20:
                        current_line.append(word)
                    else:
                        if current_line:
                            lines.append(' '.join(current_line))
                            current_line = [word]
                        else:
                            lines.append(word)
                if current_line:
                    lines.append(' '.join(current_line))
                description = '\n'.join(lines)

            ax_timeline.text(frac_x, arrow_y + 0.1,
                             description,
                             ha='center', va='bottom',
                             fontsize=8, fontweight='bold',
                             transform=ax_timeline.transAxes)

            # Timestep label below
            ax_timeline.text(frac_x, arrow_y - 0.1,
                             f"t = {event['t']}",
                             ha='center', va='top',
                             fontsize=10, color='darkblue',
                             transform=ax_timeline.transAxes)

    # --- Main stacked bar chart (bottom) ---
    if k == 0:
        ax_main.text(0.5, 0.5, 'No events to display',
                     ha='center', va='center', fontsize=12, style='italic',
                     transform=ax_main.transAxes)
        ax_main.axis('off')
    else:
        # Use index-based positioning to handle multiple events at same timestep
        bar_width = 0.6  # Decreased bar width to 0.6
        
        # Plot stacked bars for each event (using index-based positioning)
        for i, event in enumerate(extended_timeline):
            contribs = event.get('contribs', {})
            
            # Check if this is the first faulty agent (patient zero) and a fault detection event
            if (event['type'] == 'fault_detection' and 
                event['agent'] == first_faulty_agent):
                # Patient Zero - write vertically to avoid overlap
                ax_main.text(i, 0.5, 'P\na\nt\ni\ne\nn\nt\n \nZ\ne\nr\no',
                            ha='center', va='center', fontsize=10, fontweight='bold', 
                            style='italic', color='darkred', rotation=0)
                
                # Add a subtle background bar to indicate the position
                ax_main.bar(i, 1, bar_width, color='lightgray', alpha=0.3, 
                           edgecolor='darkred', linewidth=2)
                
            elif len(contribs) == 0:
                # No data case
                ax_main.text(i, 0.5, 'No\nData',
                            ha='center', va='center', fontsize=9, style='italic',
                            color='gray')
                ax_main.bar(i, 1, bar_width, color='lightgray', alpha=0.3)
                
            else:
                # Prepare data for stacked bar chart - ensure no duplicates
                # Sort by agent ID for consistent ordering
                sorted_agents = sorted(contribs.keys())
                influences = [contribs[agent_id] for agent_id in sorted_agents]
                
                # Normalize influences to sum to 1 for percentage representation
                total_influence = sum(influences)
                if total_influence > 0:
                    influences = [influence / total_influence for influence in influences]
                
                colors = [agent_colors[agent_id] for agent_id in sorted_agents]
                
                # Create stacked bar chart with proper tracking
                bottom = 0
                labeled_segments = []  # Track which segments get labels to avoid overlap
                
                for agent_id, influence, color in zip(sorted_agents, influences, colors):
                    bar = ax_main.bar(i, influence, bar_width, bottom=bottom, 
                                        color=color, edgecolor='white', linewidth=0.5)
                    
                    # Store segment info for potential labeling
                    labeled_segments.append({
                        'agent_id': agent_id,
                        'influence': influence,
                        'bottom': bottom,
                        'height': influence
                    })
                    
                    bottom += influence

                for seg in labeled_segments:
                    label_y = seg['bottom'] + seg['height'] / 2
                    ax_main.text(i, label_y, f'{seg["influence"]*100:.0f}%', 
                                ha='center', va='center', fontsize=8, fontweight='bold',
                                color='white')

            # Add title above each bar with better formatting
            title_parts = f"Influence on\nAgent {event['agent']}".split('\n')
            ax_main.text(i, 1.02, '\n'.join(title_parts),
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Customize the main plot
        ax_main.set_xlim(-0.3, k - 0.7)  # Reduced margins to decrease gaps between bars
        ax_main.set_ylim(0, 1)
        ax_main.set_ylabel('Influence', fontsize=12, fontweight='bold')
        
        # Set x-ticks to show event indices with timestep info
        ax_main.set_xticks(range(k))
        x_labels = []
        for i, event in enumerate(extended_timeline):
            # Create labels that show both index and timestep
            label = f"t={event['t']}"
            x_labels.append(label)
        ax_main.set_xticklabels(x_labels, fontsize=9, ha='center')
        
        # Add grid for better readability
        ax_main.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax_main.grid(True, axis='x', alpha=0.2, linestyle=':')

    # Create legend at the bottom
    legend_elements = [Patch(facecolor=agent_colors[i], label=f"Agent {i}") for i in range(total_agents)]
    
    # Add legend for event types
    fault_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkred', 
                             markersize=10, label='Fault Detection')
    influence_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                                 markersize=10, label=f'Vulnerable Top-{k_top} Influence')
    legend_elements.extend([fault_marker, influence_marker])
    
    fig.legend(handles=legend_elements, loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), ncol=min(len(legend_elements), 10),
               fontsize=9, frameon=True, fancybox=True, shadow=True)

    fig.suptitle('Fault Detection Timeline with Action Influence Analysis',
                 fontsize=16, fontweight='bold', y=0.98)  # Moved title up slightly

    # Adjust layout to make room for legend and reduce margins
    plt.tight_layout(rect=[0.05, 0.08, 1, 0.96])  # Adjusted top margin

    out_path = os.path.join(logdir, 'fault_timeline_action_influences_stacked.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved stacked action influences fault timeline plot to {out_path}")
    print(f"Stacked timeline includes {len([e for e in extended_timeline if e['type'] == 'fault_detection'])} fault detections and {len([e for e in extended_timeline if e['type'] == 'top_k_influence'])} top-{k_top} influence events")
    
    # Return timesteps and agents for comparison plotting
    timesteps = [event['t'] for event in extended_timeline]
    agents = [event['agent'] for event in extended_timeline]
    return timesteps, agents


def plot_normal_scenario_action_influences_stacked(timesteps, agents, action_influences_matrix_history_normal, 
                                                  total_agents, logdir):
    """
    Plot stacked bar charts for action influence in the normal (unattacked) scenario.
    Uses the same timesteps and agents as the fault detection timeline for direct comparison.
    
    Args:
        timesteps: List of timesteps to plot (from fault detection timeline)
        agents: List of agents to show influences for (from fault detection timeline)
        action_influences_matrix_history_normal: Normal scenario action influences data
        total_agents: Total number of agents
        logdir: Directory to save the plot
    """
    if len(timesteps) == 0:
        print("No timesteps provided; skipping normal scenario stacked action influences plot.")
        return

    k = len(timesteps)
    fig = plt.figure(figsize=(min(12, 0.7 * k + 4), 5))  # Reduced base width and increased coefficient for tighter spacing
    
    # Create a single plot for just the stacked bar chart
    ax_main = plt.subplot(1, 1, 1)  # Single plot without timeline

    agent_colors = get_agent_colors(total_agents)

    # --- Main stacked bar chart ---
    if k == 0:
        ax_main.text(0.5, 0.5, 'No events to display',
                     ha='center', va='center', fontsize=12, style='italic',
                     transform=ax_main.transAxes)
        ax_main.axis('off')
    else:
        # Use index-based positioning to handle multiple events at same timestep
        bar_width = 0.6  # Decreased bar width to 0.6
        
        # Plot stacked bars for each event (using index-based positioning)
        for i, (timestep, agent) in enumerate(zip(timesteps, agents)):
            # Get action influences at the specified timestep (if available)
            if timestep < len(action_influences_matrix_history_normal):
                action_influences = action_influences_matrix_history_normal[timestep][agent]
                
                # Create contributors dict from action influences (include all agents including self)
                contribs = {}
                for j in range(total_agents):
                    contribs[j] = abs(action_influences[j])  # Use absolute value of influence
            else:
                contribs = {}

            if len(contribs) == 0:
                # No data case
                ax_main.text(i, 0.5, 'No\nData',
                            ha='center', va='center', fontsize=9, style='italic',
                            color='gray')
                ax_main.bar(i, 1, bar_width, color='lightgray', alpha=0.3)
                
            else:
                # Prepare data for stacked bar chart - ensure no duplicates
                # Sort by agent ID for consistent ordering
                sorted_agents = sorted(contribs.keys())
                influences = [contribs[agent_id] for agent_id in sorted_agents]
                
                # Normalize influences to sum to 1 for percentage representation
                total_influence = sum(influences)
                if total_influence > 0:
                    influences = [influence / total_influence for influence in influences]
                
                colors = [agent_colors[agent_id] for agent_id in sorted_agents]
                
                # Create stacked bar chart with proper tracking
                bottom = 0
                labeled_segments = []  # Track which segments get labels to avoid overlap
                
                for agent_id, influence, color in zip(sorted_agents, influences, colors):
                    bar = ax_main.bar(i, influence, bar_width, bottom=bottom, 
                                        color=color, edgecolor='white', linewidth=0.5)
                    
                    # Store segment info for potential labeling
                    labeled_segments.append({
                        'agent_id': agent_id,
                        'influence': influence,
                        'bottom': bottom,
                        'height': influence
                    })
                    
                    bottom += influence

                for seg in labeled_segments:
                    label_y = seg['bottom'] + seg['height'] / 2
                    ax_main.text(i, label_y, f'{seg["influence"]*100:.0f}%', 
                                ha='center', va='center', fontsize=8, fontweight='bold',
                                color='white')

            # Add title above each bar with better formatting
            title_parts = f"Influence on\nAgent {agent}".split('\n')
            ax_main.text(i, 1.02, '\n'.join(title_parts),
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Customize the main plot
        ax_main.set_xlim(-0.5, k - 0.5)  # Restore proper margins to show all bars
        ax_main.set_ylim(0, 1)
        ax_main.set_ylabel('Influence', fontsize=12, fontweight='bold')
        
        # Set x-ticks to show event indices with timestep info
        ax_main.set_xticks(range(k))
        x_labels = []
        for i, timestep in enumerate(timesteps):
            # Create labels that show both index and timestep
            label = f"t={timestep}"
            x_labels.append(label)
        ax_main.set_xticklabels(x_labels, fontsize=9, ha='center')
        
        # Add grid for better readability
        ax_main.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax_main.grid(True, axis='x', alpha=0.2, linestyle=':')

    # Create legend at the bottom
    legend_elements = [Patch(facecolor=agent_colors[i], label=f"Agent {i}") for i in range(total_agents)]
    
    fig.legend(handles=legend_elements, loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), ncol=min(len(legend_elements), 10),
               fontsize=9, frameon=True, fancybox=True, shadow=True)

    fig.suptitle('Normal Scenario: Action Influence Analysis',
                 fontsize=16, fontweight='bold', y=0.95)

    # Adjust layout to make room for legend
    plt.tight_layout(rect=[0.05, 0.08, 1, 0.92])

    out_path = os.path.join(logdir, 'normal_scenario_action_influences_stacked.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved normal scenario stacked action influences plot to {out_path}")
    print(f"Normal scenario plot includes {k} timesteps corresponding to fault detection events")


def plot_normal_scenario_frob_norms_stacked(timesteps, agents, frob_norms_matrix_history_normal, 
                                           total_agents, logdir):
    """
    Plot stacked bar charts for Frobenius norm influences in the normal (unattacked) scenario.
    Uses the same timesteps and agents as the fault detection timeline for direct comparison.
    
    Args:
        timesteps: List of timesteps to plot (from fault detection timeline)
        agents: List of agents to show Frobenius norm influences for (from fault detection timeline)
        frob_norms_matrix_history_normal: Normal scenario Frobenius norm matrices data
        total_agents: Total number of agents
        logdir: Directory to save the plot
    """
    if len(timesteps) == 0:
        print("No timesteps provided; skipping normal scenario stacked Frobenius norms plot.")
        return

    k = len(timesteps)
    fig = plt.figure(figsize=(min(12, 0.7 * k + 4), 5))  # Reduced base width and increased coefficient for tighter spacing
    
    # Create a single plot for just the stacked bar chart
    ax_main = plt.subplot(1, 1, 1)  # Single plot without timeline

    agent_colors = get_agent_colors(total_agents)

    # --- Main stacked bar chart ---
    if k == 0:
        ax_main.text(0.5, 0.5, 'No events to display',
                     ha='center', va='center', fontsize=12, style='italic',
                     transform=ax_main.transAxes)
        ax_main.axis('off')
    else:
        # Use index-based positioning to handle multiple events at same timestep
        bar_width = 0.6  # Decreased bar width to 0.6
        
        # Plot stacked bars for each event (using index-based positioning)
        for i, (timestep, agent) in enumerate(zip(timesteps, agents)):
            # Get Frobenius norm matrix at the specified timestep (if available)
            if timestep < len(frob_norms_matrix_history_normal):
                frob_matrix = frob_norms_matrix_history_normal[timestep]
                
                # Create contributors dict from Frobenius norm influences (include all agents including self)
                contribs = {}
                for j in range(total_agents):
                    contribs[j] = abs(frob_matrix[agent][j])  # Use absolute value of Frobenius norm influence
            else:
                contribs = {}

            if len(contribs) == 0:
                # No data case
                ax_main.text(i, 0.5, 'No\nData',
                            ha='center', va='center', fontsize=9, style='italic',
                            color='gray')
                ax_main.bar(i, 1, bar_width, color='lightgray', alpha=0.3)
                
            else:
                # Prepare data for stacked bar chart - ensure no duplicates
                # Sort by agent ID for consistent ordering
                sorted_agents = sorted(contribs.keys())
                influences = [contribs[agent_id] for agent_id in sorted_agents]
                
                # Normalize influences to sum to 1 for percentage representation
                total_influence = sum(influences)
                if total_influence > 0:
                    influences = [influence / total_influence for influence in influences]
                
                colors = [agent_colors[agent_id] for agent_id in sorted_agents]
                
                # Create stacked bar chart with proper tracking
                bottom = 0
                labeled_segments = []  # Track which segments get labels to avoid overlap
                
                for agent_id, influence, color in zip(sorted_agents, influences, colors):
                    bar = ax_main.bar(i, influence, bar_width, bottom=bottom, 
                                        color=color, edgecolor='white', linewidth=0.5)
                    
                    # Store segment info for potential labeling
                    labeled_segments.append({
                        'agent_id': agent_id,
                        'influence': influence,
                        'bottom': bottom,
                        'height': influence
                    })
                    
                    bottom += influence

                for seg in labeled_segments:
                    label_y = seg['bottom'] + seg['height'] / 2
                    ax_main.text(i, label_y, f'{seg["influence"]*100:.0f}%', 
                                ha='center', va='center', fontsize=8, fontweight='bold',
                                color='white')

            # Add title above each bar with better formatting
            title_parts = f"Frob Influence\nAgent {agent}".split('\n')
            ax_main.text(i, 1.02, '\n'.join(title_parts),
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Customize the main plot
        ax_main.set_xlim(-0.5, k - 0.5)  # Restore proper margins to show all bars
        ax_main.set_ylim(0, 1)
        ax_main.set_ylabel('Frobenius Norm Influence', fontsize=12, fontweight='bold')
        
        # Set x-ticks to show event indices with timestep info
        ax_main.set_xticks(range(k))
        x_labels = []
        for i, timestep in enumerate(timesteps):
            # Create labels that show both index and timestep
            label = f"t={timestep}"
            x_labels.append(label)
        ax_main.set_xticklabels(x_labels, fontsize=9, ha='center')
        
        # Add grid for better readability
        ax_main.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax_main.grid(True, axis='x', alpha=0.2, linestyle=':')

    # Create legend at the bottom
    legend_elements = [Patch(facecolor=agent_colors[i], label=f"Agent {i}") for i in range(total_agents)]
    
    fig.legend(handles=legend_elements, loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), ncol=min(len(legend_elements), 10),
               fontsize=9, frameon=True, fancybox=True, shadow=True)

    fig.suptitle('Normal Scenario: Frobenius Norm Influence Analysis',
                 fontsize=16, fontweight='bold', y=0.95)

    # Adjust layout to make room for legend
    plt.tight_layout(rect=[0.05, 0.08, 1, 0.92])

    out_path = os.path.join(logdir, 'normal_scenario_frob_norms_stacked.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved normal scenario stacked Frobenius norms plot to {out_path}")
    print(f"Normal scenario Frobenius norms plot includes {k} timesteps corresponding to fault detection events")


def plot_attacked_scenario_frob_norms_stacked(timesteps, agents, frob_norms_matrix_history, 
                                             total_agents, logdir):
    """
    Plot stacked bar charts for Frobenius norm influences in the attacked scenario.
    Uses the same timesteps and agents as the fault detection timeline for direct comparison.
    
    Args:
        timesteps: List of timesteps to plot (from fault detection timeline)
        agents: List of agents to show Frobenius norm influences for (from fault detection timeline)
        frob_norms_matrix_history: Attacked scenario Frobenius norm matrices data
        total_agents: Total number of agents
        logdir: Directory to save the plot
    """
    if len(timesteps) == 0:
        print("No timesteps provided; skipping attacked scenario stacked Frobenius norms plot.")
        return

    k = len(timesteps)
    fig = plt.figure(figsize=(min(12, 0.7 * k + 4), 5))  # Reduced base width and increased coefficient for tighter spacing
    
    # Create a single plot for just the stacked bar chart
    ax_main = plt.subplot(1, 1, 1)  # Single plot without timeline

    agent_colors = get_agent_colors(total_agents)

    # --- Main stacked bar chart ---
    if k == 0:
        ax_main.text(0.5, 0.5, 'No events to display',
                     ha='center', va='center', fontsize=12, style='italic',
                     transform=ax_main.transAxes)
        ax_main.axis('off')
    else:
        # Use index-based positioning to handle multiple events at same timestep
        bar_width = 0.6  # Decreased bar width to 0.6
        
        # Plot stacked bars for each event (using index-based positioning)
        for i, (timestep, agent) in enumerate(zip(timesteps, agents)):
            # Get Frobenius norm matrix at the specified timestep (if available)
            if timestep < len(frob_norms_matrix_history):
                frob_matrix = frob_norms_matrix_history[timestep]
                
                # Create contributors dict from Frobenius norm influences (include all agents including self)
                contribs = {}
                for j in range(total_agents):
                    contribs[j] = abs(frob_matrix[agent][j])  # Use absolute value of Frobenius norm influence
            else:
                contribs = {}

            if len(contribs) == 0:
                # No data case
                ax_main.text(i, 0.5, 'No\nData',
                            ha='center', va='center', fontsize=9, style='italic',
                            color='gray')
                ax_main.bar(i, 1, bar_width, color='lightgray', alpha=0.3)
                
            else:
                # Prepare data for stacked bar chart - ensure no duplicates
                # Sort by agent ID for consistent ordering
                sorted_agents = sorted(contribs.keys())
                influences = [contribs[agent_id] for agent_id in sorted_agents]
                
                # Normalize influences to sum to 1 for percentage representation
                total_influence = sum(influences)
                if total_influence > 0:
                    influences = [influence / total_influence for influence in influences]
                
                colors = [agent_colors[agent_id] for agent_id in sorted_agents]
                
                # Create stacked bar chart with proper tracking
                bottom = 0
                labeled_segments = []  # Track which segments get labels to avoid overlap
                
                for agent_id, influence, color in zip(sorted_agents, influences, colors):
                    bar = ax_main.bar(i, influence, bar_width, bottom=bottom, 
                                        color=color, edgecolor='white', linewidth=0.5)
                    
                    # Store segment info for potential labeling
                    labeled_segments.append({
                        'agent_id': agent_id,
                        'influence': influence,
                        'bottom': bottom,
                        'height': influence
                    })
                    
                    bottom += influence

                for seg in labeled_segments:
                    label_y = seg['bottom'] + seg['height'] / 2
                    ax_main.text(i, label_y, f'{seg["influence"]*100:.0f}%', 
                                ha='center', va='center', fontsize=8, fontweight='bold',
                                color='white')

            # Add title above each bar with better formatting
            title_parts = f"Frob Influence\nAgent {agent}".split('\n')
            ax_main.text(i, 1.02, '\n'.join(title_parts),
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Customize the main plot
        ax_main.set_xlim(-0.5, k - 0.5)  # Restore proper margins to show all bars
        ax_main.set_ylim(0, 1)
        ax_main.set_ylabel('Frobenius Norm Influence', fontsize=12, fontweight='bold')
        
        # Set x-ticks to show event indices with timestep info
        ax_main.set_xticks(range(k))
        x_labels = []
        for i, timestep in enumerate(timesteps):
            # Create labels that show both index and timestep
            label = f"t={timestep}"
            x_labels.append(label)
        ax_main.set_xticklabels(x_labels, fontsize=9, ha='center')
        
        # Add grid for better readability
        ax_main.grid(True, axis='y', alpha=0.3, linestyle='--')
        ax_main.grid(True, axis='x', alpha=0.2, linestyle=':')

    # Create legend at the bottom
    legend_elements = [Patch(facecolor=agent_colors[i], label=f"Agent {i}") for i in range(total_agents)]
    
    fig.legend(handles=legend_elements, loc='lower center', 
               bbox_to_anchor=(0.5, 0.02), ncol=min(len(legend_elements), 10),
               fontsize=9, frameon=True, fancybox=True, shadow=True)

    fig.suptitle('Attacked Scenario: Frobenius Norm Influence Analysis',
                 fontsize=16, fontweight='bold', y=0.95)

    # Adjust layout to make room for legend
    plt.tight_layout(rect=[0.05, 0.08, 1, 0.92])

    out_path = os.path.join(logdir, 'attacked_scenario_frob_norms_stacked.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved attacked scenario stacked Frobenius norms plot to {out_path}")
    print(f"Attacked scenario Frobenius norms plot includes {k} timesteps corresponding to fault detection events")
