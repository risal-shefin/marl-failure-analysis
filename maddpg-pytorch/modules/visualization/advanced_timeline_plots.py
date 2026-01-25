"""
Advanced timeline visualization functions for fault detection analysis.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from .utils import get_agent_colors
from matplotlib.patches import Patch

def plot_fault_timeline_action_influences(fault_timeline, action_influences_matrix_history, total_agents, logdir):
    """
    Plot fault timeline with action influence contributors instead of Frobenius norm influences.
    Each fault event shows the action influences from other agents as contributors.
    Additionally flags timesteps where faulty agents are among top-k influencers on non-faulty agents.
    """
    if len(fault_timeline) == 0:
        print("No faults detected; skipping action influences fault timeline plot.")
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
    
    # Track already flagged (faulty_agent, target_agent) pairs to avoid duplicates
    flagged_pairs = set()  # Set of (faulty_agent_id, target_agent_id) tuples
    
    # Add original fault detection events with exact timestep action influences (not mean)
    for event in fault_timeline:
        # Use exact timestep action influences for fault detection events (like original version)
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
            'contribs': contribs,  # Use exact timestep influences, not mean
            'description': f"Faulty agent {event['agent']}"
        })
    
    # Find additional timesteps where faulty agents are top-k influencers on non-faulty agents
    # Only check timesteps up to (but not including) the last fault detection
    for t in range(min(len(action_influences_matrix_history), last_fault_detection_time)):
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
                continue  # Skip agents that are faulty at this timestep
                
            # Get influences on this non-faulty agent and rank them
            agent_influences = [(j, abs(influences_at_t[non_faulty_agent][j])) for j in range(total_agents)]
            # Sort by influence magnitude (descending)
            ranked_influences = sorted(agent_influences, key=lambda x: x[1], reverse=True)
            
            # Check if any faulty agent is in top-k
            top_k_agents = [agent_id for agent_id, _ in ranked_influences[:k_top]]
            faulty_in_top_k = [agent_id for agent_id in top_k_agents if agent_id in faulty_agents_at_t]
            
            if faulty_in_top_k:
                # Check if any of the faulty agents in top-k have already been flagged for this target
                new_faulty_influencers = []
                for faulty_agent in faulty_in_top_k:
                    pair = (faulty_agent, non_faulty_agent)
                    if pair not in flagged_pairs:
                        new_faulty_influencers.append(faulty_agent)
                        flagged_pairs.add(pair)  # Mark this pair as flagged
                
                # Only create an event if there are new faulty influencers to report
                if new_faulty_influencers:
                    # Check if this exact timestep+target combination is already in timeline
                    already_exists = any(event['t'] == t and event.get('target_agent') == non_faulty_agent 
                                       for event in extended_timeline)
                    if not already_exists:
                        # Use exact timestep action influences for top-k influence events (same as fault detection)
                        if t < len(action_influences_matrix_history):
                            action_influences = action_influences_matrix_history[t][non_faulty_agent]
                            
                            # Create contributors dict from action influences (include all agents including self)
                            contribs = {}
                            for j in range(total_agents):
                                contribs[j] = abs(action_influences[j])  # Use absolute value of influence
                        else:
                            contribs = {}
                        
                        faulty_list = ', '.join(map(str, new_faulty_influencers))
                        extended_timeline.append({
                            'type': 'top_k_influence',
                            'agent': non_faulty_agent,  # The affected agent
                            'faulty_influencers': new_faulty_influencers,
                            't': t,
                            'contribs': contribs,  # Use exact timestep influences, same as fault detection
                            'target_agent': non_faulty_agent,
                            "description": f"Faulty agent {faulty_list} is among the top-{k_top} influencers of Agent {non_faulty_agent}"
                        })
    
    # Sort extended timeline by timestep
    extended_timeline.sort(key=lambda x: x['t'])
    
    if len(extended_timeline) == 0:
        print("No events to display in action influences fault timeline.")
        return
    
    k = len(extended_timeline)
    fig = plt.figure(figsize=(max(8, 3*k), 6))  # Increased height for better visibility
    gs = fig.add_gridspec(
        3, k,
        height_ratios=[1.0, 2, 0.1],
        hspace=0.15
    )

    cmap = plt.get_cmap('tab20')
    agent_colors = {i: cmap(i % 20) for i in range(total_agents)}

    # --- Timeline axis (top row) ---
    ax_timeline = fig.add_subplot(gs[0, :])
    ax_timeline.axis('off')

    arrow_y = 0.5
    ax_timeline.annotate(
        '', xy=(1, arrow_y), xytext=(0, arrow_y),
        arrowprops=dict(arrowstyle='-|>', lw=2, color='gray'),
        xycoords='axes fraction', textcoords='axes fraction'
    )

    # Milestones
    for i, event in enumerate(extended_timeline):
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

        # Event description above (with line wrapping for long descriptions)
        description = event['description']
        if len(description) > 25:  # Wrap long descriptions
            words = description.split()
            lines = []
            current_line = []
            for word in words:
                if len(' '.join(current_line + [word])) <= 25:
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

        ax_timeline.text(frac_x, arrow_y + 0.15,
                         description,
                         ha='center', va='bottom',
                         fontsize=9, fontweight='bold',
                         transform=ax_timeline.transAxes)

        # Timestep label below
        ax_timeline.text(frac_x, arrow_y - 0.15,
                         f"t = {event['t']}",
                         ha='center', va='top',
                         fontsize=10, color='darkblue',
                         transform=ax_timeline.transAxes)

    # --- Contributor charts (middle row) ---
    for col, event in enumerate(extended_timeline):
        ax = fig.add_subplot(gs[1, col])
        
        contribs = event.get('contribs', {})

        # Check if this is the first faulty agent (patient zero) and a fault detection event
        if (event['type'] == 'fault_detection' and 
            event['agent'] == first_faulty_agent):
            ax.axis('off')
            ax.text(0.5, 0.5, 'Patient Zero',
                    ha='center', va='center', fontsize=12, fontweight='bold', 
                    style='italic', color='darkred')
        elif len(contribs) == 0:
            ax.axis('off')
            ax.text(0.5, 0.5, 'No Data',
                    ha='center', va='center', fontsize=10, style='italic')
        else:
            vals = np.array(list(contribs.values()), dtype=float)
            if vals.sum() > 0:
                vals /= vals.sum()  # Normalize to sum to 1
            colors = [agent_colors[a] for a in contribs.keys()]

            wedges, _, autotexts = ax.pie(
                vals, autopct='%1.1f%%', startangle=90, colors=colors,
                wedgeprops=dict(width=0.35, edgecolor='w')
            )
            for at in autotexts:
                at.set_fontsize(8)
                at.set_fontweight('bold')
            
            # Different title based on event type
            title = f"Influences on Agent {event['agent']}"
            # if event['type'] == 'fault_detection':
            #     title = 'Contributors to Fault'
            # else:
            #     title = f"Influences on Agent {event['agent']}"
            
            ax.set_title(title, fontsize=10, pad=5)
            ax.set_aspect('equal')

    # --- Legend row (bottom row) ---
    ax_legend = fig.add_subplot(gs[2, :])
    ax_legend.axis('off')
    legend_elements = [Patch(facecolor=agent_colors[i], label=f"Agent {i}") for i in range(total_agents)]
    
    # Add legend for event types
    fault_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkred', 
                             markersize=10, label='Fault Detection')
    influence_marker = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                                 markersize=10, label=f'Vulnerable Top-{k_top} Influence')
    legend_elements.extend([fault_marker, influence_marker])
    
    ax_legend.legend(handles=legend_elements, loc='center', ncol=min(len(legend_elements), 8),
                     fontsize=9, frameon=False)

    fig.suptitle('Fault Detection Timeline with Action Influence Analysis',
                 fontsize=14, fontweight='bold', y=0.96)

    out_path = os.path.join(logdir, 'fault_timeline_action_influences.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved enhanced action influences fault timeline plot to {out_path}")
    print(f"Timeline includes {len([e for e in extended_timeline if e['type'] == 'fault_detection'])} fault detections and {len([e for e in extended_timeline if e['type'] == 'top_k_influence'])} top-{k_top} influence events")


def plot_fault_timeline_second_order_action_influences(fault_timeline, second_order_action_influences_history, 
                                                      total_agents, logdir):
    """
    Plot fault timeline with second-order action influence contributors.
    Each fault event shows the second-order action influences from other agents as contributors.
    """
    if len(fault_timeline) == 0:
        print("No faults detected; skipping second-order action influences fault timeline plot.")
        return

    k = len(fault_timeline)
    fig = plt.figure(figsize=(max(6, 3*k), 5))
    gs = fig.add_gridspec(
        3, k,
        height_ratios=[0.8, 2, 0.1],
        hspace=0.1
    )

    cmap = plt.get_cmap('tab20')
    agent_colors = {i: cmap(i % 20) for i in range(total_agents)}

    # --- Timeline axis (top row) ---
    ax_timeline = fig.add_subplot(gs[0, :])
    ax_timeline.axis('off')

    arrow_y = 0.5
    ax_timeline.annotate(
        '', xy=(1, arrow_y), xytext=(0, arrow_y),
        arrowprops=dict(arrowstyle='-|>', lw=2, color='gray'),
        xycoords='axes fraction', textcoords='axes fraction'
    )

    # Milestones
    for i, event in enumerate(fault_timeline):
        frac_x = (i + 0.5) / k

        # Circle marker
        ax_timeline.plot(frac_x, arrow_y, 'o', color='darkred', markersize=10, transform=ax_timeline.transAxes)

        # Faulty agent label above
        ax_timeline.text(frac_x, arrow_y + 0.15,
                         f"Faulty agent {event['agent']}",
                         ha='center', va='bottom',
                         fontsize=11, fontweight='bold',
                         transform=ax_timeline.transAxes)

        # Timestep label below
        ax_timeline.text(frac_x, arrow_y - 0.15,
                         f"t = {event['t']}",
                         ha='center', va='top',
                         fontsize=10, color='darkblue',
                         transform=ax_timeline.transAxes)

    # --- Contributor charts (middle row) ---
    for col, event in enumerate(fault_timeline):
        ax = fig.add_subplot(gs[1, col])
        
        # For the first fault, don't show contributors
        if col == 0:
            contribs = {}
        else:
            # Calculate second-order action influence contributors for this fault event
            faulty_agent = event['agent']
            fault_timestep = event['t']
            
            # Get second-order action influences at the fault timestep
            if fault_timestep < len(second_order_action_influences_history):
                second_order_influences = second_order_action_influences_history[fault_timestep][faulty_agent]
                
                # Create contributors dict from second-order action influences (include all agents including self)
                contribs = {}
                for j in range(total_agents):
                    contribs[j] = abs(second_order_influences[j])  # Use absolute value of influence
            else:
                contribs = {}

        if len(contribs) == 0:
            ax.axis('off')
            if col == 0:
                ax.text(0.5, 0.5, 'Patient Zero',
                        ha='center', va='center', fontsize=10, style='italic')
            else:
                ax.text(0.5, 0.5, 'No 2nd-order action influences',
                        ha='center', va='center', fontsize=10, style='italic')
        else:
            vals = np.array(list(contribs.values()), dtype=float)
            if vals.sum() > 0:
                vals /= vals.sum()  # Normalize to sum to 1
            colors = [agent_colors[a] for a in contribs.keys()]

            wedges, _, autotexts = ax.pie(
                vals, autopct='%1.1f%%', startangle=90, colors=colors,
                wedgeprops=dict(width=0.35, edgecolor='w')
            )
            for at in autotexts:
                at.set_fontsize(9)
                at.set_fontweight('bold')
            ax.set_title('Contributors', fontsize=11, pad=5)
            ax.set_aspect('equal')

    # --- Legend row (bottom row) ---
    ax_legend = fig.add_subplot(gs[2, :])
    ax_legend.axis('off')
    legend_elements = [Patch(facecolor=agent_colors[i], label=f"Agent {i}") for i in range(total_agents)]
    ax_legend.legend(handles=legend_elements, loc='center', ncol=total_agents,
                     fontsize=9, frameon=False)

    fig.suptitle('Fault Detection Timeline and Second-Order Action Influence Contributors',
                 fontsize=14, fontweight='bold', y=0.96)

    out_path = os.path.join(logdir, 'fault_timeline_second_order_action_influences.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved second-order action influences fault timeline plot to {out_path}")


def plot_fault_timeline_observation_influences(fault_timeline, observation_influences_matrix_history, total_agents, logdir):
    """
    Plot fault timeline with observation influence contributors.
    Each fault event shows the observation influences from other agents as contributors.
    """
    if len(fault_timeline) == 0:
        print("No faults detected; skipping observation influences fault timeline plot.")
        return

    k = len(fault_timeline)
    fig = plt.figure(figsize=(max(6, 3*k), 5))
    gs = fig.add_gridspec(
        3, k,
        height_ratios=[0.8, 2, 0.1],
        hspace=0.1
    )

    cmap = plt.get_cmap('tab20')
    agent_colors = {i: cmap(i % 20) for i in range(total_agents)}

    # --- Timeline axis (top row) ---
    ax_timeline = fig.add_subplot(gs[0, :])
    ax_timeline.axis('off')

    arrow_y = 0.5
    ax_timeline.annotate(
        '', xy=(1, arrow_y), xytext=(0, arrow_y),
        arrowprops=dict(arrowstyle='-|>', lw=2, color='gray'),
        xycoords='axes fraction', textcoords='axes fraction'
    )

    # Milestones
    for i, event in enumerate(fault_timeline):
        frac_x = (i + 0.5) / k

        # Circle marker
        ax_timeline.plot(frac_x, arrow_y, 'o', color='darkred', markersize=10, transform=ax_timeline.transAxes)

        # Faulty agent label above
        ax_timeline.text(frac_x, arrow_y + 0.15,
                         f"Faulty agent {event['agent']}",
                         ha='center', va='bottom',
                         fontsize=11, fontweight='bold',
                         transform=ax_timeline.transAxes)

        # Timestep label below
        ax_timeline.text(frac_x, arrow_y - 0.15,
                         f"t = {event['t']}",
                         ha='center', va='top',
                         fontsize=10, color='darkblue',
                         transform=ax_timeline.transAxes)

    # --- Contributor charts (middle row) ---
    for col, event in enumerate(fault_timeline):
        ax = fig.add_subplot(gs[1, col])
        
        # For the first fault, don't show contributors
        if col == 0:
            contribs = {}
        else:
            # Calculate observation influence contributors for this fault event
            faulty_agent = event['agent']
            fault_timestep = event['t']
            
            # Get observation influences at the fault timestep
            if fault_timestep < len(observation_influences_matrix_history):
                observation_influences = observation_influences_matrix_history[fault_timestep][faulty_agent]
                
                # Create contributors dict from observation influences (include all agents including self)
                contribs = {}
                for j in range(total_agents):
                    contribs[j] = abs(observation_influences[j])  # Use absolute value of influence
            else:
                contribs = {}

        if len(contribs) == 0:
            ax.axis('off')
            if col == 0:
                ax.text(0.5, 0.5, 'Patient Zero',
                        ha='center', va='center', fontsize=10, style='italic')
            else:
                ax.text(0.5, 0.5, 'No observation influences',
                        ha='center', va='center', fontsize=10, style='italic')
        else:
            vals = np.array(list(contribs.values()), dtype=float)
            if vals.sum() > 0:
                vals /= vals.sum()  # Normalize to sum to 1
            colors = [agent_colors[a] for a in contribs.keys()]

            wedges, _, autotexts = ax.pie(
                vals, autopct='%1.1f%%', startangle=90, colors=colors,
                wedgeprops=dict(width=0.35, edgecolor='w')
            )
            for at in autotexts:
                at.set_fontsize(9)
                at.set_fontweight('bold')
            ax.set_title('Contributors', fontsize=11, pad=5)
            ax.set_aspect('equal')

    # --- Legend row (bottom row) ---
    ax_legend = fig.add_subplot(gs[2, :])
    ax_legend.axis('off')
    legend_elements = [Patch(facecolor=agent_colors[i], label=f"Agent {i}") for i in range(total_agents)]
    ax_legend.legend(handles=legend_elements, loc='center', ncol=total_agents,
                     fontsize=9, frameon=False)

    fig.suptitle('Fault Detection Timeline and Observation Influence Contributors',
                 fontsize=14, fontweight='bold', y=0.96)

    out_path = os.path.join(logdir, 'fault_timeline_observation_influences.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved observation influences fault timeline plot to {out_path}")


def plot_fault_timeline_second_order_observation_influences(fault_timeline, second_order_observation_influences_history, total_agents, logdir):
    """
    Plot fault timeline with second-order observation influence contributors.
    Each fault event shows the second-order observation influences from other agents as contributors.
    """
    if len(fault_timeline) == 0:
        print("No faults detected; skipping second-order observation influences fault timeline plot.")
        return

    k = len(fault_timeline)
    fig = plt.figure(figsize=(max(6, 3*k), 5))
    gs = fig.add_gridspec(
        3, k,
        height_ratios=[0.8, 2, 0.1],
        hspace=0.1
    )

    cmap = plt.get_cmap('tab20')
    agent_colors = {i: cmap(i % 20) for i in range(total_agents)}

    # --- Timeline axis (top row) ---
    ax_timeline = fig.add_subplot(gs[0, :])
    ax_timeline.axis('off')

    arrow_y = 0.5
    ax_timeline.annotate(
        '', xy=(1, arrow_y), xytext=(0, arrow_y),
        arrowprops=dict(arrowstyle='-|>', lw=2, color='gray'),
        xycoords='axes fraction', textcoords='axes fraction'
    )

    # Milestones
    for i, event in enumerate(fault_timeline):
        frac_x = (i + 0.5) / k

        # Circle marker
        ax_timeline.plot(frac_x, arrow_y, 'o', color='darkred', markersize=10, transform=ax_timeline.transAxes)

        # Faulty agent label above
        ax_timeline.text(frac_x, arrow_y + 0.15,
                         f"Faulty agent {event['agent']}",
                         ha='center', va='bottom',
                         fontsize=11, fontweight='bold',
                         transform=ax_timeline.transAxes)

        # Timestep label below
        ax_timeline.text(frac_x, arrow_y - 0.15,
                         f"t = {event['t']}",
                         ha='center', va='top',
                         fontsize=10, color='darkblue',
                         transform=ax_timeline.transAxes)

    # --- Contributor charts (middle row) ---
    for col, event in enumerate(fault_timeline):
        ax = fig.add_subplot(gs[1, col])
        
        # For the first fault, don't show contributors
        if col == 0:
            contribs = {}
        else:
            # Calculate second-order observation influence contributors for this fault event
            faulty_agent = event['agent']
            fault_timestep = event['t']
            
            # Get second-order observation influences at the fault timestep
            if fault_timestep < len(second_order_observation_influences_history):
                second_order_obs_influences = second_order_observation_influences_history[fault_timestep][faulty_agent]
                
                # Create contributors dict from second-order observation influences (include all agents including self)
                contribs = {}
                for j in range(total_agents):
                    contribs[j] = abs(second_order_obs_influences[j])  # Use absolute value of influence
            else:
                contribs = {}

        if len(contribs) == 0:
            ax.axis('off')
            if col == 0:
                ax.text(0.5, 0.5, 'Patient Zero',
                        ha='center', va='center', fontsize=10, style='italic')
            else:
                ax.text(0.5, 0.5, 'No 2nd-order obs influences',
                        ha='center', va='center', fontsize=10, style='italic')
        else:
            vals = np.array(list(contribs.values()), dtype=float)
            if vals.sum() > 0:
                vals /= vals.sum()  # Normalize to sum to 1
            colors = [agent_colors[a] for a in contribs.keys()]

            wedges, _, autotexts = ax.pie(
                vals, autopct='%1.1f%%', startangle=90, colors=colors,
                wedgeprops=dict(width=0.35, edgecolor='w')
            )
            for at in autotexts:
                at.set_fontsize(9)
                at.set_fontweight('bold')
            ax.set_title('Contributors', fontsize=11, pad=5)
            ax.set_aspect('equal')

    # --- Legend row (bottom row) ---
    ax_legend = fig.add_subplot(gs[2, :])
    ax_legend.axis('off')
    legend_elements = [Patch(facecolor=agent_colors[i], label=f"Agent {i}") for i in range(total_agents)]
    ax_legend.legend(handles=legend_elements, loc='center', ncol=total_agents,
                     fontsize=9, frameon=False)

    fig.suptitle('Fault Detection Timeline and Second-Order Observation Influence Contributors',
                 fontsize=14, fontweight='bold', y=0.96)

    out_path = os.path.join(logdir, 'fault_timeline_second_order_observation_influences.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved second-order observation influences fault timeline plot to {out_path}")