"""
Timeline and fault detection visualization functions.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from .utils import get_agent_colors


def plot_fault_timeline(fault_timeline, total_agents, logdir):
    if len(fault_timeline) == 0:
        print("No faults detected; skipping fault timeline plot.")
        return

    k = len(fault_timeline)
    fig = plt.figure(figsize=(max(6, 3*k), 5))  # reduce height from 6 → 5
    gs = fig.add_gridspec(
        3, k,
        height_ratios=[0.8, 2, 0.1],  # smaller top & bottom rows
        hspace=0.1  # tighter vertical spacing
    )

    cmap = plt.get_cmap('tab20')
    agent_colors = {i: cmap(i % 20) for i in range(total_agents)}

    # --- Timeline axis (top row) ---
    ax_timeline = fig.add_subplot(gs[0, :])
    ax_timeline.axis('off')

    # Horizontal arrow for timeline
    arrow_y = 0.5
    ax_timeline.annotate(
        '', xy=(1, arrow_y), xytext=(0, arrow_y),
        arrowprops=dict(arrowstyle='-|>', lw=2, color='gray'),
        xycoords='axes fraction', textcoords='axes fraction'
    )

    # Milestones
    for i, event in enumerate(fault_timeline):
        frac_x = (i + 0.5) / k  # evenly spaced

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
        contribs = event.get('contribs', {})

        if len(contribs) == 0:
            ax.axis('off')
            ax.text(0.5, 0.5, 'Patient Zero',
                    ha='center', va='center', fontsize=10, style='italic')
        else:
            vals = np.array(list(contribs.values()), dtype=float)
            if vals.sum() > 0:
                vals /= vals.sum()
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

    fig.suptitle('Fault Detection Timeline and Contributors',
                 fontsize=14, fontweight='bold', y=0.96)

    out_path = os.path.join(logdir, 'fault_timeline.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved fault timeline plot to {out_path}")


def plot_contributor_barchart(fault_timeline, total_agents, logdir):
    if len(fault_timeline) == 0:
        print("No faults detected; skipping contributor bar chart.")
        return

    k = len(fault_timeline)
    # Increase figure width for better spacing, especially with many events
    fig = plt.figure(figsize=(max(8, 4*k), 6))  # Increased from 3*k to 4*k width and 5 to 6 height
    gs = fig.add_gridspec(
        3, k,
        height_ratios=[0.8, 2.5, 0.2],  # Give more space to the middle row
        hspace=0.15,  # Increase vertical spacing
        wspace=0.3    # Add horizontal spacing between subplots
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

    for i, event in enumerate(fault_timeline):
        frac_x = (i + 0.5) / k
        ax_timeline.plot(frac_x, arrow_y, 'o', color='darkred', markersize=10, transform=ax_timeline.transAxes)
        ax_timeline.text(frac_x, arrow_y + 0.15,
                         f"Faulty agent {event['agent']}",
                         ha='center', va='bottom',
                         fontsize=11, fontweight='bold',
                         transform=ax_timeline.transAxes)
        ax_timeline.text(frac_x, arrow_y - 0.15,
                         f"t = {event['t']}",
                         ha='center', va='top',
                         fontsize=10, color='darkblue',
                         transform=ax_timeline.transAxes)

    # --- Contributor bar charts (middle row) ---
    for col, event in enumerate(fault_timeline):
        ax = fig.add_subplot(gs[1, col])
        contribs = event.get('contribs', {})

        if len(contribs) == 0:
            ax.axis('off')
            ax.text(0.5, 0.5, 'Patient Zero',
                    ha='center', va='center', fontsize=10, style='italic')
        else:
            agents = list(contribs.keys())
            scores = np.array(list(contribs.values()), dtype=float)

            colors = [agent_colors[a] for a in agents]

            # Use narrower bars with proper spacing
            bar_width = 0.6  # Make bars narrower
            x_positions = range(len(agents))
            bars = ax.bar(x_positions, scores, color=colors, width=bar_width, 
                         edgecolor='black', linewidth=0.5, alpha=0.8)

            # Set appropriate x limits with padding
            if len(agents) > 1:
                ax.set_xlim(-0.8, len(agents) - 0.2)
            else:
                ax.set_xlim(-0.8, 0.8)

            # Improved label handling
            ax.set_xticks(x_positions)
            if len(agents) <= 3:
                # For few agents, use normal labels
                ax.set_xticklabels([f"Agent {i}" for i in agents], fontsize=9)
            else:
                # For many agents, use abbreviated labels with rotation
                ax.set_xticklabels([f"A{i}" for i in agents], rotation=45, ha='right', fontsize=8)

            # Add value labels on top of bars for clarity
            for bar, score in zip(bars, scores):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(scores)*0.01,
                       f'{score:.3f}', ha='center', va='bottom', fontsize=7)

            ax.set_ylabel("Contribution", fontsize=9)
            ax.set_title('Contributors', fontsize=11, pad=10)

            # Grid for readability
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Set y-axis to start from 0 for better visual comparison
            ax.set_ylim(bottom=0)

    # --- Legend row (bottom row) ---
    ax_legend = fig.add_subplot(gs[2, :])
    ax_legend.axis('off')
    legend_elements = [Patch(facecolor=agent_colors[i], label=f"Agent {i}") for i in range(total_agents)]
    ax_legend.legend(handles=legend_elements, loc='center', ncol=total_agents,
                     fontsize=9, frameon=False)

    fig.suptitle('Fault Detection Timeline and Contributor Scores',
                 fontsize=14, fontweight='bold', y=0.97)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])  # Adjusted margins for better label visibility

    out_path = os.path.join(logdir, 'fault_contributor_barchart.png')
    plt.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.2)  # Added padding
    plt.show()
    print(f"Saved contributor bar chart to {out_path}")
