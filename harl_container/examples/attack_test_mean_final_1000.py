"""Train an algorithm."""
import argparse
from collections import deque
import sys
import os
import yaml
# Add HARL to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
from harl.utils.configs_tools import get_defaults_yaml_args, update_args
import numpy as np
import torch
from harl.utils.trans_tools import _t2n 
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from datetime import datetime
import csv

def load_taylor_history_csvs(csv_paths):
    """
    Load pre-computed Taylor history from CSV files for all agents.
    
    Args:
        csv_paths: List of paths to CSV files for each agent
        
    Returns:
        dict: {agent_id: {timestep: {'mean': float, 'std_dev': float}}}
    """
    taylor_history_data = {}
    
    for agent_id, csv_path in enumerate(csv_paths):
        agent_data = {}
        with open(csv_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                timestep = int(row['timestep'])
                mean = float(row['mean'])
                std_dev = float(row['std_dev'])
                agent_data[timestep] = {'mean': mean, 'std_dev': std_dev}
        
        taylor_history_data[agent_id] = agent_data
        print(f"Loaded Taylor history for agent {agent_id} with {len(agent_data)} timesteps")
    
    return taylor_history_data

# -------------- Ploting Starts Here -----------------
def plot_results(results, results_attacked, atk_agent_id, logdir):
        os.makedirs(logdir, exist_ok=True)
        n = len(results[0])  # number of agents
        t = len(results)     # number of time steps
        
        # Create n subplots in a row
        fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
        fig.suptitle(f'Taylor Error (Worst Action Attack | Attacked Agent ID: {atk_agent_id})', fontsize=16, y=0.95)
        
        # Ensure axes is iterable even for single agent case
        if n == 1:
            axes = [axes]
        
        for i in range(n):
            ax = axes[i]
            
            # Extract time series for agent i
            normal_series = [results[t][i] for t in range(len(results))]
            attacked_series = [results_attacked[t][i] for t in range(len(results_attacked))]
            
            # Plot the curves
            steps_normal = range(len(normal_series))
            steps_attacked = range(len(attacked_series))
            ax.plot(steps_normal, normal_series, 'b-', label='Normal', linewidth=2)
            ax.plot(steps_attacked, attacked_series, 'r-', label='Attacked', linewidth=2)
            
            ax.set_xlabel('Step')
            ax.set_ylabel('Taylor Delta Error')
            ax.set_title(f'Agent {i}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.savefig(os.path.join(logdir, f'plot_analysis_attacked_{atk_agent_id}.png'), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved analysis plot to {logdir}")
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
            ax.text(0.5, 0.5, 'No prior faults',
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
            ax.text(0.5, 0.5, 'No prior faults',
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

############# PLOTING ENDS HERE #############

def plot_taylor_error_with_historical_bounds(results_attacked, taylor_history_data, attacked_steps, atk_agent_id, logdir):
    """
    Plot Taylor expansion errors with historical bounds for all agents.
    
    Args:
        results_attacked: List of timesteps, each containing Taylor errors for all agents
        taylor_history_data: Historical statistics from CSV files
        attacked_steps: List of timesteps when attacks occurred  
        atk_agent_id: ID of the attacked agent
        logdir: Directory to save the plot
    """
    n_agents = len(results_attacked[0])  # number of agents
    n_timesteps = len(results_attacked)  # number of time steps
    
    # Create 3 subplots in a row
    fig, axes = plt.subplots(1, n_agents, figsize=(5*n_agents, 5))
    fig.suptitle(f'Taylor Error vs Historical Bounds (Attack on Agent {atk_agent_id})', fontsize=16, y=0.95)
    
    # Ensure axes is iterable even for single agent case
    if n_agents == 1:
        axes = [axes]
    
    for i in range(n_agents):
        ax = axes[i]
        
        # Extract time series for agent i
        timesteps = range(n_timesteps)
        actual_errors = [results_attacked[t][i] for t in range(n_timesteps)]
        
        # Plot historical bounds if available
        if i in taylor_history_data:
            historical_data = taylor_history_data[i]
            historical_means = []
            historical_stds = []
            
            for t in timesteps:
                if t in historical_data:
                    historical_means.append(historical_data[t]['mean'])
                    historical_stds.append(historical_data[t]['std_dev'])
                else:
                    # If no historical data for this timestep, use NaN
                    historical_means.append(np.nan)
                    historical_stds.append(np.nan)
            
            historical_means = np.array(historical_means)
            historical_stds = np.array(historical_stds)
            
            # Calculate upper and lower bounds
            upper_bound = historical_means + historical_stds
            lower_bound = historical_means - historical_stds
            
            # Plot historical bounds as green filled area
            valid_mask = ~np.isnan(historical_means)
            if np.any(valid_mask):
                ax.fill_between(timesteps, lower_bound, upper_bound, 
                               color='green', alpha=0.3, label='Historical Mean ± Std')
                ax.plot(timesteps, historical_means, 'g--', alpha=0.7, linewidth=1, label='Historical Mean')
        
        # Plot actual Taylor errors during attack as red curve
        ax.plot(timesteps, actual_errors, 'r-', linewidth=2, label='Actual Taylor Error')
        
        # Mark attacked timesteps with vertical lines
        if attacked_steps:
            for attack_step in attacked_steps:
                if attack_step < n_timesteps:
                    ax.axvline(x=attack_step, color='orange', linestyle=':', alpha=0.7, linewidth=2)
            # Add legend entry for attack markers (only once)
            if attacked_steps and attacked_steps[0] < n_timesteps:
                ax.axvline(x=attacked_steps[0], color='orange', linestyle=':', alpha=0.7, 
                          linewidth=2, label='Attack Timesteps')
        
        # Highlight the attacked agent
        if i == atk_agent_id:
            ax.set_facecolor('#fff2f2')  # Light red background for attacked agent
            ax.set_title(f'Agent {i} (ATTACKED)', fontweight='bold', color='red')
        else:
            ax.set_title(f'Agent {i}')
            
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Taylor Error')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Set y-axis to start from 0 for better comparison
        ax.set_ylim(bottom=0)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(logdir, f'taylor_error_vs_historical_bounds_attack_{atk_agent_id}.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved Taylor error vs historical bounds plot to {logdir}")

############# PLOTING ENDS HERE #############

# --------- COMPUTE TAYLOR POLICY ------------------
def compute_taylor_policy(runner, eval_obs, eval_available_actions, eval_rnn_states):
        # states_tensor = torch.stack([torch.tensor(state_dict[k], dtype=torch.float32, requires_grad=True) for k in state_dict.keys()])
        eval_obs = torch.tensor(eval_obs, dtype=torch.float32, requires_grad=True)
        delta_errors = []
        eval_actions_collector = []
        eval_masks = np.ones(
            (runner.algo_args["eval"]["n_eval_rollout_threads"], runner.num_agents, 1),
            dtype=np.float32,
        )

        for agent_id in range(runner.num_agents):
            cur_obs = eval_obs[:, agent_id]
            eval_actions, eval_actions_log_prob, temp_rnn_state = runner.actor[agent_id].get_actions(
                cur_obs,
                eval_rnn_states[:, agent_id],
                eval_masks[:, agent_id],
                eval_available_actions[:, agent_id]
                if eval_available_actions[0] is not None
                else None,
                deterministic=True,
            )
            # eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
            eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
            grad_i = torch.autograd.grad(
                outputs=eval_actions_log_prob,
                inputs=cur_obs,
                create_graph=True,
                retain_graph=True,
            )[0]

            eta_i = 0.01 * grad_i.sign() / torch.max(grad_i.norm(p=2), torch.tensor(1e-6))

            
            j_tilde = eval_actions_log_prob + torch.dot(grad_i.flatten(), eta_i.flatten())  #+ 0.5 * torch.dot(eta_i.flatten(), hvp.flatten())

            p_obs = cur_obs + eta_i
            _, perturb_log_prob, _ = runner.actor[agent_id].get_actions(
                p_obs,
                eval_rnn_states[:, agent_id],
                eval_masks[:, agent_id],
                eval_available_actions[:, agent_id]
                if eval_available_actions[0] is not None
                else None,
                deterministic=True,
            )
            # _, _, p_log_prob = runner.agents.choose_actions_attack(p_state, i)
            # # Actual value of perturbed point
            j_perturbed = perturb_log_prob

            delta_error = abs(j_perturbed - j_tilde).item()
            delta_errors.append(delta_error)

        return delta_errors
# ---------- Evaluation Loop --
def slice_avail(avail, agent_id):
    """Extract available actions for a specific agent"""
    if avail is None:
        return None
    first = avail[0]
    if first is None:
        return None
    return avail[:, agent_id]

def eval(runner, attack_status=False, attack_agent_id=0, seed=23, taylor_history_data=None):
    """Evaluate the model."""
    
    eval_episode = 0

    eval_obs, eval_share_obs, eval_available_actions = runner.eval_envs.reset(seed=seed)

    eval_rnn_states = np.zeros(
        (
            runner.algo_args["eval"]["n_eval_rollout_threads"],
            runner.num_agents,
            runner.recurrent_n,
            runner.rnn_hidden_size,
        ),
        dtype=np.float32,
    )
    eval_rnn_states_critic = np.zeros(
        (
            runner.algo_args["eval"]["n_eval_rollout_threads"],
            runner.num_agents,
            runner.recurrent_n,
            runner.rnn_hidden_size,
        ),
        dtype=np.float32,
    )
    eval_masks = np.ones(
        (runner.algo_args["eval"]["n_eval_rollout_threads"], runner.num_agents, 1),
        dtype=np.float32,
    )

    taylor_error_list = list()
    frob_norms_list = []
    sec_dir_derivatives = []
    result_deques = [deque(maxlen=5) for _ in range(runner.num_agents)]
    frob_norms_deques = [deque(maxlen=5) for _ in range(runner.num_agents)]
    sec_dir_derivatives_deques = [deque(maxlen=5) for _ in range(runner.num_agents)]

    # Additional structures to mirror get_episode_data logic
    frob_norms_matrix_history = []  # list of N x N pairwise frob matrices per timestep
    fault_first_detected = {}  # agent_id -> first detected timestep
    fault_timeline = []
    attacked_steps = []
    taylor_history = [[] for _ in range(runner.num_agents)]
    cnt = 0

    while True:
        eval_rnn_states_backup = np.copy(eval_rnn_states)
        eval_actions_collector = []
        for agent_id in range(runner.num_agents):
            eval_actions, temp_rnn_state = runner.actor[agent_id].act(
                eval_obs[:, agent_id],
                eval_rnn_states[:, agent_id],
                eval_masks[:, agent_id],
                eval_available_actions[:, agent_id]
                if eval_available_actions[0] is not None
                else None,
                deterministic=True,
            )
            eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
            eval_actions_collector.append(_t2n(eval_actions))

        eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
        
        if attack_status:
            # mark attacked step
            eval_actions[0][attack_agent_id] = runner.eval_envs.action_space[attack_agent_id].sample()  # Random action for attack agent
            print(f">>>> ",eval_actions[0][attack_agent_id])
            n_actions = runner.eval_envs.action_space[attack_agent_id].n
            avail_slice = slice_avail(eval_available_actions, attack_agent_id)
            if avail_slice is not None and avail_slice[0] is not None:
                        available_actions = np.where(avail_slice[0] > 0.5)[0]
            else:
                available_actions = list(range(n_actions))
            
            # exit("Exiting for debug")
            with torch.no_grad():
                print(f" [!!!] Attack launched on agent {attack_agent_id} at timestep: {cnt}")
                obs_tensor = torch.FloatTensor(eval_obs[:, attack_agent_id])
                rnn_tensor = torch.FloatTensor(eval_rnn_states[:, attack_agent_id])
                mask_tensor = torch.FloatTensor(eval_masks[:, attack_agent_id])
                
                action_log_probs, dist_entropy, action_distribution = runner.actor[attack_agent_id].evaluate_actions(
                            obs_tensor.to(runner.device),
                            rnn_tensor.to(runner.device),
                            available_actions,
                            mask_tensor.to(runner.device),
                            slice_avail(eval_available_actions, attack_agent_id),
                            None
                        )
                q_values = action_log_probs.squeeze()
                if q_values.numel() == 1 or len(available_actions) == 1:
                    print(f"Agent {agent_id} appears to be dead or has only one action. Using index 0.")
                    eval_actions[0][attack_agent_id] = 0
                else:
                    worst_action = torch.argmin(q_values).item()
                    eval_actions[0][attack_agent_id] = worst_action
                    print(f"Agent {attack_agent_id} worst action under current policy: {worst_action}")
            attacked_steps.append(cnt)


        # calculating taylor policy
        delta_errors = compute_taylor_policy(runner, eval_obs, eval_available_actions, eval_rnn_states_backup)
        # results_frob_norms = compute_frob_norms(runner, eval_obs, 1, eval_rnn_states_critic, eval_masks)
        # results_sec_dir_derivatives = compute_2nd_ord_dir_derivatives(runner, eval_obs, 1, eval_rnn_states_critic, eval_masks)
        # pairwise frob matrix for cascading analysis
        pairwise_frobs = compute_pairwise_frob_norms(runner, eval_obs, eval_rnn_states_critic, eval_masks)
        frob_norms_matrix_history.append(pairwise_frobs)

        for i in range(runner.num_agents):
            result_deques[i].append(delta_errors[i])
            taylor_approx_error = np.mean(result_deques[i])
            taylor_history[i].append(taylor_approx_error)

            # frob_norms_deques[i].append(results_frob_norms[i])
            sec_dir_derivatives_deques[i].append(results_sec_dir_derivatives[i])

            # Detect anomalies based on Taylor approximation error using pre-computed history
            if i not in fault_first_detected and cnt in taylor_history_data[i]:
                historical_data = taylor_history_data[i][cnt]
                historical_mean = historical_data['mean']
                historical_std = historical_data['std_dev']
                
                # Ensure minimum std deviation to avoid division by zero
                if historical_std < 1e-6:
                    historical_std = 1e-6
                
                # Check if current error is outside historical bounds (mean ± std_dev)
                lower_bound = historical_mean - historical_std
                upper_bound = historical_mean + historical_std
                
                if taylor_approx_error < lower_bound or taylor_approx_error > upper_bound:
                    print(f" [!!!] Anomaly detected for agent {i} at timestep: {cnt}. Taylor Appx. Error: {taylor_approx_error}")
                    print(f"     >> Historical bounds: [{lower_bound:.6f}, {upper_bound:.6f}], Mean: {historical_mean:.6f}, Std: {historical_std:.6f}")
                    fault_first_detected[i] = cnt
                    # Cascading Impact Analysis
                    prev_faults = [(f, tf) for f, tf in fault_first_detected.items() if f != i and tf < cnt]
                    contribs = {}
                    if len(prev_faults) > 0:
                        for f, tf in prev_faults:
                            values_over_time = [frob_norms_matrix_history[tau][i][f] for tau in range(tf, cnt + 1) if tau < len(frob_norms_matrix_history)]
                            if len(values_over_time) > 0:
                                contribs[f] = float(np.mean(values_over_time))
                        if len(contribs) > 0:
                            ranked = sorted(contribs.items(), key=lambda x: x[1], reverse=True)
                            print(f"     >> Potential contributors to fault in agent {i} (mean ||H_{{i,f}}||_F from t_f to {cnt}): {ranked}")
                    fault_timeline.append({
                        'agent': i,
                        't': cnt,
                        'contribs': contribs
                    })

        taylor_error_list.append([np.mean(list(result_deques[j])) for j in range(runner.num_agents)])
        # frob_norms_list.append([np.mean(frob_norms_deques[i]) for i in range(runner.num_agents)])
        sec_dir_derivatives.append([np.mean(sec_dir_derivatives_deques[i]) for i in range(runner.num_agents)])

        (
            eval_obs,
            eval_share_obs,
            eval_rewards,
            eval_dones,
            eval_infos,
            eval_available_actions,
        ) = runner.eval_envs.step(eval_actions)
        eval_data = (
            eval_obs,
            eval_share_obs,
            eval_rewards,
            eval_dones,
            eval_infos,
            eval_available_actions,
        )

        value, eval_rnn_states_critic = runner.critic.get_values(
            eval_share_obs,
            eval_rnn_states_critic,
            eval_masks,
        )

        eval_dones_env = np.all(eval_dones, axis=1)

        eval_rnn_states[
            eval_dones_env == True
        ] = np.zeros(  # if env is done, then reset rnn_state to all zero
            (
                (eval_dones_env == True).sum(),
                runner.num_agents,
                runner.recurrent_n,
                runner.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

        eval_masks = np.ones(
            (runner.algo_args["eval"]["n_eval_rollout_threads"], runner.num_agents, 1),
            dtype=np.float32,
        )
        eval_masks[eval_dones_env == True] = np.zeros(
            ((eval_dones_env == True).sum(), runner.num_agents, 1), dtype=np.float32
        )

        for eval_i in range(runner.algo_args["eval"]["n_eval_rollout_threads"]):
            if eval_dones_env[eval_i]:
                eval_episode += 1

        if eval_episode >= runner.algo_args["eval"]["eval_episodes"]:
            break

        cnt += 1
    
    # return taylor_error_list, frob_norms_list, sec_dir_derivatives, frob_norms_matrix_history, fault_timeline, attacked_steps
    return taylor_error_list, None, sec_dir_derivatives, frob_norms_matrix_history, fault_timeline, attacked_steps

def compute_pairwise_frob_norms(runner, eval_obs, eval_rnn_states_critic, eval_masks):
    """Compute Frobenius norms of cross-agent Hessian blocks for all (i, j).
    Returns an N x N matrix where entry [i][j] approximates || \partial^2 v_i / (\partial obs_i \partial obs_j) ||_F.
    """
    eval_obs = torch.tensor(eval_obs, dtype=torch.float32)

    agent_obs_tensors = []
    n_agents = runner.num_agents
    # assume eval_obs shape (1, n_agents, obs_dim)
    for i in range(n_agents):
        agent_obs = eval_obs[0][i].clone().detach()
        agent_obs_tensor = torch.tensor(agent_obs, dtype=torch.float32, requires_grad=True)
        agent_obs_tensors.append(agent_obs_tensor)

    concatenated_obs = torch.cat(agent_obs_tensors, dim=0)
    share_obs = concatenated_obs.unsqueeze(0).unsqueeze(0)
    share_obs = share_obs.expand(1, n_agents, -1)

    values, temp_rnn_state_critic = runner.critic.get_values(
        share_obs,
        eval_rnn_states_critic,
        eval_masks,
    )
    values = values.squeeze()

    N = n_agents
    results = [[0.0 for _ in range(N)] for _ in range(N)]

    for i in range(N):
        # gradient of v_i wrt agent i obs
        try:
            grad_i = torch.autograd.grad(values[i], agent_obs_tensors[i], create_graph=True, retain_graph=True)[0]
        except Exception:
            grad_i = torch.zeros_like(agent_obs_tensors[i])

        for j in range(N):
            hessian_matrix = []
            for k in range(grad_i.shape[0]):
                second_grad = torch.autograd.grad(
                    grad_i[k],
                    agent_obs_tensors[j],
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if second_grad is None:
                    second_grad = torch.zeros_like(agent_obs_tensors[j])
                hessian_matrix.append(second_grad.flatten())

            H = torch.stack(hessian_matrix) if len(hessian_matrix) > 0 else torch.zeros(1, 1)
            results[i][j] = H.norm(p='fro').item()
    # Normalize each row by its sum
    for i in range(N):
        row_sum = sum(results[i])
        if row_sum > 0:
            for j in range(N):
                results[i][j] /= row_sum
    return results

def compute_frob_norms(runner, eval_obs, vulnerable_agent_id, eval_rnn_states_critic, eval_masks):
    eval_obs = torch.tensor(eval_obs, dtype=torch.float32)

    # Create separate tensors for each agent's observations with gradient tracking
    agent_obs_tensors = []
    obs_dim = runner.envs.observation_space[0].shape[0]  # Assuming all agents have same obs dim
    n_agents = runner.num_agents

    for i in range(n_agents):
        # Extract agent i's observation from share_obs
        # start_idx = i * obs_dim
        # end_idx = (i + 1) * obs_dim
        agent_obs = eval_obs[0][i].clone().detach()
        agent_obs_tensor = torch.tensor(agent_obs, dtype=torch.float32, requires_grad=True)
        agent_obs_tensors.append(agent_obs_tensor)

    # Concatenate back to recreate share_obs structure with gradient tracking
    concatenated_obs = torch.cat(agent_obs_tensors, dim=0)
    share_obs = concatenated_obs.unsqueeze(0).unsqueeze(0)  # Reshape to (1, 1, obs_dim*n_agent)
    # Need to expand to match original shape (1, n_agents, obs_dim*n_agent)
    share_obs = share_obs.expand(1, n_agents, -1)
   
    values, temp_rnn_state_critic = runner.critic.get_values(
            share_obs,
            eval_rnn_states_critic,
            eval_masks,
        )
    values = values.squeeze()  # shape: (N,)

    # values = runner.agent_n.compute_value(states_tensor).squeeze(-1)  # shape: (N,)

    # Store eigenvalues for each agent pair (i, j)
    results = []

    for i in range(runner.num_agents):
        obs_dim = runner.envs.observation_space[i].shape[0]
        # Compute first-order gradient of v_i with respect to agent i's observation
        grad_i = torch.autograd.grad(values[i], agent_obs_tensors[i], create_graph=True, retain_graph=True)[0]

        # Compute cross-agent Hessian matrix for agent pair (i, j)
        # This represents ∂²v/∂obs_i∂obs_j
        hessian_matrix = []
        
        for k in range(grad_i.shape[0]):  # For each dimension of agent i's observation (has shape (obs_dim,))
            # Compute ∂²v/∂obs_i[k]∂obs_j
            second_grad = torch.autograd.grad(
                grad_i[k],
                agent_obs_tensors[vulnerable_agent_id],
                retain_graph=True,
                allow_unused=True
            )[0]
            hessian_matrix.append(second_grad.flatten())

        # Convert to tensor and compute eigenvalues
        H = torch.stack(hessian_matrix)

        # Compute eigenvalues
        eigenvalues = torch.linalg.eigvals(H)
        real_eigenvalues = eigenvalues.real
        negative_count = (real_eigenvalues < 0).sum().item()
        total_count = len(real_eigenvalues)
        negative_ratio = negative_count / total_count if total_count > 0 else 0.0
        results.append(negative_ratio)
        continue

        # Frobenius norm of the Hessian matrix
        results.append(H.norm(p='fro').item()) 

    return results

# second order directional derivative
def compute_2nd_ord_dir_derivatives(runner, eval_obs, vulnerable_agent_id, eval_rnn_states_critic, eval_masks):
    eval_obs = torch.tensor(eval_obs, dtype=torch.float32)

    # Create separate tensors for each agent's observations with gradient tracking
    agent_obs_tensors = []
    obs_dim = runner.envs.observation_space[0].shape[0]  # Assuming all agents have same obs dim
    n_agents = runner.num_agents

    for i in range(n_agents):
        # Extract agent i's observation from share_obs
        # start_idx = i * obs_dim
        # end_idx = (i + 1) * obs_dim
        agent_obs = eval_obs[0][i].clone().detach()
        agent_obs_tensor = torch.tensor(agent_obs, dtype=torch.float32, requires_grad=True)
        agent_obs_tensors.append(agent_obs_tensor)

    # Concatenate back to recreate share_obs structure with gradient tracking
    concatenated_obs = torch.cat(agent_obs_tensors, dim=0)
    share_obs = concatenated_obs.unsqueeze(0).unsqueeze(0)  # Reshape to (1, 1, obs_dim*n_agent)
    # Need to expand to match original shape (1, n_agents, obs_dim*n_agent)
    share_obs = share_obs.expand(1, n_agents, -1)
   
    values, temp_rnn_state_critic = runner.critic.get_values(
            share_obs,
            eval_rnn_states_critic,
            eval_masks,
        )
    values = values.squeeze()  # shape: (N,)

    # values = runner.agent_n.compute_value(states_tensor).squeeze(-1)  # shape: (N,)

    # Store eigenvalues for each agent pair (i, j)
    results = []

    for i in range(runner.num_agents):
        obs_dim = runner.envs.observation_space[i].shape[0]
        # Compute first-order gradient of v_i with respect to agent i's observation
        grad_i = torch.autograd.grad(values[i], agent_obs_tensors[i], create_graph=True, retain_graph=True)[0]
        v = grad_i / torch.max(grad_i.norm(p=2), torch.tensor(1e-6))

        # Compute Hessian-vector product (HVP) of grad_i and v with respect to states_tensors[j]
        hvp = torch.autograd.grad(
            outputs=grad_i,
            inputs=agent_obs_tensors[vulnerable_agent_id],
            grad_outputs=v,
            retain_graph=True,
            allow_unused=True
        )[0]

        # Compute u^T * H * v (quadratic form)
        grad_j = torch.autograd.grad(-values[i], agent_obs_tensors[vulnerable_agent_id], create_graph=True, retain_graph=True)[0]
        u = grad_j / torch.max(grad_j.norm(p=2), torch.tensor(1e-6))
        u = -u  # negative gradient direction
        curvature_val = torch.dot(u.flatten(), hvp.flatten())
        results.append(curvature_val.item())

    return results

def plot_frobs(frobs_normal, frobs_atk, attacked_steps, atk_agent_id, logdir):
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
        steps = range(len(normal_series))
        
        ax.plot(steps, attacked_series, 'r-', label='Under Attack', linewidth=2)
        ax.plot(steps, normal_series, 'g-', label='Normal', linewidth=2)
        
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

def save_matrix_to_files(matrix, attacked_steps, attacked_agent_id, total_agents, logdir, filename):
    """
    Save matrix data to a CSV file for all timesteps.
    
    Args:
        matrix: List of timesteps, where each timestep contains n_agent data
        attacked_agent_id: ID of the attacked agent
        total_agents: Total number of agents
        logdir: Directory to save the file
    """
    filepath = os.path.join(logdir, filename)
    
    # Create header row
    # header = ["timestep", "attacked_agent"]
    header = ["timestep", "is_attacked", "attacked_agent"]
    for i in range(total_agents):
        header.append(f"agent_{i}")
    
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        
        for timestep, timestep_data in enumerate(matrix):
            is_attacked = 1 if timestep in attacked_steps else 0
            row = [timestep, is_attacked, attacked_agent_id]
            for i in range(total_agents):
                row.append(timestep_data[i])
            writer.writerow(row)
    
    print(f"Saved {len(matrix)} timestep matrices to {filepath}")

def restore(runner, restore_dir, reward):
    """Restore trained model from checkpoint"""
    for agent_id in range(runner.num_agents):
        policy_actor_state_dict = torch.load(
            os.path.join(restore_dir, f"actor_agent{agent_id}_{reward}.pt"),
            weights_only=False, map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        )
        runner.actor[agent_id].actor.load_state_dict(policy_actor_state_dict)

# def restore(runner,reward,episode,filepath="/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v2-discrete/happo/installtest/seed-00042-2025-08-03-20-41-48/models"):
#         """Restore model parameters."""
#         for agent_id in range(runner.num_agents):
#             policy_actor_state_dict = torch.load(
#                 str(filepath)
#                 + "/actor_agent"
#                 + str(agent_id)
#                 + "_reward_" + str(reward)
#                 + "_episode_" + str(episode)
#                 + ".pt"
#             )
#             runner.actor[agent_id].actor.load_state_dict(policy_actor_state_dict)
#         if not runner.algo_args["render"]["use_render"]:
#             policy_critic_state_dict = torch.load(
#                 str(filepath)
#                 + "/critic_agent"
#                 + "_reward_" + str(reward)
#                 + "_episode_" + str(episode)
#                 + ".pt"
#             )
#             runner.critic.critic.load_state_dict(policy_critic_state_dict)
#             if runner.value_normalizer is not None:
#                 value_normalizer_state_dict = torch.load(
#                     str(filepath)
#                     + "/value_normalizer"
#                     + "_reward_" + str(reward)
#                     + "_episode_" + str(episode)
#                     + ".pt"
#                 )
#                 runner.value_normalizer.load_state_dict(value_normalizer_state_dict)
def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="happo",
        choices=[
            "happo",
            "hatrpo",
            "haa2c",
            "haddpg",
            "hatd3",
            "hasac",
            "had3qn",
            "maddpg",
            "matd3",
            "mappo",
        ],
        help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, matd3, mappo.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="pettingzoo_mpe",
        choices=[
            "smac",
            "mamujoco",
            "pettingzoo_mpe",
            "gym",
            "football",
            "dexhands",
            "smacv2",
            "lag",
        ],
        help="Environment name. Choose from: smac, mamujoco, pettingzoo_mpe, gym, football, dexhands, smacv2, lag.",
    )
    parser.add_argument(
        "--exp_name", type=str, default="installtest", help="Experiment name."
    )
    parser.add_argument(
        "--attack_id", type=int, default=0, help="Agent ID to attack."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    parser.add_argument(
        "--reward",
        type=float,
        default=-79.879,
        help="Reward value to restore the model."
    )
    parser.add_argument(
        "--filepath",
        type=str,
        default="/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/hatrpo/Latest_3/seed-00001-2025-08-15-22-56-55/models",
        help="Filepath to restore the model from."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed."
    )
    parser.add_argument(
        "--taylor_csv_agent0", type=str, default="", help="Path to CSV file with pre-computed Taylor history for agent 0."
    )
    parser.add_argument(
        "--taylor_csv_agent1", type=str, default="", help="Path to CSV file with pre-computed Taylor history for agent 1."
    )
    parser.add_argument(
        "--taylor_csv_agent2", type=str, default="", help="Path to CSV file with pre-computed Taylor history for agent 2."
    )
    parser.add_argument(
        "--save_dir", type=str, default=None, help="Directory to save results."
    )
    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict
    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:  # load config from corresponding yaml file
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])
    update_args(unparsed_dict, algo_args, env_args)  # update args from command line

    if args["env"] == "dexhands":
        import isaacgym  # isaacgym has to be imported before PyTorch

    # note: isaac gym does not support multiple instances, thus cannot eval separately
    if args["env"] == "dexhands":
        algo_args["eval"]["use_eval"] = False
        algo_args["train"]["episode_length"] = env_args["hands_episode_length"]

    # start training
    from harl.runners import RUNNER_REGISTRY
# 
    algo_args['eval']['n_eval_rollout_threads'] = 1
    algo_args['eval']['eval_episodes'] = 1
    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    restore(runner, args['filepath'],args['reward'])  # Restore the model with specific reward and episode
    runner.prep_training()
    
    attack_agent_id = args['attack_id']
    print(f"=== Evaluating with attack on agent {attack_agent_id} ===")
    
    # Load pre-computed Taylor history CSV files
    csv_paths = [
        args.get('taylor_csv_agent0', ''),
        args.get('taylor_csv_agent1', ''), 
        args.get('taylor_csv_agent2', '')
    ]
    taylor_history_data = load_taylor_history_csvs(csv_paths)
    
    # Run evaluation without attack
    # results_normal, frob_norms_normal, sec_dir_derivatives_normal, frob_norms_matrix_history_normal, fault_timeline_normal, attacked_steps_normal = eval(runner, False, attack_agent_id, taylor_history_data=taylor_history_data)
    # Run evaluation with attack
    results_attacked, frob_norms_atk, sec_dir_derivatives_atk, frob_norms_matrix_history_atk, fault_timeline_atk, attacked_steps_atk = eval(runner, attack_status=True, attack_agent_id=attack_agent_id, seed=args['seed'], taylor_history_data=taylor_history_data)

    log_dir = algo_args['attack']['log_dir']
    alg_name = algo_args['attack']['algo_name']
    date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    if args['save_dir'] is not None:
        log_path = os.path.join(args['save_dir'], alg_name, date)
    else:
        log_path = os.path.join(log_dir, alg_name, date)
    os.makedirs(log_path, exist_ok=True)

    # plot_results(results_normal, results_attacked, atk_agent_id=attack_agent_id, logdir=log_path)
    # plot_frobs(frob_norms_normal, frob_norms_atk, attacked_steps_atk, attack_agent_id, log_path)
    # plot_sec_dir_derivatives(sec_dir_derivatives_normal, sec_dir_derivatives_atk, attacked_steps_atk, attack_agent_id, log_path)

    # Save matrices and include attacked steps for attacked run
    save_matrix_to_files(results_attacked, attacked_steps_atk, attack_agent_id, runner.num_agents, log_path, f'happo_taylor_error_atk_{attack_agent_id}.csv')
    # save_matrix_to_files(frob_norms_atk, attacked_steps_atk, attack_agent_id, runner.num_agents, log_path, f'happo_frobenius_norms_atk_{attack_agent_id}.csv')
    # save_matrix_to_files(sec_dir_derivatives_atk, attacked_steps_atk, attack_agent_id, runner.num_agents, log_path, f'happo_sec_dir_derivatives_atk_{attack_agent_id}.csv')

    # Plot fault timeline and contributor charts for attacked run
    plot_fault_timeline(fault_timeline_atk, runner.num_agents, log_path)
    plot_contributor_barchart(fault_timeline_atk, runner.num_agents, log_path)
    
    # Plot Taylor error vs historical bounds
    plot_taylor_error_with_historical_bounds(results_attacked, taylor_history_data, attacked_steps_atk, attack_agent_id, log_path)

    # runner.run()
    runner.close()



if __name__ == "__main__":
    main()
