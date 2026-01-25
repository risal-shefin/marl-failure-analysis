"""
Data processing and file I/O functions.
"""
import os
import csv


def save_decayed_action_influence_csv(influence_tensor, logdir, filename_prefix, start_timestep=0):
    """
    Persist decayed pairwise influence tensor to per-target CSVs with timestep headers.
    
    Args:
        influence_tensor: 3D tensor [target][source][timestep]
        logdir: Directory to save files
        filename_prefix: Prefix for output files
        start_timestep: Starting timestep for data export
    """
    if not influence_tensor:
        print("No decayed action influence data to save.")
        return

    num_targets = len(influence_tensor)
    if num_targets == 0:
        print("Decayed action influence tensor is empty.")
        return

    num_sources = len(influence_tensor[0])
    num_timesteps = len(influence_tensor[0][0]) if num_sources > 0 else 0
    if num_timesteps == 0:
        print("Decayed action influence tensor has no timestep data.")
        return

    start_index = int(start_timestep)
    if start_index < 0:
        start_index = 0

    if start_index >= num_timesteps:
        print("Start timestep is beyond available influence data; skipping save.")
        return

    timestep_range = range(start_index, num_timesteps)
    header = ["agent_id"] + list(timestep_range)

    for target_agent in range(num_targets):
        filepath = os.path.join(logdir, f"{filename_prefix}_target_{target_agent}.csv")

        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)

            for source_agent in range(num_sources):
                if source_agent == target_agent:
                    continue

                row = [source_agent]
                row.extend(influence_tensor[target_agent][source_agent][start_index:])
                writer.writerow(row)

            # Include self-influence row (expected to remain zeros)
            self_row = [target_agent]
            self_row.extend(influence_tensor[target_agent][target_agent][start_index:])
            writer.writerow(self_row)

        print(f"Saved decayed action influence data for target agent {target_agent} to {filepath}")


def save_matrix_to_files(matrix, attacked_steps, attacked_agent_id, total_agents, logdir, filename):
    """
    Save matrix data to a CSV file for all timesteps.
    
    Args:
        matrix: List of timesteps, where each timestep contains n_agent data
        attacked_steps: List of timesteps when attack occurred
        attacked_agent_id: ID of the attacked agent
        total_agents: Total number of agents
        logdir: Directory to save the file
        filename: Name of the output file
    """
    filepath = os.path.join(logdir, filename)
    
    # Create header row
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


def save_q_values_csv(q_values, logdir, filename):
    """
    Save raw per-agent Q-values over timesteps to CSV format.
    
    Args:
        q_values: List of timesteps, each containing Q-values for all agents
        logdir: Directory to save the file
        filename: Name of the output file
    """
    if len(q_values) == 0:
        print(f"No Q-values recorded for {filename}; skipping save.")
        return

    num_agents = len(q_values[0])
    timesteps = len(q_values)
    header = ["agent_id"] + [f"timestep_{t}" for t in range(timesteps)]
    filepath = os.path.join(logdir, filename)

    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for agent_id in range(num_agents):
            row = [agent_id]
            for t in range(timesteps):
                row.append(q_values[t][agent_id])
            writer.writerow(row)

    print(f"Saved Q-values ({timesteps} timesteps) to {filepath}")


def save_q_value_drop_csv(normal_q_values, attacked_q_values, logdir, filename):
    """
    Save per-agent Q-value drop (attacked - normal) over timesteps to CSV format.
    
    Args:
        normal_q_values: Q-values from normal scenario
        attacked_q_values: Q-values from attacked scenario
        logdir: Directory to save the file
        filename: Name of the output file
    """
    if len(normal_q_values) == 0 or len(attacked_q_values) == 0:
        print("No Q-values available to save.")
        return

    truncated_steps = min(len(normal_q_values), len(attacked_q_values))
    if truncated_steps == 0:
        print("No overlapping timesteps for Q-value comparison.")
        return

    num_agents = len(normal_q_values[0])
    header = ["agent_id"] + [f"timestep_{t}" for t in range(truncated_steps)]
    filepath = os.path.join(logdir, filename)

    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for agent_id in range(num_agents):
            row = [agent_id]
            for t in range(truncated_steps):
                drop_val = attacked_q_values[t][agent_id] - normal_q_values[t][agent_id]
                row.append(drop_val)
            writer.writerow(row)

    print(f"Saved Q-value drops ({truncated_steps} timesteps) to {filepath}")

    if truncated_steps < len(normal_q_values) or truncated_steps < len(attacked_q_values):
        print("Warning: Episodes ended at different lengths; truncated to minimum duration for comparison.")


def load_reference_values(ref_val_dir, num_agents, detection_method):
    """
    Load reference values from CSV files.
    
    Args:
        ref_val_dir: Directory containing reference CSV files
        num_agents: Number of agents
        detection_method: Detection method ('mean_std', 'median_mad', 'diff')
        
    Returns:
        Tuple of (ref_vals, ref_std_devs) lists for each agent
    """
    ref_vals = [[] for _ in range(num_agents)]
    ref_std_devs = [[] for _ in range(num_agents)]

    for agent_id in range(num_agents):
        csv_filename = f"maddpg_taylor_error_atk_free_agent_{agent_id}.csv"
        csv_path = os.path.join(ref_val_dir, csv_filename)
        
        with open(csv_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                if detection_method == 'mean_std':
                    # Use mean and std_dev columns
                    ref_vals[agent_id].append(float(row[2]))  # mean
                    ref_std_devs[agent_id].append(float(row[4]))  # std_dev
                elif detection_method == 'median_mad':
                    # Use median and MAD columns
                    ref_vals[agent_id].append(float(row[7]))  # median
                    ref_std_devs[agent_id].append(float(row[8]))  # MAD
                elif detection_method == 'diff':
                    # Use diff_mean and diff_std columns
                    ref_vals[agent_id].append(float(row[9]))  # diff_mean
                    ref_std_devs[agent_id].append(float(row[10]))  # diff_std
                else:
                    raise ValueError(f"Unknown detection method: {detection_method}")
    
    return ref_vals, ref_std_devs