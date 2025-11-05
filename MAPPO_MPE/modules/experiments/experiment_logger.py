"""
Data logger for storing experiment results and metrics.
"""
import os
import csv


class ExperimentDataLogger:
    """
    Logs cumulative influences, directional derivatives, and Taylor deviations.
    """
    
    def __init__(self, nagents):
        """
        Initialize data logger.
        
        Args:
            nagents: Number of agents in the system
        """
        self.nagents = nagents
        self.cumulative_influences_data = []
        self.directional_derivatives_data = []
        self.taylor_deviations_data = []
    
    def log_cumulative_influences(self, action_influences_history, seed, episode_length):
        """
        Store cumulative influence sums for each agent pair at each timestep.
        
        Args:
            action_influences_history: List of action influence matrices for each timestep
            seed: Random seed for this experiment
            episode_length: Length of the episode
        """
        print(f"Storing cumulative influences for seed {seed}...")
        
        # For each agent pair (i, j), compute and store cumulative influences
        for agent_i in range(self.nagents):
            for agent_j in range(self.nagents):
                if agent_i == agent_j:
                    continue  # Skip self-influence
                
                # Compute cumulative influence sum at each timestep
                cumulative_influence = 0.0
                timestep_cumulative_values = []
                
                for t in range(min(episode_length, len(action_influences_history))):
                    # Get influence of agent_i on agent_j at timestep t
                    influence_value = action_influences_history[t][agent_j][agent_i]
                    cumulative_influence += influence_value
                    timestep_cumulative_values.append(cumulative_influence)
                
                # Store the data for this agent pair
                pair_data = {
                    'seed': seed,
                    'influencer_agent_id': agent_i,
                    'influenced_agent_id': agent_j,
                    'episode_length': episode_length,
                    'cumulative_values': timestep_cumulative_values
                }
                
                self.cumulative_influences_data.append(pair_data)
        
        print(f"Stored cumulative influences for seed {seed} "
              f"({self.nagents * (self.nagents - 1)} agent pairs)")
    
    def log_directional_derivatives(self, directional_derivatives_history, seed, episode_length):
        """
        Store directional second derivatives for each agent pair at each timestep.
        
        Args:
            directional_derivatives_history: List of directional second derivative matrices
            seed: Random seed for this experiment
            episode_length: Length of the episode
        """
        print(f"Storing directional derivatives for seed {seed}...")
        
        # For each agent pair (i, j), store directional derivatives at each timestep
        for agent_i in range(self.nagents):
            for agent_j in range(self.nagents):
                if agent_i == agent_j:
                    continue  # Skip self-influence
                
                # Get directional derivatives for each timestep
                timestep_derivative_values = []
                
                for t in range(min(episode_length, len(directional_derivatives_history))):
                    # Get directional second derivative of agent_i on agent_j at timestep t
                    derivative_value = directional_derivatives_history[t][agent_j][agent_i]
                    timestep_derivative_values.append(derivative_value)
                
                # Store the data for this agent pair
                pair_data = {
                    'seed': seed,
                    'influencer_agent_id': agent_i,
                    'influenced_agent_id': agent_j,
                    'episode_length': episode_length,
                    'derivative_values': timestep_derivative_values
                }
                
                self.directional_derivatives_data.append(pair_data)
        
        print(f"Stored directional derivatives for seed {seed} "
              f"({self.nagents * (self.nagents - 1)} agent pairs)")
    
    def log_taylor_deviations(self, high_attack_results, low_attack_results, 
                             ref_vals, ref_std_devs, seed, agent_i, agent_j):
        """
        Store Taylor deviations from mean for both high and low influence attacks.
        
        Args:
            high_attack_results: Results from high influence attack episode
            low_attack_results: Results from low influence attack episode
            ref_vals: Reference Taylor error values
            ref_std_devs: Reference standard deviations
            seed: Random seed for this experiment
            agent_i: Influencing agent (attacked agent)
            agent_j: Influenced agent (observed agent)
        """
        # Process both high and low influence attacks
        for attack_type, attack_results in [('high', high_attack_results), ('low', low_attack_results)]:
            taylor_errors_history = attack_results['taylor_errors_history']
            episode_length = attack_results['episode_length']
            
            # For each agent, compute Taylor deviations from reference mean
            for agent_id in range(self.nagents):
                timestep_deviation_values = []
                
                for t in range(min(episode_length, len(taylor_errors_history), 
                                  len(ref_vals[agent_id]))):
                    taylor_error = taylor_errors_history[t][agent_id]
                    ref_mean = ref_vals[agent_id][t]
                    # Compute absolute deviation from reference mean
                    deviation = abs(taylor_error - ref_mean)
                    timestep_deviation_values.append(deviation)
                
                # Store the data
                deviation_data = {
                    'seed': seed,
                    'attack_type': attack_type,
                    'attacked_agent_id': agent_i,
                    'observed_agent_id': agent_j,
                    'deviation_agent_id': agent_id,
                    'episode_length': episode_length,
                    'deviation_values': timestep_deviation_values
                }
                
                self.taylor_deviations_data.append(deviation_data)
        
        print(f"Stored Taylor deviations for seed {seed}, pair ({agent_i} -> {agent_j}) "
              f"for {self.nagents} agents (both attack types)")
    
    def save_cumulative_influences_csv(self, logdir):
        """
        Save all cumulative influence data to a single CSV file.
        
        Args:
            logdir: Directory to save the CSV file
        """
        if not self.cumulative_influences_data:
            print("No cumulative influences data to save.")
            return
        
        print("Saving cumulative influences to single CSV file...")
        
        # Find the maximum episode length
        max_episode_length = max(data['episode_length'] 
                                for data in self.cumulative_influences_data)
        
        # Create timestep column names
        timestep_columns = [f'timestep_{t}' for t in range(max_episode_length)]
        
        # Create CSV filename
        csv_filename = 'cumulative_influences_all_seeds.csv'
        csv_filepath = os.path.join(logdir, csv_filename)
        
        # Write to CSV file
        with open(csv_filepath, 'w', newline='') as csvfile:
            fieldnames = ['seed', 'influencer_agent_id', 'influenced_agent_id', 
                         'episode_length'] + timestep_columns
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for data in self.cumulative_influences_data:
                row = {
                    'seed': data['seed'],
                    'influencer_agent_id': data['influencer_agent_id'],
                    'influenced_agent_id': data['influenced_agent_id'],
                    'episode_length': data['episode_length']
                }
                
                # Add cumulative values for each timestep
                for t, value in enumerate(data['cumulative_values']):
                    row[f'timestep_{t}'] = value
                
                # Fill remaining timesteps with empty string if episode is shorter
                for t in range(len(data['cumulative_values']), max_episode_length):
                    row[f'timestep_{t}'] = ''
                
                writer.writerow(row)
        
        total_rows = len(self.cumulative_influences_data)
        print(f"Saved cumulative influences CSV: {csv_filename}")
        print(f"  Total rows: {total_rows}")
        print(f"  Max episode length: {max_episode_length}")
    
    def save_directional_derivatives_csv(self, logdir):
        """
        Save all directional derivatives data to a single CSV file.
        
        Args:
            logdir: Directory to save the CSV file
        """
        if not self.directional_derivatives_data:
            print("No directional derivatives data to save.")
            return
        
        print("Saving directional derivatives to single CSV file...")
        
        # Find the maximum episode length
        max_episode_length = max(data['episode_length'] 
                                for data in self.directional_derivatives_data)
        
        # Create timestep column names
        timestep_columns = [f'timestep_{t}' for t in range(max_episode_length)]
        
        # Create CSV filename
        csv_filename = 'directional_derivatives_all_seeds.csv'
        csv_filepath = os.path.join(logdir, csv_filename)
        
        # Write to CSV file
        with open(csv_filepath, 'w', newline='') as csvfile:
            fieldnames = ['seed', 'influencer_agent_id', 'influenced_agent_id', 
                         'episode_length'] + timestep_columns
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for data in self.directional_derivatives_data:
                row = {
                    'seed': data['seed'],
                    'influencer_agent_id': data['influencer_agent_id'],
                    'influenced_agent_id': data['influenced_agent_id'],
                    'episode_length': data['episode_length']
                }
                
                # Add derivative values for each timestep
                for t, value in enumerate(data['derivative_values']):
                    row[f'timestep_{t}'] = value
                
                # Fill remaining timesteps with empty string if episode is shorter
                for t in range(len(data['derivative_values']), max_episode_length):
                    row[f'timestep_{t}'] = ''
                
                writer.writerow(row)
        
        total_rows = len(self.directional_derivatives_data)
        print(f"Saved directional derivatives CSV: {csv_filename}")
        print(f"  Total rows: {total_rows}")
        print(f"  Max episode length: {max_episode_length}")
    
    def save_taylor_deviations_csv(self, logdir):
        """
        Save all Taylor deviations data to a single CSV file.
        
        Args:
            logdir: Directory to save the CSV file
        """
        if not self.taylor_deviations_data:
            print("No Taylor deviations data to save.")
            return
        
        print("Saving Taylor deviations to single CSV file...")
        
        # Find the maximum episode length
        max_episode_length = max(data['episode_length'] 
                                for data in self.taylor_deviations_data)
        
        # Create timestep column names
        timestep_columns = [f'timestep_{t}' for t in range(max_episode_length)]
        
        # Create CSV filename
        csv_filename = 'taylor_deviations_all_seeds.csv'
        csv_filepath = os.path.join(logdir, csv_filename)
        
        # Write to CSV file
        with open(csv_filepath, 'w', newline='') as csvfile:
            fieldnames = ['seed', 'attack_type', 'attacked_agent_id', 'observed_agent_id',
                         'deviation_agent_id', 'episode_length'] + timestep_columns
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for data in self.taylor_deviations_data:
                row = {
                    'seed': data['seed'],
                    'attack_type': data['attack_type'],
                    'attacked_agent_id': data['attacked_agent_id'],
                    'observed_agent_id': data['observed_agent_id'],
                    'deviation_agent_id': data['deviation_agent_id'],
                    'episode_length': data['episode_length']
                }
                
                # Add deviation values for each timestep
                for t, value in enumerate(data['deviation_values']):
                    row[f'timestep_{t}'] = value
                
                # Fill remaining timesteps with empty string if episode is shorter
                for t in range(len(data['deviation_values']), max_episode_length):
                    row[f'timestep_{t}'] = ''
                
                writer.writerow(row)
        
        total_rows = len(self.taylor_deviations_data)
        print(f"Saved Taylor deviations CSV: {csv_filename}")
        print(f"  Total rows: {total_rows}")
        print(f"  Max episode length: {max_episode_length}")
