# Patient Zero Traceback Module

This module implements the patient zero detection and influence chain traceback algorithm following the provided pseudocode. It provides a comprehensive analysis of multi-agent systems to identify the true source of anomalies by tracing back through influence chains.

## Features

- **Patient Zero Detection**: Automatically detects initial patient zero agents from fault timelines
- **Tie-Breaking**: Uses Taylor deviation to select among multiple agents detected simultaneously  
- **Influence Chain Traceback**: Recursively traces back influence chains using action influences and Frobenius norms
- **Cycle Detection**: Detects and handles cycles in influence chains
- **Comprehensive Logging**: Logs all results to CSV files with seed information for manual verification
- **Batch Processing**: Supports analysis of multiple episodes/seeds
- **Integration Ready**: Designed to be easily imported into existing codebases

## Installation

The module is located in the `modules/` directory. Simply import it into your Python scripts:

```python
from modules.patient_zero_traceback import PatientZeroTracebackAnalyzer, analyze_single_episode
```

## Usage

### Basic Single Episode Analysis

```python
from modules.patient_zero_traceback import analyze_single_episode

# Analyze a single episode
result = analyze_single_episode(
    fault_timeline=fault_timeline,
    action_influences_history=action_influences_history,
    pairwise_frob_norms_history=pairwise_frob_norms_history,
    taylor_history=taylor_history,
    num_agents=num_agents,
    seed=seed,
    episode_length=episode_length,
    log_dir='./traceback_logs'
)

print(f"True patient zero: {result['true_patient_zero']}")
print(f"Influence chain: {result['influence_chain']}")
```

### Batch Analysis

```python
from modules.patient_zero_traceback import batch_analyze_episodes

# Prepare episode data
episodes_data = [
    {
        'fault_timeline': episode1_fault_timeline,
        'action_influences_history': episode1_influences,
        'pairwise_frob_norms_history': episode1_frob_norms,
        'taylor_history': episode1_taylor,
        'num_agents': 3,
        'seed': 123,
        'episode_length': 50
    },
    # ... more episodes
]

# Analyze all episodes
results = batch_analyze_episodes(episodes_data, log_dir='./traceback_logs')
```

### Advanced Usage with Statistics

```python
from modules.patient_zero_traceback import PatientZeroTracebackAnalyzer

# Create analyzer
analyzer = PatientZeroTracebackAnalyzer(log_dir='./traceback_logs')

# Analyze multiple episodes
for seed in range(100, 200):
    result = analyzer.analyze_patient_zero_traceback(
        fault_timeline, action_influences_history, pairwise_frob_norms_history,
        taylor_history, num_agents, seed, episode_length
    )

# Get summary statistics
stats = analyzer.get_summary_statistics()
print(f"Detection rate: {stats['detection_rate']:.2%}")
print(f"Average chain length: {stats['average_chain_length']:.2f}")

# Save all results to CSV
analyzer.save_results_to_csv()
```

## Data Structures

### Input Data Formats

1. **fault_timeline**: List of fault detection events
   ```python
   [
       {'agent': 0, 't': 15, 'severity': 0.8},
       {'agent': 1, 't': 17, 'severity': 0.6}
   ]
   ```

2. **action_influences_history**: List of action influence matrices over time
   ```python
   [
       [[0.0, 0.1, 0.2], [0.3, 0.0, 0.1], [0.2, 0.3, 0.0]],  # timestep 0
       [[0.0, 0.2, 0.1], [0.2, 0.0, 0.2], [0.1, 0.4, 0.0]],  # timestep 1
       # ... more timesteps
   ]
   ```
   Format: `action_influences_history[t][influenced_agent][influencing_agent]`

3. **pairwise_frob_norms_history**: List of Frobenius norm matrices over time
   ```python
   [
       [[0.0, 0.5, 0.3], [0.4, 0.0, 0.6], [0.2, 0.7, 0.0]],  # timestep 0
       # ... more timesteps
   ]
   ```
   Format: `pairwise_frob_norms_history[t][influenced_agent][influencing_agent]`

4. **taylor_history**: List of Taylor error histories for each agent
   ```python
   [
       [0.01, 0.02, 0.15, 0.8, 0.3, ...],  # agent 0 Taylor errors over time
       [0.02, 0.01, 0.02, 0.02, 0.1, ...], # agent 1 Taylor errors over time
       # ... more agents
   ]
   ```
   Format: `taylor_history[agent_id][timestep]`

### Output Data Formats

The analysis returns a dictionary with the following structure:

```python
{
    'seed': 123,
    'episode_length': 50,
    'num_agents': 3,
    'initial_patient_zeros': [0, 1],  # Multiple agents detected at same time
    'initial_detection_time': 15,
    'selected_patient_zero': 0,       # Selected using Taylor deviation tie-breaking
    'true_patient_zero': 2,           # Final result after traceback
    'influence_chain_length': 3,
    'influence_chain': [2, 1, 0],     # Chain from true source to detected agent
    'cycle_detected': False,
    'max_taylor_deviation': 0.8,
    'analysis_timestamp': '2025-10-05T...'
}
```

## Algorithm Details

The module implements the pseudocode algorithm in three main steps:

### Step 1: Initialization
- Detect initial patient zero agents from fault timeline
- If multiple agents detected simultaneously, use Taylor deviation for tie-breaking
- Get detection time for analysis window

### Step 2: Recursive Traceback
- For each agent in the current chain, find the most influential predecessor
- Compute `Dij_rate` (positive influence rate) as primary metric
- Use `Gij_norm` (Frobenius norm) for tie-breaking
- Stop if cycle detected or no influential agent found

### Step 3: Finalize True Patient Zero
- Reverse the influence chain to get source-to-detected order
- The first agent in the reversed chain is the true patient zero

## CSV Output Files

The module generates two CSV files:

### 1. `patient_zero_traceback_results.csv`
Contains summary results for each analyzed episode:
- `seed`: Random seed identifier
- `episode_length`: Length of the episode
- `num_agents`: Number of agents in the system
- `initial_patient_zeros`: Initially detected agents
- `initial_detection_time`: When detection occurred
- `selected_patient_zero`: Agent selected after tie-breaking
- `true_patient_zero`: Final result after traceback
- `influence_chain_length`: Length of the influence chain
- `influence_chain`: Complete influence chain
- `cycle_detected`: Whether a cycle was detected
- `max_taylor_deviation`: Maximum Taylor deviation used in tie-breaking
- `analysis_timestamp`: When analysis was performed

### 2. `influence_chain_details.csv`
Contains detailed influence computations for each step:
- `seed`: Random seed identifier
- `chain_step`: Step number in the traceback process
- `current_agent`: Agent being analyzed
- `candidate_agent`: Potential influencing agent
- `dij_rate`: Computed Dij rate (primary metric)
- `gij_norm`: Computed Gij norm (tie-breaker)
- `timestep_window_start`: Start of analysis window
- `timestep_window_end`: End of analysis window
- `selected_as_most_influential`: Whether this agent was selected
- `analysis_timestamp`: When analysis was performed

## Integration with Existing Code

### With K_multi_seed_statistics_smac_actionBased.py

The module is designed to integrate seamlessly with existing experiment code. See `enhanced_multi_seed_with_traceback.py` for a complete integration example.

```python
from modules.patient_zero_traceback import PatientZeroTracebackAnalyzer

class EnhancedExperimentRunner(MultiSeedExperimentRunner):
    def __init__(self, config):
        super().__init__(config)
        self.patient_zero_analyzer = PatientZeroTracebackAnalyzer(
            log_dir=os.path.join(self.logdir, 'patient_zero_traceback')
        )
    
    def run_experiment_with_traceback(self, seed):
        # Run original experiment
        episode_results = self.eval(runner, seed=seed)
        
        # Extract data and perform traceback analysis
        patient_zero_result = self.patient_zero_analyzer.analyze_patient_zero_traceback(
            episode_results['fault_timeline'],
            episode_results['action_influences_history'],
            episode_results['pairwise_frob_norms_history'],
            episode_results['taylor_history'],
            runner.num_agents,
            seed,
            episode_results['episode_length']
        )
        
        return patient_zero_result
```

## Examples

Run the provided examples to see the module in action:

```bash
# Basic example with dummy data
python patient_zero_traceback_example.py

# Integration example with existing code
python enhanced_multi_seed_with_traceback.py --total_experiments 50 --K_SIGMA 2
```

## Configuration Parameters

- `lambda_decay`: Exponential decay factor for influence rate computation (default: 0.1)
- `window_size`: Size of timestep window for influence analysis (default: 5)
- `log_dir`: Directory for saving CSV logs

## Summary Statistics

The analyzer provides comprehensive summary statistics:
- `total_episodes_analyzed`: Total number of episodes processed
- `episodes_with_patient_zero`: Episodes where patient zero was detected
- `detection_rate`: Percentage of episodes with successful detection
- `cycles_detected`: Number of episodes with cycles in influence chains
- `cycle_rate`: Percentage of episodes with cycles
- `average_chain_length`: Average length of influence chains
- `max_chain_length`: Maximum observed chain length
- `min_chain_length`: Minimum observed chain length

## Manual Verification

The seed information in all CSV files allows for manual verification of results:
1. Filter CSV files by specific seed values
2. Examine the influence chain details for that seed
3. Verify the traceback logic and influence computations
4. Cross-reference with original episode data

This makes it easy to debug the algorithm and validate results on specific episodes of interest.