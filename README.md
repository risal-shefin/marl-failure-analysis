This is the code repository of the papers **"Interpretable Failure Analysis in Multi-Agent Reinforcement Learning Systems"** and **"When One Agent Reshapes Another: Cross-Hessian Interpretation of Local Coupling in MARL"**.

## Environment Setup
```sh
$conda create --name xmarl python=3.10
$conda activate xmarl
$pip install -r requirements.txt
$conda install -c conda-forge mesa glfw glew patchelf
$conda install -c menpo osmesa
$pip install "Cython<3"
$wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
$mkdir -p ~/.mujoco
$tar -xvzf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/
$pip install -U 'mujoco-py<2.2,>=2.1' gym torch opencv-python matplotlib plotly
# Set Environment Variables
$export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
```

## MADDPG
Set directory: <br>
```sh
$cd maddpg-pytorch
```
### Model Training
Simple Spread: <br>
```sh
$python train_pettingzoo.py simple_spread_v3 maddpg_discrete --discrete_action True
```
SMAC: <br>
```sh
$python train_smac.py 3s_vs_3z maddpg
```
### Failure Analysis
Simple Spread: <br>
```sh
$python -m scripts.pettingzoo.multi_seed_statistics simple_spread_v3 <model_path> --total_experiments 500
```
SMAC: <br>
```sh
$python -m scripts.smac.multi_seed_statistics 3s_vs_3z <model_path> --total_experiments 100
```

### Cross-Hessian Frobenius Norm Analysis
This module provides two PettingZoo experiments:

1) **Single-Episode Visualization + Frobenius Logging** (`episode_gif_frob_norm`)

Runs one seeded deterministic episode and records per-timestep cross-Hessian Frobenius norms for every ordered agent pair.

```sh
$python -m scripts.pettingzoo.frob_norm_experiments episode_gif_frob_norm simple_spread_v3 <model_path> --seed 0
```

What it does:
- Computes pairwise cross-Hessian Frobenius norms at each timestep.
- Saves an episode GIF for qualitative trajectory inspection.
- Writes a detailed per-timestep text log with per-pair max/min and overall max/min.
- Saves a heatmap over timesteps (agent-pair rows, timestep columns).

**Outputs (under** `runs/frob_norm_experiments/episode_gif_frob_norm/...`**):**
- `episode_seed<seed>.gif`
- `frob_norms_seed<seed>.txt`
- `frob_norm_heatmap_seed<seed>.png`

2) **Multi-Seed SVD Coupling Analysis** (`grad_shift_svd_coupling`)

Analyzes how cross-Hessian strength relates to policy-gradient sensitivity under SVD-directed perturbations (continuous-action models only).

```sh
$python -m scripts.pettingzoo.frob_norm_experiments grad_shift_svd_coupling simple_spread_v3 <model_path> --epsilon 0.01 --total_experiments 100
```

What it does:
- Computes cross-Hessian $H = \nabla_{a_j} \nabla_{a_i} Q_i$ for all ordered agent pairs.
- Uses SVD directions to perturb agent $j$ and measures $\|\Delta g\|_2$ for agent $i$.
- Replays perturbed episodes to record reward impact for the affected agent.
- Generates per-pair and combined scatter plots with linear fits and Pearson correlation.

**Outputs (under** `runs/frob_norm_experiments/grad_shift_svd_coupling/...`**):**
- `csv_data/raw_coupling_data.csv` (per seed/timestep/pair records)
- `csv_data/mean_coupling_by_pair.csv` (aggregated pair statistics)
- `plots/svd_coupling_frob_vs_deltag_all_pairs.png` (combined plot)
- `plots/svd_coupling_pair_<agent_j>_to_<agent_i>.png` (individual pair plots)

## HARL
### Installation / Preparation
1. Clone and install HARL (https://github.com/PKU-MARL/HARL) and confirm its example scripts run.
2. Place the trained model checkpoints into the `models/` directory. Checkpoint files for the included experiments are already present under `models/simple_spread_N_3/` and `models/simple_spread_N_5/`.
3. (Optional) Create a virtual environment and install the package requirements used for analysis.

Set directory: <br>
```sh
$cd harl_container/examples
```

### Running Evaluations

1) Simple Spread V3 — N = 3

```bash
python3 K_multi_seed_statistics_simple_new_actionBased.py \
    --N 3 \
    --total_experiments 500 \
    --total_episodes 100 \
    --filepath "models/simple_spread_N_3/" \
    --reward -274.358 \
    --K_SIGMA 3 \
    --folder_name "output_simple_spread_N3"
```

2) Simple Spread V3 — N = 5

```bash
python3 K_multi_seed_statistics_simple_new_actionBased.py \
    --N 5 \
    --total_experiments 500 \
    --total_episodes 100 \
    --filepath "models/simple_spread_N_5/" \
    --reward -274.358 \
    --K_SIGMA 3 \
    --folder_name "output_simple_spread_N5"
```

3) SMAC 3svs3z

```bash
python3 K_multi_seed_statistics_smac_actionBased.py \
    --total_experiments 500 \
    --total_episodes 100 \
    --filepath "models/<smac_model_folder>/" \
    --reward -274.358 \
    --K_SIGMA 3 \
    --folder_name "output_smac_3svs3z"
```

Replace `--filepath` with the path to the checkpoint directory containing agent model files (for example `models/simple_spread_N_3/`). The scripts will load the actor/critic checkpoints by the naming convention used in this repository (`actor_agent{i}_<score>.pt`, `critic_agent_<score>.pt`, etc.).

### Output
The evaluation scripts produce a folder (controlled by `--folder_name`) containing:
- CSV summaries (per-experiment and aggregated): `accuracy_results.csv`, `detailed_results.csv`, `failed_expectations.csv`.
- Influence and derivative analyses: `cumulative_influences_all_seeds.csv`, `directional_derivatives_all_seeds.csv`, `taylor_deviations_all_seeds.csv`.
- Patient-zero analysis and per-pair detailed files under `pair_specific_results/`.

### Cross-Hessian Frobenius Norm Analysis
Set directory: <br>
```sh
$cd harl_frob_container
```

1) **Single-Episode Visualization + Frobenius Logging** (`episode_gif_frob_norm`)

Runs one seeded deterministic episode and records per-timestep cross-Hessian Frobenius norms for every ordered agent pair.

```sh
$python examples.pettingzoo.frob_norm_experiments episode_gif_frob_norm \
    --algo hatrpo \
    --scenario simple_spread_v3 \
    --model_dir <path_to_model_checkpoint_dir> \
    --seed 0
```

What it does:
- Computes pairwise cross-Hessian Frobenius norms at each timestep: $H_{ij} = \left\| \frac{\partial^2 Q}{\partial a_i \partial a_j} \right\|_F$
- Saves an episode GIF for qualitative trajectory inspection.
- Writes a detailed per-timestep text log with per-pair max/min and overall max/min.
- Saves a heatmap over timesteps (agent-pair rows, timestep columns).

**Outputs (under** `runs/frob_norm_experiments/episode_gif_frob_norm/...`**):**
- `episode_seed<seed>.gif`
- `frob_norms_seed<seed>.txt`
- `frob_norm_heatmap_seed<seed>.png`

2) **Multi-Seed SVD Coupling Analysis** (`grad_shift_svd_coupling`)

Analyzes how cross-Hessian strength relates to policy-gradient sensitivity under SVD-directed perturbations.

```sh
$python examples.pettingzoo.frob_norm_experiments grad_shift_svd_coupling \
    --algo haddpg \
    --scenario simple_spread_v3 \
    --model_dir <path_to_model_checkpoint_dir> \
    --epsilon 0.01 \
    --total_experiments 100
```

What it does:
- Computes cross-Hessian $H_{ij} = \frac{\partial^2 Q}{\partial a_i \partial a_j}$ for all ordered agent pairs.
- Uses SVD top right singular vector to perturb agent $j$ and measures induced gradient shift $\|\Delta g\|_2$ for agent $i$.
- Replays perturbed episodes to record reward impact for the affected agent.
- Generates per-pair and combined scatter plots with linear fits and Pearson correlation.

**Outputs (under** `runs/frob_norm_experiments/grad_shift_svd_coupling/...`**):**
- `csv_data/raw_coupling_data.csv` (per seed/timestep/pair records)
- `csv_data/mean_coupling_by_pair.csv` (aggregated pair statistics)
- `plots/svd_coupling_frob_vs_deltag_all_pairs.png` (combined plot)
- `plots/svd_coupling_pair_<agent_j>_to_<agent_i>.png` (individual pair plots)

## Acknowledgements
- MADDPG: https://github.com/shariqiqbal2810/maddpg-pytorch
- HARL: https://github.com/PKU-MARL/HARL
- MAPPO: https://github.com/Lizhi-sjtu/MARL-code-pytorch/tree/main/1.MAPPO_MPE
- AdvExRL (https://github.com/asifurrahman1/AdvEx-RL.git)
- SheepRL (https://github.com/Eclectic-Sheep/sheeprl)
- Stable Baselines 3 (https://github.com/DLR-RM/stable-baselines3)
### ENVs
- MPE: https://pettingzoo.farama.org/environments/mpe/
- Multigrid: https://github.com/ArnaudFickinger/gym-multigrid
- VMAS: https://vmas.readthedocs.io/
- SMAC: https://github.com/oxwhirl/smac/tree/master
