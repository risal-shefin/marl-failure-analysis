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
