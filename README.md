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

## ENVs
- MPE: https://pettingzoo.farama.org/environments/mpe/
- Multigrid: https://github.com/ArnaudFickinger/gym-multigrid
- VMAS: https://vmas.readthedocs.io/
- SMAC: https://github.com/oxwhirl/smac/tree/master

## Acknowledgements
- MADDPG: https://github.com/shariqiqbal2810/maddpg-pytorch
- MAPPO: https://github.com/Lizhi-sjtu/MARL-code-pytorch/tree/main/1.MAPPO_MPE
- AdvExRL (https://github.com/asifurrahman1/AdvEx-RL.git)
- SheepRL (https://github.com/Eclectic-Sheep/sheeprl)
- Stable Baselines 3 (https://github.com/DLR-RM/stable-baselines3)
