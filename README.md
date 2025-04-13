## Environment Setup
```sh
$module load apps/anaconda3/2024.02
$conda create --name adversary_loss python=3.10
$conda activate adversary_loss
$conda install -c conda-forge mesa glfw glew patchelf
$conda install -c menpo osmesa
$pip install "Cython<3"
$wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
$mkdir -p ~/.mujoco
$tar -xvzf mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/
$pip install -U 'mujoco-py<2.2,>=2.1' gym torch opencv-python matplotlib plotly
# Set Environment Variables
$export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
# Load CUDA Modules
$module load nvidia/cuda11/cuda/11.8.0
$module load nvidia/cuda11/cudnn/8.7.0.84
```

## Example Run:
```sh
$python run.py --configure-env nav2 --exp-data-dir ExperimentalData
```

## Branches
- Debashis: ``git checkout develop-debashis``
- Risal: ``git checkout develop-risal``

## Additional Notes
Delete lines regarding env.seed to avoid errors.

You may need to run the training inside job. Head node might give errors.

# SKRL
```sh
git clone https://github.com/Toni-SM/skrl.git
cd skrl
pip install -e .["all"]
```

# AgileRL
```sh
git clone https://github.com/AgileRL/AgileRL.git && cd AgileRL
pip install -e .
```
I've implemented sample train and test script in AgileRL folder.

# SheepRL Setup:
Follow the upper **Environment Setup** section at first. After that,
```sh
$git clone https://github.com/Eclectic-Sheep/sheeprl.git
$cd sheeprl
$pip install .[atari,dev,mujoco]
```

## SheepRL Example Run:
To run a [PPO](https://openai.com/index/openai-baselines-ppo/) agent on [RiverRaid](https://ale.farama.org/environments/riverraid/) environment:
```sh
# Training command:
$python sheeprl.py exp=ppo env=atari env.id=RiverraidNoFrameskip-v4 algo.cnn_keys.encoder=[rgb] fabric.accelerator=gpu fabric.strategy=ddp fabric.devices=1 algo.mlp_keys.encoder=[]
# Evaluation command
$python sheeprl_eval.py checkpoint_path=logs/runs/ppo/RiverraidNoFrameskip-v4/2025-02-15_21-51-00_ppo_RiverraidNoFrameskip-v4_42/version_0/checkpoint/ckpt_6166016_0.ckpt fabric.accelerator=cpu env.capture_video=True seed=42'
# to evaluate with gradient tracking, append "disable_grads=False" to the evaluation command
```
How to docs: https://github.com/Eclectic-Sheep/sheeprl/tree/main/howto

# Stable Baseline
```sh
git clone https://github.com/DLR-RM/stable-baselines3
cd stable-baselines3
pip install -e .[extra]
```

## Acknowledgements
- AdvExRL (https://github.com/asifurrahman1/AdvEx-RL.git)
- SheepRL (https://github.com/Eclectic-Sheep/sheeprl)
- Stable Baselines 3 (https://github.com/DLR-RM/stable-baselines3)
