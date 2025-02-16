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
python sheeprl.py exp=ppo env=atari env.id=RiverraidNoFrameskip-v4 algo.cnn_keys.encoder=[rgb] fabric.accelerator=gpu fabric.strategy=ddp fabric.devices=1 algo.mlp_keys.encoder=[]
```
How to docs: https://github.com/Eclectic-Sheep/sheeprl/tree/main/howto

## Acknowledgements
- AdvExRL (https://github.com/asifurrahman1/AdvEx-RL.git)
- SheepRL (https://github.com/Eclectic-Sheep/sheeprl)
