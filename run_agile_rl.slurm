#!/bin/bash		 				
#SBATCH --job-name=gpu-%j				
#SBATCH --output=/deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/slurm_logs/out/out_rl-%j.o					#Call it output-<name>-%j.o
#SBATCH --error=/deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/slurm_logs/err_rl-%j.e					#Call it error-<name>-%j.e
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shefrs24@wfu.edu				#Change to your own email
#SBATCH --account=alqahtaniGrp
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=32gb					#Change depending on how much Memory you need
#SBATCH --gres=gpu:1
#SBATCH --time=20-00:00:00				#4 days - change if you need more time

module load apps/anaconda3/2024.02
conda activate adversary_loss_marl

module load nvidia/cuda11/cuda/11.8.0
module load nvidia/cuda11/cudnn/8.7.0.84

cd /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/AgileRL
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia

python train.py --env_id simple_speaker_listener_v4 --train_steps 20000000 --algo_name MADDPG
# python train_with_evolution.py --env_id simple_speaker_listener_v4 --train_steps 10000000 --algo_name MADDPG
# python test.py --env_id simple_speaker_listener_v4 --algo_name MADDPG --model_dir /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/AgileRL/logs/simple_speaker_listener_v4_MADDPG/20250320_020022/checkpoint.pt

exit 0
