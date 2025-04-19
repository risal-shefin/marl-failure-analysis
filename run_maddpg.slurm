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
#SBATCH --nodelist=gpu-a100-04
#SBATCH --time=10-00:00:00				#4 days - change if you need more time

module load apps/anaconda3/2024.02
conda activate adversary_loss_marl

module load nvidia/cuda11/cuda/11.8.0
module load nvidia/cuda11/cudnn/8.7.0.84

cd /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia

# python train_pettingzoo.py simple_speaker_listener_v4 maddpg_continuous
python train_pettingzoo.py simple_reference_v3 maddpg_continuous

# TESTING COMMANDS
# python test_pettingzoo.py simple_speaker_listener_v4 /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/simple_speaker_listener_v4/maddpg_continuous/run1/model_-1.6530647514892638.pt --save_gifs
# python test_pettingzoo.py simple_reference_v3 /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/simple_reference_v3/maddpg_continuous/run1/model_-8.332783496716875.pt --save_gifs

exit 0
