#!/bin/bash		 				
#SBATCH --job-name=maddpg_stats				
#SBATCH --output=/deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/slurm_logs/out/out_rl-%j.o					#Call it output-<name>-%j.o
#SBATCH --error=/deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/slurm_logs/err_rl-%j.e					#Call it error-<name>-%j.e
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shefrs24@wfu.edu				#Change to your own email
#SBATCH --account=alqahtaniGrp
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1					
#SBATCH --gres=gpu:1					#Request 1 GPU
#SBATCH --mem=20gb					#Change depending on how much Memory you need
#SBATCH --time=4-00:00:00				#4 days - change if you need more time

module load apps/anaconda3/2024.02
conda activate adversary_loss_marl

module load nvidia/cuda11/cuda/11.8.0
module load nvidia/cuda11/cudnn/8.7.0.84

cd /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
export SC2PATH=/deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/StarCraftII

python -m scripts.pettingzoo.shapley_frob_analysis_stats simple_spread_v3 /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/simple_spread_v3/maddpg_discrete_tanh_localq/run1/model_-22.824949662226082.pt --max_iterations=5000

# python -m scripts.smac.shapley_frob_analysis_stats 3m /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/3m/maddpg/run2/model_24.427397260273974.pt --max_iterations=1000

exit 0
