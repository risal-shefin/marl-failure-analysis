#!/bin/bash		 				
#SBATCH --job-name=maddpg_detection_stats				
#SBATCH --output=/deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/slurm_logs/out/out_rl-%j.o					#Call it output-<name>-%j.o
#SBATCH --error=/deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/slurm_logs/err_rl-%j.e					#Call it error-<name>-%j.e
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shefrs24@wfu.edu				#Change to your own email
#SBATCH --account=alqahtaniGrp
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1					
#SBATCH --gres=gpu:1					#Request 1 GPU
#SBATCH --nodelist=gpu-v100-03
#SBATCH --mem=8gb					#Change depending on how much Memory you need
#SBATCH --time=3-00:00:00				#3 days - change if you need more time

module load apps/anaconda3/2024.02
conda activate adversary_loss_marl

module load nvidia/cuda11/cuda/11.8.0
module load nvidia/cuda11/cudnn/8.7.0.84

cd /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
export SC2PATH=/deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/StarCraftII

python -m scripts.pettingzoo.multi_seed_statistics simple_spread_v3 /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/simple_spread_v3/maddpg_discrete_tanh_localq/run1/model_-22.824949662226082.pt --total_experiments 500 --taylor_ref_dir /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/data/precomputed_taylor_values/simple_spread_v3_discrete_nagents3_20251006-110437

# Simple Spread with 5 Agents
# python -m scripts.pettingzoo.multi_seed_statistics simple_spread_v3 /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/simple_spread_v3/maddpg_discrete_n5_tanh_localq/run1/model_-118.76917529145851.pt --total_experiments 500 --taylor_ref_dir /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/data/precomputed_taylor_values/simple_spread_v3_discrete_nagents5_20251006-105736

# test
# python -m scripts.pettingzoo.multi_seed_statistics_test simple_spread_v3 /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/simple_spread_v3/maddpg_discrete_tanh_localq/run1/model_-22.824949662226082.pt --total_experiments 100

# SMAC
# 3m
# python -m scripts.smac.multi_seed_statistics 3m /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/3m/maddpg/run2/model_24.427397260273974.pt --total_experiments 100  --precompute_taylor
# 3s_vs_3z
# python -m scripts.smac.multi_seed_statistics 3s_vs_3z /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/3s_vs_3z/maddpg/run1/model_33.64852941176473.pt --total_experiments 100 --taylor_ref_dir /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/data/precomputed_taylor_values/3s_vs_3z_discrete_20251006-184702

exit 0
