#!/bin/bash		 				
#SBATCH --job-name=mappo_stats			
#SBATCH --output=/deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/slurm_logs/out/out_rl-%j.o					#Call it output-<name>-%j.o
#SBATCH --error=/deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/slurm_logs/err_rl-%j.e					#Call it error-<name>-%j.e
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shefrs24@wfu.edu				#Change to your own email
#SBATCH --account=alqahtaniGrp
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8gb					#Change depending on how much Memory you need
#SBATCH --gres=gpu:1
#SBATCH --time=5-00:00:00				#4 days - change if you need more time

module load apps/anaconda3/2024.02
conda activate adversary_loss_marl

module load nvidia/cuda11/cuda/11.8.0
module load nvidia/cuda11/cudnn/8.7.0.84

cd /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/MAPPO_MPE
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
export SC2PATH=/deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/StarCraftII

# Multigrid
python -m scripts.multigrid.fault_analysis_statistics --model_dir /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/MAPPO_MPE/runs/train/collect_20251105-024927/models/MAPPO_seed_42_score_27.00.pt --env_id collect --use_central_q --total_experiments 100

exit 0
