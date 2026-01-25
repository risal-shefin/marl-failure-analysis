#!/bin/bash		 				
#SBATCH --job-name=mappo			
#SBATCH --output=/deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/slurm_logs/out/out_rl-%j.o					#Call it output-<name>-%j.o
#SBATCH --error=/deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/slurm_logs/err_rl-%j.e					#Call it error-<name>-%j.e
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shefrs24@wfu.edu				#Change to your own email
#SBATCH --account=alqahtaniGrp
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=80gb					#Change depending on how much Memory you need
#SBATCH --gres=gpu:1
#SBATCH --time=15-00:00:00				#4 days - change if you need more time

module load apps/anaconda3/2024.02
conda activate adversary_loss_marl

module load nvidia/cuda11/cuda/11.8.0
module load nvidia/cuda11/cudnn/8.7.0.84

cd /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/MAPPO_MPE
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
export SC2PATH=/deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/StarCraftII

# pettingzoo
# python MAPPO_MPE_main.py --env_id simple_spread_v3 --evaluate_freq 1 --evaluate_times 10 --max_train_steps 500000000

# Shapley frob computation
# python -m scripts.pettingzoo.shapley_frob_analysis --env_id simple_spread_v3 --model_dir /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/MAPPO_MPE/runs/train_simple_spread_v3_20250730-025008/models/MAPPO_seed_42_score_-18.45.pt --seed 23

# SMAC
python MAPPO_SMAC_main.py --map_name 3m --evaluate_freq 1 --evaluate_times 10 --max_train_steps 500000000

# Shapley frob computation
# python -m scripts.smac.shapley_frob_analysis --map_name 3m --model_dir /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/MAPPO_MPE/runs/train/smac_3m/20250906-015627/models/MAPPO_seed_42_score_24.99.pt --seed 23

# Reference Val Computation
# python -m scripts.compute_ref_taylor_error --env_id simple_spread_v3 --model_dir /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/MAPPO_MPE/runs/train_simple_spread_v3_N5_20250809-030114/models/MAPPO_seed_42_score_-102.64.pt

# TESTING COMMANDS
# python test_detection.py --env_id simple_spread_v3 --model_dir /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/MAPPO_MPE/runs/train_simple_spread_v3_20250730-025008/models/MAPPO_seed_42_score_-18.45.pt --ref_val_dir /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/MAPPO_MPE/data/simple_spread/20250724-233738_refs --attack_agent_id=1 
# python test_detection_by_cluster.py --reference_clusters_dir /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/MAPPO_MPE/data/simple_spread/20250803-155927_cluster_analysis --model_dir /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/MAPPO_MPE/runs/train_simple_spread_v3_20250730-025008/models/MAPPO_seed_42_score_-18.45.pt --attack_agent_id=1
# python test_detection_by_cluster.py --reference_clusters_dir /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/MAPPO_MPE/data/simple_spread/20250803-171228_cluster_analysis --model_dir /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/MAPPO_MPE/runs/train_simple_spread_v3_20250730-025008/models/MAPPO_seed_42_score_-18.45.pt --attack_agent_id=1 
# python test_detection_by_cluster.py --reference_clusters_dir /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/MAPPO_MPE/runs/simple_spread_v3_discrete/20250804-035916_cluster_analysis --model_dir /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/MAPPO_MPE/runs/train_simple_spread_v3_20250730-025008/models/MAPPO_seed_42_score_-18.45.pt --attack_agent_id=1 

# N5
# python test_detection.py --env_id simple_spread_v3 --model_dir /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/MAPPO_MPE/runs/train_simple_spread_v3_N5_20250809-030114/models/MAPPO_seed_42_score_-102.64.pt --ref_val_dir /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/MAPPO_MPE/data/simple_spread/20250809-162515_N5_refs_mad --attack_agent_id=1

# stats
# python -m scripts.compute_detection_metrics --env_id simple_spread_v3 --model_dir /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/MAPPO_MPE/runs/train_simple_spread_v3_N5_20250809-030114/models/MAPPO_seed_42_score_-102.64.pt --ref_val_dir /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/MAPPO_MPE/data/simple_spread/20250816-150325_refs_mmd_n5 --num_episodes 10000

exit 0
