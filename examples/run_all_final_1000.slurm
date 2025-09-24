#!/bin/bash		 				
#SBATCH --job-name=hap.worst.part_1			#Change to your own name
#SBATCH --output=/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/SLURM/out/%j.part_1.hap.worst.o					#Call it output-<name>-%j.o
#SBATCH --error=/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/SLURM/error/%j.part_1.hap.worst.e					#Call it error-<name>-%j.e
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=guptd23@wfu.edu				#Change to your own email
#SBATCH --account=paucaGrp
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=64gb		
#SBATCH --gres=gpu:1			#Change depending on how much Memory you need
#SBATCH --time=20-00:00:00				#4 days - change if you need more time



module load nvidia/cuda11/cuda/11.8.0
module load nvidia/cuda11/cudnn/8.7.0.84

cd /deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
export TORCH_USE_CUDA_DSA=1


echo "Using worst action"
python3 shapley_monte_carlo_smac_K_Final_Work.py \
     --restore_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/happo/smac_3s_vs_3z/seed-13123-2025-09-03-20-10-32/models" \
     --env "smac" \
     --algo "happo" \
     --restore_reward "69.562" \
     --dir_name "worst" \
     --M 100 \
     --total_seeds 1000 \
     --worst_action "worst" \
     --compute_frobenius \
     --frobenius_episodes 1 \
     --output_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/Part-1/happo/worst"


# echo "Using 2nd_best_action action"
# python3 shapley_monte_carlo_smac_K_Final_Work.py \
#      --restore_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/happo/smac_3s_vs_3z/seed-13123-2025-09-03-20-10-32/models" \
#      --env "smac" \
#      --algo "happo" \
#      --restore_reward "69.562" \
#      --dir_name "2nd_worst" \
#      --M 100 \
#      --total_seeds 1000 \
#      --worst_action "2nd_worst" \
#      --compute_frobenius \
#      --frobenius_episodes 1 \
#      --output_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/shapley_results/happo/2nd_worst"



# echo "Using 2nd best action"
# python3 shapley_monte_carlo_smac_K_Final_Work.py \
#      --restore_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/happo/smac_3s_vs_3z/seed-13123-2025-09-03-20-10-32/models" \
#      --env "smac" \
#      --algo "happo" \
#      --restore_reward "69.562" \
#      --dir_name "2nd_best_action" \
#      --M 100 \
#      --total_seeds 1000 \
#      --best_action "2nd_best" \
#      --compute_frobenius \
#      --frobenius_episodes 1 \
#      --output_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/shapley_results/happo/2nd_best_action"


echo "Using 3rd best action"
python3 shapley_monte_carlo_smac_K_Final_Work.py \
     --restore_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/happo/smac_3s_vs_3z/seed-13123-2025-09-03-20-10-32/models" \
     --env "smac" \
     --algo "happo" \
     --restore_reward "69.562" \
     --dir_name "3rd_best_action" \
     --M 100 \
     --total_seeds 1000 \
     --best_action "3rd_best" \
     --compute_frobenius \
     --frobenius_episodes 1 \
     --output_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/shapley_results/happo/3rd_best_action"

exit 0
