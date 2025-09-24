#!/bin/bash		 				
#SBATCH --job-name=2nd_best.hatrpo.final_part1.smac			#Change to your own name
#SBATCH --output=/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/SLURM/out/%j.2nd_best.hatrpo.final_part1.smac.o					#Call it output-<name>-%j.o
#SBATCH --error=/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/SLURM/error/%j.2nd_best.hatrpo.final_part1.smac.e					#Call it error-<name>-%j.e
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=guptd23@wfu.edu				#Change to your own email
#SBATCH --account=paucaGrp
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb		
#SBATCH --gres=gpu:1			#Change depending on how much Memory you need
#SBATCH --time=5-00:00:00				#4 days - change if you need more time



module load nvidia/cuda11/cuda/11.8.0
module load nvidia/cuda11/cudnn/8.7.0.84

cd /deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
export TORCH_USE_CUDA_DSA=1




# echo "Using worst happo action"
# python3 shapley_monte_carlo_smac_K_Final_Work.py \
#      --restore_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/happo/Latest_3/seed-00001-2025-08-15-21-25-27/models" \
#      --env "pettingzoo_mpe" \
#      --algo "happo" \
#      --restore_reward "-75.127" \
#      --dir_name "worst" \
#      --M 40 \
#      --total_seeds 1000 \
#      --worst_action "worst" \
#      --compute_frobenius \
#      --frobenius_episodes 1 \
#      --output_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/Part-1/pettingzoo_mpe/happo/worst"


# echo "Using worst action happo"
# python3 shapley_monte_carlo_smac_K_Final_Work.py \
#      --restore_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/happo/Latest_3/seed-00001-2025-08-15-21-25-27/models" \
#      --env "pettingzoo_mpe" \
#      --algo "happo" \
#      --restore_reward "-75.127" \
#      --dir_name "2nd_best" \
#      --M 40 \
#      --total_seeds 1000 \
#      --best_action "2nd_best" \
#      --compute_frobenius \
#      --frobenius_episodes 1 \
#      --output_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/Part-1/pettingzoo_mpe/happo/2nd_best"

# echo "Using worst hatrpo action"
# python3 shapley_monte_carlo_smac_K_Final_Work.py \
#      --restore_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/hatrpo/Latest_3/seed-00001-2025-08-15-22-56-55/models" \
#      --env "pettingzoo_mpe" \
#      --algo "hatrpo" \
#      --restore_reward "-79.879" \
#      --dir_name "worst" \
#      --M 40 \
#      --total_seeds 1000 \
#      --worst_action "worst" \
#      --compute_frobenius \
#      --frobenius_episodes 1 \
#      --output_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/Part-1/pettingzoo_mpe/hatrpo/worst"


# echo "Using 2nd_best action hatrpo"
# python3 shapley_monte_carlo_smac_K_Final_Work.py \
#      --restore_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/hatrpo/Latest_3/seed-00001-2025-08-15-22-56-55/models" \
#      --env "pettingzoo_mpe" \
#      --algo "hatrpo" \
#      --restore_reward "-79.879" \
#      --dir_name "2nd_best" \
#      --M 40 \
#      --total_seeds 1000 \
#      --best_action "2nd_best" \
#      --compute_frobenius \
#      --frobenius_episodes 1 \
#      --output_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/Part-1/pettingzoo_mpe/hatrpo/2nd_best"

# echo "Using worst happo action smac"
# python3 shapley_monte_carlo_smac_K_Final_Work.py \
#      --restore_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/happo/smac_3s_vs_3z/seed-13123-2025-09-03-20-10-32/models" \
#      --env "smac" \
#      --algo "happo" \
#      --restore_reward "69.562" \
#      --dir_name "worst" \
#      --M 40 \
#      --total_seeds 1000 \
#      --worst_action "worst" \
#      --compute_frobenius \
#      --frobenius_episodes 1 \
#      --output_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/Part-1/smac/happo/worst"



# echo "Using 2nd best happo action smac"
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
#      --output_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/Part-1/smac/happo/2nd_best_action"


# echo "Using 3rd best action"
# python3 shapley_monte_carlo_smac_K_Final_Work.py \
#      --restore_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/happo/smac_3s_vs_3z/seed-13123-2025-09-03-20-10-32/models" \
#      --env "smac" \
#      --algo "happo" \
#      --restore_reward "69.562" \
#      --dir_name "3rd_best_action" \
#      --M 100 \
#      --total_seeds 1000 \
#      --best_action "3rd_best" \
#      --compute_frobenius \
#      --frobenius_episodes 1 \
#      --output_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/Part-1/happo/happo/3rd_best_action"


# echo "Using worst hatrpo action smac"
# python3 shapley_monte_carlo_smac_K_Final_Work.py \
#      --restore_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/hatrpo/smac_3s_vs_3z/seed-00001-2025-09-03-22-49-28/models" \
#      --env "smac" \
#      --algo "hatrpo" \
#      --restore_reward "60.176" \
#      --dir_name "worst" \
#      --M 40 \
#      --total_seeds 1000 \
#      --worst_action "worst" \
#      --compute_frobenius \
#      --frobenius_episodes 1 \
#      --output_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/Part-1/smac/hatrpo/worst"



echo "Using 2nd best hatrpo action"
python3 shapley_monte_carlo_smac_K_Final_Work.py \
     --restore_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/hatrpo/smac_3s_vs_3z/seed-00001-2025-09-03-22-49-28/models" \
     --env "smac" \
     --algo "hatrpo" \
     --restore_reward "60.176" \
     --dir_name "2nd_best_action" \
     --M 40 \
     --total_seeds 1000 \
     --best_action "2nd_best" \
     --compute_frobenius \
     --frobenius_episodes 1 \
     --output_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/Part-1/smac/hatrpo/2nd_best_action"

exit 0
