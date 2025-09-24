#!/bin/bash		 				
#SBATCH --job-name=sim.happo.3rd_best			#Change to your own name
#SBATCH --output=/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/SLURM/out/%j.sim.happo.3rd_best.o					#Call it output-<name>-%j.o
#SBATCH --error=/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/SLURM/error/%j.sim.happo.3rd_best.e					#Call it error-<name>-%j.e
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=guptd23@wfu.edu				#Change to your own email
#SBATCH --account=paucaGrp
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb		
#SBATCH --gres=gpu:1			#Change depending on how much Memory you need
#SBATCH --time=20-00:00:00				#4 days - change if you need more time



module load nvidia/cuda11/cuda/11.8.0
module load nvidia/cuda11/cudnn/8.7.0.84

cd /deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
export TORCH_USE_CUDA_DSA=1


# echo "Using worst action"
# python3 shapley_monte_carlo_smac_K_Final_Work.py \
#      --restore_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/happo/Latest_3/seed-00001-2025-08-15-21-25-27/models" \
#      --env "pettingzoo_mpe" \
#      --algo "happo" \
#      --restore_reward "-75.127" \
#      --dir_name "worst_simple_happo" \
#      --M 100 \
#      --total_seeds 5000 \
#      --worst_action "worst" \
#      --compute_frobenius \
#      --frobenius_episodes 1 \
#      --output_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/shapley_results/simple_spread3/happo/worst"


# echo "Using 2nd_worst action"
# python3 shapley_monte_carlo_smac_K_Final_Work.py \
#      --restore_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/happo/Latest_3/seed-00001-2025-08-15-21-25-27/models" \
#      --env "pettingzoo_mpe" \
#      --algo "happo" \
#      --restore_reward "-75.127" \
#      --dir_name "2nd_worst_simple_happo" \
#      --M 100 \
#      --total_seeds 5000 \
#      --worst_action "2nd_worst" \
#      --compute_frobenius \
#      --frobenius_episodes 1 \
#      --output_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/shapley_results/simple_spread3/happo/2nd_worst"



# echo "Using 2nd_best action"
# python3 shapley_monte_carlo_smac_K_Final_Work.py \
#      --restore_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/happo/Latest_3/seed-00001-2025-08-15-21-25-27/models" \
#      --env "pettingzoo_mpe" \
#      --algo "happo" \
#      --restore_reward "-75.127" \
#      --dir_name "2nd_best_action_simple_happo" \
#      --M 100 \
#      --total_seeds 5000 \
#      --best_action "2nd_best" \
#      --compute_frobenius \
#      --frobenius_episodes 1 \
#      --output_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/shapley_results/simple_spread3/happo/2nd_best_action"

echo "Using 3rd_best action"
python3 shapley_monte_carlo_smac_K_Final_Work.py \
     --restore_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/happo/Latest_3/seed-00001-2025-08-15-21-25-27/models" \
     --env "pettingzoo_mpe" \
     --algo "happo" \
     --restore_reward "-75.127" \
     --dir_name "3rd_best_action_simple_happo" \
     --M 100 \
     --total_seeds 5000 \
     --best_action "3rd_best" \
     --compute_frobenius \
     --frobenius_episodes 1 \
     --output_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/shapley_results/simple_spread3/happo/3rd_best_action"


# #### HATRPO ################################

# echo "Using worst action"
# python3 shapley_monte_carlo_smac_K_Final_Work.py \
#      --restore_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/hatrpo/Latest_3/seed-00001-2025-08-15-22-56-55/models" \
#      --env "pettingzoo_mpe" \
#      --algo "hatrpo" \
#      --restore_reward "-79.879" \
#      --dir_name "worst_simple_hatrpo" \
#      --M 100 \
#      --total_seeds 5000 \
#      --worst_action "worst" \
#      --compute_frobenius \
#      --frobenius_episodes 1 \
#      --output_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/shapley_results/simple_spread3/hatrpo/worst"


# echo "Using 2nd_worst action"
# python3 shapley_monte_carlo_smac_K_Final_Work.py \
#      --restore_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/hatrpo/Latest_3/seed-00001-2025-08-15-22-56-55/models" \
#      --env "pettingzoo_mpe" \
#      --algo "hatrpo" \
#      --restore_reward "-79.879" \
#      --dir_name "2nd_worst_simple_hatrpo" \
#      --M 100 \
#      --total_seeds 5000 \
#      --worst_action "2nd_worst" \
#      --compute_frobenius \
#      --frobenius_episodes 1 \
#      --output_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/shapley_results/simple_spread3/hatrpo/2nd_worst"



# echo "Using 2nd_best action"
# python3 shapley_monte_carlo_smac_K_Final_Work.py \
#      --restore_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/hatrpo/Latest_3/seed-00001-2025-08-15-22-56-55/models" \
#      --env "pettingzoo_mpe" \
#      --algo "hatrpo" \
#      --restore_reward "-79.879" \
#      --dir_name "2nd_best_action_simple_hatrpo" \
#      --M 100 \
#      --total_seeds 5000 \
#      --best_action "2nd_best" \
#      --compute_frobenius \
#      --frobenius_episodes 1 \
#      --output_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/shapley_results/simple_spread3/hatrpo/2nd_best_action"

# echo "Using 3rd_best action"
# python3 shapley_monte_carlo_smac_K_Final_Work.py \
#      --restore_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/hatrpo/Latest_3/seed-00001-2025-08-15-22-56-55/models" \
#      --env "pettingzoo_mpe" \
#      --algo "hatrpo" \
#      --restore_reward "-79.879" \
#      --dir_name "3rd_best_action_simple_hatrpo" \
#      --M 100 \
#      --total_seeds 5000 \
#      --best_action "3rd_best" \
#      --compute_frobenius \
#      --frobenius_episodes 1 \
#      --output_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/shapley_results/simple_spread3/hatrpo/3rd_best_action"

exit 0
