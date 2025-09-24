#!/bin/bash		 				
#SBATCH --job-name=Eps_1000_Final_Hatrpo_SMAC_3sVs3z			#Change to your own name
#SBATCH --output=/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/SLURM/out/%j.Eps_1000_Final_Hatrpo_SMAC_3sVs3z.o					#Call it output-<name>-%j.o
#SBATCH --error=/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/SLURM/error/%j.Eps_1000_Final_Hatrpo_SMAC_3sVs3z.e					#Call it error-<name>-%j.e
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
echo " Running final run for shapley with fixed seed and one frobenius computation episode with that fixed seed"
echo "Current timestamp: $(date)"
echo "Running seed 7"
python3 shapley_monte_carlo_smac_newWork.py \
     --restore_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/hatrpo/smac_3s_vs_3z/seed-00001-2025-09-03-22-49-28/models" \
     --env "smac" \
     --algo "hatrpo" \
     --restore_reward "60.176" \
     --dir_name "HATRPO_Final_Result_SMAC_3sVs3z_Eps_1000" \
     --M 100 \
     --seed 7 \
     --compute_frobenius \
     --frobenius_episodes 1000 \
     --save_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/shapley_results"

echo "Running seed 23"
python3 shapley_monte_carlo_smac_newWork.py \
     --restore_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/hatrpo/smac_3s_vs_3z/seed-00001-2025-09-03-22-49-28/models" \
     --env "smac" \
     --algo "hatrpo" \
     --restore_reward "60.176" \
     --dir_name "HATRPO_Final_Result_SMAC_3sVs3z_Eps_1000" \
     --M 100 \
     --seed 23 \
     --compute_frobenius \
     --frobenius_episodes 1000 \
     --save_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/shapley_results"

echo "Running seed 3276"
python3 shapley_monte_carlo_smac_newWork.py \
     --restore_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/hatrpo/smac_3s_vs_3z/seed-00001-2025-09-03-22-49-28/models" \
     --env "smac" \
     --algo "hatrpo" \
     --restore_reward "60.176" \
     --dir_name "HATRPO_Final_Result_SMAC_3sVs3z_Eps_1000" \
     --M 100 \
     --seed 3276 \
     --compute_frobenius \
     --frobenius_episodes 1000 \
     --save_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/shapley_results"

echo "Running seed 13123"
python3 shapley_monte_carlo_smac_newWork.py \
     --restore_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/hatrpo/smac_3s_vs_3z/seed-00001-2025-09-03-22-49-28/models" \
     --env "smac" \
     --algo "hatrpo" \
     --restore_reward "60.176" \
     --dir_name "HATRPO_Final_Result_SMAC_3sVs3z_Eps_1000" \
     --M 100 \
     --seed 13123 \
     --compute_frobenius \
     --frobenius_episodes 1000 \
     --save_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/shapley_results"

echo "Running seed 2343467"
python3 shapley_monte_carlo_smac_newWork.py \
     --restore_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/hatrpo/smac_3s_vs_3z/seed-00001-2025-09-03-22-49-28/models" \
     --env "smac" \
     --algo "hatrpo" \
     --restore_reward "60.176" \
     --dir_name "HATRPO_Final_Result_SMAC_3sVs3z_Eps_1000" \
     --M 100 \
     --seed 2343467 \
     --compute_frobenius \
     --frobenius_episodes 1000 \
     --save_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/shapley_results"

exit 0
