#!/bin/bash		 				
#SBATCH --job-name=NoFixed-Happo-1000-frob			#Change to your own name
#SBATCH --output=/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/SLURM/out/NoFixed-Happo-1000-frob.%j.o					#Call it output-<name>-%j.o
#SBATCH --error=/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/SLURM/error/NoFixed-Happo-1000-frob.%j.e					#Call it error-<name>-%j.e
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
echo " Running final run for shapley with non-fixed seed and one frobenius computation episode with 1000 episodes"
echo "Current timestamp: $(date)"
echo "Running seed 7"
python3 shapley_monte_carlo.py \
     --restore_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/happo/Latest_3/seed-00001-2025-08-15-21-25-27/models" \
     --env "pettingzoo_mpe" \
     --algo "happo" \
     --restore_reward "-75.127" \
     --dir_name "F-Happo-Simple_Spread_V3-No-Fixed" \
     --save_gifs \
     --M 1000 \
     --seed 7 \
     --compute_frobenius \
     --frobenius_episodes 1000 \
     --save_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/shapley_results"

echo "Running seed 23"
python3 shapley_monte_carlo.py \
     --restore_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/happo/Latest_3/seed-00001-2025-08-15-21-25-27/models" \
     --env "pettingzoo_mpe" \
     --algo "happo" \
     --restore_reward "-75.127" \
     --dir_name "F-Happo-Simple_Spread_V3-No-Fixed" \
     --save_gifs \
     --M 1000 \
     --seed 23 \
     --compute_frobenius \
     --frobenius_episodes 1000 \
     --save_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/shapley_results"

echo "Running seed 3276"
python3 shapley_monte_carlo.py \
     --restore_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/happo/Latest_3/seed-00001-2025-08-15-21-25-27/models" \
     --env "pettingzoo_mpe" \
     --algo "happo" \
     --restore_reward "-75.127" \
     --dir_name "F-Happo-Simple_Spread_V3-No-Fixed" \
     --save_gifs \
     --M 1000 \
     --seed 3276 \
     --compute_frobenius \
     --frobenius_episodes 1000 \
     --save_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/shapley_results"

echo "Running seed 13123"
python3 shapley_monte_carlo.py \
     --restore_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/happo/Latest_3/seed-00001-2025-08-15-21-25-27/models" \
     --env "pettingzoo_mpe" \
     --algo "happo" \
     --restore_reward "-75.127" \
     --dir_name "F-Happo-Simple_Spread_V3-No-Fixed" \
     --save_gifs \
     --M 1000 \
     --seed 13123 \
     --compute_frobenius \
     --frobenius_episodes 1000 \
     --save_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/shapley_results"

echo "Running seed 2343467"
python3 shapley_monte_carlo.py \
     --restore_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/happo/Latest_3/seed-00001-2025-08-15-21-25-27/models" \
     --env "pettingzoo_mpe" \
     --algo "happo" \
     --restore_reward "-75.127" \
     --dir_name "F-Happo-Simple_Spread_V3-No-Fixed" \
     --save_gifs \
     --M 1000 \
     --seed 2343467 \
     --compute_frobenius \
     --frobenius_episodes 1000 \
     --save_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/shapley_results"

exit 0
