#!/bin/bash		 				
#SBATCH --job-name=3m.Act.K3.E.100.F.100.R.Prev_Influence			#Change to your own name
#SBATCH --output=/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/Krishna-Result-Slurm/out/%j.3m.Act.K3.E.100.F.100.R.Prev_Influence.o					#Call it output-<name>-%j.o
#SBATCH --error=/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/Krishna-Result-Slurm/error/%j.3m.Act.K3.E.100.F.100.R.Prev_Influence.e					#Call it error-<name>-%j.eSBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=guptd23@wfu.edu				#Change to your own email
#SBATCH --account=paucaGrp
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=32gb
#SBATCH --gres=gpu:A100_40:1		#Change depending on how much Memory you need
#SBATCH --time=4-00:00:00				#4 days - change if you need more time



module load nvidia/cuda11/cuda/11.8.0
module load nvidia/cuda11/cudnn/8.7.0.84

cd /deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
export TORCH_USE_CUDA_DSA=1

# python compute_taylor_ref.py --algo happo --env pettingzoo_mpe

# python attack_test.py --algo happo --env pettingzoo_mpe

# python train.py --algo hatrpo --env pettingzoo_mpe
# python train.py --algo hatrpo --env smac --exp_name q_smac-True --use_centralized_q True
# python train.py --algo hatrpo --env pettingzoo_mpe --exp_name q_five_agent --use_centralized_q True

# echo "Running smac with K=1"
# python3 K_multi_seed_statistics_smac.py --total_experiments 500 \
#     --total_episodes 100 \
#     --filepath "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/hatrpo/q_smac-True/seed-00001-2025-09-27-21-59-35/models" \
#     --reward 60.036 \
#     --K_SIGMA 1

# echo "Running smac with K=3 for 100 experiments 100 episodes 3s_vs_3z with New Q with Previous Influence"
# python3 K_multi_seed_statistics_smac_actionBased.py --total_experiments 100 \
#     --total_episodes 100 \
#     --filepath "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/hatrpo/q_smac-True-3svs3z/seed-00001-2025-10-04-16-12-38/models" \
#     --reward 60.187 \
#     --K_SIGMA 3


echo "Running smac with K=3 for 100 experiments 100 episodes 3m with New Q with Previous Influence"
python3 K_multi_seed_statistics_smac_actionBased.py --total_experiments 100 \
    --total_episodes 100 \
    --filepath "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3m/hatrpo/q_smac-True-3m_New/seed-00001-2025-10-04-16-14-38/models" \
    --reward 60.0 \
    --K_SIGMA 3

# echo "Running smac with K=5 for 100 experiments"
# python3 K_multi_seed_statistics_smac.py --total_experiments 100 \
#     --total_episodes 100 \
#     --filepath "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/hatrpo/q_smac-True-3svs3z/seed-00001-2025-10-03-21-10-39/models" \
#     --reward 60.529 \
#     --K_SIGMA 3



exit 0
