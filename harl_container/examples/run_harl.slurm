#!/bin/bash		 				
#SBATCH --job-name=S.3.Act.K3.E.500.F.100.R.Prev_Influence			#Change to your own name
#SBATCH --output=/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/Krishna-Result-Slurm/out/%j.S.3.Act.K3.E.500.F.100.R.Prev_Influence.o					#Call it output-<name>-%j.o
#SBATCH --error=/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/Krishna-Result-Slurm/error/%j.S.3.Act.K3.E.500.F.100.R.Prev_Influence.e					#Call it error-<name>-%j.e
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=guptd23@wfu.edu				#Change to your own email
#SBATCH --account=paucaGrp
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=64gb		
#SBATCH --gres=gpu:1			#Change depending on how much Memory you need
#SBATCH --time=1-00:00:00				#4 days - change if you need more time



module load nvidia/cuda11/cuda/11.8.0
module load nvidia/cuda11/cudnn/8.7.0.84

cd /deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
export TORCH_USE_CUDA_DSA=1

# python compute_taylor_ref.py --algo happo --env pettingzoo_mpe

# python attack_test.py --algo happo --env pettingzoo_mpe

# python train.py --algo hatrpo --env pettingzoo_mpe
# python train.py --algo hatrpo --env smac --exp_name q_smac-True-3m --use_centralized_q True
# python train.py --algo hatrpo --env pettingzoo_mpe --exp_name q_five_agent --use_centralized_q True




##########MAIN
# echo "Running simple spread with 5 agents K=3"
# python3 K_multi_seed_statistics_simple_new.py --N 5 \
#     --total_experiments 100 \
#     --total_episodes 100 \
#     --filepath "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/hatrpo/q_five_agent_tanh/seed-00001-2025-10-03-22-33-15/models/" \
#     --reward -356.476 \
#     --K_SIGMA 3

# echo "Running simple spread with 3 agents K=3 action based 500 experiments 100 episodes"
# python3 K_multi_seed_statistics_simple_new_actionBased.py --N 3 \
#     --total_experiments 500 \
#     --total_episodes 100 \
#     --filepath "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/hatrpo/q_three_agent_tanh_New/seed-00001-2025-10-04-16-07-38/models" \
#     --reward -44.944 \
#     --K_SIGMA 3

echo "Running simple spread with 5 agents K=3 action based 500 experiments 100 episodes with Previous Influence"
python3 K_multi_seed_statistics_simple_new_actionBased.py --N 5 \
    --total_experiments 500 \
    --total_episodes 100 \
    --filepath "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/hatrpo/q_three_agent_tanh_New/seed-00001-2025-10-04-16-07-38/models" \
    --reward -44.944 \
    --K_SIGMA 3


#####OLD
# python3 multi_seed_statistics.py
# python3 multi_seed_statistics_k_influence.py
# echo "Running simple spread with 3 agents K=1"
# python3 multi_seed_statistics.py -N 3 \
#     --total_experiments 500 \
#     --total_episodes 100 \
#     --filepath "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/hatrpo/q_three_agent/seed-00001-2025-09-23-22-29-22/models" \
#     --reward -89.021 \
#     --K_SIGMA 3

# echo "Running simple spread with 3 agents K=2"
# python3 multi_seed_statistics.py -N 3 \
#     --total_experiments 500 \
#     --total_episodes 100 \
#     --filepath "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/hatrpo/q_three_agent/seed-00001-2025-09-23-22-29-22/models" \
#     --reward -89.021 \
#     --K_SIGMA 2

# echo "Running simple spread with 3 agents K=3"
# python3 multi_seed_statistics.py -N 3 \
#     --total_experiments 100 \
#     --total_episodes 100 \
#     --filepath "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/hatrpo/q_three_agent/seed-00001-2025-09-23-22-29-22/models" \
#     --reward -89.021 \
#     --K_SIGMA 3


# echo "Running simple spread with 5 agents K=1"
# python3 multi_seed_statistics.py -N 5 \
#     --total_experiments 500 \
#     --total_episodes 100 \
#     --filepath "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/hatrpo/q_five_agent/seed-00001-2025-09-23-22-32-22/models" \
#     --reward -565.155 \
#     --K_SIGMA 1



# echo "Running simple spread with 5 agents K=2"
# python3 multi_seed_statistics.py -N 5 \
#     --total_experiments 500 \
#     --total_episodes 100 \
#     --filepath "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/hatrpo/q_five_agent/seed-00001-2025-09-23-22-32-22/models" \
#     --reward -565.155 \
#     --K_SIGMA 2


# echo "Running simple spread with 5 agents K=3"
# python3 multi_seed_statistics.py -N 5 \
#     --total_experiments 100 \
#     --total_episodes 100 \
#     --filepath "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/hatrpo/q_five_agent/seed-00001-2025-09-23-22-32-22/models" \
#     --reward -565.155 \
#     --K_SIGMA 3

exit 0
