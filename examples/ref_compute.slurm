#!/bin/bash		 				
#SBATCH --job-name=cal_hap_hatr			#Change to your own name
#SBATCH --output=/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/SLURM/out/cal_hap_hatr.%j.o					#Call it output-<name>-%j.o
#SBATCH --error=/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/SLURM/error/cal_hap_hatr.%j.e					#Call it error-<name>-%j.e
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


# python compute_taylor_ref.py --algo happo --env pettingzoo_mpe --exp_name compute --reward -75.127 --total_episodes 5000 \
#     --filepath  "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/happo/Latest_3/seed-00001-2025-08-15-21-25-27/models" \
#     --save_result_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/simple_spread_v3/0.25/" \
#     --randomness 0.25

# python compute_taylor_ref.py --algo hatrpo --env pettingzoo_mpe --exp_name compute --reward -79.879 --total_episodes 5000 \
#     --filepath  "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/hatrpo/Latest_3/seed-00001-2025-08-15-22-56-55/models" \
#     --save_result_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/simple_spread_v3/0.25/" \
#     --randomness 0.25

echo "####### Starting SMAC Taylor ref computation for HAPPO #######"
python compute_taylor_ref_smac.py --algo happo --env smac --exp_name compute --reward 69.562 --total_episodes 1000 \
    --filepath  "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/happo/smac_3s_vs_3z/seed-13123-2025-09-03-20-10-32/models" \
    --save_result_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.25/" \
    --randomness 0.25

echo "####### Starting SMAC Taylor ref computation for HATRPO #######"
python compute_taylor_ref_smac.py --algo hatrpo --env smac --exp_name compute --reward 60.176 --total_episodes 1000 \
    --filepath  "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/hatrpo/smac_3s_vs_3z/seed-00001-2025-09-03-22-49-28/models" \
    --save_result_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.25/" \
    --randomness 0.25

# echo "####### Starting SMAC Taylor ref computation for HAPPO #######"
# python compute_taylor_ref_smac.py --algo happo --env smac --exp_name compute --reward 69.562 --total_episodes 1000 \
#     --filepath  "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/happo/smac_3s_vs_3z/seed-13123-2025-09-03-20-10-32/models" \
#     --save_result_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.10/"

# echo "####### Starting SMAC Taylor ref computation for HATRPO #######"
# python compute_taylor_ref_smac.py --algo hatrpo --env smac --exp_name compute --reward 60.176 --total_episodes 1000 \
#     --filepath  "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/hatrpo/smac_3s_vs_3z/seed-00001-2025-09-03-22-49-28/models" \
#     --save_result_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.10/"

exit 0
