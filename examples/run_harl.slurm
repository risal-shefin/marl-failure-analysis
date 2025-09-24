#!/bin/bash		 				
#SBATCH --job-name=HATRPO_smac_3s_vs_3z			#Change to your own name
#SBATCH --output=/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/SLURM/out/HATRPO_smac_3s_vs_3z.%j.o					#Call it output-<name>-%j.o
#SBATCH --error=/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/SLURM/error/HATRPO_smac_3s_vs_3z.%j.e					#Call it error-<name>-%j.e
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

# python compute_taylor_ref.py --algo happo --env pettingzoo_mpe

# python attack_test.py --algo happo --env pettingzoo_mpe

# python train.py --algo hatrpo --env pettingzoo_mpe
# python train.py --algo happo --env pettingzoo_mpe --exp_name Latest_5
# python train.py --algo happo --env smac --exp_name smac_3s_vs_3z
python train.py --algo hatrpo --env smac --exp_name smac_3s_vs_3z

exit 0
