#!/bin/bash		 				
#SBATCH --job-name=gpu-%j				
#SBATCH --output=/deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/slurm_logs/out/out_rl-%j.o					#Call it output-<name>-%j.o
#SBATCH --error=/deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/slurm_logs/err_rl-%j.e					#Call it error-<name>-%j.e
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shefrs24@wfu.edu				#Change to your own email
#SBATCH --account=alqahtaniGrp
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=20gb					#Change depending on how much Memory you need
#SBATCH --gres=gpu:1
#SBATCH --time=10-00:00:00				#4 days - change if you need more time

module load apps/anaconda3/2024.02
conda activate adversary_loss_marl

module load nvidia/cuda11/cuda/11.8.0
module load nvidia/cuda11/cudnn/8.7.0.84

cd /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/skrl
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia

# python train.py --env_id simple_spread_v3 --algo_name IPPO --train_steps 10000000
# python train.py --env_id simple_spread_v3 --algo_name MAPPO --train_steps 10000000
# python train.py --env_id simple_adversary_v3 --algo_name IPPO --train_steps 10000000
python train_atari.py --env_id space_invaders_v2 --algo_name IPPO --train_steps 10000000

# python test.py --env_id simple_spread_v3 --algo_name IPPO --model_dir /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/skrl/runs/25-03-27_22-54-26-605109_IPPO/checkpoints/best_agent.pt
# python test.py --env_id simple_spread_v3 --algo_name MAPPO --model_dir /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/skrl/runs/25-03-28_01-50-57-258165_MAPPO/checkpoints/best_agent.pt

exit 0
