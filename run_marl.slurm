#!/bin/bash		 				
#SBATCH --job-name=gpu-%j				
#SBATCH --output=/deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/slurm_logs/out/out_rl-%j.o					#Call it output-<name>-%j.o
#SBATCH --error=/deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/slurm_logs/err_rl-%j.e					#Call it error-<name>-%j.e
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shefrs24@wfu.edu				#Change to your own email
#SBATCH --account=alqahtaniGrp
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=16gb					#Change depending on how much Memory you need
#SBATCH --gres=gpu:1
#SBATCH --time=20-00:00:00				#4 days - change if you need more time

module load apps/anaconda3/2024.02
conda activate adversary_loss

module load nvidia/cuda11/cuda/11.8.0
module load nvidia/cuda11/cudnn/8.7.0.84

cd /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/BenchMARL
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia

python benchmarl/run.py algorithm=maddpg task=pettingzoo/simple_adversary \
        experiment.checkpoint_at_end=True experiment.checkpoint_interval=18000 experiment.keep_checkpoints_num=10 \
        experiment.sampling_device=cuda experiment.train_device=cuda experiment.buffer_device=cuda

exit 0
