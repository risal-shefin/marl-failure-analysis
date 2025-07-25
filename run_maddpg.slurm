#!/bin/bash		 				
#SBATCH --job-name=gpu-%j				
#SBATCH --output=/deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/slurm_logs/out/out_rl-%j.o					#Call it output-<name>-%j.o
#SBATCH --error=/deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/slurm_logs/err_rl-%j.e					#Call it error-<name>-%j.e
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shefrs24@wfu.edu				#Change to your own email
#SBATCH --account=alqahtaniGrp
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=64gb					#Change depending on how much Memory you need
#SBATCH --gres=gpu:V100_32:1
#SBATCH --time=10-00:00:00				#4 days - change if you need more time

module load apps/anaconda3/2024.02
conda activate adversary_loss_marl

module load nvidia/cuda11/cuda/11.8.0
module load nvidia/cuda11/cudnn/8.7.0.84

cd /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia

# python train_pettingzoo.py simple_speaker_listener_v4 maddpg_continuous
# python train_pettingzoo.py simple_reference_v3 maddpg_continuous
# python train_pettingzoo.py simple_spread_v3 maddpg_continuous
# python train_pettingzoo.py multiwalker_v9 maddpg
# python train_pettingzoo.py waterworld_v4 maddpg --hidden_dim 256
# python train_pettingzoo.py simple_adversary_v3 maddpg_continuous
# python train_pettingzoo.py simple_tag_v3 maddpg_continuous
# python train_pettingzoo.py space_invaders_v2 maddpg --hidden_dim 512 --discrete_action True --episode_length 1000
# python train_pettingzoo.py entombed_cooperative_v3 maddpg --hidden_dim 512 --discrete_action True --episode_length 1000 --buffer_length 10000 --save_interval 10
# python train_pettingzoo.py mario_bros_v3 maddpg --hidden_dim 512 --discrete_action True --episode_length 1000 --buffer_length 50000 --save_interval 10
# python train_pettingzoo.py simple_spread_v3 maddpg_discrete --discrete_action True --episode_length 100
# python train_pettingzoo.py simple_spread_v3 maddpg_continuous_tanh_activ --episode_length 100
python train_pettingzoo.py simple_spread_v3 maddpg_discrete_tanh_activ --discrete_action True --episode_length 100


# TESTING COMMANDS
# python test_pettingzoo_by_act.py simple_speaker_listener_v4 /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/simple_speaker_listener_v4/maddpg_continuous/run1/model_-1.6530647514892638.pt --save_gifs
# python test_pettingzoo_by_act.py simple_reference_v3 /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/simple_reference_v3/maddpg_continuous/run1/model_-8.332783496716875.pt --save_gifs
# python test_pettingzoo_by_act.py simple_spread_v3 /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/simple_spread_v3/maddpg_continuous/run1/model_-14.630762016761627.pt --save_gifs
# python test_pettingzoo_cross_so.py simple_spread_v3 /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/simple_spread_v3/maddpg_discrete/run1/model_-18.883772011719756.pt --save_gifs
# python test_pettingzoo_by_act.py multiwalker_v9 /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/multiwalker_v9/maddpg/run3/model_-20.483479279322403.pt --save_gifs
# python test_pettingzoo_by_act.py waterworld_v4 /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/waterworld_v4/maddpg/run2/model_62.396000000000015.pt --save_gifs
# python test_pettingzoo_by_act.py waterworld_v4 /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/waterworld_v4/maddpg/run5-n5-256/model_64.49200000000003.pt --save_gifs
# python test_pettingzoo_by_act.py simple_adversary_v3 /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/simple_adversary_v3/maddpg_continuous/run1/model_51.000088575985245.pt --save_gifs
# python test_pettingzoo_by_act.py simple_tag_v3 /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/simple_tag_v3/maddpg_continuous/run3/model_0.0.pt --save_gifs
# python test_pettingzoo_by_act.py entombed_cooperative_v3 /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/entombed_cooperative_v3/maddpg/run4/model_8.0.pt --save_gifs
# python test_pettingzoo_cross_so.py simple_spread_v3 /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/simple_spread_v3/maddpg_continuous_tanh_activ/run1/model_-27.42235851044746.pt --save_gifs
# python test_pettingzoo_cross_so.py simple_spread_v3 /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/simple_spread_v3/maddpg_discrete_tanh_activ/run1/model_-24.24483667155664.pt --save_gifs

# python test_pettingzoo_detection.py simple_spread_v3 /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/simple_spread_v3/maddpg_discrete_tanh_activ/run1/model_-24.24483667155664.pt --ref_val_dir /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/data/20250722_simple_spread_taylor_error --attack_agent_id 0 --atk_start_step 0 --atk_end_step 10

exit 0
