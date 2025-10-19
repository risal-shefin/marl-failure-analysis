#!/bin/bash		 				
#SBATCH --job-name=maddpg				
#SBATCH --output=/deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/slurm_logs/out/out_rl-%j.o					#Call it output-<name>-%j.o
#SBATCH --error=/deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/slurm_logs/err_rl-%j.e					#Call it error-<name>-%j.e
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=shefrs24@wfu.edu				#Change to your own email
#SBATCH --account=alqahtaniGrp
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64gb					#Change depending on how much Memory you need
#SBATCH --time=20-00:00:00				#4 days - change if you need more time

module load apps/anaconda3/2024.02
conda activate adversary_loss_marl

module load nvidia/cuda11/cuda/11.8.0
module load nvidia/cuda11/cudnn/8.7.0.84

cd /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/.mujoco/mujoco210/bin:/usr/lib/nvidia
export SC2PATH=/deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/StarCraftII

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
# python train_pettingzoo.py simple_spread_v3 maddpg_discrete_tanh_activ --discrete_action True --episode_length 100
# python train_pettingzoo.py multiwalker_v9 maddpg_tanh_activ --episode_length 1000
# python train_pettingzoo.py simple_spread_v3 maddpg_discrete_n5_tanh_activ --discrete_action True --episode_length 100
# python train_pettingzoo.py simple_spread_v3 maddpg_discrete_tanh_localq --discrete_action True --episode_length 100 --local_q
# python train_pettingzoo.py simple_spread_v3 maddpg_discrete_tanh_localq_fxseed12345 --discrete_action True --episode_length 100 --local_q --seed 12345

# petingzoo test
# Shapley frob analysis
# python -m scripts.pettingzoo.shapley_frob_analysis simple_spread_v3 /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/simple_spread_v3/maddpg_discrete_tanh_localq/run1/model_-22.824949662226082.pt --save_gifs --seed 12345
# Multi seed detection stats
# python -m scripts.pettingzoo.multi_seed_statistics simple_spread_v3 /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/simple_spread_v3/maddpg_discrete_tanh_localq/run1/model_-22.824949662226082.pt --total_experiments 50

# Multigrid
python train_multigrid.py collect maddpg_multigrid_3a0_2b0 --discrete_action True --local_q --episode_length 100 --save_interval 10

#VMAS42
# python train_vmas.py flocking maddpg_discrete --discrete_action True --episode_length 100
## Testing
# python test_vmas_detection.py flocking /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/flocking/maddpg_discrete/run49/model_-12.602785110473633.pt --ref_val_dir /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/data/flocking_discrete/20250820-014511_refs_w20 --attack_agent_id 4

# SMAC
# python train_smac.py 3m maddpg
# python train_smac.py 3s_vs_3z maddpg
# smac test scripts
# python -m scripts.smac.monte_carlo_shapley 3m /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/3m/maddpg/run2/model_24.427397260273974.pt --monte_carlo_iterations 100 --save_gifs
# python -m scripts.smac.shapley_frob_analysis 3m /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/3m/maddpg/run2/model_24.427397260273974.pt --seed 12345

# Reference Computation
# python -m scripts.compute_ref_taylor_error simple_spread_v3 /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/simple_spread_v3/maddpg_discrete_tanh_localq/run1/model_-22.824949662226082.pt --seed 23
# python -m scripts.compute_ref_taylor_error multiwalker_v9 /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/multiwalker_v9/maddpg_tanh_activ/run2/model_24.50843327734619.pt
# python -m scripts.compute_vmas_ref_taylor_error flocking /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/flocking/maddpg_discrete/run49/model_-12.602785110473633.pt
# python -m scripts.compute_ref_taylor_error_localq simple_spread_v3 /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/simple_spread_v3/maddpg_discrete_tanh_localq/run1/model_-22.824949662226082.pt
# python -m scripts.smac.compute_ref_taylor_error 3m /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/3m/maddpg/run2/model_24.427397260273974.pt --seed 23
# python -m scripts.smac.compute_ref_taylor_error 3s_vs_3z /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/3s_vs_3z/maddpg/run1/model_33.64852941176473.pt --seed 10

# SMAC TEST
# python -m scripts.smac.test_detection 3s_vs_3z /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/3s_vs_3z/maddpg/run1/model_33.64852941176473.pt --ref_val_dir /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/runs/3s_vs_3z_discrete/20251003-223634_metrics_taylor_error_seed10_noise1e-4always --attack_agent_id 2 --seed 10 --save_gifs

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

# python test_pettingzoo_detection.py simple_spread_v3 /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/simple_spread_v3/maddpg_discrete_tanh_activ/run1/model_-24.24483667155664.pt --ref_val_dir /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/data/20250722_simple_spread_refs --attack_agent_id 0 --atk_start_step 0 --atk_end_step 10
# with noise
# python test_pettingzoo_detection.py simple_spread_v3 /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/simple_spread_v3/maddpg_discrete_tanh_localq/run1/model_-22.824949662226082.pt --ref_val_dir /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/data/20250914-212212_metrics_taylor_error_seed23_gauss_noise_rate0.1 --attack_agent_id 0

# N5
# python -m scripts.compute_ref_taylor_error simple_spread_v3 /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/simple_spread_v3/maddpg_discrete_n5_tanh_activ/run1/model_-111.47935966899873.pt
# python test_pettingzoo_detection.py simple_spread_v3 /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/simple_spread_v3/maddpg_discrete_n5_tanh_activ/run1/model_-111.47935966899873.pt --ref_val_dir /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/data/20250810-193211_simple_spread_refs_n5 --attack_agent_id 4
## localq
# python -m scripts.compute_ref_taylor_error_localq simple_spread_v3 /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/simple_spread_v3/maddpg_discrete_n5_tanh_localq/run1/model_-122.48782671825927.pt
# python test_pettingzoo_detection_localq.py simple_spread_v3 /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/simple_spread_v3/maddpg_discrete_n5_tanh_localq/run1/model_-118.76917529145851.pt --ref_val_dir /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/runs/simple_spread_v3_discrete/20250825-093502_metrics_taylor_error --attack_agent_id 4

# multiwalker
# python test_pettingzoo_detection.py multiwalker_v9 /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/multiwalker_v9/maddpg_tanh_activ/run2/model_24.50843327734619.pt --ref_val_dir /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/data/20250813-011645_multiwalker_v9_refs --attack_agent_id 1

# detection metrics stats
# python -m scripts.compute_detection_metrics simple_spread_v3 /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/models/simple_spread_v3/maddpg_discrete_n5_tanh_activ/run1/model_-111.47935966899873.pt --ref_val_dir /deac/csc/alqahtaniGrp/shefrs24/AdversaryLoss-Container/AdversaryLoss/maddpg-pytorch/data/20250814-210119_simple_spread_v3_refs_mmd_n5 --num_episodes 10000

exit 0
