#!/bin/bash		 				
#SBATCH --job-name=Calc_Attack_SMAC			#Change to your own name
#SBATCH --output=/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/SLURM/out/Calc_Attack_SMAC.%j.o					#Call it output-<name>-%j.o
#SBATCH --error=/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/SLURM/error/Calc_Attack_SMAC.%j.e					#Call it error-<name>-%j.e
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


echo "With 0.10 taylor ref"
python attack_test_mean_final_smac.py --algo happo \
     --env smac \
     --exp_name compute \
     --reward 69.562 \
     --filepath  "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/happo/smac_3s_vs_3z/seed-13123-2025-09-03-20-10-32/models" \
     --taylor_csv_agent0 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.10/happo/2025-09-17-10-43-24/mappo_taylor_error_atk_free_agent_0.csv" \
    --taylor_csv_agent1 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.10/happo/2025-09-17-10-43-24/mappo_taylor_error_atk_free_agent_1.csv" \
    --taylor_csv_agent2 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.10/happo/2025-09-17-10-43-24/mappo_taylor_error_atk_free_agent_2.csv" \
    --seed 23 \
    --attack_id 1 \
    --save_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/Part-2/smac/happo/0.10/"
    
echo "With 0.25 taylor ref"
python attack_test_mean_final_smac.py --algo happo \
     --env smac \
     --exp_name compute \
     --reward 69.562 \
     --filepath  "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/happo/smac_3s_vs_3z/seed-13123-2025-09-03-20-10-32/models" \
     --taylor_csv_agent0 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.25/happo/2025-09-17-14-03-14/mappo_taylor_error_atk_free_agent_0.csv" \
    --taylor_csv_agent1 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.25/happo/2025-09-17-14-03-14/mappo_taylor_error_atk_free_agent_1.csv" \
    --taylor_csv_agent2 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.25/happo/2025-09-17-14-03-14/mappo_taylor_error_atk_free_agent_2.csv" \
    --seed 23 \
    --attack_id 1 \
    --save_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/Part-2/smac/happo/0.25/"


echo "With 0.10 taylor ref"
python attack_test_mean_final_smac.py --algo happo \
     --env smac \
     --exp_name compute \
     --reward 69.562 \
     --filepath  "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/happo/smac_3s_vs_3z/seed-13123-2025-09-03-20-10-32/models" \
     --taylor_csv_agent0 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.10/happo/2025-09-17-10-43-24/mappo_taylor_error_atk_free_agent_0.csv" \
    --taylor_csv_agent1 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.10/happo/2025-09-17-10-43-24/mappo_taylor_error_atk_free_agent_1.csv" \
    --taylor_csv_agent2 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.10/happo/2025-09-17-10-43-24/mappo_taylor_error_atk_free_agent_2.csv" \
    --seed 23 \
    --attack_id 0 \
    --save_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/Part-2/smac/happo/0.10/"
    
echo "With 0.25 taylor ref"
python attack_test_mean_final_smac.py --algo happo \
     --env smac \
     --exp_name compute \
     --reward 69.562 \
     --filepath  "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/happo/smac_3s_vs_3z/seed-13123-2025-09-03-20-10-32/models" \
     --taylor_csv_agent0 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.25/happo/2025-09-17-14-03-14/mappo_taylor_error_atk_free_agent_0.csv" \
    --taylor_csv_agent1 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.25/happo/2025-09-17-14-03-14/mappo_taylor_error_atk_free_agent_1.csv" \
    --taylor_csv_agent2 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.25/happo/2025-09-17-14-03-14/mappo_taylor_error_atk_free_agent_2.csv" \
    --seed 23 \
    --attack_id 0 \
    --save_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/Part-2/smac/happo/0.25/"


echo "With 0.10 taylor ref"
python attack_test_mean_final_smac.py --algo happo \
     --env smac \
     --exp_name compute \
     --reward 69.562 \
     --filepath  "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/happo/smac_3s_vs_3z/seed-13123-2025-09-03-20-10-32/models" \
     --taylor_csv_agent0 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.10/happo/2025-09-17-10-43-24/mappo_taylor_error_atk_free_agent_0.csv" \
    --taylor_csv_agent1 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.10/happo/2025-09-17-10-43-24/mappo_taylor_error_atk_free_agent_1.csv" \
    --taylor_csv_agent2 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.10/happo/2025-09-17-10-43-24/mappo_taylor_error_atk_free_agent_2.csv" \
    --seed 23 \
    --attack_id 2 \
    --save_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/Part-2/smac/happo/0.10/"
    
echo "With 0.25 taylor ref"
python attack_test_mean_final_smac.py --algo happo \
     --env smac \
     --exp_name compute \
     --reward 69.562 \
     --filepath  "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/happo/smac_3s_vs_3z/seed-13123-2025-09-03-20-10-32/models" \
     --taylor_csv_agent0 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.25/happo/2025-09-17-14-03-14/mappo_taylor_error_atk_free_agent_0.csv" \
    --taylor_csv_agent1 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.25/happo/2025-09-17-14-03-14/mappo_taylor_error_atk_free_agent_1.csv" \
    --taylor_csv_agent2 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.25/happo/2025-09-17-14-03-14/mappo_taylor_error_atk_free_agent_2.csv" \
    --seed 23 \
    --attack_id 2 \
    --save_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/Part-2/smac/happo/0.25/"



echo "####### Starting Taylor ref computation for HATRPO #######"
echo "With 0.10 taylor ref"
python attack_test_mean_final_smac.py --algo hatrpo \
     --env smac \
     --exp_name compute \
     --reward 60.176 \
     --filepath  "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/hatrpo/smac_3s_vs_3z/seed-00001-2025-09-03-22-49-28/models" \
     --taylor_csv_agent0 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.10/hatrpo/2025-09-17-11-00-40/mappo_taylor_error_atk_free_agent_0.csv" \
    --taylor_csv_agent1 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.10/hatrpo/2025-09-17-11-00-40/mappo_taylor_error_atk_free_agent_1.csv" \
    --taylor_csv_agent2 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.10/hatrpo/2025-09-17-11-00-40/mappo_taylor_error_atk_free_agent_2.csv" \
    --seed 23 \
    --attack_id 1 \
    --save_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/Part-2/smac/happo/0.10/"
    
echo "With 0.25 taylor ref"
python attack_test_mean_final_smac.py --algo hatrpo \
     --env smac \
     --exp_name compute \
     --reward 60.176 \
     --filepath  "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/hatrpo/smac_3s_vs_3z/seed-00001-2025-09-03-22-49-28/models" \
     --taylor_csv_agent0 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.25/hatrpo/2025-09-17-14-20-32/mappo_taylor_error_atk_free_agent_0.csv" \
    --taylor_csv_agent1 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.25/hatrpo/2025-09-17-14-20-32/mappo_taylor_error_atk_free_agent_1.csv" \
    --taylor_csv_agent2 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.25/hatrpo/2025-09-17-14-20-32/mappo_taylor_error_atk_free_agent_2.csv" \
    --seed 23 \
    --attack_id 1 \
    --save_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/Part-2/smac/happo/0.25/"


echo "With 0.10 taylor ref"
python attack_test_mean_final_smac.py --algo hatrpo \
     --env smac \
     --exp_name compute \
     --reward 60.176 \
     --filepath  "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/hatrpo/smac_3s_vs_3z/seed-00001-2025-09-03-22-49-28/models" \
     --taylor_csv_agent0 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.10/hatrpo/2025-09-17-11-00-40/mappo_taylor_error_atk_free_agent_0.csv" \
    --taylor_csv_agent1 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.10/hatrpo/2025-09-17-11-00-40/mappo_taylor_error_atk_free_agent_1.csv" \
    --taylor_csv_agent2 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.10/hatrpo/2025-09-17-11-00-40/mappo_taylor_error_atk_free_agent_2.csv" \
    --seed 23 \
    --attack_id 0 \
    --save_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/Part-2/smac/happo/0.10/"
    
echo "With 0.25 taylor ref"
python attack_test_mean_final_smac.py --algo hatrpo \
     --env smac \
     --exp_name compute \
     --reward 60.176 \
     --filepath  "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/hatrpo/smac_3s_vs_3z/seed-00001-2025-09-03-22-49-28/models" \
     --taylor_csv_agent0 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.25/hatrpo/2025-09-17-14-20-32/mappo_taylor_error_atk_free_agent_0.csv" \
    --taylor_csv_agent1 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.25/hatrpo/2025-09-17-14-20-32/mappo_taylor_error_atk_free_agent_1.csv" \
    --taylor_csv_agent2 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.25/hatrpo/2025-09-17-14-20-32/mappo_taylor_error_atk_free_agent_2.csv" \
    --seed 23 \
    --attack_id 0 \
    --save_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/Part-2/smac/happo/0.25/"


echo "With 0.10 taylor ref"
python attack_test_mean_final_smac.py --algo hatrpo \
     --env smac \
     --exp_name compute \
     --reward 60.176 \
     --filepath  "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/hatrpo/smac_3s_vs_3z/seed-00001-2025-09-03-22-49-28/models" \
     --taylor_csv_agent0 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.10/hatrpo/2025-09-17-11-00-40/mappo_taylor_error_atk_free_agent_0.csv" \
    --taylor_csv_agent1 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.10/hatrpo/2025-09-17-11-00-40/mappo_taylor_error_atk_free_agent_1.csv" \
    --taylor_csv_agent2 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.10/hatrpo/2025-09-17-11-00-40/mappo_taylor_error_atk_free_agent_2.csv" \
    --seed 23 \
    --attack_id 2 \
    --save_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/Part-2/smac/happo/0.10/"
    
echo "With 0.25 taylor ref"
python attack_test_mean_final_smac.py --algo hatrpo \
     --env smac \
     --exp_name compute \
     --reward 60.176 \
     --filepath  "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/hatrpo/smac_3s_vs_3z/seed-00001-2025-09-03-22-49-28/models" \
     --taylor_csv_agent0 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.25/hatrpo/2025-09-17-14-20-32/mappo_taylor_error_atk_free_agent_0.csv" \
    --taylor_csv_agent1 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.25/hatrpo/2025-09-17-14-20-32/mappo_taylor_error_atk_free_agent_1.csv" \
    --taylor_csv_agent2 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.25/hatrpo/2025-09-17-14-20-32/mappo_taylor_error_atk_free_agent_2.csv" \
    --seed 23 \
    --attack_id 2 \
    --save_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/Part-2/smac/happo/0.25/"
exit 0
