# python attack_test_mean_final.py --algo happo \
#      --env pettingzoo_mpe \
#      --exp_name compute \
#      --reward -75.127 \
#      --filepath "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/happo/Latest_3/seed-00001-2025-08-15-21-25-27/models" \
#      --taylor_csv_agent0 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/simple_spread_v3/0.10/happo/2025-09-16-22-37-16/mappo_taylor_error_atk_free_agent_0.csv" \
#     --taylor_csv_agent1 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/simple_spread_v3/0.10/happo/2025-09-16-22-37-16/mappo_taylor_error_atk_free_agent_1.csv" \
#     --taylor_csv_agent2 "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/simple_spread_v3/0.10/happo/2025-09-16-22-37-16/mappo_taylor_error_atk_free_agent_2.csv" \
#     --seed 23 \
#     --attack_id 2

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
# python compute_taylor_ref.py --algo hatrpo --env pettingzoo_mpe --exp_name compute --reward -79.879 --total_episodes 5000 \
#     --filepath  "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/hatrpo/Latest_3/seed-00001-2025-08-15-22-56-55/models" \
#     --save_result_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/simple_spread_v3/0.10/"

