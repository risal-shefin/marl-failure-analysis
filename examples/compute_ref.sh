python compute_taylor_ref.py --algo happo --env pettingzoo_mpe --exp_name compute --reward -75.127 --total_episodes 5000 --seed 376 \
    --filepath  "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/happo/Latest_3/seed-00001-2025-08-15-21-25-27/models" \
    --save_result_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/simple_spread_v3/0.25/"


python compute_taylor_ref.py --algo hatrpo --env pettingzoo_mpe --exp_name compute --reward -79.879 --total_episodes 5000 --seed 376 \
    --filepath  "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/hatrpo/Latest_3/seed-00001-2025-08-15-22-56-55/models" \
    --save_result_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/simple_spread_v3/0.25/"

# echo "####### Starting SMAC Taylor ref computation for HAPPO #######"
# python compute_taylor_ref_smac.py --algo happo --env smac --exp_name compute --reward 69.562 --total_episodes 1000 \
#     --filepath  "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/happo/smac_3s_vs_3z/seed-13123-2025-09-03-20-10-32/models" \
#     --save_result_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.25/"

# echo "####### Starting SMAC Taylor ref computation for HATRPO #######"
# python compute_taylor_ref_smac.py --algo hatrpo --env smac --exp_name compute --reward 60.176 --total_episodes 1000 \
#     --filepath  "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/smac/3s_vs_3z/hatrpo/smac_3s_vs_3z/seed-00001-2025-09-03-22-49-28/models" \
#     --save_result_dir "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc/smac/0.25/"
