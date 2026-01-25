def restore(runner,reward,filepath="/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v2-discrete/happo/installtest/seed-00042-2025-08-03-20-41-48/models"):
        """Restore model parameters."""
        for agent_id in range(runner.num_agents):
            policy_actor_state_dict = torch.load(
                str(filepath)
                + "/actor_agent"
                + str(agent_id)
                + "_" + str(reward)
                + ".pt"
            )
            runner.actor[agent_id].actor.load_state_dict(policy_actor_state_dict)
        
            
            runner.central_q[agent_id].restore(filepath,agent_id,reward)
         
           
        if not runner.algo_args["render"]["use_render"]:
            policy_critic_state_dict = torch.load(
                str(filepath)
                + "/critic_agent"
                + "_" + str(reward)
                + ".pt"
            )
            runner.critic.critic.load_state_dict(policy_critic_state_dict)
            if runner.value_normalizer is not None:
                value_normalizer_state_dict = torch.load(
                    str(filepath)
                    + "/value_normalizer"
                    + "_" + str(reward)
                    + ".pt"
                )
                runner.value_normalizer.load_state_dict(value_normalizer_state_dict)
        

def initiate_model():
    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="hatrpo",
        choices=[
            "happo",
            "hatrpo",
            "haa2c",
            "haddpg",
            "hatd3",
            "hasac",
            "had3qn",
            "maddpg",
            "matd3",
            "mappo",
        ],
        help="Algorithm name. Choose from: happo, hatrpo, haa2c, haddpg, hatd3, hasac, had3qn, maddpg, matd3, mappo.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="pettingzoo_mpe",
        choices=[
            "smac",
            "mamujoco",
            "pettingzoo_mpe",
            "gym",
            "football",
            "dexhands",
            "smacv2",
            "lag",
        ],
        help="Environment name. Choose from: smac, mamujoco, pettingzoo_mpe, gym, football, dexhands, smacv2, lag.",
    )
    parser.add_argument(
        "--exp_name", type=str, default="installtest", help="Experiment name."
    )
    parser.add_argument(
        "--attack_id", type=int, default=0, help="Agent ID to attack."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    parser.add_argument(
        "--reward",
        type=float,
        default=-79.879,
        help="Reward value to restore the model."
    )
    parser.add_argument(
        "--filepath",
        type=str,
        default="/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/results/pettingzoo_mpe/simple_spread_v3-discrete/hatrpo/Latest_3/seed-00001-2025-08-15-22-56-55/models",
        help="Filepath to restore the model from."
    )
    parser.add_argument(
        "--seed", type=int, default=376, help="Random seed."
    )
    parser.add_argument(
        "--min_window", type=int, default=8, help="Minimum window size."
    )
    parser.add_argument(
        "--max_window", type=int, default=10, help="Maximum window size."
    )
    parser.add_argument(
        "--taylor_csv_agent0", type=str, default="/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc_new/seed-376/hatrpo/0.1/2025-09-24-20-51-51/mappo_taylor_error_atk_free_agent_0.csv", help="Path to CSV file with pre-computed Taylor history for agent 0."
    )
    parser.add_argument(
        "--taylor_csv_agent1", type=str, default="/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc_new/seed-376/hatrpo/0.1/2025-09-24-20-51-51/mappo_taylor_error_atk_free_agent_1.csv", help="Path to CSV file with pre-computed Taylor history for agent 1."
    )
    parser.add_argument(
        "--taylor_csv_agent2", type=str, default="/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/taylor_calc_new/seed-376/hatrpo/0.1/2025-09-24-20-51-51/mappo_taylor_error_atk_free_agent_2.csv", help="Path to CSV file with pre-computed Taylor history for agent 2."
    )
    parser.add_argument(
        "--save_dir", type=str, default='/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/examples/Part-2/test', help="Directory to save results."
    )
    
    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict
    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
    else:  # load config from corresponding yaml file
        algo_args, env_args = get_defaults_yaml_args(args["algo"], args["env"])
    update_args(unparsed_dict, algo_args, env_args)  # update args from command line

    if args["env"] == "dexhands":
        import isaacgym  # isaacgym has to be imported before PyTorch

    # note: isaac gym does not support multiple instances, thus cannot eval separately
    if args["env"] == "dexhands":
        algo_args["eval"]["use_eval"] = False
        algo_args["train"]["episode_length"] = env_args["hands_episode_length"]

    # start training
    from harl.runners import RUNNER_REGISTRY

# 
    algo_args['eval']['n_eval_rollout_threads'] = 1
    algo_args['eval']['eval_episodes'] = 1
    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
    # print(f"Checking if centralize q is set : {algo_args['algo']['use_centralized_q']}")
    restore(runner,args['reward'],args['filepath'])  # Restore the model with specific reward and episode
    
    runner.prep_training()
    
    return runner
