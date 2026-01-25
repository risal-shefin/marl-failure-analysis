def eval(runner, attack_status=False, attack_agent_id=0, seed=23, taylor_history_data=None, collect_q_flag=False,min_window=8,max_window=12):
    """Evaluate the model."""
    
    eval_episode = 0

    eval_obs, eval_share_obs, eval_available_actions = runner.eval_envs.reset(seed=seed)

    eval_rnn_states = np.zeros(
        (
            runner.algo_args["eval"]["n_eval_rollout_threads"],
            runner.num_agents,
            runner.recurrent_n,
            runner.rnn_hidden_size,
        ),
        dtype=np.float32,
    )
    eval_rnn_states_critic = np.zeros(
        (
            runner.algo_args["eval"]["n_eval_rollout_threads"],
            runner.num_agents,
            runner.recurrent_n,
            runner.rnn_hidden_size,
        ),
        dtype=np.float32,
    )
    eval_masks = np.ones(
        (runner.algo_args["eval"]["n_eval_rollout_threads"], runner.num_agents, 1),
        dtype=np.float32,
    )

    taylor_error_list = list()
    frob_norms_list = []
    sec_dir_derivatives = []
    result_deques = [deque(maxlen=5) for _ in range(runner.num_agents)]
    frob_norms_deques = [deque(maxlen=5) for _ in range(runner.num_agents)]
    sec_dir_derivatives_deques = [deque(maxlen=5) for _ in range(runner.num_agents)]

    # Additional structures to mirror get_episode_data logic
    frob_norms_matrix_history = []  # list of N x N pairwise frob matrices per timestep
    fault_first_detected = {}  # agent_id -> first detected timestep
    fault_timeline = []
    attacked_steps = []
    pairwise_action_value_influence_list = []
    pairwise_action_value_influence_history = []
    taylor_history = [[] for _ in range(runner.num_agents)]
    cnt = 0
    q_values_list = [] if collect_q_values else None
    total_rewards = {}
    reward_ep=0
    tt=0
    while True:
        
        eval_rnn_states_backup = np.copy(eval_rnn_states)
        eval_actions_collector = []
        for agent_id in range(runner.num_agents):
            eval_actions, temp_rnn_state = runner.actor[agent_id].act(
                eval_obs[:, agent_id],
                eval_rnn_states[:, agent_id],
                eval_masks[:, agent_id],
                eval_available_actions[:, agent_id]
                if eval_available_actions[0] is not None
                else None,
                deterministic=True,
            )
            eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
            eval_actions_collector.append(_t2n(eval_actions))

        eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
        
        if collect_q_flag:
            q_vals = collect_q_values(runner, eval_obs, eval_actions)
            q_values_list.append(q_vals)

        if attack_status and (cnt>=min_window and cnt<=max_window):
            # mark attacked step
            # eval_actions[0][attack_agent_id] = runner.eval_envs.action_space[attack_agent_id].sample()  # Random action for attack agent
            # print(f">>>> ",eval_actions[0][attack_agent_id])
            n_actions = runner.eval_envs.action_space[attack_agent_id].n
            avail_slice = slice_avail(eval_available_actions, attack_agent_id)
            if avail_slice is not None and avail_slice[0] is not None:
                        available_actions = np.where(avail_slice[0] > 0.5)[0]
            else:
                available_actions = list(range(n_actions))
            
            # exit("Exiting for debug")
            with torch.no_grad():
                print(f" [!!!] Attack launched on agent {attack_agent_id} at timestep: {cnt}")
                obs_tensor = torch.FloatTensor(eval_obs[:, attack_agent_id])
                rnn_tensor = torch.FloatTensor(eval_rnn_states[:, attack_agent_id])
                mask_tensor = torch.FloatTensor(eval_masks[:, attack_agent_id])
                
                action_log_probs, dist_entropy, action_distribution = runner.actor[attack_agent_id].evaluate_actions(
                            obs_tensor.to(runner.device),
                            rnn_tensor.to(runner.device),
                            available_actions,
                            mask_tensor.to(runner.device),
                            slice_avail(eval_available_actions, attack_agent_id),
                            None
                        )
                q_values = action_log_probs.squeeze()
                if q_values.numel() == 1 or len(available_actions) == 1:
                    print(f"Agent {agent_id} appears to be dead or has only one action. Using index 0.")
                    eval_actions[0][attack_agent_id] = 0
                else:
                    worst_action = torch.argmin(q_values).item()
                    eval_actions[0][attack_agent_id] = worst_action
                    print(f"Agent {attack_agent_id} worst action under current policy: {worst_action}")
            attacked_steps.append(cnt)


        # calculating taylor policy
        pairwise_action_value_influence = compute_pairwise_action_influence(runner, eval_obs, eval_actions)
        # exit("Exiting for debug")
        pairwise_action_value_influence_history.append(pairwise_action_value_influence)
        delta_errors = compute_taylor_policy(runner, eval_obs, eval_available_actions, eval_rnn_states_backup)
        # results_frob_norms = compute_frob_norms(runner, eval_obs, 1, eval_rnn_states_critic, eval_masks)
        # results_sec_dir_derivatives = compute_2nd_ord_dir_derivatives(runner, eval_obs, 1, eval_rnn_states_critic, eval_masks)
        # pairwise frob matrix for cascading analysis
        pairwise_frobs = compute_pairwise_frob_norms(runner, eval_obs, eval_rnn_states_critic, eval_masks)
        frob_norms_matrix_history.append(pairwise_frobs)

        for i in range(runner.num_agents):
            result_deques[i].append(delta_errors[i])
            taylor_approx_error = np.mean(result_deques[i])
            taylor_history[i].append(taylor_approx_error)

            # frob_norms_deques[i].append(results_frob_norms[i])
            # sec_dir_derivatives_deques[i].append(results_sec_dir_derivatives[i])

            # Detect anomalies based on Taylor approximation error using pre-computed history
            if i not in fault_first_detected and cnt in taylor_history_data[i]:
                historical_data = taylor_history_data[i][cnt]
                historical_mean = historical_data['mean']
                historical_std = historical_data['std_dev']
                
                # Ensure minimum std deviation to avoid division by zero
                if historical_std < 1e-6:
                    historical_std = 1e-6
                
                # Check if current error is outside historical bounds (mean ± std_dev)
                lower_bound = historical_mean - historical_std
                upper_bound = historical_mean + historical_std
                
                if taylor_approx_error < lower_bound*0.3 or taylor_approx_error > upper_bound*3.0:
                    print(f" [!!!] Anomaly detected for agent {i} at timestep: {cnt}. Taylor Appx. Error: {taylor_approx_error}")
                    print(f"     >> Historical bounds: [{lower_bound:.6f}, {upper_bound:.6f}], Mean: {historical_mean:.6f}, Std: {historical_std:.6f}")
                    fault_first_detected[i] = cnt
                    # Cascading Impact Analysis
                    prev_faults = [(f, tf) for f, tf in fault_first_detected.items() if f != i and tf < cnt]
                    contribs = {}
                    if len(prev_faults) > 0:
                        for f, tf in prev_faults:
                            values_over_time = [frob_norms_matrix_history[tau][i][f] for tau in range(tf, cnt + 1) if tau < len(frob_norms_matrix_history)]
                            if len(values_over_time) > 0:
                                contribs[f] = float(np.mean(values_over_time))
                        if len(contribs) > 0:
                            ranked = sorted(contribs.items(), key=lambda x: x[1], reverse=True)
                            print(f"     >> Potential contributors to fault in agent {i} (mean ||H_{{i,f}}||_F from t_f to {cnt}): {ranked}")
                    fault_timeline.append({
                        'agent': i,
                        't': cnt,
                        'contribs': contribs
                    })

        taylor_error_list.append([np.mean(list(result_deques[j])) for j in range(runner.num_agents)])
        # pairwise_action_value_influence_list.append(pairwise_action_value_influence)
        
        # exit("Exiting for debug")
        # frob_norms_list.append([np.mean(frob_norms_deques[i]) for i in range(runner.num_agents)])
        # sec_dir_derivatives.append([np.mean(sec_dir_derivatives_deques[i]) for i in range(runner.num_agents)])

        (
            eval_obs,
            eval_share_obs,
            eval_rewards,
            eval_dones,
            eval_infos,
            eval_available_actions,
        ) = runner.eval_envs.step(eval_actions)
        total_rewards[tt] = eval_rewards
        tt=tt+1
        reward_ep+=np.sum(eval_rewards)
        eval_data = (
            eval_obs,
            eval_share_obs,
            eval_rewards,
            eval_dones,
            eval_infos,
            eval_available_actions,
        )

        value, eval_rnn_states_critic = runner.critic.get_values(
            eval_share_obs,
            eval_rnn_states_critic,
            eval_masks,
        )

        eval_dones_env = np.all(eval_dones, axis=1)

        eval_rnn_states[
            eval_dones_env == True
        ] = np.zeros(  # if env is done, then reset rnn_state to all zero
            (
                (eval_dones_env == True).sum(),
                runner.num_agents,
                runner.recurrent_n,
                runner.rnn_hidden_size,
            ),
            dtype=np.float32,
        )

        eval_masks = np.ones(
            (runner.algo_args["eval"]["n_eval_rollout_threads"], runner.num_agents, 1),
            dtype=np.float32,
        )
        eval_masks[eval_dones_env == True] = np.zeros(
            ((eval_dones_env == True).sum(), runner.num_agents, 1), dtype=np.float32
        )

        for eval_i in range(runner.algo_args["eval"]["n_eval_rollout_threads"]):
            if eval_dones_env[eval_i]:
                eval_episode += 1

        if eval_episode >= runner.algo_args["eval"]["eval_episodes"]:
            break

        cnt += 1
    
    # print(f"Final Pairwise Action Value Influence List: {pairwise_action_value_influence_history}")
    # return taylor_error_list, frob_norms_list, sec_dir_derivatives, frob_norms_matrix_history, fault_timeline, attacked_steps
    print(f"Episode ending reward: {reward_ep}")
    return taylor_error_list, pairwise_action_value_influence_history, total_rewards, frob_norms_matrix_history, fault_timeline, attacked_steps, q_values_list

