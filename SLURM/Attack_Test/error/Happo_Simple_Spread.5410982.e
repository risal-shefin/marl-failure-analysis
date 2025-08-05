Traceback (most recent call last):
  File "attack_test.py", line 304, in <module>
    main()
  File "attack_test.py", line 297, in main
    eval(runner)  # Run evaluation
  File "attack_test.py", line 117, in eval
    delta_errors = compute_taylor_policy(runner, eval_obs, eval_available_actions)
  File "attack_test.py", line 34, in compute_taylor_policy
    eval_actions, temp_rnn_state = runner.actor[agent_id].get_actions(
ValueError: too many values to unpack (expected 2)
slurmstepd: error: *** JOB 5410982 ON gpu-v100-02 CANCELLED AT 2025-08-04T13:38:53 ***
