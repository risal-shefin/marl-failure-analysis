Traceback (most recent call last):
  File "attack_test.py", line 237, in <module>
    main()
  File "attack_test.py", line 230, in main
    eval(runner)  # Run evaluation
  File "attack_test.py", line 71, in eval
    runner.logger.eval_per_step(
  File "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/harl/common/base_logger.py", line 137, in eval_per_step
    self.one_episode_rewards[eval_i].append(eval_rewards[eval_i])
AttributeError: 'PettingZooMPELogger' object has no attribute 'one_episode_rewards'
slurmstepd: error: *** JOB 5409740 ON gpu-v100-02 CANCELLED AT 2025-08-04T12:59:08 ***
