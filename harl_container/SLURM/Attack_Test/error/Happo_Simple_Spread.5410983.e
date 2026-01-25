Traceback (most recent call last):
  File "attack_test.py", line 304, in <module>
    main()
  File "attack_test.py", line 297, in main
    eval(runner)  # Run evaluation
  File "attack_test.py", line 117, in eval
    delta_errors = compute_taylor_policy(runner, eval_obs, eval_available_actions)
  File "attack_test.py", line 48, in compute_taylor_policy
    print(f"Available actions: {eval_actions}, {eval_actions.requires_grad}")
AttributeError: 'numpy.ndarray' object has no attribute 'requires_grad'
slurmstepd: error: *** JOB 5410983 ON gpu-v100-02 CANCELLED AT 2025-08-04T13:38:51 ***
