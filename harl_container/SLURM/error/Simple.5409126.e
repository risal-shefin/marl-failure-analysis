Traceback (most recent call last):
  File "_mt19937.pyx", line 178, in numpy.random._mt19937.MT19937._legacy_seeding
TypeError: 'str' object cannot be interpreted as an integer

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "train.py", line 95, in <module>
    main()
  File "train.py", line 89, in main
    runner = RUNNER_REGISTRY[args["algo"]](args, algo_args, env_args)
  File "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/harl/runners/on_policy_base_runner.py", line 47, in __init__
    set_seed(algo_args["seed"])
  File "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/harl/utils/envs_tools.py", line 233, in set_seed
    np.random.seed(args["seed"])
  File "mtrand.pyx", line 4789, in numpy.random.mtrand.seed
  File "mtrand.pyx", line 250, in numpy.random.mtrand.RandomState.seed
  File "_mt19937.pyx", line 166, in numpy.random._mt19937.MT19937._legacy_seeding
  File "_mt19937.pyx", line 186, in numpy.random._mt19937.MT19937._legacy_seeding
TypeError: Cannot cast scalar from dtype('<U4') to dtype('int64') according to the rule 'safe'
