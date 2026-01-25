/home/guptd23/.local/lib/python3.8/site-packages/torch/cuda/__init__.py:146: UserWarning: 
NVIDIA A100-PCIE-40GB with CUDA capability sm_80 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_37 sm_50 sm_60 sm_70.
If you want to use the NVIDIA A100-PCIE-40GB GPU with PyTorch, please check the instructions at https://pytorch.org/get-started/locally/

  warnings.warn(incompatible_device_warn.format(device_name, capability, " ".join(arch_list), device_name))
Traceback (most recent call last):
  File "train.py", line 96, in <module>
    main()
  File "train.py", line 90, in main
    runner.run()
  File "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/harl/runners/on_policy_base_runner.py", line 211, in run
    ) = self.collect(step)
  File "/home/guptd23/.local/lib/python3.8/site-packages/torch/autograd/grad_mode.py", line 27, in decorate_context
    return func(*args, **kwargs)
  File "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/harl/runners/on_policy_base_runner.py", line 298, in collect
    action, action_log_prob, rnn_state = self.actor[agent_id].get_actions(
  File "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/harl/algorithms/actors/on_policy_base.py", line 64, in get_actions
    actions, action_log_probs, rnn_states_actor = self.actor(
  File "/home/guptd23/.local/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/harl/models/policy_models/stochastic_policy.py", line 77, in forward
    actor_features = self.base(obs)
  File "/home/guptd23/.local/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "/deac/csc/vanbastelaerGrp/guptd23/RL_Project/HARL/harl/models/base/mlp.py", line 66, in forward
    x = self.feature_norm(x)
  File "/home/guptd23/.local/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1130, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/guptd23/.local/lib/python3.8/site-packages/torch/nn/modules/normalization.py", line 189, in forward
    return F.layer_norm(
  File "/home/guptd23/.local/lib/python3.8/site-packages/torch/nn/functional.py", line 2503, in layer_norm
    return torch.layer_norm(input, normalized_shape, weight, bias, eps, torch.backends.cudnn.enabled)
RuntimeError: CUDA error: no kernel image is available for execution on the device
CUDA kernel errors might be asynchronously reported at some other API call,so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
slurmstepd: error: *** JOB 5421511 ON gpu-a100-03 CANCELLED AT 2025-08-05T15:25:35 ***
