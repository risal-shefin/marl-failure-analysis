import copy
import importlib
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)


def _get_gym_spaces():
    try:
        import gymnasium as gym
        return gym.spaces
    except Exception:
        try:
            from gym import spaces as gym_spaces  # legacy
            return gym_spaces
        except Exception:
            raise ImportError("Neither gymnasium nor gym spaces are available. Please install gymnasium.")


class VMASEnv:
    """HARL-compatible wrapper for VMAS environments.

    Matches the interface of PettingZooMPEEnv so runners work unchanged.
    """

    def __init__(self, args: Dict[str, Any]):
        self.args = copy.deepcopy(args)
        self.scenario = self.args["scenario"]
        # defaults
        self.discrete = not self.args.get("continuous_actions", False)
        self.max_cycles = int(self.args.get("max_cycles", 25))
        # VMAS uses max_steps; we keep an internal +1 like PettingZooMPEEnv
        self._max_steps_internal = self.max_cycles + 1
        # number of agents
        self.n_agents = int(self.args.get("N", self.args.get("n_agents", 2)))
        self.device = self.args.get("device", "cpu")
        self._seed = 0
        self.cur_step = 0

        # Compatibility shim for Python < 3.9: argparse.BooleanOptionalAction
        import argparse

        if not hasattr(argparse, "BooleanOptionalAction"):

            class BooleanOptionalAction(argparse.Action):
                def __init__(
                    self,
                    option_strings,
                    dest,
                    default=None,
                    type=None,
                    choices=None,
                    required=False,
                    help=None,
                    metavar=None,
                ):
                    # create --flag / --no-flag variants
                    opts = []
                    for opt in option_strings:
                        opts.append(opt)
                        if opt.startswith("--"):
                            opts.append("--no-" + opt[2:])
                    super().__init__(
                        option_strings=opts,
                        dest=dest,
                        nargs=0,
                        default=default,
                        type=type,
                        choices=choices,
                        required=required,
                        help=help,
                        metavar=metavar,
                    )

                def __call__(self, parser, namespace, values, option_string=None):
                    setattr(namespace, self.dest, not option_string.startswith("--no-"))

            argparse.BooleanOptionalAction = BooleanOptionalAction

        # Lazy import vmas; raise a helpful error if missing
        try:
            import vmas  # type: ignore
        except Exception as e:
            raise ImportError(
                "vmas is required for VMASEnv. Install with `pip install vmas`"
            ) from e

        # Build VMAS environment (single vectorized env: num_envs=1)
        self._vmas = vmas.make_env(
            scenario=self.scenario,
            num_envs=1,
            device=self.device,
            continuous_actions=not self.discrete,
            max_steps=self._max_steps_internal,
            seed=None,
            n_agents=self.n_agents,
        )

        # Initial reset to populate agents/spaces
        _ = self._vmas.reset()

        # Collect agent names (fallback to agent_i)
        try:
            self.agents: List[str] = [a.name for a in self._vmas.agents]
        except Exception:
            self.agents = [f"agent_{i}" for i in range(self.n_agents)]

        # Observation and action spaces per agent (robust resolution)
        self.observation_space = self._resolve_observation_spaces()
        # Per-agent action metadata corresponding to normalized action_space
        self._component_nvecs: List[Optional[List[int]]] = [None] * self.n_agents
        self._is_pure_discrete: List[bool] = [False] * self.n_agents
        self.action_space = self._resolve_action_spaces()

        # Share/global observation space (EP style similar to PettingZooMPE)
        # Prefer state_space if available; else sum local obs dims
        share_box = None
        if hasattr(self._vmas, "state_space") and self._vmas.state_space is not None:
            share_box = self._vmas.state_space
        else:
            gym_spaces = _get_gym_spaces()
            # First try from declared spaces if all have valid shapes
            try:
                shapes = [sp.shape for sp in self.observation_space]
                if any(s is None for s in shapes):
                    raise ValueError("obs space shape None")
                dim = int(sum(int(np.prod(s)) for s in shapes))
            except Exception:
                # Fallback: infer from current observation tensors
                o = self._vmas.observe() if hasattr(self._vmas, "observe") else self._vmas.reset()
                obs_list = self._obs_list_from_tensor(o)
                dim = int(sum(obs.size for obs in obs_list))
            share_box = gym_spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)
        self.share_observation_space = [share_box for _ in range(self.n_agents)]

    # --- HARL API ---
    def step(self, actions: np.ndarray):
        """Step the environment.

        Args:
            actions: (n_agents, action_dim). For discrete, may be (n_agents, 1) ints or logits.
        Returns:
            obs(list), share_obs(list), rewards(list[list[1]] per agent),
            dones(list[bool]), infos(list[dict]), available_actions(list or None)
        """
        import torch

        act_list = []
        for i in range(self.n_agents):
            if self.discrete:
                # Always derive a scalar action from possibly vector/logits
                a_i = actions[i]
                if isinstance(a_i, np.ndarray):
                    a_i = a_i.squeeze()
                    if a_i.size > 1:
                        a_scalar = int(np.argmax(a_i))
                    else:
                        a_scalar = int(a_i.item())
                else:
                    a_scalar = int(a_i)
                # Determine expected VMAS action size for this agent
                try:
                    size = int(self._vmas.get_agent_action_size(self._vmas.agents[i]))
                except Exception:
                    size = 1
                if size <= 1:
                    act_list.append(torch.tensor([a_scalar], device=self.device, dtype=torch.long))
                else:
                    # Expand scalar back to vector of discrete components using stored nvec
                    nvec = self._component_nvecs[i]
                    if not nvec or len(nvec) != size:
                        raise RuntimeError(
                            f"Discrete action needs expansion to length {size}, but component sizes are unknown. "
                            f"Please set env_args to provide component sizes (e.g., action_nvec) or use a VMAS scenario exposing MultiDiscrete."
                        )
                    comps = self._unflatten_scalar(a_scalar, nvec)
                    a_vec = np.asarray(comps, dtype=np.int64).reshape(1, -1)
                    act_list.append(torch.tensor(a_vec, device=self.device, dtype=torch.long))
            else:
                a_i = np.asarray(actions[i], dtype=np.float32).reshape(1, -1)
                act_list.append(torch.tensor(a_i, device=self.device, dtype=torch.float32))
        # VMAS expects list of per-agent tensors
        o, r, d, info = self._vmas.step(act_list)
        self.cur_step += 1

        # Convert outputs
        obs = self._obs_list_from_tensor(o)
        share_obs = self._build_share_obs(obs)

        # rewards: per-agent vector -> team reward repeated like PettingZooMPEEnv
        r_np = self._to_numpy(r)
        if r_np.ndim == 2:  # (num_envs=1, n_agents)
            r_np = r_np[0]
        total_reward = float(np.sum(r_np))
        rewards = [[total_reward]] * self.n_agents

        # dones: VMAS returns per-env bool or tensor; expand to per-agent list
        dones_flag = self._done_flag(d)
        dones = [dones_flag for _ in range(self.n_agents)]

        # infos per agent; include bad_transition on truncation by horizon
        infos = [{} for _ in range(self.n_agents)]
        if self.cur_step >= self.max_cycles:
            for i in range(self.n_agents):
                infos[i]["bad_transition"] = True

        # Available actions (only when all agents are truly Discrete originally)
        available_actions = self.get_avail_actions()
        return obs, share_obs, rewards, dones, infos, available_actions

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._seed = seed
        else:
            self._seed += 1
        self.cur_step = 0
        _ = self._vmas.reset(seed=self._seed)
        # always use observe after reset for robustness
        if hasattr(self._vmas, "observe"):
            o = self._vmas.observe()
        else:
            o = _
        obs = self._obs_list_from_tensor(o)
        share_obs = self._build_share_obs(obs)
        available_actions = self.get_avail_actions()
        return obs, share_obs, available_actions

    def render(self, mode: str = "human"):
        try:
            return self._vmas.render(mode=mode)
        except Exception:
            return None

    def close(self):
        try:
            self._vmas.close()
        except Exception:
            pass

    def seed(self, seed: int):
        self._seed = seed

    # --- Helpers ---
    def get_avail_actions(self):
        # Provide available_actions if all normalized per-agent spaces are Discrete
        names = [sp.__class__.__name__ for sp in self.action_space]
        if all(n == "Discrete" for n in names):
            avail = []
            for i in range(self.n_agents):
                n = self.action_space[i].n
                avail.append([1] * n)
            return avail
        return None

    def _resolve_observation_spaces(self):
        # Try direct container on env
        if hasattr(self._vmas, "observation_spaces"):
            cont = self._vmas.observation_spaces
            if isinstance(cont, dict):
                if all(isinstance(k, int) for k in cont.keys()):
                    return [cont[i] for i in range(self.n_agents)]
                else:
                    # map by agent names if available
                    if all(name in cont for name in self.agents):
                        return [cont[name] for name in self.agents]
                    return list(cont.values())
            elif isinstance(cont, (list, tuple)):
                return list(cont)
        # Single-space API
        if hasattr(self._vmas, "observation_space"):
            gym_spaces = _get_gym_spaces()
            sp = self._vmas.observation_space
            # If env provides a gym Tuple across agents, unpack it
            if getattr(sp.__class__, "__name__", "") == "Tuple" and hasattr(sp, "spaces"):
                if len(sp.spaces) == self.n_agents:
                    return list(sp.spaces)
            if isinstance(sp, (list, tuple)):
                return list(sp)
            if isinstance(sp, dict):
                return [sp[i] for i in range(self.n_agents)] if all(isinstance(k, int) for k in sp.keys()) else list(sp.values())
            # Only replicate if the space has a concrete shape
            if hasattr(sp, "shape") and sp.shape is not None:
                return [sp for _ in range(self.n_agents)]
        # Try per-agent attribute
        try:
            return [getattr(a, "observation_space") for a in self._vmas.agents]
        except Exception:
            pass
        # Fallback: infer from observe() result
        gym_spaces = _get_gym_spaces()
        o = self._vmas.observe() if hasattr(self._vmas, "observe") else self._vmas.reset()
        obs_list = self._obs_list_from_tensor(o)
        return [gym_spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32) for obs in obs_list]

    def _resolve_action_spaces(self):
        # Build raw per-agent spaces first
        raw_spaces = None
        if hasattr(self._vmas, "action_spaces"):
            cont = self._vmas.action_spaces
            if isinstance(cont, dict):
                if all(isinstance(k, int) for k in cont.keys()):
                    raw_spaces = [cont[i] for i in range(self.n_agents)]
                else:
                    if all(name in cont for name in self.agents):
                        raw_spaces = [cont[name] for name in self.agents]
                    else:
                        raw_spaces = list(cont.values())
            elif isinstance(cont, (list, tuple)):
                raw_spaces = list(cont)
        if raw_spaces is None and hasattr(self._vmas, "action_space"):
            sp = self._vmas.action_space
            if getattr(sp.__class__, "__name__", "") == "Tuple" and hasattr(sp, "spaces") and len(sp.spaces) == self.n_agents:
                raw_spaces = list(sp.spaces)
            elif isinstance(sp, (list, tuple)):
                raw_spaces = list(sp)
            elif isinstance(sp, dict):
                raw_spaces = [sp[i] for i in range(self.n_agents)] if all(isinstance(k, int) for k in sp.keys()) else list(sp.values())
            else:
                raw_spaces = [sp for _ in range(self.n_agents)]
        if raw_spaces is None:
            try:
                raw_spaces = [getattr(a, "action_space") for a in self._vmas.agents]
            except Exception:
                raw_spaces = []
        # Apply normalization per agent
        gym_spaces = _get_gym_spaces()
        norm_spaces = []
        # Optional overrides
        override_discrete_n = int(self.args["action_n"]) if self.discrete and ("action_n" in self.args) else None
        override_cont_dim = int(self.args["action_dim"]) if (not self.discrete) and ("action_dim" in self.args) else None
        # New: per-agent discrete component sizes
        override_nvec: Optional[Union[List[int], List[List[int]]]] = self.args.get("action_nvec", None) if self.discrete else None
        def get_nvec_for_agent(agent_idx: int) -> Optional[List[int]]:
            if override_nvec is None:
                return None
            if isinstance(override_nvec, list) and len(override_nvec) > 0 and all(isinstance(x, int) for x in override_nvec):
                return list(map(int, override_nvec))
            if isinstance(override_nvec, list) and len(override_nvec) == self.n_agents and all(isinstance(v, list) for v in override_nvec):
                return list(map(int, override_nvec[agent_idx]))
            return None
        for i in range(self.n_agents):
            # Highest priority: explicit nvec override
            nvec_i = get_nvec_for_agent(i)
            if nvec_i is not None:
                prod = int(np.prod(nvec_i))
                norm_spaces.append(gym_spaces.Discrete(prod))
                self._is_pure_discrete[i] = False
                self._component_nvecs[i] = nvec_i
                continue
            # Next priority: simple n override
            if override_discrete_n is not None:
                norm_spaces.append(gym_spaces.Discrete(override_discrete_n))
                self._is_pure_discrete[i] = True
                self._component_nvecs[i] = None
                continue
            if override_cont_dim is not None:
                norm_spaces.append(gym_spaces.Box(low=-1.0, high=1.0, shape=(override_cont_dim,), dtype=np.float32))
                self._is_pure_discrete[i] = False
                self._component_nvecs[i] = None
                continue
            s = raw_spaces[i] if i < len(raw_spaces) else None
            if s is None:
                if self.discrete:
                    norm_spaces.append(gym_spaces.Discrete(2))
                    self._is_pure_discrete[i] = True
                else:
                    norm_spaces.append(gym_spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32))
                    self._is_pure_discrete[i] = False
                self._component_nvecs[i] = None
                continue
            name = s.__class__.__name__
            if name == "Discrete":
                norm_spaces.append(s)
                self._is_pure_discrete[i] = True
                self._component_nvecs[i] = None
            elif name == "MultiDiscrete":
                nvec = list(map(int, getattr(s, "nvec", [])))
                prod = int(np.prod(nvec)) if len(nvec) > 0 else 0
                norm_spaces.append(gym_spaces.Discrete(prod))
                self._is_pure_discrete[i] = False
                self._component_nvecs[i] = nvec
        # ...existing code below...
