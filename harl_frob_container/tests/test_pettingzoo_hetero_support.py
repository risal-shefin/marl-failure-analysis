import unittest
from unittest.mock import patch


class DummyActionSpace:
    def __init__(self, n=5):
        self.n = n


class DummyPZEnv:
    def __init__(self, env_args):
        self.n_agents = 3
        self.agents = ["adversary_0", "agent_0", "agent_1"]
        self.agent_types = ["adversary", "agent", "agent"]
        self.type_to_agent_ids = {"adversary": [0], "agent": [1, 2]}
        self.agent_id_to_type = {0: "adversary", 1: "agent", 2: "agent"}
        self.share_observation_space = [None] * 3
        self.observation_space = [None] * 3
        self.action_space = [DummyActionSpace(), DummyActionSpace(), DummyActionSpace()]

    def seed(self, _):
        pass




class DummyParallelEnv:
    def __init__(self):
        self.num_agents = 1
        self.agents = ["agent_0"]
        self.state_space = object()
        self.observation_spaces = {"agent_0": object()}
        self.action_spaces = {"agent_0": DummyActionSpace()}

    def reset(self, seed=None):
        return {"agent_0": 0.0}


class DummyPZModule:
    def __init__(self, kwargs_recorder):
        self.kwargs_recorder = kwargs_recorder

    def parallel_env(self, **kwargs):
        self.kwargs_recorder.update(kwargs)
        return DummyParallelEnv()


class PettingZooHeteroSupportTests(unittest.TestCase):
    def _require_runtime_deps(self):
        import importlib.util

        missing = [
            name
            for name in ("numpy", "supersuit", "torch", "absl")
            if importlib.util.find_spec(name) is None
        ]
        if missing:
            self.skipTest(f"missing dependencies: {missing}")

    def test_legacy_mode_rejects_adversarial_scenario(self):
        self._require_runtime_deps()
        from harl.utils.envs_tools import make_train_env

        env_args = {
            "scenario": "simple_adversary_v3",
            "continuous_actions": False,
            "render_mode": "rgb_array",
            "enable_heterogeneous_agents": False,
            "reward_mode": "global_sum_shared",
        }
        with self.assertRaises(AssertionError):
            make_train_env("pettingzoo_mpe", seed=1, n_threads=1, env_args=env_args)

    def test_heterogeneous_mode_allows_adversarial_scenario(self):
        self._require_runtime_deps()
        from harl.utils.envs_tools import make_train_env

        env_args = {
            "scenario": "simple_adversary_v3",
            "continuous_actions": False,
            "render_mode": "rgb_array",
            "enable_heterogeneous_agents": True,
            "reward_mode": "team_by_type",
        }
        with patch(
            "harl.envs.pettingzoo_mpe.pettingzoo_mpe_env.PettingZooMPEEnv", DummyPZEnv
        ):
            envs = make_train_env("pettingzoo_mpe", seed=1, n_threads=1, env_args=env_args)
        self.assertEqual(envs.n_agents, 3)
        self.assertEqual(envs.type_to_agent_ids["agent"], [1, 2])


    def test_harl_only_args_not_forwarded_to_pettingzoo_constructor(self):
        self._require_runtime_deps()
        from harl.envs.pettingzoo_mpe import pettingzoo_mpe_env as env_module

        captured_kwargs = {}
        with patch.object(env_module.importlib, "import_module", return_value=DummyPZModule(captured_kwargs)), patch.object(env_module.ss, "pad_observations_v0", side_effect=lambda e: e), patch.object(env_module.ss, "pad_action_space_v0", side_effect=lambda e: e):
            env = env_module.PettingZooMPEEnv(
                {
                    "scenario": "simple_adversary_v3",
                    "continuous_actions": False,
                    "render_mode": "rgb_array",
                    "enable_heterogeneous_agents": True,
                    "reward_mode": "team_by_type",
                }
            )
            self.assertTrue(env.enable_heterogeneous_agents)

        self.assertNotIn("enable_heterogeneous_agents", captured_kwargs)
        self.assertNotIn("reward_mode", captured_kwargs)
        self.assertIn("continuous_actions", captured_kwargs)

    def test_reward_aggregation_modes(self):
        self._require_runtime_deps()
        from harl.envs.pettingzoo_mpe.pettingzoo_mpe_env import PettingZooMPEEnv

        env = PettingZooMPEEnv.__new__(PettingZooMPEEnv)
        env.n_agents = 3
        env.agents = ["adversary_0", "agent_0", "agent_1"]
        env.agent_id_to_type = {0: "adversary", 1: "agent", 2: "agent"}

        rew = {"adversary_0": -1.0, "agent_0": 0.4, "agent_1": 0.6}

        env.reward_mode = "global_sum_shared"
        self.assertEqual(env._aggregate_rewards(rew), [[0.0], [0.0], [0.0]])

        env.reward_mode = "individual"
        self.assertEqual(env._aggregate_rewards(rew), [[-1.0], [0.4], [0.6]])

        env.reward_mode = "team_by_type"
        self.assertEqual(env._aggregate_rewards(rew), [[-1.0], [1.0], [1.0]])


if __name__ == "__main__":
    unittest.main()
