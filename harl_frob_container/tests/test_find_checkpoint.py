import importlib.util
import tempfile
import unittest
from pathlib import Path

if importlib.util.find_spec("torch") is not None:
    from harl.utils.models_tools import find_checkpoint
else:
    find_checkpoint = None


class FindCheckpointTests(unittest.TestCase):
    def setUp(self):
        if find_checkpoint is None:
            self.skipTest("missing dependency: torch")


    def test_prefers_exact_plain_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "actor_agent0.pt").write_text("x")
            (root / "actor_agent0_rew1.0000.pt").write_text("x")
            resolved = find_checkpoint(root, "actor_agent0")
            self.assertEqual(resolved, str(root / "actor_agent0.pt"))

    def test_selects_highest_reward_for_type_suffixed_files(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "actor_agent0_type_agent_rew1.2500.pt").write_text("x")
            (root / "actor_agent0_type_agent_rew2.5000.pt").write_text("x")
            (root / "actor_agent0_type_agent_rew0.5000.pt").write_text("x")
            resolved = find_checkpoint(root, "actor_agent0")
            self.assertEqual(
                resolved, str(root / "actor_agent0_type_agent_rew2.5000.pt")
            )


if __name__ == "__main__":
    unittest.main()
