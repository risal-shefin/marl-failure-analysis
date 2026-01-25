"""Environment helpers for MAPPO analysis scripts."""
from make_env_pettingzoo import make_env


def create_environment(config, runner=None):
    """Create a PettingZoo environment configured for MAPPO analysis."""
    discrete = getattr(config, 'discrete_action', True)
    env = make_env(config.env_id, discrete=discrete)
    return env
