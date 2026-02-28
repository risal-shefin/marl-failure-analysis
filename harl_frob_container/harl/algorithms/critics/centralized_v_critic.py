"""Centralized V Critic (per agent)."""
from harl.algorithms.critics.v_critic import VCritic


class CentralizedVCritic(VCritic):
    """Centralized V Critic (per agent).

    A per-agent centralized critic that learns a V-function from centralized
    (global) observations. Each agent maintains its own centralized critic
    with independent parameters. Training and loss computation follow the
    same procedure as the shared VCritic.
    """
