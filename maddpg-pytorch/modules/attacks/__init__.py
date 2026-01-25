"""
Attack module initialization.
"""
from .adversarial_attacks import fgsm_attack, preprocess_env_atari

__all__ = ['fgsm_attack', 'preprocess_env_atari']