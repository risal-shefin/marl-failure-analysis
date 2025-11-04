"""Ranking accuracy utilities for MAPPO analysis."""
from __future__ import annotations

from typing import Dict, List

import numpy as np


class AccuracyComputer:
    """Compute ranking agreement metrics across different signals."""

    @staticmethod
    def compute_matching_accuracy(rankings: Dict[str, List[int]]) -> Dict[str, float]:
        reference = rankings.get('influence') or rankings.get('shapley')
        if not reference:
            return {}

        accuracy = {}
        for name, ranking in rankings.items():
            if name == 'influence' or name == 'shapley':
                continue
            matches = sum(int(a == b) for a, b in zip(reference, ranking))
            accuracy[name] = matches / len(reference)
        return accuracy

    @staticmethod
    def compute_spearman(rank_a: List[int], rank_b: List[int]) -> float:
        if len(rank_a) != len(rank_b):
            raise ValueError('Rankings must have the same length')
        diff = np.array(rank_a) - np.array(rank_b)
        n = len(rank_a)
        return 1 - (6 * np.sum(diff ** 2)) / (n * (n ** 2 - 1))
