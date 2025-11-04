"""Result saving helpers for MAPPO analysis."""
import json
import os
from typing import Dict


class ResultsSaver:
    """Persist experiment summaries to disk."""

    def __init__(self, logdir: str):
        self.logdir = logdir
        os.makedirs(self.logdir, exist_ok=True)

    def save_json(self, filename: str, data: Dict):
        path = os.path.join(self.logdir, filename)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        return path
