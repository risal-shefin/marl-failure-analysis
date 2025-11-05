"""Visualization utilities for MAPPO analysis."""
from __future__ import annotations

import imageio


def save_frames_as_gif(frames, path: str, fps: int = 10):
    imageio.mimsave(path, frames, fps=fps)
