"""MiniGrid environment wrapper.

This module provides a light-weight wrapper around environments from the
`gymnasium-minigrid` package.  Observations are converted to ``float32``
and returned in channel-first (CHW) format so that vision-based agents can
process them consistently.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import gymnasium as gym
from minigrid.wrappers import ImgObsWrapper


class MinigridEnv:
    """Simple wrapper for MiniGrid environments.

    Parameters
    ----------
    env_id: str
        Name of the MiniGrid environment to instantiate.
    view_size: Optional[int]
        Overrides the agent's view size if provided.
    tile_size: Optional[int]
        Overrides the tile size used for rendering.
    kwargs: dict
        Additional arguments forwarded to ``gym.make``.
    """

    def __init__(self, env_id: str, view_size: Optional[int] = None,
                 tile_size: Optional[int] = None, **kwargs: Any) -> None:
        self._env = gym.make(
            env_id,
            agent_view_size=view_size,
            tile_size=tile_size,
            render_mode="rgb_array",
            **kwargs,
        )
        # Convert observations to images
        self._env = ImgObsWrapper(self._env)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, dict]:
        obs, info = self._env.reset(**kwargs)
        return obs.astype(np.float32), info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self._env.step(action)
        return obs.astype(np.float32), reward, terminated, truncated, info

    def render(self) -> np.ndarray:
        """Render the current environment state as an RGB image."""
        return self._env.render()

    def close(self) -> None:
        self._env.close()
