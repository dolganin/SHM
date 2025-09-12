"""Environment factory for SHM RL experiments.

Provides a unified ``create_env`` helper that instantiates the
appropriate wrapper based on a configuration dictionary.  The goal is to
match the interface described in the user's request where different
environments (e.g. MiniGrid and VizDoom) can be created with a common
API so that experiments can be swapped easily.

Each environment wrapper exposes the standard OpenAI Gym interface with
``reset`` and ``step`` methods returning observations in ``float32``.
"""
from __future__ import annotations

from typing import Any, Dict

from .minigrid_env import MinigridEnv
from .vizdoom_env import VizdoomEnv


def create_env(config: Dict[str, Any]):
    """Create an environment from a configuration dictionary.

    Parameters
    ----------
    config: Dict[str, Any]
        Configuration with at least a ``"type"`` key.  Additional keys
        under ``"params"`` are forwarded to the environment constructor.

    Returns
    -------
    gym.Env
        Instantiated environment.
    """
    env_type = config.get("type")
    params = config.get("params", {})
    if env_type == "minigrid":
        return MinigridEnv(**params)
    if env_type == "vizdoom":
        return VizdoomEnv(**params)
    raise ValueError(f"Unknown environment type: {env_type}")
