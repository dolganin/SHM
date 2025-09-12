"""Utility functions for PPO training with environment factory and schedules."""
from __future__ import annotations

from typing import Any, Dict
import math

from environments import create_env


def polynomial_decay(initial: float, final: float, max_decay_steps: int, power: float, step: int) -> float:
    """Polynomially decay a value over steps.

    Parameters
    ----------
    initial: float
        Starting value.
    final: float
        Final value after ``max_decay_steps``.
    max_decay_steps: int
        Number of steps over which to decay.
    power: float
        Power of the polynomial.
    step: int
        Current step.
    """
    step = min(step, max_decay_steps)
    decayed = (1 - step / max_decay_steps) ** power
    return (initial - final) * decayed + final


def create_env_from_config(config: Dict[str, Any]):
    """Thin wrapper around :func:`environments.create_env` for backward compatibility."""
    return create_env(config)
