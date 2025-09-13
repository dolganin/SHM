"""Simple experience buffer for PPO training."""
from __future__ import annotations

from typing import List
import numpy as np
import torch


class Buffer:
    """Stores trajectories and computes GAE advantages."""

    def __init__(self, gamma: float = 0.99, lam: float = 0.95) -> None:
        self.gamma = gamma
        self.lam = lam
        self.reset()

    def reset(self) -> None:
        self.obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.vals: List[float] = []
        self.values: torch.Tensor = torch.tensor([])
        self.advantages: torch.Tensor = torch.tensor([])
        self.samples_flat = []

    def add(self, obs, action, reward, done, value) -> None:
        self.obs.append(np.asarray(obs))
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.vals.append(float(value))

    def prepare_batch_dict(self) -> None:
        """Compute returns and GAE advantages for logging."""
        returns: List[float] = []
        advantages: List[float] = []
        next_value = 0.0
        gae = 0.0
        for r, d, v in zip(reversed(self.rewards), reversed(self.dones), reversed(self.vals)):
            delta = r + self.gamma * next_value * (1.0 - float(d)) - v
            gae = delta + self.gamma * self.lam * (1.0 - float(d)) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + v)
            next_value = v
        self.values = torch.tensor(returns, dtype=torch.float32)
        self.advantages = torch.tensor(advantages, dtype=torch.float32)
        self.samples_flat = list(range(len(self.rewards)))
