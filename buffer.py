"""Simple experience buffer for PPO training."""
from __future__ import annotations

from typing import List
import numpy as np
import torch


class Buffer:
    """Stores trajectories and computes simple advantages."""

    def __init__(self, gamma: float = 0.99) -> None:
        self.gamma = gamma
        self.reset()

    def reset(self) -> None:
        self.obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.values: torch.Tensor = torch.tensor([])
        self.advantages: torch.Tensor = torch.tensor([])
        self.samples_flat = []

    def add(self, obs, action, reward, done, value) -> None:
        self.obs.append(np.asarray(obs))
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)

    def prepare_batch_dict(self) -> None:
        """Compute returns and dummy advantages for logging."""
        returns = []
        g = 0.0
        for r, d in zip(reversed(self.rewards), reversed(self.dones)):
            g = r + self.gamma * g * (1.0 - float(d))
            returns.append(g)
        returns.reverse()
        self.values = torch.tensor(returns, dtype=torch.float32)
        self.advantages = self.values.clone()
        self.samples_flat = list(range(len(self.rewards)))
