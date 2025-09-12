"""Minimal actor-critic model used for PPOTrainer."""
from __future__ import annotations

from typing import Tuple
import numpy as np
import torch
from torch import nn


class ActorCriticModel(nn.Module):
    def __init__(self, obs_shape: Tuple[int, ...], action_space) -> None:
        super().__init__()
        flat_dim = int(np.prod(obs_shape))
        self.flatten = nn.Flatten()
        if hasattr(action_space, "n"):
            self.discrete = True
            n_actions = action_space.n
            self.policy = nn.Linear(flat_dim, n_actions)
        else:
            self.discrete = False
            n_actions = action_space.shape[0]
            self.policy = nn.Linear(flat_dim, n_actions)
        self.value = nn.Linear(flat_dim, 1)

    def forward(self, obs: torch.Tensor, rc, device):
        x = self.flatten(obs.float().to(device))
        logits = self.policy(x)
        value = self.value(x).squeeze(-1)
        if self.discrete:
            dist = torch.distributions.Categorical(logits=logits)
        else:
            dist = torch.distributions.Normal(logits, torch.ones_like(logits))
        return [dist], value, rc

    def init_recurrent_cell_states(self, n_envs: int, device):
        hxs = torch.zeros(n_envs, 1, device=device)
        cxs = torch.zeros(n_envs, 1, device=device)
        return hxs, cxs
