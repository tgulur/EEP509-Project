"""Bottleneck projection layer for student privacy mitigation."""

from __future__ import annotations

import torch
from torch import nn


class BottleneckProjection(nn.Module):
    """Low-rank projection that limits hidden representation capacity."""

    def __init__(self, input_dim: int, rank: int) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("Bottleneck rank must be positive.")
        self.down = nn.Linear(input_dim, rank)
        self.activation = nn.ReLU()
        self.up = nn.Linear(rank, input_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.up(self.activation(self.down(inputs)))
