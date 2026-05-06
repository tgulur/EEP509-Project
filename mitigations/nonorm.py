"""NoNorm layer used to replace BatchNorm in memorization-sensitive students."""

from __future__ import annotations

import torch
from torch import nn


class NoNorm(nn.Module):
    """Identity normalization layer with the same forward shape as BatchNorm."""

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs
