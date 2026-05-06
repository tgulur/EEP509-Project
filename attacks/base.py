"""Shared interface for membership inference attacks."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import DataLoader


class MIAttack(ABC):
    """Interface implemented by all membership inference attacks."""

    @abstractmethod
    def fit(
        self,
        model: torch.nn.Module,
        member_loader: DataLoader,
        non_member_loader: DataLoader,
    ) -> "MIAttack":
        """Fit attack parameters from member and non-member data."""

    @abstractmethod
    def score(self, model: torch.nn.Module, samples: DataLoader) -> np.ndarray:
        """Return membership scores where larger means more likely member."""
