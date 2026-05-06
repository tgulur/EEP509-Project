"""Loss-threshold membership inference attack."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from attacks.base import MIAttack


class LossBasedAttack(MIAttack):
    """Yeom et al. style attack using lower loss as stronger membership evidence."""

    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self.threshold: float | None = None

    def fit(
        self,
        model: torch.nn.Module,
        member_loader: DataLoader,
        non_member_loader: DataLoader,
    ) -> "LossBasedAttack":
        member_losses = -self.score(model, member_loader)
        non_member_losses = -self.score(model, non_member_loader)
        losses = np.concatenate([member_losses, non_member_losses])
        labels = np.concatenate([np.ones_like(member_losses), np.zeros_like(non_member_losses)])

        best_threshold = float(np.median(losses))
        best_acc = -1.0
        for threshold in np.unique(losses):
            predictions = (losses <= threshold).astype(np.int64)
            acc = float((predictions == labels).mean())
            if acc > best_acc:
                best_acc = acc
                best_threshold = float(threshold)
        self.threshold = best_threshold
        return self

    @torch.no_grad()
    def score(self, model: torch.nn.Module, samples: DataLoader) -> np.ndarray:
        model.to(self.device)
        model.eval()
        scores: list[np.ndarray] = []
        for features, labels, _is_member in samples:
            features = features.to(self.device)
            labels = labels.to(self.device)
            logits = model(features)
            losses = F.cross_entropy(logits, labels, reduction="none")
            scores.append((-losses).cpu().numpy())
        return np.concatenate(scores)

    def predict(self, model: torch.nn.Module, samples: DataLoader) -> np.ndarray:
        """Predict binary membership with the fitted threshold."""

        if self.threshold is None:
            raise RuntimeError("Call fit before predict.")
        losses = -self.score(model, samples)
        return (losses <= self.threshold).astype(np.int64)
