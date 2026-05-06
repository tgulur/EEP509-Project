"""Shadow-model membership inference attack."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader

from attacks.base import MIAttack


class ShadowModelAttack(MIAttack):
    """Train a meta-classifier on model confidence features."""

    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)
        self.classifier = LogisticRegression(max_iter=1000)
        self.is_fitted = False

    def fit(
        self,
        model: torch.nn.Module,
        member_loader: DataLoader,
        non_member_loader: DataLoader,
    ) -> "ShadowModelAttack":
        member_features = self._attack_features(model, member_loader)
        non_member_features = self._attack_features(model, non_member_loader)
        features = np.vstack([member_features, non_member_features])
        labels = np.concatenate([np.ones(len(member_features)), np.zeros(len(non_member_features))])
        self.classifier.fit(features, labels)
        self.is_fitted = True
        return self

    def score(self, model: torch.nn.Module, samples: DataLoader) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Call fit before score.")
        features = self._attack_features(model, samples)
        return self.classifier.predict_proba(features)[:, 1]

    @torch.no_grad()
    def _attack_features(self, model: torch.nn.Module, loader: DataLoader) -> np.ndarray:
        model.to(self.device)
        model.eval()
        rows: list[np.ndarray] = []
        for features, _labels, _is_member in loader:
            probs = torch.softmax(model(features.to(self.device)), dim=1).cpu().numpy()
            top3 = -np.sort(-probs, axis=1)[:, :3]
            entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1, keepdims=True)
            rows.append(np.hstack([top3, entropy]))
        return np.vstack(rows)
