"""Reference model training utilities for full LiRA attack.

This module handles efficient training of multiple reference models on random
subsets of the training data, and caches confidences for LiRA scoring.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ReferenceModelCache:
    """Cache storing reference model membership and confidence information.

    For each reference model i:
      - membership_masks[i]: boolean array indicating which samples were in training
      - confidence_matrix[i, j]: confidence of model i on sample j (in logit space)

    For the target model:
      - target_confidences[j]: confidence of target model on sample j
    """

    num_models: int
    target_indices: np.ndarray
    membership_masks: np.ndarray = field(default_factory=lambda: np.array([]))
    confidence_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    target_confidences: np.ndarray = field(default_factory=lambda: np.array([]))

    def get_confidences_for_sample(self, sample_idx: int) -> tuple[np.ndarray, np.ndarray]:
        """Get in-model and out-model confidences for a specific sample.

        Args:
            sample_idx: Global sample index

        Returns:
            Tuple of (in_confidences, out_confidences) arrays
        """
        if sample_idx not in self._idx_to_pos:
            return np.array([]), np.array([])

        pos = self._idx_to_pos[sample_idx]
        in_mask = self.membership_masks[:, pos]
        out_mask = ~in_mask

        confs = self.confidence_matrix[:, pos]
        return confs[in_mask], confs[out_mask]

    def get_target_confidence(self, sample_idx: int) -> float | None:
        """Get target model's confidence for a sample."""
        if sample_idx not in self._idx_to_pos:
            return None
        pos = self._idx_to_pos[sample_idx]
        if pos >= len(self.target_confidences):
            return None
        return float(self.target_confidences[pos])

    def compute_target_model_confidences(
        self,
        model: nn.Module,
        features: np.ndarray,
        labels: np.ndarray,
        target_indices: np.ndarray,
        device: torch.device,
        batch_size: int = 256,
    ) -> None:
        """Compute and store confidences from the target model."""
        self.target_confidences = _compute_confidences(
            model, features, labels, target_indices, device, batch_size
        )

    @property
    def _idx_to_pos(self) -> dict[int, int]:
        """Map global sample index to position in arrays."""
        if not hasattr(self, "_idx_map"):
            self._idx_map = {int(idx): pos for pos, idx in enumerate(self.target_indices)}
        return self._idx_map

    def save(self, path: Path) -> None:
        """Save cache to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({
                "num_models": self.num_models,
                "target_indices": self.target_indices,
                "membership_masks": self.membership_masks,
                "confidence_matrix": self.confidence_matrix,
                "target_confidences": self.target_confidences,
            }, f)

    @classmethod
    def load(cls, path: Path) -> "ReferenceModelCache":
        """Load cache from disk."""
        with path.open("rb") as f:
            data = pickle.load(f)
        cache = cls(
            num_models=data["num_models"],
            target_indices=data["target_indices"],
            membership_masks=data["membership_masks"],
            confidence_matrix=data["confidence_matrix"],
            target_confidences=data.get("target_confidences", np.array([])),
        )
        return cache


def train_reference_models(
    features: np.ndarray,
    labels: np.ndarray,
    target_indices: np.ndarray,
    num_models: int,
    hidden_dims: list[int],
    epochs: int,
    sample_fraction: float,
    device: torch.device,
    num_classes: int,
    cache_dir: Path | None = None,
    batch_size: int = 256,
    lr: float = 0.001,
    seed: int = 509,
) -> ReferenceModelCache:
    """Train multiple reference models and build confidence cache.

    Args:
        features: Full feature array (n_samples, n_features)
        labels: Full label array (n_samples,)
        target_indices: Indices of samples to evaluate
        num_models: Number of reference models to train
        hidden_dims: Hidden layer dimensions for reference models
        epochs: Training epochs per model
        sample_fraction: Fraction of data to sample for each model
        device: Torch device
        num_classes: Number of output classes
        cache_dir: Optional directory to cache models
        batch_size: Training batch size
        lr: Learning rate
        seed: Random seed

    Returns:
        ReferenceModelCache with membership masks and confidence matrix
    """
    if cache_dir is not None:
        cache_path = cache_dir / f"lira_cache_n{num_models}_f{sample_fraction:.2f}.pkl"
        if cache_path.exists():
            print(f"Loading cached reference models from {cache_path}")
            return ReferenceModelCache.load(cache_path)

    rng = np.random.default_rng(seed)
    input_dim = features.shape[1]
    n_samples = len(labels)
    n_targets = len(target_indices)
    sample_size = int(n_samples * sample_fraction)

    membership_masks = np.zeros((num_models, n_targets), dtype=bool)
    confidence_matrix = np.zeros((num_models, n_targets), dtype=np.float32)

    target_set = set(target_indices)
    target_idx_to_pos = {int(idx): pos for pos, idx in enumerate(target_indices)}

    print(f"Training {num_models} reference models for LiRA...")
    for model_idx in range(num_models):
        subset_indices = rng.choice(n_samples, size=sample_size, replace=False)
        subset_set = set(subset_indices)

        for idx in target_indices:
            if idx in subset_set:
                membership_masks[model_idx, target_idx_to_pos[idx]] = True

        model = _build_reference_model(input_dim, num_classes, hidden_dims)
        model = _train_model(
            model, features, labels, subset_indices,
            device, epochs, batch_size, lr
        )

        confidences = _compute_confidences(
            model, features, labels, target_indices, device, batch_size
        )
        confidence_matrix[model_idx] = confidences

        if (model_idx + 1) % 10 == 0 or model_idx == num_models - 1:
            print(f"  Trained {model_idx + 1}/{num_models} reference models")

    cache = ReferenceModelCache(
        num_models=num_models,
        target_indices=target_indices,
        membership_masks=membership_masks,
        confidence_matrix=confidence_matrix,
    )

    if cache_dir is not None:
        cache.save(cache_path)
        print(f"Saved reference model cache to {cache_path}")

    return cache


def _build_reference_model(
    input_dim: int,
    num_classes: int,
    hidden_dims: list[int],
    dropout: float = 0.1,
) -> nn.Module:
    """Build a lightweight MLP for reference model training."""
    layers: list[nn.Module] = []
    prev_dim = input_dim
    for dim in hidden_dims:
        layers.extend([
            nn.Linear(prev_dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])
        prev_dim = dim
    layers.append(nn.Linear(prev_dim, num_classes))
    return nn.Sequential(*layers)


def _train_model(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
) -> nn.Module:
    """Train a single reference model."""
    model.to(device)
    model.train()

    train_features = torch.tensor(features[train_indices], dtype=torch.float32)
    train_labels = torch.tensor(labels[train_indices], dtype=torch.long)
    dataset = TensorDataset(train_features, train_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for _ in range(epochs):
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    return model


@torch.no_grad()
def _compute_confidences(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    eps: float = 1e-6,
) -> np.ndarray:
    """Compute logit-space confidence on true label for specified indices."""
    model.to(device)
    model.eval()

    subset_features = torch.tensor(features[indices], dtype=torch.float32)
    subset_labels = torch.tensor(labels[indices], dtype=torch.long)
    dataset = TensorDataset(subset_features, subset_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    confidences = []
    for batch_features, batch_labels in loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        probs = torch.softmax(model(batch_features), dim=1)
        conf = probs.gather(1, batch_labels.view(-1, 1)).squeeze(1)
        conf = torch.logit(conf.clamp(eps, 1.0 - eps))
        confidences.append(conf.cpu().numpy())

    return np.concatenate(confidences)
