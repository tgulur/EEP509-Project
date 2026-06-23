"""Reference model training + confidence cache for full LiRA.

The cache stores, for each (reference_model, target_sample) pair, whether the
sample was IN the model's training subset and what confidence the model gave it.
That's enough to build per-sample IN/OUT Gaussians at scoring time.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class ReferenceModelCache:
    # membership_masks[i, j] = was sample j in the training subset of reference model i?
    # confidence_matrix[i, j] = model i's logit-space confidence on sample j
    # correctness_matrix[i, j] = did model i argmax-predict sample j correctly? (for Feldman-Zhang)
    # target_confidences[j]   = the target model's confidence on sample j
    num_models: int
    target_indices: np.ndarray
    membership_masks: np.ndarray = field(default_factory=lambda: np.array([]))
    confidence_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    correctness_matrix: np.ndarray = field(default_factory=lambda: np.array([]))
    target_confidences: np.ndarray = field(default_factory=lambda: np.array([]))

    def get_confidences_for_sample(self, sample_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if sample_idx not in self._idx_to_pos:
            return np.array([]), np.array([])
        pos = self._idx_to_pos[sample_idx]
        in_mask = self.membership_masks[:, pos]
        confs = self.confidence_matrix[:, pos]
        return confs[in_mask], confs[~in_mask]

    def get_correctness_for_sample(self, sample_idx: int) -> tuple[np.ndarray, np.ndarray]:
        if sample_idx not in self._idx_to_pos or self.correctness_matrix.size == 0:
            return np.array([]), np.array([])
        pos = self._idx_to_pos[sample_idx]
        in_mask = self.membership_masks[:, pos]
        correct = self.correctness_matrix[:, pos]
        return correct[in_mask], correct[~in_mask]

    def get_target_confidence(self, sample_idx: int) -> float | None:
        pos = self._idx_to_pos.get(sample_idx)
        if pos is None or pos >= len(self.target_confidences):
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
        self.target_confidences = _compute_confidences(
            model, features, labels, target_indices, device, batch_size
        )

    @property
    def _idx_to_pos(self) -> dict[int, int]:
        if not hasattr(self, "_idx_map"):
            self._idx_map = {int(idx): pos for pos, idx in enumerate(self.target_indices)}
        return self._idx_map

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({
                "num_models": self.num_models,
                "target_indices": self.target_indices,
                "membership_masks": self.membership_masks,
                "confidence_matrix": self.confidence_matrix,
                "correctness_matrix": self.correctness_matrix,
                "target_confidences": self.target_confidences,
            }, f)

    @classmethod
    def load(cls, path: Path) -> "ReferenceModelCache":
        with path.open("rb") as f:
            data = pickle.load(f)
        return cls(
            num_models=data["num_models"],
            target_indices=data["target_indices"],
            membership_masks=data["membership_masks"],
            confidence_matrix=data["confidence_matrix"],
            correctness_matrix=data.get("correctness_matrix", np.array([])),
            target_confidences=data.get("target_confidences", np.array([])),
        )


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
    model_config: dict | None = None,
    feature_metadata: dict[str, object] | None = None,
) -> ReferenceModelCache:
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
    correctness_matrix = np.zeros((num_models, n_targets), dtype=np.float32)

    target_idx_to_pos = {int(idx): pos for pos, idx in enumerate(target_indices)}

    print(f"Training {num_models} reference models for LiRA...")
    for model_idx in range(num_models):
        subset_indices = rng.choice(n_samples, size=sample_size, replace=False)
        subset_set = set(subset_indices)

        for idx in target_indices:
            if idx in subset_set:
                membership_masks[model_idx, target_idx_to_pos[idx]] = True

        model = _build_reference_model(
            input_dim, num_classes, hidden_dims,
            model_config=model_config, feature_metadata=feature_metadata,
        )
        model = _train_model(
            model, features, labels, subset_indices,
            device, epochs, batch_size, lr,
            model_config=model_config,
        )

        confidences, correctness = _compute_confidences_and_correctness(
            model, features, labels, target_indices, device, batch_size
        )
        confidence_matrix[model_idx] = confidences
        correctness_matrix[model_idx] = correctness

        if (model_idx + 1) % 10 == 0 or model_idx == num_models - 1:
            print(f"  Trained {model_idx + 1}/{num_models} reference models")

    cache = ReferenceModelCache(
        num_models=num_models,
        target_indices=target_indices,
        membership_masks=membership_masks,
        confidence_matrix=confidence_matrix,
        correctness_matrix=correctness_matrix,
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
    model_config: dict | None = None,
    feature_metadata: dict[str, object] | None = None,
) -> nn.Module:
    # mirror build_teacher when a model_config is provided so the reference distribution
    # matches the target distribution. Carlini requires reference models to use the same
    # training procedure as the target; the architecture is half of that.
    if model_config is not None:
        from models.teacher import build_teacher
        return build_teacher(
            input_dim,
            num_classes,
            hidden_dims,
            float(model_config.get("dropout", dropout)),
            teacher_type=str(model_config.get("teacher_type", "mlp")),
            metadata=feature_metadata,
            embedding_dim=int(model_config.get("embedding_dim", 16)),
        )
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
    model_config: dict | None = None,
) -> nn.Module:
    model.to(device)
    model.train()

    train_features = torch.tensor(features[train_indices], dtype=torch.float32)
    train_labels = torch.tensor(labels[train_indices], dtype=torch.long)
    dataset = TensorDataset(train_features, train_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # when called with model_config, mirror the teacher's optimizer + scheduler. otherwise
    # fall back to plain Adam (legacy path used by older callers/tests).
    if model_config is not None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(model_config.get("lr", lr)),
            weight_decay=float(model_config.get("weight_decay", 0.0)),
        )
        scheduler = None
        sched_cfg = model_config.get("scheduler", {})
        if sched_cfg.get("enabled"):
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(sched_cfg.get("step_size", 20)),
                gamma=float(sched_cfg.get("gamma", 0.5)),
            )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = None

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
        if scheduler is not None:
            scheduler.step()

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
    confidences, _ = _compute_confidences_and_correctness(
        model, features, labels, indices, device, batch_size, eps
    )
    return confidences


@torch.no_grad()
def _compute_confidences_and_correctness(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    # logit-space confidence on true label, plus 0/1 correctness of argmax prediction
    model.to(device)
    model.eval()

    subset_features = torch.tensor(features[indices], dtype=torch.float32)
    subset_labels = torch.tensor(labels[indices], dtype=torch.long)
    dataset = TensorDataset(subset_features, subset_labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    confidences = []
    correctness = []
    for batch_features, batch_labels in loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)
        logits = model(batch_features)
        probs = torch.softmax(logits, dim=1)
        conf = probs.gather(1, batch_labels.view(-1, 1)).squeeze(1)
        confidences.append(torch.logit(conf.clamp(eps, 1.0 - eps)).cpu().numpy())
        correctness.append((logits.argmax(dim=1) == batch_labels).float().cpu().numpy())

    return np.concatenate(confidences), np.concatenate(correctness)
