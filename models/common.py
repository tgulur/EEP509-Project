"""Shared PyTorch model and training utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader


class MLPClassifier(nn.Module):
    """Simple MLP baseline for Texas-100X tabular classification."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 100,
        hidden_dims: list[int] | tuple[int, ...] = (256, 128),
        dropout: float = 0.2,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class TabularMLPClassifier(nn.Module):
    """Tabular MLP with embeddings for encoded categorical columns."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        categorical_indices: list[int],
        categorical_cardinalities: dict[int, int],
        max_attr_vals: list[float] | np.ndarray,
        embedding_dim: int = 16,
        hidden_dims: list[int] | tuple[int, ...] = (512, 512, 256),
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.categorical_indices = sorted(categorical_indices)
        self.continuous_indices = [idx for idx in range(input_dim) if idx not in set(self.categorical_indices)]
        max_vals = torch.as_tensor(max_attr_vals, dtype=torch.float32)
        self.register_buffer("max_attr_vals", max_vals if len(max_vals) == input_dim else torch.ones(input_dim))
        self.embeddings = nn.ModuleDict(
            {
                str(idx): nn.Embedding(max(int(categorical_cardinalities[idx]), 1), embedding_dim)
                for idx in self.categorical_indices
            }
        )
        projected_dim = len(self.continuous_indices) + len(self.categorical_indices) * embedding_dim
        self.classifier = MLPClassifier(
            input_dim=projected_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_batchnorm=True,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        if self.continuous_indices:
            parts.append(features[:, self.continuous_indices])
        for idx in self.categorical_indices:
            max_val = self.max_attr_vals[idx].clamp_min(1.0)
            category = torch.round(features[:, idx] * max_val).long()
            category = category.clamp(min=0, max=self.embeddings[str(idx)].num_embeddings - 1)
            parts.append(self.embeddings[str(idx)](category))
        return self.classifier(torch.cat(parts, dim=1))


@dataclass
class TrainResult:
    train_loss: float
    train_acc: float
    eval_loss: float
    eval_acc: float


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    criterion: nn.Module | None = None,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total = 0
    criterion = criterion or nn.CrossEntropyLoss()

    for features, labels, _is_member in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = loss_fn(logits, labels, features) if loss_fn is not None else criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.numel()
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += batch_size

    return total_loss / max(total, 1), total_correct / max(total, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total = 0
    for features, labels, _is_member in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        total_loss += float(criterion(logits, labels).item())
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.numel()
    return total_loss / max(total, 1), total_correct / max(total, 1)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    metadata: dict[str, object],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, output)


def load_checkpoint(path: str | Path, model: nn.Module, device: torch.device) -> dict[str, object]:
    checkpoint = torch.load(Path(path), map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    return dict(checkpoint.get("metadata", {}))


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_out: list[np.ndarray] = []
    labels_out: list[np.ndarray] = []
    for features, labels, _is_member in loader:
        logits = model(features.to(device)).detach().cpu().numpy()
        logits_out.append(logits)
        labels_out.append(labels.numpy())
    return np.concatenate(logits_out), np.concatenate(labels_out)
