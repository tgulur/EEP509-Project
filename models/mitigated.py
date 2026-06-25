"""Mitigated student model variants.

Each mitigation has to wrap the same input pipeline as the teacher so the
privacy comparison isn't confounded with an embedding-vs-no-embedding gap.
For tabular_mlp teachers that means going through the categorical embedding
step before whatever the mitigation does to the representation.
"""

from __future__ import annotations

import torch
from torch import nn

from mitigations.bottleneck import BottleneckProjection
from mitigations.nonorm import NoNorm
from models.common import MLPClassifier, TabularMLPClassifier


class _TabularEmbedder(nn.Module):
    """Embedding + concat step from TabularMLPClassifier, reusable on its own."""

    def __init__(
        self,
        input_dim: int,
        categorical_indices: list[int],
        categorical_cardinalities: dict[int, int],
        max_attr_vals,
        embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.categorical_indices = sorted(categorical_indices)
        self.continuous_indices = [
            idx for idx in range(input_dim) if idx not in set(self.categorical_indices)
        ]
        max_vals = torch.as_tensor(max_attr_vals, dtype=torch.float32)
        self.register_buffer(
            "max_attr_vals",
            max_vals if len(max_vals) == input_dim else torch.ones(input_dim),
        )
        self.embeddings = nn.ModuleDict(
            {
                str(idx): nn.Embedding(max(int(categorical_cardinalities[idx]), 1), embedding_dim)
                for idx in self.categorical_indices
            }
        )
        self.output_dim = (
            len(self.continuous_indices)
            + len(self.categorical_indices) * embedding_dim
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
        return torch.cat(parts, dim=1)


class BottleneckStudent(nn.Module):
    """Student with a low-rank bottleneck after the (optional) embedding step."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int],
        dropout: float,
        rank: int,
        teacher_type: str = "mlp",
        metadata: dict | None = None,
        embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        if teacher_type == "tabular_mlp" and metadata:
            self.embedder = _TabularEmbedder(
                input_dim=input_dim,
                categorical_indices=list(metadata.get("categorical_indices", [])),
                categorical_cardinalities=dict(metadata.get("categorical_cardinalities", {})),
                max_attr_vals=metadata.get("max_attr_vals", []),
                embedding_dim=embedding_dim,
            )
            bottleneck_in = self.embedder.output_dim
        else:
            self.embedder = None
            bottleneck_in = input_dim
        self.bottleneck = BottleneckProjection(bottleneck_in, rank)
        self.classifier = MLPClassifier(bottleneck_in, num_classes, hidden_dims, dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.embedder is not None:
            features = self.embedder(features)
        return self.classifier(self.bottleneck(features))


def build_nonorm_student(
    input_dim: int,
    num_classes: int,
    hidden_dims: list[int],
    dropout: float,
    teacher_type: str = "mlp",
    metadata: dict | None = None,
    embedding_dim: int = 16,
) -> nn.Module:
    """Tabular student with BatchNorm layers replaced by identity passthroughs."""
    if teacher_type == "tabular_mlp" and metadata:
        model = TabularMLPClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            categorical_indices=list(metadata.get("categorical_indices", [])),
            categorical_cardinalities=dict(metadata.get("categorical_cardinalities", {})),
            max_attr_vals=metadata.get("max_attr_vals", []),
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        # the inner MLPClassifier holds the BatchNorm layers
        inner = model.classifier.net
        for i, layer in enumerate(list(inner)):
            if isinstance(layer, nn.BatchNorm1d):
                inner[i] = NoNorm()
        return model

    model = MLPClassifier(input_dim, num_classes, hidden_dims, dropout, use_batchnorm=False)
    for i, layer in enumerate(list(model.net)):
        if isinstance(layer, nn.BatchNorm1d):
            model.net[i] = NoNorm()
    return model
