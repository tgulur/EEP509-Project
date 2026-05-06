"""Mitigated student model variants."""

from __future__ import annotations

from torch import nn

from mitigations.bottleneck import BottleneckProjection
from mitigations.nonorm import NoNorm
from models.common import MLPClassifier


class BottleneckStudent(nn.Module):
    """Student with a low-rank projection before the classifier."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int],
        dropout: float,
        rank: int,
    ) -> None:
        super().__init__()
        self.bottleneck = BottleneckProjection(input_dim, rank)
        self.classifier = MLPClassifier(input_dim, num_classes, hidden_dims, dropout)

    def forward(self, features):
        return self.classifier(self.bottleneck(features))


def build_nonorm_student(
    input_dim: int,
    num_classes: int,
    hidden_dims: list[int],
    dropout: float,
) -> MLPClassifier:
    """Build a student and replace normalization with identity layers."""

    model = MLPClassifier(input_dim, num_classes, hidden_dims, dropout, use_batchnorm=False)
    for name, module in list(model.named_children()):
        if isinstance(module, nn.BatchNorm1d):
            setattr(model, name, NoNorm())
    return model
