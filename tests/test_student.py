import pytest

pytest.importorskip("torch")

import torch

from models.student import distillation_loss


def test_distillation_loss_is_positive_and_differentiable():
    student_logits = torch.randn(4, 3, requires_grad=True)
    teacher_logits = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 2, 1])

    loss = distillation_loss(student_logits, labels, teacher_logits, temperature=2.0, alpha=0.5)
    loss.backward()

    assert loss.item() > 0
    assert student_logits.grad is not None
