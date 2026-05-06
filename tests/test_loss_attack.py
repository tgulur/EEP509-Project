import numpy as np
import pytest

pytest.importorskip("torch")

import torch
from torch import nn
from torch.utils.data import DataLoader

from attacks.loss_based import LossBasedAttack
from data_utils.texas100x import Texas100XDataset


class MemorizedTinyModel(nn.Module):
    def forward(self, features):
        return torch.stack([features[:, 0], -features[:, 0]], dim=1)


def test_loss_based_attack_scores_lower_loss_as_membership():
    features = np.array([[5.0], [4.0], [-4.0], [-5.0]], dtype=np.float32)
    labels = np.array([0, 0, 0, 0], dtype=np.int64)
    member = DataLoader(Texas100XDataset(features, labels, member_indices=[0, 1], indices=[0, 1]), batch_size=2)
    non_member = DataLoader(Texas100XDataset(features, labels, member_indices=[0, 1], indices=[2, 3]), batch_size=2)

    attack = LossBasedAttack().fit(MemorizedTinyModel(), member, non_member)
    member_scores = attack.score(MemorizedTinyModel(), member)
    non_member_scores = attack.score(MemorizedTinyModel(), non_member)

    assert member_scores.mean() > non_member_scores.mean()
