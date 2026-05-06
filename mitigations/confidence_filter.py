"""High-confidence filtering before distillation."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def select_low_confidence_indices(
    teacher: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    top_fraction: float,
) -> np.ndarray:
    """Return source indices after removing the teacher's highest-confidence samples."""

    if not 0.0 <= top_fraction < 1.0:
        raise ValueError("top_fraction must be in [0, 1).")

    teacher.eval()
    confidences: list[np.ndarray] = []
    source_indices: list[np.ndarray] = []
    dataset = loader.dataset
    if not hasattr(dataset, "source_indices"):
        raise ValueError("Loader dataset must expose `source_indices` for filtering.")

    start = 0
    all_source_indices = np.asarray(dataset.source_indices)
    for features, _labels, _is_member in loader:
        batch_size = features.shape[0]
        probs = torch.softmax(teacher(features.to(device)), dim=1)
        confidences.append(probs.max(dim=1).values.cpu().numpy())
        source_indices.append(all_source_indices[start : start + batch_size])
        start += batch_size

    conf = np.concatenate(confidences)
    idx = np.concatenate(source_indices)
    keep_count = int(round(len(idx) * (1.0 - top_fraction)))
    order = np.argsort(conf)
    return np.sort(idx[order[:keep_count]])
