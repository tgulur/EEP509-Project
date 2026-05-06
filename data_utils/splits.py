"""Deterministic split creation and persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SplitIndices:
    """Container for deterministic train/validation/test splits."""

    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def save(self, path: str | Path) -> None:
        """Persist split indices to disk."""

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output, train=self.train, val=self.val, test=self.test)

    @classmethod
    def load(cls, path: str | Path) -> "SplitIndices":
        """Load split indices from an `.npz` file."""

        data = np.load(Path(path))
        return cls(train=data["train"], val=data["val"], test=data["test"])


def make_split_indices(
    num_samples: int,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
) -> SplitIndices:
    """Create non-overlapping deterministic splits."""

    requested = train_size + val_size + test_size
    if requested > num_samples:
        raise ValueError(f"Requested {requested} samples but only {num_samples} are available.")

    rng = np.random.default_rng(seed)
    permuted = rng.permutation(num_samples)
    train_end = train_size
    val_end = train_size + val_size
    return SplitIndices(
        train=np.sort(permuted[:train_end]),
        val=np.sort(permuted[train_end:val_end]),
        test=np.sort(permuted[val_end : val_end + test_size]),
    )
