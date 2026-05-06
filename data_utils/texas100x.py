"""Texas-100X loading utilities.

The loader centralizes the privacy invariant that `THCIC_ID` must never enter
model features. It accepts either the upstream CSV or the processed pickle
artifacts published by the Texas-100X project.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class Texas100XDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Dataset returning `(features, label, is_member)` tuples."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        member_indices: Iterable[int] | None = None,
        indices: Iterable[int] | None = None,
    ) -> None:
        features_array = np.asarray(features)
        labels_array = np.asarray(labels)
        if len(features_array) != len(labels_array):
            raise ValueError("Features and labels must have the same number of rows.")

        selected = np.asarray(list(indices), dtype=np.int64) if indices is not None else np.arange(len(labels_array))
        members = set(int(idx) for idx in member_indices) if member_indices is not None else set()
        self.source_indices = selected
        self.features = torch.as_tensor(features_array[selected], dtype=torch.float32)
        self.labels = torch.as_tensor(labels_array[selected], dtype=torch.long)
        self.is_member = torch.as_tensor([int(idx in members) for idx in selected], dtype=torch.bool)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index], self.is_member[index]


def load_texas100x_arrays(
    data_root: str | Path,
    label_column: str = "PRINC_SURG_PROC_CODE",
    excluded_columns: Iterable[str] = ("THCIC_ID",),
) -> tuple[np.ndarray, np.ndarray]:
    """Load Texas-100X features and labels while excluding sensitive columns."""

    root = Path(data_root)
    csv_path = root / "texas_100x.csv"
    features_path = root / "texas_100x_features.p"
    labels_path = root / "texas_100x_labels.p"

    if features_path.exists() and labels_path.exists():
        return _load_from_pickles(features_path, labels_path, root / "texas_100x_feature_desc.p", excluded_columns)
    if csv_path.exists():
        return _load_from_csv(csv_path, label_column, excluded_columns)

    raise FileNotFoundError(
        "Could not find Texas-100X processed files. Expected `texas_100x.csv` "
        "or `texas_100x_features.p` plus `texas_100x_labels.p` under "
        f"{root}."
    )


def _load_from_csv(
    csv_path: Path,
    label_column: str,
    excluded_columns: Iterable[str],
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(csv_path)
    if label_column not in frame.columns:
        raise ValueError(f"Missing label column `{label_column}` in {csv_path}.")

    excluded = set(excluded_columns)
    feature_columns = [col for col in frame.columns if col != label_column and col not in excluded]
    leaked = excluded.intersection(feature_columns)
    if leaked:
        raise ValueError(f"Sensitive columns leaked into features: {sorted(leaked)}")

    labels = pd.Categorical(frame[label_column]).codes.astype(np.int64)
    features = frame[feature_columns].to_numpy(dtype=np.float32)
    return features, labels


def load_feature_metadata(
    data_root: str | Path,
    excluded_columns: Iterable[str] = ("THCIC_ID",),
) -> dict[str, object]:
    """Load feature metadata and adjust indices after excluded columns are dropped."""

    root = Path(data_root)
    desc_path = root / "texas_100x_feature_desc.p"
    csv_path = root / "texas_100x.csv"
    if not desc_path.exists():
        return {}

    with desc_path.open("rb") as handle:
        desc = pickle.load(handle)
    if isinstance(desc, dict):
        attribute_idx = dict(desc.get("attribute_idx", {}))
        attribute_dict = dict(desc.get("attribute_dict", {}))
        max_attr_vals = np.asarray(desc.get("max_attr_vals", []), dtype=np.float32)
    elif isinstance(desc, (list, tuple)) and len(desc) >= 3:
        attribute_idx = dict(desc[0])
        attribute_dict = dict(desc[1])
        max_attr_vals = np.asarray(desc[2], dtype=np.float32)
    else:
        return {}

    columns = list(pd.read_csv(csv_path, nrows=0).columns) if csv_path.exists() else []
    label_column = "PRINC_SURG_PROC_CODE"
    feature_columns = [col for col in columns if col != label_column and col not in set(excluded_columns)]
    original_feature_columns = [col for col in columns if col != label_column]
    original_to_new = {
        original_idx: new_idx
        for new_idx, column in enumerate(feature_columns)
        for original_idx, original_column in enumerate(original_feature_columns)
        if column == original_column
    }

    categorical_indices: list[int] = []
    categorical_cardinalities: dict[int, int] = {}
    for column, original_idx in attribute_idx.items():
        if column in set(excluded_columns) or column == "TOTAL_CHARGES":
            continue
        if original_idx not in original_to_new:
            continue
        new_idx = original_to_new[original_idx]
        categorical_indices.append(new_idx)
        values = attribute_dict.get(original_idx, {})
        cardinality = max(len(values), int(max_attr_vals[original_idx]) + 1 if original_idx < len(max_attr_vals) else 1)
        categorical_cardinalities[new_idx] = max(cardinality, 1)

    if len(max_attr_vals) == len(original_feature_columns):
        keep_positions = [original_feature_columns.index(column) for column in feature_columns]
        max_attr_vals = max_attr_vals[keep_positions]

    return {
        "feature_columns": feature_columns,
        "categorical_indices": sorted(categorical_indices),
        "categorical_cardinalities": categorical_cardinalities,
        "max_attr_vals": max_attr_vals,
    }


def _load_from_pickles(
    features_path: Path,
    labels_path: Path,
    desc_path: Path,
    excluded_columns: Iterable[str],
) -> tuple[np.ndarray, np.ndarray]:
    with features_path.open("rb") as handle:
        raw_features = pickle.load(handle)
    with labels_path.open("rb") as handle:
        raw_labels = pickle.load(handle)

    features_frame = _features_to_frame(raw_features)
    for column in excluded_columns:
        if column in features_frame.columns:
            features_frame = features_frame.drop(columns=[column])
        elif column == "THCIC_ID" and features_frame.shape[1] == 11:
            # Upstream pickle features are documented as THCIC_ID followed by 10 model features.
            features_frame = features_frame.iloc[:, 1:]

    labels = np.asarray(raw_labels, dtype=np.int64).reshape(-1)
    features = features_frame.to_numpy(dtype=np.float32)

    if desc_path.exists():
        features = _normalize_if_needed(features, desc_path)
    return features, labels


def _features_to_frame(raw_features: object) -> pd.DataFrame:
    if isinstance(raw_features, pd.DataFrame):
        return raw_features.copy()
    return pd.DataFrame(np.asarray(raw_features))


def _normalize_if_needed(features: np.ndarray, desc_path: Path) -> np.ndarray:
    with desc_path.open("rb") as handle:
        desc = pickle.load(handle)
    if isinstance(desc, dict):
        max_vals = desc.get("max_attr_vals")
    elif isinstance(desc, (list, tuple)) and len(desc) >= 3:
        max_vals = desc[2]
    else:
        max_vals = None
    if max_vals is None:
        return features

    max_vals_array = np.asarray(max_vals, dtype=np.float32).reshape(-1)
    if len(max_vals_array) == features.shape[1] + 1:
        max_vals_array = max_vals_array[1:]
    if len(max_vals_array) != features.shape[1]:
        return features
    divisor = np.where(max_vals_array == 0, 1.0, max_vals_array)
    if np.nanmax(features) > 1.0:
        return features / divisor
    return features


def make_synthetic_texas100x(
    num_samples: int = 2048,
    num_features: int = 10,
    num_classes: int = 100,
    seed: int = 509,
) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic synthetic tabular data for tests and smoke runs."""

    rng = np.random.default_rng(seed)
    features = rng.normal(size=(num_samples, num_features)).astype(np.float32)
    weights = rng.normal(size=(num_features, num_classes)).astype(np.float32)
    logits = features @ weights + rng.normal(scale=0.5, size=(num_samples, num_classes))
    labels = np.argmax(logits, axis=1).astype(np.int64)
    return features, labels
