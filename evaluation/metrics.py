"""Metrics for utility and membership inference evaluation."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


RESULT_COLUMNS = [
    "model_type",
    "attack_type",
    "mitigation",
    "seed",
    "auc",
    "tpr_at_01fpr",
    "tpr_at_05fpr",
    "tpr_at_1fpr",
    "train_acc",
    "test_acc",
]


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    """Return the best TPR observed at or below a target FPR."""

    fpr, tpr, _thresholds = roc_curve(y_true, scores)
    valid = tpr[fpr <= target_fpr]
    if len(valid) == 0:
        return 0.0
    return float(valid.max())


def attack_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    """Compute AUC and low-FPR TPR values for an attack."""

    if len(np.unique(y_true)) < 2:
        raise ValueError("Attack metrics require both member and non-member labels.")
    return {
        "auc": float(roc_auc_score(y_true, scores)),
        "tpr_at_01fpr": tpr_at_fpr(y_true, scores, 0.001),
        "tpr_at_05fpr": tpr_at_fpr(y_true, scores, 0.005),
        "tpr_at_1fpr": tpr_at_fpr(y_true, scores, 0.01),
    }


def append_result(path: str | Path, row: dict[str, object]) -> None:
    """Append a result row using the project-standard columns."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.exists()
    with output.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in RESULT_COLUMNS})
