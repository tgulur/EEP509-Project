"""Teacher model construction and training.

The teacher intentionally overfits a bit (train >> test acc) since that's
what makes MIA interesting - if there's no memorization, there's nothing to leak.
"""

from __future__ import annotations

from pathlib import Path
import csv

import torch
from torch import nn
from torch.utils.data import DataLoader

from models.common import MLPClassifier, TabularMLPClassifier, TrainResult, evaluate, save_checkpoint, train_epoch
from utils import config_hash


def build_teacher(
    input_dim: int,
    num_classes: int,
    hidden_dims: list[int],
    dropout: float,
    teacher_type: str = "mlp",
    metadata: dict[str, object] | None = None,
    embedding_dim: int = 16,
):
    """Build teacher model. Use tabular_mlp for categorical embeddings."""
    if teacher_type == "tabular_mlp" and metadata:
        return TabularMLPClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            categorical_indices=list(metadata.get("categorical_indices", [])),
            categorical_cardinalities=dict(metadata.get("categorical_cardinalities", {})),
            max_attr_vals=metadata.get("max_attr_vals", []),
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    return MLPClassifier(input_dim=input_dim, num_classes=num_classes, hidden_dims=hidden_dims, dropout=dropout)


def train_teacher(
    model: torch.nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    config: dict,
    checkpoint_path: str | Path,
) -> TrainResult:
    # saves best checkpoint by eval acc, logs history to CSV for debugging
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["model"]["lr"]),
        weight_decay=float(config["model"]["weight_decay"]),
    )
    scheduler = None
    scheduler_config = config["model"].get("scheduler", {})
    if bool(scheduler_config.get("enabled", False)):
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_config.get("step_size", 20)),
            gamma=float(scheduler_config.get("gamma", 0.5)),
        )
    epochs = int(config["model"]["epochs"])
    train_loss = train_acc = eval_loss = eval_acc = 0.0
    best_eval_acc = -1.0
    best_metadata: dict[str, object] = {}
    history_path = Path(config["paths"]["experiment_root"]) / "teacher_history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    criterion = _build_criterion(train_loader, device, config)

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "train_acc", "eval_loss", "eval_acc", "gap", "lr"])
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, device, criterion=criterion)
            eval_loss, eval_acc = evaluate(model, eval_loader, device)
            gap = train_acc - eval_acc
            lr = optimizer.param_groups[0]["lr"]
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "eval_loss": eval_loss,
                    "eval_acc": eval_acc,
                    "gap": gap,
                    "lr": lr,
                }
            )
            print(
                f"teacher epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"eval_loss={eval_loss:.4f} eval_acc={eval_acc:.4f} gap={gap:.4f} lr={lr:.6g}"
            )
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                best_metadata = {
                    "epoch": epoch,
                    "seed": config["project"]["seed"],
                    "train_acc": train_acc,
                    "test_acc": eval_acc,
                    "best_eval_acc": eval_acc,
                    "config_hash": config_hash(config),
                    "teacher_type": config["model"].get("teacher_type", "mlp"),
                }
                save_checkpoint(checkpoint_path, model, best_metadata)
            if scheduler is not None:
                scheduler.step()

    metadata = best_metadata or {
        "epoch": epochs,
        "seed": config["project"]["seed"],
        "train_acc": train_acc,
        "test_acc": eval_acc,
        "best_eval_acc": eval_acc,
        "config_hash": config_hash(config),
        "teacher_type": config["model"].get("teacher_type", "mlp"),
    }
    if not Path(checkpoint_path).exists():
        save_checkpoint(checkpoint_path, model, metadata)
    return TrainResult(train_loss, train_acc, eval_loss, eval_acc)


def _build_criterion(train_loader: DataLoader, device: torch.device, config: dict) -> nn.Module:
    if not bool(config["model"].get("class_weighted_loss", False)):
        return nn.CrossEntropyLoss()
    num_classes = int(config["model"]["num_classes"])
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for _features, labels, _is_member in train_loader:
        counts += torch.bincount(labels, minlength=num_classes).float()
    weights = counts.sum() / counts.clamp_min(1.0)
    weights = weights / weights.mean()
    return nn.CrossEntropyLoss(weight=weights.to(device))
