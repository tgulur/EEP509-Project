"""Knowledge distillation - train student on teacher's soft labels."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.common import MLPClassifier, TrainResult, evaluate, save_checkpoint
from utils import config_hash


def build_student(input_dim: int, num_classes: int, hidden_dims: list[int], dropout: float) -> MLPClassifier:
    # student is typically smaller than teacher
    return MLPClassifier(input_dim=input_dim, num_classes=num_classes, hidden_dims=hidden_dims, dropout=dropout)


def distillation_loss(
    student_logits: torch.Tensor,
    hard_labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    """KD loss: alpha * CE(hard) + (1-alpha) * KL(soft). T^2 scaling per Hinton et al."""
    hard_loss = F.cross_entropy(student_logits, hard_labels)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction="batchmean") * (temperature**2)
    return alpha * hard_loss + (1.0 - alpha) * soft_loss


@torch.no_grad()
def cache_teacher_soft_labels(
    teacher: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_path: str | Path,
    temperature: float,
) -> torch.Tensor:
    """Precompute teacher soft labels for student training."""

    teacher.eval()
    batches: list[torch.Tensor] = []
    for features, _labels, _is_member in loader:
        logits = teacher(features.to(device))
        batches.append(F.softmax(logits / temperature, dim=1).cpu())
    soft_labels = torch.cat(batches, dim=0)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"temperature": temperature, "soft_labels": soft_labels}, output)
    return soft_labels


def train_student(
    model: torch.nn.Module,
    teacher: torch.nn.Module,
    train_loader: DataLoader,
    eval_loader: DataLoader,
    device: torch.device,
    config: dict,
    checkpoint_path: str | Path,
) -> TrainResult:
    """Train a student directly from teacher logits."""

    model.to(device)
    teacher.to(device)
    teacher.eval()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["model"]["lr"]),
        weight_decay=float(config["model"]["weight_decay"]),
    )
    scheduler = None
    scheduler_config = config.get("student", {}).get("scheduler", {})
    if bool(scheduler_config.get("enabled", False)):
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_config.get("step_size", 20)),
            gamma=float(scheduler_config.get("gamma", 0.5)),
        )
    epochs = int(config["student"]["epochs"])
    temperature = float(config["student"]["temperature"])
    alpha = float(config["student"].get("alpha", 0.5))
    train_loss = train_acc = eval_loss = eval_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0
        for features, labels, _is_member in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits = teacher(features)
            student_logits = model(features)
            loss = distillation_loss(student_logits, labels, teacher_logits, temperature, alpha)
            loss.backward()
            optimizer.step()
            batch_size = labels.numel()
            total_loss += float(loss.item()) * batch_size
            total_correct += int((student_logits.argmax(dim=1) == labels).sum().item())
            total += batch_size

        train_loss = total_loss / max(total, 1)
        train_acc = total_correct / max(total, 1)
        eval_loss, eval_acc = evaluate(model, eval_loader, device)
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"student epoch={epoch} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"eval_loss={eval_loss:.4f} eval_acc={eval_acc:.4f} gap={train_acc - eval_acc:.4f} lr={lr:.6g}"
        )
        if scheduler is not None:
            scheduler.step()

    metadata = {
        "epoch": epochs,
        "seed": config["project"]["seed"],
        "train_acc": train_acc,
        "test_acc": eval_acc,
        "config_hash": config_hash(config),
        "temperature": temperature,
        "alpha": alpha,
    }
    save_checkpoint(checkpoint_path, model, metadata)
    return TrainResult(train_loss, train_acc, eval_loss, eval_acc)
