"""Stage B: sweep (T, alpha) at the new same-as-teacher student backbone.

Runs four (T, alpha) combinations to find a student that lands at >= 45% test
accuracy. Writes one checkpoint per config under student_T{T}_a{alpha}.pt and
prints a sweep summary at the end.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main import _make_teacher, get_feature_metadata, make_loaders, prepare_arrays
from models.student import build_student, train_student
from utils import load_config, resolve_device


SWEEPS = [
    {"temperature": 2.0, "alpha": 0.5},
    {"temperature": 1.0, "alpha": 0.5},
    {"temperature": 2.0, "alpha": 0.3},
    {"temperature": 4.0, "alpha": 1.0},
]


def main() -> None:
    config = load_config("configs/config.yaml")
    device = resolve_device(str(config["project"]["device"]))

    features, labels, splits = prepare_arrays(config, synthetic=False)
    input_dim = features.shape[1]
    config["model"]["input_dim"] = input_dim

    loaders = make_loaders(features, labels, splits, int(config["data"]["batch_size"]))
    feature_metadata = get_feature_metadata(config)

    teacher = _make_teacher(input_dim, config, feature_metadata)
    teacher.load_state_dict(torch.load("experiments/checkpoints/teacher.pt", map_location=device)["state_dict"])
    teacher.to(device).eval()

    checkpoint_dir = Path("experiments/checkpoints")
    results = []
    for sweep in SWEEPS:
        T = sweep["temperature"]
        alpha = sweep["alpha"]
        tag = f"T{int(T)}_a{str(alpha).replace('.', '')}"
        print(f"\n=== {tag} (T={T}, alpha={alpha}) ===")

        config["student"]["temperature"] = float(T)
        config["student"]["alpha"] = float(alpha)

        student = build_student(
            input_dim,
            int(config["model"]["num_classes"]),
            list(config["student"]["hidden_dims"]),
            float(config["model"]["dropout"]),
            teacher_type=str(config["model"].get("teacher_type", "mlp")),
            metadata=feature_metadata,
            embedding_dim=int(config["model"].get("embedding_dim", 16)),
        )
        ckpt = checkpoint_dir / f"student_{tag}.pt"
        result = train_student(student, teacher, loaders["train"], loaders["test"], device, config, ckpt)
        results.append({
            "tag": tag,
            "T": T,
            "alpha": alpha,
            "train_acc": result.train_acc,
            "test_acc": result.eval_acc,
            "gap": result.train_acc - result.eval_acc,
            "checkpoint": str(ckpt),
        })

    print("\n=== Stage B sweep summary ===")
    print(f"{'tag':<14} {'T':>4} {'alpha':>6} {'train':>8} {'test':>8} {'gap':>7}")
    for r in results:
        flag = " *" if r["test_acc"] >= 0.45 else ""
        print(f"{r['tag']:<14} {r['T']:>4} {r['alpha']:>6} "
              f"{r['train_acc']:>8.4f} {r['test_acc']:>8.4f} {r['gap']:>7.4f}{flag}")

    out = checkpoint_dir / "stage_b_sweep_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
