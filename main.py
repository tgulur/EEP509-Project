"""Main entry point for the KD-MIA experiments.

Run `python main.py --help` for usage. Most common:
  python main.py all              # full pipeline
  python main.py list-runs        # see previous experiments
  python main.py analyze --subgroups --resume-run <id>
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_utils.splits import SplitIndices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Texas-100X knowledge-distillation MIA pipeline")
    parser.add_argument(
        "stage",
        choices=[
            "prepare-data",
            "train-teacher",
            "train-student",
            "sweep-student",
            "train-mitigated",
            "run-attacks",
            "run-lira-full",
            "plot",
            "smoke",
            "all",
            "analyze",
            "list-runs",
        ],
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data for smoke testing.")
    parser.add_argument("--subgroups", action="store_true", help="Run subgroup vulnerability analysis (for analyze stage).")
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run identifier for experiment folder. Defaults to datetime stamp (YYYYMMDD_HHMMSS).",
    )
    parser.add_argument(
        "--resume-run",
        default=None,
        help="Resume a previous run by specifying its run-id (folder name under experiments/runs/).",
    )
    return parser.parse_args()


def generate_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_run_paths(config: dict, run_id: str) -> None:
    """Update config paths to use a run-specific experiment folder."""
    base_experiment_root = Path(config["paths"]["experiment_root"])
    run_dir = base_experiment_root / "runs" / run_id

    config["paths"]["experiment_root"] = str(run_dir)
    config["paths"]["checkpoint_dir"] = str(run_dir / "checkpoints")
    config["paths"]["soft_label_dir"] = str(run_dir / "soft_labels")
    config["paths"]["figure_dir"] = str(run_dir / "figures")
    config["paths"]["results_csv"] = str(run_dir / "results.csv")

    if "lira_full" in config:
        config["lira_full"]["cache_dir"] = str(run_dir / "lira_cache")

    config["run_id"] = run_id
    print(f"Experiment run: {run_id}")
    print(f"Output folder: {run_dir}")


def save_run_metadata(config: dict, args: argparse.Namespace) -> None:
    """Save config and run metadata to the run folder for reproducibility."""
    import yaml

    run_dir = Path(config["paths"]["experiment_root"])
    run_dir.mkdir(parents=True, exist_ok=True)

    config_path = run_dir / "config.yaml"
    with config_path.open("w") as f:
        yaml.dump(config, f, default_flow_style=False)

    metadata = {
        "run_id": config.get("run_id", "unknown"),
        "started_at": datetime.now().isoformat(),
        "stage": args.stage,
        "synthetic": args.synthetic,
        "subgroups": args.subgroups,
    }
    metadata_path = run_dir / "run_metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)


def list_runs(config: dict) -> None:
    runs_dir = Path(config["paths"]["experiment_root"]) / "runs"
    if not runs_dir.exists():
        print("No experiment runs found.")
        return

    runs = sorted(runs_dir.iterdir(), reverse=True)
    if not runs:
        print("No experiment runs found.")
        return

    print(f"{'Run ID':<20} {'Started':<22} {'Stage':<15} {'Results':<10}")
    print("-" * 70)

    for run_path in runs:
        if not run_path.is_dir():
            continue

        run_id = run_path.name
        metadata_path = run_path / "run_metadata.json"
        results_path = run_path / "results.csv"

        started = "unknown"
        stage = "unknown"
        has_results = "Yes" if results_path.exists() else "No"

        if metadata_path.exists():
            try:
                with metadata_path.open() as f:
                    meta = json.load(f)
                started = meta.get("started_at", "unknown")[:19].replace("T", " ")
                stage = meta.get("stage", "unknown")
            except Exception:
                pass

        print(f"{run_id:<20} {started:<22} {stage:<15} {has_results:<10}")

    print(f"\nTotal runs: {len(runs)}")
    print(f"Resume a run with: python main.py <stage> --resume-run <run-id>")


def _checkpoint_path(config: dict, filename: str) -> str:
    base = "experiments/checkpoints/_synthetic" if config.get("_synthetic") else "experiments/checkpoints"
    Path(base).mkdir(parents=True, exist_ok=True)
    return f"{base}/{filename}"


def _make_teacher(input_dim: int, config: dict, feature_metadata: dict[str, object]):
    # all five call sites used the exact same kwargs - pull it out
    from models.teacher import build_teacher

    return build_teacher(
        input_dim,
        int(config["model"]["num_classes"]),
        list(config["model"]["hidden_dims"]),
        float(config["model"]["dropout"]),
        teacher_type=str(config["model"].get("teacher_type", "mlp")),
        metadata=feature_metadata,
        embedding_dim=int(config["model"].get("embedding_dim", 16)),
    )


def main() -> None:
    args = parse_args()
    from evaluation.plots import make_all_plots
    from models.student import build_student, train_student
    from models.teacher import train_teacher
    from utils import load_config, resolve_device, set_seed

    config = load_config(args.config)

    if args.stage == "list-runs":
        list_runs(config)
        return

    if args.stage == "smoke":
        _shrink_for_smoke(config)
    # synthetic + smoke share checkpoint paths with real runs by default,
    # which means a quick smoke can stomp on a real teacher. route them to a
    # sandbox dir instead.
    config["_synthetic"] = bool(args.synthetic or args.stage == "smoke")

    if args.resume_run:
        run_id = args.resume_run
        print(f"Resuming run: {run_id}")
    elif args.run_id:
        run_id = args.run_id
    else:
        run_id = generate_run_id()

    setup_run_paths(config, run_id)

    set_seed(int(config["project"]["seed"]))
    device = resolve_device(str(config["project"]["device"]))
    ensure_output_dirs(config)
    save_run_metadata(config, args)

    if args.stage == "plot":
        make_all_plots(config["paths"]["results_csv"], config["paths"]["figure_dir"])
        return
    if args.stage == "analyze":
        from analysis.summary import run_analysis

        run_analysis(config)
        if args.subgroups:
            run_subgroup_analysis_stage(config)
        return

    features, labels, splits = prepare_arrays(config, synthetic=args.synthetic or args.stage == "smoke")
    if args.stage == "prepare-data":
        print(f"prepared {len(labels)} rows with {features.shape[1]} model features")
        return

    loaders = make_loaders(features, labels, splits, int(config["data"]["batch_size"]))
    input_dim = features.shape[1]
    teacher: torch.nn.Module | None = None
    feature_metadata = get_feature_metadata(config)

    if args.stage in {"train-teacher", "all", "smoke"}:
        teacher = _make_teacher(input_dim, config, feature_metadata)
        train_teacher(teacher, loaders["train"], loaders["test"], device, config, _checkpoint_path(config, "teacher.pt"))

    if args.stage in {"train-student", "all", "smoke"}:
        if teacher is None:
            teacher = _make_teacher(input_dim, config, feature_metadata)
            _load_if_exists(teacher, _checkpoint_path(config, "teacher.pt"), device)
        student = build_student(input_dim, int(config["model"]["num_classes"]), list(config["student"]["hidden_dims"]), float(config["model"]["dropout"]))
        train_student(student, teacher, loaders["train"], loaders["test"], device, config, _checkpoint_path(config, "student.pt"))

    if args.stage == "sweep-student":
        if teacher is None:
            teacher = _make_teacher(input_dim, config, feature_metadata)
            _load_if_exists(teacher, _checkpoint_path(config, "teacher.pt"), device)
        run_student_alpha_sweep(input_dim, teacher, loaders, device, config)

    if args.stage in {"train-mitigated", "all", "smoke"}:
        if teacher is None:
            teacher = _make_teacher(input_dim, config, feature_metadata)
            _load_if_exists(teacher, _checkpoint_path(config, "teacher.pt"), device)
        mitigated_models = _build_mitigated_models(input_dim, config)
        for name, mitigated_model in mitigated_models.items():
            print(f"\n--- Training mitigated student: {name} ---")
            train_student(
                mitigated_model,
                teacher,
                loaders["train"],
                loaders["test"],
                device,
                config,
                _checkpoint_path(config, f"student_{name}.pt"),
            )
        print("\n--- Training mitigated student: confidence_filter ---")
        _train_confidence_filtered_student(
            input_dim, teacher, features, labels, splits, loaders, device, config
        )

    if args.stage in {"run-attacks", "all", "smoke"}:
        run_attack_suite(input_dim, loaders, device, config, feature_metadata)

    if args.stage == "all":
        from analysis.summary import run_analysis
        from evaluation.plots import make_all_plots

        print("\n--- Running analysis and generating plots ---")
        run_analysis(config)
        make_all_plots(config["paths"]["results_csv"], config["paths"]["figure_dir"])
        run_subgroup_analysis_stage(config)
        print(f"\nExperiment complete! Results in: {config['paths']['experiment_root']}")

    if args.stage == "run-lira-full":
        run_full_lira_stage(features, labels, splits, input_dim, device, config, feature_metadata)


def ensure_output_dirs(config: dict) -> None:
    from utils import ensure_dir

    for key in ["processed_root", "experiment_root", "checkpoint_dir", "soft_label_dir", "figure_dir"]:
        ensure_dir(config["paths"][key])


def _shrink_for_smoke(config: dict) -> None:
    # make smoke tests fast - just sanity check that nothing crashes
    config["synthetic"]["num_samples"] = min(int(config["synthetic"]["num_samples"]), 512)
    config["data"]["train_size"] = 128
    config["data"]["val_size"] = 64
    config["data"]["test_size"] = 128
    config["data"]["batch_size"] = 64
    config["model"]["epochs"] = 1
    config["student"]["epochs"] = 1


def _load_if_exists(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> dict[str, object]:
    from models.common import load_checkpoint

    path = Path(checkpoint_path)
    if path.exists():
        return load_checkpoint(path, model, device)
    return {}


def prepare_arrays(config: dict, synthetic: bool = False) -> tuple[np.ndarray, np.ndarray, SplitIndices]:
    from data_utils.splits import make_split_indices
    from data_utils.texas100x import load_texas100x_arrays, make_synthetic_texas100x

    if synthetic:
        features, labels = make_synthetic_texas100x(
            num_samples=int(config["synthetic"]["num_samples"]),
            num_features=int(config["synthetic"]["num_features"]),
            num_classes=int(config["synthetic"]["num_classes"]),
            seed=int(config["project"]["seed"]),
        )
    else:
        ensure_processed_texas100x(config)
        features, labels = load_texas100x_arrays(
            config["paths"]["data_root"],
            label_column=config["data"]["label_column"],
            excluded_columns=config["data"]["excluded_columns"],
        )
    print(
        f"data rows={len(labels)} input_dim={features.shape[1]} classes={len(np.unique(labels))} "
        f"feature_min={float(np.nanmin(features)):.4f} feature_max={float(np.nanmax(features)):.4f}"
    )

    train_size = min(int(config["data"]["train_size"]), len(labels) // 2)
    val_size = min(int(config["data"]["val_size"]), max(len(labels) - train_size, 0) // 2)
    test_size = min(int(config["data"]["test_size"]), len(labels) - train_size - val_size)
    splits = make_split_indices(len(labels), train_size, val_size, test_size, int(config["project"]["seed"]))
    splits.save(Path(config["paths"]["processed_root"]) / "split_indices.npz")
    return features, labels, splits


def get_feature_metadata(config: dict) -> dict[str, object]:
    from data_utils.texas100x import load_feature_metadata

    return load_feature_metadata(
        config["paths"]["data_root"],
        excluded_columns=config["data"]["excluded_columns"],
    )


def ensure_processed_texas100x(config: dict) -> Path:
    """Build the processed CSV from raw PUDF tabs if it is not present."""

    from data_utils.preprocess_pudf import build_texas100x_csv

    data_root = Path(config["paths"]["data_root"])
    csv_path = data_root / str(config["data"]["csv_file"])
    features_path = data_root / str(config["data"]["features_pickle"])
    labels_path = data_root / str(config["data"]["labels_pickle"])
    if csv_path.exists() or (features_path.exists() and labels_path.exists()):
        return csv_path

    print("processed Texas-100X files not found; building texas_100x.csv from raw PUDF tabs")
    return build_texas100x_csv(
        data_root=data_root,
        raw_dir=str(config["data"].get("raw_dir", "raw")),
        raw_glob=str(config["data"].get("raw_glob", "PUDF_base*q*_tab.txt")),
        output_file=str(config["data"]["csv_file"]),
        label_column=str(config["data"]["label_column"]),
        feature_columns=list(config["data"]["raw_feature_columns"]),
        top_classes=int(config["data"]["top_procedure_classes"]),
    )


def make_loaders(features: np.ndarray, labels: np.ndarray, splits: SplitIndices, batch_size: int) -> dict[str, DataLoader]:
    from data_utils.texas100x import Texas100XDataset

    train_dataset = Texas100XDataset(features, labels, member_indices=splits.train, indices=splits.train)
    val_dataset = Texas100XDataset(features, labels, member_indices=splits.train, indices=splits.val)
    test_dataset = Texas100XDataset(features, labels, member_indices=splits.train, indices=splits.test)
    return {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_dataset, batch_size=batch_size),
        "test": DataLoader(test_dataset, batch_size=batch_size),
    }


def _train_confidence_filtered_student(
    input_dim: int,
    teacher: torch.nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    splits: SplitIndices,
    loaders: dict[str, DataLoader],
    device: torch.device,
    config: dict,
) -> None:
    # data-side mitigation: drop the top-N% teacher-confidence samples before distilling
    from data_utils.texas100x import Texas100XDataset
    from mitigations.confidence_filter import select_low_confidence_indices
    from models.student import build_student, train_student

    top_fraction = float(config["mitigations"]["confidence_filter"]["top_fraction"])
    kept = select_low_confidence_indices(teacher, loaders["train"], device, top_fraction)
    print(f"keeping {len(kept)} / {len(splits.train)} samples after dropping top {top_fraction:.0%}")

    batch_size = int(config["data"]["batch_size"])
    train_ds = Texas100XDataset(features, labels, member_indices=splits.train, indices=kept)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    student = build_student(
        input_dim,
        int(config["model"]["num_classes"]),
        list(config["student"]["hidden_dims"]),
        float(config["model"]["dropout"]),
    )
    train_student(student, teacher, train_loader, loaders["test"], device, config,
                  _checkpoint_path(config, "student_confidence_filter.pt"))


def _build_mitigated_models(input_dim: int, config: dict) -> dict[str, torch.nn.Module]:
    from models.mitigated import BottleneckStudent, build_nonorm_student

    return {
        "bottleneck": BottleneckStudent(
            input_dim,
            int(config["model"]["num_classes"]),
            list(config["student"]["hidden_dims"]),
            float(config["model"]["dropout"]),
            int(config["mitigations"]["bottleneck"]["rank"]),
        ),
        "nonorm": build_nonorm_student(
            input_dim,
            int(config["model"]["num_classes"]),
            list(config["student"]["hidden_dims"]),
            float(config["model"]["dropout"]),
        ),
    }


def compute_sample_metadata(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """Compute per-sample metadata: indices, labels, loss, confidence."""
    import torch.nn.functional as F

    model.to(device)
    model.eval()

    sample_indices: list[int] = []
    labels_list: list[int] = []
    losses: list[float] = []
    confidences: list[float] = []
    max_confidences: list[float] = []

    dataset = loader.dataset
    has_source_indices = hasattr(dataset, "source_indices")

    with torch.no_grad():
        batch_start = 0
        for features, batch_labels, _is_member in loader:
            features = features.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(features)
            probs = F.softmax(logits, dim=1)

            per_sample_loss = F.cross_entropy(logits, batch_labels, reduction="none")
            true_class_conf = probs.gather(1, batch_labels.view(-1, 1)).squeeze(1)
            max_conf = probs.max(dim=1).values

            batch_size = features.size(0)
            if has_source_indices:
                indices = dataset.source_indices[batch_start : batch_start + batch_size]
                sample_indices.extend(int(idx) for idx in indices)
            else:
                sample_indices.extend(range(batch_start, batch_start + batch_size))

            labels_list.extend(batch_labels.cpu().tolist())
            losses.extend(per_sample_loss.cpu().tolist())
            confidences.extend(true_class_conf.cpu().tolist())
            max_confidences.extend(max_conf.cpu().tolist())
            batch_start += batch_size

    return {
        "sample_idx": np.array(sample_indices, dtype=np.int64),
        "label": np.array(labels_list, dtype=np.int64),
        "loss": np.array(losses, dtype=np.float32),
        "confidence": np.array(confidences, dtype=np.float32),
        "max_confidence": np.array(max_confidences, dtype=np.float32),
    }


def run_attacks(
    model: torch.nn.Module,
    member_loader: DataLoader,
    non_member_loader: DataLoader,
    device: torch.device,
    config: dict,
    model_type: str,
    mitigation: str,
    utility_metadata: dict[str, object] | None = None,
) -> None:
    from attacks.lira import LiRAAttack
    from attacks.loss_based import LossBasedAttack
    from attacks.shadow import ShadowModelAttack
    from evaluation.metrics import append_result, attack_metrics

    attacks = {
        "loss_based": LossBasedAttack(device),
        "shadow": ShadowModelAttack(device),
        "lira": LiRAAttack(device),
    }
    y_true = np.concatenate([
        np.ones(len(member_loader.dataset), dtype=np.int64),
        np.zeros(len(non_member_loader.dataset), dtype=np.int64),
    ])

    member_metadata = compute_sample_metadata(model, member_loader, device)
    non_member_metadata = compute_sample_metadata(model, non_member_loader, device)

    for name, attack in attacks.items():
        attack.fit(model, member_loader, non_member_loader)
        member_scores = attack.score(model, member_loader)
        non_member_scores = attack.score(model, non_member_loader)
        scores = np.concatenate([member_scores, non_member_scores])
        save_attack_scores(
            config, model_type, name, mitigation,
            member_scores, non_member_scores,
            member_metadata, non_member_metadata,
        )
        row = {
            "model_type": model_type,
            "attack_type": name,
            "mitigation": mitigation,
            "seed": config["project"]["seed"],
            "train_acc": (utility_metadata or {}).get("train_acc", ""),
            "test_acc": (utility_metadata or {}).get("test_acc", ""),
        }
        row.update(attack_metrics(y_true, scores))
        append_result(config["paths"]["results_csv"], row)


def save_attack_scores(
    config: dict,
    model_type: str,
    attack_type: str,
    mitigation: str,
    member_scores,
    non_member_scores,
    member_metadata: dict[str, np.ndarray] | None = None,
    non_member_metadata: dict[str, np.ndarray] | None = None,
) -> None:
    import pandas as pd

    analysis_dir = Path(config["paths"]["experiment_root"]) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    output = analysis_dir / "attack_scores.csv"

    n_members = len(member_scores)
    n_non_members = len(non_member_scores)
    total = n_members + n_non_members

    data = {
        "model_type": [model_type] * total,
        "attack_type": [attack_type] * total,
        "mitigation": [mitigation] * total,
        "is_member": [1] * n_members + [0] * n_non_members,
        "score": list(member_scores) + list(non_member_scores),
    }

    member_meta = member_metadata or {}
    non_member_meta = non_member_metadata or {}
    for key in ["sample_idx", "label", "loss", "confidence", "max_confidence"]:
        member_vals = member_meta.get(key, np.full(n_members, np.nan))
        non_member_vals = non_member_meta.get(key, np.full(n_non_members, np.nan))
        data[key] = list(member_vals) + list(non_member_vals)

    frame = pd.DataFrame(data)
    frame.to_csv(output, mode="a", header=not output.exists(), index=False)


def run_attack_suite(
    input_dim: int,
    loaders: dict[str, DataLoader],
    device: torch.device,
    config: dict,
    feature_metadata: dict[str, object],
) -> None:
    teacher = _make_teacher(input_dim, config, feature_metadata)
    teacher_metadata = _load_if_exists(teacher, _checkpoint_path(config, "teacher.pt"), device)
    if teacher_metadata:
        run_attacks(
            teacher,
            loaders["train"],
            loaders["test"],
            device,
            config,
            model_type="teacher",
            mitigation="none",
            utility_metadata=teacher_metadata,
        )

    # baseline student + each mitigation variant. checkpoints we don't have are silently skipped.
    student_variants = [
        ("none", "student.pt"),
        ("bottleneck", "student_bottleneck.pt"),
        ("nonorm", "student_nonorm.pt"),
        ("confidence_filter", "student_confidence_filter.pt"),
    ]
    for mitigation, ckpt_name in student_variants:
        model = _build_student_for_mitigation(mitigation, input_dim, config)
        md = _load_if_exists(model, _checkpoint_path(config, ckpt_name), device)
        if not md:
            continue
        run_attacks(
            model,
            loaders["train"],
            loaders["test"],
            device,
            config,
            model_type="student",
            mitigation=mitigation,
            utility_metadata=md,
        )


def _build_student_for_mitigation(name: str, input_dim: int, config: dict) -> torch.nn.Module:
    from models.student import build_student

    if name in ("none", "confidence_filter"):
        return build_student(
            input_dim,
            int(config["model"]["num_classes"]),
            list(config["student"]["hidden_dims"]),
            float(config["model"]["dropout"]),
        )
    return _build_mitigated_models(input_dim, config)[name]


def run_full_lira_stage(
    features: np.ndarray,
    labels: np.ndarray,
    splits: SplitIndices,
    input_dim: int,
    device: torch.device,
    config: dict,
    feature_metadata: dict[str, object],
) -> None:
    """Run full LiRA with N reference models. GPU hours, not engineering."""
    from attacks.lira_full import run_full_lira
    from evaluation.metrics import append_result, attack_metrics

    print("Running full LiRA attack with reference models...")

    teacher = _make_teacher(input_dim, config, feature_metadata)
    teacher_metadata = _load_if_exists(teacher, _checkpoint_path(config, "teacher.pt"), device)

    if not teacher_metadata:
        print("Error: Teacher checkpoint not found. Train teacher first.")
        return

    target_indices = np.concatenate([splits.train, splits.test])
    lira_config = config.get("lira_full", config.get("lira", {}))
    cache_dir = Path(lira_config.get("cache_dir", "experiments/lira_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    member_scores, non_member_scores = run_full_lira(
        features=features,
        labels=labels,
        target_indices=target_indices,
        member_indices=splits.train,
        target_model=teacher,
        device=device,
        config={"lira": lira_config, "model": config["model"]},
        cache_dir=cache_dir,
    )

    y_true = np.concatenate([
        np.ones(len(member_scores), dtype=np.int64),
        np.zeros(len(non_member_scores), dtype=np.int64),
    ])
    scores = np.concatenate([member_scores, non_member_scores])

    row = {
        "model_type": "teacher",
        "attack_type": "lira_full",
        "mitigation": "none",
        "seed": config["project"]["seed"],
        "train_acc": teacher_metadata.get("train_acc", ""),
        "test_acc": teacher_metadata.get("test_acc", ""),
    }
    row.update(attack_metrics(y_true, scores))
    append_result(config["paths"]["results_csv"], row)

    print(f"Full LiRA attack complete. AUC: {row['auc']:.4f}")


def run_student_alpha_sweep(
    input_dim: int,
    teacher: torch.nn.Module,
    loaders: dict[str, DataLoader],
    device: torch.device,
    config: dict,
) -> None:
    # train one student per alpha, pick the tightest gap in the [40,45] window
    import csv
    import shutil
    from models.student import build_student, train_student

    alphas = list(config["student"].get("alpha_sweep", [0.3, 0.5, 0.7]))
    checkpoint_dir = Path(_checkpoint_path(config, "")).parent
    original_alpha = config["student"].get("alpha")

    rows = []
    for alpha in alphas:
        print(f"\n--- alpha={alpha} ---")
        config["student"]["alpha"] = float(alpha)
        student = build_student(
            input_dim,
            int(config["model"]["num_classes"]),
            list(config["student"]["hidden_dims"]),
            float(config["model"]["dropout"]),
        )
        ckpt = checkpoint_dir / f"student_alpha{str(alpha).replace('.', '')}.pt"
        result = train_student(student, teacher, loaders["train"], loaders["test"], device, config, ckpt)
        rows.append({
            "alpha": float(alpha),
            "train_acc": result.train_acc,
            "test_acc": result.eval_acc,
            "gap": result.train_acc - result.eval_acc,
            "checkpoint": str(ckpt),
        })
    if original_alpha is not None:
        config["student"]["alpha"] = original_alpha

    summary_path = checkpoint_dir / "sweep_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["alpha", "train_acc", "test_acc", "gap", "checkpoint"])
        writer.writeheader()
        writer.writerows(rows)

    in_window = [r for r in rows if 0.40 <= r["test_acc"] <= 0.45]
    best = min(in_window, key=lambda r: r["gap"]) if in_window else max(rows, key=lambda r: r["test_acc"])
    shutil.copy(best["checkpoint"], checkpoint_dir / "student.pt")

    print("\nsweep summary:")
    for r in rows:
        marker = " <- chosen" if r["checkpoint"] == best["checkpoint"] else ""
        print(f"  alpha={r['alpha']:.2f}  train={r['train_acc']:.4f}  test={r['test_acc']:.4f}  gap={r['gap']:.4f}{marker}")
    if not in_window:
        print("  none landed in [40, 45], picked highest test acc instead")


def run_subgroup_analysis_stage(config: dict) -> None:
    """Run subgroup vulnerability analysis and generate plots."""
    import pandas as pd

    from analysis.subgroup import (
        load_attack_scores_with_metadata,
        load_class_distribution,
        run_subgroup_analysis,
    )
    from evaluation.plots import make_subgroup_plots

    analysis_dir = Path(config["paths"]["experiment_root"]) / "analysis"
    figure_dir = Path(config["paths"]["figure_dir"])

    print("Running subgroup vulnerability analysis...")

    try:
        scores_df = load_attack_scores_with_metadata(analysis_dir)
        class_counts_df = load_class_distribution(analysis_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run 'run-attacks' first to generate attack scores.")
        return

    outputs = run_subgroup_analysis(scores_df, class_counts_df, analysis_dir)
    print(f"Generated {len(outputs)} subgroup analysis files")

    summary_path = analysis_dir / "subgroup_summary.csv"
    make_subgroup_plots(
        analysis_dir / "attack_scores.csv",
        analysis_dir / "class_distribution.csv",
        summary_path,
        figure_dir,
    )
    print(f"Subgroup plots saved to {figure_dir}")


if __name__ == "__main__":
    main()
