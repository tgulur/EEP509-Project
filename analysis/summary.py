"""Proposal-aligned analysis summaries and figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import RocCurveDisplay


def run_analysis(config: dict) -> dict[str, Path]:
    """Generate report-ready analysis CSVs and figures."""

    analysis_dir = Path(config["paths"]["experiment_root"]) / "analysis"
    figure_dir = Path(config["paths"]["figure_dir"])
    analysis_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    data_csv = Path(config["paths"]["data_root"]) / config["data"]["csv_file"]
    results_csv = Path(config["paths"]["results_csv"])
    history_csv = Path(config["paths"]["experiment_root"]) / "teacher_history.csv"
    attack_scores_csv = analysis_dir / "attack_scores.csv"

    outputs = {
        "dataset_summary": write_dataset_summary(data_csv, Path(config["paths"]["processed_root"]) / "split_indices.npz", analysis_dir),
        "utility_summary": write_utility_summary(results_csv, history_csv, analysis_dir),
        "kd_transfer": write_kd_transfer_summary(results_csv, analysis_dir),
    }
    if attack_scores_csv.exists():
        write_attack_score_plots(attack_scores_csv, figure_dir)
        write_roc_plots(attack_scores_csv, figure_dir)
        outputs["attack_scores"] = attack_scores_csv
    write_training_plots(history_csv, figure_dir)
    return outputs


def write_dataset_summary(data_csv: Path, split_npz: Path, output_dir: Path) -> Path:
    """Write dataset and class distribution summaries."""

    frame = pd.read_csv(data_csv)
    label_column = "PRINC_SURG_PROC_CODE"
    feature_columns = [column for column in frame.columns if column != label_column and column != "THCIC_ID"]
    class_counts = frame[label_column].value_counts().rename_axis("procedure_code").reset_index(name="count")
    class_counts["fraction"] = class_counts["count"] / len(frame)
    class_counts.to_csv(output_dir / "class_distribution.csv", index=False)

    split_sizes = _split_sizes(split_npz)
    summary = [
        ("rows", len(frame)),
        ("model_features", len(feature_columns)),
        ("procedure_classes", frame[label_column].nunique()),
        ("hospitals", frame["THCIC_ID"].nunique() if "THCIC_ID" in frame.columns else ""),
        ("top_class_fraction", float(class_counts["fraction"].iloc[0])),
        ("median_class_count", float(class_counts["count"].median())),
        ("min_class_count", int(class_counts["count"].min())),
        ("max_class_count", int(class_counts["count"].max())),
        ("train_split_rows", split_sizes.get("train", "")),
        ("val_split_rows", split_sizes.get("val", "")),
        ("test_split_rows", split_sizes.get("test", "")),
    ]
    output = output_dir / "dataset_summary.csv"
    pd.DataFrame(summary, columns=["metric", "value"]).to_csv(output, index=False)
    return output


def write_utility_summary(results_csv: Path, history_csv: Path, output_dir: Path) -> Path:
    """Write utility and overfitting summaries for teacher and student."""

    results = pd.read_csv(results_csv)
    rows = []
    for model_type, group in results.groupby("model_type"):
        train_acc = pd.to_numeric(group["train_acc"], errors="coerce").dropna().iloc[0]
        test_acc = pd.to_numeric(group["test_acc"], errors="coerce").dropna().iloc[0]
        gap = train_acc - test_acc
        rows.append(
            {
                "model_type": model_type,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "train_test_gap": gap,
                "diagnosis": _diagnose_fit(train_acc, test_acc),
            }
        )

    if history_csv.exists():
        history = pd.read_csv(history_csv)
        best = history.loc[history["eval_acc"].idxmax()]
        rows.append(
            {
                "model_type": "teacher_best_epoch",
                "train_acc": best["train_acc"],
                "test_acc": best["eval_acc"],
                "train_test_gap": best["gap"],
                "diagnosis": f"best_epoch={int(best['epoch'])}",
            }
        )

    output = output_dir / "utility_summary.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def write_kd_transfer_summary(results_csv: Path, output_dir: Path) -> Path:
    """Summarize whether KD reduced, preserved, or amplified leakage."""

    results = pd.read_csv(results_csv)
    rows = []
    for attack_type, group in results.groupby("attack_type"):
        teacher = group[group["model_type"] == "teacher"]
        student = group[group["model_type"] == "student"]
        if teacher.empty or student.empty:
            continue
        teacher_row = teacher.iloc[0]
        student_row = student.iloc[0]
        teacher_auc = float(teacher_row["auc"])
        student_auc = float(student_row["auc"])
        teacher_test = float(teacher_row["test_acc"])
        student_test = float(student_row["test_acc"])
        delta = student_auc - teacher_auc
        rows.append(
            {
                "attack_type": attack_type,
                "teacher_auc": teacher_auc,
                "student_auc": student_auc,
                "auc_delta_student_minus_teacher": delta,
                "relative_auc_change": delta / teacher_auc if teacher_auc else 0.0,
                "teacher_test_acc": teacher_test,
                "student_test_acc": student_test,
                "utility_drop": teacher_test - student_test,
                "outcome": _classify_transfer(delta),
            }
        )
    output = output_dir / "kd_transfer_summary.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def write_training_plots(history_csv: Path, figure_dir: Path) -> None:
    """Create teacher training diagnostics."""

    if not history_csv.exists():
        return
    history = pd.read_csv(history_csv)
    _line_plot(history, "epoch", ["train_acc", "eval_acc"], "Accuracy", figure_dir / "teacher_accuracy_curve")
    _line_plot(history, "epoch", ["train_loss", "eval_loss"], "Loss", figure_dir / "teacher_loss_curve")
    _line_plot(history, "epoch", ["gap"], "Train-test accuracy gap", figure_dir / "teacher_gap_curve")


def write_attack_score_plots(attack_scores_csv: Path, figure_dir: Path) -> None:
    """Create member/non-member attack score distribution figures."""

    scores = pd.read_csv(attack_scores_csv)
    for (model_type, attack_type), group in scores.groupby(["model_type", "attack_type"]):
        plt.figure(figsize=(6, 4))
        for is_member, subset in group.groupby("is_member"):
            label = "member" if bool(is_member) else "non-member"
            plt.hist(subset["score"], bins=40, alpha=0.55, density=True, label=label)
        plt.xlabel("Attack score")
        plt.ylabel("Density")
        plt.title(f"{model_type} {attack_type} score distribution")
        plt.legend()
        plt.tight_layout()
        _save_current(figure_dir / f"score_distribution_{model_type}_{attack_type}")


def write_roc_plots(attack_scores_csv: Path, figure_dir: Path) -> None:
    """Create ROC curves comparing teacher and student for each attack."""

    scores = pd.read_csv(attack_scores_csv)
    for attack_type, group in scores.groupby("attack_type"):
        plt.figure(figsize=(6, 4))
        for model_type, subset in group.groupby("model_type"):
            RocCurveDisplay.from_predictions(subset["is_member"].astype(int), subset["score"], name=model_type, ax=plt.gca())
        plt.title(f"{attack_type} ROC")
        plt.tight_layout()
        _save_current(figure_dir / f"roc_{attack_type}")


def _line_plot(frame: pd.DataFrame, x_column: str, y_columns: list[str], ylabel: str, stem: Path) -> None:
    plt.figure(figsize=(6, 4))
    for column in y_columns:
        plt.plot(frame[x_column], frame[column], label=column)
    plt.xlabel(x_column)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    _save_current(stem)


def _split_sizes(split_npz: Path) -> dict[str, int]:
    if not split_npz.exists():
        return {}
    import numpy as np

    splits = np.load(split_npz)
    return {key: len(splits[key]) for key in splits.files}


def _diagnose_fit(train_acc: float, test_acc: float) -> str:
    gap = train_acc - test_acc
    if train_acc < 0.4:
        return "underfit"
    if gap > 0.1:
        return "overfit"
    return "moderate_gap"


def _classify_transfer(delta: float, threshold: float = 0.02) -> str:
    if delta > threshold:
        return "amplified"
    if delta < -threshold:
        return "reduced"
    return "transferred"


def _save_current(stem: Path) -> None:
    plt.savefig(stem.with_suffix(".png"), dpi=300)
    plt.savefig(stem.with_suffix(".pdf"), dpi=300)
    plt.close()
