"""Plot generation - AUC bars, privacy-utility curves, etc."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def make_all_plots(results_csv: str | Path, figure_dir: str | Path) -> None:
    results_path = Path(results_csv)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    frame = pd.read_csv(results_path)
    # coerce to numeric in case there are blanks or weird values
    for column in ["auc", "train_acc", "test_acc"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    output = Path(figure_dir)
    output.mkdir(parents=True, exist_ok=True)
    _plot_auc_bars(frame, output)
    _plot_privacy_utility(frame, output)
    _plot_privacy_utility_by_attack(frame, output)


def _plot_auc_bars(frame: pd.DataFrame, output: Path) -> None:
    labels = frame["model_type"].astype(str) + "/" + frame["attack_type"].astype(str) + "/" + frame["mitigation"].astype(str)
    plt.figure(figsize=(max(8, len(labels) * 0.6), 4))
    plt.bar(range(len(frame)), frame["auc"])
    plt.xticks(range(len(frame)), labels, rotation=45, ha="right")
    plt.ylabel("AUC")
    plt.tight_layout()
    _save_current(output / "auc_bar_chart")


def _plot_privacy_utility(frame: pd.DataFrame, output: Path) -> None:
    frame = frame.dropna(subset=["test_acc", "auc"])
    if frame.empty:
        raise ValueError("No rows with numeric `test_acc` and `auc` are available for privacy-utility plotting.")

    plt.figure(figsize=(6, 4))
    for attack_type, group in frame.groupby("attack_type"):
        plt.scatter(group["test_acc"], group["auc"], label=attack_type)
        for _, row in group.iterrows():
            plt.annotate(
                _point_label(row),
                (row["test_acc"], row["auc"]),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )
    plt.xlabel("Test accuracy")
    plt.ylabel("Membership inference AUC")
    plt.title("Privacy-utility tradeoff")
    plt.legend()
    plt.tight_layout()
    _save_current(output / "privacy_utility_tradeoff")


def _plot_privacy_utility_by_attack(frame: pd.DataFrame, output: Path) -> None:
    frame = frame.dropna(subset=["test_acc", "auc"])
    for attack_type, group in frame.groupby("attack_type"):
        plt.figure(figsize=(6, 4))
        for model_type, subset in group.groupby("model_type"):
            plt.scatter(subset["test_acc"], subset["auc"], label=model_type)
            for _, row in subset.iterrows():
                plt.annotate(
                    _point_label(row),
                    (row["test_acc"], row["auc"]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                )
        plt.xlabel("Test accuracy")
        plt.ylabel("Membership inference AUC")
        plt.title(f"Privacy-utility tradeoff: {attack_type}")
        plt.legend()
        plt.tight_layout()
        _save_current(output / f"privacy_utility_{attack_type}")


def _save_current(stem: Path) -> None:
    plt.savefig(stem.with_suffix(".png"), dpi=300)
    plt.savefig(stem.with_suffix(".pdf"), dpi=300)
    plt.close()


def _point_label(row: pd.Series) -> str:
    model = str(row["model_type"])
    mitigation = str(row.get("mitigation", "none"))
    if mitigation and mitigation != "none":
        return f"{model} ({mitigation})"
    return model


def plot_subgroup_heatmap(
    subgroup_summary: pd.DataFrame,
    output_dir: Path,
    stratification: str = "class_frequency",
    metric: str = "auc",
) -> None:
    # TODO: add option to show TPR@FPR instead of just AUC
    import numpy as np

    df = subgroup_summary[subgroup_summary["stratification"] == stratification].copy()
    if df.empty:
        return

    pivot = df.pivot_table(
        values=metric,
        index="subgroup",
        columns=["model_type", "attack_type"],
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 1.2), max(4, len(pivot) * 0.6)))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0.5, vmax=1.0)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{m}/{a}" for m, a in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    plt.colorbar(im, ax=ax, label=metric.upper())
    plt.title(f"Subgroup Vulnerability ({stratification})")
    plt.tight_layout()
    _save_current(output_dir / f"subgroup_heatmap_{stratification}")


def plot_subgroup_roc_curves(
    scores_df: pd.DataFrame,
    class_counts_df: pd.DataFrame,
    output_dir: Path,
    model_type: str = "teacher",
    attack_type: str = "loss_based",
) -> None:
    """Create overlaid ROC curves for different subgroups.

    Args:
        scores_df: Attack scores with metadata
        class_counts_df: Class distribution DataFrame
        output_dir: Directory to save figures
        model_type: Filter to specific model
        attack_type: Filter to specific attack
    """
    from sklearn.metrics import RocCurveDisplay

    from analysis.subgroup import stratify_by_class_frequency

    df = scores_df[
        (scores_df["model_type"] == model_type) &
        (scores_df["attack_type"] == attack_type)
    ].copy()

    if df.empty or "label" not in df.columns:
        return

    subgroups = stratify_by_class_frequency(df, class_counts_df)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, subset in subgroups.items():
        if subset.empty or subset["is_member"].nunique() < 2:
            continue
        try:
            RocCurveDisplay.from_predictions(
                subset["is_member"].astype(int),
                subset["score"],
                name=name,
                ax=ax,
            )
        except Exception:
            continue

    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_title(f"ROC by Class Frequency: {model_type} / {attack_type}")
    ax.legend(loc="lower right")
    plt.tight_layout()
    _save_current(output_dir / f"roc_by_frequency_{model_type}_{attack_type}")


def plot_vulnerability_by_class_frequency(
    scores_df: pd.DataFrame,
    class_counts_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Scatter plot of per-class AUC vs class frequency.

    Args:
        scores_df: Attack scores with 'label' column
        class_counts_df: Class distribution DataFrame
        output_dir: Directory to save figures
    """
    from sklearn.metrics import roc_auc_score

    if "label" not in scores_df.columns:
        return

    label_to_freq = dict(zip(class_counts_df["procedure_code"], class_counts_df["fraction"]))

    results = []
    for (model_type, attack_type, label), group in scores_df.groupby(["model_type", "attack_type", "label"]):
        if group["is_member"].nunique() < 2 or len(group) < 20:
            continue
        try:
            auc = roc_auc_score(group["is_member"], group["score"])
            results.append({
                "model_type": model_type,
                "attack_type": attack_type,
                "label": label,
                "class_freq": label_to_freq.get(label, 0),
                "auc": auc,
                "n_samples": len(group),
            })
        except Exception:
            continue

    if not results:
        return

    df = pd.DataFrame(results)

    for attack_type, attack_group in df.groupby("attack_type"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for model_type, model_group in attack_group.groupby("model_type"):
            ax.scatter(
                model_group["class_freq"] * 100,
                model_group["auc"],
                alpha=0.6,
                label=model_type,
                s=model_group["n_samples"] / 10,
            )
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Class Frequency (%)")
        ax.set_ylabel("Per-Class AUC")
        ax.set_title(f"Vulnerability vs Class Frequency: {attack_type}")
        ax.legend()
        plt.tight_layout()
        _save_current(output_dir / f"vulnerability_vs_frequency_{attack_type}")


def make_subgroup_plots(
    scores_csv: str | Path,
    class_dist_csv: str | Path,
    subgroup_summary_csv: str | Path,
    output_dir: str | Path,
) -> None:
    """Generate all subgroup vulnerability plots.

    Args:
        scores_csv: Path to attack_scores.csv
        class_dist_csv: Path to class_distribution.csv
        subgroup_summary_csv: Path to subgroup_summary.csv
        output_dir: Directory to save figures
    """
    scores_path = Path(scores_csv)
    class_dist_path = Path(class_dist_csv)
    summary_path = Path(subgroup_summary_csv)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if not scores_path.exists():
        print(f"Warning: {scores_path} not found, skipping subgroup plots")
        return

    scores_df = pd.read_csv(scores_path)
    class_counts_df = pd.read_csv(class_dist_path) if class_dist_path.exists() else None

    if summary_path.exists():
        summary_df = pd.read_csv(summary_path)
        for strat in ["class_frequency", "confidence", "loss_quantile"]:
            plot_subgroup_heatmap(summary_df, output, stratification=strat)

    if class_counts_df is not None and "label" in scores_df.columns:
        plot_vulnerability_by_class_frequency(scores_df, class_counts_df, output)

        for model in scores_df["model_type"].unique():
            for attack in scores_df["attack_type"].unique():
                plot_subgroup_roc_curves(scores_df, class_counts_df, output, model, attack)
