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
    _scatter_privacy_utility(frame, output / "privacy_utility_tradeoff", group_by="attack_type", title="Privacy-utility tradeoff")


def _plot_privacy_utility_by_attack(frame: pd.DataFrame, output: Path) -> None:
    frame = frame.dropna(subset=["test_acc", "auc"])
    for attack_type, group in frame.groupby("attack_type"):
        _scatter_privacy_utility(
            group,
            output / f"privacy_utility_{attack_type}",
            group_by="model_type",
            title=f"Privacy-utility tradeoff: {attack_type}",
        )


def _join_worst_case(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    stratification: str,
    worst_subgroup: str,
) -> pd.DataFrame:
    """For each row in results.csv, look up the AUC and TPR@1%FPR of the worst-
    case bin in subgroup_summary.csv. Rows without a subgroup match are dropped."""
    sg = summary_df[
        (summary_df["stratification"] == stratification)
        & (summary_df["subgroup"] == worst_subgroup)
    ]
    keys = ["model_type", "attack_type"]
    if "mitigation" in sg.columns and "mitigation" in results_df.columns:
        keys.append("mitigation")
    sg_cols = keys + ["auc"]
    rename = {"auc": "auc_worst"}
    if "tpr_at_0.01_fpr" in sg.columns:
        sg_cols.append("tpr_at_0.01_fpr")
        rename["tpr_at_0.01_fpr"] = "tpr_worst"
    merged = results_df.merge(sg[sg_cols].rename(columns=rename), on=keys, how="inner")
    return merged.dropna(subset=["test_acc", "auc_worst"])


# marker per model variant - keeps panels readable without per-point labels.
# teacher is a star to stand out, students get distinct shapes by mitigation.
_MARKERS = {
    "teacher/none": ("*", 220),
    "student/none": ("o", 70),
    "student/bottleneck": ("s", 60),
    "student/nonorm": ("^", 70),
    "student/confidence_filter": ("D", 55),
}


def _plot_privacy_utility_comparison(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output: Path,
) -> None:
    """2x3 figure: rows = {AUC, TPR @ 1% FPR}, columns = {population, loss-q4, mem_high}.

    The visual disparate-impact argument: each column shows the same models /
    attacks, but with a more aggressive measurement going right. The TPR row
    is the one that matters most for the F/Z column - the loss-based attack's
    AUC can look misleadingly low on memorized samples because the student's
    loss on those samples isn't well-ordered, but TPR @ 1% FPR still catches
    the small fraction that *are* memorized by the student."""

    # Shadow attack is in the data but excluded from this figure: our shadow
    # implementation sits near 0.5 AUC across the board (it does not train
    # full shadow-model attacks the way Shokri 2017 prescribes - it is closer
    # to a baseline check). Including it adds two cluttered points per panel
    # without adding storytelling value. Re-enable by removing the line below.
    results_df = results_df[results_df["attack_type"] != "shadow"].copy()
    summary_df = summary_df[summary_df["attack_type"] != "shadow"].copy()

    has_mem = ((summary_df["stratification"] == "memorization")
               & (summary_df["subgroup"] == "mem_high")).any()
    n_cols = 3 if has_mem else 2

    fig, axes = plt.subplots(2, n_cols, figsize=(5.5 * n_cols, 8.2), sharex=True)

    pop = results_df.dropna(subset=["test_acc", "auc"])
    loss_q4 = _join_worst_case(results_df, summary_df, "loss_quantile", "loss_q4_high")
    mem_high = _join_worst_case(results_df, summary_df, "memorization", "mem_high") if has_mem else None

    # row 0: AUC
    _draw_pu_panel(axes[0, 0], pop, y_col="auc", title="Population")
    _draw_pu_panel(axes[0, 1], loss_q4, y_col="auc_worst", title="Loss-q4 worst-case")
    if has_mem:
        _draw_pu_panel(axes[0, 2], mem_high, y_col="auc_worst", title="F/Z mem_high worst-case")

    # row 1: TPR @ 1% FPR
    _draw_pu_panel(axes[1, 0], pop, y_col="tpr_at_1fpr", title=None)
    _draw_pu_panel(axes[1, 1], loss_q4, y_col="tpr_worst", title=None)
    if has_mem:
        _draw_pu_panel(axes[1, 2], mem_high, y_col="tpr_worst", title=None)

    for ax in axes[0]:
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    for ax in axes[1]:
        ax.axhline(0.01, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    for ax in axes[1]:
        ax.set_xlabel("Test accuracy")
    axes[0, 0].set_ylabel("MIA AUC")
    axes[1, 0].set_ylabel("TPR @ 1% FPR")

    # single legend at the bottom - attack color + model/mitigation shape
    attack_handles, attack_labels = axes[0, 0].get_legend_handles_labels()
    shape_handles = [
        plt.Line2D([0], [0], marker=m, color="black", linestyle="", markersize=8 if m == "*" else 6, label=k)
        for k, (m, _) in _MARKERS.items()
    ]
    fig.legend(
        attack_handles + shape_handles,
        attack_labels + [h.get_label() for h in shape_handles],
        loc="lower center",
        ncol=min(6, len(attack_handles) + len(shape_handles)),
        bbox_to_anchor=(0.5, -0.02),
        fontsize=9,
    )
    plt.tight_layout(rect=(0, 0.06, 1, 1))
    _save_current(output / "privacy_utility_comparison")


def _draw_pu_panel(ax, frame: pd.DataFrame, y_col: str, title: str | None) -> None:
    if frame.empty:
        if title is not None:
            ax.set_title(f"{title}\n(no data)")
        return
    for attack_type, group in frame.groupby("attack_type"):
        for _, row in group.iterrows():
            key = f"{row['model_type']}/{row.get('mitigation', 'none')}"
            marker, size = _MARKERS.get(key, ("o", 50))
            ax.scatter(
                row["test_acc"], row[y_col],
                marker=marker, s=size, alpha=0.85,
                label=attack_type if _first_use(ax, attack_type) else None,
            )
    if title is not None:
        ax.set_title(title)
    ax.set_ylim(0.0, 1.0)


def _first_use(ax, label: str) -> bool:
    existing = {t.get_label() for t in ax.collections}
    return label not in existing


def _scatter_privacy_utility(frame: pd.DataFrame, stem: Path, group_by: str, title: str) -> None:
    plt.figure(figsize=(6, 4))
    for name, group in frame.groupby(group_by):
        plt.scatter(group["test_acc"], group["auc"], label=name)
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
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    _save_current(stem)


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
    """Overlay ROC curves for class-frequency subgroups of a single model/attack."""
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
    """Per-class AUC vs class frequency. Rare classes show up top-left."""
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
    results_csv: str | Path | None = None,
) -> None:
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
        for strat in ["class_frequency", "confidence", "loss_quantile", "memorization"]:
            plot_subgroup_heatmap(summary_df, output, stratification=strat)
        if results_csv is not None:
            results_path = Path(results_csv)
            if results_path.exists():
                results_df = pd.read_csv(results_path)
                for col in ["auc", "test_acc"]:
                    if col in results_df.columns:
                        results_df[col] = pd.to_numeric(results_df[col], errors="coerce")
                _plot_privacy_utility_comparison(results_df, summary_df, output)

    if class_counts_df is not None and "label" in scores_df.columns:
        plot_vulnerability_by_class_frequency(scores_df, class_counts_df, output)

        for model in scores_df["model_type"].unique():
            for attack in scores_df["attack_type"].unique():
                plot_subgroup_roc_curves(scores_df, class_counts_df, output, model, attack)
