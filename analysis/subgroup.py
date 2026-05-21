"""Subgroup vulnerability analysis for MIA evaluation.

Addresses the "evaluating only easy members" pitfall from Carlini et al. by
stratifying attack scores. Aggregated AUC can hide that rare classes or
high-loss samples are way more vulnerable than the average suggests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def load_attack_scores_with_metadata(analysis_dir: Path) -> pd.DataFrame:
    scores_path = analysis_dir / "attack_scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"Attack scores not found: {scores_path}")
    return pd.read_csv(scores_path)


def load_class_distribution(analysis_dir: Path) -> pd.DataFrame:
    dist_path = analysis_dir / "class_distribution.csv"
    if not dist_path.exists():
        raise FileNotFoundError(f"Class distribution not found: {dist_path}")
    return pd.read_csv(dist_path)


def stratify_by_class_frequency(
    scores_df: pd.DataFrame,
    class_counts_df: pd.DataFrame,
    bins: tuple[float, ...] = (0.0, 0.01, 0.05, 1.0),
    labels: tuple[str, ...] = ("rare_<1%", "medium_1-5%", "common_>5%"),
) -> dict[str, pd.DataFrame]:
    """Bin samples by their class's frequency. Rare classes tend to be more vulnerable."""
    if "label" not in scores_df.columns:
        raise ValueError("scores_df must have 'label' column for class stratification")

    label_to_freq = dict(zip(class_counts_df["procedure_code"], class_counts_df["fraction"]))
    scores_df = scores_df.copy()
    scores_df["class_freq"] = scores_df["label"].map(label_to_freq)
    scores_df["freq_bin"] = pd.cut(
        scores_df["class_freq"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    return {label: scores_df[scores_df["freq_bin"] == label] for label in labels}


def stratify_by_confidence(
    scores_df: pd.DataFrame,
    bins: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    labels: tuple[str, ...] = ("conf_0-25%", "conf_25-50%", "conf_50-75%", "conf_75-100%"),
) -> dict[str, pd.DataFrame]:
    """Bin by model confidence on the true class."""
    if "confidence" not in scores_df.columns:
        raise ValueError("scores_df must have 'confidence' column")

    scores_df = scores_df.copy()
    scores_df["conf_bin"] = pd.cut(
        scores_df["confidence"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    return {label: scores_df[scores_df["conf_bin"] == label] for label in labels}


def stratify_by_memorization(
    scores_df: pd.DataFrame,
    memorization_scores: dict[int, float],
    bins: tuple[float, ...] = (-1.0, 0.0, 0.1, 0.3, 1.0),
    labels: tuple[str, ...] = ("mem_neg_or_zero", "mem_low", "mem_med", "mem_high"),
) -> dict[str, pd.DataFrame]:
    """Bin by Feldman-Zhang memorization intensity. See analysis/memorization.py
    for the definition and the rationale.

    Fixed thresholds rather than quantiles: F/Z scores cluster heavily at zero
    (most samples aren't memorized), so quartile boundaries collapse and leave
    the actually-memorized samples in the same bin as everything else. With
    (-1, 0, 0.1, 0.3, 1] the mem_high bin picks out the long-tail samples we
    care about - in our run, ~200 of 100k samples land there."""
    if "sample_idx" not in scores_df.columns:
        raise ValueError("scores_df must have 'sample_idx' column for memorization stratification")

    scores_df = scores_df.copy()
    scores_df["memorization"] = scores_df["sample_idx"].map(memorization_scores)
    available = scores_df.dropna(subset=["memorization"]).copy()
    if available.empty:
        return {label: available.iloc[:0] for label in labels}

    available["mem_bin"] = pd.cut(
        available["memorization"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )
    return {label: available[available["mem_bin"] == label].dropna(subset=["mem_bin"]) for label in labels}


def stratify_by_loss_quantile(
    scores_df: pd.DataFrame,
    quantiles: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    labels: tuple[str, ...] = ("loss_q1_low", "loss_q2", "loss_q3", "loss_q4_high"),
) -> dict[str, pd.DataFrame]:
    """Bin by loss quantile. Caveat: the loss-based attack is built on this same
    signal, so loss-stratified subgroups are partly circular for that attack.
    Class-frequency and confidence bins are independent and tell the same story."""
    if "loss" not in scores_df.columns:
        raise ValueError("scores_df must have 'loss' column")

    scores_df = scores_df.copy()
    quantile_values = scores_df["loss"].quantile(quantiles).values
    scores_df["loss_bin"] = pd.cut(
        scores_df["loss"],
        bins=quantile_values,
        labels=labels,
        include_lowest=True,
        duplicates="drop",
    )

    return {label: scores_df[scores_df["loss_bin"] == label].dropna(subset=["loss_bin"]) for label in labels}


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    """Compute TPR at a specific FPR threshold."""
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(0, min(idx, len(tpr) - 1))
    return float(tpr[idx])


def compute_subgroup_metrics(
    subgroups: dict[str, pd.DataFrame],
    fpr_targets: tuple[float, ...] = (0.001, 0.01),
) -> pd.DataFrame:
    """AUC + TPR@FPR per subgroup. NaN when the subgroup has only one membership class."""
    rows = []
    for name, df in subgroups.items():
        if df.empty or df["is_member"].nunique() < 2:
            rows.append({
                "subgroup": name,
                "n_samples": len(df),
                "n_members": int(df["is_member"].sum()) if not df.empty else 0,
                "auc": np.nan,
                **{f"tpr_at_{fpr}_fpr": np.nan for fpr in fpr_targets},
            })
            continue

        y_true = df["is_member"].values
        scores = df["score"].values

        try:
            auc = roc_auc_score(y_true, scores)
        except ValueError:
            auc = np.nan

        row = {
            "subgroup": name,
            "n_samples": len(df),
            "n_members": int(y_true.sum()),
            "auc": auc,
        }
        for fpr in fpr_targets:
            try:
                row[f"tpr_at_{fpr}_fpr"] = tpr_at_fpr(y_true, scores, fpr)
            except Exception:
                row[f"tpr_at_{fpr}_fpr"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def run_subgroup_analysis(
    scores_df: pd.DataFrame,
    class_counts_df: pd.DataFrame,
    output_dir: Path,
    memorization_scores: dict[int, float] | None = None,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    # wipe stale per-bin CSVs from earlier runs (the file-naming scheme changed
    # when we added mitigation to the grouping, and old files would otherwise
    # get pulled back into the combined summary with empty mitigation cells).
    for stale in output_dir.glob("subgroup_*.csv"):
        if stale.name != "subgroup_summary.csv":
            stale.unlink()
    outputs = {}

    required_cols = {"label", "confidence", "loss"}
    available_cols = set(scores_df.columns)
    has_metadata = required_cols.issubset(available_cols)

    if not has_metadata:
        print(f"Warning: Missing metadata columns {required_cols - available_cols}. "
              "Re-run attacks to generate scores with metadata.")
        return outputs

    group_keys = ["model_type", "attack_type"]
    if "mitigation" in scores_df.columns:
        group_keys.append("mitigation")

    for keys, group in scores_df.groupby(group_keys):
        if len(group_keys) == 3:
            model_type, attack_type, mitigation = keys
        else:
            model_type, attack_type = keys
            mitigation = "none"
        prefix = f"{model_type}_{mitigation}_{attack_type}"

        stratifiers = [
            ("class_frequency", "freq", lambda g: stratify_by_class_frequency(g, class_counts_df)),
            ("confidence", "conf", stratify_by_confidence),
            ("loss_quantile", "loss", stratify_by_loss_quantile),
        ]
        if memorization_scores:
            stratifiers.append(
                ("memorization", "mem", lambda g: stratify_by_memorization(g, memorization_scores))
            )

        for strat_name, file_tag, strat_fn in stratifiers:
            subgroups = strat_fn(group)
            metrics = compute_subgroup_metrics(subgroups)
            metrics["model_type"] = model_type
            metrics["attack_type"] = attack_type
            metrics["mitigation"] = mitigation
            metrics["stratification"] = strat_name
            path = output_dir / f"subgroup_{file_tag}_{prefix}.csv"
            metrics.to_csv(path, index=False)
            outputs[f"{file_tag}_{prefix}"] = path

    combined = _combine_all_subgroup_results(output_dir)
    if combined is not None:
        summary_path = output_dir / "subgroup_summary.csv"
        combined.to_csv(summary_path, index=False)
        outputs["summary"] = summary_path

    return outputs


def _combine_all_subgroup_results(output_dir: Path) -> pd.DataFrame | None:
    """Combine all subgroup CSVs into a single summary."""
    pattern = "subgroup_*.csv"
    files = list(output_dir.glob(pattern))
    if not files:
        return None

    dfs = []
    for f in files:
        if f.name == "subgroup_summary.csv":
            continue
        df = pd.read_csv(f)
        dfs.append(df)

    if not dfs:
        return None

    return pd.concat(dfs, ignore_index=True)


def get_vulnerability_ranking(
    subgroup_summary: pd.DataFrame,
    metric: str = "auc",
) -> pd.DataFrame:
    df = subgroup_summary.copy()
    df = df.dropna(subset=[metric])
    df = df.sort_values(metric, ascending=False)
    df["vulnerability_rank"] = range(1, len(df) + 1)
    return df
