"""Recompute the subgroup_summary table with 95% bootstrap CIs on top of the
existing attack_scores.csv from a finished analysis directory. Avoids re-running
the full pipeline just to add CI columns.

Usage:
    python -m analysis.add_bootstrap_cis <analysis_dir>

Writes subgroup_summary_with_ci.csv alongside the existing subgroup_summary.csv."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from analysis.memorization import compute_feldman_zhang_scores
from analysis.subgroup import (
    compute_subgroup_metrics,
    load_attack_scores_with_metadata,
    load_class_distribution,
    stratify_by_class_frequency,
    stratify_by_confidence,
    stratify_by_loss_quantile,
    stratify_by_memorization,
)
from attacks.lira_reference import ReferenceModelCache


def _load_memorization_if_available(analysis_dir: Path) -> dict[int, float] | None:
    # cache lives in the global experiments/lira_cache/ directory by default; main.py
    # writes lira_cache_*.pkl files there and loads the most recent.
    for candidate_dir in [analysis_dir.parent / "lira_cache", Path("experiments/lira_cache")]:
        if not candidate_dir.exists():
            continue
        pkls = sorted(candidate_dir.glob("lira_cache_*.pkl"))
        if not pkls:
            continue
        try:
            cache = ReferenceModelCache.load(pkls[-1])
        except Exception as exc:
            print(f"Could not load LiRA cache from {pkls[-1]}: {exc}")
            continue
        if cache.correctness_matrix.size == 0:
            print(f"Cache at {pkls[-1]} has no correctness matrix; cannot compute F/Z.")
            return None
        print(f"Loaded LiRA cache from {pkls[-1]} ({cache.correctness_matrix.shape[0]} models).")
        return compute_feldman_zhang_scores(cache)
    return None


def main(analysis_dir: Path, n_bootstrap: int = 1000) -> None:
    scores_df = load_attack_scores_with_metadata(analysis_dir)
    class_counts_df = load_class_distribution(analysis_dir)
    memorization_scores = _load_memorization_if_available(analysis_dir)

    group_keys = ["model_type", "attack_type"]
    if "mitigation" in scores_df.columns:
        group_keys.append("mitigation")

    all_rows: list[pd.DataFrame] = []
    for keys, group in scores_df.groupby(group_keys):
        if len(group_keys) == 3:
            model_type, attack_type, mitigation = keys
        else:
            model_type, attack_type = keys
            mitigation = "none"

        stratifiers = [
            ("class_frequency", lambda g: stratify_by_class_frequency(g, class_counts_df)),
            ("confidence", stratify_by_confidence),
            ("loss_quantile", stratify_by_loss_quantile),
        ]
        if memorization_scores:
            stratifiers.append(
                ("memorization", lambda g: stratify_by_memorization(g, memorization_scores))
            )

        for strat_name, strat_fn in stratifiers:
            subgroups = strat_fn(group)
            metrics = compute_subgroup_metrics(
                subgroups, bootstrap=True, n_bootstrap=n_bootstrap
            )
            metrics["model_type"] = model_type
            metrics["attack_type"] = attack_type
            metrics["mitigation"] = mitigation
            metrics["stratification"] = strat_name
            all_rows.append(metrics)

    combined = pd.concat(all_rows, ignore_index=True)
    out_path = analysis_dir / "subgroup_summary_with_ci.csv"
    combined.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(combined)} rows).")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m analysis.add_bootstrap_cis <analysis_dir>")
        sys.exit(1)
    main(Path(sys.argv[1]))
