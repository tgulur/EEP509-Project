"""Phase 2A: run the full pipeline at multiple seeds, aggregate across seeds.

Each seed gets its own run directory under experiments/runs/seed{SEED}/. After
every seed has finished, this script aggregates the population-level results.csv
files into experiments/multi_seed_aggregate.csv with mean and std per
(model_type, attack_type, mitigation) cell.

Usage:
    python -m scripts.multi_seed_run                       # default 3-seed run
    python -m scripts.multi_seed_run --seeds 509 510 511 512 513
    python -m scripts.multi_seed_run --skip-lira           # skip K=64 LiRA per seed (faster)
    python -m scripts.multi_seed_run --aggregate-only      # just rebuild the aggregate CSV
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PYTHON = sys.executable


def run_stage(stage: str, run_id: str, seed: int, extra: list[str] | None = None) -> None:
    cmd = [PYTHON, "main.py", stage, "--run-id", run_id, "--seed", str(seed)]
    if extra:
        cmd.extend(extra)
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_one_seed(seed: int, run_id: str, skip_lira: bool) -> None:
    # `all` covers train-teacher, train-student, train-mitigated, run-attacks, analysis
    run_stage("all", run_id, seed)
    if not skip_lira:
        run_stage("run-lira-full", run_id, seed)
    # rerun the subgroup analysis once the LiRA cache exists so F/Z scores get included
    run_stage("analyze", run_id, seed, extra=["--subgroups", "--resume-run", run_id])


def aggregate(seeds: list[int], output: Path) -> None:
    rows = []
    for seed in seeds:
        results_path = Path(f"experiments/runs/seed{seed}/results.csv")
        if not results_path.exists():
            print(f"WARNING: results missing for seed {seed} ({results_path})")
            continue
        df = pd.read_csv(results_path)
        df["seed"] = seed
        rows.append(df)
    if not rows:
        print("No results to aggregate.")
        return

    combined = pd.concat(rows, ignore_index=True)
    metric_cols = [c for c in combined.columns
                   if c.startswith(("auc", "tpr", "train_acc", "test_acc"))]
    grouped = combined.groupby(["model_type", "attack_type", "mitigation"])

    summary = []
    for keys, grp in grouped:
        row = {"model_type": keys[0], "attack_type": keys[1], "mitigation": keys[2],
               "n_seeds": len(grp)}
        for col in metric_cols:
            vals = grp[col].astype(float).values
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        summary.append(row)

    pd.DataFrame(summary).to_csv(output, index=False)
    print(f"\nWrote aggregate to {output} ({len(summary)} rows over {len(seeds)} seeds)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[509, 510, 511])
    p.add_argument("--skip-lira", action="store_true")
    p.add_argument("--aggregate-only", action="store_true")
    args = p.parse_args()

    if not args.aggregate_only:
        for seed in args.seeds:
            run_id = f"seed{seed}"
            print(f"\n========== SEED {seed} (run_id={run_id}) ==========")
            try:
                run_one_seed(seed, run_id, skip_lira=args.skip_lira)
            except subprocess.CalledProcessError as exc:
                print(f"Seed {seed} failed: {exc}. Continuing to next seed.")

    aggregate(args.seeds, Path("experiments/multi_seed_aggregate.csv"))


if __name__ == "__main__":
    main()
