# Texas-100X KD-MIA Pipeline

This repository implements the EE P 509 project described in `cursorrules.md`: membership inference attacks against knowledge-distilled models on Texas-100X.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place processed Texas-100X artifacts from `https://github.com/bargavj/Texas-100X` under:

```text
data/texas100x/
  texas_100x.csv
```

or:

```text
data/texas100x/
  texas_100x_features.p
  texas_100x_labels.p
  texas_100x_feature_desc.p
```

Raw and processed data are ignored by git. The loader enforces that `THCIC_ID` is excluded from model inputs.

## Commands

```bash
python main.py prepare-data
python main.py train-teacher
python main.py train-student
python main.py train-mitigated
python main.py run-attacks
python main.py plot
python main.py all
```

For local verification without Texas-100X:

```bash
python main.py smoke --synthetic
```

## Outputs

- Checkpoints: `experiments/checkpoints/`
- Cached soft labels: `experiments/soft_labels/`
- Result table: `experiments/results.csv`
- Figures: `experiments/figures/`

## Project Guarantees

- `THCIC_ID` is excluded from tensors used by models.
- Splits are deterministic from `project.seed` and saved under `data/processed/`.
- Dataset samples return `(features, label, is_member)`.
- Attacks expose a shared `MIAttack.fit(...)` and `MIAttack.score(...)` interface.
- Evaluation reports AUC plus TPR at 0.1%, 0.5%, and 1.0% FPR.
