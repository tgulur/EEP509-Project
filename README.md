# Membership Inference Attacks on Knowledge Distillation

EE P 509 A Project — Tejas Gulur

Empirical evaluation of whether knowledge distillation preserves, transfers, or amplifies membership inference vulnerability on the Texas-100X healthcare dataset.

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

## AI Note: 

The repo scaffolding was written by an AI Agent (cursor AI) as well as debugging the code with an AI agent. This README (up until the AI note) was also written by an AI agent so that I can remember how some features work. I've also asked AI to leave some helpful comments and lintering across the files so that I can come back and fix what needs to be done. 

For debugging purposes I utilized Claude Code to help as well. That's why you'll see some test files in tests folder to verify the fix went through and it won't break again in the current pipeline/implementation