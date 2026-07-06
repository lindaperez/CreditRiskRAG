# Reproducibility Guide

This project should be reproducible before any EDA, modeling, or explanation work starts. Reproducibility means the same code, environment, raw data files, schema, random seeds, and execution order produce the same results.

Reproducibility is enforced in three places:

1. **Environment:** pinned Python/package versions and raw-data checksum checks.
2. **Methods:** fixed sampling, cleaning, target, leakage, grouping, and validation rules inside the notebook.
3. **Plots:** fixed random seed, fixed row limits, fixed clipping rules, fixed top-N rules, and stable plot file exports.

## Required Baseline

- Use Python 3.11.
- Install dependencies from `requirements.lock.txt` or `environment.yml`.
- Keep raw LendingClub files unchanged under `Final/Data/archive`.
- Run `scripts/reproducibility_check.py` before running the EDA notebook.
- Treat `data_manifest.json` as the source-of-truth for expected raw file hashes and headers.

## Setup With Conda

From `Final/CreditRiskRAG`:

```bash
conda env create -f environment.yml
conda activate credit-risk-rag
python -m ipykernel install --user --name credit-risk-rag --display-name "Python (credit-risk-rag)"
```

## Setup With venv

From `Final/CreditRiskRAG`:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.lock.txt
python -m ipykernel install --user --name credit-risk-rag --display-name "Python (credit-risk-rag)"
```

## Environment File

Create a local `.env` from `.env.example`:

```bash
cp .env.example .env
```

Then set `PROJECT_ROOT` to the absolute path of `Final/CreditRiskRAG`.

## Preflight Check

Run:

```bash
python scripts/reproducibility_check.py
```

For a faster data-only check during development:

```bash
python scripts/reproducibility_check.py --skip-package-check --skip-hash-check
```

The strict check validates:

- Python version.
- Installed package versions.
- Raw data file existence.
- Raw data SHA256 checksums.
- Raw data headers.
- Required notebook reproducibility markers such as `RANDOM_STATE`, sample size, chunk size, keep-column lists, and leakage rules.

## Notebook Execution

Use the checked kernel and run the notebook from top to bottom:

```bash
jupyter nbconvert --to notebook --execute EDA/Accepted_Loan_EDA.ipynb --output Accepted_Loan_EDA.executed.ipynb
```

Do not run cells out of order when producing final EDA results. If the notebook changes, rerun the preflight check and then execute the notebook from a clean kernel.

## Reproducible Plots And Methods

The EDA notebook includes a dedicated `2A. Reproducibility Methods` section. It records:

- `RANDOM_STATE = 42`.
- `CONFIG.sample_rows = 250_000`.
- `CONFIG.chunk_size = 250_000`.
- `CONFIG.min_group_n = 500`.
- `PLOT_DPI = 150`.
- `PLOT_FORMAT = png`.
- Accepted-loan EDA output directory: `EDA/accepted_eda_outputs/`.
- Python, pandas, and NumPy versions used in the notebook run.

The plot methods are reproducible by design:

- Numeric distribution plots use the same seeded sample and p01/p99 clipping.
- Accepted-vs-rejected density plots use p01/p99 clipping.
- Categorical plots use fixed top-N limits.
- Default-rate plots suppress small groups through `CONFIG.min_group_n`.
- Every plot is saved with a stable filename before it is displayed.
- Every plot artifact includes a visible reproducibility footer with the seed, sample-row setting, chunk size, and clipping convention.

This means a reviewer can rerun the notebook and compare both the methods table and the generated plot files.

## Reproducibility Rules

1. Raw data files are immutable. If a raw file changes, update `data_manifest.json` only after confirming the source and reason.
2. Use a fixed `RANDOM_STATE` for sampling, splits, and any stochastic model step.
3. Save EDA plots with stable filenames and fixed plotting parameters.
4. Do not infer columns automatically for final modeling. Use reviewed feature lists with leakage and timestamp labels.
5. Keep raw fields and cleaned fields side by side for auditability.
6. Use time-based splits for model validation, not random splits as the primary estimate.
7. Record every target, feature, cleaning, plot, or exclusion decision in `Docs/EDA_DECISION_LOG.md`.
8. Save model-training datasets with their feature list, date range, target rule, row count, and source data manifest hash.

## What This Does Not Guarantee

This does not guarantee that the model is valid for lending decisions. It guarantees that the analysis inputs and execution conditions are controlled. Credit-risk validity still requires model validation, fair-lending review, and business sign-off.
