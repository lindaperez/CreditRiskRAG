from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELING_DIR = PROJECT_ROOT / "Modeling"
OUTPUT_NOTEBOOK_PATH = MODELING_DIR / "7_Advanced_Script_Pipeline.ipynb"


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


CELLS = [
    markdown(
        """
# Accepted Loan Advanced Script Pipeline

Run this notebook after the core modeling notebooks and before the XGBoost grade/int-rate ablation and final model comparison notebooks.

This notebook replaces the manual terminal step for the advanced modeling scripts. It runs each script in the required order and fails fast if any script fails.
"""
    ),
    markdown(
        """
## Expected Notebook Order

0. `Preprocessing/0_Preprocessing.ipynb`
1. `1_LogisticRegression_Modeling.ipynb`
2. `2_RandomForest_Modeling.ipynb`
3. `3_HistGradientBoosting_Modeling.ipynb`
4. `4_LightGBM_Modeling.ipynb`
5. `5_XGBoost_Modeling.ipynb`
6. `6_CatBoost_Modeling.ipynb`
7. `7_Advanced_Script_Pipeline.ipynb`
8. `8_XGBoost_grade_IntRate_Ablation.ipynb`
9. `9_Final_Model_Comparison.ipynb`

The final comparison notebook runs `create_operating_policy_analysis.py` internally, so there is no extra terminal step after this notebook.
"""
    ),
    code(
        """
from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time

import pandas as pd
from IPython.display import display
"""
    ),
    markdown(
        """
## Locate Project Root
"""
    ),
    code(
        """
current = Path.cwd().resolve()
PROJECT_ROOT = None
for candidate in [current, *current.parents]:
    if candidate.name == "CreditRiskRAG":
        PROJECT_ROOT = candidate
        break
if PROJECT_ROOT is None:
    candidate = current / "CreditRiskRAG"
    if candidate.exists():
        PROJECT_ROOT = candidate
if PROJECT_ROOT is None:
    raise FileNotFoundError("Could not locate the CreditRiskRAG project root from " + str(current))

SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TABLE_DIR = PROJECT_ROOT / "Modeling" / "modeling_outputs" / "final_comparison" / "tables"

print("Project root:", PROJECT_ROOT)
print("Python executable:", sys.executable)
print("Scripts directory:", SCRIPTS_DIR)
"""
    ),
    markdown(
        """
## Advanced Script Sequence

These scripts create the missingness challenger comparisons, PR-AUC optimization outputs, class-weighting audit, neutral XGBoost retuning, grade/int-rate ablation outputs, calibration tables, and economic underwriting policy tables.
"""
    ),
    code(
        """
SCRIPT_STEPS = [
    {
        "step": 1,
        "script": "create_missingness_no_grade_subgrade_dataset.py",
        "purpose": "Create the missingness challenger dataset variant that excludes grade and sub_grade.",
    },
    {
        "step": 2,
        "script": "run_pr_auc_optimized_advanced_models.py",
        "purpose": "Train PR-AUC optimized LightGBM, XGBoost, and CatBoost variants.",
    },
    {
        "step": 3,
        "script": "run_missingness_challenger_all_models.py",
        "purpose": "Evaluate all model families on the standard missingness challenger dataset.",
    },
    {
        "step": 4,
        "script": "run_missingness_no_grade_subgrade_all_models.py",
        "purpose": "Evaluate all model families on the missingness challenger no-grade/subgrade dataset.",
    },
    {
        "step": 5,
        "script": "review_xgboost_class_weighting.py",
        "purpose": "Audit XGBoost scale_pos_weight versus neutral class weighting.",
    },
    {
        "step": 6,
        "script": "tune_neutral_xgboost_missingness_challenger.py",
        "purpose": "Retune neutral XGBoost on the standard missingness challenger dataset.",
    },
    {
        "step": 7,
        "script": "tune_neutral_xgboost_no_grade_subgrade.py",
        "purpose": "Retune neutral XGBoost on the no-grade/subgrade missingness challenger ablation.",
    },
    {
        "step": 8,
        "script": "test_xgboost_grade_int_rate_combinations.py",
        "purpose": "Test grade/sub_grade and int_rate feature combinations.",
    },
    {
        "step": 9,
        "script": "create_calibration_analysis.py",
        "purpose": "Create raw, Platt, and isotonic calibration outputs.",
    },
    {
        "step": 10,
        "script": "create_economic_underwriting_policy.py",
        "purpose": "Create the calibrated economic underwriting policy outputs.",
    },
]

steps = pd.DataFrame(SCRIPT_STEPS)
steps["script_path"] = steps["script"].map(lambda name: str(SCRIPTS_DIR / name))
steps["exists"] = steps["script_path"].map(lambda path: Path(path).exists())
display(steps)

missing = steps.loc[~steps["exists"], "script_path"].tolist()
if missing:
    raise FileNotFoundError("Missing scripts: " + ", ".join(missing))
"""
    ),
    markdown(
        """
## Run Scripts

This is the long-running cell. It runs all advanced scripts in order and records runtime status.
"""
    ),
    code(
        """
RUN_ADVANCED_SCRIPTS = True

run_rows = []
if RUN_ADVANCED_SCRIPTS:
    for item in SCRIPT_STEPS:
        script_path = SCRIPTS_DIR / item["script"]
        print(f"\\n=== Step {item['step']}: {item['script']} ===")
        print(item["purpose"])
        start = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(PROJECT_ROOT),
            check=True,
            text=True,
            capture_output=True,
        )
        seconds = time.perf_counter() - start
        print(completed.stdout)
        if completed.stderr:
            print("STDERR:")
            print(completed.stderr)
        run_rows.append({
            "step": item["step"],
            "script": item["script"],
            "status": "completed",
            "seconds": round(seconds, 2),
        })
else:
    print("RUN_ADVANCED_SCRIPTS is False. No scripts were executed.")

run_summary = pd.DataFrame(run_rows)
display(run_summary)
"""
    ),
    markdown(
        """
## Check Key Outputs

This confirms that the downstream notebooks have the files they need.
"""
    ),
    code(
        """
EXPECTED_OUTPUTS = [
    TABLE_DIR / "final_model_pr_auc_optimized_ranking_by_validation_pr_auc.csv",
    TABLE_DIR / "final_model_missingness_challenger_ranking_by_validation_pr_auc.csv",
    TABLE_DIR / "final_model_xgboost_class_weighting_review.csv",
    TABLE_DIR / "neutral_xgboost_missingness_challenger_candidate_results.csv",
    TABLE_DIR / "neutral_xgboost_no_grade_subgrade_candidate_results.csv",
    TABLE_DIR / "xgboost_grade_int_rate_selected_candidates.csv",
    TABLE_DIR / "final_model_calibration_summary.csv",
    TABLE_DIR / "final_model_economic_underwriting_policy_application.csv",
]

output_check = pd.DataFrame([
    {
        "output": str(path.relative_to(PROJECT_ROOT)),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }
    for path in EXPECTED_OUTPUTS
])
display(output_check)

missing_outputs = output_check.loc[~output_check["exists"], "output"].tolist()
if missing_outputs:
    raise FileNotFoundError("Missing expected outputs: " + ", ".join(missing_outputs))
"""
    ),
    markdown(
        """
## Next Notebook

After this notebook completes, run:

`8_XGBoost_grade_IntRate_Ablation.ipynb`

Then run:

`9_Final_Model_Comparison.ipynb`

The final comparison notebook now runs `create_operating_policy_analysis.py` internally before loading the operating-policy tables.
"""
    ),
]


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    OUTPUT_NOTEBOOK_PATH.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"Wrote {OUTPUT_NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
