from __future__ import annotations

import tune_neutral_xgboost_no_grade_subgrade as base


base.OUTPUT_ROOT = base.MODELING_OUTPUT_ROOT / "xgboost_neutral_missingness_challenger"
base.TABLE_DIR = base.OUTPUT_ROOT / "tables"
base.MODEL_DIR = base.OUTPUT_ROOT / "models"
for directory in [base.TABLE_DIR, base.MODEL_DIR, base.FINAL_TABLE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

base.DATASET = "missingness_challenger"
base.DATASET_LABEL = "missingness challenger"
base.MODEL_DATASET_LABEL = "XGBoost neutral | missingness challenger"
base.FINAL_OUTPUT_PREFIX = "neutral_xgboost_missingness_challenger"


if __name__ == "__main__":
    base.main()
