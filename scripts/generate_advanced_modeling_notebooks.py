from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELING_DIR = PROJECT_ROOT / "Modeling"
BASE_NOTEBOOK_PATH = MODELING_DIR / "Accepted_Loan_HistGradientBoosting_Modeling.ipynb"


MODEL_CONFIGS = {
    "lightgbm": {
        "title": "LightGBM",
        "label": "LightGBM",
        "class_name": "LGBMClassifier",
        "import_code": "from lightgbm import LGBMClassifier",
        "build_code": """
def build_model(params: dict) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        num_leaves=params["num_leaves"],
        max_depth=params["max_depth"],
        min_child_samples=params["min_child_samples"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_lambda=params["reg_lambda"],
        scale_pos_weight=float((y_train == 0).sum() / max((y_train == 1).sum(), 1)),
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=-1,
    )

def build_fit_kwargs(y_fit: pd.Series) -> dict:
    return {}

CANDIDATES = [
    {"candidate": "lightgbm_01", "params": {"n_estimators": 250, "learning_rate": 0.04, "num_leaves": 31, "max_depth": -1, "min_child_samples": 60, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 1.0}},
    {"candidate": "lightgbm_02", "params": {"n_estimators": 300, "learning_rate": 0.035, "num_leaves": 45, "max_depth": -1, "min_child_samples": 80, "subsample": 0.85, "colsample_bytree": 0.80, "reg_lambda": 2.0}},
    {"candidate": "lightgbm_03", "params": {"n_estimators": 350, "learning_rate": 0.03, "num_leaves": 63, "max_depth": -1, "min_child_samples": 100, "subsample": 0.80, "colsample_bytree": 0.80, "reg_lambda": 3.0}},
    {"candidate": "lightgbm_04", "params": {"n_estimators": 220, "learning_rate": 0.05, "num_leaves": 31, "max_depth": 8, "min_child_samples": 80, "subsample": 0.90, "colsample_bytree": 0.90, "reg_lambda": 1.0}},
    {"candidate": "lightgbm_05", "params": {"n_estimators": 280, "learning_rate": 0.04, "num_leaves": 45, "max_depth": 10, "min_child_samples": 120, "subsample": 0.80, "colsample_bytree": 0.85, "reg_lambda": 4.0}},
    {"candidate": "lightgbm_06", "params": {"n_estimators": 400, "learning_rate": 0.025, "num_leaves": 63, "max_depth": 12, "min_child_samples": 120, "subsample": 0.80, "colsample_bytree": 0.75, "reg_lambda": 5.0}},
]
FIT_SAMPLE_ROWS = 300_000
""".strip(),
    },
    "xgboost": {
        "title": "XGBoost",
        "label": "XGBoost",
        "class_name": "XGBClassifier",
        "import_code": "from xgboost import XGBClassifier",
        "build_code": """
def build_model(params: dict) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        min_child_weight=params["min_child_weight"],
        subsample=params["subsample"],
        colsample_bytree=params["colsample_bytree"],
        reg_lambda=params["reg_lambda"],
        reg_alpha=params["reg_alpha"],
        scale_pos_weight=float((y_train == 0).sum() / max((y_train == 1).sum(), 1)),
        random_state=RANDOM_STATE,
        n_jobs=1,
    )

def build_fit_kwargs(y_fit: pd.Series) -> dict:
    return {}

CANDIDATES = [
    {"candidate": "xgboost_01", "params": {"n_estimators": 220, "learning_rate": 0.04, "max_depth": 4, "min_child_weight": 8, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 2.0, "reg_alpha": 0.0}},
    {"candidate": "xgboost_02", "params": {"n_estimators": 260, "learning_rate": 0.035, "max_depth": 4, "min_child_weight": 12, "subsample": 0.85, "colsample_bytree": 0.80, "reg_lambda": 3.0, "reg_alpha": 0.1}},
    {"candidate": "xgboost_03", "params": {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 5, "min_child_weight": 10, "subsample": 0.80, "colsample_bytree": 0.80, "reg_lambda": 4.0, "reg_alpha": 0.1}},
    {"candidate": "xgboost_04", "params": {"n_estimators": 240, "learning_rate": 0.045, "max_depth": 5, "min_child_weight": 16, "subsample": 0.90, "colsample_bytree": 0.85, "reg_lambda": 3.0, "reg_alpha": 0.2}},
    {"candidate": "xgboost_05", "params": {"n_estimators": 350, "learning_rate": 0.025, "max_depth": 6, "min_child_weight": 16, "subsample": 0.80, "colsample_bytree": 0.75, "reg_lambda": 5.0, "reg_alpha": 0.2}},
    {"candidate": "xgboost_06", "params": {"n_estimators": 180, "learning_rate": 0.06, "max_depth": 3, "min_child_weight": 8, "subsample": 0.90, "colsample_bytree": 0.90, "reg_lambda": 2.0, "reg_alpha": 0.0}},
]
FIT_SAMPLE_ROWS = 300_000
""".strip(),
    },
    "catboost": {
        "title": "CatBoost",
        "label": "CatBoost",
        "class_name": "CatBoostClassifier",
        "import_code": "from catboost import CatBoostClassifier",
        "build_code": """
def build_model(params: dict) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=params["iterations"],
        learning_rate=params["learning_rate"],
        depth=params["depth"],
        l2_leaf_reg=params["l2_leaf_reg"],
        subsample=params["subsample"],
        random_seed=RANDOM_STATE,
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
        auto_class_weights="Balanced",
    )

def build_fit_kwargs(y_fit: pd.Series) -> dict:
    return {}

CANDIDATES = [
    {"candidate": "catboost_01", "params": {"iterations": 220, "learning_rate": 0.04, "depth": 4, "l2_leaf_reg": 3.0, "subsample": 0.85}},
    {"candidate": "catboost_02", "params": {"iterations": 260, "learning_rate": 0.035, "depth": 5, "l2_leaf_reg": 4.0, "subsample": 0.85}},
    {"candidate": "catboost_03", "params": {"iterations": 300, "learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 5.0, "subsample": 0.80}},
    {"candidate": "catboost_04", "params": {"iterations": 240, "learning_rate": 0.045, "depth": 5, "l2_leaf_reg": 6.0, "subsample": 0.90}},
    {"candidate": "catboost_05", "params": {"iterations": 350, "learning_rate": 0.025, "depth": 6, "l2_leaf_reg": 8.0, "subsample": 0.80}},
    {"candidate": "catboost_06", "params": {"iterations": 180, "learning_rate": 0.06, "depth": 4, "l2_leaf_reg": 5.0, "subsample": 0.90}},
]
FIT_SAMPLE_ROWS = 300_000
""".strip(),
    },
}


def source_lines(text: str) -> list[str]:
    return [line + "\n" for line in text.splitlines()]


def new_code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines(text),
    }


def new_markdown_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source_lines(text)}


NO_GRADE_SUBGRADE_MARKDOWN = new_markdown_cell(
    "## 7. No-Grade/Subgrade Modeling\n\n"
    "Train and tune the same candidate grid on the `baseline_no_grade_subgrade` feature set so the advanced model families can be compared against the prior no-grade/subgrade baseline outputs."
)


NO_GRADE_SUBGRADE_CODE = new_code_cell(
    """
NO_GRADE_SUFFIX = "no_grade_subgrade"
CENTRAL_NO_GRADE_METRICS_PATH = MODELING_OUTPUT_ROOT / "tables" / "no_grade_subgrade_model_metrics.csv"

def save_or_replace_central_no_grade_metrics(new_metrics: pd.DataFrame) -> Path:
    CENTRAL_NO_GRADE_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if CENTRAL_NO_GRADE_METRICS_PATH.exists():
        existing = pd.read_csv(CENTRAL_NO_GRADE_METRICS_PATH)
        existing = existing[~existing["model"].str.startswith(f"{MODEL_FAMILY}_{NO_GRADE_SUFFIX}")]
        combined = pd.concat([existing, new_metrics], ignore_index=True)
    else:
        combined = new_metrics.copy()
    combined.to_csv(CENTRAL_NO_GRADE_METRICS_PATH, index=False)
    print("Saved:", CENTRAL_NO_GRADE_METRICS_PATH)
    return CENTRAL_NO_GRADE_METRICS_PATH

def run_no_grade_subgrade_modeling() -> None:
    global X_train, X_validation, X_test

    original_X_train = X_train
    original_X_validation = X_validation
    original_X_test = X_test

    try:
        X_train = load_parquet("baseline_no_grade_subgrade_train_X")
        X_validation = load_parquet("baseline_no_grade_subgrade_validation_X")
        X_test = load_parquet("baseline_no_grade_subgrade_test_X")

        no_grade_input_summary = pd.DataFrame([
            {"split": "train", "rows": len(X_train), "columns": X_train.shape[1], "bad_rate": y_train.mean()},
            {"split": "validation", "rows": len(X_validation), "columns": X_validation.shape[1], "bad_rate": y_validation.mean()},
            {"split": "test", "rows": len(X_test), "columns": X_test.shape[1], "bad_rate": y_test.mean()},
        ])
        no_grade_input_summary["bad_rate"] = no_grade_input_summary["bad_rate"].round(6)
        save_table(no_grade_input_summary, f"{NO_GRADE_SUFFIX}_input_summary")
        display(no_grade_input_summary)

        no_grade_candidates = [
            {**candidate, "candidate": f"{candidate['candidate']}_{NO_GRADE_SUFFIX}"}
            for candidate in CANDIDATES
        ]
        no_grade_candidate_results, no_grade_fitted_models = fit_candidates(
            no_grade_candidates,
            sample_rows=FIT_SAMPLE_ROWS,
        )
        save_table(no_grade_candidate_results, f"{NO_GRADE_SUFFIX}_candidate_results")
        display(no_grade_candidate_results)

        no_grade_winner = no_grade_candidate_results.iloc[0]
        no_grade_selected_model = no_grade_fitted_models[no_grade_winner.candidate]
        no_grade_selected_metrics, no_grade_review_volume_precision = evaluate_selected_model(
            no_grade_winner,
            no_grade_selected_model,
        )

        no_grade_selected_metrics = no_grade_selected_metrics[
            no_grade_selected_metrics["operating_point"] == "best_validation_f1"
        ].copy()
        no_grade_selected_metrics = no_grade_selected_metrics.drop(columns=["model_family", "operating_point"])

        save_table(pd.DataFrame([no_grade_winner]), f"{NO_GRADE_SUFFIX}_selected_candidate")
        save_table(no_grade_selected_metrics, f"{NO_GRADE_SUFFIX}_selected_model_metrics")
        save_table(no_grade_review_volume_precision, f"{NO_GRADE_SUFFIX}_review_volume_precision")
        save_or_replace_central_no_grade_metrics(no_grade_selected_metrics)
        display(no_grade_selected_metrics)
        display(no_grade_review_volume_precision)

        no_grade_model_path = MODEL_DIR / f"{MODEL_FAMILY}_{NO_GRADE_SUFFIX}_selected_model.joblib"
        joblib.dump(no_grade_selected_model, no_grade_model_path)
        no_grade_artifact_table = pd.DataFrame([{
            "model_family": MODEL_FAMILY,
            "candidate": no_grade_winner.candidate,
            "feature_set": "baseline_no_grade_subgrade",
            "artifact_path": str(no_grade_model_path),
        }])
        save_table(no_grade_artifact_table, f"{NO_GRADE_SUFFIX}_model_artifact")
        print("Saved:", no_grade_model_path)

        no_grade_confusion_matrix_long, no_grade_per_class_metrics = build_confusion_matrix_tables(
            no_grade_selected_metrics.assign(
                model_family=MODEL_FAMILY,
                operating_point="best_validation_f1",
            )
        )
        save_table(no_grade_confusion_matrix_long, f"{NO_GRADE_SUFFIX}_confusion_matrix")
        save_table(no_grade_per_class_metrics, f"{NO_GRADE_SUFFIX}_per_class_metrics")
        display(no_grade_confusion_matrix_long)
        display(no_grade_per_class_metrics)

        for metric in ["f1", "precision", "recall", "pr_auc", "roc_auc"]:
            plot_df = no_grade_selected_metrics[no_grade_selected_metrics["split"].isin(["validation", "test"])]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.bar(plot_df["split"], plot_df[metric])
            ax.set_title(f"{MODEL_LABEL} no-grade/subgrade {metric.upper()} by split")
            ax.set_ylim(0, max(plot_df[metric].max() * 1.15, 0.05))
            ax.grid(axis="y", alpha=0.25)
            save_plot(fig, f"{NO_GRADE_SUFFIX}_selected_{metric}_comparison")
    finally:
        X_train = original_X_train
        X_validation = original_X_validation
        X_test = original_X_test

run_no_grade_subgrade_modeling()
""".strip()
)


def write_model_notebooks() -> None:
    base_nb = json.loads(BASE_NOTEBOOK_PATH.read_text())
    for family, config in MODEL_CONFIGS.items():
        nb = deepcopy(base_nb)
        nb["cells"][0]["source"] = source_lines(
            f"# Accepted Loan {config['title']} Modeling And Tuning\n\n"
            f"This notebook trains and tunes `{family}` candidates using the chronological baseline preprocessing exports; "
            "candidate search may use a stratified train-period sample for runtime. Thresholds are selected on validation only."
        )
        setup = "".join(nb["cells"][2]["source"])
        setup = setup.replace("MODEL_FAMILY = 'hist_gradient_boosting'", f"MODEL_FAMILY = '{family}'")
        setup = setup.replace("MODEL_LABEL = 'HistGradientBoostingClassifier'", f"MODEL_LABEL = '{config['label']}'")
        setup = setup.replace("from sklearn.ensemble import HistGradientBoostingClassifier", config["import_code"])
        setup = setup.replace(
            f"MODEL_FAMILY = '{family}'\nMODEL_LABEL = '{config['label']}'\n\nfrom __future__ import annotations",
            f"from __future__ import annotations\n\nMODEL_FAMILY = '{family}'\nMODEL_LABEL = '{config['label']}'",
        )
        nb["cells"][2]["source"] = source_lines(setup)
        nb["cells"][8]["source"] = source_lines(config["build_code"])
        nb["cells"].insert(-1, deepcopy(NO_GRADE_SUBGRADE_MARKDOWN))
        nb["cells"].insert(-1, deepcopy(NO_GRADE_SUBGRADE_CODE))
        nb["cells"][-1]["source"] = source_lines("## 8. Notes\n\nUse validation metrics for model/threshold selection. Use test metrics only for final reporting after selection.")
        output_path = MODELING_DIR / f"Accepted_Loan_{config['title']}_Modeling.ipynb"
        output_path.write_text(json.dumps(nb, indent=1) + "\n")
        print(f"Wrote {output_path}")


FINAL_NOTEBOOK_CELLS = [
    new_markdown_cell(
        "# Accepted Loan Final Model Comparison\n\n"
        "Compare the selected Logistic Regression, Random Forest, HistGradientBoosting, LightGBM, XGBoost, and CatBoost models using their per-model notebook outputs."
    ),
    new_markdown_cell("## 1. Setup"),
    new_code_cell(
        """
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def find_project_root() -> Path:
    current = Path.cwd().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "Cleaning").exists() and (candidate / "Modeling").exists():
            return candidate
    raise FileNotFoundError("Could not find CreditRiskRAG project root")

PROJECT_ROOT = find_project_root()
MODELING_OUTPUT_ROOT = PROJECT_ROOT / "Modeling" / "modeling_outputs"
FINAL_OUTPUT_ROOT = MODELING_OUTPUT_ROOT / "final_comparison"
TABLE_DIR = FINAL_OUTPUT_ROOT / "tables"
PLOT_DIR = FINAL_OUTPUT_ROOT / "plots"
for directory in [TABLE_DIR, PLOT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

MODEL_FAMILIES = [
    "logistic_regression",
    "random_forest",
    "hist_gradient_boosting",
    "lightgbm",
    "xgboost",
    "catboost",
]
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
}
ABLATION_MODEL_FAMILIES = MODEL_FAMILIES.copy()

def save_table(df: pd.DataFrame, name: str) -> Path:
    path = TABLE_DIR / f"final_model_{name}.csv"
    df.to_csv(path, index=False)
    print("Saved:", path)
    return path

def save_plot(fig, name: str) -> Path:
    path = PLOT_DIR / f"final_model_{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)
    return path

print("Project root:", PROJECT_ROOT)
print("Final comparison outputs:", FINAL_OUTPUT_ROOT)
""".strip()
    ),
    new_markdown_cell("## 2. Load Per-Model Outputs"),
    new_code_cell(
        """
def model_table_path(model_family: str, table_name: str) -> Path:
    return MODELING_OUTPUT_ROOT / model_family / "tables" / f"{model_family}_{table_name}.csv"

def read_model_table(model_family: str, table_name: str) -> pd.DataFrame:
    path = model_table_path(model_family, table_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}. Run the {model_family} notebook first.")
    return pd.read_csv(path)

required_tables = ["candidate_results", "selected_model_metrics", "review_volume_precision", "selected_candidate"]
available_families = []
missing_rows = []

for model_family in MODEL_FAMILIES:
    missing = [table_name for table_name in required_tables if not model_table_path(model_family, table_name).exists()]
    if missing:
        missing_rows.append({
            "model_family": model_family,
            "model_label": MODEL_LABELS[model_family],
            "missing_tables": ", ".join(missing),
            "notebook_to_run": f"Accepted_Loan_{MODEL_LABELS[model_family].replace(' ', '')}_Modeling.ipynb",
        })
    else:
        available_families.append(model_family)

missing_model_outputs = pd.DataFrame(missing_rows)
save_table(missing_model_outputs, "missing_model_outputs")
if not missing_model_outputs.empty:
    display(missing_model_outputs)

if not available_families:
    raise FileNotFoundError("No per-model outputs are available. Run at least one modeling notebook first.")

candidate_tables = []
selected_metric_tables = []
review_volume_tables = []
selected_candidate_tables = []
for model_family in available_families:
    candidate_tables.append(read_model_table(model_family, "candidate_results"))
    selected_metric_tables.append(read_model_table(model_family, "selected_model_metrics"))
    review_volume_tables.append(read_model_table(model_family, "review_volume_precision"))
    selected_candidate_tables.append(read_model_table(model_family, "selected_candidate"))

all_candidates = pd.concat(candidate_tables, ignore_index=True)
selected_metrics = pd.concat(selected_metric_tables, ignore_index=True)
review_volume_precision = pd.concat(review_volume_tables, ignore_index=True)
selected_candidates = pd.concat(selected_candidate_tables, ignore_index=True)

for frame in [all_candidates, selected_metrics, review_volume_precision, selected_candidates]:
    frame["model_label"] = frame["model_family"].map(MODEL_LABELS)

save_table(all_candidates, "candidate_summary")
save_table(selected_metrics, "selected_metrics")
save_table(review_volume_precision, "review_volume_precision")
save_table(selected_candidates, "selected_candidates")
display(selected_candidates)
display(selected_metrics)
""".strip()
    ),
    new_markdown_cell("## 3. Validation And Test Comparison"),
    new_code_cell(
        """
plot_families = [family for family in MODEL_FAMILIES if family in available_families]

comparison = selected_metrics[
    selected_metrics["split"].isin(["validation", "test"])
].copy()
comparison = comparison.sort_values(
    ["operating_point", "split", "f1", "precision", "pr_auc"],
    ascending=[True, True, False, False, False],
)
save_table(comparison, "validation_test_comparison")
display(comparison)

for metric in ["f1", "precision", "recall", "pr_auc", "roc_auc", "brier_score"]:
    plot_df = comparison[comparison["operating_point"] == "best_validation_f1"]
    fig, ax = plt.subplots(figsize=(max(9, len(plot_families) * 1.35), 4.5))
    width = 0.35
    labels = [MODEL_LABELS[m] for m in plot_families]
    x = np.arange(len(labels))
    metric_by_family = plot_df.set_index(["model_family", "split"])[metric]
    val = [metric_by_family.get((m, "validation"), np.nan) for m in plot_families]
    test = [metric_by_family.get((m, "test"), np.nan) for m in plot_families]
    ax.bar(x - width/2, val, width, label="validation")
    ax.bar(x + width/2, test, width, label="test")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(f"Final model comparison: {metric}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_plot(fig, f"{metric}_comparison")
""".strip()
    ),
    new_markdown_cell("## 4. Fixed Review Volume Comparison"),
    new_code_cell(
        """
review_compare = review_volume_precision[review_volume_precision["split"].isin(["validation", "test"])].copy()
save_table(review_compare, "review_volume_comparison")
display(review_compare)

for split in ["validation", "test"]:
    plot_df = review_compare[review_compare["split"] == split]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for model_family in plot_families:
        group = plot_df[plot_df["model_family"] == model_family]
        if group.empty:
            continue
        ax.plot(group["review_pct"], group["precision"], marker="o", label=MODEL_LABELS[model_family])
    ax.set_title(f"Precision at fixed review volumes: {split}")
    ax.set_xlabel("Reviewed applications (%)")
    ax.set_ylabel("Precision")
    ax.grid(alpha=0.25)
    ax.legend()
    save_plot(fig, f"precision_by_review_volume_{split}")
""".strip()
    ),
    new_markdown_cell(
        "## 5. Grade/Subgrade Ablation\n\n"
        "Compare the current baseline feature set against the `baseline_no_grade_subgrade` feature set for every model family with no-grade/subgrade outputs."
    ),
    new_code_cell(
        """
NO_GRADE_METRICS_PATH = MODELING_OUTPUT_ROOT / "tables" / "no_grade_subgrade_model_metrics.csv"
NO_GRADE_EXPORT_SUMMARY_PATH = PROJECT_ROOT / "Modeling" / "Preprocessing" / "preprocessing_outputs" / "tables" / "preprocessing_no_grade_subgrade_export_summary.csv"

if not NO_GRADE_METRICS_PATH.exists():
    raise FileNotFoundError(
        f"Missing {NO_GRADE_METRICS_PATH}. Run the no-grade/subgrade modeling export first."
    )

no_grade_metrics = pd.read_csv(NO_GRADE_METRICS_PATH)
no_grade_metrics = no_grade_metrics[no_grade_metrics["split"].isin(["validation", "test"])].copy()

def infer_no_grade_model_family(model_name: str) -> str:
    base_name = model_name.replace("_no_grade_subgrade", "")
    for model_family in MODEL_FAMILIES:
        if base_name == model_family or base_name.startswith(f"{model_family}_"):
            return model_family
    return base_name

no_grade_metrics["model_family"] = no_grade_metrics["model"].map(infer_no_grade_model_family)
no_grade_metrics = no_grade_metrics[no_grade_metrics["model_family"].isin(ABLATION_MODEL_FAMILIES)]
no_grade_metrics["model_label"] = no_grade_metrics["model_family"].map(MODEL_LABELS)
no_grade_metrics["feature_set"] = "baseline_no_grade_subgrade"

with_grade_metrics = selected_metrics[
    (selected_metrics["model_family"].isin(ABLATION_MODEL_FAMILIES)) &
    (selected_metrics["split"].isin(["validation", "test"])) &
    (selected_metrics["operating_point"] == "best_validation_f1")
].copy()
with_grade_metrics["feature_set"] = "baseline_with_grade_subgrade"
with_grade_metrics = with_grade_metrics[[
    "model_family", "model_label", "model", "split", "feature_set",
    "roc_auc", "pr_auc", "brier_score", "precision", "recall", "f1", "threshold"
]]

no_grade_metrics = no_grade_metrics[[
    "model_family", "model_label", "model", "split", "feature_set",
    "roc_auc", "pr_auc", "brier_score", "precision", "recall", "f1", "threshold"
]]

ablation_metrics = pd.concat([with_grade_metrics, no_grade_metrics], ignore_index=True)
FEATURE_SET_LABELS = {
    "baseline_with_grade_subgrade": "with grade/subgrade",
    "baseline_no_grade_subgrade": "no grade/subgrade",
}
ablation_metrics["feature_set_label"] = ablation_metrics["feature_set"].map(FEATURE_SET_LABELS)
ablation_metrics["model_feature_label"] = ablation_metrics["model_label"] + " | " + ablation_metrics["feature_set_label"]
save_table(ablation_metrics, "grade_subgrade_ablation_metrics")
display(ablation_metrics.sort_values(["split", "model_label", "feature_set"])[[
    "model_feature_label", "model_label", "model", "split", "feature_set", "feature_set_label",
    "roc_auc", "pr_auc", "brier_score", "precision", "recall", "f1", "threshold"
]])

wide = ablation_metrics.pivot_table(
    index=["model_family", "model_label", "split"],
    columns="feature_set",
    values=["roc_auc", "pr_auc", "f1", "precision", "recall"],
    aggfunc="first",
)
wide.columns = [f"{metric}_{feature_set}" for metric, feature_set in wide.columns]
wide = wide.reset_index()

for metric in ["roc_auc", "pr_auc", "f1", "precision", "recall"]:
    wide[f"{metric}_delta"] = wide[f"{metric}_baseline_no_grade_subgrade"] - wide[f"{metric}_baseline_with_grade_subgrade"]

ablation_deltas = wide.sort_values(["split", "model_label"])
save_table(ablation_deltas, "grade_subgrade_ablation_deltas")
display(ablation_deltas)

compact_cols = [
    "model_label", "split",
    "roc_auc_baseline_with_grade_subgrade", "roc_auc_baseline_no_grade_subgrade", "roc_auc_delta",
    "pr_auc_baseline_with_grade_subgrade", "pr_auc_baseline_no_grade_subgrade", "pr_auc_delta",
    "f1_baseline_with_grade_subgrade", "f1_baseline_no_grade_subgrade", "f1_delta",
]
compact = ablation_deltas[compact_cols].copy()
compact = compact.rename(columns={"model_label": "model"})
compact["with_grade_label"] = compact["model"] + " | with grade/subgrade"
compact["no_grade_label"] = compact["model"] + " | no grade/subgrade"
save_table(compact, "grade_subgrade_ablation_compact")
display(compact[[
    "model", "split", "with_grade_label", "no_grade_label",
    "roc_auc_baseline_with_grade_subgrade", "roc_auc_baseline_no_grade_subgrade", "roc_auc_delta",
    "pr_auc_baseline_with_grade_subgrade", "pr_auc_baseline_no_grade_subgrade", "pr_auc_delta",
    "f1_baseline_with_grade_subgrade", "f1_baseline_no_grade_subgrade", "f1_delta",
]])

if NO_GRADE_EXPORT_SUMMARY_PATH.exists():
    no_grade_export_summary = pd.read_csv(NO_GRADE_EXPORT_SUMMARY_PATH)
    save_table(no_grade_export_summary, "grade_subgrade_ablation_feature_summary")
    display(no_grade_export_summary)

ablation_families = [family for family in ABLATION_MODEL_FAMILIES if family in with_grade_metrics["model_family"].unique()]
for metric in ["f1", "pr_auc", "roc_auc", "precision", "recall"]:
    plot_df = ablation_metrics[ablation_metrics["split"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = [MODEL_LABELS[m] for m in ablation_families]
    x = np.arange(len(labels))
    width = 0.35
    metric_by_family = plot_df.set_index(["model_family", "feature_set"])[metric]
    with_grade = [metric_by_family.get((m, "baseline_with_grade_subgrade"), np.nan) for m in ablation_families]
    no_grade = [metric_by_family.get((m, "baseline_no_grade_subgrade"), np.nan) for m in ablation_families]
    ax.bar(x - width/2, with_grade, width, label="with grade/subgrade")
    ax.bar(x + width/2, no_grade, width, label="without grade/subgrade")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_title(f"Grade/subgrade ablation on test: {metric}")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_plot(fig, f"grade_subgrade_ablation_test_{metric}")
""".strip()
    ),
    new_markdown_cell("## 6. Recommendation"),
    new_code_cell(
        """
preferred_operating_point = "best_validation_f1"

validation_rank = (
    selected_metrics[
        (selected_metrics["split"] == "validation") &
        (selected_metrics["operating_point"] == preferred_operating_point)
    ]
    .sort_values(["f1", "precision", "pr_auc", "roc_auc"], ascending=False)
    .reset_index(drop=True)
)

preferred_validation = validation_rank.iloc[0]
preferred_model_family = preferred_validation["model_family"]
preferred_candidate = preferred_validation["model"]
preferred_feature_set = "baseline_with_grade_subgrade"

preferred_test = selected_metrics[
    (selected_metrics["model_family"] == preferred_model_family) &
    (selected_metrics["model"] == preferred_candidate) &
    (selected_metrics["split"] == "test") &
    (selected_metrics["operating_point"] == preferred_operating_point)
].iloc[0]

winner_confusion_matrix = read_model_table(preferred_model_family, "confusion_matrix")
recommended_predicted_label_counts = (
    winner_confusion_matrix[
        (winner_confusion_matrix["candidate"] == preferred_candidate) &
        (winner_confusion_matrix["split"].isin(["validation", "test"])) &
        (winner_confusion_matrix["operating_point"] == preferred_operating_point)
    ]
    .groupby([
        "model_family", "candidate", "split", "operating_point", "threshold",
        "predicted_label", "predicted_class"
    ], as_index=False)["count"]
    .sum()
    .rename(columns={"count": "predicted_count"})
    .assign(split_order=lambda df: df["split"].map({"validation": 0, "test": 1}))
    .sort_values(["split_order", "predicted_label"])
    .drop(columns="split_order")
)
recommended_predicted_label_counts["predicted_share"] = (
    recommended_predicted_label_counts["predicted_count"] /
    recommended_predicted_label_counts.groupby("split")["predicted_count"].transform("sum")
).round(6)

def predicted_value(split: str, predicted_label: int, column: str) -> float:
    match = recommended_predicted_label_counts[
        (recommended_predicted_label_counts["split"] == split) &
        (recommended_predicted_label_counts["predicted_label"] == predicted_label)
    ]
    if match.empty:
        return np.nan
    value = match.iloc[0][column]
    return int(value) if column == "predicted_count" else float(value)

recommendation = pd.DataFrame([{
    "recommended_model_family": preferred_model_family,
    "recommended_model_label": MODEL_LABELS[preferred_model_family],
    "recommended_candidate": preferred_candidate,
    "recommended_feature_set": preferred_feature_set,
    "recommended_operating_point": preferred_operating_point,
    "selection_basis": (
        "Select the available model with the highest validation F1 at the best-validation-F1 "
        "operating point, using validation precision, PR-AUC, and ROC-AUC as tie breakers. "
        "Test metrics are reported after selection and are not used to choose the winner."
    ),
    "validation_f1": preferred_validation["f1"],
    "validation_precision": preferred_validation["precision"],
    "validation_recall": preferred_validation["recall"],
    "validation_pr_auc": preferred_validation["pr_auc"],
    "validation_roc_auc": preferred_validation["roc_auc"],
    "validation_predicted_0_count": predicted_value("validation", 0, "predicted_count"),
    "validation_predicted_0_share": predicted_value("validation", 0, "predicted_share"),
    "validation_predicted_1_count": predicted_value("validation", 1, "predicted_count"),
    "validation_predicted_1_share": predicted_value("validation", 1, "predicted_share"),
    "test_f1": preferred_test["f1"],
    "test_precision": preferred_test["precision"],
    "test_recall": preferred_test["recall"],
    "test_pr_auc": preferred_test["pr_auc"],
    "test_roc_auc": preferred_test["roc_auc"],
    "test_predicted_0_count": predicted_value("test", 0, "predicted_count"),
    "test_predicted_0_share": predicted_value("test", 0, "predicted_share"),
    "test_predicted_1_count": predicted_value("test", 1, "predicted_count"),
    "test_predicted_1_share": predicted_value("test", 1, "predicted_share"),
}])

save_table(validation_rank, "ranking_by_validation_f1")
save_table(recommended_predicted_label_counts, "predicted_label_counts")
save_table(recommendation, "recommendation")
display(validation_rank)
display(recommended_predicted_label_counts)
display(recommendation)

feature_set_rank = (
    ablation_metrics[
        (ablation_metrics["split"] == "validation")
        & (ablation_metrics["feature_set"].isin(["baseline_with_grade_subgrade", "baseline_no_grade_subgrade"]))
    ]
    .sort_values(["f1", "precision", "pr_auc", "roc_auc"], ascending=False)
    .reset_index(drop=True)
)
feature_set_rank["selection_rank"] = feature_set_rank.index + 1
feature_set_rank = feature_set_rank[[
    "selection_rank", "model_feature_label", "model_label", "model", "feature_set",
    "feature_set_label", "split", "f1", "precision", "recall", "pr_auc", "roc_auc"
]]
save_table(feature_set_rank, "ranking_by_validation_f1_with_feature_set")
display(feature_set_rank)

recommended_feature_set_row = feature_set_rank.iloc[0]
recommendation_with_feature_set = recommendation.copy()
recommendation_with_feature_set["recommended_model_feature_label"] = recommended_feature_set_row["model_feature_label"]
recommendation_with_feature_set["recommended_candidate_with_feature_set"] = recommended_feature_set_row["model"]
recommendation_with_feature_set["recommended_feature_set"] = recommended_feature_set_row["feature_set"]
recommendation_with_feature_set["recommended_feature_set_label"] = recommended_feature_set_row["feature_set_label"]
recommendation_with_feature_set["recommended_validation_f1_with_feature_set_rank"] = recommended_feature_set_row["f1"]
save_table(recommendation_with_feature_set, "recommendation_with_feature_set_label")
display(recommendation_with_feature_set[[
    "recommended_model_feature_label", "recommended_model_label", "recommended_candidate_with_feature_set",
    "recommended_feature_set", "recommended_feature_set_label", "recommended_operating_point",
    "recommended_validation_f1_with_feature_set_rank", "validation_f1", "test_f1",
]])

per_class_tables = []
for model_family in available_families:
    per_class_tables.append(read_model_table(model_family, "per_class_metrics"))
all_model_per_class_metrics = pd.concat(per_class_tables, ignore_index=True)
all_model_per_class_metrics["model_label"] = all_model_per_class_metrics["model_family"].map(MODEL_LABELS)
save_table(all_model_per_class_metrics, "per_class_metrics")
display(all_model_per_class_metrics)
""".strip()
    ),
    new_markdown_cell(
        "## 7. PR-AUC Optimized Advanced Models\n\n"
        "Compare the separate PR-AUC-monitored LightGBM, XGBoost, and CatBoost experiment against the current F1-selected recommendation."
    ),
    new_code_cell(
        """
PR_AUC_OUTPUT_ROOT = MODELING_OUTPUT_ROOT / "pr_auc_optimized" / "tables"
PR_AUC_RANKING_PATH = PR_AUC_OUTPUT_ROOT / "pr_auc_optimized_ranking_by_validation_pr_auc.csv"
PR_AUC_TEST_RANKING_PATH = PR_AUC_OUTPUT_ROOT / "pr_auc_optimized_test_ranking.csv"
PR_AUC_RECOMMENDATION_PATH = PR_AUC_OUTPUT_ROOT / "pr_auc_optimized_recommendation.csv"

missing_pr_auc_outputs = [
    path for path in [PR_AUC_RANKING_PATH, PR_AUC_TEST_RANKING_PATH, PR_AUC_RECOMMENDATION_PATH]
    if not path.exists()
]
if missing_pr_auc_outputs:
    raise FileNotFoundError(
        "Missing PR-AUC optimized outputs. Run scripts/run_pr_auc_optimized_advanced_models.py first: "
        + ", ".join(str(path) for path in missing_pr_auc_outputs)
    )

pr_auc_validation_rank = pd.read_csv(PR_AUC_RANKING_PATH)
pr_auc_test_rank = pd.read_csv(PR_AUC_TEST_RANKING_PATH)
pr_auc_recommendation = pd.read_csv(PR_AUC_RECOMMENDATION_PATH)

save_table(pr_auc_validation_rank, "pr_auc_optimized_ranking_by_validation_pr_auc")
save_table(pr_auc_test_rank, "pr_auc_optimized_test_ranking")
save_table(pr_auc_recommendation, "pr_auc_optimized_recommendation")

current_rec = recommendation_with_feature_set.iloc[0]
pr_auc_rec = pr_auc_recommendation.iloc[0]

selection_strategy_comparison = pd.DataFrame([
    {
        "selection_strategy": "F1-selected current final comparison",
        "recommended_model_feature_label": current_rec["recommended_model_feature_label"],
        "recommended_candidate": current_rec["recommended_candidate_with_feature_set"],
        "recommended_feature_set": current_rec["recommended_feature_set"],
        "validation_pr_auc": current_rec["validation_pr_auc"],
        "validation_f1": current_rec["validation_f1"],
        "validation_precision": current_rec["validation_precision"],
        "validation_recall": current_rec["validation_recall"],
        "validation_predicted_reject_share": current_rec["validation_predicted_1_share"],
        "test_pr_auc": current_rec["test_pr_auc"],
        "test_f1": current_rec["test_f1"],
        "test_precision": current_rec["test_precision"],
        "test_recall": current_rec["test_recall"],
        "test_predicted_reject_share": current_rec["test_predicted_1_share"],
    },
    {
        "selection_strategy": "PR-AUC-selected advanced experiment",
        "recommended_model_feature_label": pr_auc_rec["recommended_model_feature_label"],
        "recommended_candidate": pr_auc_rec["recommended_candidate"],
        "recommended_feature_set": pr_auc_rec["recommended_feature_set"],
        "validation_pr_auc": pr_auc_rec["validation_pr_auc"],
        "validation_f1": pr_auc_rec["validation_f1"],
        "validation_precision": pr_auc_rec["validation_precision"],
        "validation_recall": pr_auc_rec["validation_recall"],
        "validation_predicted_reject_share": pr_auc_rec["validation_predicted_reject_share"],
        "test_pr_auc": pr_auc_rec["test_pr_auc"],
        "test_f1": pr_auc_rec["test_f1"],
        "test_precision": pr_auc_rec["test_precision"],
        "test_recall": pr_auc_rec["test_recall"],
        "test_predicted_reject_share": pr_auc_rec["test_predicted_reject_share"],
    },
])
save_table(selection_strategy_comparison, "selection_strategy_comparison")

display(pr_auc_validation_rank[[
    "selection_rank", "model_feature_label", "model", "feature_set_label",
    "pr_auc", "roc_auc", "f1", "precision", "recall",
    "predicted_reject_share", "false_rejection_share_among_rejects"
]])
display(selection_strategy_comparison)
""".strip()
    ),
    new_markdown_cell(
        "## 8. Missingness Challenger Dataset Comparison\n\n"
        "Compare all six model families across baseline with grade/subgrade, baseline no grade/subgrade, and missingness challenger datasets."
    ),
    new_code_cell(
        """
MISSINGNESS_OUTPUT_ROOT = MODELING_OUTPUT_ROOT / "missingness_challenger" / "tables"
MISSINGNESS_COMPARISON_PATH = MISSINGNESS_OUTPUT_ROOT / "missingness_challenger_validation_test_dataset_comparison.csv"
MISSINGNESS_RANKING_PATH = MISSINGNESS_OUTPUT_ROOT / "missingness_challenger_ranking_by_validation_f1.csv"
MISSINGNESS_BEST_BY_MODEL_PATH = MISSINGNESS_OUTPUT_ROOT / "missingness_challenger_best_dataset_by_model_family.csv"
MISSINGNESS_RECOMMENDATION_PATH = MISSINGNESS_OUTPUT_ROOT / "missingness_challenger_recommendation.csv"
MISSINGNESS_NO_GRADE_OUTPUT_ROOT = MODELING_OUTPUT_ROOT / "missingness_challenger_no_grade_subgrade" / "tables"
MISSINGNESS_NO_GRADE_COMPARISON_PATH = MISSINGNESS_NO_GRADE_OUTPUT_ROOT / "missingness_challenger_no_grade_subgrade_validation_test_comparison.csv"

missing_missingness_outputs = [
    path for path in [
        MISSINGNESS_COMPARISON_PATH,
        MISSINGNESS_RANKING_PATH,
        MISSINGNESS_BEST_BY_MODEL_PATH,
        MISSINGNESS_RECOMMENDATION_PATH,
        MISSINGNESS_NO_GRADE_COMPARISON_PATH,
    ]
    if not path.exists()
]
if missing_missingness_outputs:
    raise FileNotFoundError(
        "Missing missingness challenger outputs. Run scripts/run_missingness_challenger_all_models.py first: "
        + ", ".join(str(path) for path in missing_missingness_outputs)
    )

missingness_dataset_comparison = pd.read_csv(MISSINGNESS_COMPARISON_PATH)
missingness_no_grade_comparison = pd.read_csv(MISSINGNESS_NO_GRADE_COMPARISON_PATH)
missingness_dataset_comparison = pd.concat(
    [missingness_dataset_comparison, missingness_no_grade_comparison],
    ignore_index=True,
    sort=False,
)
missingness_dataset_comparison = missingness_dataset_comparison.drop_duplicates(
    subset=["model_family", "model", "dataset", "split", "operating_point"],
    keep="last",
)

missingness_validation_rank = (
    missingness_dataset_comparison[
        (missingness_dataset_comparison["split"] == "validation")
        & (missingness_dataset_comparison["operating_point"] == "best_validation_f1")
    ]
    .sort_values(["f1", "precision", "pr_auc"], ascending=False)
    .reset_index(drop=True)
)
missingness_validation_rank["selection_rank"] = missingness_validation_rank.index + 1

missingness_validation_pr_auc_rank = (
    missingness_dataset_comparison[
        (missingness_dataset_comparison["split"] == "validation")
        & (missingness_dataset_comparison["operating_point"] == "best_validation_f1")
    ]
    .sort_values(["pr_auc", "roc_auc", "f1"], ascending=False)
    .reset_index(drop=True)
)
missingness_validation_pr_auc_rank["selection_rank"] = missingness_validation_pr_auc_rank.index + 1

missingness_best_by_model = (
    missingness_validation_rank
    .sort_values(["model_family", "selection_rank"])
    .groupby("model_family", as_index=False)
    .first()
)
missingness_recommendation = missingness_validation_rank.iloc[[0]].copy()
missingness_pr_auc_recommendation = missingness_validation_pr_auc_rank.iloc[[0]].copy()

save_table(missingness_dataset_comparison, "missingness_challenger_dataset_comparison")
save_table(missingness_validation_rank, "missingness_challenger_ranking_by_validation_f1")
save_table(missingness_validation_pr_auc_rank, "missingness_challenger_ranking_by_validation_pr_auc")
save_table(missingness_best_by_model, "missingness_challenger_best_dataset_by_model_family")
save_table(missingness_recommendation, "missingness_challenger_recommendation")
save_table(missingness_pr_auc_recommendation, "missingness_challenger_pr_auc_recommendation")

display(missingness_validation_rank[[
    "selection_rank", "model_dataset_label", "model", "dataset_label",
    "f1", "precision", "recall", "pr_auc", "roc_auc", "predicted_reject_share"
]].head(18))

display(missingness_best_by_model[[
    "model_label", "dataset_label", "model", "f1", "precision", "recall",
    "pr_auc", "roc_auc", "predicted_reject_share"
]])

display(missingness_recommendation[[
    "model_dataset_label", "model", "dataset_label", "f1", "precision",
    "recall", "pr_auc", "roc_auc", "predicted_reject_share"
]])

display(missingness_validation_pr_auc_rank[[
    "selection_rank", "model_dataset_label", "model", "dataset_label",
    "pr_auc", "roc_auc", "f1", "precision", "recall", "predicted_reject_share"
]].head(18))

display(missingness_pr_auc_recommendation[[
    "model_dataset_label", "model", "dataset_label", "pr_auc", "roc_auc",
    "f1", "precision", "recall", "predicted_reject_share"
]])
""".strip()
    ),
    new_markdown_cell(
        "## 9. Operating Policy Warning\n\n"
        "PR-AUC ranks models without choosing an action threshold. Reject-share values from the best-F1 threshold should not be interpreted as an automatic rejection policy."
    ),
    new_code_cell(
        """
POLICY_WARNING_PATH = TABLE_DIR / "final_model_pr_auc_ranking_with_f1_threshold_warning.csv"
FIXED_POLICY_PATH = TABLE_DIR / "final_model_top_pr_auc_fixed_review_volume_policy.csv"
FIXED_POLICY_MISSING_PATH = TABLE_DIR / "final_model_top_pr_auc_fixed_review_volume_missing.csv"

missing_policy_outputs = [
    path for path in [POLICY_WARNING_PATH, FIXED_POLICY_PATH, FIXED_POLICY_MISSING_PATH]
    if not path.exists()
]
if missing_policy_outputs:
    raise FileNotFoundError(
        "Missing operating policy outputs. Run scripts/create_operating_policy_analysis.py first: "
        + ", ".join(str(path) for path in missing_policy_outputs)
    )

pr_auc_ranking_with_warning = pd.read_csv(POLICY_WARNING_PATH)
fixed_review_policy = pd.read_csv(FIXED_POLICY_PATH)
fixed_review_policy_missing = pd.read_csv(FIXED_POLICY_MISSING_PATH)

display(pr_auc_ranking_with_warning[[
    "selection_rank", "model_dataset_label", "split", "pr_auc", "roc_auc",
    "f1_threshold_f1", "f1_threshold_precision", "f1_threshold_recall",
    "f1_threshold_predicted_reject_share", "operating_policy_warning"
]].head(12))

display(fixed_review_policy[[
    "selection_rank", "model_dataset_label", "split", "review_pct", "business_policy",
    "review_count", "captured_bad", "precision", "recall", "base_bad_rate",
    "lift_over_base_bad_rate", "avoided_auto_reject_share_vs_f1_threshold"
]].head(48))

if not fixed_review_policy_missing.empty:
    display(fixed_review_policy_missing)
""".strip()
    ),
    new_markdown_cell(
        "## 10. Calibration Check\n\n"
        "Check whether predicted default probabilities from the PR-AUC recommended ranker align with observed default rates."
    ),
    new_code_cell(
        """
CALIBRATION_BINS_PATH = TABLE_DIR / "final_model_calibration_bins.csv"
CALIBRATION_SUMMARY_PATH = TABLE_DIR / "final_model_calibration_summary.csv"

missing_calibration_outputs = [
    path for path in [CALIBRATION_BINS_PATH, CALIBRATION_SUMMARY_PATH]
    if not path.exists()
]
if missing_calibration_outputs:
    raise FileNotFoundError(
        "Missing calibration outputs. Run scripts/create_calibration_analysis.py first: "
        + ", ".join(str(path) for path in missing_calibration_outputs)
    )

calibration_bins = pd.read_csv(CALIBRATION_BINS_PATH)
calibration_summary = pd.read_csv(CALIBRATION_SUMMARY_PATH)

display(calibration_summary)
display(calibration_bins[[
    "calibration_method", "split", "probability_bin", "rows", "bad_count",
    "predicted_probability_mean", "observed_bad_rate", "abs_calibration_error"
]])

around_45 = calibration_bins[calibration_bins["probability_bin"] == "(0.4, 0.5]"].copy()
around_45["calibration_interpretation"] = np.where(
    around_45["calibration_method"].eq("raw"),
    "Raw probabilities near 0.45 materially overstate observed default risk; use raw scores for ranking, not probability decisions.",
    "Calibrated probabilities are closer to observed default risk and are more defensible for probability-based policy thresholds.",
)
display(around_45[[
    "calibration_method", "split", "probability_bin", "rows", "predicted_probability_mean",
    "observed_bad_rate", "abs_calibration_error", "calibration_interpretation"
]])
""".strip()
    ),
]


def write_final_comparison_notebook() -> None:
    base_nb = json.loads(BASE_NOTEBOOK_PATH.read_text())
    final_nb = {
        "cells": FINAL_NOTEBOOK_CELLS,
        "metadata": base_nb.get("metadata", {}),
        "nbformat": base_nb.get("nbformat", 4),
        "nbformat_minor": base_nb.get("nbformat_minor", 5),
    }
    output_path = MODELING_DIR / "Accepted_Loan_Final_Model_Comparison.ipynb"
    output_path.write_text(json.dumps(final_nb, indent=1) + "\n")
    print(f"Wrote {output_path}")


def fix_future_import_order() -> None:
    for path in MODELING_DIR.glob("Accepted_Loan_*_Modeling.ipynb"):
        nb = json.loads(path.read_text())
        changed = False
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            future_line = "from __future__ import annotations"
            if future_line not in source:
                continue
            lines = source.splitlines()
            if lines and lines[0] == future_line:
                continue
            lines = [line for line in lines if line != future_line]
            while lines and lines[0] == "":
                lines.pop(0)
            cell["source"] = source_lines(future_line + "\n\n" + "\n".join(lines))
            changed = True
        if changed:
            path.write_text(json.dumps(nb, indent=1) + "\n")
            print(f"Fixed future import order in {path}")


if __name__ == "__main__":
    write_model_notebooks()
    write_final_comparison_notebook()
    fix_future_import_order()
