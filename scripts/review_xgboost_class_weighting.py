from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "Modeling" / "Preprocessing" / "preprocessing_outputs" / "datasets"
MODELING_OUTPUT_ROOT = PROJECT_ROOT / "Modeling" / "modeling_outputs"
FINAL_TABLE_DIR = MODELING_OUTPUT_ROOT / "final_comparison" / "tables"
FINAL_TABLE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
FIT_ROWS = 300_000

XGBOOST_SELECTED_PARAMS = {
    "n_estimators": 350,
    "learning_rate": 0.025,
    "max_depth": 6,
    "min_child_weight": 16,
    "subsample": 0.80,
    "colsample_bytree": 0.75,
    "reg_lambda": 5.0,
    "reg_alpha": 0.2,
}

DATASETS = [
    {
        "dataset": "missingness_challenger",
        "dataset_label": "missingness challenger",
        "model_path": MODELING_OUTPUT_ROOT
        / "missingness_challenger"
        / "models"
        / "xgboost_05_missingness_challenger_selected_model.joblib",
    },
    {
        "dataset": "missingness_challenger_no_grade_subgrade",
        "dataset_label": "missingness challenger no grade/subgrade",
        "model_path": MODELING_OUTPUT_ROOT
        / "missingness_challenger_no_grade_subgrade"
        / "models"
        / "xgboost_05_missingness_challenger_no_grade_subgrade_selected_model.joblib",
    },
]


def load_parquet(name: str) -> pd.DataFrame:
    path = DATASET_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def save_table(df: pd.DataFrame, name: str) -> Path:
    path = FINAL_TABLE_DIR / f"final_model_{name}.csv"
    df.to_csv(path, index=False)
    print("Saved:", path)
    return path


def sample_fit_data(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    if len(X) <= FIT_ROWS:
        return X, y
    sampled_index = X.sample(n=FIT_ROWS, random_state=RANDOM_STATE).index
    return X.loc[sampled_index], y.loc[sampled_index]


def build_xgboost(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_estimators=XGBOOST_SELECTED_PARAMS["n_estimators"],
        learning_rate=XGBOOST_SELECTED_PARAMS["learning_rate"],
        max_depth=XGBOOST_SELECTED_PARAMS["max_depth"],
        min_child_weight=XGBOOST_SELECTED_PARAMS["min_child_weight"],
        subsample=XGBOOST_SELECTED_PARAMS["subsample"],
        colsample_bytree=XGBOOST_SELECTED_PARAMS["colsample_bytree"],
        reg_lambda=XGBOOST_SELECTED_PARAMS["reg_lambda"],
        reg_alpha=XGBOOST_SELECTED_PARAMS["reg_alpha"],
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def predict_positive_probability(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def best_f1_threshold(y_true: pd.Series, score: np.ndarray) -> tuple[float, float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    idx = int(np.nanargmax(f1))
    return float(thresholds[idx]), float(f1[idx]), float(precision[idx]), float(recall[idx])


def top_pct_metrics(y_true: pd.Series, score: np.ndarray, pct: float) -> dict:
    threshold = float(np.quantile(score, 1 - pct))
    flagged = score >= threshold
    y = np.asarray(y_true)
    captured_bad = int(((flagged == 1) & (y == 1)).sum())
    flagged_count = int(flagged.sum())
    return {
        "top_pct": pct,
        "top_pct_threshold": round(threshold, 6),
        "top_pct_reject_share": round(float(flagged_count / len(y)), 6),
        "top_pct_precision": round(float(captured_bad / flagged_count), 6) if flagged_count else np.nan,
        "top_pct_recall": round(float(captured_bad / y.sum()), 6) if y.sum() else np.nan,
    }


def evaluate_model(
    model,
    dataset: str,
    dataset_label: str,
    weighting_policy: str,
    scale_pos_weight: float,
    split: str,
    X: pd.DataFrame,
    y: pd.Series,
) -> dict:
    score = predict_positive_probability(model, X)
    threshold, f1, precision, recall = best_f1_threshold(y, score)
    top_20 = top_pct_metrics(y, score, 0.20)
    predicted_reject_share = float((score >= threshold).mean())
    row = {
        "model_family": "xgboost",
        "dataset": dataset,
        "dataset_label": dataset_label,
        "model_dataset_label": f"XGBoost | {dataset_label}",
        "weighting_policy": weighting_policy,
        "scale_pos_weight": round(float(scale_pos_weight), 6),
        "split": split,
        "rows": int(len(y)),
        "bad_rate": round(float(y.mean()), 6),
        "roc_auc": round(float(roc_auc_score(y, score)), 6),
        "pr_auc": round(float(average_precision_score(y, score)), 6),
        "mean_predicted_probability_raw": round(float(np.mean(score)), 6),
        "best_f1_threshold": round(threshold, 6),
        "best_f1": round(f1, 6),
        "best_f1_precision": round(precision, 6),
        "best_f1_recall": round(recall, 6),
        "best_f1_predicted_reject_share": round(predicted_reject_share, 6),
    }
    row.update(top_20)
    return row


def main() -> None:
    y_train = load_parquet("train_y")["target_bad"].astype(int)
    y_validation = load_parquet("validation_y")["target_bad"].astype(int)
    y_test = load_parquet("test_y")["target_bad"].astype(int)

    actual_train_ratio = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    rows = []

    for config in DATASETS:
        dataset = config["dataset"]
        dataset_label = config["dataset_label"]
        X_train = load_parquet(f"{dataset}_train_X")
        X_validation = load_parquet(f"{dataset}_validation_X")
        X_test = load_parquet(f"{dataset}_test_X")
        X_fit, y_fit = sample_fit_data(X_train, y_train)
        fit_ratio = float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1))

        weighted_model = joblib.load(config["model_path"])
        neutral_model = build_xgboost(scale_pos_weight=1.0)
        neutral_model.fit(X_fit, y_fit)

        for weighting_policy, model, spw in [
            ("current_balanced_scale_pos_weight", weighted_model, fit_ratio),
            ("neutral_scale_pos_weight_1", neutral_model, 1.0),
        ]:
            for split, X, y in [
                ("validation", X_validation, y_validation),
                ("test", X_test, y_test),
            ]:
                row = evaluate_model(model, dataset, dataset_label, weighting_policy, spw, split, X, y)
                row["full_train_neg_pos_ratio"] = round(actual_train_ratio, 6)
                row["fit_sample_neg_pos_ratio"] = round(fit_ratio, 6)
                rows.append(row)

    result = pd.DataFrame(rows)
    save_table(result, "xgboost_class_weighting_review")

    validation = result[result["split"].eq("validation")].copy()
    winner = validation.sort_values(["dataset", "pr_auc"], ascending=[True, False])
    save_table(winner, "xgboost_class_weighting_review_validation")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
