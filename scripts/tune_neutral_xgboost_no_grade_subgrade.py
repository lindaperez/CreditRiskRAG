from __future__ import annotations

from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "Modeling" / "Preprocessing" / "preprocessing_outputs" / "datasets"
MODELING_OUTPUT_ROOT = PROJECT_ROOT / "Modeling" / "modeling_outputs"
OUTPUT_ROOT = MODELING_OUTPUT_ROOT / "xgboost_neutral_no_grade_subgrade"
TABLE_DIR = OUTPUT_ROOT / "tables"
MODEL_DIR = OUTPUT_ROOT / "models"
FINAL_TABLE_DIR = MODELING_OUTPUT_ROOT / "final_comparison" / "tables"
for directory in [TABLE_DIR, MODEL_DIR, FINAL_TABLE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
FIT_ROWS = 300_000
DATASET = "missingness_challenger_no_grade_subgrade"
DATASET_LABEL = "missingness challenger no grade/subgrade"
MODEL_DATASET_LABEL = "XGBoost neutral | missingness challenger no grade/subgrade"
FINAL_OUTPUT_PREFIX = "neutral_xgboost_no_grade_subgrade"
DEFAULT_DEFAULT_LOSS_SAVED = 10_000.0
DEFAULT_GOOD_BORROWER_OPPORTUNITY_COST = 1_500.0
DEFAULT_MAX_REJECT_SHARE = 0.20

PARAM_GRID = [
    {"candidate": "xgb_neutral_01", "params": {"n_estimators": 350, "learning_rate": 0.025, "max_depth": 6, "min_child_weight": 16, "subsample": 0.80, "colsample_bytree": 0.75, "reg_lambda": 5.0, "reg_alpha": 0.2}},
    {"candidate": "xgb_neutral_02", "params": {"n_estimators": 500, "learning_rate": 0.020, "max_depth": 5, "min_child_weight": 16, "subsample": 0.85, "colsample_bytree": 0.80, "reg_lambda": 6.0, "reg_alpha": 0.2}},
    {"candidate": "xgb_neutral_03", "params": {"n_estimators": 550, "learning_rate": 0.018, "max_depth": 4, "min_child_weight": 24, "subsample": 0.90, "colsample_bytree": 0.85, "reg_lambda": 8.0, "reg_alpha": 0.1}},
    {"candidate": "xgb_neutral_04", "params": {"n_estimators": 450, "learning_rate": 0.025, "max_depth": 4, "min_child_weight": 16, "subsample": 0.85, "colsample_bytree": 0.90, "reg_lambda": 5.0, "reg_alpha": 0.0}},
    {"candidate": "xgb_neutral_05", "params": {"n_estimators": 700, "learning_rate": 0.015, "max_depth": 3, "min_child_weight": 24, "subsample": 0.90, "colsample_bytree": 0.90, "reg_lambda": 8.0, "reg_alpha": 0.0}},
    {"candidate": "xgb_neutral_06", "params": {"n_estimators": 400, "learning_rate": 0.030, "max_depth": 5, "min_child_weight": 24, "subsample": 0.80, "colsample_bytree": 0.80, "reg_lambda": 8.0, "reg_alpha": 0.3}},
    {"candidate": "xgb_neutral_07", "params": {"n_estimators": 300, "learning_rate": 0.035, "max_depth": 4, "min_child_weight": 32, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 10.0, "reg_alpha": 0.3}},
    {"candidate": "xgb_neutral_08", "params": {"n_estimators": 600, "learning_rate": 0.018, "max_depth": 5, "min_child_weight": 32, "subsample": 0.75, "colsample_bytree": 0.75, "reg_lambda": 12.0, "reg_alpha": 0.5}},
    {"candidate": "xgb_neutral_09", "params": {"n_estimators": 450, "learning_rate": 0.020, "max_depth": 6, "min_child_weight": 24, "subsample": 0.75, "colsample_bytree": 0.70, "reg_lambda": 10.0, "reg_alpha": 0.5}},
    {"candidate": "xgb_neutral_10", "params": {"n_estimators": 250, "learning_rate": 0.050, "max_depth": 3, "min_child_weight": 16, "subsample": 0.90, "colsample_bytree": 0.90, "reg_lambda": 4.0, "reg_alpha": 0.0}},
    {"candidate": "xgb_neutral_11", "params": {"n_estimators": 500, "learning_rate": 0.022, "max_depth": 4, "min_child_weight": 8, "subsample": 0.80, "colsample_bytree": 0.85, "reg_lambda": 6.0, "reg_alpha": 0.1}},
    {"candidate": "xgb_neutral_12", "params": {"n_estimators": 650, "learning_rate": 0.016, "max_depth": 4, "min_child_weight": 32, "subsample": 1.00, "colsample_bytree": 0.80, "reg_lambda": 12.0, "reg_alpha": 0.2}},
]


def load_parquet(name: str) -> pd.DataFrame:
    path = DATASET_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def save_table(df: pd.DataFrame, directory: Path, name: str) -> Path:
    path = directory / f"{name}.csv"
    df.to_csv(path, index=False)
    print("Saved:", path, flush=True)
    return path


def sample_fit_data(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    sampled_index = X.sample(n=min(FIT_ROWS, len(X)), random_state=RANDOM_STATE).index
    return X.loc[sampled_index], y.loc[sampled_index]


def build_model(params: dict) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        scale_pos_weight=1.0,
        random_state=RANDOM_STATE,
        n_jobs=1,
        **params,
    )


def predict_score(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def threshold_for_best_f1(y_true: pd.Series, score: np.ndarray) -> tuple[float, float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    idx = int(np.nanargmax(f1))
    return float(thresholds[idx]), float(f1[idx]), float(precision[idx]), float(recall[idx])


def review_at_top_pct(y_true: pd.Series, score: np.ndarray, pct: float) -> dict:
    threshold = float(np.quantile(score, 1 - pct))
    flagged = score >= threshold
    y = np.asarray(y_true)
    captured_bad = int(((flagged == 1) & (y == 1)).sum())
    flagged_count = int(flagged.sum())
    return {
        "review_pct": round(pct * 100, 2),
        "review_threshold": round(threshold, 6),
        "review_count": flagged_count,
        "captured_bad": captured_bad,
        "precision": round(float(captured_bad / flagged_count), 6),
        "recall": round(float(captured_bad / y.sum()), 6),
    }


def evaluate_split(model, candidate: str, split: str, X: pd.DataFrame, y: pd.Series) -> dict:
    score = predict_score(model, X)
    threshold, f1, precision, recall = threshold_for_best_f1(y, score)
    top20 = review_at_top_pct(y, score, DEFAULT_MAX_REJECT_SHARE)
    return {
        "model_family": "xgboost",
        "candidate": candidate,
        "dataset": DATASET,
        "dataset_label": DATASET_LABEL,
        "model_dataset_label": MODEL_DATASET_LABEL,
        "split": split,
        "rows": len(y),
        "bad_rate": round(float(y.mean()), 6),
        "roc_auc": round(float(roc_auc_score(y, score)), 6),
        "pr_auc": round(float(average_precision_score(y, score)), 6),
        "mean_predicted_probability_raw": round(float(score.mean()), 6),
        "best_f1_threshold": round(threshold, 6),
        "best_f1": round(f1, 6),
        "best_f1_precision": round(precision, 6),
        "best_f1_recall": round(recall, 6),
        "best_f1_predicted_reject_share": round(float((score >= threshold).mean()), 6),
        **top20,
    }


def calibrate_scores(raw_validation: np.ndarray, y_validation: pd.Series, raw_test: np.ndarray) -> dict[tuple[str, str], np.ndarray]:
    platt = LogisticRegression(solver="lbfgs", random_state=RANDOM_STATE)
    platt.fit(raw_validation.reshape(-1, 1), y_validation)
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(raw_validation, y_validation)
    return {
        ("raw", "validation"): raw_validation,
        ("raw", "test"): raw_test,
        ("platt_sigmoid", "validation"): platt.predict_proba(raw_validation.reshape(-1, 1))[:, 1],
        ("platt_sigmoid", "test"): platt.predict_proba(raw_test.reshape(-1, 1))[:, 1],
        ("isotonic", "validation"): isotonic.predict(raw_validation),
        ("isotonic", "test"): isotonic.predict(raw_test),
    }


def calibration_summary(scores: dict[tuple[str, str], np.ndarray], y_by_split: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for (method, split), score in scores.items():
        y = y_by_split[split]
        rows.append({
            "model_dataset_label": MODEL_DATASET_LABEL,
            "calibration_method": method,
            "split": split,
            "rows": len(y),
            "bad_rate": round(float(y.mean()), 6),
            "mean_predicted_probability": round(float(score.mean()), 6),
            "brier_score": round(float(brier_score_loss(y, score)), 6),
            "pr_auc": round(float(average_precision_score(y, score)), 6),
            "roc_auc": round(float(roc_auc_score(y, score)), 6),
        })
    return pd.DataFrame(rows)


def economic_policy(scores: dict[tuple[str, str], np.ndarray], y_by_split: dict[str, pd.Series]) -> pd.DataFrame:
    threshold = float(np.quantile(scores[("platt_sigmoid", "validation")], 1 - DEFAULT_MAX_REJECT_SHARE))
    rows = []
    for split in ["validation", "test"]:
        score = scores[("platt_sigmoid", split)]
        y = np.asarray(y_by_split[split]).astype(int)
        reject = score >= threshold
        tp = int(((reject == 1) & (y == 1)).sum())
        fp = int(((reject == 1) & (y == 0)).sum())
        fn = int(((reject == 0) & (y == 1)).sum())
        reject_count = int(reject.sum())
        approved_count = int(len(y) - reject_count)
        total_value = tp * DEFAULT_DEFAULT_LOSS_SAVED - fp * DEFAULT_GOOD_BORROWER_OPPORTUNITY_COST
        rows.append({
            "model_dataset_label": MODEL_DATASET_LABEL,
            "policy_type": "economic_max_20pct_reject",
            "calibration_method": "platt_sigmoid",
            "split": split,
            "threshold": round(threshold, 6),
            "predicted_reject_share": round(float(reject_count / len(y)), 6),
            "approved_share": round(float(approved_count / len(y)), 6),
            "precision_bad_rate_among_rejected": round(float(tp / reject_count), 6),
            "recall_default_capture": round(float(tp / y.sum()), 6),
            "approved_bad_rate": round(float(fn / approved_count), 6),
            "total_portfolio_value": round(float(total_value), 2),
            "value_per_applicant": round(float(total_value / len(y)), 2),
            "default_loss_saved": DEFAULT_DEFAULT_LOSS_SAVED,
            "good_borrower_opportunity_cost": DEFAULT_GOOD_BORROWER_OPPORTUNITY_COST,
            "max_reject_share_constraint": DEFAULT_MAX_REJECT_SHARE,
        })
    return pd.DataFrame(rows)


def main() -> None:
    X_train = load_parquet(f"{DATASET}_train_X")
    X_validation = load_parquet(f"{DATASET}_validation_X")
    X_test = load_parquet(f"{DATASET}_test_X")
    y_train = load_parquet("train_y")["target_bad"].astype(int)
    y_validation = load_parquet("validation_y")["target_bad"].astype(int)
    y_test = load_parquet("test_y")["target_bad"].astype(int)
    X_fit, y_fit = sample_fit_data(X_train, y_train)

    candidate_rows = []
    fitted_models = {}
    for item in PARAM_GRID:
        print(f"Training {item['candidate']}: {item['params']}", flush=True)
        start = time.perf_counter()
        model = build_model(item["params"])
        model.fit(X_fit, y_fit)
        seconds = time.perf_counter() - start
        score = predict_score(model, X_validation)
        threshold, f1, precision, recall = threshold_for_best_f1(y_validation, score)
        row = {
            "candidate": item["candidate"],
            "dataset": DATASET,
            "dataset_label": DATASET_LABEL,
            "model_dataset_label": MODEL_DATASET_LABEL,
            "params": json.dumps(item["params"], sort_keys=True),
            "scale_pos_weight": 1.0,
            "fit_rows": len(X_fit),
            "fit_bad_rate": round(float(y_fit.mean()), 6),
            "fit_seconds": round(float(seconds), 3),
            "roc_auc": round(float(roc_auc_score(y_validation, score)), 6),
            "pr_auc": round(float(average_precision_score(y_validation, score)), 6),
            "mean_predicted_probability_raw": round(float(score.mean()), 6),
            "best_f1_threshold": round(threshold, 6),
            "best_f1": round(f1, 6),
            "best_f1_precision": round(precision, 6),
            "best_f1_recall": round(recall, 6),
        }
        print(f"Finished {item['candidate']}: pr_auc={row['pr_auc']:.6f}, seconds={seconds:.1f}", flush=True)
        candidate_rows.append(row)
        fitted_models[item["candidate"]] = model

    candidates = pd.DataFrame(candidate_rows).sort_values(["pr_auc", "roc_auc"], ascending=False).reset_index(drop=True)
    winner = candidates.iloc[0]
    model = fitted_models[winner["candidate"]]
    model_path = MODEL_DIR / f"{winner['candidate']}_selected_model.joblib"
    joblib.dump(model, model_path)

    metrics = pd.DataFrame([
        evaluate_split(model, winner["candidate"], "validation", X_validation, y_validation),
        evaluate_split(model, winner["candidate"], "test", X_test, y_test),
    ])
    raw_scores = {
        "validation": predict_score(model, X_validation),
        "test": predict_score(model, X_test),
    }
    y_by_split = {"validation": y_validation, "test": y_test}
    calibrated_scores = calibrate_scores(raw_scores["validation"], y_validation, raw_scores["test"])
    calibration = calibration_summary(calibrated_scores, y_by_split)
    policy = economic_policy(calibrated_scores, y_by_split)

    save_table(candidates, TABLE_DIR, "neutral_xgboost_candidate_results")
    save_table(pd.DataFrame([winner.to_dict() | {"artifact_path": str(model_path)}]), TABLE_DIR, "neutral_xgboost_selected_candidate")
    save_table(metrics, TABLE_DIR, "neutral_xgboost_selected_model_metrics")
    save_table(calibration, TABLE_DIR, "neutral_xgboost_calibration_summary")
    save_table(policy, TABLE_DIR, "neutral_xgboost_economic_policy")

    save_table(candidates, FINAL_TABLE_DIR, f"{FINAL_OUTPUT_PREFIX}_candidate_results")
    save_table(metrics, FINAL_TABLE_DIR, f"{FINAL_OUTPUT_PREFIX}_selected_model_metrics")
    save_table(calibration, FINAL_TABLE_DIR, f"{FINAL_OUTPUT_PREFIX}_calibration_summary")
    save_table(policy, FINAL_TABLE_DIR, f"{FINAL_OUTPUT_PREFIX}_economic_policy")

    print("\nSelected candidate")
    print(winner.to_string())
    print("\nSelected metrics")
    print(metrics.to_string(index=False))
    print("\nEconomic policy")
    print(policy.to_string(index=False))


if __name__ == "__main__":
    main()
