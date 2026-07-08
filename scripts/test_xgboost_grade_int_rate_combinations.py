from __future__ import annotations

from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "Modeling" / "Preprocessing" / "preprocessing_outputs" / "datasets"
MODELING_OUTPUT_ROOT = PROJECT_ROOT / "Modeling" / "modeling_outputs"
OUTPUT_ROOT = MODELING_OUTPUT_ROOT / "xgboost_grade_int_rate_ablation"
TABLE_DIR = OUTPUT_ROOT / "tables"
MODEL_DIR = OUTPUT_ROOT / "models"
FINAL_TABLE_DIR = MODELING_OUTPUT_ROOT / "final_comparison" / "tables"
for directory in [TABLE_DIR, MODEL_DIR, FINAL_TABLE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
FIT_ROWS = 300_000
BASE_DATASET = "missingness_challenger"
DEFAULT_MAX_REJECT_SHARE = 0.20
DEFAULT_DEFAULT_LOSS_SAVED = 10_000.0
DEFAULT_GOOD_BORROWER_OPPORTUNITY_COST = 1_500.0

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

FEATURE_COMBINATIONS = [
    {
        "feature_set": "with_grade_subgrade_with_int_rate",
        "feature_set_label": "with grade/sub_grade + int_rate",
        "include_grade_subgrade": True,
        "include_int_rate": True,
    },
    {
        "feature_set": "without_grade_subgrade_with_int_rate",
        "feature_set_label": "without grade/sub_grade + int_rate",
        "include_grade_subgrade": False,
        "include_int_rate": True,
    },
    {
        "feature_set": "with_grade_subgrade_without_int_rate",
        "feature_set_label": "with grade/sub_grade without int_rate",
        "include_grade_subgrade": True,
        "include_int_rate": False,
    },
    {
        "feature_set": "without_grade_subgrade_without_int_rate",
        "feature_set_label": "without grade/sub_grade and without int_rate",
        "include_grade_subgrade": False,
        "include_int_rate": False,
    },
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


def grade_subgrade_columns(columns: list[str]) -> list[str]:
    return [c for c in columns if c.startswith("grade_") or c.startswith("sub_grade_")]


def columns_for_combination(base_columns: list[str], combo: dict) -> tuple[list[str], list[str]]:
    dropped = []
    if not combo["include_grade_subgrade"]:
        dropped.extend(grade_subgrade_columns(base_columns))
    if not combo["include_int_rate"] and "int_rate_clean" in base_columns:
        dropped.append("int_rate_clean")
    dropped = sorted(set(dropped))
    kept = [c for c in base_columns if c not in set(dropped)]
    return kept, dropped


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


def top_pct_metrics(y_true: pd.Series, score: np.ndarray, pct: float = DEFAULT_MAX_REJECT_SHARE) -> dict:
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
        "review_precision": round(float(captured_bad / flagged_count), 6),
        "review_recall": round(float(captured_bad / y.sum()), 6),
    }


def evaluate_split(model, combo: dict, candidate: str, split: str, X: pd.DataFrame, y: pd.Series) -> dict:
    score = predict_score(model, X)
    threshold, f1, precision, recall = threshold_for_best_f1(y, score)
    return {
        "model_family": "xgboost",
        "candidate": candidate,
        "base_dataset": BASE_DATASET,
        "feature_set": combo["feature_set"],
        "feature_set_label": combo["feature_set_label"],
        "include_grade_subgrade": combo["include_grade_subgrade"],
        "include_int_rate": combo["include_int_rate"],
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
        **top_pct_metrics(y, score),
    }


def platt_scores(raw_validation: np.ndarray, y_validation: pd.Series, raw_test: np.ndarray) -> dict[str, np.ndarray]:
    platt = LogisticRegression(solver="lbfgs", random_state=RANDOM_STATE)
    platt.fit(raw_validation.reshape(-1, 1), y_validation)
    return {
        "validation": platt.predict_proba(raw_validation.reshape(-1, 1))[:, 1],
        "test": platt.predict_proba(raw_test.reshape(-1, 1))[:, 1],
    }


def economic_policy(combo: dict, candidate: str, calibrated_scores: dict[str, np.ndarray], y_by_split: dict[str, pd.Series]) -> pd.DataFrame:
    threshold = float(np.quantile(calibrated_scores["validation"], 1 - DEFAULT_MAX_REJECT_SHARE))
    rows = []
    for split in ["validation", "test"]:
        score = calibrated_scores[split]
        y = np.asarray(y_by_split[split]).astype(int)
        reject = score >= threshold
        tp = int(((reject == 1) & (y == 1)).sum())
        fp = int(((reject == 1) & (y == 0)).sum())
        fn = int(((reject == 0) & (y == 1)).sum())
        reject_count = int(reject.sum())
        approved_count = int(len(y) - reject_count)
        total_value = tp * DEFAULT_DEFAULT_LOSS_SAVED - fp * DEFAULT_GOOD_BORROWER_OPPORTUNITY_COST
        rows.append({
            "candidate": candidate,
            "base_dataset": BASE_DATASET,
            "feature_set": combo["feature_set"],
            "feature_set_label": combo["feature_set_label"],
            "include_grade_subgrade": combo["include_grade_subgrade"],
            "include_int_rate": combo["include_int_rate"],
            "policy_type": "platt_economic_max_20pct_reject",
            "split": split,
            "threshold": round(threshold, 6),
            "predicted_reject_share": round(float(reject_count / len(y)), 6),
            "approved_share": round(float(approved_count / len(y)), 6),
            "precision_bad_rate_among_rejected": round(float(tp / reject_count), 6),
            "recall_default_capture": round(float(tp / y.sum()), 6),
            "approved_bad_rate": round(float(fn / approved_count), 6),
            "total_portfolio_value": round(float(total_value), 2),
            "value_per_applicant": round(float(total_value / len(y)), 2),
        })
    return pd.DataFrame(rows)


def main() -> None:
    X_train_base = load_parquet(f"{BASE_DATASET}_train_X")
    X_validation_base = load_parquet(f"{BASE_DATASET}_validation_X")
    X_test_base = load_parquet(f"{BASE_DATASET}_test_X")
    y_train = load_parquet("train_y")["target_bad"].astype(int)
    y_validation = load_parquet("validation_y")["target_bad"].astype(int)
    y_test = load_parquet("test_y")["target_bad"].astype(int)
    base_columns = list(X_train_base.columns)

    candidate_rows = []
    selected_rows = []
    selected_metric_rows = []
    policy_frames = []
    feature_summary_rows = []

    for combo in FEATURE_COMBINATIONS:
        kept_columns, dropped_columns = columns_for_combination(base_columns, combo)
        print(
            f"\nFeature set: {combo['feature_set_label']} | kept={len(kept_columns)} dropped={len(dropped_columns)}",
            flush=True,
        )
        feature_summary_rows.append({
            **combo,
            "base_dataset": BASE_DATASET,
            "kept_feature_count": len(kept_columns),
            "dropped_feature_count": len(dropped_columns),
            "dropped_features": json.dumps(dropped_columns),
        })

        X_train = X_train_base[kept_columns]
        X_validation = X_validation_base[kept_columns]
        X_test = X_test_base[kept_columns]
        X_fit, y_fit = sample_fit_data(X_train, y_train)

        fitted_models = {}
        combo_candidate_rows = []
        for item in PARAM_GRID:
            candidate_name = f"{item['candidate']}_{combo['feature_set']}"
            print(f"Training {candidate_name}: {item['params']}", flush=True)
            start = time.perf_counter()
            model = build_model(item["params"])
            model.fit(X_fit, y_fit)
            seconds = time.perf_counter() - start
            score = predict_score(model, X_validation)
            threshold, f1, precision, recall = threshold_for_best_f1(y_validation, score)
            row = {
                "model_family": "xgboost",
                "candidate": candidate_name,
                "base_candidate": item["candidate"],
                "base_dataset": BASE_DATASET,
                "feature_set": combo["feature_set"],
                "feature_set_label": combo["feature_set_label"],
                "include_grade_subgrade": combo["include_grade_subgrade"],
                "include_int_rate": combo["include_int_rate"],
                "params": json.dumps(item["params"], sort_keys=True),
                "scale_pos_weight": 1.0,
                "fit_rows": len(X_fit),
                "fit_bad_rate": round(float(y_fit.mean()), 6),
                "fit_seconds": round(float(seconds), 3),
                "feature_count": len(kept_columns),
                "roc_auc": round(float(roc_auc_score(y_validation, score)), 6),
                "pr_auc": round(float(average_precision_score(y_validation, score)), 6),
                "mean_predicted_probability_raw": round(float(score.mean()), 6),
                "best_f1_threshold": round(threshold, 6),
                "best_f1": round(f1, 6),
                "best_f1_precision": round(precision, 6),
                "best_f1_recall": round(recall, 6),
            }
            print(f"Finished {candidate_name}: pr_auc={row['pr_auc']:.6f}, seconds={seconds:.1f}", flush=True)
            combo_candidate_rows.append(row)
            fitted_models[candidate_name] = model

        combo_candidates = pd.DataFrame(combo_candidate_rows).sort_values(["pr_auc", "roc_auc"], ascending=False)
        candidate_rows.extend(combo_candidates.to_dict("records"))
        winner = combo_candidates.iloc[0]
        selected_rows.append(winner.to_dict())
        model = fitted_models[winner["candidate"]]
        model_path = MODEL_DIR / f"{winner['candidate']}_selected_model.joblib"
        joblib.dump(model, model_path)

        selected_metric_rows.extend([
            evaluate_split(model, combo, winner["candidate"], "validation", X_validation, y_validation),
            evaluate_split(model, combo, winner["candidate"], "test", X_test, y_test),
        ])
        raw_validation = predict_score(model, X_validation)
        raw_test = predict_score(model, X_test)
        calibrated = platt_scores(raw_validation, y_validation, raw_test)
        policy_frames.append(economic_policy(
            combo,
            winner["candidate"],
            calibrated,
            {"validation": y_validation, "test": y_test},
        ))

    candidate_results = pd.DataFrame(candidate_rows).sort_values(
        ["feature_set", "pr_auc", "roc_auc"],
        ascending=[True, False, False],
    )
    selected = pd.DataFrame(selected_rows).sort_values(["pr_auc", "roc_auc"], ascending=False).reset_index(drop=True)
    selected["selection_rank"] = selected.index + 1
    selected_metrics = pd.DataFrame(selected_metric_rows)
    policy = pd.concat(policy_frames, ignore_index=True, sort=False)
    feature_summary = pd.DataFrame(feature_summary_rows)

    save_table(feature_summary, TABLE_DIR, "xgboost_grade_int_rate_feature_summary")
    save_table(candidate_results, TABLE_DIR, "xgboost_grade_int_rate_candidate_results")
    save_table(selected, TABLE_DIR, "xgboost_grade_int_rate_selected_candidates")
    save_table(selected_metrics, TABLE_DIR, "xgboost_grade_int_rate_selected_model_metrics")
    save_table(policy, TABLE_DIR, "xgboost_grade_int_rate_economic_policy")

    save_table(feature_summary, FINAL_TABLE_DIR, "xgboost_grade_int_rate_feature_summary")
    save_table(candidate_results, FINAL_TABLE_DIR, "xgboost_grade_int_rate_candidate_results")
    save_table(selected, FINAL_TABLE_DIR, "xgboost_grade_int_rate_selected_candidates")
    save_table(selected_metrics, FINAL_TABLE_DIR, "xgboost_grade_int_rate_selected_model_metrics")
    save_table(policy, FINAL_TABLE_DIR, "xgboost_grade_int_rate_economic_policy")

    print("\nSelected candidates")
    print(selected[[
        "selection_rank", "feature_set_label", "candidate", "feature_count",
        "pr_auc", "roc_auc", "mean_predicted_probability_raw",
        "best_f1_precision", "best_f1_recall",
    ]].to_string(index=False))
    print("\nEconomic policy")
    print(policy.to_string(index=False))


if __name__ == "__main__":
    main()
