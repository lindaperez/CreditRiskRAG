from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


RANDOM_STATE = 42
TARGET_PRECISION = 0.40
REVIEW_RATES = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_DATASET_DIR = PROJECT_ROOT / "Modeling" / "Preprocessing" / "preprocessing_outputs" / "datasets"
MODELING_OUTPUT_ROOT = PROJECT_ROOT / "Modeling" / "modeling_outputs"
OUTPUT_ROOT = MODELING_OUTPUT_ROOT / "missingness_challenger"
TABLE_DIR = OUTPUT_ROOT / "tables"
MODEL_DIR = OUTPUT_ROOT / "models"
for directory in [TABLE_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
}

DATASET_LABELS = {
    "baseline_with_grade_subgrade": "baseline with grade/subgrade",
    "baseline_no_grade_subgrade": "baseline no grade/subgrade",
    "missingness_challenger": "missingness challenger",
}


def load_parquet(name: str) -> pd.DataFrame:
    path = PREPROCESSING_DATASET_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def save_table(df: pd.DataFrame, name: str) -> Path:
    path = TABLE_DIR / f"missingness_challenger_{name}.csv"
    df.to_csv(path, index=False)
    print("Saved:", path, flush=True)
    return path


def predict_positive_probability(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def threshold_for_best_f1(y_true: pd.Series, y_score: np.ndarray) -> tuple[float, float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5, 0.0, 0.0, 0.0
    f1 = (2 * precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    idx = int(np.nanargmax(f1))
    return float(thresholds[idx]), float(f1[idx]), float(precision[idx]), float(recall[idx])


def threshold_for_target_precision(y_true: pd.Series, y_score: np.ndarray) -> tuple[float, float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5, 0.0, 0.0, 0.0
    candidate = np.where(precision[:-1] >= TARGET_PRECISION)[0]
    if len(candidate) == 0:
        idx = int(np.nanargmax(precision[:-1]))
    else:
        idx = int(candidate[np.nanargmax(recall[:-1][candidate])])
    f1 = (2 * precision[idx] * recall[idx]) / max(precision[idx] + recall[idx], 1e-12)
    return float(thresholds[idx]), float(f1), float(precision[idx]), float(recall[idx])


def evaluate_at_threshold(
    model_family: str,
    candidate: str,
    dataset: str,
    split: str,
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float,
    operating_point: str,
) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    reject_count = int(fp + tp)
    return {
        "model_family": model_family,
        "model_label": MODEL_LABELS[model_family],
        "model": candidate,
        "dataset": dataset,
        "dataset_label": DATASET_LABELS[dataset],
        "model_dataset_label": f"{MODEL_LABELS[model_family]} | {DATASET_LABELS[dataset]}",
        "split": split,
        "operating_point": operating_point,
        "rows": len(y_true),
        "bad_rate": round(float(y_true.mean()), 6),
        "threshold": round(float(threshold), 6),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 6),
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 6),
        "brier_score": round(float(brier_score_loss(y_true, y_score)), 6),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_reject_count": reject_count,
        "predicted_reject_share": round(float(reject_count / len(y_true)), 6),
        "false_rejection_share_among_rejects": round(float(fp / reject_count), 6) if reject_count else np.nan,
    }


def review_volume_metrics(
    model_family: str,
    candidate: str,
    dataset: str,
    split: str,
    y_true: pd.Series,
    y_score: np.ndarray,
) -> pd.DataFrame:
    order = np.argsort(-y_score)
    y_sorted = np.asarray(y_true)[order]
    base_bad_rate = float(np.mean(y_sorted))
    total_bad = int(y_sorted.sum())
    rows = []
    for rate in REVIEW_RATES:
        review_count = max(1, int(np.ceil(len(y_sorted) * rate)))
        reviewed = y_sorted[:review_count]
        captured_bad = int(reviewed.sum())
        precision = captured_bad / review_count
        recall = captured_bad / total_bad if total_bad else np.nan
        rows.append({
            "model_family": model_family,
            "model_label": MODEL_LABELS[model_family],
            "model": candidate,
            "dataset": dataset,
            "dataset_label": DATASET_LABELS[dataset],
            "model_dataset_label": f"{MODEL_LABELS[model_family]} | {DATASET_LABELS[dataset]}",
            "split": split,
            "review_pct": round(rate * 100, 2),
            "review_count": int(review_count),
            "captured_bad": captured_bad,
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "base_bad_rate": round(base_bad_rate, 6),
            "lift_over_base_bad_rate": round(float(precision / base_bad_rate), 6) if base_bad_rate else np.nan,
        })
    return pd.DataFrame(rows)


def sample_fit_data(X_train: pd.DataFrame, y_train: pd.Series, sample_rows: int | None) -> tuple[pd.DataFrame, pd.Series]:
    if not sample_rows or len(X_train) <= sample_rows:
        return X_train, y_train
    sample_idx = y_train.groupby(y_train).sample(frac=sample_rows / len(y_train), random_state=RANDOM_STATE).index
    return X_train.loc[sample_idx], y_train.loc[sample_idx]


def build_model(model_family: str, params: dict, y_fit: pd.Series):
    if model_family == "logistic_regression":
        return LogisticRegression(
            solver="saga",
            penalty=params["penalty"],
            C=params["C"],
            class_weight=params.get("class_weight"),
            l1_ratio=params.get("l1_ratio"),
            max_iter=1200,
            n_jobs=1,
            random_state=RANDOM_STATE,
        )
    if model_family == "random_forest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            class_weight=params.get("class_weight", "balanced_subsample"),
            n_jobs=1,
            random_state=RANDOM_STATE,
        )
    if model_family == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=params["learning_rate"],
            max_iter=params["max_iter"],
            max_leaf_nodes=params["max_leaf_nodes"],
            l2_regularization=params["l2_regularization"],
            early_stopping=True,
            validation_fraction=0.10,
            random_state=RANDOM_STATE,
        )
    if model_family == "lightgbm":
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
            scale_pos_weight=float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1)),
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=-1,
        )
    if model_family == "xgboost":
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
            scale_pos_weight=float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1)),
            random_state=RANDOM_STATE,
            n_jobs=1,
        )
    if model_family == "catboost":
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
    raise ValueError(model_family)


MODEL_CONFIGS = {
    "logistic_regression": {
        "sample_rows": None,
        "candidates": [
            {"candidate": "logistic_regression_01", "params": {"penalty": "l2", "C": 0.25, "class_weight": None}},
            {"candidate": "logistic_regression_02", "params": {"penalty": "l2", "C": 0.50, "class_weight": None}},
            {"candidate": "logistic_regression_03", "params": {"penalty": "l2", "C": 1.00, "class_weight": None}},
            {"candidate": "logistic_regression_04", "params": {"penalty": "l2", "C": 0.25, "class_weight": "balanced"}},
            {"candidate": "logistic_regression_05", "params": {"penalty": "l2", "C": 0.50, "class_weight": "balanced"}},
            {"candidate": "logistic_regression_06", "params": {"penalty": "l2", "C": 1.00, "class_weight": "balanced"}},
            {"candidate": "logistic_regression_07", "params": {"penalty": "elasticnet", "C": 0.50, "l1_ratio": 0.15, "class_weight": "balanced"}},
            {"candidate": "logistic_regression_08", "params": {"penalty": "elasticnet", "C": 1.00, "l1_ratio": 0.15, "class_weight": "balanced"}},
        ],
    },
    "random_forest": {
        "sample_rows": 200_000,
        "candidates": [
            {"candidate": "random_forest_01", "params": {"n_estimators": 120, "max_depth": 12, "min_samples_leaf": 80, "max_features": "sqrt", "class_weight": "balanced_subsample"}},
            {"candidate": "random_forest_02", "params": {"n_estimators": 160, "max_depth": 16, "min_samples_leaf": 60, "max_features": "sqrt", "class_weight": "balanced_subsample"}},
            {"candidate": "random_forest_03", "params": {"n_estimators": 200, "max_depth": 20, "min_samples_leaf": 40, "max_features": "sqrt", "class_weight": "balanced_subsample"}},
        ],
    },
    "hist_gradient_boosting": {
        "sample_rows": 300_000,
        "candidates": [
            {"candidate": "hist_gradient_boosting_01", "params": {"learning_rate": 0.04, "max_iter": 180, "max_leaf_nodes": 31, "l2_regularization": 0.0}},
            {"candidate": "hist_gradient_boosting_02", "params": {"learning_rate": 0.05, "max_iter": 200, "max_leaf_nodes": 31, "l2_regularization": 0.0}},
            {"candidate": "hist_gradient_boosting_03", "params": {"learning_rate": 0.05, "max_iter": 220, "max_leaf_nodes": 45, "l2_regularization": 0.01}},
            {"candidate": "hist_gradient_boosting_04", "params": {"learning_rate": 0.06, "max_iter": 220, "max_leaf_nodes": 45, "l2_regularization": 0.0}},
            {"candidate": "hist_gradient_boosting_05", "params": {"learning_rate": 0.07, "max_iter": 180, "max_leaf_nodes": 45, "l2_regularization": 0.01}},
            {"candidate": "hist_gradient_boosting_06", "params": {"learning_rate": 0.04, "max_iter": 260, "max_leaf_nodes": 63, "l2_regularization": 0.01}},
        ],
    },
    "lightgbm": {
        "sample_rows": 300_000,
        "candidates": [
            {"candidate": "lightgbm_01", "params": {"n_estimators": 250, "learning_rate": 0.04, "num_leaves": 31, "max_depth": -1, "min_child_samples": 60, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 1.0}},
            {"candidate": "lightgbm_02", "params": {"n_estimators": 300, "learning_rate": 0.035, "num_leaves": 45, "max_depth": -1, "min_child_samples": 80, "subsample": 0.85, "colsample_bytree": 0.80, "reg_lambda": 2.0}},
            {"candidate": "lightgbm_03", "params": {"n_estimators": 350, "learning_rate": 0.03, "num_leaves": 63, "max_depth": -1, "min_child_samples": 100, "subsample": 0.80, "colsample_bytree": 0.80, "reg_lambda": 3.0}},
            {"candidate": "lightgbm_04", "params": {"n_estimators": 220, "learning_rate": 0.05, "num_leaves": 31, "max_depth": 8, "min_child_samples": 80, "subsample": 0.90, "colsample_bytree": 0.90, "reg_lambda": 1.0}},
            {"candidate": "lightgbm_05", "params": {"n_estimators": 280, "learning_rate": 0.04, "num_leaves": 45, "max_depth": 10, "min_child_samples": 120, "subsample": 0.80, "colsample_bytree": 0.85, "reg_lambda": 4.0}},
            {"candidate": "lightgbm_06", "params": {"n_estimators": 400, "learning_rate": 0.025, "num_leaves": 63, "max_depth": 12, "min_child_samples": 120, "subsample": 0.80, "colsample_bytree": 0.75, "reg_lambda": 5.0}},
        ],
    },
    "xgboost": {
        "sample_rows": 300_000,
        "candidates": [
            {"candidate": "xgboost_01", "params": {"n_estimators": 220, "learning_rate": 0.04, "max_depth": 4, "min_child_weight": 8, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 2.0, "reg_alpha": 0.0}},
            {"candidate": "xgboost_02", "params": {"n_estimators": 260, "learning_rate": 0.035, "max_depth": 4, "min_child_weight": 12, "subsample": 0.85, "colsample_bytree": 0.80, "reg_lambda": 3.0, "reg_alpha": 0.1}},
            {"candidate": "xgboost_03", "params": {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 5, "min_child_weight": 10, "subsample": 0.80, "colsample_bytree": 0.80, "reg_lambda": 4.0, "reg_alpha": 0.1}},
            {"candidate": "xgboost_04", "params": {"n_estimators": 240, "learning_rate": 0.045, "max_depth": 5, "min_child_weight": 16, "subsample": 0.90, "colsample_bytree": 0.85, "reg_lambda": 3.0, "reg_alpha": 0.2}},
            {"candidate": "xgboost_05", "params": {"n_estimators": 350, "learning_rate": 0.025, "max_depth": 6, "min_child_weight": 16, "subsample": 0.80, "colsample_bytree": 0.75, "reg_lambda": 5.0, "reg_alpha": 0.2}},
            {"candidate": "xgboost_06", "params": {"n_estimators": 180, "learning_rate": 0.06, "max_depth": 3, "min_child_weight": 8, "subsample": 0.90, "colsample_bytree": 0.90, "reg_lambda": 2.0, "reg_alpha": 0.0}},
        ],
    },
    "catboost": {
        "sample_rows": 100_000,
        "candidates": [
            {"candidate": "catboost_01", "params": {"iterations": 220, "learning_rate": 0.04, "depth": 4, "l2_leaf_reg": 3.0, "subsample": 0.85}},
            {"candidate": "catboost_02", "params": {"iterations": 260, "learning_rate": 0.035, "depth": 5, "l2_leaf_reg": 4.0, "subsample": 0.85}},
            {"candidate": "catboost_04", "params": {"iterations": 240, "learning_rate": 0.045, "depth": 5, "l2_leaf_reg": 6.0, "subsample": 0.90}},
        ],
    },
}


def fit_kwargs(model_family: str, y_fit: pd.Series) -> dict:
    if model_family == "hist_gradient_boosting":
        return {"sample_weight": compute_sample_weight(class_weight="balanced", y=y_fit)}
    return {}


def train_missingness_models() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train = load_parquet("missingness_challenger_train_X")
    X_validation = load_parquet("missingness_challenger_validation_X")
    X_test = load_parquet("missingness_challenger_test_X")
    y_train = load_parquet("train_y")["target_bad"].astype(int)
    y_validation = load_parquet("validation_y")["target_bad"].astype(int)
    y_test = load_parquet("test_y")["target_bad"].astype(int)

    candidate_rows = []
    selected_rows = []
    metric_rows = []
    review_rows = []

    for model_family, config in MODEL_CONFIGS.items():
        X_fit, y_fit = sample_fit_data(X_train, y_train, config["sample_rows"])
        fitted_models = {}
        family_rows = []
        for candidate in config["candidates"]:
            candidate_name = f"{candidate['candidate']}_missingness_challenger"
            print(f"Training {candidate_name}: {candidate['params']}", flush=True)
            start = time.perf_counter()
            model = build_model(model_family, candidate["params"], y_fit)
            model.fit(X_fit, y_fit, **fit_kwargs(model_family, y_fit))
            seconds = time.perf_counter() - start
            validation_score = predict_positive_probability(model, X_validation)
            best_threshold, best_f1, best_precision, best_recall = threshold_for_best_f1(y_validation, validation_score)
            precision_threshold, precision_f1, precision_value, precision_recall = threshold_for_target_precision(y_validation, validation_score)
            row = {
                "model_family": model_family,
                "model_label": MODEL_LABELS[model_family],
                "candidate": candidate_name,
                "dataset": "missingness_challenger",
                "dataset_label": DATASET_LABELS["missingness_challenger"],
                "model_dataset_label": f"{MODEL_LABELS[model_family]} | {DATASET_LABELS['missingness_challenger']}",
                "params": json.dumps(candidate["params"], sort_keys=True),
                "fit_rows": len(X_fit),
                "fit_bad_rate": round(float(y_fit.mean()), 6),
                "fit_seconds": round(float(seconds), 3),
                "roc_auc": round(float(roc_auc_score(y_validation, validation_score)), 6),
                "pr_auc": round(float(average_precision_score(y_validation, validation_score)), 6),
                "best_f1_threshold": round(best_threshold, 6),
                "best_f1": round(best_f1, 6),
                "best_f1_precision": round(best_precision, 6),
                "best_f1_recall": round(best_recall, 6),
                "target_precision_threshold": round(precision_threshold, 6),
                "target_precision_f1": round(precision_f1, 6),
                "target_precision": round(precision_value, 6),
                "target_precision_recall": round(precision_recall, 6),
            }
            family_rows.append(row)
            fitted_models[candidate_name] = model
            print(f"Finished {candidate_name}: f1={best_f1:.4f}, pr_auc={row['pr_auc']:.4f}, seconds={seconds:.1f}", flush=True)

        family_results = pd.DataFrame(family_rows).sort_values(["best_f1", "best_f1_precision", "pr_auc"], ascending=False)
        candidate_rows.extend(family_results.to_dict("records"))
        winner = family_results.iloc[0]
        selected_rows.append(winner.to_dict())
        selected_model = fitted_models[winner["candidate"]]
        model_path = MODEL_DIR / f"{winner['candidate']}_selected_model.joblib"
        joblib.dump(selected_model, model_path)

        scores = {
            "train": predict_positive_probability(selected_model, X_train),
            "validation": predict_positive_probability(selected_model, X_validation),
            "test": predict_positive_probability(selected_model, X_test),
        }
        y_parts = {"train": y_train, "validation": y_validation, "test": y_test}
        for split, y_part in y_parts.items():
            metric_rows.append(evaluate_at_threshold(model_family, winner["candidate"], "missingness_challenger", split, y_part, scores[split], winner["best_f1_threshold"], "best_validation_f1"))
            metric_rows.append(evaluate_at_threshold(model_family, winner["candidate"], "missingness_challenger", split, y_part, scores[split], winner["target_precision_threshold"], "target_validation_precision"))
        for split in ["validation", "test"]:
            review_rows.extend(review_volume_metrics(model_family, winner["candidate"], "missingness_challenger", split, y_parts[split], scores[split]).to_dict("records"))

    return pd.DataFrame(candidate_rows), pd.DataFrame(selected_rows), pd.DataFrame(metric_rows), pd.DataFrame(review_rows)


def infer_family(model: str) -> str:
    for family in MODEL_LABELS:
        if model == family or model.startswith(f"{family}_"):
            return family
    if model == "hist_gradient_boosting_no_grade_subgrade":
        return "hist_gradient_boosting"
    return model.replace("_no_grade_subgrade", "")


def load_existing_dataset_metrics() -> pd.DataFrame:
    rows = []
    for family in MODEL_LABELS:
        path = MODELING_OUTPUT_ROOT / family / "tables" / f"{family}_selected_model_metrics.csv"
        if path.exists():
            df = pd.read_csv(path)
            df = df[df["operating_point"] == "best_validation_f1"].copy()
            df["dataset"] = "baseline_with_grade_subgrade"
            df["dataset_label"] = DATASET_LABELS["baseline_with_grade_subgrade"]
            df["model_family"] = family
            df["model_label"] = MODEL_LABELS[family]
            rows.append(df)

    no_grade_path = MODELING_OUTPUT_ROOT / "tables" / "no_grade_subgrade_model_metrics.csv"
    if no_grade_path.exists():
        df = pd.read_csv(no_grade_path)
        df["dataset"] = "baseline_no_grade_subgrade"
        df["dataset_label"] = DATASET_LABELS["baseline_no_grade_subgrade"]
        df["model_family"] = df["model"].map(infer_family)
        df["model_label"] = df["model_family"].map(MODEL_LABELS)
        df["operating_point"] = "best_validation_f1"
        rows.append(df)

    combined = pd.concat(rows, ignore_index=True)
    combined["model_dataset_label"] = combined["model_label"] + " | " + combined["dataset_label"]
    return combined


def main() -> None:
    candidate_results, selected_candidates, missingness_metrics, review_volume = train_missingness_models()
    save_table(candidate_results, "candidate_results")
    save_table(selected_candidates, "selected_candidates")
    save_table(missingness_metrics, "selected_model_metrics")
    save_table(review_volume, "review_volume_precision")

    existing_metrics = load_existing_dataset_metrics()
    all_dataset_metrics = pd.concat([existing_metrics, missingness_metrics], ignore_index=True, sort=False)
    save_table(all_dataset_metrics, "all_dataset_selected_model_metrics")

    comparison = all_dataset_metrics[
        (all_dataset_metrics["split"].isin(["validation", "test"]))
        & (all_dataset_metrics["operating_point"] == "best_validation_f1")
    ].copy()
    comparison["predicted_reject_share"] = comparison.get("predicted_reject_share", np.nan)
    if comparison["predicted_reject_share"].isna().any():
        comparison["predicted_reject_share"] = comparison["predicted_reject_share"].fillna((comparison["fp"] + comparison["tp"]) / comparison["rows"])
    comparison["false_rejection_share_among_rejects"] = comparison.get("false_rejection_share_among_rejects", np.nan)
    missing_false_reject = comparison["false_rejection_share_among_rejects"].isna()
    comparison.loc[missing_false_reject, "false_rejection_share_among_rejects"] = comparison.loc[missing_false_reject, "fp"] / (comparison.loc[missing_false_reject, "fp"] + comparison.loc[missing_false_reject, "tp"])
    comparison = comparison.sort_values(["split", "f1", "precision", "pr_auc"], ascending=[True, False, False, False])
    save_table(comparison, "validation_test_dataset_comparison")

    validation_rank = comparison[comparison["split"] == "validation"].sort_values(["f1", "precision", "pr_auc"], ascending=False).reset_index(drop=True)
    validation_rank["selection_rank"] = validation_rank.index + 1
    save_table(validation_rank, "ranking_by_validation_f1")

    test_rank = comparison[comparison["split"] == "test"].sort_values(["f1", "precision", "pr_auc"], ascending=False).reset_index(drop=True)
    test_rank["test_rank"] = test_rank.index + 1
    save_table(test_rank, "ranking_by_test_f1")

    best_by_model = validation_rank.sort_values(["model_family", "selection_rank"]).groupby("model_family", as_index=False).first()
    save_table(best_by_model, "best_dataset_by_model_family")

    recommendation = validation_rank.iloc[[0]].copy()
    save_table(recommendation, "recommendation")
    print("\nTop validation dataset/model combinations")
    print(validation_rank[["selection_rank", "model_dataset_label", "model", "f1", "precision", "recall", "pr_auc", "roc_auc", "predicted_reject_share"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
