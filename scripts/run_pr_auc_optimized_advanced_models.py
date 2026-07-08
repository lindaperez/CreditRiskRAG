from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
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
from xgboost import XGBClassifier


RANDOM_STATE = 42
TARGET_PRECISION = 0.40
REVIEW_RATES = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
FIT_SAMPLE_ROWS = 300_000

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_DATASET_DIR = PROJECT_ROOT / "Modeling" / "Preprocessing" / "preprocessing_outputs" / "datasets"
OUTPUT_ROOT = PROJECT_ROOT / "Modeling" / "modeling_outputs" / "pr_auc_optimized"
TABLE_DIR = OUTPUT_ROOT / "tables"
PLOT_DIR = OUTPUT_ROOT / "plots"
MODEL_DIR = OUTPUT_ROOT / "models"
for directory in [TABLE_DIR, PLOT_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

FEATURE_SETS = {
    "baseline_with_grade_subgrade": {
        "label": "with grade/subgrade",
        "prefix": "baseline",
    },
    "baseline_no_grade_subgrade": {
        "label": "no grade/subgrade",
        "prefix": "baseline_no_grade_subgrade",
    },
}

MODEL_LABELS = {
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
}


def load_parquet(name: str) -> pd.DataFrame:
    path = PREPROCESSING_DATASET_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def save_table(df: pd.DataFrame, name: str) -> Path:
    path = TABLE_DIR / f"pr_auc_optimized_{name}.csv"
    df.to_csv(path, index=False)
    print("Saved:", path)
    return path


def save_plot(fig, name: str) -> Path:
    path = PLOT_DIR / f"pr_auc_optimized_{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)
    return path


def threshold_for_best_f1(y_true: pd.Series, y_score: np.ndarray) -> tuple[float, float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5, 0.0, 0.0, 0.0
    f1 = (2 * precision[:-1] * recall[:-1]) / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    idx = int(np.nanargmax(f1))
    return float(thresholds[idx]), float(f1[idx]), float(precision[idx]), float(recall[idx])


def threshold_for_target_precision(
    y_true: pd.Series,
    y_score: np.ndarray,
    target_precision: float,
) -> tuple[float, float, float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5, 0.0, 0.0, 0.0
    candidate = np.where(precision[:-1] >= target_precision)[0]
    if len(candidate) == 0:
        idx = int(np.nanargmax(precision[:-1]))
    else:
        idx = int(candidate[np.nanargmax(recall[:-1][candidate])])
    f1 = (2 * precision[idx] * recall[idx]) / max(precision[idx] + recall[idx], 1e-12)
    return float(thresholds[idx]), float(f1), float(precision[idx]), float(recall[idx])


def predict_positive_probability(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def evaluate_at_threshold(
    model_family: str,
    model_label: str,
    candidate: str,
    feature_set: str,
    split: str,
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float,
    operating_point: str,
) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    predicted_reject_count = int(fp + tp)
    return {
        "model_family": model_family,
        "model_label": model_label,
        "model": candidate,
        "feature_set": feature_set,
        "feature_set_label": FEATURE_SETS[feature_set]["label"],
        "model_feature_label": f"{model_label} | {FEATURE_SETS[feature_set]['label']}",
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
        "predicted_reject_count": predicted_reject_count,
        "predicted_reject_share": round(float(predicted_reject_count / len(y_true)), 6),
        "false_rejection_share_among_rejects": round(float(fp / predicted_reject_count), 6) if predicted_reject_count else np.nan,
    }


def review_volume_metrics(
    model_family: str,
    model_label: str,
    candidate: str,
    feature_set: str,
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
            "model_label": model_label,
            "model": candidate,
            "feature_set": feature_set,
            "feature_set_label": FEATURE_SETS[feature_set]["label"],
            "model_feature_label": f"{model_label} | {FEATURE_SETS[feature_set]['label']}",
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


def sample_fit_data(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    if len(X_train) <= FIT_SAMPLE_ROWS:
        return X_train, y_train
    sample_idx = y_train.groupby(y_train).sample(frac=FIT_SAMPLE_ROWS / len(y_train), random_state=RANDOM_STATE).index
    return X_train.loc[sample_idx], y_train.loc[sample_idx]


def build_lightgbm(params: dict, y_fit: pd.Series) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        boosting_type="gbdt",
        metric="average_precision",
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


def build_xgboost(params: dict, y_fit: pd.Series) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
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


def build_catboost(params: dict, y_fit: pd.Series) -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="PRAUC",
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


MODEL_CONFIGS = {
    "lightgbm": {
        "builder": build_lightgbm,
        "candidates": [
            {"candidate": "lightgbm_pr_auc_01", "params": {"n_estimators": 250, "learning_rate": 0.04, "num_leaves": 31, "max_depth": -1, "min_child_samples": 60, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 1.0}},
            {"candidate": "lightgbm_pr_auc_02", "params": {"n_estimators": 300, "learning_rate": 0.035, "num_leaves": 45, "max_depth": -1, "min_child_samples": 80, "subsample": 0.85, "colsample_bytree": 0.80, "reg_lambda": 2.0}},
            {"candidate": "lightgbm_pr_auc_03", "params": {"n_estimators": 350, "learning_rate": 0.03, "num_leaves": 63, "max_depth": -1, "min_child_samples": 100, "subsample": 0.80, "colsample_bytree": 0.80, "reg_lambda": 3.0}},
            {"candidate": "lightgbm_pr_auc_04", "params": {"n_estimators": 220, "learning_rate": 0.05, "num_leaves": 31, "max_depth": 8, "min_child_samples": 80, "subsample": 0.90, "colsample_bytree": 0.90, "reg_lambda": 1.0}},
            {"candidate": "lightgbm_pr_auc_05", "params": {"n_estimators": 280, "learning_rate": 0.04, "num_leaves": 45, "max_depth": 10, "min_child_samples": 120, "subsample": 0.80, "colsample_bytree": 0.85, "reg_lambda": 4.0}},
            {"candidate": "lightgbm_pr_auc_06", "params": {"n_estimators": 400, "learning_rate": 0.025, "num_leaves": 63, "max_depth": 12, "min_child_samples": 120, "subsample": 0.80, "colsample_bytree": 0.75, "reg_lambda": 5.0}},
        ],
    },
    "xgboost": {
        "builder": build_xgboost,
        "candidates": [
            {"candidate": "xgboost_pr_auc_01", "params": {"n_estimators": 220, "learning_rate": 0.04, "max_depth": 4, "min_child_weight": 8, "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 2.0, "reg_alpha": 0.0}},
            {"candidate": "xgboost_pr_auc_02", "params": {"n_estimators": 260, "learning_rate": 0.035, "max_depth": 4, "min_child_weight": 12, "subsample": 0.85, "colsample_bytree": 0.80, "reg_lambda": 3.0, "reg_alpha": 0.1}},
            {"candidate": "xgboost_pr_auc_03", "params": {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 5, "min_child_weight": 10, "subsample": 0.80, "colsample_bytree": 0.80, "reg_lambda": 4.0, "reg_alpha": 0.1}},
            {"candidate": "xgboost_pr_auc_04", "params": {"n_estimators": 240, "learning_rate": 0.045, "max_depth": 5, "min_child_weight": 16, "subsample": 0.90, "colsample_bytree": 0.85, "reg_lambda": 3.0, "reg_alpha": 0.2}},
            {"candidate": "xgboost_pr_auc_05", "params": {"n_estimators": 350, "learning_rate": 0.025, "max_depth": 6, "min_child_weight": 16, "subsample": 0.80, "colsample_bytree": 0.75, "reg_lambda": 5.0, "reg_alpha": 0.2}},
            {"candidate": "xgboost_pr_auc_06", "params": {"n_estimators": 180, "learning_rate": 0.06, "max_depth": 3, "min_child_weight": 8, "subsample": 0.90, "colsample_bytree": 0.90, "reg_lambda": 2.0, "reg_alpha": 0.0}},
        ],
    },
    "catboost": {
        "builder": build_catboost,
        "candidates": [
            {"candidate": "catboost_pr_auc_01", "params": {"iterations": 220, "learning_rate": 0.04, "depth": 4, "l2_leaf_reg": 3.0, "subsample": 0.85}},
            {"candidate": "catboost_pr_auc_02", "params": {"iterations": 260, "learning_rate": 0.035, "depth": 5, "l2_leaf_reg": 4.0, "subsample": 0.85}},
            {"candidate": "catboost_pr_auc_03", "params": {"iterations": 300, "learning_rate": 0.03, "depth": 6, "l2_leaf_reg": 5.0, "subsample": 0.80}},
            {"candidate": "catboost_pr_auc_04", "params": {"iterations": 240, "learning_rate": 0.045, "depth": 5, "l2_leaf_reg": 6.0, "subsample": 0.90}},
            {"candidate": "catboost_pr_auc_05", "params": {"iterations": 350, "learning_rate": 0.025, "depth": 6, "l2_leaf_reg": 8.0, "subsample": 0.80}},
            {"candidate": "catboost_pr_auc_06", "params": {"iterations": 180, "learning_rate": 0.06, "depth": 4, "l2_leaf_reg": 5.0, "subsample": 0.90}},
        ],
    },
}


def train_candidate(
    model_family: str,
    candidate: dict,
    feature_set: str,
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
):
    model_label = MODEL_LABELS[model_family]
    candidate_name = f"{candidate['candidate']}_{feature_set.replace('baseline_', '')}"
    print(f"Training {candidate_name}: {candidate['params']}", flush=True)
    start = time.perf_counter()
    model = MODEL_CONFIGS[model_family]["builder"](candidate["params"], y_fit)
    model.fit(X_fit, y_fit)
    seconds = time.perf_counter() - start
    validation_score = predict_positive_probability(model, X_validation)
    best_threshold, best_f1, best_precision, best_recall = threshold_for_best_f1(y_validation, validation_score)
    precision_threshold, precision_f1, precision_value, precision_recall = threshold_for_target_precision(
        y_validation,
        validation_score,
        TARGET_PRECISION,
    )
    row = {
        "model_family": model_family,
        "model_label": model_label,
        "candidate": candidate_name,
        "feature_set": feature_set,
        "feature_set_label": FEATURE_SETS[feature_set]["label"],
        "model_feature_label": f"{model_label} | {FEATURE_SETS[feature_set]['label']}",
        "selection_metric": "validation_pr_auc",
        "training_eval_metric": {"lightgbm": "average_precision", "xgboost": "aucpr", "catboost": "PRAUC"}[model_family],
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
    print(f"Finished {candidate_name}: pr_auc={row['pr_auc']:.4f}, best_f1={row['best_f1']:.4f}, seconds={seconds:.1f}", flush=True)
    return row, model


def main() -> None:
    y_train = load_parquet("train_y")["target_bad"].astype(int)
    y_validation = load_parquet("validation_y")["target_bad"].astype(int)
    y_test = load_parquet("test_y")["target_bad"].astype(int)

    all_candidate_rows = []
    selected_rows = []
    metric_rows = []
    review_rows = []
    artifact_rows = []

    for feature_set, feature_config in FEATURE_SETS.items():
        prefix = feature_config["prefix"]
        X_train = load_parquet(f"{prefix}_train_X")
        X_validation = load_parquet(f"{prefix}_validation_X")
        X_test = load_parquet(f"{prefix}_test_X")
        X_fit, y_fit = sample_fit_data(X_train, y_train)

        for model_family in ["lightgbm", "xgboost", "catboost"]:
            fitted_models = {}
            candidate_rows = []
            for candidate in MODEL_CONFIGS[model_family]["candidates"]:
                row, model = train_candidate(model_family, candidate, feature_set, X_fit, y_fit, X_validation, y_validation)
                candidate_rows.append(row)
                fitted_models[row["candidate"]] = model

            candidate_results = pd.DataFrame(candidate_rows).sort_values(
                ["pr_auc", "roc_auc", "best_f1"],
                ascending=False,
            )
            all_candidate_rows.extend(candidate_results.to_dict("records"))

            winner = candidate_results.iloc[0]
            selected_rows.append(winner.to_dict())
            selected_model = fitted_models[winner["candidate"]]
            model_path = MODEL_DIR / f"{winner['candidate']}_selected_model.joblib"
            joblib.dump(selected_model, model_path)
            artifact_rows.append({
                "model_family": model_family,
                "model_label": MODEL_LABELS[model_family],
                "candidate": winner["candidate"],
                "feature_set": feature_set,
                "feature_set_label": FEATURE_SETS[feature_set]["label"],
                "artifact_path": str(model_path),
            })

            scores = {
                "train": predict_positive_probability(selected_model, X_train),
                "validation": predict_positive_probability(selected_model, X_validation),
                "test": predict_positive_probability(selected_model, X_test),
            }
            y_parts = {"train": y_train, "validation": y_validation, "test": y_test}
            for split, y_part in y_parts.items():
                metric_rows.append(evaluate_at_threshold(
                    model_family,
                    MODEL_LABELS[model_family],
                    winner["candidate"],
                    feature_set,
                    split,
                    y_part,
                    scores[split],
                    winner["best_f1_threshold"],
                    "best_validation_f1_pr_auc_selected",
                ))
                metric_rows.append(evaluate_at_threshold(
                    model_family,
                    MODEL_LABELS[model_family],
                    winner["candidate"],
                    feature_set,
                    split,
                    y_part,
                    scores[split],
                    winner["target_precision_threshold"],
                    "target_validation_precision_pr_auc_selected",
                ))
            for split in ["validation", "test"]:
                review_rows.extend(review_volume_metrics(
                    model_family,
                    MODEL_LABELS[model_family],
                    winner["candidate"],
                    feature_set,
                    split,
                    y_parts[split],
                    scores[split],
                ).to_dict("records"))

    all_candidates = pd.DataFrame(all_candidate_rows).sort_values(["feature_set", "model_family", "pr_auc"], ascending=[True, True, False])
    selected_candidates = pd.DataFrame(selected_rows).sort_values(["pr_auc", "roc_auc", "best_f1"], ascending=False)
    selected_metrics = pd.DataFrame(metric_rows)
    review_volume = pd.DataFrame(review_rows)
    artifacts = pd.DataFrame(artifact_rows)

    save_table(all_candidates, "candidate_results")
    save_table(selected_candidates, "selected_candidates")
    save_table(selected_metrics, "selected_model_metrics")
    save_table(review_volume, "review_volume_precision")
    save_table(artifacts, "model_artifacts")

    validation_rank = (
        selected_metrics[
            (selected_metrics["split"] == "validation")
            & (selected_metrics["operating_point"] == "best_validation_f1_pr_auc_selected")
        ]
        .sort_values(["pr_auc", "roc_auc", "f1"], ascending=False)
        .reset_index(drop=True)
    )
    validation_rank["selection_rank"] = validation_rank.index + 1
    save_table(validation_rank, "ranking_by_validation_pr_auc")

    test_rank = (
        selected_metrics[
            (selected_metrics["split"] == "test")
            & (selected_metrics["operating_point"] == "best_validation_f1_pr_auc_selected")
        ]
        .sort_values(["pr_auc", "roc_auc", "f1"], ascending=False)
        .reset_index(drop=True)
    )
    test_rank["test_rank"] = test_rank.index + 1
    save_table(test_rank, "test_ranking")

    winner = validation_rank.iloc[0]
    winner_test = test_rank[test_rank["model"] == winner["model"]].iloc[0]
    recommendation = pd.DataFrame([{
        "recommended_model_feature_label": winner["model_feature_label"],
        "recommended_model_label": winner["model_label"],
        "recommended_candidate": winner["model"],
        "recommended_feature_set": winner["feature_set"],
        "recommended_feature_set_label": winner["feature_set_label"],
        "selection_basis": "Highest validation PR-AUC among PR-AUC-monitored LightGBM, XGBoost, and CatBoost candidates.",
        "validation_pr_auc": winner["pr_auc"],
        "validation_roc_auc": winner["roc_auc"],
        "validation_f1": winner["f1"],
        "validation_precision": winner["precision"],
        "validation_recall": winner["recall"],
        "validation_predicted_reject_share": winner["predicted_reject_share"],
        "validation_false_rejection_share_among_rejects": winner["false_rejection_share_among_rejects"],
        "test_pr_auc": winner_test["pr_auc"],
        "test_roc_auc": winner_test["roc_auc"],
        "test_f1": winner_test["f1"],
        "test_precision": winner_test["precision"],
        "test_recall": winner_test["recall"],
        "test_predicted_reject_share": winner_test["predicted_reject_share"],
        "test_false_rejection_share_among_rejects": winner_test["false_rejection_share_among_rejects"],
    }])
    save_table(recommendation, "recommendation")

    print("\\nRecommendation")
    print(recommendation.to_string(index=False))


if __name__ == "__main__":
    main()
