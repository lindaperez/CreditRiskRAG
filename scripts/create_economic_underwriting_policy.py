from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "Modeling" / "Preprocessing" / "preprocessing_outputs" / "datasets"
MODELING_OUTPUT_ROOT = PROJECT_ROOT / "Modeling" / "modeling_outputs"
FINAL_TABLE_DIR = MODELING_OUTPUT_ROOT / "final_comparison" / "tables"
FINAL_TABLE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FAMILY = "xgboost"
MODEL_LABEL = "XGBoost"
MODEL_NAME = "xgboost_05_missingness_challenger"
DATASET = "missingness_challenger"
DATASET_LABEL = "missingness challenger"
MODEL_DATASET_LABEL = f"{MODEL_LABEL} | {DATASET_LABEL}"
MODEL_PATH = MODELING_OUTPUT_ROOT / DATASET / "models" / f"{MODEL_NAME}_selected_model.joblib"

# Default business assumptions. These should be replaced with finance-approved values
# before the policy is used for production underwriting.
DEFAULT_DEFAULT_LOSS_SAVED = 10_000.0
DEFAULT_GOOD_BORROWER_OPPORTUNITY_COST = 1_500.0
DEFAULT_MAX_REJECT_SHARE = 0.20


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


def evaluate_threshold(
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float,
    default_loss_saved: float,
    good_borrower_opportunity_cost: float,
) -> dict:
    y = np.asarray(y_true).astype(int)
    reject = y_score >= threshold
    tp = int(((reject == 1) & (y == 1)).sum())
    fp = int(((reject == 1) & (y == 0)).sum())
    tn = int(((reject == 0) & (y == 0)).sum())
    fn = int(((reject == 0) & (y == 1)).sum())
    rows = int(len(y))
    reject_count = int(reject.sum())
    approved_count = rows - reject_count
    total_value = tp * default_loss_saved - fp * good_borrower_opportunity_cost
    precision = tp / reject_count if reject_count else np.nan
    recall = tp / int(y.sum()) if int(y.sum()) else np.nan
    return {
        "threshold": round(float(threshold), 6),
        "rows": rows,
        "reject_count": reject_count,
        "approved_count": approved_count,
        "predicted_reject_share": round(float(reject_count / rows), 6),
        "approved_share": round(float(approved_count / rows), 6),
        "tp_caught_defaults": tp,
        "fp_rejected_good": fp,
        "tn_approved_good": tn,
        "fn_approved_defaults": fn,
        "precision_bad_rate_among_rejected": round(float(precision), 6) if not np.isnan(precision) else np.nan,
        "recall_default_capture": round(float(recall), 6) if not np.isnan(recall) else np.nan,
        "approved_bad_rate": round(float(fn / approved_count), 6) if approved_count else np.nan,
        "rejected_bad_rate": round(float(precision), 6) if not np.isnan(precision) else np.nan,
        "total_portfolio_value": round(float(total_value), 2),
        "value_per_applicant": round(float(total_value / rows), 2),
    }


def threshold_candidates(y_score: np.ndarray) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, 1001)
    candidates = np.unique(np.quantile(y_score, quantiles))
    return np.r_[0.0, candidates, 1.0]


def optimize_threshold(
    y_true: pd.Series,
    y_score: np.ndarray,
    calibration_method: str,
    default_loss_saved: float,
    good_borrower_opportunity_cost: float,
    max_reject_share: float | None = None,
) -> pd.DataFrame:
    rows = []
    for threshold in threshold_candidates(y_score):
        metrics = evaluate_threshold(
            y_true,
            y_score,
            threshold,
            default_loss_saved,
            good_borrower_opportunity_cost,
        )
        if max_reject_share is not None and metrics["predicted_reject_share"] > max_reject_share:
            continue
        rows.append(metrics)
    out = pd.DataFrame(rows)
    out["calibration_method"] = calibration_method
    out["default_loss_saved"] = default_loss_saved
    out["good_borrower_opportunity_cost"] = good_borrower_opportunity_cost
    out["max_reject_share_constraint"] = max_reject_share
    out["economic_break_even_probability"] = round(
        float(good_borrower_opportunity_cost / (default_loss_saved + good_borrower_opportunity_cost)),
        6,
    )
    return out.sort_values(
        ["total_portfolio_value", "precision_bad_rate_among_rejected", "threshold"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def score_splits() -> tuple[dict[tuple[str, str], np.ndarray], dict[str, pd.Series]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    model = joblib.load(MODEL_PATH)
    X_validation = load_parquet("missingness_challenger_validation_X")
    X_test = load_parquet("missingness_challenger_test_X")
    y_validation = load_parquet("validation_y")["target_bad"].astype(int)
    y_test = load_parquet("test_y")["target_bad"].astype(int)

    raw_validation = model.predict_proba(X_validation)[:, 1]
    raw_test = model.predict_proba(X_test)[:, 1]

    platt = LogisticRegression(solver="lbfgs", random_state=42)
    platt.fit(raw_validation.reshape(-1, 1), y_validation)

    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(raw_validation, y_validation)

    scores = {
        ("raw", "validation"): raw_validation,
        ("raw", "test"): raw_test,
        ("platt_sigmoid", "validation"): platt.predict_proba(raw_validation.reshape(-1, 1))[:, 1],
        ("platt_sigmoid", "test"): platt.predict_proba(raw_test.reshape(-1, 1))[:, 1],
        ("isotonic", "validation"): isotonic.predict(raw_validation),
        ("isotonic", "test"): isotonic.predict(raw_test),
    }
    y_by_split = {
        "validation": y_validation,
        "test": y_test,
    }
    return scores, y_by_split


def main() -> None:
    scores, y_by_split = score_splits()

    validation_threshold_rows = []
    unconstrained_curves = []
    constrained_curves = []
    for method in ["raw", "platt_sigmoid", "isotonic"]:
        unconstrained = optimize_threshold(
            y_by_split["validation"],
            scores[(method, "validation")],
            method,
            DEFAULT_DEFAULT_LOSS_SAVED,
            DEFAULT_GOOD_BORROWER_OPPORTUNITY_COST,
            max_reject_share=None,
        )
        constrained = optimize_threshold(
            y_by_split["validation"],
            scores[(method, "validation")],
            method,
            DEFAULT_DEFAULT_LOSS_SAVED,
            DEFAULT_GOOD_BORROWER_OPPORTUNITY_COST,
            max_reject_share=DEFAULT_MAX_REJECT_SHARE,
        )
        unconstrained_curves.append(unconstrained.assign(policy_type="economic_unconstrained"))
        constrained_curves.append(constrained.assign(policy_type="economic_max_20pct_reject"))

        for policy_type, curve in [
            ("economic_unconstrained", unconstrained),
            ("economic_max_20pct_reject", constrained),
        ]:
            winner = curve.iloc[0].to_dict()
            winner["policy_type"] = policy_type
            validation_threshold_rows.append(winner)

    selected_thresholds = pd.DataFrame(validation_threshold_rows)
    selected_thresholds["selected_on_split"] = "validation"
    selected_thresholds["model_family"] = MODEL_FAMILY
    selected_thresholds["model"] = MODEL_NAME
    selected_thresholds["dataset"] = DATASET
    selected_thresholds["model_dataset_label"] = MODEL_DATASET_LABEL

    application_rows = []
    for _, selected in selected_thresholds.iterrows():
        for split in ["validation", "test"]:
            metrics = evaluate_threshold(
                y_by_split[split],
                scores[(selected["calibration_method"], split)],
                float(selected["threshold"]),
                DEFAULT_DEFAULT_LOSS_SAVED,
                DEFAULT_GOOD_BORROWER_OPPORTUNITY_COST,
            )
            metrics.update({
                "policy_type": selected["policy_type"],
                "calibration_method": selected["calibration_method"],
                "split": split,
                "selected_on_split": "validation",
                "default_loss_saved": DEFAULT_DEFAULT_LOSS_SAVED,
                "good_borrower_opportunity_cost": DEFAULT_GOOD_BORROWER_OPPORTUNITY_COST,
                "max_reject_share_constraint": selected["max_reject_share_constraint"],
                "economic_break_even_probability": selected["economic_break_even_probability"],
                "pr_auc": round(float(average_precision_score(y_by_split[split], scores[(selected["calibration_method"], split)])), 6),
                "roc_auc": round(float(roc_auc_score(y_by_split[split], scores[(selected["calibration_method"], split)])), 6),
                "model_family": MODEL_FAMILY,
                "model": MODEL_NAME,
                "dataset": DATASET,
                "model_dataset_label": MODEL_DATASET_LABEL,
            })
            application_rows.append(metrics)

    threshold_curve = pd.concat(unconstrained_curves + constrained_curves, ignore_index=True)
    threshold_curve["model_family"] = MODEL_FAMILY
    threshold_curve["model"] = MODEL_NAME
    threshold_curve["dataset"] = DATASET
    threshold_curve["model_dataset_label"] = MODEL_DATASET_LABEL

    policy_application = pd.DataFrame(application_rows)
    policy_application = policy_application.sort_values(
        ["policy_type", "calibration_method", "split"]
    ).reset_index(drop=True)

    recommendation = policy_application[
        (policy_application["policy_type"] == "economic_max_20pct_reject")
        & (policy_application["calibration_method"] == "platt_sigmoid")
        & (policy_application["split"] == "test")
    ].copy()
    recommendation["recommendation_reason"] = (
        "Use calibrated probabilities and a dollar-value objective, but constrain rejection volume "
        "to a business-approved maximum. With the example cost assumptions, the unconstrained "
        "economic optimum still rejects too many applicants because the break-even risk is only 13.04%."
    )

    save_table(threshold_curve, "economic_underwriting_threshold_curve")
    save_table(selected_thresholds, "economic_underwriting_selected_thresholds")
    save_table(policy_application, "economic_underwriting_policy_application")
    save_table(recommendation, "economic_underwriting_recommendation")

    print(policy_application.to_string(index=False))


if __name__ == "__main__":
    main()
