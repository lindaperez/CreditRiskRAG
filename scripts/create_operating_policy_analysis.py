from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELING_OUTPUT_ROOT = PROJECT_ROOT / "Modeling" / "modeling_outputs"
FINAL_TABLE_DIR = MODELING_OUTPUT_ROOT / "final_comparison" / "tables"
MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gradient_boosting": "HistGradientBoosting",
    "lightgbm": "LightGBM",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
}


def save_table(df: pd.DataFrame, name: str) -> Path:
    path = FINAL_TABLE_DIR / f"final_model_{name}.csv"
    df.to_csv(path, index=False)
    print("Saved:", path)
    return path


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def normalize_review_table(df: pd.DataFrame, dataset: str, dataset_label: str) -> pd.DataFrame:
    out = df.copy()
    if "model_label" not in out.columns:
        out["model_label"] = out["model_family"].map(MODEL_LABELS)
    out["dataset"] = out.get("dataset", dataset)
    out["dataset_label"] = out.get("dataset_label", dataset_label)
    if "model_dataset_label" not in out.columns:
        out["model_dataset_label"] = out["model_label"] + " | " + dataset_label
    return out


def main() -> None:
    pr_auc_rank_path = FINAL_TABLE_DIR / "final_model_missingness_challenger_ranking_by_validation_pr_auc.csv"
    pr_auc_rank = pd.read_csv(pr_auc_rank_path)

    renamed = pr_auc_rank.rename(columns={
        "threshold": "f1_operating_threshold",
        "precision": "f1_threshold_precision",
        "recall": "f1_threshold_recall",
        "f1": "f1_threshold_f1",
        "predicted_reject_share": "f1_threshold_predicted_reject_share",
        "false_rejection_share_among_rejects": "f1_threshold_false_rejection_share_among_rejects",
    })
    renamed["selection_metric"] = "validation_pr_auc"
    renamed["operating_policy_warning"] = (
        "Reject/share columns are from the automated best-F1 threshold, not from PR-AUC itself. "
        "Use fixed-review-volume policy tables for business action thresholds."
    )
    save_table(renamed, "pr_auc_ranking_with_f1_threshold_warning")

    review_tables = []
    baseline_review = read_csv_if_exists(FINAL_TABLE_DIR / "final_model_review_volume_comparison.csv")
    if baseline_review is not None:
        review_tables.append(normalize_review_table(
            baseline_review,
            "baseline_with_grade_subgrade",
            "baseline with grade/subgrade",
        ))

    missingness_review = read_csv_if_exists(
        MODELING_OUTPUT_ROOT / "missingness_challenger" / "tables" / "missingness_challenger_review_volume_precision.csv"
    )
    if missingness_review is not None:
        review_tables.append(missingness_review)

    missingness_no_grade_review = read_csv_if_exists(
        MODELING_OUTPUT_ROOT
        / "missingness_challenger_no_grade_subgrade"
        / "tables"
        / "missingness_challenger_no_grade_subgrade_review_volume_precision.csv"
    )
    if missingness_no_grade_review is not None:
        review_tables.append(missingness_no_grade_review)

    for family in ["lightgbm", "xgboost", "catboost"]:
        no_grade_review = read_csv_if_exists(
            MODELING_OUTPUT_ROOT / family / "tables" / f"{family}_no_grade_subgrade_review_volume_precision.csv"
        )
        if no_grade_review is not None:
            review_tables.append(normalize_review_table(
                no_grade_review,
                "baseline_no_grade_subgrade",
                "baseline no grade/subgrade",
            ))

    all_review = pd.concat(review_tables, ignore_index=True, sort=False)
    all_review = all_review[all_review["split"].isin(["validation", "test"])].copy()
    save_table(all_review, "fixed_review_volume_policy_all_available")

    policy_caps = [5.0, 10.0, 20.0, 30.0]
    top_rank = pr_auc_rank[["selection_rank", "model_family", "model", "dataset", "model_dataset_label"]].head(12)
    top_policy = all_review.merge(
        top_rank,
        on=["model_family", "model", "dataset", "model_dataset_label"],
        how="inner",
    )
    top_policy = top_policy[top_policy["review_pct"].isin(policy_caps)].copy()
    top_policy = top_policy.sort_values(["split", "review_pct", "selection_rank"])
    top_policy["business_policy"] = top_policy["review_pct"].map(lambda pct: f"review/risk-action top {pct:g}% only")
    top_policy["avoided_auto_reject_share_vs_f1_threshold"] = np.nan

    f1_reject_lookup = pr_auc_rank.set_index(["model_family", "model", "dataset"])["predicted_reject_share"].to_dict()
    for idx, row in top_policy.iterrows():
        f1_reject_share = f1_reject_lookup.get((row["model_family"], row["model"], row["dataset"]))
        if f1_reject_share is not None:
            top_policy.loc[idx, "avoided_auto_reject_share_vs_f1_threshold"] = round(
                float(f1_reject_share - row["review_pct"] / 100.0),
                6,
            )

    save_table(top_policy, "top_pr_auc_fixed_review_volume_policy")

    missing_policy_rows = []
    available_keys = set(zip(
        all_review["model_family"],
        all_review["model"],
        all_review["dataset"],
    ))
    for _, row in top_rank.iterrows():
        key = (row["model_family"], row["model"], row["dataset"])
        if key not in available_keys:
            missing_policy_rows.append({
                "selection_rank": row["selection_rank"],
                "model_dataset_label": row["model_dataset_label"],
                "model": row["model"],
                "dataset": row["dataset"],
                "missing_reason": "No fixed-review-volume table was available for this model/dataset artifact.",
            })
    save_table(pd.DataFrame(missing_policy_rows), "top_pr_auc_fixed_review_volume_missing")


if __name__ == "__main__":
    main()
