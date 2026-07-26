from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import brier_score_loss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "Modeling" / "Preprocessing" / "preprocessing_outputs" / "datasets"
MODELING_OUTPUT_ROOT = PROJECT_ROOT / "Modeling" / "modeling_outputs"
FINAL_OUTPUT_ROOT = MODELING_OUTPUT_ROOT / "final_comparison"
TABLE_DIR = FINAL_OUTPUT_ROOT / "tables"
PLOT_DIR = FINAL_OUTPUT_ROOT / "plots"
for directory in [TABLE_DIR, PLOT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

MODEL_FAMILY = "xgboost"
MODEL_LABEL = "XGBoost"
MODEL_NAME = "xgboost_05_missingness_challenger"
DATASET = "missingness_challenger"
DATASET_LABEL = "missingness challenger"
MODEL_DATASET_LABEL = f"{MODEL_LABEL} | {DATASET_LABEL}"
MODEL_PATH = MODELING_OUTPUT_ROOT / DATASET / "models" / f"{MODEL_NAME}_selected_model.joblib"


def load_parquet(name: str) -> pd.DataFrame:
    path = DATASET_DIR / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


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


def calibration_bins(y_true: pd.Series, y_score: np.ndarray, split: str, calibration_method: str = "raw") -> pd.DataFrame:
    frame = pd.DataFrame({"target_bad": np.asarray(y_true), "predicted_probability": y_score})
    frame["probability_bin"] = pd.cut(
        frame["predicted_probability"],
        bins=np.linspace(0.0, 1.0, 11),
        include_lowest=True,
        right=True,
    )
    grouped = (
        frame
        .groupby("probability_bin", observed=False)
        .agg(
            rows=("target_bad", "size"),
            predicted_probability_mean=("predicted_probability", "mean"),
            observed_bad_rate=("target_bad", "mean"),
            bad_count=("target_bad", "sum"),
        )
        .reset_index()
    )
    grouped["probability_bin"] = grouped["probability_bin"].astype(str)
    grouped["split"] = split
    grouped["model_family"] = MODEL_FAMILY
    grouped["model"] = MODEL_NAME
    grouped["model_dataset_label"] = MODEL_DATASET_LABEL
    grouped["calibration_method"] = calibration_method
    grouped["abs_calibration_error"] = (
        grouped["observed_bad_rate"] - grouped["predicted_probability_mean"]
    ).abs()
    grouped["weighted_abs_calibration_error"] = grouped["abs_calibration_error"] * grouped["rows"]
    return grouped[
        [
            "model_family",
            "model",
            "model_dataset_label",
            "calibration_method",
            "split",
            "probability_bin",
            "rows",
            "bad_count",
            "predicted_probability_mean",
            "observed_bad_rate",
            "abs_calibration_error",
            "weighted_abs_calibration_error",
        ]
    ]


def calibration_summary(
    calibration: pd.DataFrame,
    y_true_by_split: dict[str, pd.Series],
    score_by_split: dict[tuple[str, str], np.ndarray],
) -> pd.DataFrame:
    rows = []
    for (method, split), group in calibration.groupby(["calibration_method", "split"]):
        total_rows = group["rows"].sum()
        ece = group["weighted_abs_calibration_error"].sum() / total_rows
        y_true = y_true_by_split[split]
        y_score = score_by_split[(method, split)]
        rows.append({
            "model_family": MODEL_FAMILY,
            "model": MODEL_NAME,
            "model_dataset_label": MODEL_DATASET_LABEL,
            "calibration_method": method,
            "split": split,
            "rows": int(total_rows),
            "bad_rate": round(float(y_true.mean()), 6),
            "mean_predicted_probability": round(float(np.mean(y_score)), 6),
            "brier_score": round(float(brier_score_loss(y_true, y_score)), 6),
            "pr_auc": round(float(average_precision_score(y_true, y_score)), 6),
            "roc_auc": round(float(roc_auc_score(y_true, y_score)), 6),
            "expected_calibration_error": round(float(ece), 6),
            "max_bin_abs_calibration_error": round(float(group["abs_calibration_error"].max()), 6),
        })
    return pd.DataFrame(rows)


def plot_calibration(calibration: pd.DataFrame) -> None:
    for split, group in calibration.groupby("split"):
        plot_df = group[group["rows"] > 0].copy()
        fig, ax = plt.subplots(figsize=(6.5, 5.2))
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="perfect calibration")
        for method, method_df in plot_df.groupby("calibration_method"):
            ax.plot(
                method_df["predicted_probability_mean"],
                method_df["observed_bad_rate"],
                marker="o",
                label=method,
            )
        ax.set_title(f"Calibration curve: {MODEL_DATASET_LABEL} ({split})")
        ax.set_xlabel("Mean predicted default probability")
        ax.set_ylabel("Observed default rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend()
        save_plot(fig, f"calibration_curve_{MODEL_FAMILY}_{DATASET}_{split}")


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    model = joblib.load(MODEL_PATH)
    X_validation = load_parquet("missingness_challenger_validation_X")
    X_test = load_parquet("missingness_challenger_test_X")
    y_validation = load_parquet("validation_y")["target_bad"].astype(int)
    y_test = load_parquet("test_y")["target_bad"].astype(int)

    raw_score_by_split = {
        "validation": model.predict_proba(X_validation)[:, 1],
        "test": model.predict_proba(X_test)[:, 1],
    }
    y_true_by_split = {
        "validation": y_validation,
        "test": y_test,
    }

    platt = LogisticRegression(solver="lbfgs", random_state=42)
    platt.fit(raw_score_by_split["validation"].reshape(-1, 1), y_validation)

    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(raw_score_by_split["validation"], y_validation)

    score_by_method_split = {
        ("raw", "validation"): raw_score_by_split["validation"],
        ("raw", "test"): raw_score_by_split["test"],
        ("platt_sigmoid", "validation"): platt.predict_proba(raw_score_by_split["validation"].reshape(-1, 1))[:, 1],
        ("platt_sigmoid", "test"): platt.predict_proba(raw_score_by_split["test"].reshape(-1, 1))[:, 1],
        ("isotonic", "validation"): isotonic.predict(raw_score_by_split["validation"]),
        ("isotonic", "test"): isotonic.predict(raw_score_by_split["test"]),
    }

    calibration = pd.concat(
        [
            calibration_bins(y_true_by_split[split], score, split, method)
            for (method, split), score in score_by_method_split.items()
        ],
        ignore_index=True,
    )
    summary = calibration_summary(calibration, y_true_by_split, score_by_method_split)

    calibrator_rows = pd.DataFrame([
        {
            "calibration_method": "platt_sigmoid",
            "calibrator_type": "LogisticRegression",
            "coefficient": float(platt.coef_[0][0]),
            "intercept": float(platt.intercept_[0]),
        },
        {
            "calibration_method": "isotonic",
            "calibrator_type": "IsotonicRegression",
            "coefficient": np.nan,
            "intercept": np.nan,
        },
    ])

    save_table(calibration, "calibration_bins")
    save_table(summary, "calibration_summary")
    save_table(calibrator_rows, "calibration_models")
    try:
        plot_calibration(calibration)
    except Exception as exc:
        print(f"Skipping calibration plot generation after error: {exc}")

    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
