from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import run_missingness_challenger_all_models as base


DATASET = "missingness_challenger_no_grade_subgrade"
DATASET_LABEL = "missingness challenger no grade/subgrade"
base.DATASET_LABELS[DATASET] = DATASET_LABEL
OUTPUT_ROOT = base.MODELING_OUTPUT_ROOT / DATASET
TABLE_DIR = OUTPUT_ROOT / "tables"
MODEL_DIR = OUTPUT_ROOT / "models"
for directory in [TABLE_DIR, MODEL_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def save_table(df: pd.DataFrame, name: str) -> Path:
    path = TABLE_DIR / f"{DATASET}_{name}.csv"
    df.to_csv(path, index=False)
    print("Saved:", path, flush=True)
    return path


def dataset_label_row(model_family: str) -> str:
    return f"{base.MODEL_LABELS[model_family]} | {DATASET_LABEL}"


def train_models() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train = base.load_parquet(f"{DATASET}_train_X")
    X_validation = base.load_parquet(f"{DATASET}_validation_X")
    X_test = base.load_parquet(f"{DATASET}_test_X")
    y_train = base.load_parquet("train_y")["target_bad"].astype(int)
    y_validation = base.load_parquet("validation_y")["target_bad"].astype(int)
    y_test = base.load_parquet("test_y")["target_bad"].astype(int)

    candidate_rows = []
    selected_rows = []
    metric_rows = []
    review_rows = []

    for model_family, config in base.MODEL_CONFIGS.items():
        X_fit, y_fit = base.sample_fit_data(X_train, y_train, config["sample_rows"])
        fitted_models = {}
        family_rows = []
        for candidate in config["candidates"]:
            candidate_name = f"{candidate['candidate']}_{DATASET}"
            print(f"Training {candidate_name}: {candidate['params']}", flush=True)
            start = time.perf_counter()
            model = base.build_model(model_family, candidate["params"], y_fit)
            model.fit(X_fit, y_fit, **base.fit_kwargs(model_family, y_fit))
            seconds = time.perf_counter() - start

            validation_score = base.predict_positive_probability(model, X_validation)
            best_threshold, best_f1, best_precision, best_recall = base.threshold_for_best_f1(
                y_validation,
                validation_score,
            )
            precision_threshold, precision_f1, precision_value, precision_recall = base.threshold_for_target_precision(
                y_validation,
                validation_score,
            )
            row = {
                "model_family": model_family,
                "model_label": base.MODEL_LABELS[model_family],
                "candidate": candidate_name,
                "dataset": DATASET,
                "dataset_label": DATASET_LABEL,
                "model_dataset_label": dataset_label_row(model_family),
                "params": json.dumps(candidate["params"], sort_keys=True),
                "fit_rows": len(X_fit),
                "fit_bad_rate": round(float(y_fit.mean()), 6),
                "fit_seconds": round(float(seconds), 3),
                "roc_auc": round(float(base.roc_auc_score(y_validation, validation_score)), 6),
                "pr_auc": round(float(base.average_precision_score(y_validation, validation_score)), 6),
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
            print(
                f"Finished {candidate_name}: f1={best_f1:.4f}, pr_auc={row['pr_auc']:.4f}, seconds={seconds:.1f}",
                flush=True,
            )

        family_results = pd.DataFrame(family_rows).sort_values(
            ["best_f1", "best_f1_precision", "pr_auc"],
            ascending=False,
        )
        candidate_rows.extend(family_results.to_dict("records"))
        winner = family_results.iloc[0]
        selected_rows.append(winner.to_dict())
        selected_model = fitted_models[winner["candidate"]]
        model_path = MODEL_DIR / f"{winner['candidate']}_selected_model.joblib"
        joblib.dump(selected_model, model_path)

        scores = {
            "train": base.predict_positive_probability(selected_model, X_train),
            "validation": base.predict_positive_probability(selected_model, X_validation),
            "test": base.predict_positive_probability(selected_model, X_test),
        }
        y_parts = {"train": y_train, "validation": y_validation, "test": y_test}
        for split, y_part in y_parts.items():
            metric_rows.append(base.evaluate_at_threshold(
                model_family,
                winner["candidate"],
                DATASET,
                split,
                y_part,
                scores[split],
                winner["best_f1_threshold"],
                "best_validation_f1",
            ))
            metric_rows.append(base.evaluate_at_threshold(
                model_family,
                winner["candidate"],
                DATASET,
                split,
                y_part,
                scores[split],
                winner["target_precision_threshold"],
                "target_validation_precision",
            ))

        for split in ["validation", "test"]:
            review_rows.extend(base.review_volume_metrics(
                model_family,
                winner["candidate"],
                DATASET,
                split,
                y_parts[split],
                scores[split],
            ).to_dict("records"))

    for frame_rows in [candidate_rows, selected_rows, metric_rows, review_rows]:
        for row in frame_rows:
            row["dataset_label"] = DATASET_LABEL
            row["model_dataset_label"] = dataset_label_row(row["model_family"])

    return pd.DataFrame(candidate_rows), pd.DataFrame(selected_rows), pd.DataFrame(metric_rows), pd.DataFrame(review_rows)


def main() -> None:
    candidate_results, selected_candidates, selected_metrics, review_volume = train_models()
    save_table(candidate_results, "candidate_results")
    save_table(selected_candidates, "selected_candidates")
    save_table(selected_metrics, "selected_model_metrics")
    save_table(review_volume, "review_volume_precision")

    comparison = selected_metrics[
        (selected_metrics["split"].isin(["validation", "test"]))
        & (selected_metrics["operating_point"] == "best_validation_f1")
    ].copy()
    comparison = comparison.sort_values(["split", "f1", "precision", "pr_auc"], ascending=[True, False, False, False])
    save_table(comparison, "validation_test_comparison")

    validation_rank = comparison[comparison["split"] == "validation"].sort_values(
        ["f1", "precision", "pr_auc"],
        ascending=False,
    ).reset_index(drop=True)
    validation_rank["selection_rank"] = validation_rank.index + 1
    save_table(validation_rank, "ranking_by_validation_f1")

    test_rank = comparison[comparison["split"] == "test"].sort_values(
        ["f1", "precision", "pr_auc"],
        ascending=False,
    ).reset_index(drop=True)
    test_rank["test_rank"] = test_rank.index + 1
    save_table(test_rank, "ranking_by_test_f1")

    save_table(validation_rank.iloc[[0]], "recommendation")
    print(validation_rank[[
        "selection_rank", "model_dataset_label", "model", "f1", "precision", "recall", "pr_auc", "roc_auc",
        "predicted_reject_share",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
