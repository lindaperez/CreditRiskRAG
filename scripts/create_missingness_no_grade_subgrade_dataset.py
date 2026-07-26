from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_OUTPUT_ROOT = PROJECT_ROOT / "Modeling" / "Preprocessing" / "preprocessing_outputs"
DATASET_DIR = PREPROCESSING_OUTPUT_ROOT / "datasets"
TABLE_DIR = PREPROCESSING_OUTPUT_ROOT / "tables"


def save_table(df: pd.DataFrame, name: str) -> Path:
    path = TABLE_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print("Saved:", path)
    return path


def grade_subgrade_columns(columns: pd.Index) -> list[str]:
    return [column for column in columns if column.startswith("grade_") or column.startswith("sub_grade_")]


def main() -> None:
    export_rows = []
    dropped_feature_rows = []
    encoded_feature_rows = []

    for split in ["train", "validation", "test"]:
        source_path = DATASET_DIR / f"missingness_challenger_{split}_X.parquet"
        output_path = DATASET_DIR / f"missingness_challenger_no_grade_subgrade_{split}_X.parquet"
        if not source_path.exists():
            raise FileNotFoundError(source_path)

        source = pd.read_parquet(source_path)
        dropped_columns = grade_subgrade_columns(source.columns)
        if not dropped_columns:
            raise ValueError(f"No grade/sub_grade one-hot columns found in {source_path}")

        output = source.drop(columns=dropped_columns)
        if any("grade" in column.lower() for column in output.columns):
            remaining = [column for column in output.columns if "grade" in column.lower()]
            raise AssertionError(f"Grade/subgrade columns remain after drop: {remaining}")

        output.to_parquet(output_path, index=True)
        print("Saved:", output_path)

        export_rows.append({
            "split": split,
            "source_artifact": source_path.name,
            "export_artifact": output_path.name,
            "rows": output.shape[0],
            "original_columns": source.shape[1],
            "dropped_columns": len(dropped_columns),
            "export_columns": output.shape[1],
        })
        dropped_feature_rows.extend({"feature": column} for column in dropped_columns)
        if split == "train":
            encoded_feature_rows.extend({"feature": column} for column in output.columns)

    dropped_features = pd.DataFrame(dropped_feature_rows).drop_duplicates().sort_values("feature")
    encoded_features = pd.DataFrame(encoded_feature_rows)
    export_summary = pd.DataFrame(export_rows)

    save_table(dropped_features, "preprocessing_missingness_no_grade_subgrade_dropped_features")
    save_table(encoded_features, "preprocessing_missingness_no_grade_subgrade_encoded_features")
    save_table(export_summary, "preprocessing_missingness_no_grade_subgrade_export_summary")


if __name__ == "__main__":
    main()
