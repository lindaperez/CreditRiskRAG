# Preprocessing Decisions

This file documents the preprocessing decisions implemented in `0_Preprocessing.ipynb`.

The preprocessing scope starts after Cleaning. It consumes the accepted-loan starter matrix, binary target, and traceability files produced by `Cleaning/Accepted_Loan_Cleaning.ipynb`, then creates model-ready train, validation, and test matrices for binary `target_bad` prediction.

The core principle is simple: all learned preprocessing parameters must be fit on the training split only. Validation and test rows are transformed using training medians, training category levels, and training scaling statistics to avoid leakage.

## Source Notebook Scope

`0_Preprocessing.ipynb` prepares the cleaned accepted-loan features for modeling.

| Input Artifact | Source | Preprocessing Role |
| --- | --- | --- |
| `accepted_X_starter.parquet` | Cleaning output | Leakage-controlled starter feature matrix. |
| `accepted_y_target_bad.parquet` | Cleaning output | Binary target, where `1` means charged off and `0` means fully paid. |
| `accepted_traceability.parquet` | Cleaning output | Traceability and issue-date fields used for chronological splitting and audit. |

Current input summary:

| Metric | Value |
| --- | ---: |
| Rows | 1,345,310 |
| Starter features | 38 |
| Target bad rate | 19.9626% |
| Earliest issue date | 2007-06-01 |
| Latest issue date | 2018-12-01 |

## Executive Decisions

| Decision Group | Decision | Reason |
| --- | --- | --- |
| Split method | Use chronological train, validation, and test split by `issue_d_dt` | Credit-risk models must be evaluated forward in time. Random splits can overstate performance when portfolio mix, underwriting policy, or macro conditions drift. |
| Split proportions | Target 70% train, 15% validation, 15% test | Provides enough training history while preserving forward validation and holdout test periods. |
| Leakage control | Do not fit imputation, scaling, or categorical levels before splitting | Fitting preprocessing on the full dataset would leak future validation/test distribution into training. |
| Baseline sparse-feature policy | Exclude `mths_since_last_record` from baseline | It is highly missing and can behave as a structural public-record absence signal; baseline should remain simpler and more stable. |
| Missingness challenger | Include `mths_since_last_record` and add `mths_since_last_record_is_missing` | Tests whether the high-missingness public-record feature provides stable lift when missingness is explicitly represented. |
| Missing indicators | Add indicators for `mths_since_last_delinq` and `emp_length_years` in baseline | Missingness can be informative: no delinquency record and missing employment history are not always random missing values. |
| Numeric treatment | Median-impute numeric features, then standardize using train mean and train standard deviation | Produces complete numeric model matrices while avoiding validation/test leakage. |
| Categorical treatment | Fill missing categories as `Missing`, one-hot encode using train categories, and align validation/test columns to train | Prevents category discovery from using future data and guarantees consistent model columns across splits. |
| Grade/sub-grade challenger | Export no-grade/sub-grade versions by dropping encoded `grade_*` and `sub_grade_*` columns | Tests dependence on LendingClub's internal risk grade, which may embed prior underwriting and pricing decisions. |
| Export policy | Save model-ready matrices, split targets, split labels, traceability, feature manifests, QA tables, and metadata | Keeps modeling reproducible and auditable. |

## Chronological Split Decision

The notebook sorts loans by `issue_d_dt` and creates forward-looking train, validation, and test windows.

| Split | Rule | Rows | Bad Rate | Issue Date Range |
| --- | --- | ---: | ---: | --- |
| Train | `issue_d_dt <= 2016-04-01` | 962,641 | 18.8300% | 2007-06-01 to 2016-04-01 |
| Validation | `2016-04-01 < issue_d_dt <= 2017-02-01` | 186,920 | 24.6763% | 2016-05-01 to 2017-02-01 |
| Test | `issue_d_dt > 2017-02-01` | 195,749 | 21.0315% | 2017-03-01 to 2018-12-01 |

The higher validation and test bad rates are expected in a time-based split and are part of why chronological validation is required. The model must generalize to later vintages, not only randomly held-out historical loans.

## Feature Set Decisions

The notebook builds two primary feature sets from the cleaned 38-feature starter matrix.

| Feature Set | Input Feature Count | Encoded Column Count | Missing Values After Preprocessing | Decision |
| --- | ---: | ---: | ---: | --- |
| `baseline` | 37 | 100 | 0 | Excludes `mths_since_last_record`; uses lower-risk sparse-feature policy. |
| `missingness_challenger` | 38 | 102 | 0 | Includes `mths_since_last_record` plus a missingness indicator. |

## Baseline Feature Policy

The baseline starts from the cleaned starter features and excludes:

| Feature | Decision | Reason |
| --- | --- | --- |
| `mths_since_last_record` | Exclude from baseline | Very high missingness. Missingness can mean no public record, so the feature is informative but structurally complex. It is better tested in a challenger model. |

Baseline missingness indicators:

| Indicator | Source Feature | Decision | Reason |
| --- | --- | --- | --- |
| `mths_since_last_delinq_is_missing` | `mths_since_last_delinq` | Add to baseline | Missing can mean no recent delinquency record, which is risk-relevant rather than ordinary missingness. |
| `emp_length_years_is_missing` | `emp_length_years` | Add to baseline | Employment-length missingness can reflect reporting behavior or data quality and may carry weak risk signal. |

## Missingness Challenger Policy

The missingness challenger keeps all baseline features and adds `mths_since_last_record`.

| Feature / Indicator | Decision | Reason |
| --- | --- | --- |
| `mths_since_last_record` | Include in challenger | Tests whether months since last public record has stable predictive value despite high missingness. |
| `mths_since_last_record_is_missing` | Add in challenger | Separates the value of public-record absence from the numeric months-since value. |

This challenger should be accepted only if validation and test performance improve without unstable feature reliance, calibration damage, or poor reason-code behavior.

## Numeric Preprocessing Decisions

Numeric columns are inferred from the selected feature set and transformed with train-only statistics.

| Step | Decision | Implementation Detail |
| --- | --- | --- |
| Numeric coercion | Convert selected numeric columns with numeric coercion | Invalid numeric parses become missing and are handled by train median imputation. |
| Imputation | Fit median on train only | Validation and test use the train median, not their own median. |
| Scaling | Standardize with train mean and train standard deviation | Zero or missing standard deviations are replaced with `1.0` to avoid division errors. |
| Missing indicators | Add selected indicators before imputation/scaling output is finalized | Indicators preserve informative missingness that median imputation would otherwise hide. |

Baseline numeric source features:

`loan_amnt`, `term_months`, `int_rate_clean`, `installment`, `emp_length_years`, `annual_inc`, `dti`, `delinq_2yrs`, `fico_mean`, `inq_last_6mths`, `mths_since_last_delinq`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util_clean`, `total_acc`, `credit_history_years`, `collections_12_mths_ex_med`, `acc_now_delinq`, `tot_coll_amt`, `tot_cur_bal`, `acc_open_past_24mths`, `avg_cur_bal`, `bc_open_to_buy`, `bc_util`, `mort_acc`, `pub_rec_bankruptcies`, `tax_liens`, `total_bal_ex_mort`, `total_bc_limit`, `total_il_high_credit_limit`.

The missingness challenger adds `mths_since_last_record` to this numeric list.

## Categorical Preprocessing Decisions

Categorical columns are one-hot encoded with train-only category discovery.

| Step | Decision | Reason |
| --- | --- | --- |
| Missing categories | Fill missing categorical values as `Missing` | Allows missing category status to remain visible to the model. |
| Category discovery | Learn category levels from train only | Prevents validation/test category information from leaking into training. |
| One-hot encoding | Create binary indicator columns for train categories | Produces model-ready numeric matrices. |
| Column alignment | Reindex validation/test to train encoded columns | Guarantees the same feature order and shape across all splits. |
| Unseen categories | Encode as all-zero across known train category columns | Avoids creating new validation/test columns that the model was not trained on. |

Categorical source features:

`grade`, `sub_grade`, `home_ownership`, `verification_status`, `purpose`, `application_type`.

## Grade And Sub-Grade Decision

`grade` and `sub_grade` are strong predictors but policy-sensitive because they are LendingClub-assigned risk grades. They may encode prior underwriting, pricing, or proprietary credit policy. The notebook therefore exports two additional no-grade/sub-grade matrix families by dropping all encoded `grade_*` and `sub_grade_*` columns.

| Export Family | Source Matrix | Original Columns | Dropped Grade/Sub-Grade Columns | Export Columns | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| `baseline_no_grade_subgrade` | `baseline_*_X.parquet` | 100 | 42 | 58 | Use to test baseline performance without direct grade/sub-grade dependence. |
| `missingness_challenger_no_grade_subgrade` | `missingness_challenger_*_X.parquet` | 102 | 42 | 60 | Use to test the missingness challenger without direct grade/sub-grade dependence. |

Dropped encoded columns include `grade_A` through `grade_G` and `sub_grade_A1` through `sub_grade_G5`.

## QA Decisions

The notebook enforces these checks before downstream modeling:

| QA Check | Required Result | Current Result |
| --- | --- | --- |
| No missing values in preprocessed matrices | All train, validation, and test matrices must have zero missing values | Passed |
| Consistent columns by feature set | Validation and test columns must exactly match train columns | Passed |
| Chronological split available | `issue_d_dt` must exist and parse as valid dates | Passed |
| Target available | `target_bad` must exist in the target frame | Passed |
| No grade/sub-grade leakage in no-grade exports | No columns containing grade/sub-grade remain after drop | Passed |

Current QA summary:

| Feature Set | Split | Rows | Columns | Missing Values | Same Columns As Train |
| --- | --- | ---: | ---: | ---: | --- |
| `baseline` | Train | 962,641 | 100 | 0 | True |
| `baseline` | Validation | 186,920 | 100 | 0 | True |
| `baseline` | Test | 195,749 | 100 | 0 | True |
| `missingness_challenger` | Train | 962,641 | 102 | 0 | True |
| `missingness_challenger` | Validation | 186,920 | 102 | 0 | True |
| `missingness_challenger` | Test | 195,749 | 102 | 0 | True |

## Output Artifacts

Artifacts are generated under `Modeling/Preprocessing/preprocessing_outputs/`.

### Model Matrices

| Artifact Pattern | Purpose |
| --- | --- |
| `baseline_train_X.parquet`, `baseline_validation_X.parquet`, `baseline_test_X.parquet` | Main baseline encoded/scaled matrices. |
| `missingness_challenger_train_X.parquet`, `missingness_challenger_validation_X.parquet`, `missingness_challenger_test_X.parquet` | Challenger matrices with `mths_since_last_record`. |
| `baseline_no_grade_subgrade_train_X.parquet`, `baseline_no_grade_subgrade_validation_X.parquet`, `baseline_no_grade_subgrade_test_X.parquet` | Baseline matrices with encoded grade/sub-grade columns removed. |
| `missingness_challenger_no_grade_subgrade_train_X.parquet`, `missingness_challenger_no_grade_subgrade_validation_X.parquet`, `missingness_challenger_no_grade_subgrade_test_X.parquet` | Missingness challenger matrices with encoded grade/sub-grade columns removed. |

### Targets And Traceability

| Artifact Pattern | Purpose |
| --- | --- |
| `train_y.parquet`, `validation_y.parquet`, `test_y.parquet` | Split target vectors. |
| `train_traceability.parquet`, `validation_traceability.parquet`, `test_traceability.parquet` | Split traceability frames for audit and date-based review. |
| `train_labels.parquet`, `validation_labels.parquet`, `test_labels.parquet` | Split label markers. |

### Tables And Metadata

| Table | Purpose |
| --- | --- |
| `preprocessing_input_summary.csv` | Input row count, feature count, target bad rate, and issue-date range. |
| `preprocessing_chronological_split_rule.csv` | Date rules used for train, validation, and test. |
| `preprocessing_split_summary.csv` | Split row counts, bad rates, and date ranges. |
| `preprocessing_feature_set_summary.csv` | Baseline and challenger feature counts and exclusions. |
| `preprocessing_baseline_feature_manifest.csv` | Baseline feature roles before final one-hot expansion. |
| `preprocessing_challenger_feature_manifest.csv` | Missingness challenger feature roles before final one-hot expansion. |
| `preprocessing_baseline_shapes.csv` | Baseline output shapes and missing-value counts. |
| `preprocessing_challenger_shapes.csv` | Missingness challenger output shapes and missing-value counts. |
| `preprocessing_qa_summary.csv` | Final QA checks for missing values and column consistency. |
| `preprocessing_metadata.json` | Split shares, cutoffs, missingness indicator settings, and numeric/categorical source columns. |
| `preprocessing_no_grade_subgrade_export_summary.csv` | Baseline no-grade/sub-grade export summary. |
| `preprocessing_missingness_no_grade_subgrade_export_summary.csv` | Missingness challenger no-grade/sub-grade export summary. |
| `preprocessing_*_dropped_features.csv` | Encoded grade/sub-grade columns dropped from no-grade/sub-grade exports. |

## Modeling Guidance

Use these datasets in the following order:

1. Start with `baseline_*_X.parquet` for the primary leakage-controlled benchmark.
2. Compare against `missingness_challenger_*_X.parquet` to test whether `mths_since_last_record` provides stable lift.
3. Compare against no-grade/sub-grade exports to quantify dependence on LendingClub-assigned risk grades.
4. Select models using validation performance, calibration, stability, and reason-code suitability.
5. Report final performance on the test period only after model and threshold choices are frozen.

## Final Recommendation

The recommended first modeling dataset is the baseline matrix because it avoids the highest-missingness feature while preserving explicit indicators for informative missingness. The missingness challenger and no-grade/sub-grade exports should be used as sensitivity analyses:

1. `baseline`: primary benchmark.
2. `missingness_challenger`: tests high-missingness public-record signal.
3. `baseline_no_grade_subgrade`: tests model value without LendingClub risk grades.
4. `missingness_challenger_no_grade_subgrade`: tests both high-missingness treatment and grade/sub-grade independence.

For credit-risk governance, a challenger should replace the baseline only if it improves validation and test results without introducing unstable missingness reliance, circular underwriting logic, calibration deterioration, or weak borrower-facing explanations.
