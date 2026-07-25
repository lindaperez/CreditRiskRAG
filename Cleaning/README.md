# Accepted Loan Cleaning README

This folder documents and stores the repeatable cleaning workflow for the LendingClub accepted-loan credit-risk dataset. The primary notebook is `Accepted_Loan_Cleaning.ipynb`.

The cleaning scope is accepted/originated loans only. Rejected applications are intentionally excluded because they do not have observed repayment outcomes and therefore cannot be labeled as good or bad loans without a separate reject-inference framework.

## Business Objective

The notebook converts raw LendingClub accepted-loan records into a leakage-controlled modeling frame for supervised default-risk modeling. From a bank credit-risk perspective, the cleaning process is designed to answer one question:

> Based on information plausibly available at or near underwriting, can we predict whether an originated loan will become a bad loan?

The output is not a fully preprocessed machine-learning matrix. It is a clean, auditable feature and target handoff for downstream chronological splitting, train-only imputation, encoding, scaling, model fitting, calibration, and explainability.

## Source Data

The notebook reads the accepted-loan file from the project data archive:

- `Data/archive/accepted_2007_to_2018Q4.csv.gz`, or
- the extracted equivalent under `Data/archive/accepted_2007_to_2018q4.csv/`.

Raw data is preserved. All treated outputs are written under:

- `CreditRiskRAG/Cleaning/cleaning_outputs/`

## Target Definition

The supervised baseline uses only stable terminal repayment outcomes:

| Loan Status | Target Mapping | Reason |
| --- | ---: | --- |
| `Fully Paid` | `target_bad = 0` | Terminal good outcome. |
| `Charged Off` | `target_bad = 1` | Terminal bad outcome. |

The strict baseline excludes unresolved or unstable statuses such as `Current`, `In Grace Period`, `Late (16-30 days)`, `Late (31-120 days)`, `Issued`, `Default`, policy-exception statuses, and missing statuses. These are not treated as good loans because doing so would contaminate the target with loans that have not reached a stable final repayment outcome.

In the current full-cleaning output:

| Metric | Value |
| --- | ---: |
| Completed modeling rows | 1,345,310 |
| Fully paid rows | 1,076,751 |
| Charged off rows | 268,559 |
| Observed bad rate | 19.9626% |
| Starter feature count | 38 |
| Exact duplicate rows removed | 0 |

## What The Notebook Does

### 1. Configures Paths And Execution Mode

The notebook discovers the project root, locates the accepted-loan source file, creates the cleaning output directories, and supports both development sampling and full cleaning. The safer development default is sample cleaning; full cleaning should be used when regenerating final modeling artifacts.

### 2. Defines Feature Governance

The workflow explicitly separates columns into governance groups:

- Application-time candidate fields
- Identifiers
- Target and outcome fields
- Clear post-origination leakage fields
- Fields requiring timing, compliance, or business review
- Approved 38-feature starter model set

This prevents accidental inclusion of identifiers, servicing fields, recovery information, credit updates after origination, hardship information, or settlement fields in the model matrix.

### 3. Audits Source Schema And Loan Status Counts

The notebook reads the raw header, checks which requested columns are available, scans full-file `loan_status` counts in chunks, and exports the status distribution. This allows the cleaning sample and target definition to be tied back to the full accepted-loan population.

### 4. Builds A Stratified Cleaning Sample When Needed

For local development and QA, the notebook can build a stratified sample by `loan_status`. The sample is restricted to the strict terminal training statuses so that charged-off outcomes remain represented while non-training statuses stay outside the baseline target.

### 5. Performs Structural Data Quality Checks

Before type conversion, the notebook checks for structural defects such as duplicate column names, missing required fields, unexpected target statuses, and text/category casing or whitespace issues. Serious structural failures are designed to stop the workflow early.

### 6. Creates Clean Helper Fields

Raw LendingClub fields are preserved for audit, but cleaner helper fields are created for modeling:

| Raw Field | Clean Helper | Purpose |
| --- | --- | --- |
| `term` | `term_months` | Converts strings like `36 months` into numeric loan term. |
| `int_rate` | `int_rate_clean` | Converts percentage strings into numeric interest rates. |
| `revol_util` | `revol_util_clean` | Converts revolving utilization percentages into numeric values. |
| `emp_length` | `emp_length_years` | Converts employment-length labels into numeric years. |
| `fico_range_low`, `fico_range_high` | `fico_mean` | Creates one FICO summary field from the reported range. |
| `issue_d`, `earliest_cr_line` | `credit_history_years` | Measures bureau history age at loan issue. |

The notebook validates expected dtypes, valid term values, FICO consistency, date parsing, and non-negative credit-history years.

## 36-Feature Data Dictionary

This project also exports a no-grade/sub-grade starter feature set with 36 features in `cleaning_starter_features_no_grade_subgrade.csv`. It removes `grade` and `sub_grade` from the 38-feature starter set to support a challenger model that does not directly rely on LendingClub's internal risk grade. Definitions below are based on `../LCDataDictionary.xlsx`; derived fields are mapped back to their raw dictionary fields.

| Clean Feature | Source Dictionary Field | Definition |
| --- | --- | --- |
| `loan_amnt` | `loan_amnt` | Listed amount of the loan applied for by the borrower. If the credit department reduced the amount, the reduction is reflected here. |
| `term_months` | `term` | Number of payments on the loan, in months. LendingClub values are 36 or 60. |
| `int_rate_clean` | `int_rate` | Interest rate on the loan, parsed from the raw percent field. |
| `installment` | `installment` | Monthly payment owed by the borrower if the loan originates. |
| `emp_length_years` | `emp_length` | Employment length in years, where 0 means less than one year and 10 means ten or more years. |
| `home_ownership` | `home_ownership` | Home ownership status provided by the borrower during registration or obtained from the credit report, such as rent, own, mortgage, or other. |
| `annual_inc` | `annual_inc` | Self-reported annual income provided by the borrower during registration. |
| `verification_status` | `verification_status` | Indicates whether income was verified by LendingClub, not verified, or whether the income source was verified. |
| `purpose` | `purpose` | Borrower-provided category for the loan request. |
| `dti` | `dti` | Debt-to-income ratio using total monthly debt obligations, excluding mortgage and the requested LendingClub loan, divided by self-reported monthly income. |
| `delinq_2yrs` | `delinq_2yrs` | Number of 30+ days past-due delinquency incidences in the borrower's credit file over the past two years. |
| `fico_mean` | `fico_range_low`, `fico_range_high` | Average of the lower and upper FICO range boundaries reported at loan origination. |
| `inq_last_6mths` | `inq_last_6mths` | Number of credit inquiries in the past six months, excluding auto and mortgage inquiries. |
| `mths_since_last_delinq` | `mths_since_last_delinq` | Number of months since the borrower's last delinquency. |
| `mths_since_last_record` | `mths_since_last_record` | Number of months since the last public record. |
| `open_acc` | `open_acc` | Number of open credit lines in the borrower's credit file. |
| `pub_rec` | `pub_rec` | Number of derogatory public records. |
| `revol_bal` | `revol_bal` | Total revolving credit balance. |
| `revol_util_clean` | `revol_util` | Revolving line utilization rate, or credit used relative to available revolving credit. |
| `total_acc` | `total_acc` | Total number of credit lines currently in the borrower's credit file. |
| `credit_history_years` | `issue_d`, `earliest_cr_line` | Years between loan funding month and the borrower's earliest reported credit-line opening month. |
| `collections_12_mths_ex_med` | `collections_12_mths_ex_med` | Number of collections in the past 12 months, excluding medical collections. |
| `acc_now_delinq` | `acc_now_delinq` | Number of accounts on which the borrower is currently delinquent. |
| `tot_coll_amt` | `tot_coll_amt` | Total collection amounts ever owed. |
| `tot_cur_bal` | `tot_cur_bal` | Total current balance of all accounts. |
| `acc_open_past_24mths` | `acc_open_past_24mths` | Number of trades opened in the past 24 months. |
| `avg_cur_bal` | `avg_cur_bal` | Average current balance of all accounts. |
| `bc_open_to_buy` | `bc_open_to_buy` | Total open-to-buy amount on revolving bankcards. |
| `bc_util` | `bc_util` | Ratio of total current balance to high credit or credit limit for all bankcard accounts. |
| `mort_acc` | `mort_acc` | Number of mortgage accounts. |
| `pub_rec_bankruptcies` | `pub_rec_bankruptcies` | Number of public-record bankruptcies. |
| `tax_liens` | `tax_liens` | Number of tax liens. |
| `total_bal_ex_mort` | `total_bal_ex_mort` | Total credit balance excluding mortgage. |
| `total_bc_limit` | `total_bc_limit` | Total bankcard high credit or credit limit. |
| `total_il_high_credit_limit` | `total_il_high_credit_limit` | Total installment high credit or credit limit. |
| `application_type` | `application_type` | Indicates whether the loan is an individual application or a joint application with two co-borrowers. |

### 7. Standardizes Missing Labels And Removes Duplicates

Conservative text missing-value tokens are converted to nulls. Exact duplicate rows are removed so duplicate records do not overweight a borrower outcome. The current full-cleaning output found no exact duplicate rows.

### 8. Creates The Modeling Frame

The notebook creates four key objects:

- `completed`: cleaned rows with strict terminal outcomes only
- `X_starter`: 38 approved starter predictors
- `y`: binary `target_bad`
- `traceability`: non-feature audit fields needed to connect rows back to loan status and issue timing

The starter matrix intentionally preserves missing values. Imputation, categorical encoding, scaling, and split-specific transformations are handled later in the modeling pipeline to avoid train/test leakage.

### 9. Documents Missingness Strategy

Missingness is profiled and exported with recommended handling. Important credit-risk examples:

- `mths_since_last_record` has high missingness and is marked for challenger use only because missingness can structurally mean no public record.
- `mths_since_last_delinq` is kept with a missingness indicator because missingness can mean no recent delinquency record.
- `emp_length_years` has low but potentially informative missingness and should be imputed after split, with an optional missingness indicator.
- Low-missingness numeric fields should be imputed only after the chronological train/validation/test split.

## Output Artifacts

### Datasets

Generated under `Cleaning/cleaning_outputs/datasets/`:

| Artifact | Description |
| --- | --- |
| `accepted_cleaned_completed_frame.parquet` | Full cleaned terminal-outcome frame with audit and helper fields. |
| `accepted_X_starter.parquet` | Leakage-controlled starter feature matrix. |
| `accepted_y_target_bad.parquet` | Binary target frame. |
| `accepted_traceability.parquet` | Traceability fields such as loan id, status, issue date, and target definition. |

### QA Tables

Generated under `Cleaning/cleaning_outputs/tables/`:

| Table | Purpose |
| --- | --- |
| `cleaning_full_loan_status_counts.csv` | Full accepted-loan status distribution before strict target filtering. |
| `cleaning_status_sample_plan.csv` | Stratified sample plan for development cleaning runs. |
| `cleaning_structural_assertions.csv` | Pass/fail structural checks. |
| `cleaning_structural_column_audit.csv` | Column-name and availability audit. |
| `cleaning_categorical_value_audit.csv` | Category whitespace/casing review. |
| `cleaning_type_conversion_assertions.csv` | Type-conversion QA checks. |
| `cleaning_type_conversion_dtype_audit.csv` | Dtype audit after conversion. |
| `cleaning_derived_field_quality.csv` | Quality checks for helper fields. |
| `cleaning_missing_label_standardization.csv` | Missing-label standardization summary. |
| `cleaning_duplicate_summary.csv` | Exact duplicate removal summary. |
| `cleaning_target_summary.csv` | Final binary target distribution. |
| `cleaning_feature_exclusion_audit.csv` | Identifier, target, leakage, helper-replacement, and review-required exclusions. |
| `cleaning_missingness_strategy.csv` | Feature-level missingness treatment recommendations. |
| `cleaning_modeling_frame_summary.csv` | Final row counts, feature count, and bad rate. |
| `cleaning_export_paths.csv` | Paths to exported datasets. |

### Plot

Generated under `Cleaning/cleaning_outputs/plots/`:

- `cleaning_starter_feature_missingness.png`: missingness profile for the starter feature set.

## Leakage And Compliance Controls

The starter feature matrix excludes:

- Direct identifiers: `id`, `member_id`, `url`
- Target/outcome fields: `loan_status`, `target_bad`, `target_definition`
- Payment and balance fields observed after origination
- Recovery and collections fields
- Last-payment and last-credit-pull fields
- Hardship and settlement fields
- Raw fields replaced by cleaner helper fields
- Text, geography, marketplace, or policy fields requiring additional review

Fields such as `grade`, `sub_grade`, and `int_rate_clean` are included in the starter feature set but remain policy-sensitive. They may encode prior LendingClub underwriting or pricing decisions, so later modeling should compare baseline results against challenger models that remove these fields.

## Downstream Handoff

The modeling notebooks should use the cleaning outputs as follows:

1. Load `accepted_X_starter.parquet`, `accepted_y_target_bad.parquet`, and `accepted_traceability.parquet`.
2. Split chronologically using issue timing from traceability or the completed cleaned frame.
3. Fit all imputers, encoders, scalers, class balancing, and feature selectors on training data only.
4. Validate calibration, rank ordering, stability over time, and explainability.
5. Run challenger models excluding policy-sensitive fields such as `grade`, `sub_grade`, and potentially `int_rate_clean`.

## Related Documentation

- `FEATURE_CLEANING_DECISIONS.md`: detailed feature-level cleaning and modeling governance.
- `../EDA/Accepted_Loan_EDA.ipynb`: accepted-loan exploratory analysis that informed cleaning decisions.
- `../Modeling/Preprocessing/0_Preprocessing.ipynb`: downstream preprocessing workflow.
