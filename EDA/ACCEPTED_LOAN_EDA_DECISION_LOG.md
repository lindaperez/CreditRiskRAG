# Accepted Loan EDA Decision Log

This document records the decisions made in `Accepted_Loan_EDA.ipynb` for the LendingClub accepted-loan population. It is focused on repayment-risk modeling decisions for originated loans and should be read alongside the exported evidence in `accepted_eda_outputs/`.

## Scope

| Item | Decision |
| --- | --- |
| Source notebook | `Final/CreditRiskRAG/EDA/Accepted_Loan_EDA.ipynb` |
| Source data | `Final/Data/archive/accepted_2007_to_2018Q4.csv.gz` |
| Raw file size | 374.4 MB |
| Raw rows / columns | 2,260,701 rows / 151 columns |
| Working EDA sample | 250,000 accepted loans |
| Full loan-status scan | 2,260,701 accepted-loan rows |
| Random seed | 42 |
| Output folder | `Final/CreditRiskRAG/EDA/accepted_eda_outputs/` |

The accepted-loan EDA is an originated-loan repayment-risk analysis. It is not a reject-inference study and should not be interpreted as modeling all applicants who applied to LendingClub.

## Decision Summary

| Area | Decision | Rationale | Follow-up Before Modeling |
| --- | --- | --- | --- |
| Population | Use accepted/originated LendingClub loans only. | Repayment outcomes are observable only after origination. | Document that the model estimates risk conditional on approval. |
| Working sample | Use a reproducible 250,000-row sample for EDA. | The raw accepted file is large, and the sample is sufficient for exploratory summaries and plots. | Run final training and validation on the full eligible modeling population if feasible. |
| Full loan-status imbalance | Compute `loan_status` distribution on all 2,260,701 accepted-loan rows. | The 250,000-row EDA sample is useful for exploration, but class imbalance should be measured on the full dataset. | Use the full-distribution table when reporting target/class imbalance. |
| Current-loan removal | Remove rows where `loan_status == "Current"` from the modeling-oriented EDA dataframe. | `Current` loans are active/censored and do not have final repayment outcomes. Treating them as good outcomes would bias the target. | Keep raw source data unchanged; apply this as an analysis/modeling filter. |
| Feature loading | Load application-time candidates, identifiers, target fields, and selected leakage-demo fields rather than all 151 columns. | Keeps EDA memory-aware and focused on credit-risk variables while still auditing leakage. | Re-run the leakage audit on the full schema before feature freeze. |
| Target | Create `target_bad` for completed outcomes only. | `Fully Paid` is good; charge-off/default/late statuses are bad; unresolved loans are censored. | Confirm status mapping with project/domain reviewer. |
| Censored loans | Exclude `Current`, `In Grace Period`, and `Issued` from binary default modeling. | These loans do not yet have final repayment outcomes. | Consider survival or censoring-aware modeling only if unresolved loans must be included. |
| Cleaning | Preserve raw fields and add cleaned helper fields. | Maintains traceability from LendingClub source values to analysis-ready values. | Keep source-to-derived-field mapping in the modeling pipeline. |
| Missingness | Treat structural missingness separately from random missingness. | Joint-applicant fields and delinquency-history fields are missing for business reasons, not necessarily data errors. | Add missingness indicators where appropriate and validate by `application_type`. |
| Suspicious values | Flag suspicious values but do not automatically delete them. | Extreme values may be real, miscoded, or parsing artifacts. | Review `dti`, high income, high revolving utilization, and high credit-limit outliers before training. |
| Leakage | Exclude post-origination servicing, payment, recovery, hardship, settlement, last-FICO, and last-credit-pull fields. | These fields are unavailable at application time and would inflate model performance. | Enforce exclusions in a feature registry or pipeline config. |
| Ambiguous fields | Require review before using funding, text, geography, policy, listing, and disbursement fields. | These can introduce timing ambiguity, compliance risk, proxy risk, or platform-policy leakage. | Decide keep/drop/transform after business, timestamp, and fair-lending review. |
| Validation | Use chronological issue-date splits. | LendingClub borrower mix, credit policy, pricing, and macro conditions change over time. | Backtest vintage stability and compare with random split only as a diagnostic. |
| Starter features | Use 38 safe starter features for first-pass modeling. | They are mostly application-time borrower, loan, and credit-bureau variables. | Review proxy-risk and high-missingness features before production framing. |

## Target Definition Decisions

`target_bad` is defined only where the loan has an observed completed or delinquent outcome.

| Loan status | Target treatment | Sample rows | Sample % |
| --- | --- | ---: | ---: |
| `Fully Paid` | Good terminal | 148,646 | 59.46 |
| `Charged Off` | Bad terminal or delinquent | 38,709 | 15.48 |
| `Late (31-120 days)` | Bad terminal or delinquent | 1,752 | 0.70 |
| `Late (16-30 days)` | Bad terminal or delinquent | 343 | 0.14 |
| `Default` | Bad terminal or delinquent | 2 | 0.00 |
| `Current` | Censored unresolved | 59,851 | 23.94 |
| `In Grace Period` | Censored unresolved | 697 | 0.28 |

Completed modeling sample:

| Metric | Value |
| --- | ---: |
| Completed rows | 189,452 |
| Good rows | 148,646 |
| Bad rows | 40,806 |
| Observed bad rate | 21.54% |
| Completed share of original 250,000-row EDA sample | 75.78% |
| Completed share of post-`Current` modeling EDA frame | 99.63% |

Decision: unresolved statuses are excluded from the binary PD/default target because their final repayment outcomes are unknown at the EDA snapshot date. `Late (16-30 days)` is treated as a bad/delinquent target event in the current notebook, but it is also flagged as an early-delinquency status that should be confirmed with the domain reviewer before final model training.

## Full-Dataset Loan-Status Imbalance

The full accepted file has 2,260,701 rows. `loan_status` is almost fully populated: 33 rows are missing, or 0.0015% of the full dataset.

| Loan status | Loan-status definition | Target treatment | Repayment-risk interpretation | Full rows | Full % |
| --- | --- | --- | --- | ---: | ---: |
| `Fully Paid` | The borrower completed repayment and the loan reached a successful terminal outcome. | Good terminal | Low observed repayment risk | 1,076,751 | 47.6291 |
| `Current` | The loan is still active/open and has not reached a final repayment or failure outcome. | Censored unresolved | Unresolved, not risk-ranked for binary PD | 878,317 | 38.8515 |
| `Charged Off` | The lender has written off the loan as a loss after serious nonpayment. | Bad terminal or delinquent | High observed repayment risk | 268,559 | 11.8795 |
| `Late (31-120 days)` | The borrower is materially delinquent, 31 to 120 days past due. | Bad terminal or delinquent | High observed repayment risk | 21,467 | 0.9496 |
| `In Grace Period` | The payment is recently overdue but still within the grace-period window. | Censored unresolved | Medium/current delinquency review, not final target | 8,436 | 0.3732 |
| `Late (16-30 days)` | The borrower is early delinquent, 16 to 30 days past due. | Bad terminal or delinquent | Early delinquency review; included as bad in current target mapping | 4,349 | 0.1924 |
| `Does not meet the credit policy. Status:Fully Paid` | The loan did not meet LendingClub credit policy, but the borrower ultimately fully repaid it. | Good terminal | Low observed repayment risk | 1,988 | 0.0879 |
| `Does not meet the credit policy. Status:Charged Off` | The loan did not meet LendingClub credit policy and was later charged off. | Bad terminal or delinquent | High observed repayment risk | 761 | 0.0337 |
| `Default` | The borrower defaulted under the loan's default definition. | Bad terminal or delinquent | High observed repayment risk | 40 | 0.0018 |
| Missing | No `loan_status` value is available. | Missing status | Missing status | 33 | 0.0015 |

Aggregated full-dataset view:

| Group | Rows | Full % |
| --- | ---: | ---: |
| Low observed repayment risk / good terminal | 1,078,739 | 47.7170 |
| Unresolved censored and not risk-ranked | 878,317 | 38.8515 |
| High observed repayment risk / bad terminal or delinquent | 290,827 | 12.8645 |
| Censored grace-period review | 8,436 | 0.3732 |
| Bad-target early delinquency review | 4,349 | 0.1924 |
| Missing status | 33 | 0.0015 |

Decision: the approximately 60% value in the sample target table is not nullness. It is the share of the 250,000-row working sample with `loan_status = Fully Paid`. For full-dataset imbalance reporting, use `accepted_loan_status_full_distribution.csv` and `accepted_loan_status_full_distribution_by_risk_bucket.csv`.

## Current-Loan Removal Decision

Rows with `loan_status = Current` are removed from the modeling-oriented EDA dataframe after the loan-status imbalance section. This does not modify the raw LendingClub source file.

| Scope | Filter rule | Rows before | Current rows removed | Rows after | % removed |
| --- | --- | ---: | ---: | ---: | ---: |
| Working sample | `loan_status != "Current"` | 250,000 | 59,851 | 190,149 | 23.9404 |
| Full dataset reference | `loan_status != "Current"` | 2,260,701 | 878,317 | 1,382,384 | 38.8515 |

Decision: `Current` loans are censored active loans. They should not be included as low-risk/good observations in binary default modeling. They are removed before downstream modeling-oriented EDA while terminal and delinquent outcomes remain available for target construction.

## Structural Data Quality Decisions

Section 5.1 adds explicit structural checks before target and missingness interpretation.

| Check Area | Current EDA Status | Decision |
| --- | --- | --- |
| Column names | Raw accepted-loan columns are inventoried and audited for leading/trailing spaces, uppercase letters, internal spaces, non-snake-case characters, and duplicates. | Do not globally rename accepted-loan columns unless the audit flags a real issue; LendingClub accepted columns are mostly already lowercase snake_case. |
| Text/categorical values | Selected categorical fields are audited for unique counts before/after stripping whitespace, values changed by strip, and case collisions after lowercasing. | Strip whitespace in `add_clean_features`; use explicit category maps only if mixed-case category collisions appear. |
| Data types | Percentages, term strings, employment length, dates, FICO ranges, and numeric credit fields are converted into analysis-ready helper fields. | Preserve raw fields and use helper fields such as `int_rate_clean`, `revol_util_clean`, `term_months`, `emp_length_years`, `issue_d_dt`, and `fico_mean`. |

## Irrelevant Data Decisions

Section 5.2 adds an audit for columns that are not useful as repayment-risk model features. This is separate from the leakage audit: irrelevant fields are identifiers, metadata, constant/unavailable fields, target columns, or raw fields superseded by cleaner helper features.

| Column Type | Examples | Decision |
| --- | --- | --- |
| Identifiers and lookup metadata | `id`, `member_id`, `url` | Drop from model features. Keep only for traceability outside the feature matrix if needed. |
| Target or outcome columns | `loan_status`, `target_bad`, `target_definition` | Drop from model features; use only to construct or audit the target. |
| Raw fields superseded by helper fields | `term`, `emp_length`, `int_rate`, `revol_util`, `issue_d`, `earliest_cr_line`, FICO range endpoints | Prefer helper fields such as `term_months`, `emp_length_years`, `int_rate_clean`, `revol_util_clean`, `credit_history_years`, and `fico_mean`. |
| Free-text fields | `desc`, `title`, `emp_title` | Exclude from first-pass modeling unless a separate NLP, privacy, and fair-lending review is completed. |
| Fully missing or constant fields | Example depends on the working dataframe audit | Drop unless a business reviewer provides a reason to keep. |

## Missing Data Label Decisions

Section 5.3 standardizes common text missing-value labels in text-like columns before duplicate removal, target construction, and downstream missingness analysis.

| Check | Decision |
| --- | --- |
| Text missing labels | Convert conservative tokens such as empty strings, `N/A`, `NA`, `NO DATA`, `NULL`, `nan`, `MISSING`, and `UNKNOWN` to `pd.NA`. |
| Whitespace | Strip leading/trailing whitespace before checking for missing-label tokens. |
| Legitimate business categories | Do not automatically convert `NONE`, because it can be a valid LendingClub category in fields such as home ownership. |
| Raw source file | Do not modify the raw LendingClub CSV. |

## Duplicate Row Decisions

Section 5.4 removes exact duplicate rows from the in-memory accepted-loan working dataframe before target construction and downstream EDA.

| Check | Decision |
| --- | --- |
| Exact duplicate records | Count with `accepted.duplicated().sum()` and remove with `accepted.drop_duplicates().reset_index(drop=True)`. |
| Raw source file | Do not modify the raw LendingClub CSV. |
| Partial duplicates | Do not automatically collapse records that share an identifier but differ in other fields; those require separate data-lineage review. |

## Imputation Decisions

Section 5.5 prepares an imputation plan but does not apply imputation inside the EDA dataframe. Actual imputation should happen later in the modeling-prep notebook after final feature selection and train/validation/test splitting.

| Data Type | Recommendation | Reason |
| --- | --- | --- |
| Numeric fields | Use median imputation by default; consider mean only after distribution review. | Credit-risk variables such as income, balances, DTI, and utilization are often skewed and outlier-prone. |
| Categorical fields | Use mode imputation by default. | The most frequent category is simple and auditable for first-pass modeling. |
| Informative missingness fields | Add missingness indicators before imputation. | Missingness may carry weak repayment-risk signal for fields such as `emp_length_years` and selected bureau variables. |
| Structural missingness fields | Handle conditionally by business logic or exclude. | Joint-applicant fields such as `annual_inc_joint` and `dti_joint` are expected to be missing for individual applications. |
| EDA dataframe | Do not overwrite missing values during EDA. | Imputation would hide missingness patterns and make EDA conclusions harder to audit. |

## High-Missingness Feature Decisions

Section 5.6 defines the decision framework for high-missingness columns. High missingness is not an automatic drop rule. The first question is whether the column actually helps predict repayment after leakage, identifier, target, and compliance exclusions.

| Step | Decision | Reason |
| --- | --- | --- |
| Feature-importance save test | In modeling prep, run a quick leakage-clean tree-based model and inspect validation feature importance. Drop high-missingness columns with zero or negligible importance. | Avoid spending modeling complexity on columns that do not help predict `target_bad`. |
| Categorical pivot | For important high-missingness columns, consider converting missing values to `Not_Reported` and binning observed values as `Low`, `Medium`, `High`. | Treats missingness as the dominant signal instead of hiding it with median/mean imputation. |
| Strict indicator method | For important high-missingness columns, create a missingness flag and fill original gaps with a distinct sentinel such as `-999` or `Unknown`. | Separates “was missing” from the observed magnitude while preserving model visibility into missingness. |
| Domain-knowledge feature | Convert known business missingness patterns into explicit features where defensible. | Examples include joint-applicant fields missing for individual applications or delinquency-recency fields missing because no event is recorded. |
| EDA dataframe | Do not drop or impute high-missingness columns inside EDA. | EDA should expose the issue; modeling prep should apply the approved transformation after validation. |

## Cleaning and Derived-Feature Decisions

| Derived field | Source field(s) | Decision |
| --- | --- | --- |
| `int_rate_clean` | `int_rate` | Parse percentage values to numeric rates. |
| `revol_util_clean` | `revol_util` | Parse percentage values to numeric utilization. |
| `term_months` | `term` | Extract numeric term length from strings such as `36 months`. |
| `emp_length_years` | `emp_length` | Convert `< 1 year` to 0, `10+ years` to 10, and parse numeric years. |
| `fico_mean` | `fico_range_low`, `fico_range_high` | Average the FICO range endpoints for EDA. |
| `issue_d_dt` | `issue_d` | Parse LendingClub month-year date. |
| `earliest_cr_line_dt` | `earliest_cr_line` | Parse LendingClub month-year date. |
| `credit_history_years` | `issue_d_dt`, `earliest_cr_line_dt` | Compute credit history age at issue date. |
| `issue_year`, `issue_quarter`, `issue_month` | `issue_d_dt` | Create time fields for temporal and vintage analysis. |

Decision: cleaning is additive. Raw columns remain available for audit, and derived fields are used for numeric EDA, plots, and first-pass feature screening.

## Missingness Decisions

Important missingness patterns from the working sample:

These percentages are from the post-`Current`-removal modeling EDA frame with 190,149 rows, matching `accepted_working_sample_missingness.csv`.

| Column / pattern | Missing % | Decision |
| --- | ---: | --- |
| `member_id` | 100.00 | Exclude as an unavailable identifier. The LC data dictionary defines it as a unique borrower-member identifier, not a borrower-risk attribute. In this public accepted-loan file it is fully unpopulated. |
| Joint-applicant fields such as `verification_status_joint`, `annual_inc_joint`, `dti_joint` | About 98.85 | Treat as structurally missing unless `application_type` is joint. |
| `mths_since_last_record` | 81.80 | Interpret as absence/recency of public record rather than simple random missingness. |
| Recent installment and revolving delinquency fields | About 74.28 to 83.16 | Review availability by vintage and credit-file condition before use. |
| `mths_since_last_delinq` | 48.44 | Use with careful missingness encoding if retained. |
| `target_bad` | 0.37 | Missing because `In Grace Period` remains censored after `Current` removal. In the original 250,000-row sample, `target_bad` was missing for `Current` plus `In Grace Period` loans. |
| `emp_length_years` | 6.15 | Keep as candidate feature with missingness handling. |
| `revol_util_clean` | 0.05 | Keep as candidate feature; handle small missing share. |
| `dti` | 0.02 | Keep as candidate feature; review suspicious values separately. |

Decision: do not blanket-drop every high-missingness field during EDA. Missingness should be interpreted by field meaning, application type, and credit-file structure.

Loan-status slice insight: missingness is somewhat higher for `Charged Off` than `Fully Paid` on selected fields, especially `emp_length_years`:

| Feature | Charged Off missing | Fully Paid missing | Interpretation |
| --- | ---: | ---: | --- |
| `emp_length_years` | 3,004 / 38,709 (7.76%) | 8,503 / 148,646 (5.72%) | Potential weak risk signal; test a missingness indicator. |
| `dti` | 12 / 38,709 (0.03%) | 30 / 148,646 (0.02%) | Difference is tiny; treat as data-quality monitoring. |
| `revol_util_clean` | 23 / 38,709 (0.06%) | 74 / 148,646 (0.05%) | Difference is tiny; treat as data-quality monitoring. |
| `bc_util` | 426 / 38,709 (1.10%) | 1,546 / 148,646 (1.04%) | Difference is small; consider missingness indicator only if validated. |

Modeling suggestion: preserve missingness indicators for selected variables where missingness may carry information, especially `emp_length_years`, `bc_util`, and joint-application fields if joint applications are modeled. Do not blindly impute all missing values without retaining whether the value was originally missing.

Automated mechanism screening decision: Section 7.1 screens missingness as `consistent_with_MCAR_screen`, `potential_MAR_observed_pattern`, `structural_missingness`, or `no_missingness_observed`. This is a diagnostic screen, not proof. MCAR and MAR can be supported or challenged using observed patterns, but MNAR cannot be confirmed from the observed dataset alone because it depends on unobserved values or the data-collection process. Missingness indicators are not applied in Section 7.1; they are deferred to later modeling preparation.

| Mechanism label | What the EDA can automate | Modeling decision |
| --- | --- | --- |
| `consistent_with_MCAR_screen` | Missingness is low and does not vary materially across observed slices. | Simple imputation may be acceptable; validate during modeling. |
| `potential_MAR_observed_pattern` | Missingness varies by observed fields such as `application_type`, `loan_status`, `issue_year`, `grade`, or target outcome. | Preserve missingness indicator later and validate imputation/model impact. |
| `structural_missingness` | Missingness follows business structure, such as joint-applicant fields missing for individual applications. | Handle conditionally by business rule or exclude from first-pass model. |
| MNAR review needed | Cannot be proven automatically from observed data. | Use domain review, collection-process review, and sensitivity analysis. |

## Suspicious-Value Decisions

The EDA flags suspicious values but does not remove them automatically.

| Column | Finding | Decision |
| --- | --- | --- |
| `dti` | 152 suspicious values; max 999.0; 99th percentile 39.35. | Review and cap/filter only after defining a modeling rule. |
| `annual_inc` | Max 9,573,072 while 99th percentile is 260,000. | Review high-income outliers; avoid arbitrary deletion in EDA. |
| `revol_util_clean` | Max 193.0 while 99th percentile is 98.7. | Review utilization above 100%; may reflect reporting or credit-limit timing. |
| `open_acc` | Max 81. | Review but not automatically invalid. |
| `pub_rec` | Max 28. | Review but not automatically invalid. |

Decision: use documented winsorization, capping, or robust preprocessing only in the modeling pipeline, not silently in EDA.

## Leakage and Exclusion Decisions

Clear exclusions:

| Category | Examples | Decision |
| --- | --- | --- |
| Identifiers / target fields | `id`, `member_id`, `loan_status`, `target_bad`, `target_definition` | Exclude from model features. |
| Payment and principal fields | `out_prncp`, `total_pymnt`, `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee` | Exclude as post-origination leakage. |
| Recovery and collections fields | `recoveries`, `collection_recovery_fee` | Exclude as post-outcome leakage. |
| Payment timing fields | `last_pymnt_d`, `last_pymnt_amnt`, `next_pymnt_d` | Exclude as post-origination leakage. |
| Credit update fields | `last_credit_pull_d`, `last_fico_range_high`, `last_fico_range_low` | Exclude because they are updated after application/origination. |
| Hardship and settlement fields | `hardship_*`, `debt_settlement_*`, `settlement_*` | Exclude as post-origination servicing/outcome fields. |

Review-required fields:

| Field(s) | Reason for review |
| --- | --- |
| `funded_amnt`, `funded_amnt_inv` | May reflect funding process rather than requested application information. |
| `url` | Identifier/platform metadata; not a borrower risk attribute. |
| `desc`, `title`, `emp_title` | Text fields can contain privacy, proxy, stability, and explainability risk. |
| `zip_code`, `addr_state` | Geography can introduce fair-lending proxy risk. |
| `initial_list_status` | May reflect platform/listing mechanics. |
| `policy_code` | May encode policy decisions or platform rules. |
| `disbursement_method` | Requires timing and business-process review. |

Decision: the first modeling dataset should use only application-time features and exclude clear leakage. Ambiguous fields require explicit approval before use.

## Starter Feature Decision

The accepted EDA exports 38 starter features in `accepted_safe_starter_features.csv`:

`loan_amnt`, `term_months`, `int_rate_clean`, `installment`, `grade`, `sub_grade`, `emp_length_years`, `home_ownership`, `annual_inc`, `verification_status`, `purpose`, `dti`, `delinq_2yrs`, `fico_mean`, `inq_last_6mths`, `mths_since_last_delinq`, `mths_since_last_record`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util_clean`, `total_acc`, `credit_history_years`, `collections_12_mths_ex_med`, `acc_now_delinq`, `tot_coll_amt`, `tot_cur_bal`, `acc_open_past_24mths`, `avg_cur_bal`, `bc_open_to_buy`, `bc_util`, `mort_acc`, `pub_rec_bankruptcies`, `tax_liens`, `total_bal_ex_mort`, `total_bc_limit`, `total_il_high_credit_limit`, `application_type`.

Decision: use this list for first-pass leakage-clean modeling only. It is not a production-approved feature registry.

## Time-Split Decision

The notebook recommends chronological splits based on the completed sample:

| Split | Rule |
| --- | --- |
| Train | `issue_d <= 2015-11-01` |
| Validation | `2015-11-01 < issue_d <= 2015-12-01` |
| Test | `issue_d > 2015-12-01` |

Decision: use issue-date chronological validation because credit-risk performance must generalize across vintages and changing portfolio conditions.

## Temporal EDA Notes

The post-`Current` working sample is concentrated in later issue years, matching `accepted_yearly_portfolio_summary.csv`:

| Issue year | Sample loans | Avg loan amount | Avg interest rate | Avg FICO | Avg DTI |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2015 | 163,399 | 14,671.49 | 12.35 | 695.74 | 18.94 |
| 2016 | 11,844 | 14,462.57 | 12.79 | 697.22 | 19.01 |
| 2017 | 8,709 | 14,243.70 | 14.42 | 699.79 | 19.29 |
| 2018 | 6,197 | 15,379.06 | 13.46 | 709.05 | 18.83 |

Decision: report vintage stability and avoid relying only on pooled average performance.

## Evidence Artifacts

Key tables created by the notebook:

| Artifact | Purpose |
| --- | --- |
| `accepted_file_overview.csv` | Raw file size, row count, and column count. |
| `accepted_schema_sample.csv` | Initial schema, type, missingness, uniqueness, and examples. |
| `accepted_working_sample_missingness.csv` | Missingness profile of the EDA working sample. |
| `accepted_loan_status_target_mapping.csv` | Mapping from `loan_status` to target treatment. |
| `accepted_loan_status_full_distribution.csv` | Full 2.26M-row `loan_status` distribution with target and repayment-risk interpretation. |
| `accepted_loan_status_full_distribution_by_risk_bucket.csv` | Aggregated full-dataset imbalance by repayment-risk bucket. |
| `accepted_current_status_removal_summary.csv` | Row counts before and after removing `loan_status = Current`. |
| `accepted_completed_target_summary.csv` | Completed-loan good/bad counts and observed bad rate. |
| `accepted_suspicious_value_report.csv` | Outlier and suspicious-value checks. |
| `accepted_leakage_audit_all_raw_columns.csv` | Raw-column leakage and review recommendations. |
| `accepted_safe_starter_features.csv` | First-pass safe starter feature list. |
| `accepted_modeling_readiness_recommendations.csv` | High-level modeling-readiness recommendations. |
| `accepted_recommended_time_split_plan.csv` | Recommended chronological split rules. |

Decision: these artifacts should be regenerated whenever the accepted EDA notebook, source data, or feature-screening rules change.

## Open Decisions

| Open item | Needed decision |
| --- | --- |
| Full-population modeling | Decide whether final training uses all eligible completed accepted loans or a sampled training frame. |
| Outlier treatment | Define capping, filtering, or robust transformation rules for `dti`, `annual_inc`, `revol_util_clean`, and credit-line extremes. |
| High-missingness credit fields | Decide which delinquency/public-record recency fields to retain and how to encode missingness. |
| Geography | Decide whether to exclude, coarsen, or include geography after fair-lending review. |
| Text fields | Decide whether `desc`, `title`, and `emp_title` are excluded entirely or handled in a separate NLP/privacy-reviewed workflow. |
| Grade/sub-grade and interest rate | Decide whether to include LendingClub-assigned pricing/risk grades depending on the model's intended decision timing. |
| Reason codes | Map final model explanations to approved adverse-action style reason codes if the project presents denial or risk reasons. |
