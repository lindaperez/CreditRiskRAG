# Feature Cleaning And Modeling Decisions

This file defines which LendingClub accepted-loan columns are key for the credit-risk project, which columns must be removed, and which columns require treatment or review before modeling.

The cleaning scope is the accepted-loan workflow in `EDA/Accepted_Loan_EDA.ipynb`. The project target is default risk on accepted/originated LendingClub loans.

Rejected applications are not part of this notebook's cleaning feature set. They do not have repayment outcomes, so they are useful for separate population comparison and demo context, but not for supervised default labels.

## Source Notebook Scope

`EDA/Accepted_Loan_EDA.ipynb` builds the working accepted-loan sample from these notebook-defined column groups:

| Notebook Group | Columns | Cleaning Decision |
| --- | --- | --- |
| `IDENTIFIER_COLS` | `id`, `member_id`, `url` | Keep only for audit or traceability outside the model matrix. Do not use as predictors. |
| `TARGET_COLS` | `loan_status` | Use only to create `target_bad`; do not use as a predictor. |
| `APPLICATION_TIME_CANDIDATES` | Application, credit-bureau, pricing, policy, geography, and joint-applicant candidates listed in the notebook | Review for application-time availability, missingness, business meaning, and fair-lending risk before final modeling. |
| `LEAKAGE_DEMO_COLS` | `out_prncp`, `total_pymnt`, `total_rec_prncp`, `total_rec_int`, `recoveries`, `collection_recovery_fee`, `last_pymnt_d`, `last_pymnt_amnt`, `last_credit_pull_d`, `last_fico_range_high`, `last_fico_range_low`, `debt_settlement_flag`, `hardship_flag` | Included in the notebook to demonstrate leakage risk; remove from modeling. |

The notebook then creates cleaned helper fields such as `term_months`, `int_rate_clean`, `emp_length_years`, `fico_mean`, `credit_history_years`, and `revol_util_clean`. These derived fields are preferred over raw string/date/range fields where noted below.

## Accepted Notebook Candidate Inventory

These are the accepted-loan column families analyzed by `Accepted_Loan_EDA.ipynb` before final feature selection.

| Decision | Accepted-Loan Columns From Notebook | Reason |
| --- | --- | --- |
| Key or strong candidate after cleaning | `loan_amnt`, `term`, `int_rate`, `installment`, `grade`, `sub_grade`, `emp_length`, `home_ownership`, `annual_inc`, `verification_status`, `purpose`, `dti`, `delinq_2yrs`, `earliest_cr_line`, `fico_range_low`, `fico_range_high`, `inq_last_6mths`, `mths_since_last_delinq`, `mths_since_last_record`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util`, `total_acc`, `collections_12_mths_ex_med`, `application_type`, `acc_now_delinq`, `tot_coll_amt`, `tot_cur_bal`, `acc_open_past_24mths`, `avg_cur_bal`, `bc_open_to_buy`, `bc_util`, `mort_acc`, `pub_rec_bankruptcies`, `tax_liens`, `total_bal_ex_mort`, `total_bc_limit`, `total_il_high_credit_limit` | These are borrower, loan-request, credit-history, or credit-bureau style fields that can plausibly exist at underwriting time. |
| Broader candidate after vintage/missingness review | `mths_since_last_major_derog`, `annual_inc_joint`, `dti_joint`, `verification_status_joint`, `open_acc_6m`, `open_act_il`, `open_il_12m`, `open_il_24m`, `mths_since_rcnt_il`, `total_bal_il`, `il_util`, `open_rv_12m`, `open_rv_24m`, `max_bal_bc`, `all_util`, `total_rev_hi_lim`, `inq_fi`, `total_cu_tl`, `inq_last_12m`, `chargeoff_within_12_mths`, `delinq_amnt`, `mo_sin_old_il_acct`, `mo_sin_old_rev_tl_op`, `mo_sin_rcnt_rev_tl_op`, `mo_sin_rcnt_tl`, `mths_since_recent_bc`, `mths_since_recent_bc_dlq`, `mths_since_recent_inq`, `mths_since_recent_revol_delinq`, `num_accts_ever_120_pd`, `num_actv_bc_tl`, `num_actv_rev_tl`, `num_bc_sats`, `num_bc_tl`, `num_il_tl`, `num_op_rev_tl`, `num_rev_accts`, `num_rev_tl_bal_gt_0`, `num_sats`, `num_tl_120dpd_2m`, `num_tl_30dpd`, `num_tl_90g_dpd_24m`, `num_tl_op_past_12m`, `pct_tl_nvr_dlq`, `percent_bc_gt_75`, `tot_hi_cred_lim` | These may be useful, but several have high missingness or vintage availability problems. Add after the 38-feature baseline, not before. |
| Review before use | `funded_amnt`, `funded_amnt_inv`, `issue_d`, `zip_code`, `addr_state`, `initial_list_status`, `policy_code`, `disbursement_method`, `emp_title`, `desc`, `title` | These have timing, proxy-risk, text/privacy, or business-policy concerns. |
| Remove from model | `id`, `member_id`, `url`, `loan_status`, `out_prncp`, `total_pymnt`, `total_rec_prncp`, `total_rec_int`, `recoveries`, `collection_recovery_fee`, `last_pymnt_d`, `last_pymnt_amnt`, `last_credit_pull_d`, `last_fico_range_high`, `last_fico_range_low`, `debt_settlement_flag`, `hardship_flag` | These are identifiers, target/outcome fields, or clear post-origination leakage fields in the notebook's working sample. |

## Executive Decision

| Decision Group | Columns | Decision | Reason |
| --- | --- | --- | --- |
| Modeling population | Accepted loans only | Use for supervised default-risk model | Repayment outcome is only observed after a loan is originated. |
| Rejected applications | Not in `Accepted_Loan_EDA.ipynb` cleaning scope | Use only in a separate rejected-applicant EDA/comparison workflow; do not label as default/non-default | A rejected applicant never receives a LendingClub loan in this file, so there is no observed repayment outcome. |
| Target | `loan_status` transformed to `target_bad` | Use only to create the label; never use as a feature | `loan_status` is the outcome we are trying to predict. |
| Terminal training outcomes | `Fully Paid`, `Charged Off` | Keep for binary modeling | These statuses provide the cleanest and most stable good/bad terminal repayment information for the project baseline. |
| Excluded statuses | `Current`, `In Grace Period`, `Late (31-120 days)`, `Late (16-30 days)`, `Default`, `Issued`, policy-exception statuses, and missing statuses | Exclude from binary target training | Late/current/grace statuses are snapshot states rather than final repayment outcomes. `Default` is excluded because it has only 40 full-file rows, which is too small for stable baseline training. Policy-exception statuses are excluded from the strict two-status baseline to keep the training definition aligned with `Fully Paid` and `Charged Off` only. |

## Key Starter Features

These are the approved starter features from `EDA/accepted_eda_outputs/tables/accepted_safe_starter_features.csv`. They should be the first model feature set because they are mostly application-time or credit-bureau style variables and avoid obvious post-origination leakage.

| Feature | Decision | Treatment / Condition |
| --- | --- | --- |
| `loan_amnt` | Key | Requested/originated amount. Check positive values; cap or winsorize only after p01/p99 review. |
| `term_months` | Key derived feature | Parse from `term`; valid values should be 36 or 60 months. Invalid parses become missing and must be investigated. |
| `int_rate_clean` | Key, but policy-sensitive | Parse from `int_rate`. Use only if the project treats LendingClub pricing as known at decision time; otherwise run a challenger model without it because pricing may partly encode prior underwriting. |
| `installment` | Key, but correlated | Keep for baseline; monitor correlation with `loan_amnt`, `term_months`, and `int_rate_clean`. If correlation causes instability, keep the more interpretable set. |
| `grade`, `sub_grade` | Key, but policy-sensitive | Strong risk rankers. Use for baseline and compare against a model without them because they may embed LendingClub's prior credit decisioning. |
| `emp_length_years` | Key treated feature | Parse `< 1 year` as 0 and `10+ years` as 10. Missing rate is about 6.25%; impute and add a missingness flag. |
| `home_ownership` | Key categorical | Trim text; group rare/ambiguous categories if frequency is below 1% or unstable across time. |
| `annual_inc` | Key | Check non-negative values and extreme upper tail. Apply log transform candidate: `log1p(annual_inc)`. |
| `verification_status` | Key categorical | Keep as categorical; monitor for policy changes across origination year. |
| `purpose` | Key categorical | Keep for model and reason-code mapping; group rare purposes if count is below 500 or share is below 1%. |
| `dti` | Key | Must be numeric and non-negative. Current missingness is about 0.04%; impute only if needed. Investigate values above business-plausible caps before clipping. |
| `delinq_2yrs` | Key | Credit-bureau delinquency signal. Check non-negative integer-like values. |
| `fico_mean` | Key derived feature | Average `fico_range_low` and `fico_range_high`. Keep the derived feature; do not need both raw bounds in the baseline model. |
| `inq_last_6mths` | Key | Credit inquiry signal. Check non-negative integer-like values. |
| `mths_since_last_delinq` | Key treated feature | Missingness is structural: missing usually means no recent delinquency record. Add missingness flag and impute with a sentinel or model-native missing handling. |
| `mths_since_last_record` | Treat carefully | Missingness is about 82.86%; keep only with missingness flag/model-native missing handling, or drop in a simple baseline. |
| `open_acc` | Key | Check non-negative integer-like values. |
| `pub_rec` | Key | Check non-negative integer-like values; heavy zero inflation expected. |
| `revol_bal` | Key | Check non-negative values; review p99 outliers. |
| `revol_util_clean` | Key treated feature | Parse from `revol_util`; current missingness is about 0.06%. Values above 100 can occur but require review. |
| `total_acc` | Key | Check non-negative integer-like values. |
| `credit_history_years` | Key derived feature | Derived from `issue_d` and `earliest_cr_line`; must be non-negative. |
| `collections_12_mths_ex_med` | Key | Check non-negative integer-like values. |
| `acc_now_delinq` | Key | Check non-negative integer-like values. |
| `tot_coll_amt` | Key | Check non-negative values; many zeros expected. |
| `tot_cur_bal` | Key | Check non-negative values; review upper-tail outliers. |
| `acc_open_past_24mths` | Key | Check non-negative integer-like values. |
| `avg_cur_bal` | Key | Check non-negative values; review upper-tail outliers. |
| `bc_open_to_buy` | Key treated feature | Missingness is about 1.00%; impute and add missing flag only if model does not handle missing natively. |
| `bc_util` | Key treated feature | Missingness is about 1.06%; parse numeric and review values above 100. |
| `mort_acc` | Key | Mortgage account count; check non-negative integer-like values. |
| `pub_rec_bankruptcies` | Key | Public-record bankruptcy signal; check non-negative integer-like values. |
| `tax_liens` | Key | Public-record tax lien signal; check non-negative integer-like values. |
| `total_bal_ex_mort` | Key | Check non-negative values. |
| `total_bc_limit` | Key | Check non-negative values. |
| `total_il_high_credit_limit` | Key | Check non-negative values. |
| `application_type` | Key categorical | Use to distinguish individual vs joint applications. Joint-only fields must be interpreted conditionally. |

## Remove Before Modeling

These columns should not enter the model matrix.

| Column Group | Columns | Removal Rule | Reason |
| --- | --- | --- | --- |
| Identifiers | `id`, `member_id`, `url` | Exclude from model; keep only for traceability outside the feature matrix | Identifiers do not represent borrower risk and can cause memorization or lookup leakage. `member_id` is 100% missing in the working sample. |
| Target / outcome | `loan_status`, `target_bad` | Use only for label construction/evaluation | Direct outcome leakage if used as a predictor. |
| Payment and balance after origination | `out_prncp`, `out_prncp_inv`, `total_pymnt`, `total_pymnt_inv`, `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee`, `last_pymnt_d`, `last_pymnt_amnt`, `next_pymnt_d` | Always remove | These are known only after loan issuance. |
| Recovery / collections after default | `recoveries`, `collection_recovery_fee` | Always remove | They reveal default and recovery behavior after the event. |
| Credit updates after origination | `last_credit_pull_d`, `last_fico_range_high`, `last_fico_range_low` | Always remove | These are updated after the lending decision. |
| Hardship program fields | `hardship_flag`, `hardship_type`, `hardship_reason`, `hardship_status`, `deferral_term`, `hardship_amount`, `hardship_start_date`, `hardship_end_date`, `payment_plan_start_date`, `hardship_length`, `hardship_dpd`, `hardship_loan_status`, `orig_projected_additional_accrued_interest`, `hardship_payoff_balance_amount`, `hardship_last_payment_amount` | Always remove | Hardship status is post-origination servicing information. |
| Settlement fields | `debt_settlement_flag`, `debt_settlement_flag_date`, `settlement_status`, `settlement_date`, `settlement_amount`, `settlement_percentage`, `settlement_term` | Always remove | Settlement is post-default/post-origination information. |
| Unavailable or structurally absent fields | Any column with 95%+ missingness unless it has a documented structural meaning and passes review | Remove from baseline | Too sparse for stable baseline modeling; high risk of unstable splits and misleading imputations. |

## Treat Or Review Before Use

These columns are not automatically wrong, but they need explicit business, timing, or compliance review.

| Column | Decision | Condition For Use |
| --- | --- | --- |
| `funded_amnt`, `funded_amnt_inv` | Review before use | Use only if confirmed available at decision time. Prefer `loan_amnt` for requested credit amount because funding amount may reflect marketplace/investor behavior after approval. |
| `issue_d` | Use for time split, not as a direct feature | Use to derive train/validation/test chronology and vintage reporting. Avoid using raw issue date as a predictive feature unless modeling temporal drift intentionally. |
| `earliest_cr_line` | Use only as derived `credit_history_years` | Raw date is less interpretable and can leak calendar effects; derived tenure is safer. |
| `emp_title` | High-risk text/category | Do not use in baseline. If used later, normalize, screen for PII/proxy risk, cap cardinality, and run fairness tests. |
| `desc`, `title` | High-risk text | Do not use in baseline. Borrower text can contain sensitive or unstable language and creates explainability risk. |
| `zip_code` | Proxy-risk geography | Do not use in baseline adverse-action model. If used for research, test with/without it and run disparate-impact review. |
| `addr_state` | Proxy-risk geography | Allowed for EDA; for modeling, use only after fairness review and with a challenger model excluding geography. |
| `initial_list_status` | Review timing/business meaning | Use only if confirmed as known at origination and not a marketplace artifact. |
| `policy_code` | Review business meaning | Use only if it does not encode historical LendingClub policy approval/denial logic that would make explanations circular. |
| `disbursement_method` | Review timing/business meaning | Use only if confirmed available before final decision and stable over time. |
| Joint-applicant fields | Conditional use | Use only when `application_type` indicates joint application. Add structural missingness indicators; do not globally impute as ordinary missing values. |

## Broader Candidate Features

The EDA leakage audit marks many credit-bureau style fields as candidate application-time features beyond the 38-feature starter set. These can be added after the first baseline if they improve validation stability and pass missingness checks.

Candidate families include:

| Family | Example Columns | Use Condition |
| --- | --- | --- |
| Installment/revolving account activity | `open_acc_6m`, `open_act_il`, `open_il_12m`, `open_il_24m`, `open_rv_12m`, `open_rv_24m`, `max_bal_bc`, `all_util`, `total_rev_hi_lim` | Add after checking missingness by issue year; many are unavailable in older vintages. |
| Additional inquiry fields | `inq_fi`, `inq_last_12m`, `mths_since_recent_inq` | Add if available consistently across train/test periods. |
| Account age and recency | `mo_sin_old_il_acct`, `mo_sin_old_rev_tl_op`, `mo_sin_rcnt_rev_tl_op`, `mo_sin_rcnt_tl`, `mths_since_recent_bc` | Add if non-negative and stable across vintages. |
| Delinquency detail | `mths_since_last_major_derog`, `mths_since_recent_bc_dlq`, `mths_since_recent_revol_delinq`, `num_accts_ever_120_pd`, `num_tl_90g_dpd_24m`, `num_tl_30dpd`, `num_tl_120dpd_2m`, `pct_tl_nvr_dlq` | Treat missingness as structural; add missing flags or use model-native missing handling. |
| Account counts | `num_actv_bc_tl`, `num_actv_rev_tl`, `num_bc_sats`, `num_bc_tl`, `num_il_tl`, `num_op_rev_tl`, `num_rev_accts`, `num_rev_tl_bal_gt_0`, `num_sats`, `num_tl_op_past_12m` | Check integer-like non-negative values and temporal availability. |
| Balance and limit fields | `tot_hi_cred_lim`, `total_bal_il`, `total_cu_tl`, `percent_bc_gt_75` | Add only after outlier review and stability check. |

## Cleaning Thresholds And Rules

These thresholds should be applied before final model training.

| Check | Threshold / Rule | Action |
| --- | --- | --- |
| Clear leakage | Any field created or updated after origination | Remove regardless of predictive power. High AUC from these fields is invalid. |
| Missingness: baseline | 0-20% missing | Keep with median/mode imputation or model-native missing handling. Add missing flag if missingness is informative. |
| Missingness: moderate | >20-60% missing | Use only if business meaning is strong. Add missingness flag and validate feature importance stability. |
| Missingness: high | >60-95% missing | Exclude from baseline unless missingness is structural and expected, such as no derogatory record. Require challenger comparison. |
| Missingness: extreme | >95% missing | Drop from baseline. Use only with documented structural reason and enough non-missing observations by time split. |
| Categorical cardinality | More than 50 levels or rare levels below 1% share | Group rare levels into `Other` or exclude. Text-like high-cardinality fields require separate NLP review. |
| Numeric outliers | Values outside p01-p99 or impossible business ranges | Do not delete automatically. Flag, inspect, and decide cap/winsorize/remove with documented rule. |
| Percent fields | Strings ending in `%` or percent-like values | Strip `%`, convert to numeric, and count conversion failures. |
| Date fields | Raw dates | Convert to datetime; derive age/tenure/year/month features. Do not use future dates. |
| Temporal validation | Any model evaluation | Use chronological train/validation/test split by `issue_d`; random split is diagnostic only. |
| Fair-lending proxy risk | Geography, employment text, free text, policy-code variables | Train with and without these fields; review disparate impact and reason-code suitability before adverse-action framing. |
| Reason-code suitability | Any feature used in SHAP/adverse-action explanation | Must map to a clear, applicant-understandable reason. If it cannot be explained fairly, exclude from explanation or model. |

## Target Mapping

| `loan_status` | Target Decision |
| --- | --- |
| `Fully Paid` | Good terminal outcome: `target_bad = 0` |
| `Charged Off` | Bad outcome: `target_bad = 1` |
| `Default` | Exclude from strict baseline because the full file has only 40 rows; revisit only as sensitivity analysis |
| `Late (31-120 days)` | Exclude from binary training because this is a delinquency snapshot, not a final repayment state |
| `Late (16-30 days)` | Exclude from binary training because this is an early-delinquency snapshot, not a final repayment state |
| `Current` | Exclude from binary training because outcome is unresolved |
| `In Grace Period` | Exclude from binary training because outcome is unresolved |
| `Does not meet the credit policy. Status:Fully Paid` | Exclude from strict three-status baseline; keep for sensitivity analysis only if policy-exception loans are explicitly in scope |
| `Does not meet the credit policy. Status:Charged Off` | Exclude from strict three-status baseline; keep for sensitivity analysis only if policy-exception loans are explicitly in scope |
| Missing status | Exclude because the repayment outcome cannot be mapped reliably |

## Final Feature Freeze Conditions

Before final modeling, every column must have one of these statuses:

| Status | Meaning |
| --- | --- |
| `approved_application_time_feature` | Available at or before underwriting, cleaned, and suitable for modeling. |
| `approved_derived_feature` | Created only from approved application-time fields. |
| `target_or_evaluation_only` | Used only for label construction, splitting, or evaluation. |
| `excluded_leakage` | Removed because it contains future/post-origination information. |
| `excluded_identifier` | Removed because it is an ID, URL, or traceability field. |
| `review_required` | Not used until timing, business meaning, missingness, and compliance concerns are resolved. |
| `eda_only` | Allowed for descriptive comparison but not for supervised default-risk training. |

## Recommended Baseline

Start with the 38 key starter features, keep only `Fully Paid` and `Charged Off` for the strict baseline target, exclude unresolved/rare statuses, and evaluate with a chronological split by `issue_d`. Then run challenger models:

1. Without LendingClub-assigned pricing/risk grades: remove `int_rate_clean`, `installment`, `grade`, and `sub_grade`.
2. Without geography/proxy-risk fields: remove `addr_state` and do not add `zip_code`.
3. Without sparse fields: remove features with more than 60% missingness, including `mths_since_last_record`, unless model-native missing handling shows stable lift.
4. Expanded credit-bureau model: add broader candidate credit-bureau fields only after vintage missingness and stability checks.

The production-style RAG/adverse-action demo should use only leakage-clean, application-time features that can be mapped to clear borrower-facing reason codes.
