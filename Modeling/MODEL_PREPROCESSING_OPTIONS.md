# Model And Preprocessing Options For Accepted Loan Default Prediction

## Objective

Predict `target_bad` for accepted LendingClub loans:

| Target value | Meaning |
| --- | --- |
| `0` | `Fully Paid` |
| `1` | `Charged Off` |

The cleaned starter frame has 38 leakage-screened features. Modeling should use chronological train/validation/test splits by `issue_d_dt`, then fit preprocessing on train only.

## Required Preprocessing Before Any Model

| Step | Requirement | Reason |
| --- | --- | --- |
| Chronological split | Split by `issue_d_dt` before fitting imputers or encoders. | Prevents future-period leakage. |
| Baseline feature set | Exclude `mths_since_last_record` from baseline. | About 83% missing; use challenger only. |
| Missingness indicators | Add indicators for `mths_since_last_delinq` and optionally `emp_length_years`. | Missingness can carry repayment-risk signal. |
| Numeric imputation | Fit median imputer on train only. | Robust to skewed credit-risk variables. |
| Categorical imputation | Fill missing categories as `Missing` on train/validation/test. | Keeps missingness explicit and auditable. |
| Categorical encoding | One-hot encode low-cardinality categorical features. | Required for linear models and sklearn tree models. |
| Scaling | Use for logistic regression, SVM, KNN, and neural nets; not required for tree models. | Distance/gradient models are scale-sensitive. |
| Evaluation | Report ROC-AUC, PR-AUC, recall at fixed precision, calibration, confusion matrix, and bad-rate by score band. | Class imbalance and risk ranking matter more than accuracy. |

## Recommended Baseline

| Model | Use | Preprocessing | Pros | Cons |
| --- | --- | --- | --- | --- |
| Logistic Regression | First benchmark and explainable baseline. | Median impute numeric, add missing flags, one-hot encode categoricals, scale numeric. | Interpretable, fast, good sanity check, easy coefficients. | Linear; weaker on nonlinear credit interactions; sensitive to correlated grade/pricing features. |
| HistGradientBoostingClassifier | Strong sklearn tree baseline. | Median impute or model-compatible missing handling, ordinal/one-hot categorical encoding. | Captures nonlinearities, no scaling needed, strong tabular performance. | Less transparent than logistic regression; categorical handling needs care. |
| Random Forest | Diagnostic benchmark. | Impute numeric, one-hot encode categoricals; no scaling. | Robust, easy feature importance, handles nonlinearities. | Can be large/slow; probability calibration often weak. |

## Strong Candidate Models

| Model | Best Role | Preprocessing | Pros | Cons |
| --- | --- | --- | --- | --- |
| LightGBM | Main production-style credit-risk model if available. | Can handle missing numeric values; categorical features can be encoded or passed as categorical. | Excellent tabular performance, fast, handles missingness well, SHAP-compatible. | Extra dependency; needs careful validation, calibration, and leakage review. |
| XGBoost | Strong challenger to LightGBM. | Handles missing numeric values; encode categoricals unless using categorical support. | Strong performance, mature tooling, SHAP-compatible. | Can be slower; categorical handling less straightforward. |
| CatBoost | Good categorical-heavy challenger. | Minimal categorical preprocessing; pass categorical feature indices. | Strong with categoricals, handles missingness, less one-hot explosion. | Extra dependency; may be less familiar; still needs calibration. |
| Explainable Boosting Machine | Interpretability-focused challenger. | Impute/encode according to package requirements. | More interpretable than black-box boosting; captures nonlinear shape functions. | Extra dependency; may underperform gradient boosting. |

## Models To Use Carefully

| Model | Preprocessing | Pros | Cons |
| --- | --- | --- | --- |
| Linear SVM | Impute, one-hot encode, scale. | Can be strong for high-dimensional one-hot data. | Poor probability estimates unless calibrated; less natural for credit-risk PD. |
| KNN | Impute, one-hot encode, scale. | Simple diagnostic. | Poor fit for large credit dataset; distance metrics weak with mixed feature types. |
| Naive Bayes | Encode categoricals and handle numeric assumptions. | Fast baseline. | Independence assumptions are unrealistic for credit features. |
| Neural Network / MLP | Impute, one-hot encode, scale, tune regularization. | Can model interactions. | Harder to explain, calibrate, and justify for this project. |

## Feature Set Plan

| Feature set | Contents | Purpose |
| --- | --- | --- |
| Baseline | 38 starter features minus `mths_since_last_record`. | Stable first model. |
| Missingness challenger | Baseline plus `mths_since_last_record` and missingness indicator. | Tests whether high-missingness public-record recency adds stable lift. |
| No grade/pricing challenger | Remove `grade`, `sub_grade`, `int_rate_clean`, `installment`. | Tests dependence on LendingClub prior underwriting/pricing. |
| No geography/proxy challenger | Exclude geography fields if added later. | Supports fair-lending/proxy-risk review. |

## Recommended First Modeling Sequence

1. Build chronological split from `accepted_traceability.parquet`.
2. Create baseline feature set excluding `mths_since_last_record`.
3. Fit logistic regression pipeline.
4. Fit gradient boosting pipeline.
5. Compare ROC-AUC, PR-AUC, calibration, and score-band bad rates.
6. Add challenger with `mths_since_last_record`.
7. Select model based on validation/test stability, not only highest AUC.

## Current Recommendation

Start with:

```text
Baseline 1: Logistic Regression
Baseline 2: HistGradientBoostingClassifier or LightGBM
```

Use LightGBM as the preferred final model if dependency and runtime are acceptable, because it fits tabular credit-risk data well and supports SHAP explanations for the later RAG/adverse-action workflow.
