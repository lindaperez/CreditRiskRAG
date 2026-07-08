# Interpretation Plan

This folder is for model interpretation work after the final credit-risk model has been selected.

## Recommendation

Use SHAP as the next interpretation method, but use it carefully. SHAP is appropriate for this project because the selected model is an XGBoost tabular model and SHAP can explain both global feature importance and borrower-level risk drivers. SHAP should not be presented as a legal adverse-action explanation by itself. It is a technical bridge between the model and a future reason-code / RAG explanation layer.

## Current Model To Interpret

Preferred model:

```text
xgb_neutral_09_without_grade_subgrade_with_int_rate
```

Relevant artifacts:

```text
Modeling/modeling_outputs/xgboost_grade_int_rate_ablation/models/xgb_neutral_09_without_grade_subgrade_with_int_rate_selected_model.joblib
Modeling/Preprocessing/preprocessing_outputs/datasets/baseline_no_grade_subgrade_train_X.parquet
Modeling/Preprocessing/preprocessing_outputs/datasets/baseline_no_grade_subgrade_validation_X.parquet
Modeling/Preprocessing/preprocessing_outputs/datasets/baseline_no_grade_subgrade_test_X.parquet
Modeling/Preprocessing/preprocessing_outputs/tables/preprocessing_baseline_feature_manifest_no_grade_subgrade.csv
```

## Correct Use Of SHAP

| Use | Status | Notes |
| --- | --- | --- |
| Global feature importance | Recommended | Identify which features generally drive default-risk ranking. |
| Local borrower-level drivers | Recommended | Explain why a specific accepted loan was scored high risk. |
| Reason-code discovery | Recommended | Use repeated SHAP drivers to design human-readable reason categories. |
| Legal adverse-action explanation | Not enough by itself | Requires reason-code governance, compliance review, and regulatory support. |
| Rejected-applicant explanations | Out of scope | Rejected applications do not have observed repayment labels in this dataset. |

## Interpretation Workflow

1. Load the selected XGBoost model and the no-grade/subgrade feature matrices.
2. Run SHAP on a manageable validation/test sample.
3. Produce global outputs:
   - SHAP mean absolute importance table.
   - SHAP summary plot.
   - Top positive risk drivers.
4. Produce local outputs:
   - A few high-risk accepted-loan examples.
   - Top features increasing risk for each example.
   - Top features lowering risk for each example.
5. Map technical features to draft reason codes.
6. Flag features that should not be used directly in applicant-facing language.

## Draft Reason-Code Examples

| Technical Feature Family | Possible Human-Readable Reason |
| --- | --- |
| `dti` | High debt burden relative to income. |
| `revol_util_clean` | High revolving credit utilization. |
| `fico_mean` | Lower credit score range. |
| `credit_history_years` | Shorter credit history. |
| `mths_since_last_delinq` | Recent or missing delinquency history signal. |
| `annual_inc` | Lower reported income relative to requested credit. |
| `loan_amnt` / `installment` | Larger requested loan or payment obligation. |
| `int_rate_clean` | Higher priced loan, interpreted only as a pricing/risk signal for accepted loans. |

## Outputs To Add Here

```text
Interpretation/
├── README.md
├── shap_outputs/
│   ├── tables/
│   └── plots/
└── reason_codes/
    └── draft_reason_code_mapping.csv
```

## Important Boundary

The interpretation layer should support the final project story: the model helps rank accepted loans by repayment risk and identify the features associated with high-risk accepted-loan segments. It should not claim that the system is production-ready for lending decisions or legally compliant adverse-action notices.
