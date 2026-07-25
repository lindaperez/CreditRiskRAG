# SHAP Decision And Results

## Decision

Use SHAP as the technical interpretation method for the preferred credit-risk model:

```text
xgb_neutral_09_without_grade_subgrade_with_int_rate
```

SHAP is appropriate because the final preferred model is an XGBoost tabular model and SHAP can explain both global model behavior and local accepted-loan risk drivers. SHAP should be used as an interpretation and reason-discovery layer, not as a standalone legal adverse-action explanation.

## Final Model Context

The preferred model is neutral XGBoost without `grade/sub_grade` and with `int_rate_clean`.

This feature policy was selected because it preserves nearly equivalent predictive performance while improving governance and parsimony. LightGBM remains the best technical F1 benchmark among the main model families, but the final preferred model is XGBoost because it is stronger for governance, calibration, SHAP interpretation, and business-policy explanation.

## SHAP Results

SHAP interpretation was run on a reproducible 5,000-row validation sample with seed 42, with test-period SHAP used for stability checking.

Top global risk drivers by mean absolute SHAP:

| Rank | Feature | Mean Absolute SHAP | Interpretation |
| ---: | --- | ---: | --- |
| 1 | `int_rate_clean` | 0.393 | Higher priced loan; accepted-loan pricing/risk signal. |
| 2 | `term_months` | 0.246 | Longer repayment term. |
| 3 | `acc_open_past_24mths` | 0.164 | More recently opened credit accounts. |
| 4 | `dti` | 0.116 | Higher debt burden relative to income. |
| 5 | `fico_mean` | 0.098 | Lower credit score range increases modeled risk. |
| 6 | `annual_inc` | 0.082 | Lower reported income can increase risk. |
| 7 | `loan_amnt` | 0.079 | Larger requested loan amount. |
| 8 | `avg_cur_bal` | 0.065 | Lower average balance across existing accounts. |

The top drivers are plausible credit-risk factors. With `grade` and `sub_grade` removed, `int_rate_clean` becomes the dominant pricing/risk signal. This matches the grade/subgrade and interest-rate ablation results.

## Stability Result

The validation and test SHAP rankings are highly stable:

```text
Spearman rank correlation: 0.9987
```

The top eight SHAP drivers have zero rank shift between validation and test. This supports the conclusion that the explanation pattern is not just a validation-period artifact.

## Reason-Code Decision

Use SHAP outputs to support draft reason-code families for the future RAG explanation layer.

Highest mapped reason-code families:

| Reason Family | Share Of Mapped SHAP Importance |
| --- | ---: |
| Higher-priced loan / interest rate signal | 29.17% |
| Longer loan repayment term | 18.25% |
| Multiple recently opened credit accounts | 12.20% |
| High debt burden relative to income | 8.61% |
| Lower credit score range | 7.25% |
| Lower reported income | 6.10% |
| Larger requested loan amount | 5.86% |
| Lower average balance across existing accounts | 4.86% |

These mappings are useful for technical explanation and downstream reason-code drafting. They still require compliance, fair-lending, and applicant-facing language review before any production use.

## Business Interpretation

SHAP confirms that the preferred model is ranking accepted loans using reasonable credit-risk signals: loan pricing, term, recent credit activity, debt burden, credit score, income, and loan size.

The results support using the model as a risk-ranking and policy-analysis tool for accepted/funded LendingClub loans. They do not support using the model as a production lending approval system without additional validation, monitoring, fairness testing, reason-code governance, and compliance approval.

## Limitations

- SHAP explains model behavior, not causal default drivers.
- SHAP outputs are not legal adverse-action explanations by themselves.
- The model estimates default risk conditional on historical LendingClub approval.
- Rejected applicants do not have observed repayment outcomes in this dataset, so SHAP results cannot represent rejected-applicant default risk.
- `int_rate_clean` is valid as an accepted-loan pricing/risk signal, but it should be clearly positioned as known at acceptance/origination and reviewed for circularity in any pre-approval use case.
- Geography, home ownership, and other possible proxy-risk features need fair-lending review before applicant-facing use.

## Output Artifacts

Primary SHAP files:

```text
Interpretation_SHAP/1_SHAP_Interpretation.ipynb
Interpretation_SHAP/shap_outputs/tables/shap_global_mean_abs_importance.csv
Interpretation_SHAP/shap_outputs/tables/shap_validation_vs_test_stability.csv
Interpretation_SHAP/shap_outputs/tables/shap_reason_code_family_importance.csv
Interpretation_SHAP/shap_outputs/tables/shap_local_examples.csv
Interpretation_SHAP/shap_outputs/plots/shap_mean_abs_bar.png
Interpretation_SHAP/shap_outputs/plots/shap_summary_beeswarm.png
Interpretation_SHAP/shap_outputs/plots/shap_waterfall_top_risk_example.png
Interpretation_SHAP/shap_outputs/plots/shap_waterfall_low_risk_example.png
Interpretation_SHAP/reason_codes/draft_reason_code_mapping.csv
```

