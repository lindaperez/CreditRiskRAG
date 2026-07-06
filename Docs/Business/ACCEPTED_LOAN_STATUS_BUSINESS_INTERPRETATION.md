# Accepted Loan Status Business Interpretation

This document translates the accepted-loan EDA `loan_status` values into business language for repayment-risk analysis. It is based on the full accepted LendingClub dataset, not only the 250,000-row EDA working sample.

## Business Decision

For binary repayment/default modeling, use only loans with sufficiently observed outcomes. Do not treat active or newly issued loans as good outcomes just because they have not failed yet.

- `Fully Paid` statuses are low observed repayment risk and can be treated as good terminal outcomes.
- `Charged Off`, `Default`, and late delinquency statuses are elevated repayment risk and can be treated as bad outcomes.
- `Current`, `Issued`, and `In Grace Period` are unresolved or not fully mature outcomes. They should not be labeled as low risk in a binary default model.

## Full-Dataset Loan-Status Interpretation

Full accepted dataset size: 2,260,701 rows.

| Loan status | Loan-status definition | Target treatment | Repayment-risk interpretation | Full rows | Full % |
| --- | --- | --- | --- | ---: | ---: |
| `Fully Paid` | The borrower completed repayment and the loan reached a successful terminal outcome. | Good terminal | Low observed repayment risk | 1,076,751 | 47.6291 |
| `Current` | The loan is still active/open and has not reached a final repayment or failure outcome. | Censored unresolved | Unresolved, not risk-ranked for binary PD | 878,317 | 38.8515 |
| `Charged Off` | The lender has written off the loan as a loss after serious nonpayment. | Bad terminal or delinquent | High observed repayment risk | 268,559 | 11.8795 |
| `Late (31-120 days)` | The borrower is materially delinquent, 31 to 120 days past due. | Bad terminal or delinquent | High observed repayment risk | 21,467 | 0.9496 |
| `In Grace Period` | The payment is recently overdue but still within the grace-period window. | Censored unresolved | Medium/current delinquency review, not final target | 8,436 | 0.3732 |
| `Late (16-30 days)` | The borrower is early delinquent, 16 to 30 days past due. | Bad terminal or delinquent | Medium/current delinquency review | 4,349 | 0.1924 |
| `Does not meet the credit policy. Status:Fully Paid` | The loan did not meet LendingClub credit policy, but the borrower ultimately fully repaid it. | Good terminal | Low observed repayment risk | 1,988 | 0.0879 |
| `Does not meet the credit policy. Status:Charged Off` | The loan did not meet LendingClub credit policy and was later charged off. | Bad terminal or delinquent | High observed repayment risk | 761 | 0.0337 |
| `Default` | The borrower defaulted under the loan's default definition. | Bad terminal or delinquent | High observed repayment risk | 40 | 0.0018 |
| Missing | No `loan_status` value is available. | Missing status | Missing status | 33 | 0.0015 |

## Aggregated Business View

| Business group | Rows | Full % | Modeling treatment |
| --- | ---: | ---: | --- |
| Low observed repayment risk / good terminal | 1,078,739 | 47.7170 | Eligible as good class for completed-loan binary modeling. |
| Unresolved censored and not risk-ranked | 878,317 | 38.8515 | Exclude from binary target or handle with censoring-aware methods. |
| High observed repayment risk / bad terminal or delinquent | 290,827 | 12.8645 | Eligible as bad class for completed-loan binary modeling. |
| Medium/current or early delinquency review | 12,785 | 0.5656 | Review carefully; some statuses are unresolved while others are early delinquency. |
| Missing status | 33 | 0.0015 | Exclude or investigate before modeling. |

## Key Interpretation

The accepted-loan dataset is imbalanced, but the issue is not `loan_status` missingness. `loan_status` is missing for only 33 rows, or 0.0015% of the full dataset.

The main modeling issue is censoring: 38.8515% of the full dataset is `Current`. These loans are still active and have not yet become either fully paid or charged off/defaulted. Labeling them as good loans would create target bias.

