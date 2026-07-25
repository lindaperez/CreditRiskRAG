# Modeling Interpretation From The Preferred Final Model

## Purpose

This document explains how to interpret the final modeling results for the accepted-loan credit-risk project. It focuses on the preferred production-style model:

```text
xgb_neutral_09_without_grade_subgrade_with_int_rate
```

This model predicts:

```text
P(target_bad = 1)
```

Where:

| Target | Meaning | Project Interpretation |
| ---: | --- | --- |
| `0` | `Fully Paid` | Good repayment outcome |
| `1` | `Charged Off` | Bad repayment outcome / default-risk event |

Because this is a binary problem, predicting default risk and predicting repayment probability are complements:

```text
P(Fully Paid) = 1 - P(Charged Off)
```

The project reports the model as a default-risk model because the business objective is to identify loans with elevated risk of charge-off. That is the more important and harder class to detect.

## Final Model Selected For Interpretation

Preferred model:

```text
xgb_neutral_09_without_grade_subgrade_with_int_rate
```

Model family:

```text
XGBoost
```

Feature policy:

- Uses the `missingness_challenger` feature set.
- Removes `grade` and `sub_grade`.
- Keeps `int_rate_clean`.
- Uses neutral class weighting with `scale_pos_weight = 1`.
- Supports SHAP interpretation.

This model was preferred because it gives almost the same ranking performance as the strict statistical alternatives while being more defensible for governance. Explicit LendingClub `grade` and `sub_grade` fields are removed, but the continuous interest-rate signal is retained.

LightGBM achieved the strongest validation F1 among the main model families, so it is the best technical F1 benchmark. The final preferred model remains neutral XGBoost without `grade/sub_grade` and with `int_rate_clean`, because it gives nearly equivalent ranking performance while being stronger for governance, calibration, and business-policy explanation.

## Why This Model Is Defensible

The grade/subgrade and interest-rate ablation showed that `grade`, `sub_grade`, and `int_rate_clean` carry highly overlapping information. Keeping either grade/subgrade or interest rate preserved nearly all predictive performance. Keeping both did not materially improve performance.

| Feature Policy | Validation PR-AUC | Test PR-AUC | Test ROC-AUC | Test Top-20% Bad Rate |
| --- | ---: | ---: | ---: | ---: |
| With `grade/sub_grade` + `int_rate_clean` | 0.426323 | 0.380704 | 0.710495 | 39.98% |
| Without `grade/sub_grade` + `int_rate_clean` | 0.426423 | 0.380806 | 0.710201 | 40.09% |
| With `grade/sub_grade`, without `int_rate_clean` | 0.426470 | 0.381304 | 0.710762 | 39.98% |
| Without both | 0.422215 | 0.375358 | 0.703070 | 39.64% |

The strict best PR-AUC result is only slightly higher, and the difference is not practically meaningful. The preferred model removes explicit grade buckets, retains predictive power, and remains compatible with SHAP explanations. That is a reasonable tradeoff for a project concerned with both performance and interpretability.

## Main Metrics For The Preferred Model

| Metric | Validation | Test |
| --- | ---: | ---: |
| Rows | 186,920 | 195,749 |
| Bad/default rate | 24.68% | 21.03% |
| ROC-AUC | 0.702038 | 0.710201 |
| PR-AUC | 0.426423 | 0.380806 |
| Mean raw predicted probability | 0.199975 | 0.197260 |
| Best-F1 threshold | 0.173938 | 0.191332 |
| F1 for `Charged Off` | 0.474964 | 0.440096 |
| Precision for `Charged Off` | 0.360500 | 0.330063 |
| Recall for `Charged Off` | 0.695935 | 0.660181 |
| Predicted risky share at best-F1 threshold | 47.64% | 42.07% |

## Interpretation Of Each Metric

### Bad / Default Rate

The bad rate is the share of loans that ended as `Charged Off`.

Validation bad rate:

```text
24.68%
```

Test bad rate:

```text
21.03%
```

This tells us the dataset is imbalanced. Most loans are fully paid, and the default-risk class is the minority class. Because of that, accuracy alone would be misleading. A model could predict most loans as fully paid and still look superficially good while failing the actual risk-detection objective.

Project impact:

The project correctly emphasizes PR-AUC, precision, recall, review-volume metrics, and economic policy instead of plain accuracy.

### ROC-AUC

ROC-AUC measures how well the model separates charged-off loans from fully paid loans across all possible thresholds.

Test ROC-AUC:

```text
0.710201
```

Plain interpretation:

If we randomly choose one charged-off loan and one fully paid loan, the model will assign a higher risk score to the charged-off loan about 71% of the time.

Project impact:

This is a solid result for noisy consumer-credit tabular data. It shows the model has real ranking signal, but it is not perfect. The score should be used as a risk-ranking tool, not as an unquestioned automated lending decision.

### PR-AUC

PR-AUC measures precision-recall performance for the minority class, `Charged Off`.

Test PR-AUC:

```text
0.380806
```

Plain interpretation:

The model is meaningfully better than the base default rate of 21.03%, because it can concentrate bad loans in higher-risk score ranges. PR-AUC is especially important here because defaults are less common than fully paid loans.

Project impact:

The PR-AUC result supports using the model to rank loans by default risk. It does not mean every flagged loan will default. It means the high-risk part of the score distribution has a much higher bad rate than the portfolio average.

### Mean Predicted Probability

The mean raw predicted probability on the test set is:

```text
0.197260
```

The actual test bad rate is:

```text
0.210315
```

Plain interpretation:

The model's average raw score is close to the actual bad rate, especially compared with earlier weighted XGBoost models that inflated predicted probabilities. This is one reason neutral class weighting was preferred.

Project impact:

The model is more reasonable as a probability input than the earlier class-weighted version, but probability-based decisions should still use calibration.

### Precision For `Charged Off`

Test precision for `Charged Off` at the best-F1 threshold:

```text
0.330063
```

Plain interpretation:

Of all loans predicted as risky, about 33.0% actually became charged off.

This can sound low, but it must be compared to the test bad rate of 21.03%. The model is concentrating risk: the predicted-risky group has a higher bad rate than the overall test portfolio.

Project impact:

This precision is not high enough to justify saying every flagged borrower will default. It is appropriate for prioritizing review, setting risk thresholds, and supporting policy analysis.

### Recall For `Charged Off`

Test recall for `Charged Off`:

```text
0.660181
```

Plain interpretation:

Of all loans that actually charged off, the model identified about 66.0% as risky at the best-F1 threshold.

Project impact:

The model catches a majority of future charge-offs, which is useful. However, the best-F1 threshold flags 42.07% of all test loans as risky, which is too broad for a realistic rejection or manual-review policy.

### F1 For `Charged Off`

Test F1:

```text
0.440096
```

Plain interpretation:

F1 balances precision and recall for the charged-off class. It is useful as a technical model comparison metric, but it is not automatically the correct business threshold.

Project impact:

The project correctly moved beyond best-F1. The best-F1 threshold catches many defaults, but it reviews or rejects too many loans. That is why the final policy uses calibrated probabilities and a capped economic-review strategy.

### Predicted Risky Share

At the best-F1 threshold, the test predicted risky share is:

```text
42.07%
```

Plain interpretation:

The model would flag about 42 out of every 100 test loans as risky.

Project impact:

This is too high for a realistic underwriting or review operation. It would create substantial false positives and could reject many borrowers who would have fully repaid. This is the main reason the project does not use the best-F1 threshold as the final business policy.

## Fully Paid Class Interpretation

The model also implies predictions for `Fully Paid`, because:

```text
P(Fully Paid) = 1 - P(Charged Off)
```

At the best-F1 threshold, the derived `Fully Paid` class metrics are:

| Metric | Validation | Test |
| --- | ---: | ---: |
| Fully Paid support | 140,795 | 154,580 |
| Predicted Fully Paid count | 97,877 | 113,404 |
| Fully Paid precision | 85.67% | 87.66% |
| Fully Paid recall | 59.56% | 64.31% |
| Fully Paid F1 | 70.27% | 74.19% |

Test interpretation:

- Fully Paid precision of 87.66% means that among loans predicted as fully paid, about 87.7% actually were fully paid.
- Fully Paid recall of 64.31% means that among all truly fully paid loans, the model identified about 64.3% as fully paid.

These numbers are higher than charged-off precision because fully paid is the majority class. That does not mean the model's main job should be described as predicting repayment. The business risk objective is to detect the minority bad-outcome class.

## Top-20% Review Policy

Instead of using the best-F1 threshold, the project also evaluates a fixed review-volume policy: review the riskiest 20% of loans by predicted default risk.

| Metric | Validation | Test |
| --- | ---: | ---: |
| Review percent | 20.0% | 20.0% |
| Review count | 37,384 | 39,150 |
| Review threshold | 0.297223 | 0.300715 |
| Defaults captured | 16,819 | 15,697 |
| Review precision / bad rate among reviewed | 44.99% | 40.09% |
| Review recall / default capture | 36.46% | 38.13% |

Plain interpretation:

The top-20% policy reviews fewer loans than the best-F1 threshold, but the reviewed group is more concentrated with true charge-offs. On the test set, the reviewed group has a 40.09% bad rate, compared with the overall test bad rate of 21.03%.

Project impact:

This is more operationally realistic. It creates a manageable review queue and focuses attention on the riskiest segment instead of flagging nearly half the portfolio.

## Capped Economic Policy

The capped economic policy uses Platt-calibrated probabilities, a dollar-value objective, and a maximum reject/review cap around 20%.

| Metric | Validation | Test |
| --- | ---: | ---: |
| Calibrated policy threshold | 0.327936 | 0.327936 |
| Predicted reject/review share | 20.00% | 20.49% |
| Approved share | 80.00% | 79.51% |
| Bad rate among rejected/reviewed | 44.99% | 39.91% |
| Default capture | 36.46% | 38.88% |
| Approved bad rate | 19.60% | 16.17% |
| Estimated portfolio value | $137,342,500 | $123,907,500 |
| Value per applicant | $734.77 | $632.99 |

Plain interpretation:

The policy rejects or reviews roughly the riskiest 20% of loans. Within that group, about 39.91% charged off on the test set. The loans still approved have a lower bad rate of 16.17%.

Project impact:

The economic policy is a better final decision rule than best-F1 because it connects model predictions to business consequences. It also prevents the model from becoming too aggressive by limiting the review/reject share.

## Defense Of The Results

The results are defensible for four reasons.

First, the model generalizes reasonably from validation to test. Validation ROC-AUC is 0.702038 and test ROC-AUC is 0.710201. Validation PR-AUC is 0.426423 and test PR-AUC is 0.380806. The lower test PR-AUC is expected because the test period has a lower bad rate, and PR-AUC is sensitive to class prevalence.

Second, the model improves risk concentration. The overall test bad rate is 21.03%, but the top-20% reviewed group has a bad rate of 40.09%. That means the model nearly doubles the concentration of bad outcomes in the review queue.

Third, the model avoids overclaiming. The charged-off precision of 33.0% at the best-F1 threshold means many predicted risky loans are still fully paid. The project therefore should not claim the model can perfectly identify defaults. Instead, it should claim the model ranks default risk meaningfully and supports review prioritization.

Fourth, the selected feature policy is reasonable. Removing `grade` and `sub_grade` barely changes performance, while improving governance and parsimony. Keeping `int_rate_clean` retains an important accepted-loan pricing signal. The final choice balances prediction, interpretability, and practical policy use.

## Limitations

The model is not a production lending approval system.

Key limitations:

- It predicts outcomes for accepted LendingClub loans, not all applicants.
- Rejected applications do not have observed repayment outcomes, so they cannot be used directly to train this default label.
- Because the training labels exist only after LendingClub accepted and funded a loan, the model estimates default risk conditional on historical approval. It cannot represent the risk distribution of rejected applicants without additional outcome data, reject-inference assumptions, or a separate applicant-population study.
- `Charged Off` is treated as the default-risk event, but it is a loan-status proxy for default, not a full bank regulatory default definition.
- Some useful features may encode policy, pricing, or proxy-risk information.
- The model's precision shows that many loans flagged as risky would still fully repay.
- Any applicant-facing explanation requires compliance and fair-lending review.

## How Performance Could Be Improved

### 1. Improve Probability Calibration

The project already uses Platt calibration, but calibration could be strengthened by comparing:

- Platt sigmoid calibration.
- Isotonic calibration.
- Time-period-specific calibration.
- Calibration by score band and borrower segment.

Better calibration would improve threshold decisions and economic policy reliability.

### 2. Optimize For PR-AUC And Review Precision

Because charge-off is the minority class, model tuning should emphasize:

- PR-AUC.
- Precision at fixed review volume.
- Recall at fixed review volume.
- Bad rate among the top 5%, 10%, and 20% risk bands.

This would align tuning more directly with the business use case than best-F1 alone.

### 3. Tune The Review Policy, Not Only The Model

Performance can improve operationally even if model AUC does not change. The policy layer can be tuned by testing:

- 10%, 15%, 20%, and 25% review caps.
- Different false-positive opportunity costs.
- Different default-loss assumptions.
- Separate thresholds for different loan terms or risk bands.

This is important because the model is only one part of the decision system.

### 4. Add Better Application-Time Features

Additional leakage-safe features could improve performance, especially if they are known at origination:

- More robust credit-history summaries.
- Income-to-installment ratios.
- Recent credit-seeking behavior summaries.
- Utilization and balance trend features.
- Policy-compliant borrower capacity measures.

Any new feature should pass leakage review and fair-lending review.

### 5. Improve Feature Engineering

The current model uses strong tabular features, but performance may improve with better transformations:

- Nonlinear bins for DTI, FICO, utilization, and income.
- Interaction terms such as `term_months x int_rate_clean`.
- Payment burden measures such as `installment / monthly_income`.
- Robust winsorization for extreme values.
- Missingness mechanism features for sparse credit-history fields.

Tree models can learn many interactions, but explicit domain features can still help.

### 6. Use Time-Aware Validation And Monitoring

Credit risk changes over time. The model should be evaluated across issue-year or issue-quarter slices:

- PR-AUC by time period.
- Bad rate by score band over time.
- Calibration drift over time.
- Feature distribution drift.
- Stability of SHAP feature importance.

If drift is large, retraining or recalibration may improve performance.

### 7. Consider Ensemble Or Stacking Approaches

LightGBM, XGBoost, and CatBoost performed similarly. A carefully validated ensemble may improve ranking performance:

- Average calibrated probabilities.
- Stack model outputs with a simple logistic meta-model.
- Use ensemble ranking only for review prioritization.

This should be done carefully because ensembles are harder to explain and govern.

### 8. Evaluate Simpler Interpretable Models As Challengers

If interpretability becomes more important than small performance gains, the project could test:

- Explainable Boosting Machine.
- Regularized logistic regression with engineered bins.
- Monotonic gradient boosting constraints.

These may slightly reduce AUC but improve transparency and policy defensibility.

## Final Interpretation

The preferred XGBoost model is best understood as a calibrated default-risk ranking model for accepted LendingClub loans. It does not perfectly predict who will default, but it meaningfully separates high-risk loans from lower-risk loans.

The strongest project result is not the best-F1 classifier. The strongest result is the policy interpretation:

```text
Use the model to rank accepted loans by default risk,
calibrate probabilities,
review roughly the riskiest 20%,
and use SHAP to explain the main risk drivers.
```

That framing is honest, technically defensible, and aligned with the project's business objective.
