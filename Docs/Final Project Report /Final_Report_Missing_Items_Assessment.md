# Final Report Missing Items Assessment

This checklist maps the current project artifacts to the rubric in `Final Project Report.md` and identifies what still needs to be completed before drafting or submitting the final PDF.

## High-Priority Missing Items

| Item | Status | Why It Matters | Suggested Action |
| --- | --- | --- | --- |
| 2-3 Kaggle related implementations | Missing / not found in repo | The rubric assigns 11 points to comparing against Kaggle work on the same dataset. | Select 2-3 Kaggle notebooks/kernels/posts, save URLs, summarize methods, and explain why this project is stronger. |
| Public GitHub repository URL | Needs verification | The appendix must include a public repo link or a zip submission if the code cannot be public. | Confirm final public URL and add it to the report appendix. |
| Exact notebook cell links for experimental setup | Missing | The rubric asks for links to cells containing setup details. | Add GitHub notebook links or anchor-style references for preprocessing and model setup cells. |
| Complete AI prompts used | Partial | The assignment requires all AI prompts used for coding, debugging, explanations, and research. | Start from `CreditRiskRAG/Docs/Prompts/Prompt.md`, then add any additional prompts used by the team. |
| Final contribution statements | Partial | Required for groups of 2-3. | Verify Linda, Yashaswi, and Siddharth contributions with the team before submission. |
| Final report screenshots/figures | Available but not selected | Results section requires figures/screenshots. | Choose 3-4 final figures: EDA risk plot, model comparison, SHAP plot, and policy/review-volume plot. |

## Rubric Coverage

| Rubric Criterion | Points | Current Evidence | Gap / Required Work |
| --- | ---: | --- | --- |
| Introduction | 5 | Strong material exists in `README.md`, `Docs/Business/`, and EDA decision logs. Target and challenges are documented. | Draft 2-3 coherent paragraphs. Explicitly define `target_bad`, explain `Fully Paid` vs `Charged Off`, and name 2-3 challenges: leakage, class imbalance, temporal drift, selection bias. |
| Proposed Method summary | 5 | Strong material exists in `README.md` and `Modeling/MODELING_INTERPRETATION.md`. | Draft one paragraph explaining why leakage-aware chronological modeling plus neutral XGBoost, calibration, SHAP, and capped policy is better than best-F1 or grade/subgrade-heavy alternatives. |
| Related Work | 11 | `Docs/Problem_Research/Papers.md` and `Paper_Reading_Summary.md` contain 6+ candidate papers. | Convert summaries into 3-5 paragraphs with numbered references. Verify any claims copied from paper summaries before final citation. |
| Related implementations | 11 | No complete 2-3 Kaggle implementation comparisons were found. | Required gap. Find Kaggle notebooks/kernels/posts for the LendingClub dataset, cite URLs, and write 2-3 paragraphs. |
| Data Analysis | 8 | EDA plots/tables and `EDA/ACCEPTED_LOAN_EDA_DECISION_LOG.md` are strong. | Select 1-2 most useful plots and write 5-6 paragraphs. Include formulas for derived features such as `fico_mean` and `credit_history_years`. |
| Proposed Method / ML algorithms | 8 | Model notebooks, output tables, and `MODELING_INTERPRETATION.md` cover algorithms and metrics. | Write 3-5 paragraphs explaining algorithms, feature extraction, tuning, metrics, calibration, and economic thresholding. Include formulas. |
| Analysis | 15 | Strong reasoning exists in README, modeling interpretation, SHAP README, and business summary. | Write 3-5 insight-heavy paragraphs explaining why the features and model family fit this dataset and what generalizes to similar credit-risk datasets. Avoid only recounting metrics. |
| Experimental Setup | 8 | Chronological split table, preprocessing outputs, model notebooks, and reproducibility docs exist. | Draft 3-4 paragraphs and add exact notebook/cell links. Include train/validation/test dates, row counts, bad rates, selection process, and holdout policy. |
| Results | 11 | Final comparison, ablation, calibration, SHAP, and economic policy outputs exist. | Build one compact results table, one ablation table, and 2-3 figures. Write 5-6 interpretive paragraphs. |
| Conclusion | 4 | Recommendation material exists in `README.md`, final comparison outputs, and business docs. | Draft exactly 3 paragraphs with model recommendation, capped policy recommendation, and future work. |
| References | 4 | Paper references and Kaggle dataset link exist. | Need final numbered bibliography with at least 6 references and all in-text citations. Add Kaggle implementation references. |
| Completion / Style | 10 | Project has strong artifacts and documentation. | Keep final report formal, concise, citation-backed, and under 8 pages excluding allowed sections. |

## Recommended Report Evidence To Use

| Report Section | Best Supporting Artifacts |
| --- | --- |
| Introduction | `CreditRiskRAG/README.md`, `Docs/Business/1.-ACCEPTED_LOAN_STATUS_BUSINESS_INTERPRETATION.md`, `Others/POSSIBLE_RISKS.md` |
| Dataset / Inputs | `CreditRiskRAG/data_manifest.json`, `CreditRiskRAG/LCDataDictionary.xlsx`, `CreditRiskRAG/EDA/accepted_eda_outputs/tables/accepted_file_overview.csv` |
| Cleaning / Leakage | `CreditRiskRAG/Cleaning/FEATURE_CLEANING_DECISIONS.md`, `CreditRiskRAG/EDA/ACCEPTED_LOAN_EDA_DECISION_LOG.md` |
| Experimental Setup | `CreditRiskRAG/Modeling/Preprocessing/0_Preprocessing.ipynb`, `preprocessing_split_summary.csv`, `Docs/REPRODUCIBILITY.md` |
| Model Results | `CreditRiskRAG/Modeling/modeling_outputs/final_comparison/tables/final_model_selected_metrics.csv`, `final_model_selected_candidates.csv`, `final_model_validation_test_comparison.csv` |
| Ablation | `final_model_grade_subgrade_ablation_metrics.csv`, `final_model_grade_subgrade_ablation_deltas.csv` |
| Calibration / Policy | `final_model_calibration_summary.csv`, `final_model_economic_underwriting_recommendation.csv`, `final_model_review_volume_precision.csv` |
| Interpretation | `CreditRiskRAG/Interpretation_SHAP/README.md`, `shap_global_mean_abs_importance.csv`, `shap_reason_code_family_importance.csv`, SHAP plots |
| Related Work | `CreditRiskRAG/Docs/Problem_Research/Papers.md`, `Paper_Reading_Summary.md` |
| AI Prompts | `CreditRiskRAG/Docs/Prompts/Prompt.md` plus any team prompt logs |

## Suggested Figures

| Purpose | Candidate File |
| --- | --- |
| EDA risk relationship | `CreditRiskRAG/EDA/accepted_eda_outputs/plots/observed_bad_rate_by_grade.png` |
| EDA numeric risk relationship | `CreditRiskRAG/EDA/accepted_eda_outputs/plots/observed_bad_rate_by_fico_mean_band.png` |
| Final model comparison | `CreditRiskRAG/Modeling/modeling_outputs/final_comparison/plots/final_model_pr_auc_comparison.png` |
| Business threshold behavior | `CreditRiskRAG/Modeling/modeling_outputs/final_comparison/plots/final_model_precision_by_review_volume_test.png` |
| SHAP global interpretation | `CreditRiskRAG/Interpretation_SHAP/shap_outputs/plots/shap_mean_abs_bar.png` |
| SHAP local example | `CreditRiskRAG/Interpretation_SHAP/shap_outputs/plots/shap_waterfall_top_risk_example.png` |

## Core Numbers To Carry Into The Draft

| Fact | Value |
| --- | --- |
| Raw accepted-loan rows | 2,260,701 |
| Raw accepted-loan columns | 151 |
| Completed modeling rows | 1,345,310 |
| Train split | 962,641 rows, 18.83% bad rate, 2007-06 to 2016-04 |
| Validation split | 186,920 rows, 24.68% bad rate, 2016-05 to 2017-02 |
| Test split | 195,749 rows, 21.03% bad rate, 2017-03 to 2018-12 |
| Preferred model | `xgb_neutral_09_without_grade_subgrade_with_int_rate` |
| Preferred model test ROC-AUC | 0.710201 |
| Preferred model test PR-AUC | 0.380806 |
| Preferred model test F1 | 0.440096 |
| Preferred model test precision | 0.330063 |
| Preferred model test recall | 0.660181 |
| Preferred model top-20% bad rate | 40.09% |
| SHAP validation-test stability | Spearman rank correlation 0.9987 |
| Top SHAP drivers | `int_rate_clean`, `term_months`, `acc_open_past_24mths`, `dti`, `fico_mean`, `annual_inc`, `loan_amnt` |

## Do Not Overclaim

- Do not say the system is production-ready for lending decisions.
- Do not say the RAG adverse-action system is legally compliant or fully complete.
- Do not use rejected applications as labeled default outcomes.
- Do not present LendingClub `grade` or `sub_grade` as independent borrower attributes; they are LendingClub risk/pricing outputs.
- Do not make fairness or compliance claims without explicit evaluation.
- Do not fabricate Kaggle implementation references, GitHub URLs, paper details, or notebook cell links.

