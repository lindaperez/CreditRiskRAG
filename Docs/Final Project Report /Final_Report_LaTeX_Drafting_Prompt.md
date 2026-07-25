# Prompt: Draft The Final Project Report In LaTeX

Use this prompt with an AI writing assistant to draft the final report as a scientific paper in LaTeX. The report must follow the grading rubric in `CreditRiskRAG/Docs/Final Project Report /Final Project Report.md`.

```text
Act as a senior machine-learning researcher 15+ years of experience from Bank of america and scientific editor. Draft a polished final project report in LaTeX for a Northeastern CS 6140 machine-learning final project. Use formal academic writing, concise technical explanations, and evidence from the repository and current project. The final report must be no more than 8 pages at 11 pt, single-spaced, excluding references, appendix, AI prompts used, and statement of contributions if those are allowed outside the page limit by the assignment.

Project title:
CreditRisk Interpretation RAG System: A Leakage-Aware Credit-Risk Modeling And Policy Analysis Project For LendingClub Loan Outcomes

Authors:
Linda Perez Penaranda, Yashaswi Aryan, Siddharth Agarwal

Repository context:
The project uses the Kaggle "All Lending Club loan data" dataset:
https://www.kaggle.com/datasets/wordsforthewise/lending-club

Current completed scope:
- Accepted-loan default-risk modeling for completed LendingClub loans.
- EDA, cleaning, leakage audit, preprocessing, chronological train/validation/test split, baseline model comparison, advanced gradient boosting, calibration, SHAP interpretation, grade/subgrade and interest-rate ablation, and capped economic underwriting policy analysis.
- The RAG adverse-action explanation layer is planned or partially exploratory, not the main completed production result. Do not overclaim that a legally compliant RAG system is complete unless supported by the artifacts.

Primary target:
Binary target `target_bad`, where:
- `0` means `Fully Paid`
- `1` means `Charged Off`
The report should explain that unresolved/current loans are excluded because final repayment outcomes are unknown. Rejected applications are not used for supervised default labels because repayment outcomes are not observed for rejected applicants.

Preferred model to emphasize:
`xgb_neutral_09_without_grade_subgrade_with_int_rate`

Interpretation:
This model predicts `P(target_bad = 1)`, or the probability of charge-off/default for accepted loans. It uses a no-grade/subgrade feature policy, retains `int_rate_clean`, uses neutral class weighting, and supports SHAP interpretation.

Important metrics and facts to use:
- Raw accepted-loan source: 2,260,701 accepted-loan rows and 151 columns.
- Clean completed modeling frame: 1,345,310 rows.
- Starter feature count after cleaning: 38.
- Chronological split:
  - Train: 962,641 rows, bad rate 18.83%, issue dates 2007-06-01 to 2016-04-01.
  - Validation: 186,920 rows, bad rate 24.68%, issue dates 2016-05-01 to 2017-02-01.
  - Test: 195,749 rows, bad rate 21.03%, issue dates 2017-03-01 to 2018-12-01.
- Preferred model validation metrics:
  - ROC-AUC 0.702038
  - PR-AUC 0.426423
  - F1 for Charged Off 0.474964
  - Precision for Charged Off 0.360500
  - Recall for Charged Off 0.695935
- Preferred model test metrics:
  - ROC-AUC 0.710201
  - PR-AUC 0.380806
  - F1 for Charged Off 0.440096
  - Precision for Charged Off 0.330063
  - Recall for Charged Off 0.660181
  - Mean raw predicted probability 0.197260 versus actual test bad rate 0.210315
  - Predicted risky share at best-F1 threshold 42.07%, which is too high for a realistic business rejection/review policy.
- Grade/subgrade and interest-rate ablation:
  - With grade/sub_grade + int_rate_clean: validation PR-AUC 0.426323, test PR-AUC 0.380704, test ROC-AUC 0.710495, test top-20% bad rate 39.98%.
  - Without grade/sub_grade + int_rate_clean: validation PR-AUC 0.426423, test PR-AUC 0.380806, test ROC-AUC 0.710201, test top-20% bad rate 40.09%.
  - With grade/sub_grade, without int_rate_clean: validation PR-AUC 0.426470, test PR-AUC 0.381304, test ROC-AUC 0.710762, test top-20% bad rate 39.98%.
  - Without both: validation PR-AUC 0.422215, test PR-AUC 0.375358, test ROC-AUC 0.703070, test top-20% bad rate 39.64%.
- Economic policy example:
  - Use calibrated probabilities and a capped economic decision rule, not unconstrained best-F1.
  - With a max reject/review share around 20%, example test policy rejected 19.47% of loans, captured 37.43% of defaults, produced 40.44% bad rate among rejected loans, and estimated value of about $120,043,000 under the example assumptions.
  - Example cost assumptions: catching a default saves $10,000; rejecting a good borrower costs $1,500; break-even default probability is 13.04%.
- SHAP interpretation:
  - SHAP was run on a reproducible 5,000-row validation sample with seed 42 and checked against test stability.
  - Top global drivers by mean absolute SHAP: `int_rate_clean`, `term_months`, `acc_open_past_24mths`, `dti`, `fico_mean`, `annual_inc`, and `loan_amnt`.
  - Validation vs test SHAP rank stability Spearman correlation: 0.9987.
  - Directional findings: higher interest rate, longer term, higher recent account-opening activity, higher DTI, and larger requested loan increase model risk; higher FICO and income lower model risk.

Required source artifacts to consult and cite in the report narrative:
- `CreditRiskRAG/README.md`
- `CreditRiskRAG/EDA/ACCEPTED_LOAN_EDA_DECISION_LOG.md`
- `CreditRiskRAG/Cleaning/FEATURE_CLEANING_DECISIONS.md`
- `CreditRiskRAG/Modeling/Preprocessing/preprocessing_outputs/tables/preprocessing_split_summary.csv`
- `CreditRiskRAG/Modeling/MODELING_INTERPRETATION.md`
- `CreditRiskRAG/Interpretation_SHAP/README.md`
- `CreditRiskRAG/Docs/Problem_Research/Papers.md`
- `CreditRiskRAG/Docs/Problem_Research/Paper_Reading_Summary.md`
- `CreditRiskRAG/Modeling/modeling_outputs/final_comparison/tables/final_model_selected_metrics.csv`
- `CreditRiskRAG/Modeling/modeling_outputs/final_comparison/tables/final_model_grade_subgrade_ablation_metrics.csv`
- `CreditRiskRAG/Modeling/modeling_outputs/final_comparison/tables/final_model_economic_underwriting_recommendation.csv`

Figures and tables to include, using LaTeX `figure` and `table` environments:
1. One EDA plot showing a strong risk relationship, such as:
   - `CreditRiskRAG/EDA/accepted_eda_outputs/plots/observed_bad_rate_by_grade.png`
   - `CreditRiskRAG/EDA/accepted_eda_outputs/plots/observed_bad_rate_by_fico_mean_band.png`
   - `CreditRiskRAG/EDA/accepted_eda_outputs/plots/observed_bad_rate_by_dti_band.png`
2. One final model comparison plot, such as:
   - `CreditRiskRAG/Modeling/modeling_outputs/final_comparison/plots/final_model_pr_auc_comparison.png`
   - `CreditRiskRAG/Modeling/modeling_outputs/final_comparison/plots/final_model_roc_auc_comparison.png`
   - `CreditRiskRAG/Modeling/modeling_outputs/final_comparison/plots/final_model_precision_by_review_volume_test.png`
3. One SHAP interpretation plot, such as:
   - `CreditRiskRAG/Interpretation_SHAP/shap_outputs/plots/shap_mean_abs_bar.png`
   - `CreditRiskRAG/Interpretation_SHAP/shap_outputs/plots/shap_summary_beeswarm.png`
4. A compact results table comparing Logistic Regression, Random Forest, HistGradientBoosting, LightGBM, XGBoost, CatBoost, and the preferred neutral XGBoost no-grade/subgrade model using validation and test ROC-AUC, PR-AUC, precision, recall, and F1.
5. A compact ablation table showing the grade/subgrade and interest-rate tradeoff.
6. A compact economic policy table showing best-F1 is too broad and the capped economic policy is more realistic.

Required report structure:

Use this LaTeX structure:

\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{times}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{hyperref}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{enumitem}
\setlist{nosep}
\title{CreditRisk Interpretation RAG System: Leakage-Aware Default-Risk Modeling for LendingClub Loans}
\author{Linda Perez Penaranda \and Yashaswi Aryan \and Siddharth Agarwal}
\date{}

\begin{document}
\maketitle

\begin{abstract}
Write a concise abstract summarizing the problem, data, approach, best model, strongest result, and main limitation.
\end{abstract}

\section{Introduction}
Write 2-3 paragraphs. Explain why LendingClub default prediction is important, define the primary task and target variable, and identify 2-3 dataset challenges: class imbalance, temporal drift, leakage from post-origination fields, selection bias because rejected applications lack repayment outcomes, and compliance/explainability constraints. Include a non-technical summary of the solution.

\section{Proposed Method Summary}
Write one paragraph. State that the final method is a leakage-controlled accepted-loan default-risk pipeline using chronological splits, cleaned application-time features, multiple model families, calibration, SHAP interpretation, and a capped economic policy. Explain why this is better than simply optimizing F1 or using LendingClub grade/subgrade directly.

\section{Related Work}
Write 3-5 paragraphs. Discuss 5-6 published papers from `Docs/Problem_Research/Papers.md`. For each, describe the dataset, features, algorithms, and relevance in 1-2 sentences. Compare the current project against them: this project emphasizes leakage control, chronological validation, no-grade/subgrade governance, calibration, SHAP stability, and policy thresholds. Use numbered citations.

\section{Related Implementations}
Write 2-3 paragraphs. Include 2-3 Kaggle notebooks, kernels, or posts that use the same Kaggle LendingClub dataset. For each, cite the URL and summarize what they did in 1-2 sentences. Compare against this project. If exact Kaggle implementation sources have not been selected yet, insert clear TODO placeholders and do not fabricate URLs.

\section{Dataset and Inputs}
Describe the accepted and rejected LendingClub files, the source, years covered, target mapping, excluded statuses, and why rejected applications are not supervised labels. Cite Kaggle and the LendingClub data dictionary if used. Explain cleaning, type conversion, missingness review, and leakage removal.

\section{Data Analysis}
Write 5-6 paragraphs. Describe EDA findings and include 1-2 plots. Discuss loan status distribution, grade/subgrade, interest rate, FICO, DTI, income, term, purpose, missingness, and temporal changes. Include formulas for derived features where relevant, for example:
\[
\text{fico\_mean} = \frac{\text{fico\_range\_low} + \text{fico\_range\_high}}{2}
\]
\[
\text{credit\_history\_years} = \frac{\text{issue\_date} - \text{earliest\_credit\_line}}{365.25}
\]
Explain that leakage fields such as payments, recoveries, last payment dates, hardship, settlement, and collection fields were excluded.

\section{Methods}
Write 3-5 detailed paragraphs. Explain Logistic Regression, Random Forest, HistGradientBoosting, LightGBM, XGBoost, CatBoost, calibration, SHAP, and threshold selection. Include formulas for logistic regression probability, binary cross-entropy, precision, recall, F1, ROC-AUC/PR-AUC definitions in words, and expected-value policy:
\[
\hat{p}(x)=P(y=1\mid x)
\]
\[
F_1 = 2\cdot\frac{\text{precision}\cdot\text{recall}}{\text{precision}+\text{recall}}
\]
\[
\text{EV} = TP\cdot L_{\text{saved}} - FP\cdot C_{\text{opportunity}}
\]
Explain why PR-AUC and top-risk review precision are more useful than accuracy for imbalanced default prediction.

\section{Experimental Setup}
Write 3-4 paragraphs. Describe the chronological train/validation/test split, row counts, bad rates, feature preprocessing, target definition, candidate model families, validation-based model selection, test-set holdout, calibration, and reproducibility. Include links or TODOs to exact notebook cells for setup:
- `Modeling/Preprocessing/0_Preprocessing.ipynb`
- `Modeling/1_LogisticRegression_Modeling.ipynb`
- `Modeling/2_RandomForest_Modeling.ipynb`
- `Modeling/3_HistGradientBoosting_Modeling.ipynb`
- `Modeling/4_LightGBM_Modeling.ipynb`
- `Modeling/5_XGBoost_Modeling.ipynb`
- `Modeling/6_CatBoost_Modeling.ipynb`
- `Modeling/8_XGBoost_grade_IntRate_Ablation.ipynb`
- `Modeling/9_Final_Model_Comparison.ipynb`

\section{Results}
Write 5-6 paragraphs. Include the model-comparison table, ablation table, calibration/policy discussion, and SHAP summary. Interpret results rather than just listing metrics. Explain why the strict validation-F1 winner is not necessarily the best business model, why the preferred no-grade/subgrade neutral XGBoost is defensible, and why capped policy thresholds are more realistic than best-F1 thresholds. Include whether the project could have done better and why some options were not pursued: full RAG/legal letter generation, larger ensembles, text fields, and production fairness evaluation require more governance and time.

\section{Analysis and Discussion}
Write 3-5 paragraphs. Provide insight into why the algorithms and features work for this dataset. Discuss nonlinear tabular structure, credit bureau variables, pricing/risk information in interest rate, temporal drift, class imbalance, and selection bias. State which lessons generalize to similar consumer-credit datasets: leakage audits, time-based validation, calibrated probabilities, and threshold policies. Compare with existing solutions and state-of-the-art approaches without overstating novelty.

\section{Conclusion}
Write exactly 3 paragraphs. Summarize the findings, recommend the preferred model and capped operating policy, and explain future work: complete RAG explanation generation, compliance-reviewed reason codes, fairness analysis, external validation, monitoring, and deployment-ready documentation.

\section{AI Prompts Used}
This section is mandatory if AI tools were used. Include the project prompts from `CreditRiskRAG/Docs/Prompts/Prompt.md` and any other known prompts. If the full prompt list is incomplete, write a TODO stating that all coding, debugging, explanation, and research prompts must be collected before submission.

\section{References}
Use numbered references. Include at least 6 references: the Kaggle dataset, 5-6 published papers, and 2-3 Kaggle implementation references if required by the rubric. Cite only references used in the text. Use a consistent format.

\appendix
\section{Appendix}
Include the public GitHub repository URL or a TODO if not public yet. Include installation and setup instructions, a brief artifact inventory, and links to notebooks. Keep extra plots here if needed.

\section{Statement of Contributions}
For the group members, describe each person's contribution. Use the repository README as the starting point, but verify with the team before final submission.

\end{document}

Rubric compliance requirements:
- Introduction: 2-3 paragraphs, clear motivation, target variable, and 2-3 dataset challenges.
- Proposed Method summary: one paragraph stating method and why it is better than alternatives tried.
- Related Work: 5-6 published papers, paragraph form, numbered citations.
- Related Implementations: 2-3 Kaggle implementations, URLs, paragraph form.
- Data Analysis: 5-6 paragraphs, 1-2 plots, derived-feature formulas.
- Methods: 3-5 paragraphs with formulas and algorithm explanations.
- Analysis/Discussion: 3-5 paragraphs with real insight, not only result repetition.
- Experimental Setup: 3-4 paragraphs, reproducible enough for another person to recreate, with notebook cell links or TODOs.
- Results: table, plots, and 5-6 interpretive paragraphs.
- Conclusion: exactly 3 paragraphs with recommendation and future work.
- References: at least 6, including papers and Kaggle-related references.
- Completion: formal academic style, no unsupported claims, no fabricated citations, no overclaiming that RAG/legal compliance is complete.

Important writing constraints:
- Do not describe the project as production-ready for lending decisions.
- Do not claim legal adverse-action compliance.
- Do not use rejected applications as labeled default outcomes.
- Do not report accuracy as the main metric.
- Do not fabricate Kaggle implementation URLs, paper claims, GitHub links, screenshots, or notebook cell anchors.
- Use TODO markers where evidence is missing.
- Keep the body concise enough for the 8-page limit.
```

