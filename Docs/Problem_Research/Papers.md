| Paper | Dataset Link Strength | What They Did |
|---|---:|---|
| **Classification based credit risk analysis: The case of Lending Club** — Gupta, Gulati, Chakrabarty, 2022 | **Direct match** | Explicitly says the dataset was obtained from Kaggle “All Lending Club Loan Data”; uses accepted loans, Logistic Regression, Random Forest, PD/LGD/EAD/Expected Loss, and CDS-style credit derivative pricing. Sources: Kaggle dataset page and arXiv paper. ([kaggle.com](https://www.kaggle.com/datasets/wordsforthewise/lending-club)) ([arxiv.org](https://arxiv.org/abs/2210.05136)) |
| **Explainable AI in Credit Risk Management** — Hadji Misheva et al., 2021 | **Direct match** | States the Lending Club dataset was obtained from Kaggle, with over 2.2M P2P loans; applies Logistic Regression, XGBoost, Random Forest, SVM, neural networks, plus LIME/SHAP explanations. ([arxiv.org](https://arxiv.org/abs/2103.00949)) |
| **Explainable AI for Interpretable Credit Scoring** — Demajo, Vella, Dingli, 2020 | Likely same/similar Lending Club dataset | Uses Lending Club and HELOC datasets; applies XGBoost plus global/local explainability methods for credit scoring. ([arxiv.org](https://arxiv.org/abs/2012.03749?utm_source=openai)) |
| **Credit Risk Meets Large Language Models: Building a Risk Indicator from Loan Descriptions in P2P Lending** — Sanz-Guerrero & Arroyo, 2024/2025 | Lending Club dataset, not clearly Kaggle in abstract | Uses Lending Club loan descriptions; fine-tunes BERT to produce a risk score and adds it to XGBoost for credit-risk classification. ([arxiv.org](https://arxiv.org/abs/2401.16458)) |
| **Enhancing ML Models Interpretability for Credit Scoring** — Schwartz, Wang, Fang, 2025 | Lending Club dataset, not clearly Kaggle in abstract | Uses SHAP for feature selection, XGBoost as black-box benchmark, and glass-box models like EBM and PLTR; reports comparable performance using only 10 features. ([arxiv.org](https://arxiv.org/abs/2509.11389)) |
| **An Optimised Greedy-Weighted Ensemble Framework for Financial Loan Default Prediction** — Nortey et al., 2026 | Lending Club dataset, not clearly Kaggle in abstract | Uses ensemble learning, PSO tuning, greedy model weighting, SMOTE/cost-sensitive learning; reports BlendNet AUC of 0.80 and default recall of 0.81. ([arxiv.org](https://arxiv.org/abs/2603.18927)) |
| **Label-Free Detection of Governance Evidence Degradation in Risk Decision Systems** — Solozobov, 2026 | Lending Club credit-scoring dataset, not clearly Kaggle in abstract | Uses Lending Club data for drift/governance monitoring in credit scoring; focuses on label-free degradation detection. ([arxiv.org](https://arxiv.org/abs/2604.17836?utm_source=openai)) |





| Paper | Dataset Match | What It Is About | Problem Addressed | Techniques / Tools | Main Results |
|---|---|---|---|---|---|
| **Classification based credit risk analysis: The case of Lending Club** — Gupta, Gulati, Chakrabarty, 2022 | **Direct Kaggle Lending Club match** | Credit-risk analysis using Lending Club loan data. | Predict borrower default and estimate credit risk for loan portfolios. | Exploratory Data Analysis, Logistic Regression, Random Forest, Probability of Default, credit derivative / CDS-style hedging idea. | Shows ML classifiers can estimate default probability and uses that probability to design a credit-risk hedge. Source: [arXiv](https://arxiv.org/abs/2210.05136) |
| **Explainable AI in Credit Risk Management** — Hadji Misheva et al., 2021 | **Direct / very close Lending Club open dataset** | Applying explainable AI to credit scoring models. | ML models can predict credit risk well, but banks need explanations for decisions. | LIME, SHAP, machine-learning credit scoring models, local and global explanations. | Demonstrates how SHAP and LIME explain borrower-level and global model behavior for Lending Club credit-risk predictions. Source: [arXiv](https://arxiv.org/abs/2103.00949) |
| **Explainable AI for Interpretable Credit Scoring** — Demajo, Vella, Dingli, 2020 | Lending Club dataset | Builds an accurate but interpretable credit scoring framework. | Credit scoring models face noisy, imbalanced data and legal/regulatory explainability requirements. | XGBoost, global explanations, local feature explanations, local instance-based explanations. | Reports strong classification performance on Lending Club and HELOC datasets, with a “360-degree” explanation framework. Source: [arXiv](https://arxiv.org/abs/2012.03749) |
| **Credit Risk Meets Large Language Models: Building a Risk Indicator from Loan Descriptions in P2P Lending** — Sanz-Guerrero & Arroyo, 2024/2025 | Lending Club dataset | Uses borrower loan descriptions as text signals for credit risk. | Traditional credit models may miss information contained in borrower-written descriptions. | BERT, fine-tuning, text classification, BERT-generated risk score, XGBoost. | Adding the BERT risk score improves balanced accuracy and AUC; text features provide useful extra risk information. Source: [arXiv](https://arxiv.org/abs/2401.16458) |
| **Enhancing ML Models Interpretability for Credit Scoring** — Schwartz, Wang, Fang, 2025 | Lending Club dataset | Uses explainability to build simpler, more transparent credit models. | Black-box models perform well but may be too opaque for regulated credit scoring. | SHAP, XGBoost, Explainable Boosting Machine, Penalized Logistic Tree Regression, feature interaction analysis. | Achieves performance comparable to black-box XGBoost using only 10 features, an 88.5% feature reduction. Source: [arXiv](https://arxiv.org/abs/2509.11389) |
| **An Optimised Greedy-Weighted Ensemble Framework for Financial Loan Default Prediction** — Nortey et al., 2026 | Lending Club dataset | Builds an optimized ensemble model for loan default prediction. | Loan-default data has nonlinear patterns, class imbalance, and changing borrower behavior. | Particle Swarm Optimization, greedy weighted ensemble, stacked ensemble, neural-network meta-learner, Recursive Feature Elimination. | Best model, BlendNet, achieved AUC 0.80, macro F1 0.73, and default recall 0.81. Source: [arXiv](https://arxiv.org/abs/2603.18927) |
| **Label-Free Detection of Governance Evidence Degradation in Risk Decision Systems** — Solozobov, 2026 | Lending Club credit-scoring dataset | Monitors credit-risk systems when labels are delayed or unavailable. | In credit scoring, true outcomes may arrive months later, so model degradation can go unnoticed. | Drift monitoring, score distribution monitoring, feature PSI, prediction entropy, confidence distribution, governance alerts. | On 1.37M Lending Club loans, proxy metrics detected covariate degradation but could not detect pure concept drift without labels. Source: [arXiv](https://arxiv.org/abs/2604.17836) |


## Detailed Reading Summary

This section summarizes the Lending Club / credit-risk papers using a consistent reading structure: problem, methods, tools, solution architecture, and results.

| Paper | Problem / Research Goal | Methods | Tools / Techniques | Solution Architecture | Results / Findings | Relevance to CreditRiskRAG |
|---|---|---|---|---|---|---|
| [Classification based credit risk analysis: The case of Lending Club](https://arxiv.org/abs/2210.05136) — Gupta, Gulati, Chakrabarty, 2022 | Predict borrower default risk using Lending Club loan data, then translate default probability into a portfolio credit-risk measure. | Exploratory data analysis, binary classification, probability of default estimation, credit derivative pricing idea. | Logistic Regression, Random Forest, PD, LGD, EAD, Expected Loss, CDS-style hedging. | Data preprocessing and EDA -> train classification models -> estimate probability of default -> calculate expected loss -> use default probability to design a credit-risk hedge. | Shows that standard ML classifiers can estimate credit risk from Lending Club features and connect predictions to financial risk management. The paper reports test-set performance using classification metrics. | Useful baseline for our project because it uses the same Kaggle Lending Club dataset and frames the core problem as default prediction plus credit-risk interpretation. |
| [Explainable AI in Credit Risk Management](https://arxiv.org/abs/2103.00949) — Hadji Misheva et al., 2021 | Credit-risk ML models can be accurate but difficult to explain, which is a problem in regulated financial decision-making. | Train credit-scoring models and apply post-hoc explainability to understand predictions locally and globally. | LIME, SHAP, Logistic Regression, Random Forest, XGBoost, SVM, neural networks, model-agnostic explainability. | Lending Club data -> ML credit scoring models -> model predictions -> LIME for local explanations -> SHAP for local and global feature importance -> compare explanation outputs. | Demonstrates that LIME and SHAP can explain individual borrower decisions and global model behavior, while also documenting implementation challenges. | Important for our project because RAG explanations should not only return predictions, but also explain why borrower features affect risk. |
| [Explainable AI for Interpretable Credit Scoring](https://arxiv.org/abs/2012.03749) — Demajo, Vella, Dingli, 2020 | Credit scoring needs high predictive performance while satisfying interpretability expectations from regulations and business users. | Use a strong black-box classifier, then add a multi-perspective explanation framework. | XGBoost, global explanations, local feature-based explanations, local instance-based explanations, human-grounded and application-grounded evaluation. | Lending Club / HELOC data -> train XGBoost classifier -> generate global explanations for model behavior -> generate local explanations for individual applications -> evaluate explanation quality. | Achieves strong classification performance with XGBoost and proposes a 360-degree explanation framework that is simple, consistent, and understandable. | Useful for designing explanation layers in CreditRiskRAG: global feature summaries, individual borrower explanations, and user-facing reasoning can be separated. |
| [Credit Risk Meets Large Language Models: Building a Risk Indicator from Loan Descriptions in P2P Lending](https://arxiv.org/abs/2401.16458) — Sanz-Guerrero & Arroyo, 2024/2025 | Traditional tabular credit models may miss useful risk information contained in borrower-written loan descriptions. | Fine-tune a language model on borrower descriptions, convert text into a risk score, and add that score to a tabular credit model. | BERT, text classification, fine-tuning, LLM-derived risk score, XGBoost, balanced accuracy, AUC. | Borrower loan description text -> fine-tuned BERT predicts default vs non-default risk -> BERT score becomes a new feature -> XGBoost combines text score with traditional loan variables -> evaluate predictive lift. | Adding the BERT-generated risk score improves balanced accuracy and AUC. The paper finds text contains borrower-specific, purpose-specific, and linguistic risk signals. | Highly relevant if our RAG system uses text fields, borrower explanations, or retrieved context to enrich tabular risk prediction. |
| [Enhancing ML Models Interpretability for Credit Scoring](https://arxiv.org/abs/2509.11389) — Schwartz, Wang, Fang, 2025 | Post-hoc explanations of black-box models may still be too complex for regulated credit-risk models. | Use SHAP on a black-box model to select important features, then train simpler glass-box models. | SHAP, XGBoost, Explainable Boosting Machine, Penalized Logistic Tree Regression, feature interaction analysis, correlation checks, expert review. | Train XGBoost benchmark -> use SHAP to rank/select features -> reduce feature set to 10 features -> train glass-box models -> refine with interactions, correlations, and expert input. | Achieves performance comparable to XGBoost while using only 10 features, an 88.5% feature reduction. Shows that interpretability can be improved without losing much predictive power. | Useful for feature-selection decisions in our project and for justifying simpler models or explainable model components alongside RAG. |
| [An Optimised Greedy-Weighted Ensemble Framework for Financial Loan Default Prediction](https://arxiv.org/abs/2603.18927) — Nortey et al., 2026 | Loan-default prediction is hard because of nonlinear relationships, class imbalance, and changing borrower behavior. | Optimize multiple classifiers, combine them with greedy performance-based weights, and compare against individual models. | Particle Swarm Optimization, greedy weighted ensemble, stacked ensemble, neural-network meta-learner, Recursive Feature Elimination, calibration analysis. | Lending Club data -> feature preprocessing and selection -> optimize base classifiers with PSO -> combine predictions through greedy weighted blending -> train stacked ensemble / neural meta-learner -> evaluate classification and calibration. | BlendNet achieved AUC 0.80, macro F1 0.73, and default recall 0.81. Important predictors included revolving utilization, annual income, and debt-to-income ratio. | Useful if we want a stronger predictive backend, but its complexity may be harder to explain than simpler models. |
| [Label-Free Detection of Governance Evidence Degradation in Risk Decision Systems](https://arxiv.org/abs/2604.17836) — Solozobov, 2026 | Credit models can degrade before true repayment/default labels arrive, making governance and monitoring difficult. | Monitor model and data behavior without labels using multiple proxy signals and governance-calibrated thresholds. | Governance Drift Toolkit, feature PSI, score PSI, prediction entropy, confidence distribution, composite monitoring score. | Deployed risk model -> collect unlabeled production signals -> monitor score distribution, feature drift, entropy, and confidence -> combine proxies into governance alerts -> escalate operational response when thresholds trigger. | On 1.37M Lending Club loans over 11 years, proxy metrics detected covariate degradation but could not detect pure concept drift in P(Y\|X). Composite severity increased as more monitors triggered. | Useful for project discussion about model monitoring after deployment. It also clarifies what cannot be solved without labels. |


## Key Patterns Across the Papers

| Theme | What the Papers Show | Project Takeaway |
|---|---|---|
| Default prediction is the core task | Most papers model Lending Club risk as a binary classification problem: default vs non-default. | Our target definition and leakage control must be explicit before modeling. |
| Tree-based models are common strong baselines | Random Forest, XGBoost, Extra Trees, and Gradient Boosting appear repeatedly. | Use Logistic Regression as an interpretable baseline and tree/boosting models as stronger predictors. |
| Explainability is central in credit risk | SHAP, LIME, glass-box models, and local/global explanation frameworks are repeatedly used. | CreditRiskRAG should explain both feature-level risk drivers and document/context-based reasoning. |
| Text can add signal | The BERT paper shows borrower descriptions contain useful risk information. | If text fields are available and safe to use, RAG/LLM components can add value beyond tabular features. |
| Simpler models may be preferred in regulated settings | The 2025 interpretability paper reduces the model to 10 features while keeping comparable performance. | A smaller, explainable model may be easier to defend in the final project than a highly complex ensemble. |
| Monitoring matters after deployment | Drift/governance work shows that models can degrade before labels arrive. | Include a deployment-monitoring section: feature drift, score drift, confidence drift, and limits of label-free monitoring. |


## Suggested Reading Order

1. Start with Gupta et al. 2022 to understand the basic Lending Club default-prediction workflow.
2. Read Hadji Misheva et al. 2021 and Demajo et al. 2020 for explainability design.
3. Read Schwartz et al. 2025 for feature selection and glass-box modeling ideas.
4. Read Sanz-Guerrero and Arroyo 2024/2025 if using text, RAG, or LLM-generated risk signals.
5. Read Nortey et al. 2026 only if considering advanced ensembles.
6. Read Solozobov 2026 for deployment monitoring and governance discussion.


## Note on Expected Loss Formula and Paper Trustworthiness

### Why Gupta et al. 2022 uses `EL = EAD x LGD x PD`

The first paper predicts borrower default risk because default prediction gives the model's estimate of **PD**, or probability of default. In credit-risk management, predicting default is only one part of the business question. A lender also needs to estimate how much money is at risk and how much would actually be lost if the borrower defaults.

The expected loss formula connects the machine-learning prediction to financial risk:

| Term | Meaning | Example in Lending Club Context |
|---|---|---|
| `PD` | Probability of Default: the chance that the borrower defaults. | A model predicts a borrower has a 12% default probability. |
| `EAD` | Exposure at Default: the amount outstanding when default happens. | The borrower still owes $10,000 when default occurs. |
| `LGD` | Loss Given Default: the percentage of exposure the lender does not recover after collections or recoveries. | If the lender expects to lose 60% after recovery, LGD = 0.60. |
| `EL` | Expected Loss: the average expected dollar loss before the default happens. | `0.12 x 10000 x 0.60 = $720`. |

So the paper is not only asking, "Will this borrower default?" It is also asking, "If this borrower defaults, how much money should the lender expect to lose?" That is why `PD`, `EAD`, and `LGD` appear together.

For our project, this means a classification model can support two different outputs:

| Output | What It Answers |
|---|---|
| Default prediction | Is the loan likely to become bad? |
| Expected loss estimate | How much financial loss should we expect from that risk? |

### Are these papers verifiable and trustworthy?

They are **verifiable enough for a class project and literature review**, but they should not all be treated as equally authoritative.

| Check | What I Found | Trust Level |
|---|---|---|
| Public access | The papers are publicly accessible through arXiv or DOI links. | Good for verification. |
| Dataset traceability | Several papers explicitly use Lending Club data; Gupta et al. 2022 directly says it uses the Kaggle "All Lending Club Loan Data" dataset. | Good, but still verify exact preprocessing. |
| Reproducibility | Most papers describe methods, but not all provide full code, exact splits, or full preprocessing decisions. | Moderate. |
| Peer review | arXiv papers are public preprints and are not automatically peer-reviewed. Demajo et al. 2020 has a related conference DOI. | Mixed. |
| Method credibility | Logistic Regression, Random Forest, XGBoost, SHAP, LIME, BERT, and expected-loss modeling are established techniques. | Good. |
| Risk of overclaiming | Some papers use complex ensembles or limited evaluation details. Claims should be cited carefully, not copied as final truth. | Requires caution. |

Recommended way to cite them:

| Use Case | Best Practice |
|---|---|
| Background / motivation | Safe to cite these papers as examples of Lending Club credit-risk modeling. |
| Exact performance numbers | Only cite numbers if the paper clearly reports the dataset split, target definition, and evaluation metric. |
| Project design | Use them to justify model choices such as Logistic Regression, Random Forest, XGBoost, SHAP, and RAG/text-based explanations. |
| Strong claims | Avoid saying "this proves." Say "this paper reports" or "this paper demonstrates on Lending Club data." |

Bottom line: these papers are useful and checkable, but for a rigorous final project we should treat them as supporting literature, then validate our own results with clear preprocessing, leakage control, train/test split, and evaluation metrics.
