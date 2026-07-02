Act as a Principal / Staff Data Scientist with 15+ years of experience at Google, specializing in credit risk, marketplace lending, large-scale tabular data, statistical inference, and production-grade ML readiness.

I am working with the Kaggle dataset “All Lending Club loan data”:
https://www.kaggle.com/datasets/wordsforthewise/lending-club

Your task is to create a rigorous, executive-quality Exploratory Data Analysis notebook for this dataset.

The dataset may contain separate files for accepted loans and rejected loans. Treat them as related but analytically distinct datasets unless there is a clear, defensible reason to combine them.

Primary objectives:

1. Understand the business context
   - Explain what LendingClub does.
   - Define the key analytical questions:
     - What types of loans are issued?
     - What borrower characteristics are associated with loan outcomes?
     - What factors correlate with default, charge-off, late payment, or full repayment?
     - What distinguishes accepted from rejected applications?
     - What data leakage risks exist for credit-risk modeling?

2. Perform dataset inventory
   - Load all available files.
   - Report shape, column names, data types, memory usage, duplicate rows, duplicate IDs, and file-level differences.
   - Identify primary keys or near-primary keys.
   - Detect columns that appear only in accepted loans or only in rejected loans.
   - Provide a concise data dictionary summary when possible.

3. Handle large-data constraints professionally
   - Use efficient loading strategies.
   - Prefer pandas with dtype optimization, or use Polars / DuckDB / PyArrow when appropriate.
   - Avoid loading unnecessary columns when profiling.
   - Include memory-safe code patterns.
   - If sampling is used, explain the sampling strategy and validate that the sample is representative.

4. Clean and standardize variables
   - Parse dates such as issue date, earliest credit line, last payment date, and credit pull dates.
   - Convert percentage fields such as interest rate and revolving utilization into numeric values.
   - Normalize employment length, loan term, grade, sub-grade, home ownership, verification status, loan purpose, state, and loan status.
   - Detect impossible, suspicious, or inconsistent values.
   - Separate raw fields from cleaned fields.

5. Define target variables carefully
   - For accepted loans, analyze `loan_status`.
   - Create defensible target definitions, for example:
     - Fully Paid vs Charged Off
     - Good loans vs bad loans
     - Excluding Current loans when the final outcome is unknown
   - Explain why certain statuses should or should not be included.
   - Identify censoring issues caused by loans that are still current.
   - Avoid leakage from post-origination fields.

6. Perform missing-value analysis
   - Calculate missingness by column.
   - Visualize missingness patterns.
   - Identify columns with structural missingness versus random missingness.
   - Compare missingness across loan issue years, loan status, grade, purpose, and verification status.
   - Recommend whether each high-missingness feature should be dropped, imputed, flagged, or retained.

7. Analyze distributions
   - Produce univariate analysis for key numerical variables:
     - loan amount
     - funded amount
     - installment
     - annual income
     - debt-to-income ratio
     - interest rate
     - revolving balance
     - revolving utilization
     - open accounts
     - delinquencies
     - inquiries
     - public records
     - FICO-related variables, if available
   - Produce categorical analysis for:
     - loan status
     - grade and sub-grade
     - term
     - home ownership
     - verification status
     - purpose
     - state
     - employment length
     - application type
   - Use appropriate visualizations and summary statistics.
   - Highlight skewness, long tails, caps, outliers, and suspicious values.

8. Analyze time trends
   - Analyze loan origination volume over time.
   - Show trends by issue year and month.
   - Track changes in:
     - loan amount
     - interest rate
     - grade mix
     - borrower income
     - DTI
     - default / charge-off rates
     - loan purpose
     - term distribution
   - Identify regime shifts, policy changes, or macro-period effects visible in the data.
   - Recommend time-based validation strategies for future modeling.

9. Analyze credit risk relationships
   - Compare good vs bad loan outcomes across:
     - grade
     - sub-grade
     - interest rate
     - term
     - FICO bands
     - DTI bands
     - income bands
     - loan amount bands
     - purpose
     - verification status
     - home ownership
     - state
     - employment length
   - Use default-rate tables, lift charts, and grouped summaries.
   - Include confidence intervals or minimum sample-size thresholds where useful.
   - Distinguish correlation from causation.

10. Detect data leakage
   - Identify fields unavailable at loan origination.
   - Flag post-origination variables such as payment history, recoveries, collection amounts, last payment date, hardship flags, settlement fields, and similar variables.
   - Create three feature groups:
     - Safe at application / origination
     - Potentially risky or ambiguous
     - Clear leakage / post-outcome
   - Explain why leakage matters for credit-risk modeling.

11. Analyze accepted vs rejected loans
   - If rejected-loan data is available, compare accepted and rejected applications.
   - Standardize comparable fields across files.
   - Analyze differences in:
     - requested amount
     - risk score, if available
     - DTI
     - employment length
     - state
     - application date
     - loan purpose
   - Be explicit about columns that cannot be compared.
   - Discuss selection bias: accepted-loan outcomes are observed only after LendingClub approval.

12. Fairness and responsible AI considerations
   - Do not infer protected attributes directly.
   - Discuss potential proxy variables such as ZIP code, state, income, employment length, and home ownership.
   - Identify fairness risks in credit modeling.
   - Recommend responsible evaluation slices and governance checks.
   - Avoid making discriminatory or legally unsupported conclusions.

13. Prepare modeling-readiness recommendations
   - Recommend target definition.
   - Recommend train / validation / test split strategy, preferably time-based.
   - Recommend features to keep, transform, drop, or investigate.
   - Recommend encoding strategies for categorical variables.
   - Recommend transformations for skewed numeric variables.
   - Recommend baseline models only at a high level; do not train a model unless explicitly requested.
   - Produce a final feature-readiness table.

14. Deliverables
   Create a complete EDA notebook with:
   - Clear markdown explanations.
   - Production-quality Python code.
   - Reusable helper functions.
   - Tables and visualizations.
   - Executive summary at the top.
   - Technical appendix at the bottom.
   - Clear assumptions and caveats.
   - Actionable recommendations.

Required output structure:

A. Executive Summary
   - 5 to 10 key findings.
   - Major data quality issues.
   - Major leakage risks.
   - Recommended modeling target.
   - Recommended next steps.

B. Dataset Overview
   - Files loaded.
   - Shapes.
   - Schema.
   - Memory usage.
   - Key columns.
   - Data-quality summary.

C. Cleaning and Type Conversion
   - Parsing logic.
   - Before / after schema.
   - Known caveats.

D. Missingness Analysis
   - Missingness tables.
   - Missingness charts.
   - Structural missingness interpretation.

E. Univariate EDA
   - Numeric features.
   - Categorical features.
   - Outlier discussion.

F. Bivariate EDA
   - Numeric-numeric relationships.
   - Numeric-categorical relationships.
   - Categorical-categorical relationships.
   - Default-rate comparisons across key variables.
   - Statistical and practical significance discussion.

G. Multivariate EDA
   - Risk segmentation heatmaps.
   - Interaction analysis.
   - Confounding analysis.
   - Feature redundancy and multicollinearity checks.
   - Segment-level default-rate interpretation.
H. Temporal EDA
   - Origination trends.
   - Portfolio composition over time.
   - Outcome trends over time.

I. Loan Outcome EDA
   - Loan status distribution.
   - Good / bad target construction.
   - Default or charge-off rate by major borrower and loan attributes.

J. Accepted vs Rejected Analysis
   - Comparable fields.
   - Distributional differences.
   - Selection-bias discussion.

K. Leakage Audit
   - Safe features.
   - Ambiguous features.
   - Leakage features.
   - Recommended exclusion list.

L. Modeling Readiness
   - Final target recommendation.
   - Candidate feature groups.
   - Split strategy.
   - Preprocessing strategy.
   - Risks and assumptions.

M. Final Recommendations
   - Business insights.
   - Data improvements.
   - Modeling roadmap.

Coding requirements:

- Use Python.
- Use pandas, numpy, matplotlib, and seaborn unless a faster library is needed.
- Use clear function names and comments.
- Avoid hardcoding paths except for a single configurable `DATA_DIR`.
- Include robust error handling for missing files or missing columns.
- Make visualizations readable and not overly cluttered.
- For high-cardinality categorical variables, show top N categories and group the rest as “Other.”
- For very large files, include an option to run on a sample first and then scale to the full dataset.

Important analytical constraints:

- Be skeptical of every variable.
- Do not assume that all columns are available at loan origination.
- Do not use post-outcome variables when discussing predictive modeling.
- Separate EDA for explanation from EDA for model design.
- Clearly distinguish observed patterns, hypotheses, and recommendations.
- Use precise credit-risk terminology.
- Prioritize business usefulness over decorative charts.

Now generate the full EDA notebook plan and Python code step by step.