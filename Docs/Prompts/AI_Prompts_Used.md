# AI Prompts Used

This appendix documents the AI prompts and prompt templates recoverable from the repository for coding, debugging, explanations, research, report drafting, and RAG evaluation. The project requirement says to include all prompts used with Claude or other AI tools. The repository contains the prompts below as saved artifacts or executable prompt templates.

Important limitation: if team members used Claude, ChatGPT, Gemini, or another assistant in a separate chat and did not save the exact prompt text into the repository, that exact text cannot be recovered from the project files. Those prompts should be added below from each team member's chat history before final submission.

## Prompt Inventory

| ID | Category | AI tool/model | Prompt type | Source |
|---|---|---|---|---|
| P1 | Research, EDA, coding | AI assistant, exact tool not recorded | Full notebook-generation prompt | `CreditRiskRAG/Docs/Prompts/Prompt.md` |
| P2 | Report drafting, explanation | AI writing assistant, exact tool not recorded | Full LaTeX report-drafting prompt | `CreditRiskRAG/Docs/Final Project Report /Final_Report_LaTeX_Drafting_Prompt.md` |
| P3 | RAG research/retrieval | Local retriever query strings | Fixed legal retrieval queries | `CreditRiskRAG/generation/generate_letter.py` |
| P4 | Generation | Gemini `gemini-3-flash-preview` | RAG adverse-action-style writer prompt | `CreditRiskRAG/generation/generate_letter.py` |
| P5 | Generation/control experiment | Gemini `gemini-3-flash-preview` | No-RAG adverse-action-style writer prompt | `CreditRiskRAG/generation/generate_norag.py` |
| P6 | Evaluation/debugging | Gemini `gemini-3.1-flash-lite` | Compliance judge rubric prompt | `CreditRiskRAG/generation/judge.py` |
| P7 | Coding/debugging/explanations | Claude/ChatGPT/other, if used | Team prompt logs not yet added | TODO: add exact prompts from team chat histories |

## P1. EDA Notebook Prompt

Category: research, coding, explanation.

Full source: `CreditRiskRAG/Docs/Prompts/Prompt.md`

Summary: The prompt asks an AI assistant to act as a Principal/Staff Data Scientist with 15+ years of experience at Google and create a rigorous, executive-quality EDA notebook for the Kaggle LendingClub dataset. It requests business context, dataset inventory, memory-safe loading, cleaning and type conversion, target construction, missing-value analysis, univariate and bivariate EDA, temporal trends, credit-risk relationships, leakage detection, accepted-vs-rejected analysis, fairness/responsible AI considerations, and modeling-readiness recommendations.

Verbatim prompt: retained in the source file above.

## P2. Final Report LaTeX Drafting Prompt

Category: report drafting, explanation, research synthesis.

Full source: `CreditRiskRAG/Docs/Final Project Report /Final_Report_LaTeX_Drafting_Prompt.md`

Summary: The prompt asks an AI writing assistant to act as a senior machine-learning researcher and scientific editor and draft a polished LaTeX final report for a Northeastern CS 6140 final project. It provides the project title, authors, dataset, completed scope, target definition, preferred model, validation/test metrics, grade/subgrade and interest-rate ablation results, economic policy example, SHAP interpretation facts, required source artifacts, required figures/tables, report structure, rubric constraints, and warnings against unsupported or production/legal-compliance claims.

Verbatim prompt: retained in the source file above.

## P3. Legal Retrieval Queries

Category: research, retrieval.

Source: `CreditRiskRAG/generation/generate_letter.py`

```text
what must an adverse action notice contain
statement of specific principal reasons requirement
credit score disclosure adverse action
```

Purpose: retrieve governing notice requirements for the RAG adverse-action-style writer. These queries retrieve the legal notice envelope rather than per-feature legal text.

## P4. RAG Adverse-Action-Style Writer Prompt

Category: generation, explanation.

Source: `CreditRiskRAG/generation/generate_letter.py`

Model: `gemini-3-flash-preview`

Dynamic inputs:

- retrieved legal chunks from the hybrid retriever,
- SHAP-selected applicant principal reasons.

```text
You are a compliance officer drafting an adverse-action notice for a declined credit application, following US law.

GOVERNING LAW (ground your letter strictly in these authorities):
{law}

APPLICANT'S PRINCIPAL REASONS FOR THE DECISION (state these; do not invent others):
{reason_lines}

Write a professional adverse-action letter that:
- states the credit decision is adverse (application declined),
- gives the specific, principal reasons above (do not add reasons or cite numeric values),
- includes the ECOA notice language and the FCRA disclosures the governing law requires,
- is addressed generically ("Dear Applicant"), signed generically.
Return only the letter text.
```

## P5. No-RAG Control Writer Prompt

Category: generation, debugging/control experiment.

Source: `CreditRiskRAG/generation/generate_norag.py`

Model: `gemini-3-flash-preview`

Dynamic input:

- SHAP-selected applicant principal reasons.

```text
You are a compliance officer drafting an adverse-action notice for a declined credit application, following US law.

APPLICANT'S PRINCIPAL REASONS FOR THE DECISION (state these; do not invent others):
{reason_lines}

Write a professional adverse-action letter that:
- states the credit decision is adverse (application declined),
- gives the specific, principal reasons above (do not add reasons or cite numeric values),
- includes the ECOA notice language and the FCRA disclosures required for adverse-action notices,
- is addressed generically ("Dear Applicant"), signed generically.
Return only the letter text.
```

## P6. Compliance Judge Rubric Prompt

Category: evaluation, debugging, explanation.

Source: `CreditRiskRAG/generation/judge.py`

Model: `gemini-3.1-flash-lite`

Dynamic inputs:

- applicant principal reasons,
- generated letter text.

```text
You are a compliance auditor grading the STATUTORY ACCURACY of a credit adverse-action letter.
Do not reward a disclosure merely for being present - grade whether its legal specifics are CORRECT.
Score each item 1 (correct) or 0 (missing, vague, or wrong). Judge ONLY the letter text.
Score the 5 items INDEPENDENTLY: a single defect should cost only the item(s) it specifically
violates, not multiple items at once.

Return ONLY a JSON object, no other text, exactly this shape:
{"reasons_correct": 0 or 1, "fcra_window_correct": 0 or 1, "real_agency_named": 0 or 1, "ecoa_classes_correct": 0 or 1, "no_legal_errors": 0 or 1, "note": "one short sentence citing the deciding detail"}

Grade strictly on these specifics:
- reasons_correct: states exactly the applicant's principal reasons listed below, no more, no fewer.
- fcra_window_correct: states the applicant's right to a free consumer report within exactly 60 days. Wrong number (30, 90, etc.) or no window = 0.
- real_agency_named: names a REAL federal enforcement agency (e.g. Consumer Financial Protection Bureau, FTC) as the agency administering compliance. A bracketed blank placeholder like [Agency Name] with no real agency = 0. An invented/wrong agency = 0.
- ecoa_classes_correct: the ECOA anti-discrimination notice lists the correct protected bases (race, color, religion, national origin, sex, marital status, age; receipt of public assistance; good-faith exercise of Consumer Credit Protection Act rights). Materially wrong or missing list = 0.
- no_legal_errors: contains a legal or factual error OTHER than the ones already scored above (e.g. a fabricated statute/citation, an invented numeric deadline, or an incorrect claim not covered by the four items above). Do NOT mark this 0 just because fcra_window_correct, real_agency_named, ecoa_classes_correct, or reasons_correct already failed for their own reasons - only mark it 0 for a DIFFERENT error not already captured by those four items. If the only problems in the letter are already covered by the other four items, no_legal_errors = 1.

The applicant's principal reasons were:
<<REASONS>>

LETTER:
<<LETTER>>
```

## P7. Team Chat Prompts To Add Before Final Submission

Category: coding, debugging, explanations, research.

Status: not recoverable from repository files unless team members paste them from their AI chat histories.

Use this format for each additional prompt:

```text
Tool:
Date:
Category: coding | debugging | explanation | research | report drafting
Prompt:
<paste exact prompt here>
Output used in project:
<briefly describe file, notebook, report text, code, or decision affected>
```

Suggested team collection checklist:

- prompts used to write or revise notebooks,
- prompts used to debug Python errors,
- prompts used to interpret model metrics,
- prompts used for SHAP explanation wording,
- prompts used for paper/background research,
- prompts used for report writing,
- prompts used for slide/video scripting,
- prompts used for RAG implementation or evaluation.
