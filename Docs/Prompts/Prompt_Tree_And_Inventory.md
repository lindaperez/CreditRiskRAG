# Prompt Tree And Inventory

This file collects the important AI prompts and prompt templates used or documented in the CreditRiskRAG project. It is a curated inventory, not a dump of every generated letter, API response, notebook markdown cell, or project artifact.

## Prompt Tree

```text
CreditRiskRAG project prompts
├── 1. Project analysis and notebook creation
│   └── EDA notebook prompt
│       └── Source: CreditRiskRAG/Docs/Prompts/Prompt.md
│
├── 2. Final report drafting
│   └── LaTeX final-report drafting prompt
│       └── Source: CreditRiskRAG/Docs/Final Project Report /Final_Report_LaTeX_Drafting_Prompt.md
│
└── 3. RAG adverse-action-style generation workflow
    ├── Retrieval queries
    │   └── Source: CreditRiskRAG/generation/generate_letter.py
    ├── RAG writer prompt
    │   └── Source: CreditRiskRAG/generation/generate_letter.py
    ├── No-RAG control writer prompt
    │   └── Source: CreditRiskRAG/generation/generate_norag.py
    └── LLM compliance judge rubric
        └── Source: CreditRiskRAG/generation/judge.py
```

## Length Estimate

The core prompt files/templates currently total about 695 source lines:

| Prompt source | Lines | Importance |
|---|---:|---|
| `CreditRiskRAG/Docs/Prompts/Prompt.md` | 269 | High |
| `CreditRiskRAG/Docs/Final Project Report /Final_Report_LaTeX_Drafting_Prompt.md` | 224 | High |
| `CreditRiskRAG/generation/generate_letter.py` | 61 | High |
| `CreditRiskRAG/generation/generate_norag.py` | 63 | High |
| `CreditRiskRAG/generation/judge.py` | 78 | High |
| **Total** | **695** |  |

An ideal prompt appendix for a final report should not include all 695 lines. A compact version should be about 1-2 pages, roughly 80-150 lines, containing:

- prompt purpose,
- model used,
- static prompt text or summarized template,
- dynamic variables injected into the prompt,
- source file path,
- whether the prompt was used for generation, control, or evaluation.

## Important Prompts

### 1. EDA Notebook Prompt

Source: `CreditRiskRAG/Docs/Prompts/Prompt.md`

Purpose: Ask an AI assistant to create a rigorous exploratory data analysis notebook for the LendingClub accepted/rejected loan dataset.

Scope:

- business context,
- dataset inventory,
- memory-safe loading,
- cleaning/type conversion,
- target definition,
- missingness,
- univariate/bivariate/multivariate EDA,
- temporal analysis,
- leakage audit,
- accepted-vs-rejected comparison,
- fairness/responsible AI risks,
- modeling-readiness recommendations.

Recommended appendix treatment: summarize the prompt and cite the full source file. It is long and covers the whole EDA workflow, so copying it verbatim into the report would likely be too much.

### 2. Final Report LaTeX Drafting Prompt

Source: `CreditRiskRAG/Docs/Final Project Report /Final_Report_LaTeX_Drafting_Prompt.md`

Purpose: Ask an AI writing assistant to draft the final report as a scientific LaTeX paper using project metrics, artifacts, and the class rubric.

Scope:

- project title/authors,
- completed modeling scope,
- target definition,
- preferred model,
- validation/test metrics,
- ablation results,
- economic policy example,
- SHAP interpretation summary,
- source artifacts to consult,
- required figures/tables,
- required report structure,
- rubric compliance constraints,
- warnings against unsupported claims.

Recommended appendix treatment: summarize with the source path and include only the major constraints, because this prompt is also long.

### 3. Retrieval Queries

Source: `CreditRiskRAG/generation/generate_letter.py`

Model: retrieval step, not a generation model.

Purpose: Retrieve governing adverse-action notice requirements before letter generation.

```text
what must an adverse action notice contain
statement of specific principal reasons requirement
credit score disclosure adverse action
```

These queries retrieve the legal notice envelope rather than law for each individual SHAP reason.

### 4. RAG Writer Prompt

Source: `CreditRiskRAG/generation/generate_letter.py`

Model: `gemini-3-flash-preview`

Purpose: Generate an adverse-action-style letter using retrieved legal text plus SHAP-selected principal reasons.

Dynamic inputs:

- retrieved law chunks,
- selected applicant reasons.

Template:

```text
You are a compliance officer drafting an adverse-action notice for a declined credit application, following US law.

GOVERNING LAW (ground your letter strictly in these authorities):
{retrieved_law_chunks}

APPLICANT'S PRINCIPAL REASONS FOR THE DECISION (state these; do not invent others):
{reason_lines}

Write a professional adverse-action letter that:
- states the credit decision is adverse (application declined),
- gives the specific, principal reasons above (do not add reasons or cite numeric values),
- includes the ECOA notice language and the FCRA disclosures the governing law requires,
- is addressed generically ("Dear Applicant"), signed generically.
Return only the letter text.
```

### 5. No-RAG Control Writer Prompt

Source: `CreditRiskRAG/generation/generate_norag.py`

Model: `gemini-3-flash-preview`

Purpose: Generate comparison letters without retrieved legal context, isolating the effect of retrieval.

Dynamic inputs:

- selected applicant reasons.

Template:

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

### 6. Compliance Judge Rubric Prompt

Source: `CreditRiskRAG/generation/judge.py`

Model: `gemini-3.1-flash-lite`

Purpose: Blindly score generated RAG and no-RAG letters for statutory accuracy.

Dynamic inputs:

- applicant principal reasons,
- generated letter text.

Scored items:

- `reasons_correct`
- `fcra_window_correct`
- `real_agency_named`
- `ecoa_classes_correct`
- `no_legal_errors`

Required output shape:

```json
{
  "reasons_correct": 0,
  "fcra_window_correct": 0,
  "real_agency_named": 0,
  "ecoa_classes_correct": 0,
  "no_legal_errors": 0,
  "note": "one short sentence citing the deciding detail"
}
```

Recommended appendix treatment: include the scored items and source file path. The full rubric can be included if there is room, because it is shorter and central to the RAG-vs-no-RAG evaluation.

## Recommended Final Report Prompt Appendix

For the final report, the most important prompts are:

1. The RAG writer prompt.
2. The no-RAG control prompt.
3. The compliance judge rubric.
4. A short citation to the EDA prompt.
5. A short citation to the final-report drafting prompt.

That gives the reader enough evidence to understand how AI was used without making the appendix longer than the project results.
