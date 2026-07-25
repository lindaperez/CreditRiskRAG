# RAG Letter Generation Decisions And Results

## Decision

Use RAG-grounded generation for adverse-action-style letter drafting, but keep the output positioned as a research prototype and compliance-support artifact, not a production adverse-action notice.

The project uses the preferred credit-risk model and SHAP reason selection to identify principal risk reasons for high-risk accepted loans. The RAG layer then retrieves governing regulatory text and asks a writer model to draft a letter using only those borrower reasons and the retrieved legal context.

## Pipeline Summary

The letter-generation workflow has four stages:

1. Build the regulatory corpus.
2. Build the retrieval index.
3. Select borrower-level SHAP reasons.
4. Generate and evaluate RAG vs no-RAG letters.

Relevant scripts:

```text
corpus/chunk_corpus.py
retrieval/build_index.py
retrieval/hybrid_retriever.py
generation/select_reasons.py
generation/generate_letter.py
generation/generate_all.py
generation/generate_norag.py
generation/judge.py
generation/results.py
```

## Corpus Decision

The corpus is limited to public U.S. regulatory sources relevant to adverse-action notices:

| Source File | Citation |
| --- | --- |
| `reg_b_1002_2.html` | 12 CFR 1002.2 |
| `reg_b_1002_9.html` | 12 CFR 1002.9 |
| `reg_b_appendix_a.html` | 12 CFR pt.1002 App. A |
| `reg_b_appendix_c.html` | 12 CFR pt.1002 App. C |
| `reg_b_supplement_i_9.html` | 12 CFR pt.1002 Supp. I, comment 9 |
| `fcra_615.html` | 15 USC 1681m, FCRA 615 |
| `cfpb_circular_2022_03.html` | CFPB Circular 2022-03 |
| `cfpb_circular_2023_03.html` | CFPB Circular 2023-03 |

Decision: keep the corpus narrow and authoritative. These sources cover adverse-action notice content, specific principal reasons, sample notices, federal-agency information, credit-report disclosures, and CFPB guidance for algorithmic credit decisions.

Corpus result:

```text
corpus/chunks.jsonl: 70 chunks
```

## Chunking Decision

`corpus/chunk_corpus.py` performs structure-aware chunking:

- Reads saved HTML or TXT files from `corpus/raw/`.
- Strips scripts, navigation, forms, headers, footers, and boilerplate.
- Splits on CFR/USC paragraph markers when available.
- Falls back to sentence splitting for prose sources such as CFPB circulars.
- Preserves citation metadata per chunk.

Decision: chunk for legal grounding, not just semantic similarity. Each generated chunk keeps a citation such as `12 CFR 1002.9`, `15 USC 1681m`, or CFPB circular metadata so generated letters can be tied back to authority.

## Retrieval Decision

`retrieval/build_index.py` embeds all chunks using:

```text
sentence-transformers/all-MiniLM-L6-v2
```

`retrieval/hybrid_retriever.py` combines:

- Dense cosine similarity over MiniLM embeddings.
- BM25 lexical retrieval.
- Reciprocal Rank Fusion to merge dense and lexical rankings.

Decision: use hybrid retrieval because legal text often depends on exact terms such as "adverse action", "specific reasons", "60 days", and agency names, while dense retrieval helps capture semantically similar legal passages.

The RAG letter generator uses fixed legal-envelope queries:

```text
what must an adverse action notice contain
statement of specific principal reasons requirement
credit score disclosure adverse action
```

Each query retrieves the top law chunks, deduplicates them by `chunk_id`, and injects them into the writer prompt.

## Reason-Selection Decision

`generation/select_reasons.py` selects reasons from SHAP local explanations without an LLM.

Policy:

- Use high-risk accepted-loan examples only.
- Keep only SHAP drivers that increase modeled risk.
- Keep only mapped features with draft human-readable reason codes.
- Drop `int_rate_clean` from letter reasons because it is a lender pricing output.
- Sort by absolute SHAP contribution.
- Cap the letter at 4 principal reasons.

Decision: reason selection should be deterministic, auditable, and constrained. The generator should not invent reasons beyond the selected SHAP-supported reason list.

Example selected reasons:

```text
High debt burden relative to income.
Lower credit score range.
Larger requested loan amount.
Larger payment obligation.
```

## Generation Decision

The writer model is:

```text
gemini-3-flash-preview
```

RAG mode:

- Uses SHAP-selected borrower reasons.
- Uses retrieved legal chunks.
- Instructs the model to ground the letter strictly in the authorities.
- Requires the letter to state the adverse decision, list the specific principal reasons, include ECOA notice language, include FCRA disclosures, and avoid invented reasons or numeric feature values.

No-RAG mode:

- Uses the same borrower reasons.
- Does not include retrieved law.
- Is used as the control condition to isolate the effect of retrieval.

Output files:

```text
generation/output/letters.jsonl
generation/output/letters_norag.jsonl
generation/output/txt/letter_<row>.txt
generation/output/txt/norag_<row>.txt
```

Current run size:

```text
RAG letters: 5
No-RAG letters: 5
```

## Evaluation Decision

`generation/judge.py` uses a separate judge model:

```text
gemini-3.1-flash-lite
```

The judge is blind and shuffled across RAG/no-RAG records. It scores statutory accuracy, not just whether disclosure text exists.

Rubric items:

| Item | Meaning |
| --- | --- |
| `reasons_correct` | Letter states exactly the selected principal reasons, no more and no fewer. |
| `fcra_window_correct` | Letter states the right to a free consumer report within exactly 60 days. |
| `real_agency_named` | Letter names a real federal enforcement agency instead of a placeholder or fake agency. |
| `ecoa_classes_correct` | ECOA protected-class language is materially correct. |
| `no_legal_errors` | No other incorrect legal or factual claims. |

Decision: evaluate legal specificity. Earlier presence-only checks were too weak because both RAG and no-RAG letters could include generic disclosures while still getting legal details wrong.

## Results

Current aggregate results from `generation/output/results.json`:

| Rubric Item | No-RAG | RAG |
| --- | ---: | ---: |
| `reasons_correct` | 1.00 | 1.00 |
| `fcra_window_correct` | 1.00 | 1.00 |
| `real_agency_named` | 0.00 | 0.40 |
| `ecoa_classes_correct` | 1.00 | 1.00 |
| `no_legal_errors` | 0.00 | 0.40 |
| **Overall** | **0.60** | **0.76** |

Interpretation:

- RAG improved the overall statutory-accuracy score from `0.60` to `0.76`.
- Both modes correctly preserved the selected principal reasons.
- Both modes correctly included the FCRA 60-day free-report window.
- Both modes correctly included ECOA protected-class language.
- The main weakness was federal-agency specificity: no-RAG scored `0.00`; RAG improved to `0.40` but still sometimes used placeholders instead of naming a real agency.
- The `no_legal_errors` item followed the same pattern: no-RAG scored `0.00`; RAG improved to `0.40`, but the remaining agency-placeholder issue still creates legal-quality risk.

## Final Letter-Generation Position

The RAG workflow is better than no-RAG for this project because it improves legal grounding and statutory-accuracy scoring while preserving the model-selected SHAP reasons.

However, the current generation layer is not compliance-ready. The sample is small, and the agency-name failure shows that retrieved context alone does not guarantee a complete legally correct notice.

Recommended final project wording:

```text
RAG-grounded adverse-action-style letter generation improved statutory-accuracy scores versus a no-RAG control, increasing the overall judge score from 0.60 to 0.76 on a 5-letter-per-mode evaluation. The generated letters correctly preserved selected SHAP reasons and core ECOA/FCRA disclosures, but federal-agency specificity remained a weakness. Therefore, the letter-generation layer should be presented as a research prototype and explanation bridge, not as a production or compliance-approved adverse-action system.
```

## Next Improvements

1. Add a deterministic federal-agency block instead of relying on free-form generation.
2. Retrieve Appendix A agency information explicitly when generating notice templates.
3. Increase the evaluation sample beyond 5 letters per mode.
4. Add rule-based validators for required fields before LLM judging.
5. Separate technical SHAP reasons from applicant-facing approved reason-code language.
6. Add compliance review before presenting any output as an adverse-action notice.

