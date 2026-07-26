# RAG Letter Generation Demo

Open `rag_letter_demo.html` in a browser to view the local demo.

The demo summarizes the completed generation run:

- 5 RAG letters from `generation/output/letters.jsonl`
- 5 no-RAG control letters from `generation/output/letters_norag.jsonl`
- Judge results from `generation/output/judged.jsonl`
- Aggregate scores from `generation/output/results.json`

The page is self-contained and does not call Gemini, run retrieval, or require API keys.

## Demo Story

The demo shows how the project moves from model interpretation to letter generation:

```text
XGBoost risk score
-> SHAP borrower reasons
-> regulatory retrieval
-> RAG letter generation
-> blind LLM judge scoring
```

Current measured result:

```text
No-RAG overall statutory-accuracy score: 0.60
RAG overall statutory-accuracy score:    0.76
```

The main remaining weakness is federal-agency specificity. Some letters still used placeholders instead of naming a real federal enforcement agency, so the generation layer should be presented as a research prototype rather than a compliance-approved adverse-action system.

