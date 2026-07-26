# RAG vs no-RAG — adverse-action letter compliance

Judge: gemini-3.1-flash-lite (blind, shuffled). Scores are mean of 5 letters/mode, each item 0/1.

| item | norag | rag |
|---|---|---|
| reasons_correct | 1.00 | 1.00 |
| fcra_window_correct | 1.00 | 1.00 |
| real_agency_named | 0.00 | 0.40 |
| ecoa_classes_correct | 1.00 | 1.00 |
| no_legal_errors | 0.00 | 0.40 |
| **overall** | **0.60** | **0.76** |
