# Regulatory corpus

All sources are US-government works → public domain, no copyright/licensing concern (note this in the report).

## What to save (into `corpus/raw/`)

Open each URL, save the page text. Two easy ways:
- Browser: File → Save Page As → **Web Page, HTML Only**, or
- Terminal: `curl -sL "<url>" -o corpus/raw/<name>.html`

Use these exact filenames (the chunker parses the citation from them):

| Save as (`corpus/raw/`)        | Citation              | URL |
| ------------------------------ | --------------------- | --- |
| `reg_b_1002_2.html`            | 12 CFR 1002.2         | https://www.ecfr.gov/current/title-12/chapter-X/part-1002/subpart-A/section-1002.2 |
| `reg_b_1002_9.html`            | 12 CFR 1002.9         | https://www.ecfr.gov/current/title-12/chapter-X/part-1002/subpart-A/section-1002.9 |
| `reg_b_appendix_a.html`        | 12 CFR pt.1002 App. A | https://www.ecfr.gov/current/title-12/chapter-X/part-1002/subpart-B/appendix-Appendix%20A%20to%20Part%201002 |
| `reg_b_appendix_c.html`        | 12 CFR pt.1002 App. C | https://www.ecfr.gov/current/title-12/chapter-X/part-1002/subpart-B/appendix-Appendix%20C%20to%20Part%201002 |
| `reg_b_supplement_i_9.html`    | 12 CFR pt.1002 Supp.I §9 | https://www.consumerfinance.gov/rules-policy/regulations/1002/Interp-9/ |
| `fcra_615.html`                | 15 USC 1681m (FCRA 615) | https://www.law.cornell.edu/uscode/text/15/1681m |
| `cfpb_circular_2022_03.html`   | CFPB Circular 2022-03 | https://www.consumerfinance.gov/compliance/circulars/circular-2022-03-adverse-action-notification-requirements-in-connection-with-credit-decisions-based-on-complex-algorithms/ |
| `cfpb_circular_2023_03.html`   | CFPB Circular 2023-03 | https://www.consumerfinance.gov/compliance/circulars/circular-2023-03-adverse-action-notification-requirements-and-the-proper-use-of-the-cfpbs-sample-forms-provided-in-regulation-b/ |

## Why these, briefly
- **1002.9** — the core rule: notice content, timing, "specific principal reasons".
- **Appendix C** — the official sample adverse-action letters (C-1…C-5). Highest-value retrieval targets for generation.
- **Appendix A** — the federal-agency address block your letter is required to include.
- **1002.2** — definition of "adverse action".
- **Supplement I (§9)** — official commentary; includes the "≤4 reasons" and "credit-score factors ≠ ECOA reasons" points.
- **FCRA 615** — adverse action based on a consumer report / credit-score disclosure.
- **Circulars 2022-03, 2023-03** — the ML/algorithmic angle: black-box models still owe specific, accurate reasons. Ties RAG to your research question.

## Then run
```bash
python corpus/chunk_corpus.py
```
Produces `corpus/chunks.jsonl` — one JSON object per chunk with `{chunk_id, source, citation, text}`. That file is the input to the retrieval step (BM25 + vector index).
