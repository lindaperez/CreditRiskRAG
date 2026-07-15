# generation/generate_letter.py
"""Step 3b — RAG letter generation (single borrower test).
Retrieves the governing notice law, injects it + the borrower's SHAP
reasons into the prompt, asks Gemini to draft an ECOA-compliant
adverse-action letter. Writer model only; judge comes in Step 4."""
import os, pathlib
from dotenv import load_dotenv
from google import genai
from retrieval.hybrid_retriever import HybridRetriever
from generation.select_reasons import select_reasons

load_dotenv()
ROOT = pathlib.Path(__file__).resolve().parent.parent
WRITER_MODEL = "gemini-3-flash-preview"

# fixed queries: retrieve the *notice requirements* (envelope), not per-reason law
LAW_QUERIES = [
    "what must an adverse action notice contain",
    "statement of specific principal reasons requirement",
    "credit score disclosure adverse action",
]

def retrieve_law(retriever, k_per_query=2):
    seen, chunks = set(), []
    for q in LAW_QUERIES:
        for h in retriever.search(q, k=k_per_query):
            if h["chunk_id"] not in seen:
                seen.add(h["chunk_id"]); chunks.append(h)
    return chunks

def build_prompt(reasons, law_chunks):
    law = "\n\n".join(f"[{c['citation']}] {c['text']}" for c in law_chunks)
    reason_lines = "\n".join(f"- {r['reason']}" for r in reasons["reasons"])
    return f"""You are a compliance officer drafting an adverse-action notice for a declined credit application, following US law.

GOVERNING LAW (ground your letter strictly in these authorities):
{law}

APPLICANT'S PRINCIPAL REASONS FOR THE DECISION (state these; do not invent others):
{reason_lines}

Write a professional adverse-action letter that:
- states the credit decision is adverse (application declined),
- gives the specific, principal reasons above (do not add reasons or cite numeric values),
- includes the ECOA notice language and the FCRA disclosures the governing law requires,
- is addressed generically ("Dear Applicant"), signed generically.
Return only the letter text."""

def generate(row_index="443554"):
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    retriever = HybridRetriever()
    reasons = select_reasons(row_index)
    law = retrieve_law(retriever)
    prompt = build_prompt(reasons, law)
    print(f"borrower {row_index}: {reasons['n_reasons']} reasons, "
          f"{len(law)} law chunks retrieved\n" + "="*60)
    resp = client.models.generate_content(model=WRITER_MODEL, contents=prompt)
    print(resp.text)

if __name__ == "__main__":
    generate()
