# generation/generate_all.py
"""Step 3c — generate RAG letters for all high-risk borrowers.
Canonical output: generation/output/letters.jsonl (one record per borrower).
Human-readable: generation/output/txt/letter_<row>.txt
Reuses retrieval + reason-selection + prompt from the single-borrower test."""
import os, json, time, pathlib
from dotenv import load_dotenv
from google import genai
from retrieval.hybrid_retriever import HybridRetriever
from generation.select_reasons import select_reasons, high_risk_row_indices, _load_borrowers
from generation.generate_letter import retrieve_law, build_prompt, WRITER_MODEL

load_dotenv()
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT  = ROOT / "generation" / "output"
TXT  = OUT / "txt"

def main():
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    retriever = HybridRetriever()                 # build ONCE, reuse
    borrowers = _load_borrowers()
    rows = high_risk_row_indices(borrowers)
    law = retrieve_law(retriever)                 # same law envelope for all
    law_ids = [c["chunk_id"] for c in law]
    OUT.mkdir(parents=True, exist_ok=True); TXT.mkdir(exist_ok=True)

    records = []
    for i, rid in enumerate(rows, 1):
        reasons = select_reasons(rid, borrowers=borrowers)
        prompt  = build_prompt(reasons, law)
        print(f"[{i}/{len(rows)}] borrower {rid} ... ", end="", flush=True)
        resp = client.models.generate_content(model=WRITER_MODEL, contents=prompt)
        letter = resp.text
        rec = {"row_index": rid,
               "mode": "rag",
               "predicted_default_prob": reasons["predicted_default_prob"],
               "reasons": [r["reason"] for r in reasons["reasons"]],
               "reason_features": [r["feature"] for r in reasons["reasons"]],
               "law_chunk_ids": law_ids,
               "writer_model": WRITER_MODEL,
               "letter_text": letter}
        records.append(rec)
        (TXT / f"letter_{rid}.txt").write_text(letter)
        print("ok")
        if i < len(rows): time.sleep(7)           # stay under free-tier RPM

    with open(OUT / "letters.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(records)} letters -> {OUT/'letters.jsonl'}")
    print(f"readable copies -> {TXT}/")

if __name__ == "__main__":
    main()
