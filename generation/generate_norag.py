# generation/generate_norag.py
"""Step 4a — no-RAG control letters.
Same borrowers + same reasons as the RAG run, but NO retrieved law in the
prompt. Isolates the effect of retrieval: the model must rely on its own
knowledge of adverse-action requirements instead of grounded statute text.
Output: generation/output/letters_norag.jsonl (+ txt/norag_<row>.txt)."""
import os, json, time, pathlib
from dotenv import load_dotenv
from google import genai
from generation.select_reasons import select_reasons, high_risk_row_indices, _load_borrowers
from generation.generate_letter import WRITER_MODEL

load_dotenv()
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT  = ROOT / "generation" / "output"
TXT  = OUT / "txt"

def build_prompt_norag(reasons):
    reason_lines = "\n".join(f"- {r['reason']}" for r in reasons["reasons"])
    return f"""You are a compliance officer drafting an adverse-action notice for a declined credit application, following US law.

APPLICANT'S PRINCIPAL REASONS FOR THE DECISION (state these; do not invent others):
{reason_lines}

Write a professional adverse-action letter that:
- states the credit decision is adverse (application declined),
- gives the specific, principal reasons above (do not add reasons or cite numeric values),
- includes the ECOA notice language and the FCRA disclosures required for adverse-action notices,
- is addressed generically ("Dear Applicant"), signed generically.
Return only the letter text."""

def main():
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    borrowers = _load_borrowers()
    rows = high_risk_row_indices(borrowers)
    OUT.mkdir(parents=True, exist_ok=True); TXT.mkdir(exist_ok=True)

    records = []
    for i, rid in enumerate(rows, 1):
        reasons = select_reasons(rid, borrowers=borrowers)
        prompt  = build_prompt_norag(reasons)
        print(f"[{i}/{len(rows)}] borrower {rid} (no-rag) ... ", end="", flush=True)
        resp = client.models.generate_content(model=WRITER_MODEL, contents=prompt)
        letter = resp.text
        records.append({"row_index": rid,
                        "mode": "norag",
                        "predicted_default_prob": reasons["predicted_default_prob"],
                        "reasons": [r["reason"] for r in reasons["reasons"]],
                        "reason_features": [r["feature"] for r in reasons["reasons"]],
                        "law_chunk_ids": [],
                        "writer_model": WRITER_MODEL,
                        "letter_text": letter})
        (TXT / f"norag_{rid}.txt").write_text(letter)
        print("ok")
        if i < len(rows): time.sleep(7)

    with open(OUT / "letters_norag.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(records)} no-rag letters -> {OUT/'letters_norag.jsonl'}")

if __name__ == "__main__":
    main()
