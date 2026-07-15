# generation/judge.py
"""Step 4b (v2) - LLM judge scoring statutory ACCURACY, not mere presence.
Earlier rubric checked whether disclosures existed (both modes maxed out).
This version checks specifics only the actual statute pins down - the kind
of detail a model must either retrieve or recall correctly:
  - free-report window stated as exactly 60 days (FCRA 615)
  - a REAL federal enforcement agency named (not vague/invented)
  - ECOA protected-class list matches the statute
  - no incorrect statutory claims (wrong windows, fake citations, invented reasons)
Blind + shuffled. Judge model differs from writer.
Output: generation/output/judged.jsonl"""
import os, json, random, time, pathlib
from dotenv import load_dotenv
from google import genai

load_dotenv()
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT  = ROOT / "generation" / "output"
JUDGE_MODEL = "gemini-3.1-flash-lite"
ITEMS = ["reasons_correct", "fcra_window_correct", "real_agency_named",
         "ecoa_classes_correct", "no_legal_errors"]

RUBRIC = """You are a compliance auditor grading the STATUTORY ACCURACY of a credit adverse-action letter.
Do not reward a disclosure merely for being present - grade whether its legal specifics are CORRECT.
Score each item 1 (correct) or 0 (missing, vague, or wrong). Judge ONLY the letter text.

Return ONLY a JSON object, no other text, exactly this shape:
{"reasons_correct": 0 or 1, "fcra_window_correct": 0 or 1, "real_agency_named": 0 or 1, "ecoa_classes_correct": 0 or 1, "no_legal_errors": 0 or 1, "note": "one short sentence citing the deciding detail"}

Grade strictly on these specifics:
- reasons_correct: states exactly the applicant's principal reasons listed below, no more, no fewer.
- fcra_window_correct: states the applicant's right to a free consumer report within exactly 60 days. Wrong number (30, 90, etc.) or no window = 0.
- real_agency_named: names a REAL federal enforcement agency (e.g. Consumer Financial Protection Bureau, FTC) as the agency administering compliance. A bracketed blank placeholder like [Agency Name] with no real agency = 0. An invented/wrong agency = 0.
- ecoa_classes_correct: the ECOA anti-discrimination notice lists the correct protected bases (race, color, religion, national origin, sex, marital status, age; receipt of public assistance; good-faith exercise of Consumer Credit Protection Act rights). Materially wrong or missing list = 0.
- no_legal_errors: contains no incorrect statutory claims, no fabricated citations, no invented reasons, and does not assert specific numeric values as facts about this applicant. (Empty lender-fill placeholders for score/date are acceptable and NOT errors.)

The applicant's principal reasons were:
<<REASONS>>

LETTER:
<<LETTER>>
"""

def _load(path):
    return [json.loads(l) for l in open(path)]

def score_one(client, rec):
    prompt = (RUBRIC
              .replace("<<REASONS>>", "\n".join(f"- {r}" for r in rec["reasons"]))
              .replace("<<LETTER>>", rec["letter_text"]))
    resp = client.models.generate_content(model=JUDGE_MODEL, contents=prompt)
    raw = resp.text.strip().replace("```json", "").replace("```", "").strip()
    return json.loads(raw)

def main():
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    recs = _load(OUT / "letters.jsonl") + _load(OUT / "letters_norag.jsonl")
    random.seed(42); random.shuffle(recs)

    judged = []
    for i, rec in enumerate(recs, 1):
        print(f"[{i}/{len(recs)}] judging {rec['row_index']} ({rec['mode']}) ... ", end="", flush=True)
        try:
            s = score_one(client, rec); print("ok")
        except Exception as e:
            print("ERR", repr(e)); s = {"error": str(e)}
        judged.append({**{k: rec[k] for k in ("row_index","mode","reasons")}, "scores": s})
        if i < len(recs): time.sleep(7)

    with open(OUT / "judged.jsonl", "w") as f:
        for r in judged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(judged)} judgments -> {OUT/'judged.jsonl'}")

if __name__ == "__main__":
    main()
