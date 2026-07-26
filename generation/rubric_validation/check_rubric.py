# generation/rubric_validation/check_rubric.py
"""Runs the hand-scored minimal-pair examples in gold_examples.py through
the live judge (generation.judge.score_one) and reports whether the judge's
scores match the expected human-scored labels, item by item.

Uses the JUDGE_MODEL only (not the writer model that hit its daily quota),
so this is cheap: 6 calls total for the 6 gold examples.

Output: generation/rubric_validation/rubric_check_results.json
"""
import json, time, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
import os
from google import genai
from generation.judge import score_one, ITEMS
from generation.rubric_validation.gold_examples import EXAMPLES, REASONS

load_dotenv()
OUT = pathlib.Path(__file__).resolve().parent / "rubric_check_results.json"


def main():
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    results = []
    names = list(EXAMPLES)
    for i, name in enumerate(names, 1):
        letter_text, expected, note = EXAMPLES[name]
        rec = {"reasons": REASONS, "letter_text": letter_text}
        print(f"[{i}/{len(names)}] {name} ... ", end="", flush=True)
        try:
            actual = score_one(client, rec)
        except Exception as e:
            print("ERR", repr(e))
            results.append({"example": name, "note": note, "error": str(e)})
            if i < len(names):
                time.sleep(7)
            continue

        mismatches = {it: {"expected": expected[it], "actual": actual.get(it)}
                      for it in ITEMS if actual.get(it) != expected[it]}
        agree = not mismatches
        print("MATCH" if agree else f"MISMATCH {mismatches}")
        results.append({
            "example": name,
            "note": note,
            "expected": expected,
            "actual": {it: actual.get(it) for it in ITEMS},
            "judge_note": actual.get("note"),
            "agree": agree,
            "mismatches": mismatches,
        })
        if i < len(names):
            time.sleep(7)

    n_ok = sum(1 for r in results if r.get("agree"))
    n_total = sum(1 for r in results if "error" not in r)
    print(f"\n{n_ok}/{n_total} examples matched expected rubric scores exactly.")
    OUT.write_text(json.dumps(results, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
