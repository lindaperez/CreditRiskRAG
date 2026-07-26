# generation/select_reasons.py
"""Step 3a — reason selection (no LLM).
For one borrower: pick the principal adverse-action reasons from SHAP.
Policy: high_risk group only -> risk-INCREASING drivers -> mapped features
only -> drop int_rate_clean (lender pricing output) -> sort by |shap| ->
cap at 4. Uses `direction` qualitatively; never the z-scored feature_value."""
import csv, pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOCAL = ROOT / "Interpretation_SHAP/shap_outputs/tables/shap_local_examples.csv"
MAP   = ROOT / "Interpretation_SHAP/reason_codes/draft_reason_code_mapping.csv"
DROP  = {"int_rate_clean"}
MAX_REASONS = 4

def _load_mapping():
    rows = csv.DictReader(open(MAP))
    return {r["feature"]: r["draft_reason_code"].strip()
            for r in rows if r["draft_reason_code"].strip()}

def _load_borrowers():
    borrowers = {}
    for r in csv.DictReader(open(LOCAL)):
        borrowers.setdefault(r["row_index"], []).append(r)
    return borrowers

def high_risk_row_indices(borrowers=None):
    borrowers = borrowers or _load_borrowers()
    return [rid for rid, rows in borrowers.items()
            if rows[0]["group"] == "high_risk"]

def select_reasons(row_index, mapping=None, borrowers=None):
    mapping   = mapping   or _load_mapping()
    borrowers = borrowers or _load_borrowers()
    rows = borrowers[str(row_index)]
    cands = [r for r in rows
             if r["direction"] == "increases_risk"
             and r["feature"] in mapping
             and r["feature"] not in DROP]
    cands.sort(key=lambda r: abs(float(r["shap_value"])), reverse=True)
    reasons = [{"feature": r["feature"],
                "reason": mapping[r["feature"]],
                "direction": r["direction"]}
               for r in cands[:MAX_REASONS]]
    return {"row_index": str(row_index),
            "group": rows[0]["group"],
            "predicted_default_prob": float(rows[0]["predicted_default_prob"]),
            "n_reasons": len(reasons),
            "reasons": reasons}

if __name__ == "__main__":
    mp, bw = _load_mapping(), _load_borrowers()
    hr = high_risk_row_indices(bw)
    print(f"{len(hr)} high_risk borrowers: {hr}\n")
    for rid in hr:
        s = select_reasons(rid, mp, bw)
        print(f"row {rid:8s} p={s['predicted_default_prob']:.3f} n={s['n_reasons']}")
        for r in s["reasons"]:
            print(f"    - {r['feature']:22s} | {r['reason']}")
        print()
