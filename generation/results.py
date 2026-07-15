# generation/results.py
"""Step 4c — aggregate judge scores into RAG vs no-RAG results.
Pure local computation (no API). Reads judged.jsonl, averages each rubric
item per mode, prints a table, and writes results.json + results.md."""
import json, pathlib
from collections import defaultdict

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT  = ROOT / "generation" / "output"
ITEMS = ["reasons_correct", "fcra_window_correct", "real_agency_named", "ecoa_classes_correct", "no_legal_errors"]

def main():
    recs = [json.loads(l) for l in open(OUT / "judged.jsonl")]
    by_mode = defaultdict(list)
    for r in recs:
        if "error" not in r["scores"]:
            by_mode[r["mode"]].append(r["scores"])

    summary = {}
    for mode, rows in by_mode.items():
        n = len(rows)
        per_item = {it: sum(x.get(it, 0) for x in rows) / n for it in ITEMS}
        overall = sum(per_item.values()) / len(ITEMS)
        summary[mode] = {"n": n, "per_item": per_item, "overall": overall}

    # console table
    hdr = f"{'item':20s} " + " ".join(f"{m:>8s}" for m in summary)
    print(hdr); print("-" * len(hdr))
    for it in ITEMS:
        print(f"{it:20s} " + " ".join(f"{summary[m]['per_item'][it]:8.2f}" for m in summary))
    print("-" * len(hdr))
    print(f"{'OVERALL':20s} " + " ".join(f"{summary[m]['overall']:8.2f}" for m in summary))
    print(f"{'n letters':20s} " + " ".join(f"{summary[m]['n']:8d}" for m in summary))

    # markdown for the writeup
    modes = list(summary)
    md = ["# RAG vs no-RAG — adverse-action letter compliance\n",
          f"Judge: gemini-3.1-flash-lite (blind, shuffled). Scores are mean of {summary[modes[0]]['n']} letters/mode, each item 0/1.\n",
          "| item | " + " | ".join(modes) + " |",
          "|" + "---|" * (len(modes) + 1)]
    for it in ITEMS:
        md.append(f"| {it} | " + " | ".join(f"{summary[m]['per_item'][it]:.2f}" for m in modes) + " |")
    md.append(f"| **overall** | " + " | ".join(f"**{summary[m]['overall']:.2f}**" for m in modes) + " |")
    (OUT / "results.md").write_text("\n".join(md) + "\n")
    (OUT / "results.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT/'results.md'} and results.json")

if __name__ == "__main__":
    main()
