# Rubric validation (gold examples)

Hand-scored minimal-pair test of `generation/judge.py`'s 5-item rubric, built to check early whether the rubric actually captures what a real adverse-action letter needs — before relying on it for the RAG-vs-no-RAG headline metric.

## Method

`gold_examples.py` has one fully-compliant baseline letter and 5 single-defect variants, each changing exactly one rubric-relevant detail and leaving everything else correct:

| Example | Change from baseline |
| --- | --- |
| `baseline_all_pass` | none — should score 1 on all 5 items |
| `fail_fcra_window` | "60 days" -> "30 days" |
| `fail_real_agency_named` | real agency/address -> `[Federal Enforcement Agency Name], [Address]` (the placeholder pattern seen in real no-RAG output) |
| `fail_ecoa_classes` | full 8-basis ECOA list -> short, materially wrong list |
| `fail_no_legal_errors` | appends a fabricated citation + invented numeric deadline |
| `fail_reasons_correct` | adds a reason not in the applicant's actual reason list |

`check_rubric.py` runs all 6 through the live judge model (`JUDGE_MODEL`, separate quota from the writer model) and diffs actual vs. expected scores.

## Result (before fix)

**2/6 matched exactly.** The baseline passed cleanly, and the deliberate `no_legal_errors` case correctly failed only that item. But all 4 *other* failure cases also tripped `no_legal_errors = 0`, even though only one specific item was meant to fail:

| Example | Only mismatch |
| --- | --- |
| `fail_fcra_window` | `no_legal_errors` expected 1, got 0 |
| `fail_real_agency_named` | `no_legal_errors` expected 1, got 0 |
| `fail_ecoa_classes` | `no_legal_errors` expected 1, got 0 |
| `fail_reasons_correct` | `no_legal_errors` expected 1, got 0 |

The judge's free-text `note` in every case correctly diagnosed the intended defect (e.g. "incorrectly cites a 30-day window", "included an extra reason not listed"), so **this was not a judge error** — the judge was reading the rubric prompt correctly. The rubric prompt's own original definition of `no_legal_errors` ("no incorrect statutory claims... no invented reasons...") was written broadly enough to structurally overlap with `fcra_window_correct`, `ecoa_classes_correct`, and `reasons_correct`.

## Why this mattered for the headline result

`no_legal_errors` was not an independent 5th signal — it behaved as a catch-all that mirrored whichever other item already failed. Two consequences:

1. **Double-counting.** A single defect (e.g. a wrong FCRA window) reduced the mean "overall" score by 2/5 instead of 1/5, inflating the apparent size of any one flaw.
2. **This is very likely what drove the original 0.76 vs 0.60 result.** In every judged letter so far, `real_agency_named` and `no_legal_errors` moved in lockstep — never split. If those two are really one underlying behavior (does the model name a real agency or not) counted twice, the RAG-vs-no-RAG gap was being driven by a single behavior double-weighted, not two independent compliance dimensions.

## Fix applied

Narrowed the `no_legal_errors` bullet in `judge.py`'s `RUBRIC` to explicitly exclude the failure modes already covered by the other 4 items, and added an instruction that the 5 items should be scored independently. New wording only marks `no_legal_errors = 0` for a *different* error not already captured by the other four (fabricated citations, invented numeric deadlines, etc.).

## Result (after fix)

**6/6 matched exactly**, including a corrected `fail_ecoa_classes` test case (the first version bundled two error types — omitting classes *and* inventing a fake one, `"income level"` — which made `no_legal_errors=0` a legitimate call rather than overlap; isolating just the omission confirmed the fix is clean).

## Effect on the actual RAG-vs-no-RAG result

Re-judged the same 5+5 letters (`letters.jsonl` / `letters_norag.jsonl`) with the fixed rubric:

| | no-RAG | RAG | gap |
| --- | ---: | ---: | ---: |
| Old rubric (double-counted) | 0.92 | 0.76 | 0.16 |
| Fixed rubric | 0.96 | 0.88 | 0.08 |

`no_legal_errors` is now 1.00/1.00 for both modes, confirming it was never independently informative on this data — it was mirroring `real_agency_named` exactly. With the overlap removed, the gap is exactly half of what it was, and **`real_agency_named` is now the only item separating RAG from no-RAG.** The direction of the finding (RAG names the real agency more reliably) still holds, but its previously-reported size was inflated 2x by the rubric bug. This is still n=5, so the number itself should not be treated as final either way.
