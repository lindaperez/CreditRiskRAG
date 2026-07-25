# generation/rubric_validation/gold_examples.py
"""Hand-authored minimal-pair examples for judge.py's 5-item rubric.

Each variant is the BASELINE letter with exactly one rubric-relevant detail
changed to a realistic failure mode (the same kinds of errors observed in
real generated letters, e.g. bracket placeholders, wrong day-counts). All
other rubric-relevant content is held identical, so a correct judge should
flip exactly one item per variant and leave the other four at 1.

This is a hand-scored gold set to sanity-check the automated LLM judge,
not a claim that these are the only ways a letter can fail.
"""

REASONS = [
    "High debt burden relative to income.",
    "Lower credit score range.",
    "Longer loan repayment term.",
]

BASELINE_LETTER = """Dear Applicant,

We are writing to inform you that your recent credit application has been declined.

The principal reasons for this decision were:
- High debt burden relative to income.
- Lower credit score range.
- Longer loan repayment term.

Under the Fair Credit Reporting Act, you have the right to obtain a free copy of your consumer report from the reporting agency within 60 days of the date of this notice.

The federal Equal Credit Opportunity Act prohibits creditors from discriminating against credit applicants on the basis of race, color, religion, national origin, sex, marital status, or age (provided the applicant has the capacity to enter into a binding contract); because all or part of the applicant's income derives from any public assistance program; or because the applicant has in good faith exercised any right under the Consumer Credit Protection Act. The federal agency that administers compliance with this law concerning this creditor is the Consumer Financial Protection Bureau, 1700 G Street NW, Washington, DC 20552.

Sincerely,
Credit Department"""

# name -> (letter_text, expected_scores, what_changed)
EXAMPLES = {
    "baseline_all_pass": (
        BASELINE_LETTER,
        {"reasons_correct": 1, "fcra_window_correct": 1, "real_agency_named": 1,
         "ecoa_classes_correct": 1, "no_legal_errors": 1},
        "Fully compliant control letter; every item should score 1.",
    ),
    "fail_fcra_window": (
        BASELINE_LETTER.replace("within 60 days", "within 30 days"),
        {"reasons_correct": 1, "fcra_window_correct": 0, "real_agency_named": 1,
         "ecoa_classes_correct": 1, "no_legal_errors": 1},
        "60-day FCRA window changed to the wrong number (30 days).",
    ),
    "fail_real_agency_named": (
        BASELINE_LETTER.replace(
            "the Consumer Financial Protection Bureau, 1700 G Street NW, Washington, DC 20552",
            "[Federal Enforcement Agency Name], [Address]",
        ),
        {"reasons_correct": 1, "fcra_window_correct": 1, "real_agency_named": 0,
         "ecoa_classes_correct": 1, "no_legal_errors": 1},
        "Real agency name/address replaced with a bracket placeholder "
        "(the exact failure mode observed in real no-RAG generations).",
    ),
    "fail_ecoa_classes": (
        BASELINE_LETTER.replace(
            "prohibits creditors from discriminating against credit applicants on the basis of "
            "race, color, religion, national origin, sex, marital status, or age (provided the "
            "applicant has the capacity to enter into a binding contract); because all or part "
            "of the applicant's income derives from any public assistance program; or because "
            "the applicant has in good faith exercised any right under the Consumer Credit "
            "Protection Act.",
            "prohibits discriminating against credit applicants on the basis of race, color, "
            "and sex.",
        ),
        {"reasons_correct": 1, "fcra_window_correct": 1, "real_agency_named": 1,
         "ecoa_classes_correct": 0, "no_legal_errors": 1},
        "Correct 8-item ECOA protected-class list replaced with an incomplete list "
        "(only 3 of 8 bases, no invented/fake categories) to isolate omission from invention.",
    ),
    "fail_no_legal_errors": (
        BASELINE_LETTER + (
            "\n\nP.S. Pursuant to 15 U.S.C. Section 1691(a)(12), you are entitled to a full "
            "written explanation of this decision within 10 business days."
        ),
        {"reasons_correct": 1, "fcra_window_correct": 1, "real_agency_named": 1,
         "ecoa_classes_correct": 1, "no_legal_errors": 0},
        "Appends a fabricated statutory citation and an invented numeric deadline.",
    ),
    "fail_reasons_correct": (
        BASELINE_LETTER.replace(
            "- Longer loan repayment term.",
            "- Longer loan repayment term.\n- High revolving credit utilization.",
        ),
        {"reasons_correct": 0, "fcra_window_correct": 1, "real_agency_named": 1,
         "ecoa_classes_correct": 1, "no_legal_errors": 1},
        "Adds a reason not present in the applicant's actual reason list "
        "(should state exactly the given reasons, no more, no fewer).",
    ),
}
