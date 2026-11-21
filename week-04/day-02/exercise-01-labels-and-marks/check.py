"""The done check for exercise 1. In Databricks, cell 8 of the notebook runs it: it counts the label
exam_changed_tax again in the training set that cell 2 loaded (course.returns.model_train, or the same
rows from the course's setup file), and reads your origin, your column of cell 5, your trace and your marks. Without Databricks,
from the course folder:

    python3 week-04/day-02/exercise-01-labels-and-marks/check.py --start
    python3 week-04/day-02/exercise-01-labels-and-marks/check.py <facts.json>

reads the same facts from a JSON file (or the notebook's starting state). No model is called.
"""
import json
import sys
from pathlib import Path

LABEL = "exam_changed_tax"
SUSPECT = "account_change_amt"
MARKS = {"tax_to_agi": "before", "wages_amt": "before", "deduction_amt": "before",
         "deduction_missing": "before", "tax_added_by_exam": "after", "account_change_amt": "after"}
MADE_BY = ("course", "synthetic", "made up", "made-up", "simulated", "fake", "not real", "generated", "invented")
START = {"schema": "course.learner_NN", "label_column": "refund_held",
         "origin": {"decision": "", "made by": "", "when": ""},
         "suspect": "refund_hold_days", "agreement_label": "refund_held", "agree": None, "labelled": None,
         "trace": {"which table holds it": "", "which step wrote it": "", "when": ""},
         "marks": {name: ["", ""] for name in MARKS}, "counts": {"no": 320, "yes": 80}, "empty": 100}


def left(f: dict) -> list[str]:
    """Return what is left: one line for each thing that is not yet right.

    Args:
        f: the facts: schema, label_column, origin, suspect, agreement_label (the label cell 5 used),
            agree and labelled (cell 5's counts), trace, marks, and the counts of exam_changed_tax in train.

    Returns:
        One line per problem; an empty list when the exercise is done.
    """
    out = []
    if f["schema"].endswith("_NN"):
        out.append("Cell 1 still names course.learner_NN. Type your own number. Step 1.")
    if f["label_column"] != LABEL:
        out.append(f"Cell 3 counts {f['label_column']}. Type {LABEL} in its place, then run cell 3. Step 3.")
    origin = {k: str(v).strip().lower() for k, v in f["origin"].items()}
    decision = origin.get("decision", "")
    missing = []
    if not ("exam" in decision or "audit" in decision or ("tax" in decision and "chang" in decision)):
        missing.append("decision (which decision: the examination's result)")
    if not any(word in origin.get("made by", "") for word in MADE_BY):
        missing.append("made by (who made these labels: the returns are made up)")
    if len(origin.get("when", "").split()) < 2:
        missing.append("when (two words or more)")
    if missing:
        out.append("Cell 4: answer " + "; ".join(missing) + ". Then run cell 4. Step 4.")
    if f["suspect"] != SUSPECT or f["agreement_label"] != LABEL:
        out.append(f"Cell 5 has not put {SUSPECT} beside {LABEL} yet. Type {SUSPECT} in cell 5, then run cells 3 "
                   "and 5. Step 5.")
    trace = {k: str(v).strip().lower() for k, v in f["trace"].items()}
    missing = []
    if "fields" not in trace.get("which table holds it", ""):
        missing.append("which table (here a file beside the notebook)")
    if not any(w in trace.get("which step wrote it", "") for w in ("exam", "audit")):
        missing.append("which step (what writes it)")
    if not any(w in trace.get("when", "") for w in ("after", "clos", "later", "month")):
        missing.append("when (before or after selection)")
    if missing:
        out.append("Cell 6: answer " + "; ".join(missing) + ". What you have, at the top, describes the column. "
                   "Then run cell 6. Step 6.")
    wrong = []
    for name, right in MARKS.items():
        mark, reason = (list(f["marks"].get(name, ["", ""])) + ["", ""])[:2]
        if str(mark).strip().lower().strip(".") != right or len(str(reason).split()) < 3:
            wrong.append(name)
    if wrong:
        out.append("Cell 7: give " + ", ".join(wrong) + " the mark its source says, before or after, and a "
                   "reason of three words or more. Then run cell 7. Step 7.")
    return out


def report(f: dict) -> str:
    """Return what the check cell prints: Done, or Not yet with each thing left."""
    out = left(f)
    if not out:
        counts = f["counts"]
        return (f"Done: {LABEL} counts no {counts.get('no', 0)}, yes {counts.get('yes', 0)}, and {f['empty']} empty, "
                f"not counted. {SUSPECT} agrees with it on {f['agree']} of {f['labelled']} returns, was written after "
                "selection, and is marked after, with tax_added_by_exam. Train on the four features marked before.")
    return "\n".join([f"Not yet: {len(out)} left."] + ["- " + item for item in out])


def in_notebook(schema: str, values: dict) -> str:
    """Count the label in the training set the notebook uses, then return what the check cell prints.

    Args:
        schema: the learner's schema, such as course.learner_07.
        values: train (the DataFrame cell 2 loaded), label_column, origin, suspect, agreement_label,
            agree, labelled, trace and marks, as the cells hold them.
    """
    values = dict(values)
    train = values.pop("train")
    facts = dict(START, schema=schema, **values)
    facts["marks"] = {k: list(v) if isinstance(v, (tuple, list)) else [v, ""] for k, v in facts["marks"].items()}
    labels = train[LABEL]
    facts["counts"] = {str(k): int(v) for k, v in labels[labels.notna()].value_counts().items()}
    facts["empty"] = int(labels.isna().sum())
    return report(facts)


if __name__ == "__main__":
    given = START if sys.argv[1:] == ["--start"] else json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print(report(given))
