"""The done check for exercise 1. In Databricks, cell 6 of the notebook runs it: it reads the ranked
list and case #123's chart from the notebook's model, and compares them with what you typed. Without
Databricks, from the course folder:

    python3 week-04/day-04/exercise-01-explain-case-123/check.py --start
    python3 week-04/day-04/exercise-01-explain-case-123/check.py <facts.json>

reads the same facts from a JSON file (or the notebook's starting state). No model is called.
"""
import json
import re
import sys
from pathlib import Path

FEATURES = ("tax_to_agi", "wages_change_pct", "withholding_amt", "withholding_missing")
# The logistic regression's ranked list and case #123's chart, as the notebook computes them.
START = {"ranked": ["tax_to_agi", "wages_change_pct", "withholding_missing", "withholding_amt"],
         "chart": {"average": 0.05000, "score": 0.82119,
                   "pushes": {"tax_to_agi": 0.39991, "wages_change_pct": 0.29000, "withholding_amt": -0.03903,
                              "withholding_missing": 0.12032}},
         "top_three": ["", "", ""], "axis": "", "average": None, "score": None,
         "pushes": {f: None for f in FEATURES}, "sentences": ""}
FORM_1 = re.compile(r"scored case #?123 at (\d*\.\d+),? against an average of (\d*\.\d+) for all returns", re.I)
FORM_2 = re.compile(r"two largest reasons were (.+?),? which (?:raised|lowered) the score by (\d*\.\d+),? and (.+?),? "
                    r"which (?:raised|lowered) it by (\d*\.\d+)", re.I | re.S)
FINDING = re.compile(r"underreport|under-report|fraud|evad|evasion|cheat|\blied\b|\bhid\b|guilty|"
                     r"(tax|return) is wrong|wrong tax", re.I)


def near(a, b, by: float = 0.006) -> bool:
    """Whether a typed number is the expected one, to two decimals."""
    try:
        return abs(float(a) - float(b)) <= by
    except (TypeError, ValueError):
        return False


def left(f: dict) -> list[str]:
    """Return what is left: one line for each thing that is not yet right.

    Args:
        f: the facts: ranked (the model's features in order), chart (case #123's average, pushes and
            score), and what the learner typed in cells 2, 4 and 5.

    Returns:
        One line per problem; an empty list when the exercise is done.
    """
    out = []
    typed = [str(t).strip().strip("`'\"") for t in (f["top_three"] or [])]
    if typed != list(f["ranked"][:3]):
        out.append("Cell 2: type the three longest bars of the ranked list, the longest first, as the chart names "
                   "them. The ranked list is the whole model's chart, not case #123's. Step 2.")
    axis = str(f["axis"]).lower()
    if not re.search(r"score|probabilit", axis) or "log" in axis:
        out.append("Cell 4: type what the chart's axis is measured in: its label is printed under cell 3. Step 4.")
    chart = f["chart"]
    if not near(f["average"], chart["average"]):
        out.append("Cell 4: the average score is the first line of cell 3's table. Step 4.")
    pushes = f["pushes"] or {}
    wrong = [name for name in FEATURES if not near(pushes.get(name), chart["pushes"][name])]
    if wrong:
        out.append(f"Cell 4: read the push of {', '.join(wrong)} again from cell 3, with its sign. Step 4.")
    if not near(f["score"], chart["score"]):
        out.append("Cell 4: case #123's score is the last line of cell 3's table. Step 4.")
    text = str(f["sentences"])
    first, second = FORM_1.search(text), FORM_2.search(text)
    largest = sorted(chart["pushes"].values(), key=lambda v: -abs(v))[:2]
    if not first:
        out.append("Cell 5: write the first sentence in the form: The model scored case #123 at ..., against an "
                   "average of ... for all returns. Step 5.")
    elif not (near(first.group(1), chart["score"]) and near(first.group(2), chart["average"])):
        out.append("Cell 5: the first sentence gives case #123's score and the average score, from cell 3. Step 5.")
    if not second:
        out.append("Cell 5: write the second sentence in the form: The two largest reasons were ..., which raised the "
                   "score by ..., and ..., which raised it by .... Step 5.")
    else:
        if not (near(second.group(2), abs(largest[0])) and near(second.group(4), abs(largest[1]))):
            out.append("Cell 5: the second sentence gives the two largest pushes, the largest first, as amounts of "
                       "score. Step 5.")
        if min(len(second.group(1).split()), len(second.group(3).split())) < 2 or "..." in text:
            out.append("Cell 5: name each reason in two words or more, in words an examiner reads. Step 5.")
    if FINDING.search(text):
        out.append("Cell 5: a push says where the model looked, not what the taxpayer did. Leave out any finding "
                   "about the taxpayer. Step 5.")
    return out


def report(f: dict) -> str:
    """Return what the check cell prints: Done, or Not yet with each thing left."""
    out = left(f)
    if not out:
        return (f"Done: your top three are {', '.join(f['ranked'][:3])}; case #123's pushes add up from "
                f"{f['chart']['average']:.2f} to {f['chart']['score']:.2f}, and your two sentences give the two "
                "largest in the axis's unit.")
    return "\n".join([f"Not yet: {len(out)} left."] + ["- " + item for item in out])


def facts_of(g: dict) -> dict:
    """The facts, from the notebook's values: the model's ranked list and case #123's chart."""
    chart = g["numbers"]("123")
    return {"ranked": list(g["weights"]().index),
            "chart": {"average": chart["average"], "score": chart["score"], "pushes": chart["pushes"]},
            **{name: g.get(name) for name in ("top_three", "axis", "average", "score", "pushes", "sentences")}}


def in_notebook(g: dict) -> str:
    """Return what the check cell prints, from the notebook's values (cell 6 passes them)."""
    return report(facts_of(g))


if __name__ == "__main__":
    given = START if sys.argv[1:] == ["--start"] else json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print(report(given))
