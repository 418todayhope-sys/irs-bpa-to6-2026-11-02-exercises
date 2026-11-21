"""The done check for exercise 1. In Databricks, cell 7 of the notebook runs it: it sorts the ten
returns at 0.5 and counts every validation return again, and compares what you typed with that.
Without Databricks, from the course folder:

    python3 week-04/day-03/exercise-01-four-cells/check.py --start
    python3 week-04/day-03/exercise-01-four-cells/check.py <facts.json>

reads the same facts from a JSON file (or the notebook's starting state). No model is called.
"""
import json
import sys
from pathlib import Path

CELLS = ("TP", "FP", "FN", "TN")
NAMES = {"tp": "TP", "true positive": "TP", "fp": "FP", "false positive": "FP",
         "fn": "FN", "false negative": "FN", "tn": "TN", "true negative": "TN"}
PLACES = {"on": "on", "on the list": "on", "off": "off", "off the list": "off"}
TEN = {"#311": (0.91, "change"), "#388": (0.84, "change"), "#458": (0.74, "no change"), "#206": (0.63, "change"),
       "#270": (0.55, "no change"), "#502": (0.46, "change"), "#189": (0.38, "no change"),
       "#644": (0.27, "no change"), "#135": (0.15, "no change"), "#717": (0.08, "no change")}
START = {"threshold": 0.5, "ten": {k: list(v) for k, v in TEN.items()}, "my_sort": {k: ["", ""] for k in TEN},
         "my_counts": {c: None for c in CELLS}, "meaning": {c: "" for c in CELLS},
         "all_counts": {c: None for c in CELLS}, "expected_all": {"TP": 22, "FP": 16, "FN": 28, "TN": 934}}


def plain(text) -> str:
    """A typed answer as the check reads it: lower case, with no spaces or full stop around it."""
    return str(text).strip().strip(".!").strip().lower()


def sort_one(score: float, label: str, threshold: float) -> tuple[str, str]:
    """Where one return lands at the threshold, and its cell. A score equal to the threshold goes on."""
    on = score >= threshold
    change = label == "change"
    return ("on" if on else "off"), {(True, True): "TP", (True, False): "FP",
                                     (False, True): "FN", (False, False): "TN"}[(on, change)]


def left(f: dict) -> list[str]:
    """Return what is left: one line for each thing that is not yet right.

    Args:
        f: the facts: threshold, ten (each return's score and label), my_sort, my_counts, meaning,
            all_counts, and expected_all (every validation return's four counts, counted again).

    Returns:
        One line per problem; an empty list when the exercise is done.
    """
    out = []
    if f["threshold"] != 0.5:
        out.append("Cell 2's threshold must stay 0.5. Run cells 2 to 6 again. Step 2.")
    wrong_place, wrong_cell, blank = [], [], []
    for rid, (score, label) in f["ten"].items():
        place, cell = (list(f["my_sort"].get(rid) or ["", ""]) + ["", ""])[:2]
        if not plain(place) or not plain(cell):
            blank.append(rid)
            continue
        want_place, want_cell = sort_one(score, label, 0.5)
        if PLACES.get(plain(place)) != want_place:
            wrong_place.append(rid)
        elif NAMES.get(plain(cell)) != want_cell:
            wrong_cell.append(rid)
    if blank:
        out.append(f"Cell 2: type on or off, then the cell, for {', '.join(blank)}. Step 2.")
    if wrong_place:
        out.append(f"Cell 2: look again at where {', '.join(wrong_place)} landed: is its score at or above 0.5? Step 2.")
    if wrong_cell:
        out.append(f"Cell 2: look again at the cell of {', '.join(wrong_cell)}: where did it land, and what does its "
                   "label say? Step 2.")
    want = {c: 0 for c in CELLS}
    for score, label in f["ten"].values():
        want[sort_one(score, label, 0.5)[1]] += 1
    if any(f["my_counts"].get(c) is None for c in CELLS):
        out.append("Cell 3: type the four counts of your sort. Step 3.")
    elif {c: f["my_counts"][c] for c in CELLS} != want:
        out.append("Cell 3: a count does not match the ten returns at 0.5. Check the two sums, then cell 2. Step 3.")
    short = [c for c in CELLS if len(str(f["meaning"].get(c, "")).split()) < 4]
    if short:
        out.append(f"Cell 4: write what {', '.join(short)} means for an examiner, four words or more. Step 4.")
    if any(f["all_counts"].get(c) is None for c in CELLS):
        out.append("Cell 6: type the four counts of cell 5's table. Step 6.")
    elif {c: f["all_counts"][c] for c in CELLS} != f["expected_all"]:
        out.append("Cell 6: a count is not where the table puts it. Its rows are the labels: first change, then "
                   "no change. Check the two sums. Step 6.")
    return out


def report(f: dict) -> str:
    """Return what the check cell prints: Done, or Not yet with each thing left."""
    out = left(f)
    if not out:
        m, a = f["my_counts"], f["all_counts"]
        return (f"Done: at 0.5, the ten returns are {m['TP']} TP, {m['FP']} FP, {m['FN']} FN, {m['TN']} TN; "
                f"every validation return is {a['TP']} TP, {a['FP']} FP, {a['FN']} FN, {a['TN']} TN. "
                "Both sums hold on both.")
    return "\n".join([f"Not yet: {len(out)} left."] + ["- " + item for item in out])


def facts_of(g: dict) -> dict:
    """The facts, from the notebook's values."""
    valid, threshold = g["valid"], 0.5
    expected = {c: 0 for c in CELLS}
    for score, label in zip(valid["logistic-regression"], valid["label"]):
        expected[sort_one(float(score), label, threshold)[1]] += 1
    number = lambda v: None if v is None else int(v)
    return {"threshold": g["threshold"],
            "ten": {r: [float(s), l] for r, s, l in zip(g["ten"]["return_id"], g["ten"]["score"], g["ten"]["label"])},
            "my_sort": {k: list(v) for k, v in g["my_sort"].items()},
            "my_counts": {c: number(g["my_counts"].get(c)) for c in CELLS},
            "meaning": dict(g["meaning"]),
            "all_counts": {c: number(g["all_counts"].get(c)) for c in CELLS},
            "expected_all": expected}


def in_notebook(g: dict) -> str:
    """Return what the check cell prints, from the notebook's values (cell 7 passes them)."""
    return report(facts_of(g))


if __name__ == "__main__":
    given = START if sys.argv[1:] == ["--start"] else json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print(report(given))
