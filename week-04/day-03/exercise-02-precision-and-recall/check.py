"""The done check for exercise 2. In Databricks, cell 5 of the notebook runs it: it counts each
candidate's four cells at 0.5 again, divides, and compares that with your divisions and with what
each run recorded. Without Databricks, from the course folder:

    python3 week-04/day-03/exercise-02-precision-and-recall/check.py --start
    python3 week-04/day-03/exercise-02-precision-and-recall/check.py <facts.json>

reads the same facts from a JSON file (or the notebook's starting state). No model is called.
"""
import json
import sys
from pathlib import Path

NAMES = ("logistic-regression", "decision-tree")
COUNTS = {"logistic-regression": {"TP": 22, "FP": 16, "FN": 28, "TN": 934},
          "decision-tree": {"TP": 8, "FP": 5, "FN": 42, "TN": 945}}
START = {"guess": {"higher recall": "", "higher precision": ""},
         "by_hand": {n: {"precision": None, "recall": None} for n in NAMES},
         "expected": COUNTS, "runs": {}}


def plain(text) -> str:
    """A typed answer as the check reads it: lower case, with no spaces or full stop around it."""
    return str(text).strip().strip(".!").strip().lower().replace(" ", "-")


def measures(c: dict) -> dict:
    """Precision and recall from four counts; precision is None when the list is empty."""
    listed, change = c["TP"] + c["FP"], c["TP"] + c["FN"]
    return {"precision": c["TP"] / listed if listed else None, "recall": c["TP"] / change if change else None}


def near(a, b, by: float = 0.006) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) <= by


def left(f: dict) -> list[str]:
    """Return what is left: one line for each thing that is not yet right.

    Args:
        f: the facts: guess, by_hand (the four divisions of cell 3), expected (each candidate's four
            counts, counted again by this check) and runs (each run name's threshold, precision and
            recall, read back from the experiment).

    Returns:
        One line per problem; an empty list when the exercise is done.
    """
    out = []
    if any(plain(f["guess"].get(k, "")) not in NAMES for k in ("higher recall", "higher precision")):
        out.append("Cell 1: type logistic-regression or decision-tree for each guess, then run cell 1 again. Step 1.")
    typed_all = True
    for name in NAMES:
        want, typed = measures(f["expected"][name]), f["by_hand"].get(name, {})
        p, r = typed.get("precision"), typed.get("recall")
        if p is None or r is None:
            typed_all = False
            out.append(f"Cell 3: type the precision and the recall of {name} as divisions. Step 3.")
        elif near(p, want["recall"]) and near(r, want["precision"]):
            out.append(f"Cell 3: the precision and the recall of {name} are swapped. Precision divides by the "
                       "returns on the list; recall by the returns labelled change. Step 3.")
        elif not near(p, want["precision"]):
            out.append(f"Cell 3: the precision of {name} is not TP divided by TP + FP, from cell 2. Step 3.")
        elif not near(r, want["recall"]):
            out.append(f"Cell 3: the recall of {name} is not TP divided by TP + FN, from cell 2. Step 3.")
    for name in NAMES:
        run = f["runs"].get(name)
        typed = f["by_hand"].get(name, {})
        if not run or "precision" not in run.get("metrics", {}):
            if typed_all:
                out.append(f"No run {name} holds a precision yet. Run cell 4. Step 5.")
            continue
        if run.get("params", {}).get("threshold") != "0.5":
            out.append(f"The run {name} holds no threshold 0.5. Run cell 4 again. Step 5.")
        elif not (near(run["metrics"].get("precision"), typed.get("precision"), 0.0005)
                  and near(run["metrics"].get("recall"), typed.get("recall"), 0.0005)):
            out.append(f"The run {name} holds other numbers than cell 3. Run cells 3 and 4 again. Step 5.")
    return out


def report(f: dict) -> str:
    """Return what the check cell prints: Done, or Not yet with each thing left."""
    out = left(f)
    if not out:
        parts = []
        for name in NAMES:
            m = measures(f["expected"][name])
            parts.append(f"{name} precision {m['precision']:.2f}, recall {m['recall']:.2f}")
        return ("Done: at 0.5, " + "; ".join(parts) + ". Both runs hold them, with the threshold. "
                "Compare each with your guess.")
    return "\n".join([f"Not yet: {len(out)} left."] + ["- " + item for item in out])


def runs_of(g: dict) -> dict:
    """Each run name's newest run in the notebook's experiment: its params and metrics."""
    if "mlflow" not in g or "experiment" not in g:
        return {}
    runs = g["mlflow"].search_runs(experiment_names=[g["experiment"]])
    out = {}
    for _, run in (runs.iterrows() if len(runs) else []):
        name = run.get("tags.mlflow.runName")
        if name in NAMES and name not in out:
            out[name] = {"params": {"threshold": run.get("params.threshold")},
                         "metrics": {k: float(run[f"metrics.{k}"]) for k in ("precision", "recall")
                                     if f"metrics.{k}" in run and run[f"metrics.{k}"] == run[f"metrics.{k}"]}}
    return out


def facts_of(g: dict) -> dict:
    """The facts, from the notebook's values."""
    valid = g["valid"]
    expected = {}
    for name in NAMES:
        on = valid[name] >= 0.5
        change = valid["label"] == "change"
        expected[name] = {"TP": int((on & change).sum()), "FP": int((on & ~change).sum()),
                          "FN": int((~on & change).sum()), "TN": int((~on & ~change).sum())}
    number = lambda v: None if v is None else float(v)
    return {"guess": dict(g["guess"]),
            "by_hand": {n: {k: number(v) for k, v in g["by_hand"].get(n, {}).items()} for n in NAMES},
            "expected": expected, "runs": runs_of(g)}


def in_notebook(g: dict) -> str:
    """Return what the check cell prints, from the notebook's values (cell 5 passes them)."""
    return report(facts_of(g))


if __name__ == "__main__":
    given = START if sys.argv[1:] == ["--start"] else json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print(report(given))
