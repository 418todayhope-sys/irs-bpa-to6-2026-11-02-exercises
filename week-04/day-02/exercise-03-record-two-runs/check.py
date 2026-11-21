"""The done check for exercise 3. In Databricks, cell 8 of the notebook runs it: it reads your
experiment's runs back from MLflow (the newest run of each name), and compares them with the two
candidates cell 2 fitted and the name you typed in cell 7. Without Databricks, from the course folder:

    python3 week-04/day-02/exercise-03-record-two-runs/check.py --start
    python3 week-04/day-02/exercise-03-record-two-runs/check.py <facts.json>

reads the same facts from a JSON file (or the notebook's starting state). No model is called.
"""
import json
import sys
from pathlib import Path

RUNS = {"logistic-regression": "LogisticRegression", "decision-tree": "DecisionTreeClassifier"}
START = {"experiment": "/Users/<you>/week-04-candidates", "runs": {},
         "accuracy": {"logistic-regression": 0.955, "decision-tree": 0.895}, "higher": ""}


def left(f: dict) -> list[str]:
    """Return what is left: one line for each thing that is not yet right.

    Args:
        f: the facts: runs (the newest run of each name: its params and its metrics), accuracy
            (each candidate's, as cell 2 fitted it) and higher (the run name typed in cell 7).

    Returns:
        One line per problem; an empty list when the exercise is done.
    """
    out = []
    for name, model_type in RUNS.items():
        run = f["runs"].get(name)
        if run is None:
            line = "as given" if name == "logistic-regression" else "with the run name decision-tree"
            out.append(f"No run named {name} in your experiment yet. Run cell 5 {line}. Step 2.")
            continue
        p, m = run["params"], run["metrics"]
        if (p.get("model_type") != model_type or not p.get("split") or p.get("measure") != "accuracy"
                or (name == "decision-tree" and str(p.get("max_depth")) != "3")):
            out.append(f"The run {name} does not record its model type, split and measure as cell 5 gives them. "
                       "Run cell 5 again for it. Step 2.")
        elif m.get("validation_accuracy") is None or abs(m["validation_accuracy"] - f["accuracy"][name]) > 0.0005:
            out.append(f"The run {name} records another accuracy than its candidate's. Run cells 2 and 5 again for "
                       "it. Step 2.")
    recorded = {n: r["metrics"].get("validation_accuracy") for n, r in f["runs"].items() if n in RUNS}
    values = [v for v in recorded.values() if v is not None]
    best = [n for n, v in recorded.items() if v is not None and v == max(values)] if values else []
    if str(f["higher"]).strip() not in best or len(values) < 2:
        out.append("Cell 7: type the name of the run with the higher validation accuracy in cell 6. Step 5.")
    return out


def report(f: dict) -> str:
    """Return what the check cell prints: Done, or Not yet with each thing left."""
    out = left(f)
    if not out:
        acc = {n: f["runs"][n]["metrics"]["validation_accuracy"] for n in RUNS}
        higher = str(f["higher"]).strip()
        other = [n for n in RUNS if n != higher][0]
        return (f"Done: two runs in {f['experiment']}, each with its split and its measure. The runs record "
                f"{acc[higher]:.3f} for {higher} and {acc[other]:.3f} for {other}: you named {higher}, the higher. "
                "Wednesday judges the two.")
    return "\n".join([f"Not yet: {len(out)} left."] + ["- " + item for item in out])


def facts_of(g: dict) -> dict:
    """The facts, from the notebook's values and the runs MLflow holds."""
    import mlflow
    runs = mlflow.search_runs(experiment_names=[g["experiment"]])  # newest first
    newest = {}
    for _, row in runs.iterrows() if len(runs) else []:
        name = row.get("tags.mlflow.runName")
        if name in newest:
            continue
        newest[name] = {"params": {k[7:]: v for k, v in row.items() if k.startswith("params.") and isinstance(v, str)},
                        "metrics": {k[8:]: float(v) for k, v in row.items() if k.startswith("metrics.") and v == v}}
    return {"experiment": g["experiment"], "runs": newest,
            "accuracy": {n: float(a) for n, (_, a) in g["candidates"].items()}, "higher": g.get("higher", "")}


def in_notebook(g: dict) -> str:
    """Return what the check cell prints, from the notebook's values (cell 8 passes them)."""
    return report(facts_of(g))


if __name__ == "__main__":
    given = START if sys.argv[1:] == ["--start"] else json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print(report(given))
