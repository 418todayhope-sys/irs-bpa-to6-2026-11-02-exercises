# Exercise 2: Count both candidates' matrices, then precision and recall for each

**Goal:** After this exercise, you can compute each candidate's precision and recall at 0.5 from its four counts and set them beside its accuracy in its run.

**Scenario:** The made-up examination group's manager asks two questions of each candidate's list: how many of the returns it sends are worth an examiner's time, and how many of the returns worth the time it sends. You count both candidates' matrices at 0.5, divide by hand, and log the answers into each candidate's run.

**AI tool:** none.

**Time:** 25 minutes: 21 to do it, 4 for setup problems.

**Where you work:** this notebook, `week-04/day-03/exercise-02-precision-and-recall/exercise.ipynb`, in your Git folder of the course in Databricks, then the workspace's Experiments page.

**Done when:** cell 5 prints `Done:`.

**Optional part:** Finished early? Cell 6 checks your divisions with scikit-learn.

**What you have:**

- `validation-scores.csv`: the 1,000 made-up validation returns of exercise 1, each with its label and its score from each of Tuesday's two candidates.
- Tuesday's two runs, `logistic-regression` and `decision-tree`, in your experiment `week-04-candidates`. If they are not there, cell 4 makes them and tells you.
- `load.py` and `check.py`: given code that cells 1 and 5 run.

**Start from:** this notebook as the course gives it, attached to the compute the trainer names. No cell changes your tables. New here: precision and recall.

### Step 1. Read the validation returns, then guess

Why: you understand a result when you said what you expected first.

1. Run cell 1. You do not need to read its first three lines. You should see `Read 1000 made-up validation returns`.
2. In cell 1, type your guess between the quotes: which candidate has the higher recall, and which the higher precision? Type `logistic-regression` or `decision-tree` in each.
3. Run cell 1 again. You should see your guess.

Notebook cell 1:

```python
# 1. The 1,000 made-up validation returns, then your guess.
import os
folder = os.getcwd().split("/week-04/")[0] + "/week-04/day-03/exercise-02-precision-and-recall"
exec(open(f"{folder}/load.py", encoding="utf-8").read())  # reads the file beside this notebook

guess = {
    "higher recall": "",     # logistic-regression or decision-tree
    "higher precision": "",  # logistic-regression or decision-tree
}
print("Your guess:", guess)
```

### Step 2. Count both candidates' matrices

Why: precision and recall are each one division on the four counts.

Run cell 2. It counts each candidate's four cells at 0.5 with exercise 1's counting code, once for each candidate. You should see, for each candidate, scikit-learn's table, then its four counts by name and the two sums. Sum 1 is 1000, and sum 2, `TP + FN`, is the 50 returns labelled change.

Notebook cell 2:

```python
# 2. The counting code of exercise 1, for each candidate, at 0.5.
from sklearn.metrics import confusion_matrix
threshold = 0.5
counts = {}
for name in ("logistic-regression", "decision-tree"):
    on_list = ["change" if score >= threshold else "no change" for score in valid[name]]
    table = confusion_matrix(valid["label"], on_list, labels=["change", "no change"])
    (tp, fn), (fp, tn) = table  # rows: the label, change then no change
    counts[name] = {"TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}
    print(name, "at", threshold)
    print(table)
    print("  ", counts[name], " Sum 1:", table.sum(), " Sum 2, TP + FN:", tp + fn)
```

### Step 3. Divide by hand

Why: "*Precision* is the share of the returns on the list that are labelled *change*." "*Recall* is the share of the returns labelled *change* that are on the list." (the day's reading)

| | You divide | Example: the ten returns at 0.5 |
|---|---|---|
| Precision | `TP` by all the returns on the list, `TP + FP` | `3 / 5` |
| Recall | `TP` by all the returns labelled change, `TP + FN` | `3 / 4` |

1. In cell 3, type each candidate's precision and recall in place of `None`, as a division of two counts from cell 2, such as `3 / 5`. A result such as `0.6` passes too, but the division shows what you divided by.
2. Run cell 3. You should see each division's result, beside the candidate's accuracy on the same returns.

Notebook cell 3:

```python
# 3. Precision and recall at 0.5, by hand. Type each as a division of two counts, such as 3 / 5.
by_hand = {
    "logistic-regression": {"precision": None, "recall": None},
    "decision-tree": {"precision": None, "recall": None},
}
for name, measures in by_hand.items():
    c = counts[name]
    accuracy = (c["TP"] + c["TN"]) / sum(c.values())
    shown = {k: (round(v, 3) if v is not None else "not yet") for k, v in measures.items()}
    print(name, "| accuracy", round(accuracy, 3), "| precision", shown["precision"], "| recall", shown["recall"])
```

### Step 4. Compare with your guess

Look at cell 3's two lines beside your guess of cell 1. This step has no cell.

- Which candidate picks out more of the 50 returns labelled change? That is the higher recall.
- Which candidate's list holds the larger share of returns labelled change? That is the higher precision.
- The accuracies are almost the same. Precision and recall are not.

### Step 5. Log the measures into each candidate's run

Why: "the comparison still comes from the record" (the day's reading). Each run keeps its measures, with the threshold they belong to.

Run cell 4. For each candidate, it finds Tuesday's run of that name and adds:

1. `log_param("threshold", 0.5)`: the threshold, a parameter.
2. `log_metric` three times: `precision` and `recall`, your divisions of cell 3, and `accuracy_same_rows`, the accuracy on the same 1,000 returns.
3. `log_param("rows", ...)`: the returns they were counted on.

The `validation_accuracy` already in each run was measured on Tuesday, on your own week 3 validation rows, not on these 1,000 made-up returns. So these three keep names of their own.

You should see `Logged into Tuesday's run` twice, with each run's measures. If a line says `made a new run`, Tuesday's run was not found; that is fine.

Notebook cell 4:

```python
# 4. Log precision, recall and the threshold into each candidate's run from Tuesday.
import mlflow
assert None not in [v for m in by_hand.values() for v in m.values()], "Type the four divisions of cell 3 first."
user = spark.sql("SELECT current_user()").first()[0]  # your user name in the workspace
experiment = f"/Users/{user}/week-04-candidates"
mlflow.set_experiment(experiment)
runs = mlflow.search_runs(experiment_names=[experiment])  # every run, newest first
for name, measures in by_hand.items():
    found = runs[runs["tags.mlflow.runName"] == name] if len(runs) else runs
    run_id = found["run_id"].iloc[0] if len(found) else None  # Tuesday's run, the newest of this name
    c = counts[name]
    with mlflow.start_run(run_id=run_id, run_name=None if run_id else name):
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("rows", "the course's 1,000 made-up validation returns")
        mlflow.log_metric("precision", measures["precision"])
        mlflow.log_metric("recall", measures["recall"])
        mlflow.log_metric("accuracy_same_rows", (c["TP"] + c["TN"]) / sum(c.values()))
    shown = {k: round(v, 3) for k, v in measures.items()}
    print("Logged into Tuesday's run" if run_id else "No run from Tuesday: made a new run", name, shown)
```

### Step 6. Read the measures side by side on the Experiments page

1. In the workspace's left sidebar, select **Experiments**, then open `week-04-candidates`.
2. Select the boxes of the newest run named `logistic-regression` and the newest named `decision-tree`, then select **Compare**. You should see each run's `threshold`, `precision`, `recall` and `accuracy_same_rows`, the numbers of cell 3.

This step has no cell.

### Step 7. Check

Run cell 5. It prints `Done:` when your guess is typed, your four divisions are right, and each run holds them with the threshold. Otherwise it prints `Not yet:` and what is left. If you change a division, run cells 3, 4 and 5 again.

Keep this notebook. Exercise 3 uses these measures.

Notebook cell 5:

```python
# 5. Check. You do not need to read this code. It runs check.py, beside this notebook.
_check = {"__name__": "check"}
exec(open(f"{folder}/check.py", encoding="utf-8").read(), _check)
check_now = _check["in_notebook"]
print(check_now(globals()))
```

### Finished early?

Optional: cell 5 does not check it.

1. Run cell 6. scikit-learn computes each candidate's precision and recall: compare them with yours.
2. In cell 6, type the baseline's recall in place of `None`, as a division. The baseline says no change for every return. Run cell 6 again. Why can its precision not be computed?

Notebook cell 6:

```python
# 6. Optional: scikit-learn's precision and recall, then the baseline's recall.
from sklearn.metrics import precision_score, recall_score
for name in ("logistic-regression", "decision-tree"):
    on_list = ["change" if score >= threshold else "no change" for score in valid[name]]
    print(name, "| precision", round(precision_score(valid["label"], on_list, pos_label="change"), 3),
          "| recall", round(recall_score(valid["label"], on_list, pos_label="change"), 3))
baseline_recall = None  # the baseline puts no return on the list
print("The baseline's recall, yours:", baseline_recall)
```

### Afterwards, together

1. Which candidate would you send to the examiners if a missed return costs more?
2. Why can a precision not be computed for an empty list?
3. Why write the threshold beside every precision and recall?

The answer, after you try: [`answers/exercise-02-precision-and-recall`](../answers/exercise-02-precision-and-recall/).
