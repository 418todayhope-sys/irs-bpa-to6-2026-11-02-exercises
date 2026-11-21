# Exercise 2: Count both candidates' matrices, then precision and recall for each (we do)

**Goal:** After this exercise, you can compute each candidate's precision and recall at 0.5 from its four counts and set them beside its accuracy in its run.

**Scenario:** The examiners' manager asks two questions of each candidate's list: how many of the returns it sends would be worth an examiner's time, and how many of the returns worth the time it sends. A given cell counts both candidates' matrices on the 200 examined returns of `model_valid` at 0.5; you divide by hand and log the answers into each candidate's run from Tuesday.

**AI tool:** none.

**Time:** 15 minutes: 11 to do it, 4 for setup problems.

**Where you work:** this notebook, `week-04/day-03/exercise-02-precision-and-recall/exercise.ipynb`, in your Git folder of the course in Databricks, then the workspace's Experiments page.

**Done when:** cell 7 prints `Done:`.

**Cells:** a cell's number is the `# N.` comment on its first line, such as `# 1.`; text cells have no number.

**Optional part:** Finished early? Cell 8 checks your divisions with scikit-learn, then asks for the baseline's recall.

**What you have:**

- `course.returns.model_train` and `course.returns.model_valid`, as in exercise 1. If they cannot be read, cell 1 makes the same rows from the course's setup file and says so.
- Your experiment `week-04-candidates`, with Tuesday's two runs, `logistic-regression` and `decision-tree`. If a run is not there, cell 6 makes one of that name and says so.
- `load.py` and `check.py`, given code that cells 1 and 7 run.

**Start from:** this notebook as the course gives it, attached to the compute the trainer names. No cell changes your tables. New here: precision and recall, from each candidate's own counts.

### Step 1. Score the validation returns

Run cells 1 and 2. You do not need to change them. You should see:

1. Which tables cell 1 uses, then 500 returns in `model_train` and 250 in `model_valid`.
2. Cell 2 fits Tuesday's two candidates again, with Tuesday's lines, on the 400 examined returns of `model_train`. Then each one gives a score to every return of `model_valid`: `predict_proba` gives the score from 0 to 1, where Tuesday's `predict` cut it at 0.5. You should see 40 returns labelled `change`, 160 `no change` and 50 `never examined`.

A score is one return's number from 0 to 1: a relative risk indicator that a person acts on, never a finding about the taxpayer. A label records a past examination: `change` means it changed the tax, `exam_changed_tax` `yes`.

Notebook cell 1:

```python
# 1. Read the course's training and validation sets. You do not need to change this cell.
import os
course_folder = os.getcwd().split("/week-04/")[0]  # your Git folder of the course
folder = f"{course_folder}/week-04/day-03/exercise-02-precision-and-recall"
try:
    train_table = spark.table("course.returns.model_train")  # the training set: tax years 2020 and 2021
    valid_table = spark.table("course.returns.model_valid")  # the validation set: tax year 2022
except Exception:
    train_table, valid_table = None, None  # not there: load.py makes the same rows
exec(open(f"{folder}/load.py", encoding="utf-8").read())  # you do not need to read load.py
train, valid = load_sets(train_table, valid_table)
```

Notebook cell 2:

```python
# 2. Tuesday's two candidates, fitted again as Tuesday fitted them; each scores every return of model_valid.
assert "valid" in globals(), "Run cell 1 first."
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

features = ["filing_status", "wages_amt", "taxable_interest_amt", "other_income_amt",
            "total_income", "agi_amt", "deduction_amt", "deduction_missing",
            "total_tax_amt", "withholding_amt", "tax_to_agi", "deduction_to_agi"]  # Tuesday's twelve
examined = train[train["exam_changed_tax"].notna()]  # the 400 examined returns of model_train
X_train, y_train = examined[features].fillna(0), examined["exam_changed_tax"]
logistic = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_train, y_train)
tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X_train, y_train)

scores = valid[["case_id", "exam_changed_tax"]].copy()  # all 250 returns of model_valid
scores["label"] = scores["exam_changed_tax"].map({"yes": "change", "no": "no change"}).fillna("never examined")
scores["logistic_score"] = logistic.predict_proba(valid[features].fillna(0))[:, 1]  # column 1 is "yes"
scores["tree_score"] = tree.predict_proba(valid[features].fillna(0))[:, 1]
print(len(scores), "returns of model_valid scored by both candidates:", scores["label"].value_counts().to_dict())
```

### Step 2. Guess, then count both candidates' matrices

1. In cell 3, type your guess between the quotes: at 0.5, which candidate has the higher recall, and which the higher precision? Type `logistic-regression` or `decision-tree` in each. Run cell 3.
2. Run cell 4. It counts each candidate's four cells at 0.5 on the 200 examined returns, as exercise 1's code did. You should see, for each candidate, its four counts by name and the two sums: 200, and `TP + FN` 40.

Notebook cell 3:

```python
# 3. Your guess, before the counts: which candidate has the higher recall, and which the higher precision?
guess = {
    "higher recall": "",     # logistic-regression or decision-tree
    "higher precision": "",  # logistic-regression or decision-tree
}
print("Your guess:", guess)
```

Notebook cell 4:

```python
# 4. Each candidate's four counts at 0.5, on the 200 examined returns. You do not need to change this cell.
assert "scores" in globals(), "Run cells 1 and 2 first."
examined_scores = scores[scores["label"] != "never examined"]  # the 200 examined returns
change = examined_scores["label"] == "change"
counts = {}
for name, column in (("logistic-regression", "logistic_score"), ("decision-tree", "tree_score")):
    on = examined_scores[column] >= 0.5  # on the list at 0.5
    counts[name] = {"TP": int((on & change).sum()), "FP": int((on & ~change).sum()),
                    "FN": int((~on & change).sum()), "TN": int((~on & ~change).sum())}
if "" in guess.values():
    print("Type your guess in cell 3 and run it, then run this cell again.")
else:
    for name, c in counts.items():
        print(f"{name} at 0.5: {c}  Sum 1: {sum(c.values())}  Sum 2, TP + FN: {c['TP'] + c['FN']}")
```

### Step 3. Divide by hand

"*Precision* is the share of the returns on the list that are labelled *change*." "*Recall* is the share of the returns labelled *change* that are on the list." (the day's reading)

1. In cell 5, type each candidate's precision and recall in place of `None`, as a division of two counts from cell 4, such as `3 / 5`. The table says what to divide.

| Value | Type this | Why |
|---|---|---|
| `precision` | `TP / (TP + FP)`, with the candidate's two numbers from cell 4 | of the returns on the list, the share labelled change |
| `recall` | `TP / (TP + FN)`, with the candidate's two numbers from cell 4 | of the returns labelled change, the share on the list |

2. Run cell 5. You should see each division's result beside the candidate's accuracy on the same 200 returns, Tuesday's 0.955 and 0.895. Compare with your guess of cell 3.

Notebook cell 5:

```python
# 5. Precision and recall at 0.5, by hand. Type each as a division of two counts from cell 4, such as 3 / 5.
by_hand = {
    "logistic-regression": {"precision": None, "recall": None},
    "decision-tree": {"precision": None, "recall": None},
}
assert "counts" in globals(), "Run cell 4 first."
for name, measures in by_hand.items():
    c = counts[name]
    shown = {k: ("not yet" if v is None else round(v, 3)) for k, v in measures.items()}
    print(f"{name} at 0.5 | accuracy {(c['TP'] + c['TN']) / sum(c.values()):.3f} | precision {shown['precision']} | recall {shown['recall']}")
```

### Step 4. Log the measures into each candidate's run, and compare them

"write the threshold beside every precision and every recall" (the day's reading). Each run keeps its measures with the threshold they belong to.

1. Run cell 6. For each candidate, it opens the newest run of that name in `week-04-candidates`, Tuesday's run, and adds `threshold` 0.5 as a parameter and your `precision` and `recall` as metrics. You should see `Logged into Tuesday's run` twice.
2. In the workspace's left sidebar, click **Experiments**, then open `week-04-candidates`. Tick the newest run of each name and click **Compare**. You should see each run's `threshold`, `precision` and `recall` beside Tuesday's `validation_accuracy`: the same 200 returns.
3. Run cell 7. It prints `Done:`, or `Not yet:` and what is left. If you change a division, run cells 5, 6 and 7 again.

Notebook cell 6:

```python
# 6. Log the threshold, precision and recall into each candidate's run from Tuesday.
import mlflow
assert "by_hand" in globals(), "Run cell 5 first."
user = spark.sql("SELECT current_user()").first()[0]  # your user name in the workspace
experiment = f"/Users/{user}/week-04-candidates"      # Tuesday's experiment
mlflow.set_experiment(experiment)
runs = mlflow.search_runs(experiment_names=[experiment])  # every run, newest first
for name, measures in by_hand.items():
    if None in measures.values():
        print(f"Type the two divisions of {name} in cell 5 first.")
        continue
    found = runs[runs["tags.mlflow.runName"] == name] if len(runs) else runs
    run_id = found["run_id"].iloc[0] if len(found) else None  # the newest run of this name
    with mlflow.start_run(run_id=run_id, run_name=None if run_id else name):
        mlflow.log_param("threshold", 0.5)
        mlflow.log_metric("precision", measures["precision"])
        mlflow.log_metric("recall", measures["recall"])
    print("Logged into Tuesday's run" if run_id else "No run from Tuesday, so made a new run:", name,
          {k: round(v, 3) for k, v in measures.items()})
```

Notebook cell 7:

```python
# 7. Check. You do not need to read this code. It runs check.py, beside this notebook.
_check = {"__name__": "check"}
exec(open(f"{folder}/check.py", encoding="utf-8").read(), _check)
check_now = _check["in_notebook"]
print(check_now(globals()))
```

### Finished early?

Optional: cell 7 does not check it.

1. Run cell 8. scikit-learn computes each candidate's precision and recall at 0.5: compare them with yours.
2. In cell 8, type the baseline's recall in place of `None`, as a division. The baseline says no change for every return. Run cell 8 again. Why can its precision not be computed?

Notebook cell 8:

```python
# 8. Optional: scikit-learn's precision and recall, then the baseline's recall.
from sklearn.metrics import precision_score, recall_score
for name, column in (("logistic-regression", "logistic_score"), ("decision-tree", "tree_score")):
    on_list = ["change" if s >= 0.5 else "no change" for s in examined_scores[column]]
    print(name, "| precision", round(precision_score(examined_scores["label"], on_list, pos_label="change"), 3),
          "| recall", round(recall_score(examined_scores["label"], on_list, pos_label="change"), 3))
baseline_recall = None  # the baseline puts no return on the list
print("The baseline's recall, yours:", baseline_recall)
```

### Afterwards, together

1. Which candidate is ahead at 0.5, and on which measure?
2. Why can a precision not be computed for an empty list?
3. Why write the threshold beside every precision and recall?

The answer, after you try: [`answers/exercise-02-precision-and-recall`](../answers/exercise-02-precision-and-recall/).
