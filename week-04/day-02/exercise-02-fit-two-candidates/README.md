# Exercise 2: Fit two candidates and compare each with the baseline

**Goal:** After this exercise, you can fit two candidates from given cells and compare each one's validation accuracy with the baseline's.

**Scenario:** Your labels are counted and your features are marked. Now you train. You fit a logistic regression, change one line to fit a decision tree, and set both accuracies beside the baseline's, on the same validation rows.

**AI tool:** none.

**Time:** 25 minutes: 18 to do it, 7 for setup problems.

**Where you work:** this notebook, `week-04/day-02/exercise-02-fit-two-candidates/exercise.ipynb`, in your Git folder of the course in Databricks.

**Done when:** cell 8 prints `Done:`.

**Optional part:** Finished early? Cell 9 asks for one sentence.

**What you have:**

- `course.returns.model_train` and `course.returns.model_valid`, the course's training and validation sets, as on Monday. If they cannot be read, cell 2 makes the same rows from the course's setup file and says so. Cell 2 adds Monday's flag `deduction_missing`.
- `load.py` and `check.py`, given code that cells 2 and 8 run.

**Start from:** this notebook as the course gives it, attached to the compute the trainer names. No cell changes your tables. New here: fitting a model.

### Step 1. Find the course's files

Why: cell 2 runs a file beside this notebook, in your Git folder of the course.

Run cell 1. You should see your Git folder of the course.

Notebook cell 1:

```python
# 1. Where the course's files are. You do not need to change this cell.
import os
course_folder = os.getcwd().split("/week-04/")[0]  # your Git folder of the course
print("Your Git folder of the course:", course_folder)
```

### Step 2. Read the training and validation sets

Why: a candidate is fitted on the training rows and measured on the validation rows.

Run cell 2. You do not need to read its code. You should see which tables it uses, then how many returns are in each.

Notebook cell 2:

```python
# 2. Read the course's training and validation sets, with Monday's flag deduction_missing.
try:
    train_table = spark.table("course.returns.model_train")  # the training set: tax years 2020 and 2021
    valid_table = spark.table("course.returns.model_valid")  # the validation set: tax year 2022
except Exception:
    train_table, valid_table = None, None  # not there: load.py makes the same rows

# You do not need to read the lines below. They run load.py, beside this notebook.
folder = f"{course_folder}/week-04/day-02/exercise-02-fit-two-candidates"
exec(open(f"{folder}/load.py", encoding="utf-8").read())
train, valid = load_sets(train_table, valid_table)
```

### Step 3. Keep the features known `before`, and read an empty amount as 0

Why: "Train only on the columns marked *before*." (the day's reading) These twelve are the filing status, every amount of the filed return, two ratios built from them, and Monday's flag: each known before selection. A model cannot read an empty value, so an empty amount is read as 0. In the course's data only the deduction line is ever empty, and `deduction_missing` says which deduction lines were, as on Monday.

Run cell 3. It keeps the returns with a label, puts the features into `X` and the label into `y`, and reads each empty amount as 0. You should see how many rows have a label, then how many deduction lines were empty.

Notebook cell 3:

```python
# 3. The features known before selection, and the label. An empty amount is read as 0.
assert "train" in globals(), "Run cells 1 and 2 first."
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

features = ["filing_status", "wages_amt", "taxable_interest_amt", "other_income_amt",
            "total_income", "agi_amt", "deduction_amt", "deduction_missing",
            "total_tax_amt", "withholding_amt", "tax_to_agi", "deduction_to_agi"]  # each known before selection
train = train[train["exam_changed_tax"].notna()]  # leave out the returns with an empty label
valid = valid[valid["exam_changed_tax"].notna()]
X_train, y_train = train[features].fillna(0), train["exam_changed_tax"]  # an empty amount read as 0;
X_valid, y_valid = valid[features].fillna(0), valid["exam_changed_tax"]  # deduction_missing says which
print(len(X_train), "training rows and", len(X_valid), "validation rows have a label")
print("Deduction lines read as 0 because they were empty:", int(train["deduction_missing"].sum()), "in training,",
      int(valid["deduction_missing"].sum()), "in validation")
```

### Step 4. Measure the baseline again, on the validation rows

Why: "A comparison is fair only on the same rows." (the day's reading)

Run cell 4. It measures Monday's baseline, always `no`, on the validation rows. You should see:

1. How many validation returns have a label. One more return right moves the accuracy by 1 divided by that number.
2. How many of them are `no` and `yes`.
3. The baseline's accuracy Monday's way, on the training rows, then on the validation rows.

Notebook cell 4:

```python
# 4. The baseline, always "no", measured again on the validation rows.
assert "y_valid" in globals(), "Run cell 3 first."
n = len(y_valid)
print(n, "validation returns have a label. One more right moves the accuracy by", round(1 / n, 3))
print("Their labels:", y_valid.value_counts().to_dict())
baseline_accuracy = accuracy_score(y_valid, ["no"] * n)
results = {**globals().get("results", {}), "baseline": baseline_accuracy}  # keeps any fit of cell 6
fitted = globals().get("fitted", {})
print("The baseline on the training rows, Monday's way:", round((y_train == "no").mean(), 3))
print("The baseline on the validation rows:", round(baseline_accuracy, 3))
```

### Step 5. Guess, before you fit

Why: you understand a result when you said what you expected first.

1. In cell 5, for each candidate, type your guess: will it beat the baseline of cell 4?
   - `a lot`: twenty or more returns more right than the baseline, of 200.
   - `a little`: one to nineteen returns more right.
   - `not better`: the same or fewer. A tie is `not better`.
2. Run cell 5.

Notebook cell 5:

```python
# 5. Your guess: will each candidate beat the baseline? Type "a lot", "a little" or "not better".
guess = {
    "LogisticRegression": "",
    "DecisionTreeClassifier": "",
}
print("Your guess:", guess)
```

### Step 6. Fit two candidates

Why: fit trains the model on the training rows; predict asks it for an outcome on each validation return.

1. Run cell 6. It fits a logistic regression, with the amounts put on a common scale first: `StandardScaler` does that, and `make_pipeline` joins the two steps into one model. You should see its validation accuracy, then how many returns it got right.
2. In cell 6, change the one line that names the model. Type this in place of `make_pipeline(StandardScaler(), LogisticRegression())`:

```python
DecisionTreeClassifier(max_depth=3, random_state=0)
```

`max_depth=3` is a setting: a tree at most three questions deep. `random_state=0` makes every run grow the same tree.

3. Run cell 6 again. You should see the tree's accuracy and how many it got right.

Run cell 6 once for each model, by itself. Do not use Run all after the change: it would fit only the tree.

Notebook cell 6:

```python
# 6. Fit a candidate on the training rows, then measure its accuracy on the validation rows.
assert "results" in globals(), "Run cells 3 and 4 first."
model = make_pipeline(StandardScaler(), LogisticRegression())  # the one line that names the model
model.fit(X_train, y_train)                      # fit: training, on the training rows
predictions = model.predict(X_valid)             # predict: one outcome for each validation return
accuracy = accuracy_score(y_valid, predictions)  # accuracy: the share of predictions that match the label
name = type(model[-1] if hasattr(model, "steps") else model).__name__  # its type, inside the pipeline
results[name], fitted[name] = accuracy, model
print(name, "validation accuracy:", round(accuracy, 3))
print("Right:", int((predictions == y_valid).sum()), "of", len(y_valid), "validation returns")
```

### Step 7. Set the three numbers side by side

Why: each candidate is read against the baseline, on the same rows.

Run cell 7. You should see three accuracies. Compare them with your guess.

- A large gain is the moment to ask whether a column gave the label away. Every column of cell 3 was known before selection, so the gain comes from what was filed.
- A candidate that does not beat the baseline is a result too: keep it.
- A candidate close to 1.000 is a warning: suspect a leak.

Notebook cell 7:

```python
# 7. Three numbers side by side, on the same validation rows.
assert "results" in globals(), "Run cells 3 and 4 first."
for name in ("baseline", "LogisticRegression", "DecisionTreeClassifier"):
    if name in results:
        print(f"{name}: {results[name]:.3f}")
    else:
        print(f"{name}: not fitted yet. Step 6.")
```

### Step 8. Check

Run cell 8. It prints `Done:` when your guess is typed and both candidates are fitted on the twelve features. Otherwise it prints `Not yet:` and what is left.

Keep this notebook. Exercise 3 asks you for the two accuracies of cell 7.

Notebook cell 8:

```python
# 8. Check. You do not need to read this code. It runs check.py, beside this notebook.
_check = {"__name__": "check"}
exec(open(f"{folder}/check.py", encoding="utf-8").read(), _check)
check_now = _check["in_notebook"]
print(check_now(globals()))
```

### Finished early?

Optional: cell 8 does not check it.

In cell 9, between the quotes, write one sentence: which candidate beat the baseline, and by how much? Then run cell 9.

Notebook cell 9:

```python
# 9. Optional: one sentence.
by_how_much = ""
print(by_how_much or "Write your sentence between the quotes.")
```

### Afterwards, together

1. Why measure the baseline again on the validation rows?
2. What does a gain of 31 returns right not tell you?
3. What would you check first if a candidate's accuracy were 0.999?

The answer, after you try: [`answers/exercise-02-fit-two-candidates`](../answers/exercise-02-fit-two-candidates/).
