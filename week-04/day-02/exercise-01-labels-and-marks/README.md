# Exercise 1: Count your labels, say their origin, mark each feature

**Goal:** After this exercise, you can record where your labels came from and mark each feature as known before or after selection.

**Scenario:** You were handed made-up returns, most of them labelled with an examination's result, and asked for a model. First you check the labels, and whether each feature was known before selection.

**AI tool:** none.

**Time:** 25 minutes: 21 to do it, 4 for setup problems.

**Where you work:** this notebook, `week-04/day-02/exercise-01-labels-and-marks/exercise.ipynb`, in your Git folder of the course in Databricks.

**Done when:** cell 8 prints `Done:`.

**Optional part:** Finished early? Cell 9 asks for one sentence.

**What you have:**

- `course.returns.model_train`, the course's training set, tax years 2020 and 2021, as on Monday. Its label `exam_changed_tax` is `yes` when an examination changed the tax, `no` when it did not, and empty when nobody examined the return. If it cannot be read, cell 2 makes the same rows from the course's setup file and says so.
- `course.learner_NN.model_feature_record`, the feature record you saved on Monday (`NN`: your number).
- `fields.csv`: a small table, kept in a file, of one made-up column. `account_change_amt` is a change to the tax on the taxpayer's account: much the same fact as `tax_added_by_exam`, under another name, in another place. An examination writes it when it closes, months after the case was selected. It is 0 when no tax was added, or when none was written yet on the day the course made the data.
- `refund-held.csv` (the trainer's demo), `load.py` and `check.py` (given code).

**Start from:** this notebook as the course gives it, on the trainer's compute. No cell changes your tables. New here: a feature that gives the label away.

### Step 1. Name your schema

1. In cell 1, type your number in place of `NN`.
2. Run cell 1. You should see `Your tables are in` and your schema.

Notebook cell 1:

```python
# 1. Your schema, where your tables are. Type your number in place of NN.
schema = "course.learner_NN"

import os
course_folder = os.getcwd().split("/week-04/")[0]  # your Git folder of the course
print("Your tables are in", schema)
```

### Step 2. Load the training set and your feature record

Run cell 2. You should see which table it uses, then your feature record's rows.

Notebook cell 2:

```python
# 2. Load the course's training set, the made-up columns, and your feature record from Monday.
try:
    train_table = spark.table("course.returns.model_train")  # the course's training set
except Exception:
    train_table = None  # not there: load.py makes the same rows
try:
    record_table = spark.table(f"{schema}.model_feature_record")  # Monday's feature record
except Exception:
    record_table = None

# You do not need to read the lines below. They run load.py.
folder = f"{course_folder}/week-04/day-02/exercise-01-labels-and-marks"
exec(open(f"{folder}/load.py", encoding="utf-8").read())
train, record = load_day(train_table, record_table, schema)
```

### Step 3. Count each value of your label

"A return with an empty label is not counted as either outcome, and it is left out of training." (the day's reading)

1. In cell 3, type `exam_changed_tax` in place of `refund_held`. Keep the quotes.
2. Run cell 3. You should see a count for `no`, for `yes` and for `empty`.

Notebook cell 3:

```python
# 3. Count the values of one label in your training set.
label_column = "refund_held"  # the demo's label. For the exercise, type exam_changed_tax in its place.

counts = train[label_column].value_counts(dropna=False)
for value, n in counts.items():
    print(f"{value if isinstance(value, str) else 'empty'}: {n} returns")
```

### Step 4. Write the labels' origin

"A *label's origin* is the answer to those three questions: which past decision made the label, who made it, and when." (the day's reading)

1. In cell 4, type one answer between each pair of quotes, in your own words, two words or more each. Use the table below.
2. Run cell 4. You should see your three answers.

| Question | The fact that decides it |
|---|---|
| `decision` | `yes` means "the examination changed the tax" |
| `made by` | "The course's data gives some returns a label that stands for an examination's result." (the day's reading) The returns are made up, so say who made the label in their place. |
| `when` | an examination closes after the return was filed and selected |

Notebook cell 4:

```python
# 4. The labels' origin: one answer between each pair of quotes.
origin = {
    "decision": "",  # which past decision made the label
    "made by": "",   # who made that decision
    "when": "",      # when it was made
}
for question, answer in origin.items():
    print(f"{question}: {answer or '(not yet)'}")
```

### Step 5. Put one column beside the label

Why: "A feature that matches the label almost exactly is a warning sign of leakage." (the day's reading)

Cell 5 counts how often a column agrees with the label: above zero with `yes`, or zero with `no`.

1. In cell 5, type `account_change_amt` in place of `refund_hold_days`.
2. Run cell 5. You should see four counts, then how many returns agree. Nearly all of them agree.

Notebook cell 5:

```python
# 5. Put one column beside the label, and count how often the two agree.
suspect = "refund_hold_days"  # the demo's column. For the exercise, type account_change_amt in its place.

labelled = train[train[label_column].notna()]
above = labelled[suspect] > 0
for side, is_above in (("above zero", True), ("zero", False)):
    for value in ("yes", "no"):
        n = int(((above == is_above) & (labelled[label_column] == value)).sum())
        print(f"{suspect} {side}, label {value}: {n}")
agree = int((above == (labelled[label_column] == "yes")).sum())
agreement_label = label_column
print(f"{suspect} agrees with {label_column} on {agree} of {len(labelled)} returns with a label.")
```

### Step 6. Trace the column, then judge it

Why: "trace it to where it came from: which table holds it, which step wrote it, and when. The answer to *when* decides." (the day's reading) Here *step* means a stage of the tax work, not a step of this exercise.

1. In cell 6, answer the three questions in your own words. What you have, at the top, describes `account_change_amt`. Here the table is a file beside the notebook, not a table in the catalog.
2. Run cell 6. You should see your three answers.

Notebook cell 6:

```python
# 6. Trace account_change_amt: where it came from.
trace = {
    "which table holds it": "",
    "which step wrote it": "",
    "when": "",
}
for question, answer in trace.items():
    print(f"{question}: {answer or '(not yet)'}")
```

### Step 7. Mark each feature `before` or `after`

Why: "was it known before the case was selected? Mark each feature *before* or *after*. Train only on the features marked *before*." (the day's reading)

1. In cell 7, for each of your six features, type `before` or `after`, then a reason of three words or more. The demo rows show the form.
2. Run cell 7. You should see four features under `Train on:` and two under `Leave out:`.

| Feature | Where its value comes from |
|---|---|
| `tax_to_agi` | total tax and adjusted gross income, on the filed return |
| `wages_amt` | wages, line 1a of the filed return |
| `deduction_amt` | the deduction, line 12e of the filed return |
| `deduction_missing` | Monday's flag: 1 when the filed line 12e is empty |
| `tax_added_by_exam` | the course's table: the tax the examination added; you struck it on Monday |
| `account_change_amt` | your trace, cell 6 |

<details><summary>Stuck? Five of the marks</summary>

`before` for the first four: each comes from the filed return. `after` for `tax_added_by_exam`: the examination writes it.
For `account_change_amt`, the answer to *when* in cell 6 decides.

</details>

Notebook cell 7:

```python
# 7. Mark each feature: "before" or "after" the case was selected, and why.
demo_marks = {  # the trainer's demo: two other features, as an example of the form
    "filing_status": ("before", "it is on the filed return"),
    "refund_hold_days": ("after", "counted from the day a refund hold started"),
}
marks = {  # your six: type before or after, then a reason of three words or more
    "tax_to_agi": ("", ""),
    "wages_amt": ("", ""),
    "deduction_amt": ("", ""),
    "deduction_missing": ("", ""),
    "tax_added_by_exam": ("", ""),
    "account_change_amt": ("", ""),
}
print("Train on:", ", ".join(f for f, (mark, _) in marks.items() if mark.strip().lower() == "before"))
print("Leave out:", ", ".join(f for f, (mark, _) in marks.items() if mark.strip().lower() == "after"))
```

### Step 8. Check

Run cell 8. It prints `Done:`, or `Not yet:` and what is left. Exercise 2 trains on these four and eight more from the filed return, all known before selection.

Notebook cell 8:

```python
# 8. Check. You do not need to read this code. It runs check.py, beside this notebook.
_check = {"__name__": "check"}
exec(open(f"{folder}/check.py", encoding="utf-8").read(), _check)
print(_check["in_notebook"](schema, {
    "train": train, "label_column": label_column, "origin": origin, "suspect": suspect,
    "agreement_label": agreement_label, "agree": agree, "labelled": len(labelled), "trace": trace, "marks": marks}))
```

### Finished early?

Optional: cell 8 does not check it.

In cell 9, between the quotes, write one sentence: why does `account_change_amt` miss one return labelled `yes`? What you have holds the fact. Then run cell 9.

Notebook cell 9:

```python
# 9. Optional: one sentence.
why_one_missed = ""
print(why_one_missed or "Write your sentence between the quotes.")
```

### Afterwards, together

1. Why is an empty label not counted as a no?
2. What would change your answer on who made these labels?
3. Why trace a matching feature before you strike it?

The answer, after you try: [`answers/exercise-01-labels-and-marks`](../answers/exercise-01-labels-and-marks/).
