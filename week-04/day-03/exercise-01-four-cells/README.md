# Exercise 1: Sort ten returns into four cells, then count every validation return

**Goal:** After this exercise, you can sort scored returns into the four cells at 0.5 and say what each cell means for an examiner.

**Scenario:** A made-up examination group asks how Tuesday's logistic regression would have done on ten returns whose examinations have closed. You put each one on the list or off it at 0.5, sort it into its cell, and say what each cell means for the examiners. Then a given cell counts all 1,000 made-up validation returns.

**AI tool:** none.

**Time:** 25 minutes: 18 to do it, 7 for setup problems.

**Where you work:** this notebook, `week-04/day-03/exercise-01-four-cells/exercise.ipynb`, in your Git folder of the course in Databricks.

**Done when:** cell 7 prints `Done:`.

**Optional part:** Finished early? Cell 8 moves the threshold to 0.7.

**What you have:**

- `ten-returns.csv`: the ten made-up returns of the day's reading, each with its score from Tuesday's logistic regression and its label.
- `validation-scores.csv`: 1,000 made-up validation returns, each with its label and its score from each of Tuesday's two candidates.
- `load.py` and `check.py`: given code that cells 1 and 7 run.

**Start from:** this notebook as the course gives it, attached to the compute the trainer names. No cell reads or changes your tables. New here: the four cells, counted at one threshold.

### Step 1. Read the ten returns

Why: you judge the model on returns whose label you hold, and that it never learned from.

Run cell 1. You do not need to read its first three lines. You should see `Read 10 made-up returns and 1000 made-up validation returns.`, then a table of the ten returns, highest score first: `return_id`, `score` and `label`.

A score is a relative risk indicator, never a finding about the taxpayer. A label records a past examination: `change` means it changed the tax.

Notebook cell 1:

```python
# 1. The ten made-up returns, highest score first.
import os
folder = os.getcwd().split("/week-04/")[0] + "/week-04/day-03/exercise-01-four-cells"
exec(open(f"{folder}/load.py", encoding="utf-8").read())  # reads the files beside this notebook

spark.sql("SELECT return_id, score, label FROM ten_returns ORDER BY score DESC").show()
```

### Step 2. Put each return on or off the list, then in its cell

Why: "The first word of each cell's name, *true* or *false*, says whether the list agreed with the label." (the day's reading)

The threshold is 0.5. A return with a score at or above 0.5 goes on the list; below it, the return stays off.

| Cell | Where the return landed | The label |
|---|---|---|
| `TP`, true positive | on the list | change |
| `FP`, false positive | on the list | no change |
| `FN`, false negative | off the list | change |
| `TN`, true negative | off the list | no change |

1. In cell 2, for each return, type `on` or `off` in the first quotes, then its cell in the second. The comment beside each return gives its score and label. For example, #388, score 0.84, label change: `["on", "TP"]`.
2. Run cell 2. You should see one line for each return, with what you typed.

Notebook cell 2:

```python
# 2. Your sort at 0.5. For each return: "on" or "off" the list, then its cell: TP, FP, FN or TN.
threshold = 0.5
my_sort = {
    "#311": ["", ""],  # score 0.91, label change
    "#388": ["", ""],  # score 0.84, label change
    "#458": ["", ""],  # score 0.74, label no change
    "#206": ["", ""],  # score 0.63, label change
    "#270": ["", ""],  # score 0.55, label no change
    "#502": ["", ""],  # score 0.46, label change
    "#189": ["", ""],  # score 0.38, label no change
    "#644": ["", ""],  # score 0.27, label no change
    "#135": ["", ""],  # score 0.15, label no change
    "#717": ["", ""],  # score 0.08, label no change
}
for return_id, (place, cell) in my_sort.items():
    print(return_id, place or "?", cell or "?")
```

### Step 3. Count your four cells, then check two sums

Why: two sums tell you that a count is right: "The four counts add up to the number of returns. The true positives plus the false negatives equal the number of returns labelled *change*." (the day's reading)

1. By hand, count how many of the ten you marked `TP`, `FP`, `FN` and `TN` in cell 2.
2. In cell 3, type each count in place of `None`.
3. Run cell 3. You should see sum 1 beside the number of returns, 10, and sum 2 beside the number of returns labelled change. If a sum does not match, look again at cell 2.

Notebook cell 3:

```python
# 3. Your four counts, from cell 2. Type each count in place of None.
my_counts = {"TP": None, "FP": None, "FN": None, "TN": None}

if None in my_counts.values():
    print("Type the four counts first.")
else:
    print("Sum 1, all four cells:", sum(my_counts.values()), " The returns:", len(ten))
    print("Sum 2, TP + FN:", my_counts["TP"] + my_counts["FN"], " Labelled change:", (ten["label"] == "change").sum())
```

### Step 4. Say what each cell means for an examiner

Why: the two ways of being wrong "affect different people" (the day's reading). An examiner acts on the list, not on a score.

1. In cell 4, between the quotes, write what each cell means for an examiner, in your own words, four words or more. Think of who spends hours, and who looks at the return. A true positive, for example: `a case that was worth the examiner's hours`.
2. Run cell 4. You should see your four sentences.

*True* means the list agreed with the label. It says nothing about a taxpayer.

Notebook cell 4:

```python
# 4. What each cell means for an examiner, in your words.
meaning = {
    "TP": "",  # on the list, labelled change
    "FP": "",  # on the list, labelled no change
    "FN": "",  # off the list, labelled change
    "TN": "",  # off the list, labelled no change
}
for cell, words in meaning.items():
    print(cell, ":", words or "(not yet)")
```

### Step 5. Count every validation return with the given code

Why: you sorted the ten by hand, so you know what the counting code does. Now it counts all 1,000.

Run cell 5. It puts each of the 1,000 returns on the list or off it at 0.5, by the logistic regression's score, and `confusion_matrix` from scikit-learn counts the four cells. You should see a table of two rows, two numbers in each, then how many returns are labelled change.

How to read the table: scikit-learn puts the label in the rows, "with each row corresponding to the true class" (the day's reading), the *true class* being the label. The first row is the returns labelled `change`, the second those labelled `no change`. In each row, the first number is the returns on the list, the second those off it.

Notebook cell 5:

```python
# 5. The counting code: every validation return at 0.5, the logistic regression's scores against the labels.
from sklearn.metrics import confusion_matrix
threshold = 0.5

on_list = ["change" if score >= threshold else "no change" for score in valid["logistic-regression"]]
# a return on the list counts as "change", a return off it as "no change"
table = confusion_matrix(valid["label"], on_list, labels=["change", "no change"])
print(table)
print(len(valid), "validation returns,", (valid["label"] == "change").sum(), "of them labelled change")
```

### Step 6. Name each count of the table, then check the two sums

Why: "tools arrange the rows and columns differently" (the day's reading). The two sums tell you that you read the table right.

1. In cell 6, type each count from cell 5's table in place of `None`. Read it as step 5 says, and name each count by the table of step 2.
2. Run cell 6. You should see sum 1 beside 1000, and sum 2 beside the returns labelled change. If a sum does not match, read the table again.

Notebook cell 6:

```python
# 6. The four counts of every validation return, read from cell 5's table.
all_counts = {"TP": None, "FP": None, "FN": None, "TN": None}

if None in all_counts.values():
    print("Type the four counts first.")
else:
    print("Sum 1, all four cells:", sum(all_counts.values()), " The returns:", len(valid))
    print("Sum 2, TP + FN:", all_counts["TP"] + all_counts["FN"], " Labelled change:", (valid["label"] == "change").sum())
```

### Step 7. Check

Run cell 7. It prints `Done:` when your sort, your counts and your four meanings are there and right. Otherwise it prints `Not yet:` and what is left.

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

Move the threshold to 0.7. In cell 8, type the four counts of the ten at 0.7, and between the quotes, which returns changed cell. Then run cell 8. You should see your counts beside the counts at 0.7.

Notebook cell 8:

```python
# 8. Optional: the ten returns at a threshold of 0.7.
at_07 = {"TP": None, "FP": None, "FN": None, "TN": None}
changed = ""  # which returns changed cell, and from which cell to which

on_07 = ten["score"] >= 0.7
change = ten["label"] == "change"
counted = {"TP": (on_07 & change).sum(), "FP": (on_07 & ~change).sum(), "FN": (~on_07 & change).sum(), "TN": (~on_07 & ~change).sum()}
print("Yours:", at_07)
print("At 0.7:", {cell: int(n) for cell, n in counted.items()})
print("Changed:", changed or "(not yet)")
```

### Afterwards, together

1. Why is #458 a false positive, not a finding about a taxpayer?
2. Which cell does nobody see once the model is in use?
3. Which two sums show the tool's table is read right?

The answer, after you try: [`answers/exercise-01-four-cells`](../answers/exercise-01-four-cells/).
