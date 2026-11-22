# Exercise 1: Name the top three features, then explain case #123 in two sentences

**Goal:** After this exercise, you can name your model's top three features and explain case #123's score in two sentences an examiner could read.

**Scenario:** A made-up examiner opens the made-up case #123 on the list, sees one number, 0.82, and asks why it is there. You read your candidate's ranked chart and name its top three features. Then you read case #123's chart and write the two sentences the examiner reads.

**AI tool:** none.

**Time:** 25 minutes: 21 to do it, 4 for setup problems.

**Where you work:** this notebook, `week-04/day-04/exercise-01-explain-case-123/exercise.ipynb`, in your Git folder of the course in Databricks.

**Done when:** cell 6 prints `Done:`.

**Optional part:** Finished early? Cell 7 explains a return that the model left off the list.

**What you have:**

- `train-returns.csv`: 2,000 made-up training returns with three of Tuesday's features and the label. Cell 1 adds the fourth, Monday's flag `withholding_missing`: 1 where the withholding amount is missing. The average score in the chart is taken over them.
- `case-123.csv`: case #123's four features and no label. It is a new return, not yet examined. Its withholding amount is missing.
- `ten-returns.csv`: Wednesday's ten made-up returns, with their features.
- `load.py` and `check.py`: given code that cells 1 and 6 run. Cell 1 fits Tuesday's two candidates again, on the training returns.

**Start from:** this notebook as the course gives it, attached to the compute the trainer names. No cell reads or changes your tables. New here: the ranked chart, and case #123's chart.

### Step 1. Fit your candidate

Why: you explain the model that gave the score, the candidate you chose on Wednesday.

1. In cell 1, after `chosen =`, type the candidate you chose on Wednesday: `logistic-regression` or `decision-tree`. If you chose none, keep `logistic-regression`.
2. Run cell 1. You do not need to read its code. You should see `Read 2000 made-up training returns, case #123 and Wednesday's ten`, then your candidate.

The slides follow the logistic regression. If you chose the tree, your charts differ from the slides. The method is the same.

Notebook cell 1:

```python
# 1. Your candidate from Wednesday, fitted again on the made-up training returns.
chosen = "logistic-regression"  # or "decision-tree", the candidate you chose on Wednesday
import os
folder = os.getcwd().split("/week-04/")[0] + "/week-04/day-04/exercise-01-explain-case-123"
exec(open(f"{folder}/load.py", encoding="utf-8").read())  # reads the files beside this notebook
```

### Step 2. Name the top three features

Why: "The chart is a *ranked list*: the features in order, with the one the model leans on most at the top." (the day's reading)

1. Run cell 2. You should see a chart of four bars, the longest at the top, then the same list as text.
2. In cell 2, between the quotes, type the names of the three longest bars, the longest first.
3. Run cell 2 again. You should see your top three.

The number beside each bar is its weight. Read the order, not the amounts. The list describes the model. It does not say what made a tax wrong.

Notebook cell 2:

```python
# 2. The ranked list of your model's features. Then type the top three, the longest bar first.
ranked = draw_ranking()
top_three = ["", "", ""]
print("Your top three:", top_three)
```

### Step 3. Draw case #123's chart

Why: "Your job is to read the chart." (the day's reading) Where the SHAP library is on the compute, the given code uses it and draws its chart. Where it is not, the given code computes the same pushes and says so.

1. In cell 3, type `123` in place of `None`. Until you do, the cell stops with `Type the case id in place of None.`
2. Run cell 3. You should see case #123's row as read, with its missing withholding amount empty (`NaN` or `NULL`), then a chart with one bar for each feature, then the same chart as a table: the average score, one line for each feature with its push, and the score of case #123.
3. Read the axis label under the chart first. The table repeats it on its last line.

Case #123's withholding amount was missing, so its line shows the value that Tuesday's fill put in its place, marked `(filled)`: 937, the middle value of the training returns' amounts. So one missing amount moves the score twice: through the flag, and through the filled amount.

Notebook cell 3:

```python
# 3. One return's chart: the average score, each feature's push, the return's score. Change only the case id.
case_id = None  # the return to explain, such as 311
assert case_id is not None, "Type the case id in place of None."
spark.sql(f"SELECT * FROM returns_to_explain WHERE return_id = '#{case_id}'").show()  # the return as read
chart = explain(case_id)
```

### Step 4. Read the chart, and check the sum

Why: "the average score plus all the pushes equals the return's score." (the day's reading) The sum is how you check your reading.

1. In cell 4, between the quotes after `axis`, type the unit the axis label names, such as `score` or `log-odds`.
2. Type in place of each `None` the number from cell 3's table, to two decimals: the average score, each feature's push with its sign, and case #123's score. Type a minus sign before a push that lowers the score, such as `-0.07`.
3. Run cell 4. You should see the average plus the pushes, and the score. The two agree to about 0.01. If not, read the table again.

Notebook cell 4:

```python
# 4. Cell 3's chart, as numbers. Type each to two decimals, with its sign.
axis = ""        # what the axis is measured in
average = None   # the average score of all returns
pushes = {
    "tax_to_agi": None,
    "wages_change_pct": None,
    "withholding_missing": None,
    "withholding_amt": None,
}
score = None     # case #123's score
if None in (average, score, *pushes.values()):
    print("Type each number in place of None.")
else:
    print(f"The average plus the pushes: {average + sum(pushes.values()):.2f}. The score: {score:.2f}.")
```

### Step 5. Write the two sentences

Why: "So the explanation is a comparison. You do not say that 0.82 is high. You say what made it higher than 0.05." (the day's reading)

1. In cell 5, replace each `...` in the two sentences. Give the score and the average score from cell 3.
2. For each of the two largest pushes, write what the feature says about the return, in words an examiner reads, such as "the fall in its wages since last year". Then give its push as an amount of score.
3. Run cell 5. You should see your two sentences.

If you chose the tree, give your own score and pushes, and where one of the two largest pushes lowers the score, write `lowered` in place of `raised`. The axis is the score, so you may give amounts. The sentences say what moved the score. They do not say why the taxpayer did something, or that the return is wrong: the examiner decides.

Notebook cell 5:

```python
# 5. The two sentences an examiner reads. Replace each ... .
sentences = "The model scored case #123 at ..., against an average of ... for all returns. The two largest reasons were ..., which raised the score by ..., and ..., which raised it by ...."
print(sentences)
```

### Step 6. Check

Run cell 6. It prints `Done:`, or `Not yet:` and what is left.

Keep this notebook. Friday's challenge asks for the same two sentences.

Notebook cell 6:

```python
# 6. Check. You do not need to read this code. It runs check.py, beside this notebook.
_check = {"__name__": "check"}
exec(open(f"{folder}/check.py", encoding="utf-8").read(), _check)
check_now = _check["in_notebook"]
print(check_now(globals()))
```

### Finished early?

Optional: cell 6 does not check it.

1. In cell 7, type in place of `None` the id of a return from Wednesday's ten that is off the list, such as `644`.
2. Run cell 7. You should see that return's chart.
3. Between the quotes, write two sentences of the same form. Where a push lowers the score, write `lowered` in place of `raised`.

Notebook cell 7:

```python
# 7. Optional: explain a return that the model left off the list.
other_id = None  # a return off the list, such as 644
if other_id is not None:
    other = explain(other_id)
other_sentences = ""
print(other_sentences or "Write your two sentences between the quotes.")
```

### Afterwards, together

1. Why does the top feature say nothing about what the taxpayer did?
2. How do you check that case #123's pushes are read right?
3. When would you give a push's direction but not its amount?

The answer, after you try: [`answers/exercise-01-explain-case-123`](../answers/exercise-01-explain-case-123/).
