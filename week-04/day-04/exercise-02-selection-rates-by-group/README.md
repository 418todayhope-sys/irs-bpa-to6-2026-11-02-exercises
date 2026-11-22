# Exercise 2: Compare selection rates by group, then say whether a feature could stand in for the group

**Goal:** After this exercise, you can compute each group's selection rate at your threshold, state the gap with its counts, and say whether one feature could stand in for the group.

**Scenario:** Before a made-up examination group puts the model in use, its manager asks whether the list falls more on one group of returns than on another. Each validation return carries a synthetic group, A or B, made up for this course and kept out of the features. You compute each group's selection rate, write the result with its counts, then compare one feature's typical value in the two groups.

**AI tool:** none.

**Time:** 25 minutes: 21 to do it, 4 for setup problems.

**Where you work:** this notebook, `week-04/day-04/exercise-02-selection-rates-by-group/exercise.ipynb`, in your Git folder of the course in Databricks.

**Done when:** cell 6 prints `Done:`.

**Optional part:** Finished early? Cell 7 compares a second feature in the two groups.

**What you have:**

- `validation-returns.csv`: Wednesday's 1,000 made-up validation returns, each with its score from each candidate, its synthetic group, and two columns that hold 0 or 1: `amended_last_year`, a feature the trainer built for this check, and `withholding_missing`, Monday's flag.
- `ten-returns.csv`: Wednesday's ten returns, each with a synthetic group, for the trainer's demo.
- `load.py` and `check.py`: given code that cells 1 and 6 run.

Groups A and B are made up. They describe no real person and stand for no protected class. The model never reads the group column.

**Start from:** this notebook as the course gives it, attached to the compute the trainer names. No cell reads or changes your tables. New here: every validation return split by group, and the feature.

### Step 1. Put the validation returns on the list

Why: "with Wednesday's whole-set example of 30 per 1,000 returns, the 30 returns with the highest scores" are on the list (the day's reading).

1. In cell 1, after `chosen =`, type the candidate you chose on Wednesday. If you chose none, keep `logistic-regression`.
2. Run cell 1. You do not need to read its code. You should see `Read 1000 made-up validation returns`, the threshold, how many returns are on the list, and the columns.

The list never holds more than 30 returns, because that is all the examiners can take. With the tree, many returns share the 30th score, so not all of them fit: a person breaks the tie and writes down how. Cell 1 breaks it by return ID, the lower IDs first, and prints a line when it does.

Notebook cell 1:

```python
# 1. The 1,000 made-up validation returns, each on the list or off it at your candidate's threshold.
chosen = "logistic-regression"  # or "decision-tree", the candidate you chose on Wednesday
import os
folder = os.getcwd().split("/week-04/")[0] + "/week-04/day-04/exercise-02-selection-rates-by-group"
exec(open(f"{folder}/load.py", encoding="utf-8").read())  # reads the files beside this notebook
```

### Step 2. Guess

Why: you understand a result better when you said what you expected first.

1. In cell 2, between the quotes, type `A` or `B`: which group do you expect to have the higher share of its returns on the list?
2. Run cell 2. You should see your guess.

Notebook cell 2:

```python
# 2. Your guess: which group has the higher share of its returns on the list? A or B.
guess = ""
print("Your guess:", guess or "(not yet)")
```

### Step 3. Count each group's returns, and its returns on the list

Why: "Split the held-out returns by group." (the day's reading) The validation returns are the held-out returns.

1. In cell 3, between the quotes, type the name of the column that holds each return's group. Cell 1 printed the columns.
2. Run cell 3. You should see one row for each group: its returns, and how many of them are on the list.

Notebook cell 3:

```python
# 3. Each group's returns, and its returns on the list. Change only group_column.
group_column = ""  # the column that holds each return's group
table = "validation_returns"  # keep it: only the trainer's demo uses ten_returns
assert group_column, "Type the column's name between the quotes."
spark.sql(f"""
    SELECT {group_column}, COUNT(*) AS returns_in_group, SUM(on_list) AS on_the_list
    FROM {table}
    GROUP BY {group_column}
    ORDER BY {group_column}
""").show()
```

### Step 4. Compute each rate, and state the result

Why: "The *selection rate* of a group is the share of its returns that the model puts on the list." (the day's reading) Groups of different sizes compare by rates, not by counts.

1. In cell 4, type in place of each `None` the group's rate as a division: its returns on the list, divided by its returns in the group, such as `3 / 50`.
2. Run cell 4. You should see each rate in 100, and the gap: the difference between the two rates.
3. Between the quotes after `result`, write the result in one sentence. Give each group's returns on the list and its returns in the group beside its rate. Give the gap as a number, such as `1.5 in 100`. End with the words `needs a reason`.
4. Run cell 4 again. You should see your sentence.

A gap is a result that needs a reason. It does not prove that the model is unfair. In a small group, one return more or less moves the rate a great deal, so the counts go beside the rates.

<details><summary>Stuck? One way to start</summary>

The model puts ... of group A's ... returns on the list, ... in 100, ...

</details>

Notebook cell 4:

```python
# 4. Each group's rate as a division, then the result in one sentence.
rate_a = None  # group A: on the list, divided by its returns in the group
rate_b = None  # group B: the same
if rate_a is not None and rate_b is not None:
    print(f"Group A: {rate_a * 100:.1f} in 100. Group B: {rate_b * 100:.1f} in 100. "
          f"The gap: {abs(rate_a - rate_b) * 100:.1f} in 100.")
result = ""  # one sentence: each rate with its counts, the gap, and that the gap needs a reason
print(result or "Write your sentence between the quotes.")
```

### Step 5. Compare the feature in the two groups

Why: "A *proxy* is a feature that names no group and still tracks one. You see it when the feature's typical value differs between the groups." (the day's reading)

The trainer built the feature `amended_last_year`: 1 where the taxpayer amended last year's return, 0 where not. Today's model does not read it, so it does not explain the gap of step 4. A later model could read it: before anyone adds it, you compare it in the two groups. Values far apart, such as 30 against 5 in 100, mean it could stand in; values close together mean it could not. For a column that holds 0 or 1, the typical value is the share of the group's returns that hold 1.

1. Run cell 5. You should see, for each group, how many of its returns in 100 hold 1.
2. Between the quotes after `proxy_sentence`, write one sentence: `amended_last_year could stand in for the group, because ...`, or `could not stand in`. After `because`, give its value in each group.
3. Run cell 5 again. You should see your sentence.

Write *could*. The comparison shows that the feature tracks the group. It does not show that a model uses it in the group's place.

Notebook cell 5:

```python
# 5. The feature's typical value in each group: how many returns in 100 hold 1. Then one sentence.
feature = "amended_last_year"
spark.sql(f"""
    SELECT synthetic_group, ROUND(AVG({feature}) * 100, 1) AS in_100_hold_1
    FROM validation_returns
    GROUP BY synthetic_group
    ORDER BY synthetic_group
""").show()
proxy_sentence = ""  # could stand in, or could not stand in, for the group, because ...
print(proxy_sentence or "Write your sentence between the quotes.")
```

### Step 6. Check

Run cell 6. It prints `Done:`, or `Not yet:` and what is left.

Keep this notebook. Friday's challenge asks for the result sentence of cell 4.

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

1. Run cell 7. It compares Monday's flag `withholding_missing` in the two groups, as cell 5 did. The model reads this flag: the ranked list of exercise 1 put it third.
2. Between the quotes, write one sentence of the same form for the flag.

Notebook cell 7:

```python
# 7. Optional: a second feature, Monday's flag, in the two groups.
second = "withholding_missing"
spark.sql(f"""
    SELECT synthetic_group, ROUND(AVG({second}) * 100, 1) AS in_100_hold_1
    FROM validation_returns
    GROUP BY synthetic_group
    ORDER BY synthetic_group
""").show()
second_sentence = ""
print(second_sentence or "Write your sentence between the quotes.")
```

### Afterwards, together

1. Why compare rates and not the counts on the list?
2. What does a gap need before anyone calls it unfair?
3. How can a list fall more on one group without its column?

The answer, after you try: [`answers/exercise-02-selection-rates-by-group`](../answers/exercise-02-selection-rates-by-group/).
