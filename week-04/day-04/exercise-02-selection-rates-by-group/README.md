# Exercise 2: Compare selection rates by channel, then state the gap (we do)

**Goal:** After this exercise, you can compute each group's selection rate at your threshold and state the gap with its counts.

**Scenario:** Before a made-up examination group puts the model in use, its manager asks whether the list of 25 falls more on one group of returns than on another. Each return carries the course's column `made_up_channel`, A or B. You compute each channel's selection rate at Wednesday's threshold and write the result with its counts.

**The channel:** `made_up_channel` is a code the course invented and calls a filing channel. "It describes no real person or group, and it stands for no protected class such as race, sex or age; the arithmetic is the same for any grouping." (the day's reading) It is never a feature: Tuesday's twelve leave it out, and the model never reads it.

**AI tool:** none.

**Time:** 15 minutes: 11 to do it, 4 for setup problems.

**Where you work:** this notebook, `week-04/day-04/exercise-02-selection-rates-by-group/exercise.ipynb`, in your Git folder of the course in Databricks.

**Done when:** cell 6 prints `Done:`.

**Cells:** a cell's number is the `# N.` comment on its first line, such as `# 1.`; text cells have no number.

**What you have:**

- `course.returns.model_train` and `course.returns.model_valid`, as on Wednesday, and `course.returns.model_features`, all 1,000 returns of the four tax years. If they cannot be read, cell 1 makes the same rows from the course's setup file and says so.
- `load.py` and `check.py`, given code that cells 1 and 6 run.

**Start from:** this notebook as the course gives it, attached to the compute the trainer names. No cell changes your tables. New here: the list of 25 split by channel.

### Step 1. Put the validation returns on the list

Run cells 1 and 2. You do not need to change them. You should see:

1. Which tables cell 1 uses, then 500 returns in `model_train`, 250 in `model_valid` and 1,000 in `model_features`.
2. Cell 2 fits Tuesday's logistic regression again, with Tuesday's lines, and scores every return of `model_valid`. Wednesday's threshold is the 25th score: you should see `Threshold 0.844: 25 of 250 returns on the list`, then the columns.

A score is one return's number from 0 to 1: a relative risk indicator that a person acts on, never a finding about the taxpayer. The model puts a return on the list or leaves it off; a person selects.

Notebook cell 1:

```python
# 1. Read the course's tables. You do not need to change this cell.
import os
course_folder = os.getcwd().split("/week-04/")[0]  # your Git folder of the course
folder = f"{course_folder}/week-04/day-04/exercise-02-selection-rates-by-group"
try:
    train_table = spark.table("course.returns.model_train")  # the training set: tax years 2020 and 2021
    valid_table = spark.table("course.returns.model_valid")  # the validation set: tax year 2022
    all_table = spark.table("course.returns.model_features")  # all 1,000 returns, the four tax years
except Exception:
    train_table, valid_table, all_table = None, None, None  # not there: load.py makes the same rows
exec(open(f"{folder}/load.py", encoding="utf-8").read())  # you do not need to read load.py
train, valid, all_returns = load_sets(train_table, valid_table, all_table)
```

Notebook cell 2:

```python
# 2. Tuesday's logistic regression, fitted again as Tuesday fitted it; it scores every return of model_valid.
assert "valid" in globals(), "Run cell 1 first."
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

features = ["filing_status", "wages_amt", "taxable_interest_amt", "other_income_amt",
            "total_income", "agi_amt", "deduction_amt", "deduction_missing",
            "total_tax_amt", "withholding_amt", "tax_to_agi", "deduction_to_agi"]  # Tuesday's twelve
examined = train[train["exam_changed_tax"].notna()]  # the 400 examined returns of model_train
X_train, y_train = examined[features].fillna(0), examined["exam_changed_tax"]
logistic = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_train, y_train)

scores = valid[["case_id", "made_up_channel", "exam_changed_tax"]].copy()  # all 250 returns of model_valid
scores["logistic_score"] = logistic.predict_proba(valid[features].fillna(0))[:, 1]  # column 1 is "yes"
threshold = scores["logistic_score"].sort_values(ascending=False).iloc[24]  # Wednesday's: the 25th score
scores["on_list"] = (scores["logistic_score"] >= threshold).astype(int)  # 1 on the list, 0 off it
print(f"Threshold {threshold:.3f}: {scores['on_list'].sum()} of {len(scores)} returns on the list.")
print("Columns:", ", ".join(scores.columns))
```

### Step 2. Guess

"So a gap belongs to a threshold, as precision and recall did on Wednesday: write the threshold beside it." (the day's reading)

1. In cell 3, between the quotes, type your guess: if the examiners could take 50 returns, not 25, would the gap between the two channels' rates be `wider` or `narrower`? Either counts.
2. Run cell 3. You should see your guess. Cell 5 shows the answer.

Notebook cell 3:

```python
# 3. Your guess: on a list of 50, not 25, is the gap between the two channels' rates wider or narrower?
guess_50 = ""  # wider or narrower
print("Your guess:", guess_50 or "(not yet)")
```

### Step 3. Count each channel's returns, and its returns on the list

"Split the held-out returns by group." (the day's reading) The 250 returns of `model_valid` are the held-out returns.

1. In cell 4, between the quotes, type the name of the column that holds each return's channel. Cell 2 printed the columns.
2. Run cell 4. You should see one row for each channel: its returns, and how many of them are on the list.

Channel A has 15 on the list and channel B 10. Counts alone do not answer the manager: the two channels are not the same size.

Notebook cell 4:

```python
# 4. Each channel's returns, and its returns on the list at Wednesday's threshold. Change only group_column.
group_column = ""  # the column that holds each return's channel
assert group_column in scores.columns, f"Type the channel's column between the quotes. The columns: {', '.join(scores.columns)}."
counts = scores.groupby(group_column).agg(returns_in_group=("case_id", "size"), on_the_list=("on_list", "sum"))
print(counts.to_string())
```

### Step 4. Compute each rate, and state the result

"The *selection rate* of a group is the share of its returns that the model puts on the list." (the day's reading)

1. In cell 5, type in place of each `None` the channel's rate as a division: its returns on the list, divided by its returns in `model_valid`, such as `3 / 50`.
2. Run cell 5. You should see each rate and the gap, then the same at two other thresholds (compare the list of 50 with your guess), then the past labels by channel.
3. Between the quotes after `result`, write the result in one sentence. The table says what it holds.

| Part | Give | Read it from |
|---|---|---|
| each channel's rate | with its counts beside it: on the list, of its returns | cells 4 and 5 |
| the threshold | 0.844 | cell 2 |
| the gap | the difference between the two rates | cell 5 |
| the end | that the gap needs a reason | "A gap is a result that needs a reason." (the day's reading) |

4. Run cell 5 again. You should see your sentence.

The labels' line is why a gap needs a reason: "So a gap in the list can come from the labels as well as from the model." (the day's reading) Do not call the model fair or unfair from one number.

<details><summary>Stuck? One way to start</summary>

At a threshold of 0.844, the model puts ... of channel A's ... returns on the list, ..., and ...

</details>

Notebook cell 5:

```python
# 5. Each channel's rate as a division, then the result in one sentence.
rate_a = None  # channel A: its returns on the list, divided by its returns in model_valid
rate_b = None  # channel B: the same
result = ""    # one sentence: each rate with its counts, the threshold, the gap, and that it needs a reason

if rate_a is None or rate_b is None:
    print("Type each rate as a division, then run this cell again.")
else:
    print(f"Channel A: {rate_a:.3f}. Channel B: {rate_b:.3f}. The gap: {abs(rate_a - rate_b):.3f}, "
          f"at a threshold of {threshold:.3f}.")
    if str(guess_50).strip().lower() in ("wider", "narrower"):
        compare()
    else:
        print("Type your guess in cell 3 and run it, then run this cell again to see the comparison.")
print(result or "Write your sentence between the quotes.")
```

### Step 5. Check

Run cell 6. It prints `Done:`, or `Not yet:` and what is left.

Keep this notebook. Friday's challenge asks for the result sentence of cell 5.

Notebook cell 6:

```python
# 6. Check. You do not need to read this code. It runs check.py, beside this notebook.
_check = {"__name__": "check"}
exec(open(f"{folder}/check.py", encoding="utf-8").read(), _check)
check_now = _check["in_notebook"]
print(check_now(globals()))
```

### Step 6. Proxy step: goes with the proxy segment

Could one feature stand in for the channel? "A *proxy* is a feature that names no group and still tracks one. You see it when the feature's typical value differs between the groups." (the day's reading) A group's *typical value* is its median: half its returns lie below it, half above.

1. Run cell 7. You should see the median of `other_income_amt`, the feature at the top of exercise 1's ranked list, in each channel, over all 1,000 returns.
2. Between the quotes, write one sentence: `other_income_amt could stand in for the channel, because ...`, or `could not stand in`. After `because`, give its median in each channel.
3. Run cell 7, then cell 6 again. Cell 6 now checks your sentence too.

"So your sentence says that the feature *could* stand in for the group, and the ranked list tells you how much the model leans on it." (the day's reading)

Notebook cell 7:

```python
# 7. Proxy step: other_income_amt's typical value in each channel, over all 1,000 returns. Then one sentence.
print(medians("other_income_amt"))
proxy_sentence = ""  # other_income_amt could stand in, or could not stand in, for the channel, because ...
print(proxy_sentence or "Write your sentence between the quotes.")
```

### Finished early? (proxy step)

Optional, with the proxy step: cell 6 does not check it.

1. Run cell 8. It compares `wages_amt`, third on exercise 1's ranked list, in the two channels, as cell 7 did.
2. Between the quotes, write one sentence of the same form for `wages_amt`.

Notebook cell 8:

```python
# 8. Optional, with the Proxy step: a second feature's typical value in each channel.
print(medians("wages_amt"))
second_sentence = ""
print(second_sentence or "Write your sentence between the quotes.")
```

### Afterwards, together

1. Why compare rates and not the counts on the list?
2. What does a gap need before anyone calls it unfair?
3. (proxy step) How can a list lean toward one channel without its column?

The answer, after you try: [`answers/exercise-02-selection-rates-by-group`](../answers/exercise-02-selection-rates-by-group/).
