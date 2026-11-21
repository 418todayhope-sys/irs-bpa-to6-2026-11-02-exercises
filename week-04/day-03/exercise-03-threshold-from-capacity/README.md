# Exercise 3: Say which error costs more, then read the threshold from capacity

**Goal:** After this exercise, you can state which error costs more in a scenario and set a threshold from the examiners' capacity, with the error it trades.

**Scenario:** A made-up examination group can take three cases for every ten returns scored. Its manager asks where the line goes, what it misses, whether case #123 makes the list, and which candidate to use. You write which error costs more for this group and why, read the threshold from its capacity, then choose the candidate.

**AI tool:** none.

**Time:** 25 minutes: 21 to do it, 4 for setup problems.

**Where you work:** this notebook, `week-04/day-03/exercise-03-threshold-from-capacity/exercise.ipynb`, in your Git folder of the course in Databricks.

**Done when:** cell 8 prints `Done:`.

**Optional part:** Finished early? Cell 9 reads the threshold on every validation return.

**What you have:**

- `ten-returns.csv`: exercise 1's ten made-up returns, scored by Tuesday's logistic regression.
- `case-123.csv`: Monday's made-up case #123, score 0.82, no label.
- `candidate-counts.csv`: the four counts at 0.5 of exercise 2.
- `validation-scores.csv`: exercise 1's 1,000 made-up validation returns, for the optional part.
- `load.py` and `check.py`: given code that cells 1 and 8 run.

**Start from:** this notebook as the course gives it, attached to the compute the trainer names. No cell reads or changes your tables. New here: a threshold set from capacity.

### Step 1. Read the files and the group's capacity

Run cell 1. You do not need to read its code. You should see `Read the ten returns, case #123 and exercise 2's counts`, then the group's capacity.

The group's two errors, in its terms:

- **A false positive:** one of its examiners spends hours on a return whose examination changes nothing, and the taxpayer goes through an examination that changes nothing.
- **A false negative:** a return whose tax an examination would have changed stays as filed, and nobody looks at it.

Notebook cell 1:

```python
# 1. The made-up files, and the group's capacity. You do not need to read this code.
import os
folder = os.getcwd().split("/week-04/")[0] + "/week-04/day-03/exercise-03-threshold-from-capacity"
exec(open(f"{folder}/load.py", encoding="utf-8").read())  # reads the files beside this notebook
print("The group can take 3 cases for every 10 returns scored.")
```

### Step 2. Say which error costs more, and why

Why: "No model or formula makes that judgment. The people who run the examinations make it, and you must be able to state it in a sentence and defend it." (the day's reading)

1. In cell 2, between the quotes, write one sentence in this form: `In this scenario a false negative costs more than a false positive, because ...`, or the other way round. After `because`, give a reason of five words or more: who bears the cost, and what it is. Either direction can be right for this group; the reason is what a reviewer questions.
2. Run cell 2. You should see your sentence.

Notebook cell 2:

```python
# 2. Which error costs more for this group, and why. One sentence.
cost_sentence = ""
print(cost_sentence or "Write your sentence between the quotes.")
```

### Step 3. Sort the ten returns, and take the top k

Why: "A *threshold from capacity* is the score of the k-th return when the returns are sorted by score, highest first." (the day's reading) *Capacity* is the number of cases the examiners can take, written k.

1. In cell 3, type in place of `None` the number of cases the group can take from these ten returns.
2. Run cell 3. The SQL sorts the ten by score, highest first, and keeps the top k. You should see k returns.

Notebook cell 3:

```python
# 3. Sort the ten returns, highest score first, and keep the top k. Change only k.
k = None  # the cases the group can take from these ten returns
assert isinstance(k, int), "Type a whole number in place of None."
spark.sql(f"SELECT return_id, score, label FROM ten_returns ORDER BY score DESC LIMIT {k}").show()
```

### Step 4. Read the threshold, count, and name the error it trades

Why: "every return at or above" the k-th score goes on the list (the day's reading).

1. In cell 4, type in place of each `None`:
   - `threshold`: the score of the last return cell 3 printed, the k-th.
   - `listed_change`: how many of the k returns are labelled change.
   - `missed_change`: how many of the ten returns labelled change are not among the k.
2. Between the quotes, write which error this threshold trades, compared with 0.5. At 0.5 the ten had 2 false positives and 1 false negative (exercise 1). Use the words `false positive` and `false negative`.
3. Run cell 4. You should see your answers.

Notebook cell 4:

```python
# 4. The threshold read from cell 3, the counts, and the error it trades.
threshold = None      # the k-th score
listed_change = None  # how many of the k are labelled change
missed_change = None  # how many returns labelled change are not on the list
trade = ""            # compared with 0.5: which error goes down, which goes up
print("Threshold:", threshold, "| labelled change on the list:", listed_change, "| missed:", missed_change)
print("The trade:", trade or "(not yet)")
```

### Step 5. Put case #123 in with the ten

Why: case #123 is "scored in the same batch as the ten, and the examiners still take three cases" (the day's reading).

Run cell 5. It sorts the eleven returns, the ten and case #123, with your k of cell 3. You should see k returns. Case #123's label says `no label yet`: it has not been examined.

Notebook cell 5:

```python
# 5. The same sort, on the ten returns and case #123, with the same k.
assert isinstance(k, int), "Type k in cell 3 and run it first."
spark.sql(f"SELECT return_id, score, label FROM today_batch ORDER BY score DESC LIMIT {k}").show()
```

### Step 6. Say what case #123 changes

1. In cell 6, type in place of `None` the new threshold, the k-th score of cell 5. Between the quotes, type the return that case #123 pushes off the list, such as `#311`.
2. Run cell 6. You should see your answers.

Case #123 has no label, so it is in no cell. Nobody can say yet whether it is a true positive. The list is a recommendation, and a person decides.

Notebook cell 6:

```python
# 6. What case #123 changes.
threshold_with_123 = None  # the k-th score of cell 5
pushed_off = ""            # the return case #123 pushes off the list
print("New threshold:", threshold_with_123, "| pushed off:", pushed_off or "(not yet)")
```

### Step 7. Choose the candidate, and say why

Why: "If a missed return costs more, recall matters more to you", and if the wasted hours cost more, precision does (the day's reading).

1. Run cell 7. You should see exercise 2's counts at 0.5 for the baseline and both candidates.
2. In cell 7, type the candidate you put in front of the manager: `logistic-regression` or `decision-tree`. Choose by your sentence of cell 2.
3. Between the quotes, write why, in one sentence of eight words or more, with the counts.
4. Run cell 7 again. You should see your choice.

The ten returns were scored by the logistic regression, so its threshold is the one of cell 6. The tree's threshold is read from its own scores: the optional part does it.

Notebook cell 7:

```python
# 7. Exercise 2's counts, then your choice.
for _, c in counts.iterrows():
    tp, fp, fn = c["true_positives"], c["false_positives"], c["false_negatives"]
    precision = f"{tp} of {tp + fp}" if tp + fp else "cannot be computed"
    print(f"{c['candidate']}: on the list {tp + fp}, precision {precision}, recall {tp} of {tp + fn}")

chosen = ""  # logistic-regression or decision-tree
why = ""     # one sentence, with the counts
print("Chosen:", chosen or "(not yet)", "| why:", why or "(not yet)")
```

### Step 8. Check

Run cell 8. It prints `Done:`, or `Not yet:` and what is left.

Keep this notebook. Thursday starts from the candidate you chose and the threshold of cell 6.

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

1. In cell 9, type in place of `None` how many of the top k of cell 3 are labelled change, as a division, such as `1 / 3`. This share is called precision at k.
2. Run cell 9. It reads the threshold of the candidate you chose on all 1,000 validation returns, at 3 cases for every 10: k is 300. The list never holds more than 300 returns, because that is all the examiners can take. When returns share the k-th score, not all of them may fit: you break the tie and write down how. Cell 9 takes the tied returns in order of their ID; in `tie_rule`, type which of the tied returns you would put on the list, and why. You should see the threshold, how many returns share it, how many returns are on the list, and how many of them are labelled change.

Notebook cell 9:

```python
# 9. Optional: precision at k on the ten, then the threshold on all 1,000 validation returns.
precision_at_k = None  # on the ten: how many of the top k are labelled change, divided by k
print("Precision at k on the ten, yours:", precision_at_k if precision_at_k is None else round(precision_at_k, 2))

k_all = 300  # 3 cases for every 10 of the 1,000 returns: the list never holds more
tie_rule = None  # if returns share the k-th score: which of them go on the list, and why
if chosen in valid:
    kth = valid[chosen].sort_values(ascending=False).iloc[k_all - 1]
    above = valid[valid[chosen] > kth]
    tied = valid[valid[chosen] == kth]
    room = k_all - len(above)
    on_list = pd.concat([above, tied.sort_values("return_id").head(room)])  # the tie broken by return ID
    print(f"{chosen}: threshold {kth}; {len(tied)} returns share it, room for {room} of them")
    print(f"{len(on_list)} returns on the list, {(on_list['label'] == 'change').sum()} of them labelled change, of 50")
    print("How you broke the tie:", tie_rule)
else:
    print("Type your choice in cell 7 and run it first.")
```

### Afterwards, together

1. What would change your sentence on which error costs more?
2. Why read the threshold from capacity, not keep 0.5?
3. What can you say about case #123 here, and what not?

The answer, after you try: [`answers/exercise-03-threshold-from-capacity`](../answers/exercise-03-threshold-from-capacity/).
