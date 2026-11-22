# Exercise 3: For two scenario tasks, choose a trained model or prompting, and give the reason (we do)

**Goal:** After this exercise, you can choose a trained model or prompting for a task and defend the choice by labelled data, cost or risk.

**Scenario:** Two made-up teams each bring a task and ask for a model. A collection team wants to know which new payment plans are likely to be missed. It holds many past plans, each labelled kept or missed. A training team wants each new procedure page summarized in five lines for new staff. It holds no store of approved summaries. For each task you answer the three questions, choose a trained model or prompting, and name the row that decided it.

**AI tool:** none. This is thinking work: you type text in this notebook. No cell reads a table, and no cell calls a model.

**Time:** 15 minutes: 13 to do it, 2 for setup problems.

**Where you work:** this notebook, `week-04/day-04/exercise-03-choose-the-technique/exercise.ipynb`, in your Git folder of the course in Databricks.

**Done when:** cell 4 prints `Done:`.

**Cells:** a cell's number is the `# N.` comment on its first line, such as `# 1.`; text cells have no number.

**Optional part:** Finished early? Cell 5 asks where retrieval would help.

**What you have:** the two tasks above; case #123's two tasks as a filled table, which cell 1 prints; and `check.py`, which cell 4 runs. The check reads only your text.

**Start from:** this notebook as the course gives it, attached to the compute the trainer names. New here: the two tasks.

### Step 1. Read the filled table

"Both tools serve the same return, each for its own task. Your answer in the exercise takes the same form: a choice, and the row that decided it." (the day's reading)

Run cell 1. You should see case #123's two tasks from the day's reading, each answered in the three rows, then the choice. The last line, the row that decided it, is the form your answer takes.

The three questions, in order:

- **Labelled data:** are there many past cases, each with a label? A trained model cannot exist without them.
- **Cost:** what does it cost to build, and to run? Count the calls.
- **Risk:** what does a wrong answer cost, and who catches it?

Notebook cell 1:

```python
# 1. Case #123's two tasks, decided by the three questions (the day's reading's table, and the deciding row).
filled = """
                Score every return, to choose which      Write the two sentences that explain
                ones the examiners see first             one return on the list
Labelled data:  yes: past examined returns               no: no store of approved explanations
Cost:           training once; then each score costs     nothing to train; a small cost for each call,
                almost nothing, for every return filed   and only the returns on the list need one
Risk:           examiner hours on a return labelled      the text may state a reason that is not in
                no change, or a missed return labelled   the chart; a person checks the sentences
                change; the examiner reviews each case   against the chart
                on the list
Choice:         a trained model                          prompting
Decided by:     labelled data: many past examined        labelled data: no approved explanations
                returns, each with its label             to learn from
"""
print(filled)
```

### Step 2. Decide the payment plans

A trained model needs many past cases, each with its label. Prompting needs a written instruction and few labelled cases or none.

The task: **the collection team wants to know which new payment plans are likely to be missed. It holds many past plans, each labelled kept or missed.** It gets many new plans each week, and a person on the team calls each taxpayer whose plan the model picks out.

1. In cell 2, after `Labelled data:` and `Risk:`, write a few words, three or more. The `Cost:` line is given.
2. After `Choice:`, write `a trained model` or `prompting`.
3. After `Decided by:`, write one sentence of six words or more: the row that decided it, as in cell 1, and why.
4. Run cell 2. You should see your table.

A reason such as "it is newer" or "it handles text" is not yet a reason. It belongs to none of the three rows.

Notebook cell 2:

```python
# 2. The payment plans. Write after each colon, one line each.
payment_plans = """
Labelled data:
Cost: training once; then each new plan costs almost nothing to run, every week
Risk:
Choice:
Decided by:
"""
print(payment_plans)
```

### Step 3. Decide the summaries

"A language model fails in its own way: as week 1 showed, it can write an answer that reads well and is invented." (the day's reading)

The task: **the training team wants each new procedure page summarized in five lines for new staff. It holds no store of approved summaries. Each page has an owner who wrote it.**

1. In cell 3, write the three rows, the choice and the deciding row, as in step 2. In the `Risk:` line, say who catches a wrong summary, and against what.
2. Run cell 3. You should see your table.

Notebook cell 3:

```python
# 3. The summaries. Write after each colon, one line each.
summaries = """
Labelled data:
Cost:
Risk:
Choice:
Decided by:
"""
print(summaries)
```

### Step 4. Check

Run cell 4. It prints `Done:`, or `Not yet:` and what is left. It checks that each line is there, the choice that labelled data points to, and that each reason uses the words of a row: `label`, `cost` or `calls`, or `risk`. It does not judge your reasons: the discussion afterwards does.

Keep this notebook. Friday's challenge asks for your choice for two tasks.

Notebook cell 4:

```python
# 4. Check. You do not need to read this code. It runs check.py, beside this notebook.
import os
folder = os.getcwd().split("/week-04/")[0] + "/week-04/day-04/exercise-03-choose-the-technique"
_check = {"__name__": "check"}
exec(open(f"{folder}/check.py", encoding="utf-8").read(), _check)
check_now = _check["in_notebook"]
print(check_now(globals()))
```

### Finished early?

Optional: cell 4 does not check it.

*Retrieval* gives the language model documents to read before it answers. You build it in week 6. In cell 5, between the quotes, write one sentence: for one of the two tasks, where would retrieval help, and why?

Notebook cell 5:

```python
# 5. Optional: where would retrieval help?
retrieval = ""
print(retrieval or "Write your sentence between the quotes.")
```

### Afterwards, together

1. Which of the three questions decided each task?
2. What would change your choice for the summaries?
3. Who checks a wrong answer in each task, and how?

The answer, after you try: [`answers/exercise-03-choose-the-technique`](../answers/exercise-03-choose-the-technique/).
