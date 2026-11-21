# Exercise 3: Record both fits as runs and compare them from the record

**Goal:** After this exercise, you can record two fits as MLflow runs and choose a candidate from what the runs recorded.

**Scenario:** Three weeks from now, someone asks which of your two models was better, on which rows, and with which settings. You record both fits now, so that the answer comes from the record and not from memory.

**AI tool:** none.

**Time:** 20 minutes: 15 to do it, 5 for setup problems.

**Where you work:** this notebook, `week-04/day-02/exercise-03-record-two-runs/exercise.ipynb`, in your Git folder of the course in Databricks, then the workspace's Experiments page.

**Done when:** cell 8 prints `Done:`.

**Optional part:** Finished early? Cell 9 asks for one sentence on retraining.

**What you have:**

- `course.returns.model_train` and `course.returns.model_valid`, the course's training and validation sets. If they cannot be read, cell 2 makes the same rows from the course's setup file and says so.
- Your notebook of exercise 2, with the two candidates' accuracies in cell 7.
- `load.py` and `check.py`: given code.

**Start from:** this notebook as the course gives it, on the trainer's compute. No cell changes your tables. New here: a run and its record.

### Step 1. Find the course's files

Run cell 1. You should see your Git folder of the course.

Notebook cell 1:

```python
# 1. Where the course's files are. You do not need to change this cell.
import os
course_folder = os.getcwd().split("/week-04/")[0]  # your Git folder of the course
print("Your Git folder of the course:", course_folder)
```

### Step 2. Fit the two candidates again

Why: this is a new notebook, so the two candidates of exercise 2 are not in its memory yet.

Run cell 2. You should see the two candidates' names, then the split: the tax years of the training and validation rows.

Notebook cell 2:

```python
# 2. Fit the two candidates of exercise 2 again, with the same cells.
try:
    train_table = spark.table("course.returns.model_train")  # the training set
    valid_table = spark.table("course.returns.model_valid")  # the validation set
except Exception:
    train_table, valid_table = None, None  # not there: load.py makes the same rows

# You do not need to read the lines below. They run load.py.
folder = f"{course_folder}/week-04/day-02/exercise-03-record-two-runs"
exec(open(f"{folder}/load.py", encoding="utf-8").read())
candidates, split = fit_candidates(train_table, valid_table)
```

### Step 3. Open your experiment

Why: "An experiment groups together runs and models for a specific task." (the day's reading)

Run cell 3. You should see your experiment's name. Cell 3 makes it the first time.

Notebook cell 3:

```python
# 3. Your experiment: the place that holds your runs for this task.
import mlflow

user = spark.sql("SELECT current_user()").first()[0]  # your user name in the workspace
experiment = f"/Users/{user}/week-04-candidates"
mlflow.set_experiment(experiment)  # makes the experiment the first time; after that it opens it
print("Your experiment:", experiment)
```

### Step 4. Write down what exercise 2 showed

Why: each run must show the same number as exercise 2. If one does not, something changed between the two.

1. Open your notebook of exercise 2 and find cell 7's two accuracies. If cell 7 shows nothing, run its cells 1 to 7 again, with cell 6 once for each model.
2. In cell 4, type each one in place of `None`, as a number from 0 to 1, such as `0.9`.
3. Run cell 4.

Notebook cell 4:

```python
# 4. Each candidate's validation accuracy, from exercise 2, cell 7.
from_exercise_2 = {
    "logistic-regression": None,  # LogisticRegression's accuracy, such as 0.9
    "decision-tree": None,        # DecisionTreeClassifier's accuracy
}
print("From exercise 2:", from_exercise_2)
```

### Step 5. Record each candidate as a run

Why: a run keeps what produced a number. "Two runs compare fairly only when they use the same data, the same split and the same measure." (the day's reading) So beside the model's settings, each run records two facts about the comparison: the split and the measure.

1. Run cell 5. You should see `Recorded the run logistic-regression`.
2. In cell 5, type `decision-tree` in place of `logistic-regression`. Change nothing else.
3. Run cell 5 again. You should see `Recorded the run decision-tree`.

MLflow may print red warnings and a link to the run while it saves the model. They are normal: the run is recorded when `Recorded the run` prints. Do not run the cell again because of them.

In cell 5:

1. `start_run` starts a run with your run name.
2. Each `log_param` writes one *parameter*, a name and its value, into the run: the model type, the tree's depth, the split and the measure. `model_name` is given code: it reads `LogisticRegression` from inside its pipeline.
3. `log_metric` writes the measured result, the validation accuracy.
4. `log_model` saves the fitted model with the run.

Notebook cell 5:

```python
# 5. Record one candidate as a run. Change only the run name, then run this cell again.
run_name = "logistic-regression"  # the first run. For the second run, type decision-tree in its place.

assert run_name in candidates, "Type logistic-regression or decision-tree as the run name."
model, accuracy = candidates[run_name]
with mlflow.start_run(run_name=run_name):
    mlflow.log_param("model_type", model_name(model))     # a setting of the model: its type
    if run_name == "decision-tree":
        mlflow.log_param("max_depth", model.max_depth)    # the tree's setting
    mlflow.log_param("split", split)                      # a fact you record: the rows it was fitted and measured on
    mlflow.log_param("measure", "accuracy")               # a fact you record: how it was measured
    mlflow.log_metric("validation_accuracy", accuracy)    # a measured result
    mlflow.sklearn.log_model(model, name="model")         # the fitted model itself
print("Recorded the run", run_name)
```

### Step 6. Read both runs back

Why: the comparison comes from what the runs recorded, not from memory.

In cell 6:

1. `search_runs` reads every run in your experiment, newest first. If you ran cell 5 more than twice, there are more than two.
2. MLflow keeps each run's name in a *tag*, a note stored with the run: `tags.mlflow.runName`. `drop_duplicates` keeps the newest run of each name. Those are the two that count; you need not delete the others.
3. The loop prints, for each run, its model type, its measure and accuracy, and its split.
4. The last lines compare each run's accuracy with cell 4. You do not need to read them.

Run cell 6. You should see how many runs your experiment holds, then one line for each run name. If a line says `Not what cell 4 holds`, check exercise 2's cell 7 and your typing in cell 4, then run cells 4 and 6 again.

Notebook cell 6:

```python
# 6. Read both runs back from the experiment. The newest run of each name counts.
runs = mlflow.search_runs(experiment_names=[experiment])          # every run, newest first
newest = runs.drop_duplicates("tags.mlflow.runName") if len(runs) else runs  # the newest run of each name
print(len(runs), "runs in your experiment. The newest of each name:")
for _, run in newest.iterrows():
    name, recorded = run["tags.mlflow.runName"], run["metrics.validation_accuracy"]
    print(f"{name}: {run['params.model_type']}, {run['params.measure']} {recorded:.3f}. Split: {run['params.split']}")
    mine = from_exercise_2.get(name)  # you do not need to read this line and the three below
    mine = mine / 100 if isinstance(mine, (int, float)) and mine > 1 else mine
    if not isinstance(mine, (int, float)) or abs(mine - recorded) > 0.01:
        print(f"  Not what cell 4 holds ({from_exercise_2.get(name)}). Check exercise 2's cell 7 and your typing in cell 4.")
```

### Step 7. Compare the two runs on the Experiments page

1. In the workspace's left sidebar, select **Experiments**, then open `week-04-candidates`. You should see one row for each run.
2. Select the boxes of the two newest runs, one of each name, then select **Compare**. You should see each run's parameters and its `validation_accuracy`, the same numbers as cell 6.
3. Open one run and find its logged model. The trainer shows where it is.

This step has no cell, so from step 8 on, each step's cell is the one before its number: step 8 uses cell 7.

### Step 8. Choose a candidate, and say why

1. In cell 7, type the candidate you choose: the one whose run has the higher validation accuracy in cell 6. If the two are equal, choose either one.
2. Write why, in one sentence of six words or more, with the two accuracies cell 6 printed from the runs.
3. Run cell 7. You should see the accuracy its run recorded.

Notebook cell 7:

```python
# 7. The candidate you choose, and why. Take the numbers from cell 6.
chosen = ""  # type logistic-regression or decision-tree
why = ""     # one sentence, with the two accuracies the runs recorded

picked = newest[newest["tags.mlflow.runName"] == chosen] if len(newest) else newest
if len(picked):
    print(f"Its run recorded validation accuracy {picked['metrics.validation_accuracy'].iloc[0]:.3f}.")
print("Why:", why or "(not yet)")
```

### Step 9. Check

Run cell 8. It prints `Done:`, or `Not yet:` and what is left.

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

"*Retraining* is the same recorded training, run again on newer labelled returns, on a schedule." (the day's reading) In cell 9, between the quotes, write one sentence: when should this model be retrained? Then run cell 9.

Notebook cell 9:

```python
# 9. Optional: one sentence on retraining.
retrain = ""
print(retrain or "Write your sentence between the quotes.")
```

### Afterwards, together

1. Why compare from the runs and not from your notebook?
2. What makes two runs a fair comparison?
3. Asked for a model's parameters, what do you ask back?

The answer, after you try: [`answers/exercise-03-record-two-runs`](../answers/exercise-03-record-two-runs/).
