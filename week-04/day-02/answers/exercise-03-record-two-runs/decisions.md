# Exercise 3: decisions

The answer is `exercise.ipynb` beside this file. Its cell 5 holds the second run's name, as a learner leaves it; the first run is recorded with the name as the course gives it.

## Each answer, and why

| Cell | Answer | Why |
|---|---|---|
| 2 | run as given | The two candidates of exercise 2, fitted again with the same cells, so the run cell has them in memory. Split: train, tax years 2020 to 2021; validation, tax year 2022, 200 returns with a label. |
| 4 | run as given | It prints what each run should record, exercise 2's two accuracies as cell 2 fitted them again: `logistic-regression` 0.955, `decision-tree` 0.895. The record must show the same numbers; if it does not, a run recorded the wrong value. |
| 5, first run | `run_name = "logistic-regression"` | One run, with its model type (`LogisticRegression`, read from inside its pipeline by the given `model_name`), split and measure as parameters, its validation accuracy as a metric, and the fitted pipeline. |
| 5, second run | `run_name = "decision-tree"` | The second run, with `max_depth` 3 too. Only the name changes: the cell takes the candidate of that name. |
| 6 | run as given | `2 runs in your experiment`, then the newest run of each name: logistic-regression 0.955, decision-tree 0.895, both on the same split and measure, and `Not what cell 4 expects`, with what to do, where a run differs from cell 4. |
| 7 | `higher = "logistic-regression"` | Its run's validation accuracy is the higher, 0.955 against 0.895, on the same split and the same measure, so the comparison is fair. The learner names the higher; Wednesday judges the two candidates. |
| 9, optional | one sentence: when should it be retrained | For example: once a year, after the next tax year's examinations close and give new labels. The reading's own answer is that the schedule follows the tax year; any reasoned schedule is a good answer. |

What else the check accepts: either run name when the two accuracies are equal; more than two runs in the experiment (the newest of each name counts), so a learner who runs cell 5 again loses nothing.

The trainer's demo (deck slide 44) records one run with cell 5 as the course gives it: `logistic-regression`, validation accuracy 0.955, split "train: tax years 2020 to 2021; validation: tax year 2022, 200 returns with a label", measure accuracy. Then the Experiments page, then the retraining job, which the learners watch.

## Where each rule comes from (for the trainer, never in a learner cell)

| Quote in the exercise | From |
|---|---|
| the run cell's lines `start_run`, `log_param`, `log_metric`, `mlflow.sklearn.log_model(model, name="model")` | `days/day-02/curated.md` § Record each training run in MLflow, its plain form; deck slide 38 |
| "Two runs compare fairly only when they use the same data, the same split and the same measure." | `days/day-02/curated.md` § Read a run's record on the Experiments page, Compare two runs |
| ticking both runs and clicking Compare | the same section, Compare two runs; deck slide 40 |
| "*Retraining* is the same recorded training, run again on newer labelled returns, on a schedule." | `days/day-02/curated.md` § Say when the model is trained again, In general; deck slide 42 |

The exercise is deck slide 45; its debrief is slide 46; the demo is slide 44.

The run records `split` and `measure` as parameters, and cell 6 reads the runs back with `mlflow.search_runs`, because the Checkpoint `thin-air-cp-124b1089`, "Record training runs and compare them from the record", asks that each run records the split and the measure it was measured on, and that the comparison reads its numbers from the runs. The day's reading names the fair-comparison rule only and does not show `search_runs`; cell 6 is given code, and its step says what it prints.

## What version 2.0 cut, and why

Learners of this cohort could not finish some exercises. Same goal and same check; 15 minutes instead of 20.

- Typed values, 5 to 2: the run name and the name of the higher run. Cell 4 no longer asks for exercise 2's two accuracies from the other notebook: it prints them from cell 2's fits, and cell 6 compares each run with them. Cell 7 no longer asks for a sentence of why: it prints both runs' accuracies beside the name.
- Steps, 9 to 5: cells 1 to 4, which the learner only runs, are one step, and cells 7 and 8 are one step.
- The quote on experiments is cut; the reading and the trainer's demo give it.

## Simulated data added

None beyond the course's tables.

## Where it ran

- Not run in Databricks: a trainer gap. What only a run there confirms: `spark.table` on `course.returns.model_train` and `model_valid` for a learner in `training-students`, and `toPandas()` on them; MLflow's `set_experiment`, `log_param`, `log_metric`, `sklearn.log_model` and `search_runs` on the workspace.
- Run on a laptop, 2026-11-21, Python 3.11.5, scikit-learn 1.3.0, pandas 2.0.3, on a stand-in for the Spark calls and for MLflow, over the course's week 4 tables made by `week-04/setup/day-01-course-data.py`, and again with no table readable (load.py makes the same rows from that file): start `Not yet: 2 left.` (the notebook as given records the run `logistic-regression`), answer `Done:` on both. The same numbers as the week's setup measurement, which used scikit-learn 1.5.2.

## Modelled on

None found that is proven. Closest: Microsoft Learn, "Use MLflow in Azure Databricks", its unit "Exercise - Use MLflow in Azure Databricks", https://learn.microsoft.com/en-us/training/modules/mlflow-azure-databricks/5-exercise-use-mlflow (opened 2026-11-20; 45 minutes; "you'll run an MLflow experiment and use MLflow to deploy a model"). No count of learners on the page. Shape taken: run a given logging cell, then read the runs in the workspace. Its deployment part is not used.
