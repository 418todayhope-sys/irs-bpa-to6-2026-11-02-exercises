# Exercise 2: decisions

The answer is `exercise.ipynb` beside this file. It reads the made-up `validation-scores.csv` beside the exercise's notebook, and writes into the learner's experiment `week-04-candidates`.

## Each answer, and why

| Cell | Answer | Why |
|---|---|---|
| 1 | higher recall `logistic-regression`, higher precision `decision-tree` | Any of the two names passes: a guess cannot be wrong. The reading's table already shows the made-up counts; the guess is still asked before the divisions, as the demo asks for it in the chat. |
| 2 | run as given | logistic regression 22, 16, 28, 934; tree 8, 5, 42, 945; sum 1 is 1000, sum 2 is 50 for both. |
| 3 | `22 / 38`, `22 / 50`; `8 / 13`, `8 / 50` | Precision divides TP by all returns on the list (TP + FP); recall by all returns labelled change (TP + FN). 0.579 and 0.44; 0.615 and 0.16. A decimal within 0.006 passes; a swapped pair gets its own `Not yet` line. |
| 4 | run as given | It finds the newest run of each name in `/Users/<user>/week-04-candidates` and adds the parameters `threshold` (0.5) and `rows`, and the metrics `precision`, `recall` and `accuracy_same_rows` (0.956, 0.953). With no run of that name, it makes one and says so. |
| 5 | run as given | `Done:` when the guess is typed, the four divisions are right, and each newest run holds them with `threshold` 0.5. |
| 6, optional | `0 / 50` | The baseline lists nothing: recall 0. Its precision is 0 of 0, which cannot be computed: an empty list, not a precision of zero. |

Two things for the trainer to know:

- The runs already hold Tuesday's `validation_accuracy`, measured on each learner's own week 3 validation rows (4 labelled returns). Today's numbers are counted on the course's 1,000 made-up returns, so the accuracy on those rows is logged as `accuracy_same_rows`, and `rows` says which rows. MLflow refuses to change a logged parameter, so nothing of Tuesday's is overwritten. Say it: a comparison is fair only on the same rows.
- Running cell 4 again with the same divisions is accepted (the same parameter values). With other divisions the metrics take the new values; the parameters are unchanged.

## Where each rule comes from (for the trainer, never in a learner cell)

| Quote in the exercise | From |
|---|---|
| "*Precision* is the share ..."; "*Recall* is the share ..." | `days/day-03/curated.md` § Count precision and recall from the matrix, In general |
| "the comparison still comes from the record" | the same section, Two candidates, side by side |

The exercise is deck slide 32; its debrief is slide 33; the demo is slide 31.

## Simulated data added

`validation-scores.csv`, exercise 1's file.

## Where it ran

- Not run in Databricks or MLflow: a trainer gap. What only a run there confirms: `mlflow.start_run(run_id=...)` resuming Tuesday's finished run and adding parameters and metrics; `search_runs` ordering newest first; the Experiments page's **Compare** showing the new metrics; `spark.sql("SELECT current_user()")`.
- Run on a laptop, 2026-11-20, Python 3.11.5, scikit-learn 1.3.0, on stand-ins for Spark and MLflow (seeded with Tuesday's two runs, and again with none): start `Not yet: 3 left.`, answer `Done:` both ways. Rerun after the cold read (a decimal accepted, the newest run named, the rows of Tuesday's accuracy said). `check.py --start` prints `Not yet: 3 left.` and `check.py values.json` prints `Done:` on Python 3.10 and 3.11.

## Pages behind the steps (opened 2026-11-20)

| Step | Page |
|---|---|
| precision and recall | Google Machine Learning Crash Course, https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall (a Source of the day's reading) |
| logging into a run | MLflow Tracking documentation (a Source of Tuesday's reading) |

## Modelled on

None found that is proven. Closest: the outline's reference, the Google page above, whose "Check your understanding" items ask which measure to use for a stated cost. Shape taken: predict, count with given code, divide by hand, record.
