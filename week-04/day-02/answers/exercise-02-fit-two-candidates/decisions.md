# Exercise 2: decisions

The answer is `exercise.ipynb` beside this file. Its cell 6 holds the tree's line, as a learner leaves it after the second fit; the logistic regression is fitted first, with the line as the course gives it. It uses the course's shared tables, the setup's twelve features, and its fill (an empty amount read as 0, with `deduction_missing`).

## Each answer, and why

| Cell | Answer | Why |
|---|---|---|
| 3 | run as given | 400 training rows and 200 validation rows have a label. The twelve features are the ones the week's setup measures: every amount of the filed return, two ratios and Monday's flag, each known before selection. An empty amount is read as 0: 20 deduction lines in training and 16 in validation. |
| 4 | run as given | It prints 200 validation returns with a label, so one return moves the accuracy by 0.005, and their labels, 160 `no` and 40 `yes`. Monday's baseline is 0.8 on the training rows and 0.8 on the validation rows, so on these tables the two agree; the cell still measures it on the rows the candidates are measured on. |
| 5 | any of `a lot` (twenty or more returns more right, of 200), `a little` (one to nineteen), `not better` (the same or fewer) | A guess cannot be wrong; the check asks only that it is written before the fit, in any case, with or without a full stop. The answer types `a lot` and `a little`. |
| 6, first run | `make_pipeline(StandardScaler(), LogisticRegression())` | Validation accuracy 0.955, 191 of 200 right. Unscaled, `LogisticRegression()` did not converge on these dollar amounts (The week's setup note). |
| 6, second run | `DecisionTreeClassifier(max_depth=3, random_state=0)` | Validation accuracy 0.895, 179 of 200 right. `random_state=0` makes every run grow the same tree, so exercise 3's record matches. The week's setup measured the full-depth tree at 0.910; Monday says Tuesday keeps the tree short. |
| 7 | run as given | baseline 0.800, LogisticRegression 0.955, DecisionTreeClassifier 0.895. |
| 9, optional | one sentence | The logistic regression got 31 more right than the baseline, the tree 19 more. |

What else the check accepts: nothing else in cell 6; the tree must have `max_depth=3`; the check fits both candidates again on the notebook's rows and compares. A learner who adds `tax_added_by_exam` to the features list gets `Not yet` (cell 3's list), and an accuracy near 1.000 if they look: the debrief's third question.

The trainer's demo (slide 31) fits the logistic regression on the same notebook and stops there; the tree is the learners'.

## Where each rule comes from (for the trainer, never in a learner cell)

| Quote in the exercise | From |
|---|---|
| "Train only on the features marked *before*." | `days/day-02/curated.md` § Ask each feature when it was known, The check |
| "it is important that these transformations are only learnt from the training data" | `days/day-02/curated.md` § Fit two models from the given cells, The fill and the scaling, named only (the scikit-learn guide's words) |
| "A comparison is fair only on the same rows." | the same section, Compare with the baseline; deck slide 29 |
| the cells' plain form: `features`, `notna()`, `X_train`, `model = LogisticRegression()`, `fit`, `predict`, `accuracy_score`, `DecisionTreeClassifier(max_depth=3)` | the same section, In general; deck slide 28 |
| "If a candidate does not beat the baseline, write that down. It is a result." and the accuracy close to 1.000 (our words) | the same section, Compare with the baseline |

The exercise is deck slide 32; its debrief is slide 33; the demo is slide 31.

## Simulated data added

None beyond the course's tables. Cell 2 adds `deduction_missing` to `train` and `valid` in the notebook by Monday's rule; no table changes.

## Where it ran

- Not run in Databricks: a trainer gap. What only a run there confirms: `spark.table` on `course.returns.model_train` and `model_valid` for a learner in `training-students`, and `toPandas()` on them; scikit-learn on the course compute giving 0.955 and 0.895.
- Run on a laptop, 2026-11-20, Python 3.11.5, scikit-learn 1.3.0, pandas 2.0.3, on a stand-in for the Spark calls and for MLflow, over the course's week 4 tables made by `week-04/setup/day-01-course-data.py`, and again with no table readable (load.py makes the same rows from that file): start `Not yet: 2 left.` (the logistic regression is fitted when the cells run as given), answer `Done:` on both. The same numbers as the week's setup measurement, which used scikit-learn 1.5.2.

## Pages behind the steps (opened 2026-11-20)

| Step | Page |
|---|---|
| fit, predict | scikit-learn, Getting Started, https://scikit-learn.org/stable/getting_started.html (a Source of the day's reading) |
| the fill | scikit-learn, Common pitfalls and recommended practices, https://scikit-learn.org/stable/common_pitfalls.html (a Source of the day's reading) |

## Modelled on

None found that is proven. Closest: Microsoft Learn, "Train a machine learning model in Azure Databricks", its unit "Exercise - Train a machine learning model in Azure Databricks", https://learn.microsoft.com/en-us/training/modules/machine-learning-azure-databricks/7-exercise-machine-learning-databricks (opened 2026-11-20; 45 minutes; "you'll use a Databricks Spark cluster to train a classification model"). No count of learners on the page. Shape taken: run given cells, change one line, compare. Its case (Spark MLlib on its own data) is not used.
