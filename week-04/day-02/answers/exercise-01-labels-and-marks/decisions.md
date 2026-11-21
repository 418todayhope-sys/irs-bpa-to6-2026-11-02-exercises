# Exercise 1: decisions

The answer is `exercise.ipynb` beside this file, with the schema `course.learner_00`. This day follows Monday, so the training set is the course's shared `course.returns.model_train` and Monday's record is `model_feature_record`.

## Each answer, and why

| Cell | Answer | Why |
|---|---|---|
| 3 | `label_column = "exam_changed_tax"` | The course's label, as on Monday. `refund_held` is the demo's, made up. Cell 3 then prints `empty: 100`, `no: 320`, `yes: 80`, Monday's counts. An empty label is not a no: nobody examined that return, so its outcome is unknown. |
| 4 | decision: an examination's result, did it change the tax; made by: the course's data, made up in place of an examiner; when: after the return was filed, selected and examined | The three questions of a label's origin. `yes` means the examination changed the tax. The returns are synthetic, so no examiner made these labels: the course did. |
| 5 | `suspect = "account_change_amt"` | It is the one column beside the training set that no earlier day named. It agrees with the label on 399 of 400 labelled returns: above zero on 79 of the 80 `yes` returns, zero on all 320 `no` returns. The 80th `yes` return, `F-41683-2020`, had no change written when the course made the data. |
| 6 | table: `fields.csv`, beside the notebook; step: the examination, when it closed; when: after the case was selected, months later | What you have describes the column: the examination writes it when it closes, months after selection. The answer to when decides the mark of cell 7. The check accepts `fields` in the table, `exam` or `audit` in the step, and `after`, `clos`, `later` or `month` in when. |
| 7 | `before` for `tax_to_agi`, `wages_amt`, `deduction_amt`, `deduction_missing`; `after` for `tax_added_by_exam`, `account_change_amt`, each with a reason | The first four are Monday's record: each comes from the filed return, which exists before selection; Monday's flag only records whether line 12e was empty. The last two are written when an examination closes. `account_change_amt` is both ways of leakage at once: recorded after selection, and close to a copy of the decision. |
| 9, optional | one sentence | On the day the course made the data, no tax was written yet for that examination (What you have says the column is 0 then). |

What else the check accepts: any wording in cell 4 whose `decision` names the examination or an audit, or says whether the tax changed, whose `made by` says the course made it ("course", "synthetic", "made up", "simulated", "fake", "not real", "generated", "invented"), and whose `when` has two words or more; any reason of three words or more in cell 7; `Before` or `AFTER` in any case.

The trainer's demo, on the same notebook as the course gives it: cell 3 counts `refund_held`, 14 `no`, 2 `yes` and 484 empty (only 16 training returns carry the demo label: the empty ones show that an empty label is not counted). Its origin: decision, a refund was held for a review; made by, the course's data, made up in place of a refund reviewer; when, after the return was filed and before the refund was paid. Cell 5 puts `refund_hold_days` beside it: 16 of 16 agree, as the column only exists once a hold started. Cell 7's two demo rows: `filing_status` before (on the filed return), `refund_hold_days` after (counted from the day a hold started).

## Where each rule comes from (for the trainer, never in a learner cell)

| Quote in the exercise | From |
|---|---|
| "A return with an empty label is not counted as either outcome, and it is left out of training." | `days/day-02/curated.md` § Examine the labels before you train, Count the two outcomes; deck slide 12 |
| "A *label's origin* is the answer to those three questions: which past decision made the label, who made it, and when." | the same section, In general; deck slides 10 and 11 |
| "The course's data gives some returns a label that stands for an examination's result." | the same section, One return first |
| "A feature that matches the label almost exactly is a warning sign of leakage." | `days/day-02/curated.md` § Ask each feature when it was known, The warning sign; deck slide 16 |
| "No cell can answer the before-or-after question for you." | the same section, Who decides |
| "was it known before the case was selected? Mark each feature *before* or *after*. Train only on the features marked *before*." | the same section, The check; deck slide 15 |
| "When you find such a feature, trace it to where it came from: which table holds it, which step wrote it, and when." | the same section, The warning sign |

The exercise is deck slide 19; its debrief is slide 20; the demo is slide 18.

## Simulated data added

- `fields.csv`: for the 750 returns of `model_train` and `model_valid`, `account_change_amt`, the planted column recorded after selection: the tax the examination added (`tax_added_by_exam`) on a `yes` return, 0 on every other return, and 0 on one `yes` return of the training set (`F-41683-2020`), so the column matches the label almost exactly, not exactly.
- `refund-held.csv`: the demo label `refund_held` for 16 training returns, with `refund_hold_days`, above zero only on a held refund.

Each file says on its first line that the course made it up. Cell 2 joins them to the training set in the notebook only; no table changes. The week's setup note names `tax_added_by_exam` as Tuesday's planted leak; here learners struck it on Monday, so the check's surprise is the same fact under another name.

## Where it ran

- Not run in Databricks: a trainer gap. What only a run there confirms: `spark.table` on `course.returns.model_train` and `model_valid` for a learner in `training-students`, and `toPandas()` on them; `model_feature_record` in the learner's own schema; `pd.read_csv` on a CSV file in the Git folder by the path from `os.getcwd()`.
- Run on a laptop, 2026-11-20, Python 3.11.5, scikit-learn 1.3.0, pandas 2.0.3, on a stand-in for the Spark calls and for MLflow, over the course's week 4 tables made by `week-04/setup/day-01-course-data.py`, and again with no table readable (load.py makes the same rows from that file): start `Not yet: 6 left.`, answer `Done:` on both. The same numbers as the week's setup measurement, which used scikit-learn 1.5.2.

## Modelled on

None found that is proven. The outline's reference: Databricks documentation, Point-in-time feature joins, https://docs.databricks.com/aws/en/machine-learning/feature-store/time-series, a documentation page with no task. Shape: classify each item and give the reason (Monday's exercise 1), after a given count.
