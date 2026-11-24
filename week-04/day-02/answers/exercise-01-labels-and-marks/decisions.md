# Exercise 1: decisions

The answer is `exercise.ipynb` beside this file. This day follows Monday, so the training set is the course's shared `course.returns.model_train`, and Monday's record, four rows, is given by `load.py` (nothing saved from Monday is read): `tax_to_agi`, `tax_added_by_exam` (struck on timing), `deduction_amt` and `deduction_missing`.

## Each answer, and why

| Cell | Answer | Why |
|---|---|---|
| 1, 2 | cells 1 and 2 run as given | Cell 2 prints Monday's record, four rows. |
| 3 | `label_column = "exam_changed_tax"` | The course's label. `refund_held` is the demo's, made up. Cell 3 then prints `no: 320`, `empty: 100`, `yes: 80`, Monday's counts. An empty label is not a no: nobody examined that return, so its outcome is unknown. |
| 4 | decision: an examination's result, did it change the tax; made by: the course's data, made up in place of an examiner; when: after the return was filed, selected and examined | The three questions of a label's origin. The returns are synthetic, so no examiner made these labels: the course did. |
| 5 | `beside = "fields.csv"` | Cell 5 puts both columns of the file beside the label: `days_to_file` agrees on 80 of 400 labelled returns (it is above zero on every return, so it agrees only with the 80 `yes`), `account_change_amt` on 399 of 400 (above zero on 79 of the 80 `yes`, zero on all 320 `no`). It then prints what the file says of each column: what it holds, not when it is written. |
| 6 | `suspect = "account_change_amt"`; step: the examination, when it closed | The column that matches the label almost exactly. The file's note says only that it is a change to the tax on the account; what writes such a change after filing is the learner's knowledge. The table is given: the file cell 5 read. Its timing is asked once, as the mark of cell 7. The check accepts `exam`, `audit`, `assess` or `adjust` in the step. |
| 7 | `file_marks`: (`account_change_amt`, `after`, written when the examination closed, months after selection); (`days_to_file`, `before`, the return is received before any selection) | The four rows of Monday's record are given, as Monday marked them. The first slot holds the column traced in cell 6; the learner types the other column's name from cell 5's count, so no cell names either column before the count. `account_change_amt` is written when an examination closes: both ways of leakage at once, recorded after selection and close to a copy of the decision, much the same fact as `tax_added_by_exam` under another name. `days_to_file` is written when the return is received. Train on: four; leave out: two. |
| 9, optional | one sentence | On the day the course made the data, no tax was written yet for that examination (the file's note says the column is 0 when the account showed no change on the day the course made the data). The `yes` return is `F-41683-2020`. |

What else the check accepts: any wording in cell 4 whose `decision` names the examination or an audit, or says whether the tax changed, whose `made by` says the course made it ("course", "synthetic", "made up", "simulated", "fake", "not real", "generated", "invented"), and whose `when` has two words or more; an examiner as `made by` too (the right answer for a real return), and the check then adds one line saying that in this course the labels themselves are made up by the course; any reason of three words or more in cell 7; `Before` or `AFTER` in any case.

The planted column stays hidden until the learner's own count shows it: no README step, no check message before cell 6 and no load code names it. The check takes it to be the column of `fields.csv` that agrees most with the label, counted by the check itself on the rows cell 2 loaded, so `check.py` does not hold its name. Only this folder and the `Done:` line name it.

The trainer's demo, on the same notebook as the course gives it: cell 3 counts `refund_held`, 14 `no`, 2 `yes` and 484 empty (only 16 training returns carry the demo label). Its origin: decision, a refund was held for a review; made by, the course's data, made up in place of a refund reviewer; when, after the return was filed and before the refund was paid. Cell 5 puts `refund_hold_days` beside it: 16 of 16 agree, and the file's note says it is counted from the day the hold started. Cell 7's two demo rows: `filing_status` before (on the filed return), `refund_hold_days` after (counted from the day a hold started).

## What version 2.0 cut, and why

Learners of this cohort could not finish some exercises. Same goal and same check; 15 minutes instead of 25.

- Typed values, 21 to 13: the four rows of Monday's record (and `wages_amt`, which is not in the new record) are given in cell 7, so the learner marks the two columns of the file only; the trace's table is given (it is the file cell 5 read); the leak's timing is asked once, as its mark, not again in the trace.
- Steps, 8 to 7: cells 1 and 2 are one step.
- The hidden column is shown by a count, not named: cell 5 counts every column of the file at once; the file's first lines say what each column holds and not when it is written, and cell 7's slots name neither column, so counting is the only way to tell which one matches.

## Where each rule comes from (for the trainer, never in a learner cell)

| Quote in the exercise | From |
|---|---|
| "A return with an empty label is not counted as either outcome, and it is left out of training." | `days/day-02/curated.md` § Examine the labels before you train, Count the two outcomes; deck slide 12 |
| "A *label's origin* is the answer to those three questions: which past decision made the label, who made it, and when." | the same section, In general; deck slides 10 and 11 |
| "The course's data gives some returns a label that stands for an examination's result." | the same section, One return first |
| "A feature that matches the label almost exactly is a warning sign of leakage." | `days/day-02/curated.md` § Ask each feature when it was known, The warning sign; deck slide 16 |
| "When you find such a feature, trace it to where it came from: which table holds it, which step wrote it, and when. The answer to *when* decides." | the same section, The warning sign |
| "was it known before the case was selected? Mark each feature *before* or *after*. Train only on the features marked *before*." | the same section, The check; deck slide 15 |

The exercise is deck slide 19; its debrief is slide 20; the demo is slide 18.

## Simulated data added

- `fields.csv`: for the 750 returns of `model_train` and `model_valid`, two made-up columns, each described on its own first line. `account_change_amt`, the planted column recorded after selection: the tax the examination added (`tax_added_by_exam`) on a `yes` return, 0 on every other return, and 0 on one `yes` return of the training set (`F-41683-2020`), so it matches the label almost exactly, not exactly. `days_to_file`, a column known before selection: days from 1 January to the day the return was received, 20 to 105, taken from a hash of `return_id`, so it bears no relation to the label.
- `refund-held.csv`: the demo label `refund_held` for 16 training returns, with `refund_hold_days`, above zero only on a held refund, and a note line for it.

Each file says on its first line that the course made it up. Cell 2 joins them to the training set in the notebook only; no table changes.

## Where it ran

- Not run in Databricks: a trainer gap. What only a run there confirms: `spark.table` on `course.returns.model_train` for a learner in `training-students`, and `toPandas()` on it; reading a CSV file in the Git folder by the path from `os.getcwd()`.
- Run on a laptop, 2026-11-21, Python 3.11.5, scikit-learn 1.3.0, pandas 2.0.3, on a stand-in for the Spark calls, over the course's week 4 tables made by `week-04/setup/day-01-course-data.py`, and again with no table readable (load.py makes the same rows from that file): start `Not yet: 7 left.`, answer `Done:` on both.

## Modelled on

None found that is proven. The outline's reference: Databricks documentation, Point-in-time feature joins, https://docs.databricks.com/aws/en/machine-learning/feature-store/time-series, a documentation page with no task. Shape: classify each item and give the reason (Monday's exercise 1), after a given count.
