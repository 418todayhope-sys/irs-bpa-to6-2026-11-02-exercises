# Exercise 2: decisions

The answer is `exercise.ipynb` beside this file.

## Each answer, and why

| Cell | Answer | Why |
|---|---|---|
| 1 | `column = "exam_changed_tax"` | The baseline predicts the most common label, so the label is the column to count. It prints `no` 320, `NULL` 100, `yes` 80. |
| 2 | `guess = 0.64` (any share from 0 to 1) | The guess is the learner's own. 0.64 is 320 of 500, the guess of a learner who keeps the empty rows: cell 3 then shows it beside 0.8. |
| 3 | `right = 320`, `labelled = 400` | The baseline always predicts `no`, so it is right on the 320 `no` returns. The 100 empty rows have no label, so the count leaves them out: 320 + 80 = 400. Accuracy 320 / 400 = 0.8. |

What else the check accepts: a guess typed as `0.8`, `"0.8"`, `"80%"` or `80`. A `labelled` of 500 prints the line on the empty rows.

## Where each rule comes from (for the trainer, never in a learner cell)

Lines are those of `days/day-01/curated.md` in the Program repository, version 2.0 (2026-11-21).

| Quote or rule in the exercise | From |
|---|---|
| "the share of predictions that are right." | line 24, Today's words, Accuracy (also line 131, § Compute the accuracy every model must beat) |
| "the simplest prediction; its accuracy is the one every model must beat." | line 23, Today's words, Baseline |
| a guess written before the cell divides (Step 2, in plain words) | line 139, § Compute the accuracy every model must beat, How you compute it |
| "A prediction is right or wrong only against a recorded decision." (Step 3) | line 125, same section, Why only the 200? |
| `NULL`: never examined, so no label (Step 1, in plain words) | line 98, § Name the label a question can learn from, A label exists only where someone decided |
| Debrief 1, a rarer yes | line 135, § Compute the accuracy, The wrong belief (Brownlee's 99 percent) |
| Debrief 2, what a model must show | line 137, same section: a model level with the baseline "has not yet shown that it picks out anything the baseline does not" |
| Debrief 3, the returns nobody examined | line 100, § Name the label, What a label is, and what it is not; line 307, § The trade-off |

The debrief questions are `days/day-01/activities.md`'s, word for word, and deck slide 33's. The worked example is deck slide 31; the exercise is slide 32. The reading gives `model_valid`'s counts (160, 40, 50) and the whole table's, never `model_train`'s.

## Where it ran

- Not run in Databricks: a trainer gap (`week-04/setup/day-01-setup.md` § Checked in Databricks, and not yet checked).
- Run on a laptop, 2026-11-21 UTC, by `answers/stand_in.py` (as `commands.sh` runs it): the start prints `Not yet: 4 left.` The answer prints `Done:`. `labelled = 500` with `guess = "80%"` prints `Not yet: 1 left.`, the line on the 100 empty rows.

## Pages behind the steps

`spark.sql(...).show()` with `GROUP BY` and `COUNT(*)`, as in week 3. We cite no other page.

## Modelled on

Google Machine Learning Crash Course, "Classification: Accuracy, recall, precision, and related metrics", https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall. Its accuracy section shows a model that predicts the common outcome every time: high accuracy, and of no use. Its "Check your understanding" tasks show each answer with its reason. The course page says "Since 2018, millions of people worldwide have relied on Machine Learning Crash Course" (https://developers.google.com/machine-learning/crash-course). Shape taken: count, guess, then compute by hand. `days/day-01/activities.md` names the scikit-learn MOOC's exercise M1.03 as the closest proven shape (guess, then run and compare).
