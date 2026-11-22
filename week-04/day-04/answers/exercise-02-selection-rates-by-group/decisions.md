# Exercise 2: decisions

The answer is `exercise.ipynb` beside this file. It reads the course's shared tables, `course.returns.model_train`, `model_valid` and `model_features`, fits Tuesday's logistic regression again in cell 2 with Tuesday's lines, and puts on the list the 25 returns of `model_valid` at or above Wednesday's threshold, the 25th score, 0.844. The column `made_up_channel` is a code the course invented; it describes no real person or group, and it is not one of the twelve features.

## Each answer, and why

| Cell | Answer | Why |
|---|---|---|
| 1, 2 | run as given | 500, 250 and 1,000 returns; `Threshold 0.844: 25 of 250 returns on the list`. |
| 3 | `wider` or `narrower` (the answer types `narrower`) | The guess: the gap on a list of 50 against the list of 25. The reading gives the gap at 0.844 and at 0.5, not on a list of 50. Cell 5 shows it once the guess is typed: 27 of 162 (0.167) against 23 of 88 (0.261), a gap of 0.095, wider. A guess cannot be wrong. |
| 4 | `group_column = "made_up_channel"` | Channel A: 162 returns, 15 on the list; channel B: 88, 10. |
| 5 | `rate_a = 15 / 162`, `rate_b = 10 / 88`; one sentence | 0.093 and 0.114, a gap of 0.021 toward channel B. The cell then prints the list of 50 and 0.5 (22 of 162, 0.136; 20 of 88, 0.227; a gap of 0.091), and the past labels over the examined returns of all four tax years: 83 of 512 (0.162) and 77 of 288 (0.267). The sentence must hold the four counts, the threshold 0.844, the gap (0.021, or 2.1 in 100) and the word reason; it may not call the model fair or unfair, and may not say the model selected. |
| 7, proxy step | `could stand in`, with 0 and 16,140 | The median of `other_income_amt` over all 1,000 returns: 0 in channel A's 648, 16,140 in channel B's 352. Checked only when the cell has run. |
| 8, optional, proxy step | `wages_amt could not stand in`, with 52,695 and 53,140 | The third feature of exercise 1's ranked list, close in the two channels: the contrast to cell 7. |

Two things for the trainer to know:

- Counts first: 15 against 10 looks like a lean toward channel A; the rates point the other way. The debrief's first question.
- The labels' line answers the debrief's second question: the past decisions already differ by channel, so a gap can come from the labels as well as from the model.

## The proxy step: how to cut it

Step 6 (cell 7), the optional part (cell 8) and the debrief's third question go with the reading's section "Spot a feature that stands in for a group". To cut them: delete step 6 and its cell 7, the optional part and its cell 8, and question 3 of "Afterwards, together", from the README and both notebooks. Nothing renumbers: the check is cell 6, before them. `check.py` checks the proxy sentence only when `proxy_sentence` exists in the notebook, so it needs no change; `values-without-proxy-step.json` is the answer's run with cells 7 and 8 removed, and `check.py` prints `Done:` on it. The day's README names the step in its last line.

## What version 2.0 cut, and why

Learners of this cohort could not finish some exercises. 25 minutes became 15.

- The made-up files (`validation-returns.csv`, 1,000 made-up returns with `synthetic_group` and a built feature `amended_last_year`; `ten-returns.csv`) are gone. The groups are the course's column `made_up_channel` on the shared tables, and the feature that tracks it is `other_income_amt`, one of Tuesday's twelve, as the reading says.
- The tree is gone: Wednesday chose the logistic regression.
- The proxy step is the last step, marked, and removable as one piece; the check does not need it.
- Typed values, 6 to 5 (with the proxy step, 6): guess, column, two rates, one sentence. The comparison at two other thresholds and the labels by channel are printed, not typed.

## Where each rule comes from (for the trainer, never in a learner cell)

| Quote in the exercise | From |
|---|---|
| "It describes no real person or group, and it stands for no protected class such as race, sex or age; the arithmetic is the same for any grouping." | `days/day-04/curated.md` § Check the model for bias before it is used, One return first |
| "So a gap belongs to a threshold, as precision and recall did on Wednesday: write the threshold beside it." | § Compare selection rates across two groups, One return first |
| "Split the held-out returns by group." | § Check the model for bias before it is used, In general |
| "The *selection rate* of a group is the share of its returns that the model puts on the list." | § Compare selection rates across two groups, One return first |
| "A gap is a result that needs a reason." | the same section, The wrong belief |
| "So a gap in the list can come from the labels as well as from the model." | the same section, Where a gap can come from: the labels |
| "A *proxy* is a feature that names no group and still tracks one. You see it when ..." | § Spot a feature that stands in for a group, In general |
| "So your sentence says that the feature *could* stand in for the group, ..." | the same section, last paragraph |

The exercise is deck slide 40; its debrief is slide 41; the demo is slide 39. The deck's slide 40 notes list "the proxy segment" steps as this exercise's last step, as here; its notes' success and stuck lines carry a literal `" + PX + "` left from a template, which the deck's fixer should remove.

## Simulated data added

None beyond the course's tables.

## Where it ran

- Not run in Databricks: a trainer gap. What only a run there confirms: `course.returns.model_features` readable by every learner (the activities' "made_up_channel readable in every learner's notebook").
- Run on a laptop, 2026-11-21, Python 3.11.5, scikit-learn 1.3.0, pandas 2.0.3, on a stand-in for Spark over the course's week 4 tables made by `week-04/setup/day-01-course-data.py`, and with no table readable: start `Not yet: 8 left.`, answer `Done:` (with the proxy sentence, when cell 6 is run again after cell 7), and the answer with cells 7 and 8 removed `Done:`. The same in Python 3.11.5 with scikit-learn 1.9.1: every count is the same. `check.py --start` prints `Not yet: 8 left.`; `check.py values.json` and `check.py values-without-proxy-step.json` print `Done:`. Wrong answers tried: `could not stand in` for 0 and 16,140, a result that calls the model unfair, a result without the threshold, "selected", 15 / 88 as a rate, an empty proxy sentence: each `Not yet` with its line.

## Modelled on

None found that is proven (the day plan: no lab found in the Source library's captured pages; the web was not searched). Closest: the Program's common task, a bias check before use. Shape taken: split, divide, state the result; then compare one feature.
