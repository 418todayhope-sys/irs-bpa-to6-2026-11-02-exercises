# Exercise 3: decisions

The answer is `exercise.ipynb` beside this file. Every value it reads is in the made-up files beside the exercise's notebook.

## Each answer, and why

| Cell | Answer | Why |
|---|---|---|
| 2 | "In this scenario a false negative costs more than a false positive, because a missed return leaves tax an examination would have changed as filed, and nobody ever sees it." | Either direction passes; the check asks for the form and a reason of five words or more after `because`. `missed return` and `wrongly listed return` are accepted for the two errors. |
| 3 | `k = 3` | Three cases for every ten returns scored, and there are ten. The SQL prints #311, #388, #458. |
| 4 | threshold 0.74; 2 labelled change on the list; 2 missed (#206, #502); the trade names both errors | The k-th score. Against 0.5 (2 false positives, 1 false negative), one false positive fewer (#270 off) and one false negative more (#206 off). The check asks that the trade names both kinds. |
| 5 | run as given | #311, #388, #123 (0.82, `no label yet`). |
| 6 | 0.82; `#458` | The new k-th score; #458 is on cell 3's list and not on cell 5's. `458` without `#` passes. Case #123 is in no cell. |
| 7 | `logistic-regression`, with a sentence that gives the counts | A sentence that says a missed return costs more must choose the logistic regression (22 of 50 picked out, against 8): the check says so. A sentence that says a false positive costs more may choose either: the tree is a little ahead on precision (8 of 13 against 22 of 38) but lists far fewer returns. Eight words or more, with a number. |
| 9, optional | `2 / 3`; a tie rule in a sentence | Precision at k on the ten. On the 1,000 at k of 300 the list never holds more than 300: the logistic regression's threshold 0.169, a score 4 returns share and all 4 fit, 300 on the list, 42 labelled change; the tree's 0.09, a score 150 returns share with room for 127 of them, so the learner breaks the tie and writes down how (the cell takes them by return ID): 300 on the list, 44 labelled change. Any rule passes if it says which tied returns go on and why. |

Two things for the trainer to know:

- The ten returns and case #123 were scored by the logistic regression, so 0.74 and 0.82 are its thresholds. A learner who chooses the tree has its threshold only from the optional cell (its own scores on the 1,000): say so, and Thursday reads it again.
- On the 1,000, three in ten is 300 returns, and at most 50 of them can be labelled change: the honest report the reading describes (few of the list are worth the hours). The tree's scores have one value per leaf, so ties at the k-th score are large: the list still holds 300, and the learner breaks the tie and writes down how.

## Where each rule comes from (for the trainer, never in a learner cell)

| Quote in the exercise | From |
|---|---|
| "No model or formula makes that judgment ..." | `days/day-03/curated.md` § Say which error costs more, Why this matters to you |
| "A *threshold from capacity* is the score of the k-th return ..." | § Set the threshold from the examiners' capacity, In general |
| "every return at or above"; "scored in the same batch as the ten, and the examiners still take three cases" | the same section, One return first |
| "If a missed return costs more, recall matters more to you" | § Say which error costs more, Which costs more |
| the list never holds more than k; a tie at the k-th score is broken and written down | § Set the threshold from the examiners' capacity, its two practical points; the Checkpoint `thin-air-cp-9897150a` |

The exercise is deck slide 46; its debrief is slide 47; the demo is slide 45.

## Simulated data added

`ten-returns.csv` and `validation-scores.csv` (exercise 1's), `case-123.csv` (Monday's made-up case, score 0.82, no label) and `candidate-counts.csv` (the reading's four counts for the baseline and both candidates). The group and its two errors are made up; nothing describes how the IRS selects returns.

## Where it ran

- Not run in Databricks: a trainer gap. What only a run there confirms: the two views and their SQL (`ORDER BY score DESC LIMIT k`), `.show()`'s layout, a pandas table with the text `no label yet` passed to `spark.createDataFrame`.
- Run on a laptop, 2026-11-20, Python 3.11.5, pandas 2.0.3, on a stand-in for Spark (sqlite): start `Not yet: 10 left.`, answer `Done:`. Rerun after the cold read (either direction said to be possible, the trade's words named, the eight-word rule stated, cell 9's message with no choice). `check.py --start` prints `Not yet: 10 left.` and `check.py values.json` prints `Done:` on Python 3.10 and 3.11. Wrong answers tried: the tree with a missed-return sentence, a trade naming one error, a swapped precision in exercise 2: each `Not yet` with its line.

## Pages behind the steps (opened 2026-11-20)

| Step | Page |
|---|---|
| the threshold | scikit-learn example, Post-tuning the decision threshold for cost-sensitive learning, https://scikit-learn.org/stable/auto_examples/model_selection/plot_cost_sensitive_learning.html (the outline's reference; it prices each error, which this exercise does not) |

## Modelled on

None found that is proven. Closest: the outline's reference above, an example with no learner task. Shape taken: state the cost, apply a rule given in the reading, choose and give the reason.
