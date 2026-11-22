# Exercise 2: decisions

The answer is `exercise.ipynb` beside this file. Every value it reads is in the made-up files beside the exercise's notebook.

## Each answer, and why

| Cell | Answer | Why |
|---|---|---|
| 1 | `logistic-regression` | The threshold from capacity, 30 of the 1,000 (the reading's whole-set example): the 30th score, 0.599, and 30 returns on the list. With the tree, the 30th score is 0.33, which 40 returns share and the list has room for 17 of them: the tie is broken by return ID, so 30 are on the list. |
| 2 | `A` | Either passes; the check asks for A or B before cell 3. |
| 3 | `synthetic_group` | Group A: 600 returns, 24 on the list; group B: 400, 6 (the reading's numbers). With the tree: 18 of 600 and 12 of 400. |
| 4 | `24 / 600`, `6 / 400`; "The model puts 24 of group A's 600 returns on the list, 4 in 100, and 6 of group B's 400, 1.5 in 100: a gap of 2.5 in 100, a result that needs a reason." | The rates as divisions, within 0.0005. The check finds the four counts and the gap (2.5, or 0.025) among the sentence's numbers and the word reason, and refuses fair, unfair or biased unless the sentence negates it. With the tree: 3 and 3 in 100, a gap of 0 in 100 (18 of 600, 12 of 400), the tie broken by return ID. |
| 5 | "amended_last_year could stand in for the group, because 35 in 100 of group A's returns hold 1, against 5 in 100 of group B's." | 35 against 5 in 100: far apart, so could. The check asks for could (or could not), the right one of the two (10 points apart or more is could), and both values. It refuses a sentence with no could: the reading separates a feature that tracks a group from one a model uses in its place. |
| 7, optional | `withholding_missing`: 30 against 4 in 100, could stand in | The reading's own example of the flag; the model reads it, third on exercise 1's ranked list. |

Two things for the trainer to know:

- The validation returns are Wednesday's 1,000 with Wednesday's scores; the group and the two 0-or-1 columns were added for today, so the groups are the same whichever candidate a learner chose, and only the list changes.
- `amended_last_year` is the trainer's built feature (the day plan's trainer gap: a 0-or-1 column). It is not one of Tuesday's four features: the step frames it as a feature a later model could read, compared before anyone adds it.

## Where each rule comes from (for the trainer, never in a learner cell)

| Quote in the exercise | From |
|---|---|
| "with Wednesday's whole-set example of 30 per 1,000 returns, the 30 returns with the highest scores" | `days/day-04/curated.md` § Check the model for bias before it is used, One return first |
| "Split the held-out returns by group." | the same section, In general, step 1 |
| "The *selection rate* of a group is the share of its returns that the model puts on the list." | § Compare selection rates across two groups, One return first |
| "A *proxy* is a feature that names no group and still tracks one. You see it when ..." | § Spot a feature that stands in for a group, In general |

The exercise is deck slide 36; its debrief is slide 37; the demo is slide 35, whose ten returns and groups are `ten-returns.csv`: the trainer sets `table` to `ten_returns` in cell 3 (A 2 of 6, B 1 of 4 at the threshold 0.74).

## Simulated data added

`validation-returns.csv`: Wednesday's 1,000 validation returns and scores, with `synthetic_group` (600 A, 400 B, 24 and 6 of the logistic regression's top 30), `amended_last_year` (210 of A, 20 of B) and `withholding_missing` (180 of A, 16 of B, the reading's 30 and 4 in 100). `ten-returns.csv`: Wednesday's ten with the demo's groups. Groups A and B are made up, describe no real person and stand for no protected class; nothing describes how the IRS selects returns.

## Where it ran

- Not run in Databricks: a trainer gap. What only a run there confirms: the two views and their SQL (`GROUP BY`, `ROUND(AVG(...) * 100, 1)`), and `.show()`'s layout.
- Run on a laptop, 2026-11-20, Python 3.11.5, pandas 2.0.3, on a stand-in for Spark (sqlite): start `Not yet: 9 left.`, answer `Done:`, with both candidates, and the demo's ten. `check.py --start` prints `Not yet: 9 left.` and `check.py values.json` prints `Done:` on Python 3.10 and 3.11. Wrong answers tried: the counts typed as rates, a sentence that compares counts and calls the model unfair, a proxy sentence with no could, could not for a feature far apart, the column named `group`: each `Not yet` with its line. After the cold read: step 4 names what the result sentence must hold, step 5 says the model does not read the feature today and how far apart is far, cell 3 says to keep `validation_returns`.

## Modelled on

None found that is proven (the day plan: no lab found in the Source library's captured pages; the web was not searched). Closest: the Program's common task, a bias check before use. Shape taken: split, divide, state the result; then compare one feature.
