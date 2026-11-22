# Exercise 1: decisions

The answer is `exercise.ipynb` beside this file. It reads the course's shared tables, `course.returns.model_train` and `model_valid`, and fits Tuesday's logistic regression again in cell 2 with Tuesday's lines: `make_pipeline(StandardScaler(), LogisticRegression())` on the 400 examined returns of `model_train` and Tuesday's twelve features, as Wednesday's exercises do. Wednesday chose the logistic regression, so the tree is not fitted here.

## Each answer, and why

| Cell | Answer | Why |
|---|---|---|
| 1, 2 | run as given | 500 and 250 returns; case #123 scores 0.964, place 11 of 250, on Wednesday's list of 25. |
| 3 | `["other_income_amt", "tax_to_agi", "wages_amt"]` | The ranked list: the average size of each feature's push over the 250 returns of `model_valid`, 1.51, 1.14, 1.14, as the reading's chart. `tax_to_agi` and `wages_amt` are level at two places, so the check takes them in either order; the first must be `other_income_amt`. Case #123's own order (other income, deduction, `tax_to_agi`) is refused: the ranked list is the whole model's. |
| 4 | `case_id = 123` | The chart and its table: -4.553, then +3.82, +2.65, +1.24, +1.01, -0.45, -0.22, -0.20, -0.19, +0.11, +0.08, +0.01, -0.01, output 3.28, a score of 0.964; the sum line: 3.30 rounded, 3.28 exact. `wages_amt`, level second on the ranked list, pushes case #123 +0.11. |
| 5 | `axis = "log-odds"`; a guess from 0 to 1 (the answer types 0.6) | The guess: case #123's score with its other income at the training average, 7,584, everything else as filed. The reading does not give it (its 0.904 is a different thing: the model fitted again without the feature). The cell then shows 0.369: a push of +3.82 on the axis moved the score by 0.595. The reading's line on why follows the guess. |
| 6 | the reading's own two sentences | The check asks for: two sentences; 0.964 and the word average; other income named before the deduction; a direction (up, raised, higher); no amount of push or score (no number within 0.015 of the three largest pushes, no "raised ... by 3"); no finding about the taxpayer. |
| 8, optional | `other_id = 471`; two sentences | Case 471, labelled change, 0.179, off the list: `tax_to_agi` +1.50 up, other income 0 against 7,584, -1.38 down. |

Two things for the trainer to know:

- The third decimal of a push depends on the scikit-learn version. With scikit-learn 1.3 the pushes are the reading's (+3.82, +1.01; rounded sum 3.30); with scikit-learn 1.9 they are +3.81 and +1.00, and the rounded sum is 3.28. The exact sum, 3.28, the score, 0.964, the ranking and every count are the same. Step 3 tells the learner the rounded sum may show 3.28 or 3.29, and the check reads the pushes from the learner's own run.
- The SHAP library's `LinearExplainer` subsamples a background of more than 100 rows unless told otherwise; `load.py` passes `shap.maskers.Independent(background, max_samples=400)`, so its numbers equal the exact formula's to 0.001 (it prints a note for the trainer if they ever differ). Called with the bare training array, it gave a starting point of -5.31 and pushes off by up to 0.22.

## What version 2.0 cut, and why

Learners of this cohort could not finish some exercises. 25 minutes became 15.

- The made-up files (`train-returns.csv`, 2,000 made-up returns; `case-123.csv`; `ten-returns.csv`) are gone, and with them the four made-up features, the filled withholding amount and the chart in score units. Case #123 is read from `model_valid`, with Tuesday's twelve features, and its chart is in log-odds, as the reading's.
- The tree is gone: Wednesday's exercise 3 chose the logistic regression, and its check refuses the tree.
- Typed values, 11 to 5: version 1 asked for the average, four pushes and the score to be typed back; cell 4 now prints the sum and the learner reads it. The sentences give order and direction, not amounts. A guess comes before cell 5 shows the score without the largest push.
- Steps, 6 to 5.

## Where each rule comes from (for the trainer, never in a learner cell)

| Quote in the exercise | From |
|---|---|
| "The chart is a *ranked list*: the features in order, ..." | `days/day-04/curated.md` § Name the features the model leans on most, One return first |
| "So read the axis label before you write a number: here, log-odds." | § Explain the score of case #123 from its pushes, Check the units before you write an amount |
| "The exact pushes add up to 3.28; the rounded ones give 3.30, and the difference is rounding." | the same section, One return first |
| "The list describes the whole model; one return's pushes may come in another order." | § Name the features the model leans on most, One return first |
| "A step of the same size in log-odds moves the score a great deal near the middle ..." | § Explain the score of case #123 from its pushes, Check the units before you write an amount |
| "The pushes are in log-odds, so you give their order and direction, and the values that made them, not an amount of score" | the same section, One return first |
| "A push says that the model moved the score because of this value, not that the value is wrong." | the same section, The wrong belief |
| the answer's two sentences | the same section, One return first (the reading's own example) |

The exercise is deck slide 23; its debrief is slide 24; the demo is slide 22. The deck's slide 23 notes (step 4, "Type the starting point, the three largest pushes and the output on the axis") and the talk plan's exercise instructions ask for typed numbers this version does not: the deck should say "read the sum under cell 4's table; type the axis; guess; write the two sentences".

## Simulated data added

None beyond the course's tables.

## Where it ran

- Not run in Databricks: a trainer gap. What only a run there confirms: that the SHAP library is on the class compute, and that `shap.LinearExplainer` and `shap.plots.waterfall(..., show=False)` draw as called here; the scikit-learn version, which sets the third decimal of the pushes.
- Run on a laptop, 2026-11-21, on a stand-in for Spark over the course's week 4 tables made by `week-04/setup/day-01-course-data.py`, and with no table readable, in two Pythons: Python 3.11 with scikit-learn 1.3.0 and no SHAP library (the exact formula), and Python 3.11 with SHAP 0.51.0 and scikit-learn 1.9.1 (the library): start `Not yet: 8 left.`, answer `Done:` in each. `check.py --start` prints `Not yet: 8 left.` and `check.py values.json` prints `Done:`. Wrong answers tried: case #123's order as the top three, `score` as the axis, the guess in quotes, "raised the score by 3.82", a push's number in the sentences, the deduction before other income, a sentence that says the taxpayer underreported: each `Not yet` with its line.

## Pages behind the steps

The day's reading's Sources, by their ids in the Source library: `shap-waterfall-plot` (the chart and how to read it) and `shap-intro-explainable-ai-shapley-values` (the pushes add up from the expected output; the units of the output). This run did not open the pages again.

## Modelled on

The outline's reference, roadmap.sh, MLOps roadmap, its SHAP node, https://roadmap.sh/mlops: a topic node with no learner task, so not proven. Shape taken: read a given chart, check it, write the reading in a fixed form.
