# Exercise 1: decisions

The answer is `exercise.ipynb` beside this file. Every value it reads is in the made-up files beside the exercise's notebook.

## Each answer, and why

| Cell | Answer | Why |
|---|---|---|
| 1 | `logistic-regression` | The candidate the slides follow, and the one for a learner with none. |
| 2 | `tax_to_agi`, `wages_change_pct`, `withholding_missing` | The three longest bars: weights 2.73, 2.10, 0.62, the features first put on one scale; `withholding_amt` is last at 0.57. With the tree: `wages_change_pct`, `tax_to_agi`, `withholding_amt` (importance 42.8, 38.8, 18.4 of 100). The check compares with the model's own list, so either candidate passes. |
| 3 | `case_id = 123` | The chart reads average 0.05; `tax_to_agi` = 0.03, +0.40; `wages_change_pct` = -0.45, +0.29; `withholding_missing` = 1, +0.12; `withholding_amt` = 937 (filled), -0.04; score 0.82. With 311: +0.52, +0.21, +0.15, -0.02, score 0.91, the deck's worked example. |
| 4 | axis `the score, from 0 to 1`; 0.05; 0.40, 0.29, 0.12, -0.04; 0.82 | The axis label says score; the check accepts any text with `score` or `probability` and refuses `log-odds`. Each number within 0.006 of the chart's, sign included. The cell prints the sum, 0.82. |
| 5 | "The model scored case #123 at 0.82, against an average of 0.05 for all returns. The two largest reasons were its low tax for its income, which raised the score by 0.40, and the fall in its wages since last year, which raised it by 0.29." | The reading's form. The check reads the two numbers of each sentence and asks for two words or more for each reason; it refuses a finding about the taxpayer (underreported, fraud, the tax is wrong). With the tree: 0.21, 0.05, 0.15, 0.09. |
| 7, optional | `644`, and two sentences | #644 scores 0.27: average 0.05, `tax_to_agi` +0.12, `wages_change_pct` +0.10, `withholding_amt` +0.04, `withholding_missing` -0.04. Off the list, and both largest pushes still raise it: a return off the list is not one with no reasons. |

## How the pushes are computed

The given code explains the model's probability output, the score, with the training returns as the background: for each of the 16 groups of features, case #123's own values are put in place of every training return's, and the scores are averaged; each push is the Shapley average of the changes one feature makes. The pushes add up exactly from the average score (0.0500) to the score (0.8212): 0.0500 + 0.3999 + 0.2900 + 0.1203 - 0.0390 = 0.8212. This is what the SHAP library's exact explainer computes on `predict_proba` for the change class with an independent masker over the same 2,000 rows (`max_samples` set to all of them). Where the library is installed, cell 3 uses it, draws its waterfall plot, and prints a note to the trainer if its numbers differ from the given code's by more than 0.001; where it is not, the given code's numbers and a plain bar chart stand in, so the exercise works on any compute.

The training returns were made so that this fit (median fill, `StandardScaler`, scikit-learn's default `LogisticRegression`) gives the reading's numbers for case #123 and the deck's for #311, and Wednesday's ten their Wednesday scores. The same numbers came out with the `lbfgs` and `newton-cg` solvers and with a tolerance of 1e-4 or 1e-10. The features are put on one scale inside the candidate, so the ranked chart reads the weights as the reading describes; Tuesday's cell fitted the regression on the raw features, which these made-up data would not let reach the reading's numbers.

The average is taken over the training returns, not over Wednesday's 1,000 validation scores, whose average is 0.15: the reading says "the average score of all returns", and the SHAP library's own examples take the background from the training data.

## Where each rule comes from (for the trainer, never in a learner cell)

| Quote in the exercise | From |
|---|---|
| "The chart is a *ranked list* ..." | `days/day-04/curated.md` § Name the features the model leans on most, One return first |
| "Your job is to read the chart." | § Explain the score of case #123 from its pushes, The tool, before you use it |
| "the average score plus all the pushes equals the return's score." | § Where today goes wrong |
| "So the explanation is a comparison ..." | § Explain the score of case #123 from its pushes, In general |

The exercise is deck slide 22; its debrief is slide 23; the demo is slide 21.

## Simulated data added

`train-returns.csv` (2,000 made-up training returns, 100 labelled change, 617 with no withholding amount), `case-123.csv` (the reading's values, no label) and `ten-returns.csv` (Wednesday's ten, each given features that the model scores at its Wednesday score). Nothing describes how the IRS selects returns.

## Where it ran

- Not run in Databricks: a trainer gap. What only a run there confirms: the view `returns_to_explain` and cell 3's SQL, and how `.show()` prints the missing amount; that the SHAP library is on the class compute, that its exact explainer and waterfall plot run as called here (`shap.maskers.Independent`, `shap.Explainer(..., algorithm="exact")`, `shap.plots.waterfall(..., show=False)`) and give the given code's numbers, and that the axis label set under the waterfall plot shows.
- Run on a laptop, 2026-11-20, Python 3.11.5, scikit-learn 1.3.0, pandas 2.0.3, with no SHAP library: start `Not yet: 7 left.`, answer `Done:`, with both candidates, on a stand-in for Spark (sqlite). After the cold read: the axis step names its unit, the tree's `lowered`, the filled 937 explained, the ranked bars drawn finer. `check.py --start` prints `Not yet: 7 left.` and `check.py values.json` prints `Done:` on Python 3.10 and 3.11. Wrong answers tried: case #123's order as the top three, a push with the wrong sign, `log-odds` as the axis, the average and score swapped, the two pushes swapped, a sentence that says the taxpayer underreported: each `Not yet` with its line.

## Pages behind the steps

The day's reading's Sources, by their ids in the Source library: `shap-waterfall-plot` (the chart and how to read it) and `shap-intro-explainable-ai-shapley-values` (the pushes add up from the expected output; the probability output of a logistic regression). This run did not open the pages again.

## Modelled on

The outline's reference, roadmap.sh, MLOps roadmap, its SHAP node, https://roadmap.sh/mlops: a topic node with no learner task, so not proven. Shape taken: read a given chart, check it, write the reading in a fixed form.
