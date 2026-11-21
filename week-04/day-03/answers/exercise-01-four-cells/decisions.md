# Exercise 1: decisions

The answer is `exercise.ipynb` beside this file. Every value it reads is in the made-up files beside the exercise's notebook; no learner table is read.

## Each answer, and why

| Cell | Answer | Why |
|---|---|---|
| 1 | run as given | It prints the ten returns of the day's reading, highest score first, by Spark SQL over a view of `ten-returns.csv`. |
| 2 | #311, #388, #206: `["on", "TP"]`; #458, #270: `["on", "FP"]`; #502: `["off", "FN"]`; #189, #644, #135, #717: `["off", "TN"]` | At 0.5 a score at or above the threshold goes on the list. #502 (0.46) is below it and labelled change, so it is the one false negative. The check accepts `on`/`off` or `on the list`/`off the list`, and `TP` or `true positive`, in any case. |
| 3 | `{"TP": 3, "FP": 2, "FN": 1, "TN": 4}` | The reading's matrix. Sum 1 is 10; sum 2, TP + FN, is the 4 returns labelled change. |
| 4 | four meanings of four words or more | The reading's table, "What it means for an examiner", in the learner's words. The check asks only for four words each: it is the learner's wording, read aloud in the debrief. |
| 5 | run as given | `[[22 28] [16 934]]`: rows are the labels, change then no change; columns on the list, then off it. 1000 returns, 50 labelled change. |
| 6 | `{"TP": 22, "FP": 16, "FN": 28, "TN": 934}` | Row 1 is TP then FN; row 2 is FP then TN. A learner who reads the table as the day's layout types FP 28 and FN 16: sum 2 is then 38, not 50, and the check says the rows are the labels. Accuracy (22 + 934) / 1000 is 0.956, Tuesday's made-up number. |
| 8, optional | `{"TP": 2, "FP": 1, "FN": 2, "TN": 5}`; #270 went from FP to TN, #206 from TP to FN | The reading's table at 0.7. |

Two things for the trainer to know:

- The 1,000 returns are made up so that the counts at 0.5 are the reading's (logistic regression 22, 16, 28, 934; tree 8, 5, 42, 945). They are not the learner's own week 3 rows; the notebook needs no week 3 table. The ten are a separate made-up set, chosen so that every cell appears.
- The debrief's second question: the false negatives are the cell nobody sees. True negatives are also off the list, but they were rightly left alone; a false negative is a return an examination would have changed.

## Where each rule comes from (for the trainer, never in a learner cell)

| Quote in the exercise | From |
|---|---|
| "The first word of each cell's name, *true* or *false*, says whether the list agreed with the label." | `days/day-03/curated.md` § Sort ten returns into four cells, One return first |
| the two sums | the same section, By hand, then by the counting code |
| "affect different people" | the same section, Why this matters to you |
| "with each row corresponding to the true class"; "tools arrange the rows and columns differently" | the same section, By hand, then by the counting code (scikit-learn's guide) |

The exercise is deck slide 19; its debrief is slide 20; the demo is slide 18.

## Simulated data added

`ten-returns.csv` (the reading's ten, as its tables give them) and `validation-scores.csv` (1,000 made-up returns: label, the logistic regression's score to three decimals, the tree's score, one per leaf of a tree three questions deep). Each file says on its first line that the course made it up. Written by a generator outside this repository from a fixed seed.

## Where it ran

- Not run in Databricks: a trainer gap. What only a run there confirms: `spark.createDataFrame` of a pandas table and `createOrReplaceTempView` on the course compute; `.show()`'s layout; `os.getcwd()` in a Git folder notebook giving the notebook's folder (Tuesday's cells rest on it too).
- Run on a laptop, 2026-11-20, Python 3.11.5, scikit-learn 1.3.0, pandas 2.0.3, on a stand-in for Spark (sqlite behind `spark.sql`): start `Not yet: 4 left.`, answer `Done:`. Rerun after the cold read (wording of steps 1, 3 and 6; cell 5 sets its own threshold). `check.py --start` prints `Not yet: 4 left.` and `check.py values.json` prints `Done:` on Python 3.10 and 3.11.

## Pages behind the steps (opened 2026-11-20)

| Step | Page |
|---|---|
| the four cells | Google Machine Learning Crash Course, Thresholds and the confusion matrix, https://developers.google.com/machine-learning/crash-course/classification/thresholding (a Source of the day's reading) |
| the counting code | scikit-learn, `confusion_matrix`, in its Model evaluation guide (a Source of the day's reading) |

## Modelled on

None found that is proven. Closest: the outline's reference, Google Machine Learning Crash Course, Classification: Accuracy, recall, precision, and related metrics, https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall, and its sibling page on thresholds, whose "Check your understanding" items ask which outcome a misclassified example is. No count of learners on the page. Shape taken: classify each item, then count, then check with given code.
