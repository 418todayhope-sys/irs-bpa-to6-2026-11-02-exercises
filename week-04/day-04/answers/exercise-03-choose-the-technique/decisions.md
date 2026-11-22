# Exercise 3: decisions

The answer is `exercise.ipynb` beside this file. The exercise reads no file and calls no model: it is thinking work in a notebook's text, as Monday's exercise 1.

## Each answer, and why

| Cell | Answer | Why |
|---|---|---|
| 2 | Payment plans: labelled data yes, many past plans labelled kept or missed; cost: training once, then each new plan almost free to run; risk: a wrongly flagged plan, reviewed by a person; **a trained model**, decided by labelled data | A decision with many labelled past cases, made many times: where a trained model fits. Cost points the same way. The check asks for each row in three words or more, the choice that labelled data points to, and a reason of six words or more that names labelled data, cost or risk. |
| 3 | Summaries: no labelled data, no store of approved summaries; cost: nothing to train, a small cost per call; risk: a summary that states a step not on the page; **prompting**, decided by labelled data; who checks: the page's owner, against the procedure page | Writing text with no labelled cases: where prompting fits. A summary that reads well and is invented is week 1's failure, so a person checks each against its page. The check asks who checks, in six words or more, with the word against. |
| 5, optional | retrieval for the summaries | The language model reads the procedure page and the pages it names before it writes. |

The check does not judge a reason's quality: "it handles text" names no row and is refused, but any sentence that names a row passes. The debrief judges the reasons. A learner who chooses prompting for the payment plans, or a trained model for the summaries, gets the facilitator's hint as the check's line.

## Where each rule comes from (for the trainer, never in a learner cell)

| Quote in the exercise | From |
|---|---|
| "Both tools serve the same return, each for its own task. Your answer in the exercise takes the same form: ..." and the filled table of case #123's two tasks (its last row, the deciding row, is the exercise's own) | `days/day-04/curated.md` § Choose the technique by three questions, One return first |
| "A language model fails in its own way: as week 1 showed, ..." | the same section, In general, 3. Risk |

The exercise is deck slide 54; its debrief is slide 55; the demo is slide 53.

## What version 2.0 changed, and why

- 25 minutes became 15 (13 to do it, 2 for setup: no table is read). The two tasks, the three rows and the check are unchanged.
- Cell 1's filled table now copies the reading's table row for row (version 1 had "past decided returns" and a shorter risk row), and step 1 quotes the reading as it now stands.
- The cells line and "(we do)" were added as in days 1 to 3.

## Simulated data added

None: the two teams and their tasks are made up; nothing describes how the IRS works.

## Where it ran

Run on a laptop, 2026-11-21, Python 3.11.5, version 2.0: start `Not yet: 11 left.`, answer `Done:`. Version 1, 2026-11-20: start `Not yet: 11 left.`, answer `Done:`. `check.py --start` prints `Not yet: 11 left.` and `check.py values.json` prints `Done:` on Python 3.10 and 3.11. Wrong answers tried: prompting for the payment plans, a reason that names no row, no one who checks: each `Not yet` with its line. After the cold read: cell 1's filled table gives each deciding row with its reason (it gave two words, below the six the check asks), step 4 names the words a row uses, and the scenario says who acts on a flagged plan and who owns a page. In Databricks the notebook only prints text; its run there is the class's.

## Modelled on

The outline's reference, Google for Developers, Rules of Machine Learning, Rule #1, https://developers.google.com/machine-learning/guides/rules-of-ml: a guide with no learner task, so not proven. Shape taken: answer fixed questions in a table, choose, give the deciding row.
