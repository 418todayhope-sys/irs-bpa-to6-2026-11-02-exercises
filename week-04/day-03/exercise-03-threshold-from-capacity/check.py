"""The done check for exercise 3. In Databricks, cell 8 of the notebook runs it: it sorts the ten
returns, and the ten with case #123, at the group's capacity again, and compares that with what you
typed. Without Databricks, from the course folder:

    python3 week-04/day-03/exercise-03-threshold-from-capacity/check.py --start
    python3 week-04/day-03/exercise-03-threshold-from-capacity/check.py <facts.json>

reads the same facts from a JSON file (or the notebook's starting state). No model is called.
"""
import json
import re
import sys
from pathlib import Path

NAMES = ("logistic-regression", "decision-tree")
CAPACITY = 3  # cases for every ten returns scored
TEN = {"#311": (0.91, "change"), "#388": (0.84, "change"), "#458": (0.74, "no change"), "#206": (0.63, "change"),
       "#270": (0.55, "no change"), "#502": (0.46, "change"), "#189": (0.38, "no change"),
       "#644": (0.27, "no change"), "#135": (0.15, "no change"), "#717": (0.08, "no change")}
CASE_123 = ("#123", 0.82)
START = {"ten": {r: list(v) for r, v in TEN.items()}, "case_123": list(CASE_123), "cost_sentence": "", "k": None,
         "threshold": None, "listed_change": None, "missed_change": None, "trade": "",
         "threshold_with_123": None, "pushed_off": "", "chosen": "", "why": ""}
ERROR = r"(false negative|false positive|missed return|wrongly listed return)"
SENTENCE = re.compile(rf"\ba {ERROR} costs more than a {ERROR}\b.*?\bbecause\b(.*)", re.I | re.S)
MISSED = ("false negative", "missed return")


def plain(text) -> str:
    """A typed answer as the check reads it: lower case, with no spaces or full stop around it."""
    return str(text).strip().strip(".!").strip().lower()


def top(scores: dict, k: int) -> list[str]:
    """The returns at or above the k-th score, highest first: the list at the threshold from capacity."""
    ranked = sorted(scores, key=lambda r: -scores[r])
    kth = scores[ranked[k - 1]]
    return [r for r in ranked if scores[r] >= kth]


def near(a, b) -> bool:
    """Whether a typed number is the expected one, to two decimals."""
    try:
        return abs(float(a) - float(b)) <= 0.005
    except (TypeError, ValueError):
        return False


def left(f: dict) -> list[str]:
    """Return what is left: one line for each thing that is not yet right.

    Args:
        f: the facts: ten (each return's score and label), case_123 (its id and score), and what the
            learner typed in cells 2 to 7.

    Returns:
        One line per problem; an empty list when the exercise is done.
    """
    out = []
    m = SENTENCE.search(str(f["cost_sentence"]))
    more = None
    if not m:
        out.append("Cell 2: write the sentence in the form: In this scenario a false negative costs more than a "
                   "false positive, because ... (or the other way round). Step 2.")
    elif (m.group(1).lower() in MISSED) == (m.group(2).lower() in MISSED):
        out.append("Cell 2: the sentence must set the two kinds of error against each other. Step 2.")
    elif len(m.group(3).split()) < 5:
        out.append("Cell 2: after because, give a reason of five words or more: who bears the cost, and what it is. Step 2.")
    else:
        more = "false negative" if m.group(1).lower() in MISSED else "false positive"
    scores = {r: v[0] for r, v in f["ten"].items()}
    if f["k"] != CAPACITY:
        out.append("Cell 3: k is the number of cases the group can take from ten returns: 3 for every 10. "
                   "Run cells 3 to 6 again. Step 3.")
    listed = top(scores, CAPACITY)
    kth = scores[listed[CAPACITY - 1]]
    change = [r for r, v in f["ten"].items() if v[1] == "change"]
    if not near(f["threshold"], kth):
        out.append("Cell 4: the threshold is the score of the k-th return of cell 3, the last one it printed. Step 4.")
    if f["listed_change"] != sum(r in change for r in listed[:CAPACITY]):
        out.append("Cell 4: count again how many of the k returns of cell 3 are labelled change. Step 4.")
    if f["missed_change"] != sum(r not in listed[:CAPACITY] for r in change):
        out.append("Cell 4: count again the returns labelled change that are not among the k. There are 4 in the "
                   "ten. Step 4.")
    trade = plain(f["trade"])
    if not ("false positive" in trade and ("false negative" in trade or "missed" in trade)):
        out.append("Cell 4: name both kinds of error in the trade: which goes down and which goes up, "
                   "compared with 0.5. Step 4.")
    batch = dict(scores, **{CASE_123[0]: CASE_123[1]})
    new = top(batch, CAPACITY)
    if not near(f["threshold_with_123"], batch[new[CAPACITY - 1]]):
        out.append("Cell 6: the new threshold is the k-th score of cell 5. Step 6.")
    off = [r for r in listed[:CAPACITY] if r not in new[:CAPACITY]]
    typed = plain(f["pushed_off"]).lstrip("#")
    if not off or typed != off[0].lstrip("#"):
        out.append("Cell 6: which return of cell 3's list is not on cell 5's? Step 6.")
    chosen = plain(f["chosen"]).replace(" ", "-")
    if chosen not in NAMES:
        out.append("Cell 7: type logistic-regression or decision-tree as your choice. Step 7.")
    elif more == "false negative" and chosen != "logistic-regression":
        out.append("Cell 7: your sentence says a missed return costs more. Which candidate picks out more of the "
                   "returns labelled change? Step 7.")
    why = str(f["why"])
    if len(why.split()) < 8 or not re.search(r"\d", why):
        out.append("Cell 7: write why in one sentence of eight words or more, with the counts. Step 7.")
    return out


def report(f: dict) -> str:
    """Return what the check cell prints: Done, or Not yet with each thing left."""
    out = left(f)
    if not out:
        return (f"Done: at k of {f['k']} the threshold is {float(f['threshold']):.2f}, with {f['listed_change']} of "
                f"{f['k']} on the list labelled change and {f['missed_change']} missed; with case #123 it is "
                f"{float(f['threshold_with_123']):.2f} and {f['pushed_off']} is off the list. You chose {f['chosen']}.")
    return "\n".join([f"Not yet: {len(out)} left."] + ["- " + item for item in out])


def facts_of(g: dict) -> dict:
    """The facts, from the notebook's values."""
    ten, c123 = g["ten"], g["case_123"]
    return {"ten": {r: [float(s), l] for r, s, l in zip(ten["return_id"], ten["score"], ten["label"])},
            "case_123": [c123["return_id"].iloc[0], float(c123["score"].iloc[0])],
            **{name: g.get(name) for name in ("cost_sentence", "k", "threshold", "listed_change", "missed_change",
                                               "trade", "threshold_with_123", "pushed_off", "chosen", "why")}}


def in_notebook(g: dict) -> str:
    """Return what the check cell prints, from the notebook's values (cell 8 passes them)."""
    return report(facts_of(g))


if __name__ == "__main__":
    given = START if sys.argv[1:] == ["--start"] else json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print(report(given))
