"""The done check for exercise 2. In Databricks, cell 6 of the notebook runs it: it counts each group's
returns and its returns on the list again, and each group's share of returns that hold 1 in the
feature of cell 5, and compares them with what you typed. Without Databricks, from the course folder:

    python3 week-04/day-04/exercise-02-selection-rates-by-group/check.py --start
    python3 week-04/day-04/exercise-02-selection-rates-by-group/check.py <facts.json>

reads the same facts from a JSON file (or the notebook's starting state). No model is called.
"""
import json
import re
import sys
from pathlib import Path

FEATURE = "amended_last_year"
# The logistic regression's list on the 1,000 validation returns, and the feature's share holding 1.
START = {"groups": {"A": {"returns": 600, "on_list": 24}, "B": {"returns": 400, "on_list": 6}},
         "feature_in_100": {"A": 35.0, "B": 5.0},
         "guess": "", "group_column": "", "rate_a": None, "rate_b": None, "result": "", "proxy_sentence": ""}
VERDICT = re.compile(r"\b(unfair|fair|biased|discriminat\w*)\b", re.I)
NOT = re.compile(r"\b(not|cannot|can't|no|without)\b", re.I)


def numbers(text: str) -> list[float]:
    """Return every number in a sentence, with its commas removed."""
    return [float(n) for n in re.findall(r"\d+(?:\.\d+)?", str(text).replace(",", ""))]


def has(found: list[float], *wanted: float) -> bool:
    """Whether the sentence holds one of the wanted numbers, to two decimals."""
    return any(abs(n - w) <= 0.006 for n in found for w in wanted)


def left(f: dict) -> list[str]:
    """Return what is left: one line for each thing that is not yet right.

    Args:
        f: the facts: groups (each group's returns and returns on the list), feature_in_100 (each
            group's share holding 1 in the feature, in 100), and what the learner typed in cells 2 to 5.

    Returns:
        One line per problem; an empty list when the exercise is done.
    """
    out = []
    a, b = f["groups"]["A"], f["groups"]["B"]
    rate_a, rate_b = a["on_list"] / a["returns"], b["on_list"] / b["returns"]
    if str(f["guess"]).strip().upper() not in ("A", "B"):
        out.append("Cell 2: type A or B as your guess, before you run cell 3. Step 2.")
    if str(f["group_column"]).strip() != "synthetic_group":
        out.append("Cell 3: type the name of the column that holds each return's group. Cell 1 printed the "
                   "columns. Step 3.")
    for name, typed, rate, group in (("rate_a", f["rate_a"], rate_a, "A"), ("rate_b", f["rate_b"], rate_b, "B")):
        try:
            ok = abs(float(typed) - rate) < 0.0005
        except (TypeError, ValueError):
            ok = False
        if not ok:
            out.append(f"Cell 4: {name} is group {group}'s returns on the list divided by its returns in the group, "
                       "such as 3 / 50. Step 4.")
    result = str(f["result"])
    found = numbers(result)
    gap = abs(rate_a - rate_b)
    if not all(has(found, n) for n in (a["on_list"], a["returns"], b["on_list"], b["returns"])):
        out.append("Cell 4: give the counts beside the rates: each group's returns on the list and its returns in "
                   "the group. Step 4.")
    if not has(found, round(gap * 100, 2), round(gap * 100, 1), round(gap, 3)):
        out.append("Cell 4: state the gap, the difference between the two rates, such as 1.5 in 100. Step 4.")
    if not re.search(r"\breason", result, re.I):
        out.append("Cell 4: say that the gap needs a reason. Step 4.")
    if VERDICT.search(result) and not NOT.search(result):
        out.append("Cell 4: a gap is a result that needs a reason. Do not call the model fair or unfair from it. Step 4.")
    proxy = str(f["proxy_sentence"])
    share_a, share_b = f["feature_in_100"]["A"], f["feature_in_100"]["B"]
    could = abs(share_a - share_b) >= 10
    said_not = re.search(r"could not stand in|couldn't stand in|cannot stand in", proxy, re.I)
    said_could = re.search(r"could stand in", proxy, re.I)
    if not (said_could or said_not):
        out.append("Cell 5: write could stand in, or could not stand in. The comparison shows that a feature "
                   "tracks a group, not that the model uses it in the group's place. Step 5.")
    elif bool(said_could) != could:
        out.append("Cell 5: compare the two values of cell 5 again. Are they close, or far apart? Step 5.")
    if not (has(numbers(proxy), share_a, share_a / 100) and has(numbers(proxy), share_b, share_b / 100)):
        out.append("Cell 5: after because, give the feature's typical value in each group, from cell 5. Step 5.")
    return out


def report(f: dict) -> str:
    """Return what the check cell prints: Done, or Not yet with each thing left."""
    out = left(f)
    a, b = f["groups"]["A"], f["groups"]["B"]
    if not out:
        return (f"Done: group A has {a['on_list']} of {a['returns']} on the list and group B {b['on_list']} of "
                f"{b['returns']}; you stated the gap with its counts, and whether {FEATURE} could stand in for the group.")
    return "\n".join([f"Not yet: {len(out)} left."] + ["- " + item for item in out])


def facts_of(g: dict) -> dict:
    """The facts, from the notebook's values: the validation returns as cell 1 put them on the list."""
    v = g["validation"]
    groups = {name: {"returns": int(len(part)), "on_list": int(part["on_list"].sum())}
              for name, part in v.groupby("synthetic_group")}
    shares = {name: round(float(part[FEATURE].mean()) * 100, 1) for name, part in v.groupby("synthetic_group")}
    return {"groups": groups, "feature_in_100": shares,
            **{name: g.get(name) for name in ("guess", "group_column", "rate_a", "rate_b", "result", "proxy_sentence")}}


def in_notebook(g: dict) -> str:
    """Return what the check cell prints, from the notebook's values (cell 6 passes them)."""
    return report(facts_of(g))


if __name__ == "__main__":
    given = START if sys.argv[1:] == ["--start"] else json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print(report(given))
