"""The done check for exercise 2. In Databricks, cell 6 of the notebook runs it: it counts each channel's
returns and its returns on the list again, at Wednesday's threshold, and compares them with what you
typed. When the notebook has the proxy step and you ran it, it checks that sentence too; without the
step, the rest is checked alone. Without Databricks, from the course folder:

    python3 week-04/day-04/exercise-02-selection-rates-by-group/check.py --start
    python3 week-04/day-04/exercise-02-selection-rates-by-group/check.py <facts.json>

reads the same facts from a JSON file (or the notebook's starting state). No model is called.
"""
import json
import re
import sys
from pathlib import Path

# The logistic regression's list of 25 on the 250 returns of model_valid, as the day's reading gives it.
START = {"groups": {"A": {"returns": 162, "on_list": 15}, "B": {"returns": 88, "on_list": 10}},
         "threshold": 0.844, "guess_50": "", "group_column": "", "rate_a": None, "rate_b": None, "result": ""}
VERDICT = re.compile(r"\b(unfair|fair|biased|discriminat\w*)\b", re.I)
NOT = re.compile(r"\b(not|cannot|can't|no|without)\b", re.I)


def numbers(text: str) -> list[float]:
    """Return every number in a sentence, with its commas removed."""
    return [float(n) for n in re.findall(r"\d+(?:\.\d+)?", str(text).replace(",", ""))]


def has(found: list[float], *wanted: float) -> bool:
    """Whether the sentence holds one of the wanted numbers, to the places it is given."""
    return any(abs(n - w) <= 0.0006 for n in found for w in wanted)


def proxy_left(f: dict) -> list[str]:
    """The proxy step's line, when the notebook has the step and it ran: could the feature stand in?"""
    if "proxy_sentence" not in f:
        return []
    proxy = str(f["proxy_sentence"])
    a, b = f["medians"]["A"], f["medians"]["B"]
    out = []
    said_not = re.search(r"could not stand in|couldn't stand in|cannot stand in", proxy, re.I)
    if not re.search(r"could stand in", proxy, re.I) and not said_not:
        out.append("Cell 7: write could stand in, or could not stand in. The comparison shows that a feature "
                   "tracks a group, not that the model uses it in the group's place. Step 6.")
    elif said_not:
        out.append("Cell 7: compare the two medians of cell 7 again. Are they close, or far apart? Step 6.")
    if not (has(numbers(proxy), a) and has(numbers(proxy), b)):
        out.append("Cell 7: after because, give the median in each channel, from cell 7. Step 6.")
    return out


def left(f: dict) -> list[str]:
    """Return what is left: one line for each thing that is not yet right.

    Args:
        f: the facts: groups (each channel's returns and returns on the list), threshold, and what the
            learner typed in cells 3 to 5; with the proxy step, medians and proxy_sentence too.

    Returns:
        One line per problem; an empty list when the exercise is done.
    """
    out = []
    a, b = f["groups"]["A"], f["groups"]["B"]
    rate_a, rate_b = a["on_list"] / a["returns"], b["on_list"] / b["returns"]
    if str(f["guess_50"]).strip().lower() not in ("wider", "narrower"):
        out.append("Cell 3: type wider or narrower as your guess. Either counts. Step 2.")
    if str(f["group_column"]).strip() != "made_up_channel":
        out.append("Cell 4: type the name of the column that holds each return's channel. Cell 2 printed the "
                   "columns. Step 3.")
    for name, typed, rate, group in (("rate_a", f["rate_a"], rate_a, "A"), ("rate_b", f["rate_b"], rate_b, "B")):
        try:
            ok = not isinstance(typed, (str, bool)) and abs(float(typed) - rate) < 0.0006
        except (TypeError, ValueError):
            ok = False
        if not ok:
            out.append(f"Cell 5: {name} is channel {group}'s returns on the list divided by its returns in "
                       "model_valid, such as 3 / 50. Step 4.")
    result = str(f["result"])
    found = numbers(result)
    gap = abs(rate_a - rate_b)
    if not all(has(found, n) for n in (a["on_list"], a["returns"], b["on_list"], b["returns"])):
        out.append("Cell 5: give the counts beside the rates: each channel's returns on the list and its returns. "
                   "Step 4.")
    if not has(found, round(f["threshold"], 3)):
        out.append(f"Cell 5: give the threshold, {f['threshold']:.3f}: a gap belongs to a threshold. Step 4.")
    if not has(found, round(gap, 3), round(gap * 100, 1)):
        out.append("Cell 5: state the gap, the difference between the two rates, to three places. Step 4.")
    if not re.search(r"\breason", result, re.I):
        out.append("Cell 5: say that the gap needs a reason. Step 4.")
    if VERDICT.search(result) and not NOT.search(result):
        out.append("Cell 5: a gap is a result that needs a reason. Do not call the model fair or unfair from it. "
                   "Step 4.")
    if re.search(r"\bselect(ed|s)\b", result, re.I):
        out.append("Cell 5: the model puts a return on the list or leaves it off; selecting is a person's decision. "
                   "Step 4.")
    return out + proxy_left(f)


def report(f: dict) -> str:
    """Return what the check cell prints: Done, or Not yet with each thing left."""
    out = left(f)
    a, b = f["groups"]["A"], f["groups"]["B"]
    if not out:
        proxy = (" and whether other_income_amt could stand in for the channel" if "proxy_sentence" in f else "")
        return (f"Done: at {f['threshold']:.3f}, channel A has {a['on_list']} of {a['returns']} on the list and "
                f"channel B {b['on_list']} of {b['returns']}; you stated the gap with its counts{proxy}.")
    return "\n".join([f"Not yet: {len(out)} left."] + ["- " + item for item in out])


def facts_of(g: dict) -> dict:
    """The facts, from the notebook's values: the 250 returns of model_valid as cell 2 put them on the list."""
    s = g["scores"]
    groups = {name: {"returns": int(len(part)), "on_list": int(part["on_list"].sum())}
              for name, part in s.groupby("made_up_channel")}
    facts = {"groups": groups, "threshold": float(g["threshold"]),
             **{name: g.get(name) for name in ("guess_50", "group_column", "rate_a", "rate_b", "result")}}
    if "proxy_sentence" in g:  # the proxy step is in this notebook, and it ran
        medians = g["all_returns"].groupby("made_up_channel")["other_income_amt"].median()
        facts.update(proxy_sentence=g["proxy_sentence"], medians={k: float(v) for k, v in medians.items()})
    return facts


def in_notebook(g: dict) -> str:
    """Return what the check cell prints, from the notebook's values (cell 6 passes them)."""
    if "scores" not in g:
        return "Not yet: 1 left.\n- Run cells 1 and 2 first. Step 1."
    return report(facts_of(g))


if __name__ == "__main__":
    given = START if sys.argv[1:] == ["--start"] else json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    print(report(given))
