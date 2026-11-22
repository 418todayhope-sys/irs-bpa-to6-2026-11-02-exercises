"""Given code for exercise 1: cell 1 runs it. It reads the made-up files beside this notebook, fills
the missing withholding amounts as Tuesday did, and fits Tuesday's two candidates on the 2,000
training returns. It gives cells 2 and 3 their two charts: the ranked list of features, and one
return's pushes. It makes one view, returns_to_explain, that cell 3's SQL reads, and changes no table. You do not need to read it."""
import itertools
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["tax_to_agi", "wages_change_pct", "withholding_amt", "withholding_missing"]
NAMES = ("logistic-regression", "decision-tree")
AXIS = "Score: the model's probability of change, from 0 to 1"


def read_csv(name: str) -> pd.DataFrame:
    """Read a CSV file beside the notebook, with Monday's flag. Its first line says the course made it up."""
    table = pd.read_csv(f"{folder}/{name}", skiprows=1, dtype={"return_id": str, "label": str})
    table["withholding_missing"] = table["withholding_amt"].isna().astype(int)  # Monday's flag: 1 where missing
    return table


train = read_csv("train-returns.csv")
returns = pd.concat([read_csv("case-123.csv"), read_csv("ten-returns.csv")], ignore_index=True)
_view = returns.assign(label=returns["label"].fillna("no label yet"))  # case #123 has not been examined
spark.createDataFrame(_view[["return_id", *FEATURES, "label"]]).createOrReplaceTempView("returns_to_explain")
fill = SimpleImputer(strategy="median").fit(train[FEATURES])  # Tuesday's fill: the middle value of each column
background = fill.transform(train[FEATURES])  # the training returns: the average score is taken over them
fitted = {
    "logistic-regression": make_pipeline(StandardScaler(), LogisticRegression()),  # the features first put on one scale
    "decision-tree": DecisionTreeClassifier(max_depth=3, random_state=0),
}
for _model in fitted.values():
    _model.fit(background, train["label"] == "change")
assert chosen in NAMES, "In cell 1, type logistic-regression or decision-tree as chosen."
model = fitted[chosen]


def model_score(rows: np.ndarray) -> np.ndarray:
    """Return the chosen model's score for each row: its probability of change, from 0 to 1."""
    return model.predict_proba(rows)[:, 1]


def weights() -> pd.Series:
    """Return the chosen model's ranked list: the logistic regression's weights, the features first put
    on one scale, without their sign; or the tree's feature importance, as shares of 100."""
    if chosen == "logistic-regression":
        values = np.abs(model[-1].coef_[0])
    else:
        values = model.feature_importances_ * 100
    return pd.Series(values, index=FEATURES).sort_values(ascending=False)


def draw_ranking() -> list[str]:
    """Draw and print the ranked list of features, the longest bar first, and return the names in order."""
    ranked = weights()
    what = "weight, the features first put on one scale" if chosen == "logistic-regression" else "importance, a share of 100"
    plt.figure(figsize=(6, 2.4))
    plt.barh(ranked.index[::-1], ranked.values[::-1], color="#1f77b4")
    plt.title(f"Features the model leans on most ({chosen})")
    plt.xlabel(what)
    plt.tight_layout()
    plt.show()
    print(f"Features the model leans on most ({chosen}; {what}):")
    for name, value in ranked.items():
        print(f"{name:<21} {'#' * max(1, round(30 * value / ranked.iloc[0])):<31} {value:.2f}")
    return list(ranked.index)


def row_of(case_id) -> tuple[str, pd.Series, np.ndarray]:
    """Return a return's id, its row as read, and its four features after the fill."""
    rid = "#" + str(case_id).strip().lstrip("#")
    found = returns[returns["return_id"] == rid]
    assert len(found) == 1, f"No return {rid} here. The returns you can explain: {', '.join(returns['return_id'])}."
    return rid, found.iloc[0], fill.transform(found[FEATURES])[0]


def given_pushes(x: np.ndarray) -> tuple[float, np.ndarray]:
    """Return the average score over the training returns and each feature's push for one return.

    The same method as the SHAP library's exact explainer: for every group of features, the return's
    own values are put in place of each training return's, and the scores are averaged. A push is the
    average change one feature's own value makes, over all the groups. The pushes add up from the
    average score to the return's score.
    """
    n = len(x)
    average = {}
    for size in range(n + 1):
        for group in itertools.combinations(range(n), size):
            rows = background.copy()
            rows[:, list(group)] = x[list(group)]
            average[group] = model_score(rows).mean()
    push = np.zeros(n)
    for j in range(n):
        others = [k for k in range(n) if k != j]
        for size in range(n):
            share = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            for group in itertools.combinations(others, size):
                push[j] += share * (average[tuple(sorted(group + (j,)))] - average[group])
    return average[()], push


def library_pushes(x: np.ndarray):
    """Return the same numbers from the SHAP library, with the library, or None where it is not installed."""
    try:
        import shap
    except ImportError:
        return None
    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.Explainer(model_score, masker, algorithm="exact")
    result = explainer(x.reshape(1, -1))
    return float(np.ravel(result.base_values)[0]), np.asarray(result.values[0], dtype=float), shap


def numbers(case_id) -> dict:
    """Return one return's chart as numbers: its id, the average score, each feature's push, its score."""
    rid, row, x = row_of(case_id)
    base, push = given_pushes(x)
    return {"case": rid, "average": base, "pushes": dict(zip(FEATURES, push)), "score": float(model_score(x.reshape(1, -1))[0]),
            "values": dict(zip(FEATURES, x)), "filled": bool(pd.isna(row["withholding_amt"]))}


def shown(name: str, value: float, filled: bool) -> str:
    """Return a feature's value as the chart line shows it."""
    if name == "withholding_amt":
        return f"{value:,.0f}" + (" (filled)" if filled else "")
    if name == "withholding_missing":
        return f"{value:.0f}"
    return f"{value:g}"


def explain(case_id) -> dict:
    """Draw one return's chart and print it as a table: the average score, each push, the score."""
    chart = numbers(case_id)
    x = np.array(list(chart["values"].values()))
    from_library = library_pushes(x)
    order = sorted(FEATURES, key=lambda f: -abs(chart["pushes"][f]))
    if from_library:
        base, values, shap = from_library
        if abs(base - chart["average"]) > 0.001 or np.abs(values - np.array(list(chart["pushes"].values()))).max() > 0.001:
            print("Note for the trainer: the SHAP library's numbers differ from the given code's by more than 0.001.")
        chart["average"], chart["pushes"] = base, dict(zip(FEATURES, values))
        names = [f"{f} = {shown(f, chart['values'][f], chart['filled'])}" for f in FEATURES]
        shap.plots.waterfall(shap.Explanation(values=values, base_values=base, data=x, feature_names=names), show=False)
        print(f"The pushes are computed by the SHAP library, version {shap.__version__}.")
    else:
        plt.figure(figsize=(6, 2.6))
        lefts = chart["average"] + np.cumsum([0] + [chart["pushes"][f] for f in order[:-1]])
        for i, f in enumerate(order):
            plt.barh(len(order) - i, chart["pushes"][f], left=lefts[i], color="#ff0051" if chart["pushes"][f] > 0 else "#008bfb")
        plt.yticks(range(len(order), 0, -1), [f"{f} = {shown(f, chart['values'][f], chart['filled'])}" for f in order])
        plt.axvline(chart["average"], color="grey", linewidth=0.8)
        print("The SHAP library is not on this compute: the given code computed the pushes, by the same method.")
    plt.xlabel(AXIS)
    plt.title(f"{chart['case']}: from the average score to its score")
    plt.tight_layout()
    plt.show()
    label = ("case " if chart["case"] == "#123" else "return ") + chart["case"]
    print(f"\n{label:<32} {'push':>6}  score so far")
    so_far = chart["average"]
    print(f"{'average score of all returns':<32} {'':>6}  {so_far:.2f}")
    for f in order:
        so_far += chart["pushes"][f]
        print(f"{f + ' = ' + shown(f, chart['values'][f], chart['filled']):<32} {chart['pushes'][f]:+.2f}  {so_far:.2f}")
    print(f"{'score of ' + label:<32} {'':>6}  {chart['score']:.2f}")
    print(f"\nThe axis: {AXIS}.")
    return chart


print(f"Read {len(train)} made-up training returns, case #123 and Wednesday's ten.")
print(f"Fitted Tuesday's two candidates on the training returns. Your candidate: {chosen}.")
