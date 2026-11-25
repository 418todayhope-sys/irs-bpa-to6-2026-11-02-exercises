"""Given code for exercise 3: cell 2 runs it. It reads the course's training and validation sets and
fits the same two candidates as exercise 2, with the same cells: the twelve features marked before, an
empty amount read as 0, make_pipeline(StandardScaler(), LogisticRegression()) and
DecisionTreeClassifier(max_depth=3, random_state=0). It changes no table. You do not need to read it."""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["filing_status", "wages_amt", "taxable_interest_amt", "other_income_amt", "total_income", "agi_amt",
            "deduction_amt", "deduction_missing", "total_tax_amt", "withholding_amt", "tax_to_agi", "deduction_to_agi"]
COPY = ("The tables course.returns.model_train and model_valid couldn't be read, so this notebook makes the "
        "same rows from the course's setup file, week-04/setup/day-01-course-data.py. Every number is the same.")
SETUP = f"{course_folder}/week-04/setup/day-01-course-data.py"


def course_rows(name: str) -> pd.DataFrame:
    """The rows of one of the course's sets, made by the course's setup file, as the table holds them."""
    made = {"__name__": "course_data"}
    exec(open(SETUP, encoding="utf-8").read(), made)
    return pd.DataFrame(made["set_rows"](name))


def read_sets(train_table, valid_table):
    """Return (train, valid) as pandas tables, from the course's tables or, when they cannot be read,
    from the course's setup file, with the flag deduction_missing: 1 where deduction_amt is empty."""
    try:
        train, valid = train_table.toPandas(), valid_table.toPandas()
        own = len(train) > 0 and len(valid) > 0
    except Exception:
        own = False
    if own:
        print("Using the course's tables course.returns.model_train and course.returns.model_valid.")
    else:
        train, valid = course_rows("model_train"), course_rows("model_valid")
        print(COPY)
    out = []
    for table in (train, valid):
        table = table.copy()
        table["deduction_missing"] = table["deduction_amt"].isna().astype(int)  # Monday's flag: 1 where empty
        table["exam_changed_tax"] = table["exam_changed_tax"].where(table["exam_changed_tax"].notna(), None)
        out.append(table.reset_index(drop=True))
    return out[0], out[1]


def model_name(model) -> str:
    """The model's type: LogisticRegression inside its pipeline, or DecisionTreeClassifier."""
    return type(model[-1] if hasattr(model, "steps") else model).__name__


def fit_candidates(train_table, valid_table):
    """Return (candidates, split): {run name: (fitted model, validation accuracy)} for the two
    candidates of exercise 2, and the split they were fitted and measured on, in words."""
    train, valid = read_sets(train_table, valid_table)
    train, valid = train[train["exam_changed_tax"].notna()], valid[valid["exam_changed_tax"].notna()]
    X_train, X_valid = train[FEATURES].fillna(0), valid[FEATURES].fillna(0)
    candidates = {}
    for name, model in (("logistic-regression", make_pipeline(StandardScaler(), LogisticRegression())),
                        ("decision-tree", DecisionTreeClassifier(max_depth=3, random_state=0))):
        model.fit(X_train, train["exam_changed_tax"])
        candidates[name] = (model, accuracy_score(valid["exam_changed_tax"], model.predict(X_valid)))
    years = sorted(int(y) for y in train["tax_year"].unique())
    split = (f"train: tax years {years[0]} to {years[-1]}; validation: tax year "
             f"{int(valid['tax_year'].iloc[0])}, {len(valid)} returns with a label")
    print("Fitted again, as in exercise 2:", ", ".join(model_name(m) for m, _ in candidates.values()))
    print("Split:", split)
    return candidates, split
