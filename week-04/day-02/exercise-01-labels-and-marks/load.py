"""Given code for exercise 1: cell 2 runs it. It reads the course's training set,
course.returns.model_train, and puts the made-up columns of two files beside it, matched by
return_id: the columns of fields.csv, and the demo's label refund_held with refund_hold_days from
refund-held.csv. It also gives Monday's feature record, as Monday's exercise 3 ends, so nothing saved
from Monday is read. It changes no table. You do not need to read it."""
import pandas as pd

FOLDER = f"{course_folder}/week-04/day-02/exercise-01-labels-and-marks"
COPY = ("The table course.returns.model_train couldn't be read, so this notebook makes the same rows from "
        "the course's setup file, week-04/setup/day-01-course-data.py. Every count is the same.")
# Monday's feature record as the answer to Monday's exercise 3 gives it. Nothing saved from Monday is read:
# learners found it hard when one day's exercises carried into the next.
MONDAY_RECORD = [
    ("tax_to_agi", "Form 1040 line 24, total tax; line 11a, adjusted gross income", "yes"),
    ("tax_added_by_exam", "the examination's result", "no"),
    ("deduction_amt", "Form 1040 line 12e, standard deduction or itemized deductions", "yes"),
    ("deduction_missing", "Form 1040 line 12e, empty or not", "yes"),
]
SETUP = f"{course_folder}/week-04/setup/day-01-course-data.py"


def course_rows(name: str) -> pd.DataFrame:
    """The rows of one of the course's sets, made by the course's setup file, as the table holds them."""
    made = {"__name__": "course_data"}
    exec(open(SETUP, encoding="utf-8").read(), made)
    return pd.DataFrame(made["set_rows"](name))


def read_csv(name: str) -> pd.DataFrame:
    """Read a CSV file beside the notebook. Its first lines, which start with #, are skipped."""
    return pd.read_csv(f"{FOLDER}/{name}", comment="#", dtype={"return_id": str})


def read_notes(name: str) -> dict:
    """What a CSV file beside the notebook says of each column: its lines '# <column>: <note>'."""
    notes = {}
    with open(f"{FOLDER}/{name}", encoding="utf-8") as lines:
        for line in lines:
            if not line.startswith("#"):
                break
            column, _, note = line[1:].strip().partition(": ")
            if note and " " not in column:
                notes[column] = note
    return notes


def load_day(train_table):
    """Return (the training set with the files' made-up columns, Monday's feature record as
    (feature, source field, known at selection)). Print which table is used, then the record."""
    try:
        train = train_table.toPandas() if train_table is not None else None
    except Exception:
        train = None
    if train is None or not len(train):
        train = course_rows("model_train")
        print(COPY)
    else:
        print("Using the course's training set, course.returns.model_train:", len(train), "returns.")
    train = train.merge(read_csv("fields.csv"), on="return_id", how="left")
    train = train.merge(read_csv("refund-held.csv"), on="return_id", how="left")
    for column in ("exam_changed_tax", "refund_held"):
        train[column] = train[column].where(train[column].notna(), None)
    record = MONDAY_RECORD
    print("Monday's feature record:")
    for feature, source, known in record:
        print(f"  {feature}: {source}. Known when the case is selected: {known}")
    return train, record
