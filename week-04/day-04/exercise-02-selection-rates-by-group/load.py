"""Given code for exercise 2: cell 1 runs it. It reads the made-up files beside this notebook: the
1,000 validation returns with Wednesday's scores and a synthetic group, and Wednesday's ten returns.
It puts each return on the list or off it at the threshold from capacity, and makes two views that
the SQL of cells 3 and 5 reads. It changes no table. You do not need to read it."""
import pandas as pd

NAMES = ("logistic-regression", "decision-tree")
CAPACITY = {"validation": 30, "ten": 3}  # the cases the examiners take: 30 of the 1,000, 3 of the ten
assert chosen in NAMES, "In cell 1, type logistic-regression or decision-tree as chosen."


def read_csv(name: str) -> pd.DataFrame:
    """Read a CSV file beside the notebook. Its first line says the course made it up; it is skipped."""
    return pd.read_csv(f"{folder}/{name}", skiprows=1, dtype={"return_id": str, "label": str})


def on_the_list(table: pd.DataFrame, k: int) -> pd.DataFrame:
    """Return the table with its score and a column on_list: 1 for each of the k returns the examiners
    take, 0 for the others. The list never holds more than k. Returns above the k-th score all go on it;
    when returns share the k-th score, a person breaks the tie: here, in order of return ID."""
    candidate = chosen if chosen in table.columns else "logistic-regression"  # the ten have the regression's scores only
    table = table.assign(score=table[candidate])
    threshold = table["score"].sort_values(ascending=False).iloc[k - 1]
    above = table["score"] > threshold
    tied = table[table["score"] == threshold].sort_values("return_id")  # the tie broken by return ID
    room = k - int(above.sum())
    chosen_ids = set(tied["return_id"].head(room))
    on_list = (above | table["return_id"].isin(chosen_ids)).astype(int)
    return table.assign(on_list=on_list, tied_at_threshold=len(tied)), threshold


validation, threshold = on_the_list(read_csv("validation-returns.csv"), CAPACITY["validation"])
ten, threshold_ten = on_the_list(read_csv("ten-returns.csv"), CAPACITY["ten"])
spark.createDataFrame(validation.drop(columns=list(NAMES) + ["tied_at_threshold"], errors="ignore")).createOrReplaceTempView("validation_returns")
spark.createDataFrame(ten.drop(columns=list(NAMES) + ["tied_at_threshold"], errors="ignore")).createOrReplaceTempView("ten_returns")
print(f"Read {len(validation)} made-up validation returns, scored by your candidate, {chosen}.")
print(f"The examiners take {CAPACITY['validation']} of them: the threshold is the {CAPACITY['validation']}th score, "
      f"{threshold}, and {validation['on_list'].sum()} returns are on the list.")
if validation["tied_at_threshold"].iloc[0] > 1:
    print(f"{validation['tied_at_threshold'].iloc[0]} returns share the threshold, more than the list has room for: "
          "the tie is broken by return ID, the lower IDs first.")
print("Columns:", ", ".join(c for c in validation.columns if c not in NAMES))
