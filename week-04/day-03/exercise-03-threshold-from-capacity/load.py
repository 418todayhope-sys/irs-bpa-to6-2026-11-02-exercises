"""Given code for exercise 3: cell 1 runs it. It reads the made-up files beside this notebook: the ten
returns, case #123, exercise 2's four counts and the 1,000 validation returns. It makes two views that
the SQL of cells 3 and 5 reads: the ten returns, and today's batch, the ten with case #123. It changes
no table. You do not need to read it."""
import pandas as pd


def read_csv(name: str) -> pd.DataFrame:
    """Read a CSV file beside the notebook. Its first line says the course made it up; it is skipped."""
    return pd.read_csv(f"{folder}/{name}", skiprows=1, dtype={"return_id": str, "label": str, "candidate": str})


ten = read_csv("ten-returns.csv")
case_123 = read_csv("case-123.csv").fillna({"label": "no label yet"})  # not examined, so no label
counts = read_csv("candidate-counts.csv")
valid = read_csv("validation-scores.csv")
spark.createDataFrame(ten).createOrReplaceTempView("ten_returns")
spark.createDataFrame(pd.concat([ten, case_123], ignore_index=True)).createOrReplaceTempView("today_batch")
print("Read the ten returns, case #123 and exercise 2's counts.")
