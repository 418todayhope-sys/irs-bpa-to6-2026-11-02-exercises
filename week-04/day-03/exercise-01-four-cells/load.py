"""Given code for exercise 1: cell 1 runs it. It reads the made-up files beside this notebook: the ten
returns of the day's reading, and the 1,000 validation returns with each candidate's score. It makes
the ten a view that cell 1's SQL reads. It changes no table. You do not need to read it."""
import pandas as pd


def read_csv(name: str) -> pd.DataFrame:
    """Read a CSV file beside the notebook. Its first line says the course made it up; it is skipped."""
    return pd.read_csv(f"{folder}/{name}", skiprows=1, dtype={"return_id": str, "label": str})


ten = read_csv("ten-returns.csv")
valid = read_csv("validation-scores.csv")
spark.createDataFrame(ten).createOrReplaceTempView("ten_returns")  # the ten, for the SQL of cell 1
print("Read", len(ten), "made-up returns and", len(valid), "made-up validation returns.")
