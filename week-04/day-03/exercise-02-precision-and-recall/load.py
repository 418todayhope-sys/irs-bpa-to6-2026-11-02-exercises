"""Given code for exercise 2: cell 1 runs it. It reads the made-up file beside this notebook: the 1,000
validation returns, each with its label and its score from each of Tuesday's two candidates. It
changes no table. You do not need to read it."""
import pandas as pd


def read_csv(name: str) -> pd.DataFrame:
    """Read a CSV file beside the notebook. Its first line says the course made it up; it is skipped."""
    return pd.read_csv(f"{folder}/{name}", skiprows=1, dtype={"return_id": str, "label": str})


valid = read_csv("validation-scores.csv")
print("Read", len(valid), "made-up validation returns,", (valid["label"] == "change").sum(), "of them labelled change.")
