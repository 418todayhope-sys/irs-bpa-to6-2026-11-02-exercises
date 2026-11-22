# Databricks notebook source
# MAGIC %md
# MAGIC # The Harbor group's list
# MAGIC Follow CHALLENGE.md. These headings are a starting point; implement and check each stage yourself.

# COMMAND ----------
import hashlib

INPUT_SCHEMA = "training.to6_challenge4_inputs"
CLOSED_TABLE = f"{INPUT_SCHEMA}.closed_returns"
NEW_TABLE = f"{INPUT_SCHEMA}.new_returns"
workspace_user = spark.sql("SELECT current_user() AS username").first()["username"]
EXPERIMENT = f"/Users/{workspace_user}/week-04-challenge"  # your MLflow experiment for this task
OUTPUT_SCHEMA = "training.to6_c4_" + hashlib.sha256(workspace_user.encode("utf-8")).hexdigest()[:12]  # if you save tables
print({"closed": CLOSED_TABLE, "new": NEW_TABLE, "experiment": EXPERIMENT, "output_schema": OUTPUT_SCHEMA})

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Know the group's result

# COMMAND ----------
# Your work here.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. Choose what the model may learn from

# COMMAND ----------
# Your work here.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Train and record two candidates
# MAGIC Outcome 3 also asks for one sentence in SUBMISSION.md: when the chosen model should be retrained, and why.

# COMMAND ----------
# Your work here.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Set the group's list

# COMMAND ----------
# Your work here.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Make the list defensible

# COMMAND ----------
# Your work here.

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Two tasks: trained model or prompting; the report and the exports
# MAGIC For each task, write your choice, your reasons and who checks the output in SUBMISSION.md:
# MAGIC - the cover-sheet summary;
# MAGIC - letters about paying over time, each sent to one of three teams.
# MAGIC
# MAGIC Write SUBMISSION.md and the exports under outputs/ beside this notebook, in week-04/day-05/ of your fork's Git folder.

# COMMAND ----------
# Your work here.
