#!/bin/sh
# Exercise 0's answer, checked with no Databricks and no model call:
#   sh week-04/day-04/answers/exercise-01-explain-case-123/commands.sh
# Run it from the course folder. It runs the exercise's check on the notebook's starting state
# (it prints Not yet), then on values.json beside this file (it prints Done:). values.json holds
# the facts the check reads from the answer's run, computed from the made-up files beside the
# exercise's notebook. In Databricks, the notebook's check cell reads the same facts from the notebook.
set -e
python3 week-04/day-04/exercise-01-explain-case-123/check.py --start
python3 week-04/day-04/exercise-01-explain-case-123/check.py week-04/day-04/answers/exercise-01-explain-case-123/values.json
