#!/bin/sh
# Exercise 0's answer, checked with no Databricks and no model call:
#   sh week-04/day-03/answers/exercise-02-precision-and-recall/commands.sh
# Run it from the course folder. It runs the exercise's check on the notebook's starting state
# (it prints Not yet), then on values.json beside this file (it prints Done:). values.json holds
# the facts the check reads from the answer's run, as a stand-in in plain Python computed them
# from the made-up files beside the exercise's notebook. In Databricks, the notebook's check cell
# reads the same facts from the notebook.
set -e
python3 week-04/day-03/exercise-02-precision-and-recall/check.py --start
python3 week-04/day-03/exercise-02-precision-and-recall/check.py week-04/day-03/answers/exercise-02-precision-and-recall/values.json
