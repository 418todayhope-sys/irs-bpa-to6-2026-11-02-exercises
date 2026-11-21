#!/bin/sh
# Exercise 2's answer, checked with no Databricks and no model call:
#   sh week-04/day-02/answers/exercise-02-fit-two-candidates/commands.sh
# Run it from the course folder. It runs the exercise's check on the notebook's starting state
# (it prints Not yet), then on values.json beside this file (it prints Done:). values.json holds
# the facts the check reads from the answer's run, as a stand-in in plain Python computed them
# from the course's week 4 tables (week-04/setup/day-01-course-data.py). In Databricks, the
# notebook's cell 9 reads the same facts from the notebook.
set -e
python3 week-04/day-02/exercise-02-fit-two-candidates/check.py --start
python3 week-04/day-02/exercise-02-fit-two-candidates/check.py week-04/day-02/answers/exercise-02-fit-two-candidates/values.json
