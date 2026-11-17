#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
STABLE_PYTHON="${STABLE_PYTHON:-/home/jkp/data/limt/.venv/bin/python}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2
mkdir -p logs results/best_effort checkpoints
"$STABLE_PYTHON" verify.py > logs/equation_checks.log 2>&1
"$STABLE_PYTHON" -m unittest test_empirical -v > logs/empirical_checks.log 2>&1
"$STABLE_PYTHON" -u run_empirical.py | tee logs/empirical_run.log
"$STABLE_PYTHON" rescore_forecasts.py
"$STABLE_PYTHON" summarize_empirical.py
