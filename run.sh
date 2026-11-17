#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir -p logs
.venv/bin/python scripts/prepare_data.py > logs/prepare_data.log 2>&1
.venv/bin/python -m pytest -q tests/test_ocp.py > logs/pytest.log 2>&1
.venv/bin/python scripts/reproduce.py | tee logs/reproduce.log
.venv/bin/python scripts/make_figures.py | tee logs/figures.log
.venv/bin/python scripts/validate_and_report.py | tee logs/validation.log
