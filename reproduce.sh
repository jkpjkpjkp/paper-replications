#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python 3.11 .venv
fi
uv pip sync --python .venv/bin/python requirements.lock.txt
.venv/bin/python -m pytest -q
.venv/bin/python scripts/build_data.py --universe csi300 --workers 2
.venv/bin/python scripts/build_data.py --universe csi500 --workers 2
.venv/bin/python scripts/fetch_benchmarks.py
.venv/bin/python scripts/run_suite.py --jobs 2
.venv/bin/python scripts/baseline.py --universe csi300
.venv/bin/python scripts/baseline.py --universe csi500
.venv/bin/python scripts/finish.py
