#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
PYTHON_CZ=${PYTHON_CZ:-/home/jkp/data/.replication-local/artifacts/ordered-reproductions/.venv-tra/bin/python}
"$PYTHON_CZ" -u scripts/run_cz202.py "$@"
"$PYTHON_CZ" scripts/verify_cz202_source.py
"$PYTHON_CZ" scripts/audit_cz202.py
"$PYTHON_CZ" scripts/report_cz202.py
