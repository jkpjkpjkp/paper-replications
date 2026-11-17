#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MLFLOW_DISABLE_AGENT_HINT=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.11 .venv
  uv pip sync --python .venv/bin/python requirements.lock.txt
fi
.venv/bin/python scripts/prepare_data.py
.venv/bin/python scripts/prepare_execution.py
.venv/bin/python -m pytest -q
if [ "$#" -eq 0 ]; then
  .venv/bin/python -u scripts/train.py --universes csi500,csi1000 --variants linear --device cpu
fi
.venv/bin/python -u scripts/train.py "$@"
.venv/bin/python scripts/audit.py
.venv/bin/python scripts/report.py
