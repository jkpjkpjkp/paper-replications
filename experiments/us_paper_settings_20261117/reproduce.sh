#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
STABLE_PYTHON="${STABLE_PYTHON:-/home/jkp/data/.replication-local/artifacts/limt/.venv/bin/python}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=2
mkdir -p logs
"$STABLE_PYTHON" -u -c 'import runpy, torch; torch.cuda.set_per_process_memory_fraction(0.075) if torch.cuda.is_available() else None; runpy.run_path("run_empirical.py", run_name="__main__")' --markets US --seeds 20261120 20261121 | tee logs/two_seeds.log
"$STABLE_PYTHON" verify_results.py
"$STABLE_PYTHON" publish_results.py
