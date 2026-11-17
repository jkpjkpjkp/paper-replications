#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
mkdir -p logs results evidence
export TF_CPP_MIN_LOG_LEVEL=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=2
if [[ -d results/smoke-point || -d results/smoke-uq ]]; then
    archive=$(mktemp -d results/previous-smoke-XXXXXX)
    for item in results/smoke-point results/smoke-uq results/verification.json; do
        if [[ -e "$item" ]]; then mv -- "$item" "$archive/"; fi
    done
fi
.venv/bin/python scripts/make_smoke_data.py
.venv/bin/python -m pytest -q scripts/test_dataset.py scripts/test_training.py > logs/tests.log 2>&1
for mode in point uq; do
    ./run.sh --config "configs/smoke-$mode.conf" > "logs/smoke-$mode-train.log" 2>&1
    ./run.sh --config "configs/smoke-$mode.conf" --train False --start_date 200001 --end_date 200212 > "logs/smoke-$mode-predict.log" 2>&1
done
.venv/bin/python scripts/verify_runs.py | tee logs/verification.log
