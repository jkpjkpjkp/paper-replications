#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export LFM_QUANT_ROOT="$PWD/runtime"
export TF_CPP_MIN_LOG_LEVEL=2
export TF_NUM_INTRAOP_THREADS=2
export TF_NUM_INTEROP_THREADS=2
export OMP_NUM_THREADS=2
export PYTHONHASHSEED=521
export TF_ENABLE_ONEDNN_OPTS=0
exec .venv/bin/python runtime/scripts/lfm_quant.py "$@"
