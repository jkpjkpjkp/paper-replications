#!/usr/bin/env bash
set -euo pipefail
run_dir="${1:-/mnt/data/home/jkp/replications/kronos-base-20261117}"
export PYTHONPATH=/home/jkp/data/.replication-local/artifacts/prism-vq/.venv/lib/python3.11/site-packages:/home/jkp/data/.replication-local/artifacts/alphaprobe/.venv/lib/python3.11/site-packages
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2
export HF_HUB_OFFLINE=1 MLFLOW_DISABLE_AGENT_HINT=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
exec 9>"$run_dir/run.lock"
flock -n 9 || { echo 'Another process owns this run.' >&2; exit 1; }
exec /home/jkp/data/.replication-local/artifacts/prism-vq/.venv/bin/python -u \
  "$run_dir/source/scripts/reproduce_pretrained.py" \
  --output "$run_dir" --stage infer --device cuda:0 \
  --threads 2 --batch-size 8 --resident-layers 0 --seeds 0 1 2
