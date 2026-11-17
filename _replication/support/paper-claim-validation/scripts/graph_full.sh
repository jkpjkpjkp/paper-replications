#!/usr/bin/env bash
set -euo pipefail
task_root=/home/jkp/data
task_out="$task_root/paper-claim-validation/results/graph_full"
mkdir -p "$task_out"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=''
cd "$task_root/financial-graph-reproduction"
.venv/bin/python scripts/run_experiment.py --models CS_GAT --profile full \
  --threads 2 --output "$task_out" >> "$task_out/training.log" 2>&1
.venv/bin/python scripts/summarize.py --input "$task_out" >> "$task_out/summary.log" 2>&1
.venv/bin/python "$task_root/paper-claim-validation/scripts/verify_graph_full.py" >> "$task_out/verification.log" 2>&1
.venv/bin/python "$task_root/paper-claim-validation/scripts/compare_claims.py"
