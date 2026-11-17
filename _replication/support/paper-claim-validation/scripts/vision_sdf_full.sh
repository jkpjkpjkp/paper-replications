#!/usr/bin/env bash
set -euo pipefail
while systemctl --user is-active --quiet paper-claim-vision-sdf-backend-holdout.service; do
  sleep 20
done
cd /home/jkp/data
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
if financial-graph-reproduction/.venv/bin/python -c 'import json,pathlib,sys; p=pathlib.Path("paper-claim-validation/results/vision_sdf_backend_equivalence.json"); sys.exit(0 if p.exists() and json.loads(p.read_text())["passed"] else 1)'; then
  financial-graph-reproduction/.venv/bin/python -u paper-claim-validation/scripts/vision_sdf_native.py --validated-gram \
    --workers 2 \
    >> paper-claim-validation/results/vision_sdf_full.log 2>&1
else
  financial-graph-reproduction/.venv/bin/python -u paper-claim-validation/scripts/vision_sdf_native.py --workers 2 \
    >> paper-claim-validation/results/vision_sdf_full.log 2>&1
fi
financial-graph-reproduction/.venv/bin/python paper-claim-validation/scripts/compare_claims.py
