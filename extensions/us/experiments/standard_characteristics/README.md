# LiMT with Chen–Zimmermann characteristics

Uses lagged monthly Chen–Zimmermann characteristics, the historical US500 universe and the previous-month NYSE20 capitalization screen. Three validation-selected seeds feed value-weighted D10−D1 portfolios. Retain the full feature-selection, lag, transaction-cost and FF5+UMD inference settings in `protocol.json`.

## Workspace and inputs

Follow the parent [LiMT US guide](../../README.md) to prepare a runtime workspace
and set `LIMT_MODEL_PYTHON` and `LIMT_ANALYSIS_PYTHON`. Run the commands below
from that workspace's **limt-us root**, not this subdirectory.

Base LiMT US data, shared CRSP monthly/daily panels, French factors and the pinned Chen–Zimmermann 2026.10 stock-level release. The preparation code checks source hashes; source acquisition is separate from retraining.

## Run and check

```bash
"$LIMT_MODEL_PYTHON" experiments/standard_characteristics/run.py --checks-only
"$LIMT_MODEL_PYTHON" experiments/standard_characteristics/run.py
"$LIMT_MODEL_PYTHON" experiments/standard_characteristics/run.py --status
```

Checks may include real-data CUDA steps; they are not an environment-free test.
The coordinator holds a run lock and resumes matching checkpoints. Do not start
separate trainers concurrently with it. Changes to data, source or settings can
invalidate the saved fingerprint; use a fresh workspace for a different design.

## Outputs

Outputs stay under `experiments/standard_characteristics/` in the working directory. Inspect
`data/universe_audit.csv`, `analysis/alpha.csv`, `analysis/performance.csv`, daily ledgers and `analysis/verification.json`. The experiment is complete only after its status reports completion
and the final verification passes. Saved historical evidence is kept in the
local archive, outside this source folder.
