# Market-adjusted return and next-month volatility

Trains joint and separate systems for raw and market-adjusted labels, with three seeds and 21 fits. Return remains daily total return at t+2. The volatility label is the next calendar month’s sample standard deviation of daily total returns (ddof=1), without annualization. It is not a monthly return. Split purging covers the full future month.

## Workspace and inputs

Follow the parent [LiMT US guide](../../README.md) to prepare a runtime workspace
and set `LIMT_MODEL_PYTHON` and `LIMT_ANALYSIS_PYTHON`. Run the commands below
from that workspace's **limt-us root**, not this subdirectory.

Base `limt-us/data/`, shared CRSP daily returns and verified French daily factors. The raw and adjusted regimes use the same training-only scales declared in `protocol.json`.

## Run and check

```bash
"$LIMT_MODEL_PYTHON" experiments/market_adjusted_monthly/run.py --checks-only
"$LIMT_MODEL_PYTHON" experiments/market_adjusted_monthly/run.py
"$LIMT_MODEL_PYTHON" experiments/market_adjusted_monthly/run.py --status
```

Checks may include real-data CUDA steps; they are not an environment-free test.
The coordinator holds a run lock and resumes matching checkpoints. Do not start
separate trainers concurrently with it. Changes to data, source or settings can
invalidate the saved fingerprint; use a fresh workspace for a different design.

## Outputs

Outputs stay under `experiments/market_adjusted_monthly/` in the working directory. Inspect
`analysis/summary.csv`, forecast diagnostics, daily ledgers and `analysis/verification.json`. The experiment is complete only after its status reports completion
and the final verification passes. Saved historical evidence is kept in the
local archive, outside this source folder.
