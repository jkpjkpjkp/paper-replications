# Joint return and volatility versus two specialists

Compares one joint LiMT model with independent return-only and volatility-only models. Three seeds per variant, nine fits, use the fixed settings in `protocol.json`. Return is the source price return at t+2 after t+1-close execution; volatility is the t+2 high–low range divided by OHLC4, not a realized monthly standard deviation. Volume is excluded.

## Workspace and inputs

Follow the parent [LiMT US guide](../../README.md) to prepare a runtime workspace
and set `LIMT_MODEL_PYTHON` and `LIMT_ANALYSIS_PYTHON`. Run the commands below
from that workspace's **limt-us root**, not this subdirectory.

Base `limt-us/data/` arrays and source hashes. No previously fitted model is needed; the experiment trains its own variants.

## Run and check

```bash
"$LIMT_MODEL_PYTHON" experiments/joint_price_volatility/run.py --checks-only
"$LIMT_MODEL_PYTHON" experiments/joint_price_volatility/run.py
"$LIMT_MODEL_PYTHON" experiments/joint_price_volatility/run.py --status
```

Checks may include real-data CUDA steps; they are not an environment-free test.
The coordinator holds a run lock and resumes matching checkpoints. Do not start
separate trainers concurrently with it. Changes to data, source or settings can
invalidate the saved fingerprint; use a fresh workspace for a different design.

## Outputs

Outputs stay under `experiments/joint_price_volatility/` in the working directory. Inspect
`analysis/comparison.csv`, `analysis/paired_daily.csv`, `analysis/per_seed_comparison.csv` and `analysis/verification.json`. The experiment is complete only after its status reports completion
and the final verification passes. Saved historical evidence is kept in the
local archive, outside this source folder.
