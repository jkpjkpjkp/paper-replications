# Three specialists versus three-head LiMT

Adds independent volume-only models to the completed return/volatility specialists and compares them with the original joint LiMT through the same APO portfolio engine. Preserve gross, buy-only 5 bp and all-trade 10 bp cost definitions in `protocol.json`.

## Workspace and inputs

Follow the parent [LiMT US guide](../../README.md) to prepare a runtime workspace
and set `LIMT_MODEL_PYTHON` and `LIMT_ANALYSIS_PYTHON`. Run the commands below
from that workspace's **limt-us root**, not this subdirectory.

The completed base three-head model and all seeds of the `joint_price_volatility` specialists are required, as well as the original base data. First run those experiments or prepare a workspace with `--reuse-results`.

## Run and check

```bash
"$LIMT_MODEL_PYTHON" experiments/separate_three_task_portfolio/run.py --checks-only
"$LIMT_MODEL_PYTHON" experiments/separate_three_task_portfolio/analysis_support/launch.py
"$LIMT_MODEL_PYTHON" experiments/separate_three_task_portfolio/run.py --status
```

Checks may include real-data CUDA steps; they are not an environment-free test.
The coordinator holds a run lock and resumes matching checkpoints. Do not start
separate trainers concurrently with it. Changes to data, source or settings can
invalidate the saved fingerprint; use a fresh workspace for a different design.

## Outputs

Outputs stay under `experiments/separate_three_task_portfolio/` in the working directory. Inspect
`analysis/summary.csv`, `analysis/paired_profitability.csv`, daily ledgers and `analysis/verification.json`. The experiment is complete only after its status reports completion
and the final verification passes. Saved historical evidence is kept in the
local archive, outside this source folder.
