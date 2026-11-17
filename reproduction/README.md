# LFM Quant reproduction adapter

Run the released-code integration for [Uncertainty-Aware Lookahead Factor Models for Quantitative Investing](https://proceedings.mlr.press/v119/chauhan20a.html). This folder provides explicit repairs to the author source, regression tests and a reproducible synthetic smoke run. The original investment experiment still requires historical Snapshot data and missing empirical construction details; the smoke outputs are not financial evidence.

## Prepare and install

From this source folder:

```bash
python ../../_replication/run.py prepare --project lfm-quant
cd ../../.replication-local/workspace/lfm-quant
mkdir -p logs results evidence
```

The launcher copies code into the run workspace and links the archived environment and shared inputs. Historical runs, evidence and the old status README are preserved under `.replication-local/artifacts/lfm-quant/` at the collection root. It does not duplicate raw data. If no working environment is linked:

```bash
uv venv --python 3.11 .venv
uv pip sync --python .venv/bin/python requirements.lock
```

The recorded environment uses TensorFlow CPU 2.15.1 and Keras 2.15. `runtime/scripts/` contains the tested repaired source. `../Lookahead/` is the unchanged source checkout at commit `ac6f47c9a36f681920314423e2502f3654c5b592`.

## Rerun the implemented integration check

```bash
bash run_smoke.sh
```

This creates a deterministic synthetic panel of twenty companies, runs the dataset/model regression tests, trains point and uncertainty models for two epochs, reloads their weights, makes predictions and independently verifies labels and prediction arithmetic. The launcher archives earlier smoke outputs before replacing them.

For individual stages:

```bash
.venv/bin/python scripts/make_smoke_data.py
.venv/bin/python -m pytest -q scripts/test_dataset.py scripts/test_training.py
./run.sh --config configs/smoke-point.conf
./run.sh --config configs/smoke-point.conf --train False --start_date 200001 --end_date 200212
./run.sh --config configs/smoke-uq.conf
./run.sh --config configs/smoke-uq.conf --train False --start_date 200001 --end_date 200212
.venv/bin/python scripts/verify_runs.py
```

Run these commands from the workspace root. The configurations use relative `results/` paths. `run.sh` sets `LFM_QUANT_ROOT` to the repaired runtime and pins the CPU thread and randomization environment.

To reconstruct the patched source into a new directory:

```bash
.venv/bin/python scripts/prepare_runtime.py --output runtime-rebuilt
```

The target must not exist. This checks the source commit and writes `evidence/runtime.patch`. The test and normal launch scripts continue to use the included `runtime/` until explicitly changed; generating `runtime-rebuilt/` alone does not select it.

## Outputs and checks

Read `results/verification.json`, `logs/tests.log`, the training/prediction logs, and the two result directories `results/smoke-point/` and `results/smoke-uq/`. The checks cover company/date alignment, contiguous histories, label cutoffs, scaling, batch-normalization updates, early stopping and ensemble variance. These are software checks rather than a replication of the paper's portfolio performance.

## Empirical inputs

A full run needs the original historical Snapshot panel, availability/vintage provenance, prices, short interest, industry and capitalization data, plus validated population filters and field mappings. `configs/paper-settings.json` records the extracted paper settings; it is not an executable full empirical configuration.

Existing inputs remain shared under `../compustat/` and `../crsp-*`; do not copy them into this folder. The standard Compustat download is current/restated data and does not establish the original Snapshot history. Optional `scripts/check_wrds_snapshot.py` and `scripts/check_crsp_access.py` use the existing shared WRDS bridge and credential file for read-only access checks; these are not required for the offline synthetic run and do not fetch an empirical panel.

Because the original investment experiment has no validated realized-return panel, this folder does not fabricate a month–firm–return table from synthetic forecasts. The source implementation and dataset contract are documented in [the parent source guide](../README.md).
