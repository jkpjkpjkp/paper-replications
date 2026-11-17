# Retrieve the Right Signal: Decomposition-Guided Retrieval-Augmented Forecasting for Financial Markets

Completed methodological reconstruction: 640 neural fits across eight methods, two benchmarks, two layouts, four horizons and five seeds, plus 48 classical configurations / 4,495 ARIMA fits. The exact original tuning and preprocessing are not fully released; these are declared TFB comparisons, not an exact original-paper replication.

The 2026-11-17 protocol audit explains 32 published baseline metric cells using
date-complete data, a 36-observation test warm-up, and training-fitted drift.
These choices were inferred after inspecting the paper; the main RISE mismatch
remains. See [the diagnosis](.agent/PROTOCOL_DIAGNOSIS_20261117.txt) and `STATUS.txt`
for the fresh fixed-configuration run and its limitations.

## Setup

From the collection root, stage the code and link the existing inputs:

```bash
python _replication/run.py prepare --project 'rise'
cd '.replication-local/workspace/rise'
```

The command prints the run directory. Run everything below there. Source folders stay free of data and generated artifacts. Inputs are linked from shared storage and `.replication-local/artifacts/paper-reproduction-audit/workspaces/405`; preparation does not duplicate them. A clean checkout still requires these external inputs. Keep their release filenames and directory layout when supplying them.

Use Python 3.11 and a compatible existing environment, or create an isolated environment **inside the run directory**:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
```

Skip environment creation and installation when using a compatible environment already provided by preparation. The requirements record the retained execution environment where known; upstream dependency files remain with the corresponding source. Dependency installation was not repeated during this cleanup.

The full neural suite requires a CUDA-capable PyTorch environment. The classical stage uses statsmodels. The historical run used separate neural and classical environments; full pins are preserved in `requirements-neural.txt`, `requirements-classical.txt` and `environment-lock.json`. For that library match, create separate environments from those files and invoke `run_classical_tfb.py` with the classical interpreter; invoke the other stages with the neural interpreter. The shorter `requirements.txt` is the combined dependency specification, not an exact reconstruction of both original environments.

## Inputs

Prepared TFB benchmark data under `data/`; pinned TFB and GTR/RAFT/DUET implementation trees under `official-tfb/` and `official-extra/`. The frozen configs identify both datasets, layouts, horizons, seeds and model settings. Reuse the prepared arrays; no financial raw data copy is needed.

Inputs and historical outcomes are not redistributed by this source folder. Preserve the retained upstream license files and the separate terms for datasets and model weights.

## Rerun

Execute the stages sequentially:

```bash
python run_rise_tfb.py --device cuda
python run_classical_tfb.py
python run_tfb_neural_baselines.py
python run_remaining_tfb.py
python verify_and_compare_tfb.py
```

Use a new `--workspace` when you need a completely independent run. To inspect saved outcomes, prepare a separate workspace with `--reuse-results`; this restores historical output copies and may trigger the runners' resume/skip behavior. Use the default fresh results for a new experiment. Acquisition/preparation helpers are included, but the commands above reuse the retained inputs. If rebuilding inputs, prepare a new workspace with `--fresh-inputs` so generated files cannot overwrite linked archives.

## Outputs and validation

`results/tfb_rise/`, `results/tfb_classical/`, `results/tfb_neural_baselines/`, `results/tfb_remaining/` and comparison/validation summaries. Completed configurations are checked and skipped on resumption.

The final verifier recomputes forecast metrics from saved predictions and checks the frozen grids. The full neural baseline/remaining stages require CUDA; only the RISE entry point accepts `--device cpu`. Benchmark rows are ordinal and lack reliable calendar/firm metadata for a month-firm table.

For the separate exploratory protocol, use `frozen_tfb_protocol36_config.json`
with the RISE runner's `--config` option, and select a new `--output` directory.
`--data-root` points to the retained input CSV directory; `--threads 1` limits
CPU threads. The `within_split` setting requires both the input and forecast
window to lie within validation/test partitions. Historical defaults use L=96
and allow preceding input context. This configuration does not establish the
unreleased authors' exact protocol. The historical `verify_and_compare_tfb.py`
expects the original grid and borders; use the independent protocol audit for
diagnosis and verify new-run targets against their saved forecast origins.

The cleanup checked source syntax and command paths. It did not start training or rerun the complete experiments. The original reports, logs, environments and outcome files remain in the archive for comparison; they are not substitutes for results from your new run.

## Month–firm–return table

Not exported: the prepared TFB arrays use ordinal row positions and do not establish calendar-month and firm identities. Forecast horizons are overlapping short-horizon labels; their MSE/MAE or predictions cannot be relabeled as monthly realized returns.
