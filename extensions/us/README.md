# LiMT on US CRSP data

Re-run the US transfer of LiMT with a dynamic historical 500-security universe,
three prediction tasks and a matched LightGBM comparator. This is a separate
experiment from the paper's CSI300/CSI500 benchmark. `protocol.json` fixes the
training/validation/test dates and portfolio assumptions; OHLC4 substitutes for
the unobserved CRSP VWAP field.

## Prepare a working directory

Run from this source folder:

```bash
python ../../../_replication/run.py prepare --project limt-us
cd ../../../.replication-local/workspace/limt-us
```

The preparation command copies code into the working directory and links existing
local inputs and the archived environment. Raw inputs are reused, not copied into
this source package. New outputs are written in the working directory. Historical
runs and the previous README are preserved under
`../../../.replication-local/artifacts/limt-us/` relative to this source folder.
Use `prepare --project limt-us --reuse-results` to create a working copy of saved
results for replay. To reconstruct prepared inputs from shared raw sources, use
`prepare --project limt-us --fresh-inputs` and run the preparation steps below.
When switching from cached to fresh inputs, pass a new `--workspace PATH` so
existing input links remain untouched.
Keep each changed experiment in a separate working directory; retained fingerprints
may reject mixing changed code or settings with old results.

## Environment and shared inputs

Use Python 3.11. The model/data and inference environments are pinned separately:

```bash
uv venv --python 3.11 .venv-model
uv pip install --python .venv-model/bin/python -r requirements-model.lock.txt
uv venv --python 3.11 .venv-analysis
uv pip install --python .venv-analysis/bin/python -r requirements-analysis.lock.txt
export LIMT_MODEL_PYTHON="$PWD/.venv-model/bin/python"
export LIMT_ANALYSIS_PYTHON="$PWD/.venv-analysis/bin/python"
```

Existing compatible archived environments can be selected through the same two
variables. The coordinator defaults to its own interpreter when either variable
is absent; use explicit variables when analysis dependencies are in another
environment. The model lock expects the recorded CUDA/PyTorch setup.

The raw source is the shared `crsp-daily/parquet/` dataset. Additional extension
inputs include shared CRSP monthly data, factor returns and the Chen–Zimmermann
characteristic panel. The exact paths are declared in each extension's `common.py`
and `protocol.json`. Prepared `data/` arrays are reused from the archive by default.
No licensed CRSP or feature inputs are copied into this source folder.

## Re-run the base experiment

From the prepared workspace, with the two interpreter variables set:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$LIMT_MODEL_PYTHON" run.py
```

`run.py` holds an exclusive coordinator lock. It builds or reuses the checked
input panel, runs tests, launches at most two training processes, fits all three
seeds, runs the LightGBM baseline, computes portfolio inference and audits results.
It no longer depends on an interpreter under a machine-specific old folder name.
For individual stages, run:

```bash
"$LIMT_MODEL_PYTHON" usdata.py
"$LIMT_MODEL_PYTHON" -m pytest tests -q
"$LIMT_MODEL_PYTHON" train.py --seed 0
"$LIMT_MODEL_PYTHON" train.py --seed 1
"$LIMT_MODEL_PYTHON" train.py --seed 2
"$LIMT_MODEL_PYTHON" baseline.py
"$LIMT_ANALYSIS_PYTHON" analyze.py
"$LIMT_ANALYSIS_PYTHON" audit_results.py
```

Training uses 2008–2014, validation 2015–2016 and testing 2017–2025. The portfolio
assumes a decision at t, execution at t+1 close and payoff at t+2. Outputs and
source/input fingerprints preserve that timing.

## Extensions

Run extensions only after their documented base inputs exist. Each directory has
its own `README.md`, `protocol.json`, output folders and checks:

| Directory under `experiments/` | Purpose |
| --- | --- |
| `joint_price_volatility/` | Joint versus separate return/volatility prediction |
| `separate_three_task_portfolio/` | Three specialists through the same portfolio engine |
| `market_adjusted_monthly/` | Market-adjusted daily return and next-month volatility |
| `standard_characteristics/` | CZ202, lagged NYSE20 screen, value-weighted D10−D1 |
| `rfs_frontier/` | Additional portfolio/economic-value diagnostics |

## Outputs and verification

`results/seed*/` contains LiMT checkpoints and predictions, `baseline/` contains
LightGBM artifacts, and `analysis/` contains forecasts, ledgers, paired comparisons
and verification. Require `run_status.json` to say `complete`, and check both
`analysis/verification.json` and `analysis/final-checks.json`.

Month–firm–return tables must use realized calendar-month CRSP returns. Keep
PERMNO/security identity and formation timing; do not relabel daily predictions
or next-month realized volatility as monthly returns. Preserve missing outcomes
and identify any model score separately from the realized return.

## Export month–firm–return tables

From this clean source folder, use an interpreter with pandas >= 2.1, NumPy and
PyArrow:

```bash
python ../../../_replication/month_firm_returns.py --project limt-us --data-root ../../..
```

The exporter reads existing inputs and archived artifacts in place and writes
derived tables under `../../../.replication-local/tables/limt-us/`. Use `--list` to inspect
available market specifications or `--output-root PATH` to choose another output
location. See each table's JSON sidecar for its return definition, coverage and
source paths. Daily-model scores are not represented as monthly return forecasts.
