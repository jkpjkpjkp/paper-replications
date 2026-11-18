# PDU stock-ranking reproduction

Re-run the repaired implementation and reconstructed NASDAQ/NYSE inputs for the
PDU paper identified by the source record in the archived `sources/provenance.json`.
The original release has missing-data and implementation inconsistencies;
`repaired/PDUArch.py`, `model.py` and `metrics.py` contain the explicit repairs.
The original code remains under `official/`. This package does not claim that the
paper's unreleased frozen input bundle has been recovered.

## Prepare a working directory

Run from this source folder:

```bash
python ../_replication/run.py prepare --project pdu
cd ../.replication-local/workspace/pdu
```

The preparation command copies code into the working directory and links existing
local inputs and the archived environment. Raw inputs are reused, not copied into
this source package. New outputs are written in the working directory. Historical
runs and the previous README are preserved under
`../.replication-local/artifacts/pdu/` relative to this source folder.
Use `prepare --project pdu --reuse-results` to create a working copy of saved
results for replay. To reconstruct prepared inputs from shared raw sources, use
`prepare --project pdu --fresh-inputs` and run the preparation steps below.
When switching from cached to fresh inputs, pass a new `--workspace PATH` so
existing input links remain untouched.
Keep each changed experiment in a separate working directory; retained fingerprints
may reject mixing changed code or settings with old results.

## Environment

Use the linked `.venv/bin/python` when the original environment is available.
For a fresh Python 3.11 environment, run these commands **in the working directory**:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
```

The lock file records the tested versions. GPU runs need a compatible PyTorch/CUDA
installation; installing the environment is separate from running experiments.

## Inputs

`data/NASDAQ.npz` and `data/NYSE.npz` are the cached reconstructed arrays.
Their JSON sidecars record source hashes, split dates and preprocessing.
The raw RSR data and relation files live in `reference-rsr/data/`, pinned at
commit `cfbb01bdf194b81bc5893a1b37aff1c0d0d2a82d` from
`https://github.com/fulifeng/Temporal_Relational_Stock_Ranking`.

The prepared workspace restores the existing upstream checkout metadata and
cached inputs. If acquiring the source afresh, create that pinned Git checkout at
`reference-rsr/` before running `acquire_rsr.py`; the script expects an actual Git
checkout and expands its sparse data selection. `reference-hats/` records the
related HATS reference, not a replacement for the unavailable PDU S&P 500 bundle.

`run.sh` uses `PDU_PYTHON`, or `python3` when it is unset. Set it explicitly:

```bash
export PDU_PYTHON="$PWD/.venv/bin/python"
```

## Audit and prepare

```bash
mkdir -p results
./run.sh audit_release.py
./run.sh -m unittest -v test_reproduction
# Only when raw RSR files are absent:
./run.sh acquire_rsr.py
# Omit these two when reusing the checked NPZ cache:
test -f data/NASDAQ.npz || ./run.sh prepare_reference.py --market NASDAQ
test -f data/NYSE.npz || ./run.sh prepare_reference.py --market NYSE
./run.sh audit_data.py
```

## Train and summarize

```bash
./run.sh train.py --market NASDAQ --tag nasdaq-seed42 --seed 42
./run.sh train.py --market NASDAQ --tag nasdaq-seed43 --seed 43
./run.sh train.py --market NASDAQ --tag nasdaq-seed44 --seed 44
./run.sh train.py --market NYSE --tag nyse-seed42 --seed 42
./run.sh train.py --market NYSE --tag nyse-seed43 --seed 43
./run.sh train.py --market NYSE --tag nyse-seed44 --seed 44
./run.sh summarize.py
```

Training defaults to 300 maximum epochs, patience 20 and automatic CUDA/CPU
selection. Pass `--device cpu` explicitly when needed. `--stocks` and
`--train-steps` produce reduced diagnostics and must not be labeled full runs.
Use a unique `--tag` for a changed run. The summary consumes available saved runs;
inspect its coverage rather than assuming all requested runs completed.

For one bounded NASDAQ cloud case, `cloud_case.py` runs the seven correctness
tests, benchmarks checkpointed versus batched graph execution, trains seed 42
on the full universe, and audits the saved predictions with `verify_case.py`.
It selects execution mode using throughput only, before training, and retains
the same model equations. The default remains checkpointed execution in
`train.py`; `--graph-mode batched` requires substantially more GPU memory.

```bash
./run.sh cloud_case.py --max-hours 23 --cpu-threads 4
```

The runner writes `case-receipt.json`, `benchmark.json`, `environment.txt`,
`logs/`, `results/`, and `case-artifacts.tar.gz`. Its process time limit does
not stop cloud billing: set a provider shutdown deadline separately and stop
the instance after retrieving artifacts. The predeclared paper comparison
screens both SR and MRR against two published seed standard deviations;
this is descriptive, not a confidence interval or proof of exact replication.
Input, trading-horizon and metric discrepancies still apply.

## Outputs and calendar-month returns

Each `results/<tag>/` retains checkpoints, predictions and metrics. The summary
and `results/data_audit.json` provide run and input checks. Preserve the distinction
between close-to-close returns and the paper's described close-to-next-open trade.

The prepared arrays have shape `(firm, date, feature)`. Monthly price returns can
be computed from `price` at exact adjacent month-end dates using `mask`, with
`tickers` as security identifiers. Missing endpoints remain missing. Neither the
`truth` prediction label nor `overnight` is a calendar-month return by itself.

## Export month–firm–return tables

From this clean source folder, use an interpreter with pandas >= 2.1, NumPy and
PyArrow:

```bash
python ../_replication/month_firm_returns.py --project pdu --data-root ..
```

The exporter reads existing inputs and archived artifacts in place and writes
derived tables under `../.replication-local/tables/pdu/`. Use `--list` to inspect
available market specifications or `--output-root PATH` to choose another output
location. See each table's JSON sidecar for its return definition, coverage and
source paths. Daily-model scores are not represented as monthly return forecasts.
