# LiMT replication on dated CSI constituents

Re-run the released LiMT model on the existing CSI300/CSI500 community snapshot,
with three-task labels, validation-selected checkpoints and an independently
implemented APO portfolio evaluator. Author code is pinned to
`c0232623d50db26b02275b6f77b682f25effeec5` in `upstream/`.
The community price snapshot starts in 2010 while the paper starts training in 2008, and
the authors' execution engine is not released. `PROTOCOL.md` records these limits
and the fixed modeling/accounting choices.

## Prepare a working directory

Run from this source folder:

```bash
python ../_replication/run.py prepare --project limt
cd ../.replication-local/workspace/limt
```

The preparation command copies code into the working directory and links existing
local inputs and the archived environment. Raw inputs are reused, not copied into
this source package. New outputs are written in the working directory. Historical
runs and the previous README are preserved under
`../.replication-local/artifacts/limt/` relative to this source folder.
Use `prepare --project limt --reuse-results` to create a working copy of saved
results for replay. To reconstruct prepared inputs from shared raw sources, use
`prepare --project limt --fresh-inputs` and run the preparation steps below.
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

The shared raw provider is `csi/qlib/cn_data/`. `scripts/build_data.py` reads it
without changing it and writes Alpha158 features, labels, dated membership and
normalization arrays under `data/csi300/` and `data/csi500/`. Existing arrays and
`sources/benchmarks/` are restored as archive links by workspace preparation.
The benchmark downloader uses cached responses where available.

The separate `published-replication/` intake requires the authors' processed
pickle files and exact source/protocol matching. All six files are available at
`../.replication-local/artifacts/limt/AAAI 2026-HiMT-Dataset/` relative to this
source folder. On 2026-11-17 their sizes, hashes and sample indexes were checked:
both CSI300 and CSI500 training sets span 2008-01-02 through 2014-12-31.
See [the inventory](.agent/AUTHOR_DATA_INVENTORY.json) for all split dates and
file hashes. The archived 2026-11-10 missing-file/download-blocker record predates
these local files and is stale. These inputs are not included with the source
package and have not been used in the completed community-data experiment below.

The released-data run started on 2026-11-17. Its
[protocol](published-replication/AUTHOR_RUN_PROTOCOL.md) uses the original
samplers, released model/training code and stated paper settings, with all 36
loss-weight combinations and three seeds on each universe. Live artifacts are
under `/mnt/data/home/jkp/replications/limt-author-2008-20261117/experiment/`;
`suite_status.json` records progress. The entry point is
`scripts/run_author_suite.py` in the separate PyTorch 1.11/pandas 1.5.3 runtime.
The paper's private configuration and APO execution remain incompletely specified.

## Run the community-data experiment

With cached arrays, start at the tests and training. To reconstruct arrays, first
prepare a fresh-input workspace and run the two `build_data.py` commands:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
test -f data/csi300/manifest.json || .venv/bin/python scripts/build_data.py --universe csi300 --workers 2
test -f data/csi500/manifest.json || .venv/bin/python scripts/build_data.py --universe csi500 --workers 2
.venv/bin/python scripts/fetch_benchmarks.py
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_suite.py --jobs 2
.venv/bin/python scripts/baseline.py --universe csi300
.venv/bin/python scripts/baseline.py --universe csi500
.venv/bin/python scripts/finish.py
```

The suite uses CUDA and three seeds per universe. Interrupted fits resume from
the last saved epoch. For a CPU execution check, use a separate run name:

```bash
.venv/bin/python scripts/train.py --run-name smoke --epochs 2 --smoke 4 --device cpu
.venv/bin/python scripts/backtest.py --run-name smoke
```

This reduced check is not a paper performance estimate. `bash reproduce.sh`
performs setup and the complete sequence; use the explicit commands when reusing
an environment without reinstalling packages.

## Joint versus separate models

```bash
.venv/bin/python scripts/compare_separate.py --jobs 1
.venv/bin/python scripts/compare_separate.py --report-only
```

This extension trains a separate encoder for each task, using settings from the
paired joint run. Results live under `results/separate/`. It has its own run lock
and is separate from the main three-task experiment.

## Outputs and verification

`results/main/<universe>/seed*/` stores checkpoints, predictions, training histories
and metrics. `scripts/finish.py` creates portfolio evaluations and reports, runs
checks, and writes completion evidence. `scripts/status.py` only reads progress;
a status message is not a substitute for `results/main/completion.json` and the
verification artifacts. `scripts/verify.py` checks retained inputs and predictions.

For monthly firm returns, use the underlying realized daily returns or price
panel, retain dated security identifiers, and aggregate by realization month.
The model's return, volume and range/VWAP labels must retain their native horizon.
The US extension source lives in [`extensions/us/`](extensions/us/README.md); its runtime project name is `limt-us`.

## Export month–firm–return tables

From this clean source folder, use an interpreter with pandas >= 2.1, NumPy and
PyArrow:

```bash
python ../_replication/month_firm_returns.py --project limt --data-root ..
```

The exporter reads existing inputs and archived artifacts in place and writes
derived tables under `../.replication-local/tables/limt/`. Use `--list` to inspect
available market specifications or `--output-root PATH` to choose another output
location. See each table's JSON sidecar for its return definition, coverage and
source paths. Daily-model scores are not represented as monthly return forecasts.

## US extension

The separate US experiment is consolidated under [extensions/us/README.md](extensions/us/README.md). Stage it with `--project limt-us`; it keeps its own input contract and protocol.
