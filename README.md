# PRISM-VQ stock ranking replication

Re-run the pinned released PRISM-VQ model with the existing CSI300 and S&P 500
inputs, a two-stage training pipeline, five stage-two seeds and audited portfolio
evaluation. This is an adaptation of the released method, with explicit data and
runtime differences from the published experiment. Frozen market settings are in
`configs/paper-csi300.json` and `configs/paper-sp500.json`.

## Prepare a working directory

Run from this source folder:

```bash
python ../_replication/run.py prepare --project prism-vq
cd ../.replication-local/workspace/prism-vq
```

The preparation command copies code into the working directory and links existing
local inputs and the archived environment. Raw inputs are reused, not copied into
this source package. New outputs are written in the working directory. Historical
runs and the previous README are preserved under
`../.replication-local/artifacts/prism-vq/` relative to this source folder.
Use `prepare --project prism-vq --reuse-results` to create a working copy of saved
results for replay. To reconstruct prepared inputs from shared raw sources, use
`prepare --project prism-vq --fresh-inputs` and run the preparation steps below.
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

The CSI source is shared `csi/qlib/cn_data/`. The S&P 500 construction uses shared
CRSP daily data and historical membership. Prepared arrays are under
`data/csi300/` and `data/sp500/`. Source memberships, JKP thematic-prior archives
and hashes are retained under the archived `sources/` tree and linked into the
workspace. Reuse these sources rather than copying them into the public package.

The JKP inputs have their own source-data license; the upstream code's MIT license
in `upstream/LICENSE` does not grant rights to redistribute those datasets.
The author source is retained under `upstream/`; original version metadata is
restored from the archive in local prepared workspaces.

## Prepare inputs

When using the cached arrays, omit the build commands. To rebuild from existing
shared raw sources, first prepare a fresh-input workspace, then run:

```bash
.venv/bin/python scripts/fetch_membership.py
test -f data/csi300/manifest.json || .venv/bin/python scripts/build_data.py --universe csi300 --workers 2
test -f data/sp500/manifest.json || .venv/bin/python scripts/build_data.py --universe sp500 --workers 2
test -f data/sp500/terminal_events.npy || .venv/bin/python scripts/terminal_events.py
.venv/bin/python -m pytest -q tests -k "not frozen_stage1_and_cached_native_agreement"
```

The omitted cache-equivalence test requires `results/smoke/best.pt`. To run it,
first create a stage-one smoke checkpoint with the command below, then run
`.venv/bin/python -m pytest -q tests` without the filter.

For a new machine, use the scripts' source manifests and pinned upstream data
instructions to obtain any missing inputs before training. The `fetch_membership`
step may contact an external provider; it is not needed for a verified existing
membership cache.

## Train and evaluate

```bash
mkdir -p results
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
.venv/bin/python -u scripts/supervise.py
```

The supervisor runs stage 1 at seed 42 and stage 2 at seeds 0–4 for both markets,
with 50 maximum epochs per stage. It then runs ridge/ensemble evaluations,
verification and reporting. Jobs run serially on CUDA. Completed matching jobs
are resumable; use a new working directory after changing the protocol.

A separate CPU execution check can call `scripts/train.py --universe csi300
--stage 1 --seed 42 --epochs 1 --device cpu --smoke-days 4` which writes
`results/smoke/best.pt`. Smoke outputs are separate from full-suite results.

## CZ202 extension

`experiments/cz_vw/README.md` describes the separate characteristic/value-weighted
experiment. It needs the shared Chen–Zimmermann data, CRSP and factors, plus the
base S&P 500 panel. Run it separately; it does not certify the original-paper
numbers.

## Outputs and verification

Checkpoints, prediction files, ledgers and metrics are generated under `results/`.
`scripts/verify.py` replays held-out forecasts and checks input/model/prediction
hashes and portfolio accounting. `scripts/report.py` rebuilds the summary. The
supervisor writes `.suite-complete` only after successful final verification.

Daily `totalret.npy`, dates and security symbols are available in each prepared
market panel for monthly aggregation. Calendar-month returns compound realized
daily returns; five-day ranking labels remain five-day labels. Missing daily
outcomes must not be silently replaced by zero.

## Export month–firm–return tables

From this clean source folder, use an interpreter with pandas >= 2.1, NumPy and
PyArrow:

```bash
python ../_replication/month_firm_returns.py --project prism-vq --data-root ..
```

The exporter reads existing inputs and archived artifacts in place and writes
derived tables under `../.replication-local/tables/prism-vq/`. Use `--list` to inspect
available market specifications or `--output-root PATH` to choose another output
location. See each table's JSON sidecar for its return definition, coverage and
source paths. Daily-model scores are not represented as monthly return forecasts.
