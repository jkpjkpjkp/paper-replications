# HGA-MT price-only diagnostic

Re-run the independent price-only diagnostic for Ma et al., *Quantitative stock
portfolio optimization by multi-task learning risk and return*, Information Fusion
104 (2024), 102165. The available code compares return-only and multitask models.
Original RESSET features, Hundsun news/MSC embeddings and typed Qichacha relations
are not available here, so this does not reproduce the full published HGA-MT model.
`protocol.json` fixes the local experiment and six fits.

## Prepare a working directory

Run from this source folder:

```bash
python ../_replication/run.py prepare --project hga-mt
cd ../.replication-local/workspace/hga-mt
```

The preparation command copies code into the working directory and links existing
local inputs and the archived environment. Raw inputs are reused, not copied into
this source package. New outputs are written in the working directory. Historical
runs and the previous README are preserved under
`../.replication-local/artifacts/hga-mt/` relative to this source folder.
Use `prepare --project hga-mt --reuse-results` to create a working copy of saved
results for replay. To reconstruct prepared inputs from shared raw sources, use
`prepare --project hga-mt --fresh-inputs` and run the preparation steps below.
When switching from cached to fresh inputs, pass a new `--workspace PATH` so
existing input links remain untouched.
Keep each changed experiment in a separate working directory; retained fingerprints
may reject mixing changed code or settings with old results.

## Environment

Use the linked `.venv/bin/python` when the original environment is available.
For a fresh Python 3.11 environment, run these commands **in the working directory**:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock.txt
```

The lock file records the tested versions. GPU runs need a compatible PyTorch/CUDA
installation; installing the environment is separate from running experiments.

## Inputs

The raw price source is the shared `csi/qlib/cn_data/` snapshot. The preparation
script reads `calendars/day.txt` and per-security `*.day.bin` fields in place.
The universe is the 86 securities extracted from Table B.9 in the locally retained
`sources/paper.txt`. Supply the same published-paper text if rebuilding on another
machine. Its archived provenance is in `sources/provenance.json`.

`data/price_only.npz`, `data/universe.csv` and `data/audit.json` are generated inputs.
The common preparation command links the existing cache. Do not redistribute the
raw snapshot with these sources. `requirements.lock.txt` records the original
reused CPU environment; the `torch==2.4.0+cpu` wheel requires PyTorch's CPU wheel
index when rebuilding that exact environment.

## Run

With cached inputs, omit `scripts/prepare.py`. To rebuild them, prepare a fresh
input workspace first. Then execute, in order:

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p results
test -f data/price_only.npz || .venv/bin/python scripts/prepare.py
.venv/bin/python scripts/audit_paper.py
.venv/bin/python scripts/checks.py
.venv/bin/python scripts/run.py
.venv/bin/python scripts/verify.py
.venv/bin/python scripts/evaluate.py
```

`run.py` reuses completed matching fits. Its input/source lock prevents combining
old predictions with a changed design. Use a new workspace for changed settings.
No real relationship-graph model is trained by this price-only workflow.

## Outputs and validation

`results/return_only_seed*/` and `results/multitask_seed*/` contain model weights,
training histories and predictions. `results/daily_metrics.csv` and the generated
report contain ranking and portfolio comparisons. `checks.py` exercises the
method and metric helpers; `verify.py` replays predictions and checks input
alignment. The generated report distinguishes the diagnostic from the unavailable
original experiment.

For calendar-month exports, `price_only.npz` contains `symbols`, `yret` and
`target_dates`. `yret` is a next-day price return, so assign it to `target_dates`,
not the feature date, before compounding. Do not use the future risk-label mask
to select the return sample.

## Export month–firm–return tables

From this clean source folder, use an interpreter with pandas >= 2.1, NumPy and
PyArrow:

```bash
python ../_replication/month_firm_returns.py --project hga-mt --data-root ..
```

The exporter reads existing inputs and archived artifacts in place and writes
derived tables under `../.replication-local/tables/hga-mt/`. Use `--list` to inspect
available market specifications or `--output-root PATH` to choose another output
location. See each table's JSON sidecar for its return definition, coverage and
source paths. Daily-model scores are not represented as monthly return forecasts.
