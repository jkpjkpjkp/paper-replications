# DoubleAdapt — native CSI300 and monthly CZ adaptation

This folder contains the DoubleAdapt experiment wrappers and pinned author source.
The native daily CSI300 check and monthly US CZ202 adaptation are separate
experiments. The retained scientific review does not establish full numerical
agreement with the original paper and blocks new monthly training.

## Prepare the workspace

From the collection root:

```bash
python _replication/run.py prepare --project doubleadapt
cd .replication-local/workspace/doubleadapt
```

The launcher copies code and links existing shared inputs and archived environments.
Use a new `--workspace /path/to/new-run` for an independent run. Add `--fresh-inputs`
only when rebuilding derived inputs; shared CRSP/CZ inputs remain at the collection
root. No raw data need to be copied into this paper folder.

## Environment and source prerequisites

The native runner uses `.venv-doubleadapt/bin/python` (Python 3.8, PyTorch 1.9 CPU)
and the primary Qlib fork in `sources/DoubleAdapt-qlib/`, including its Git revision
metadata. It imports the installed fork before loading the author example at
`examples/benchmarks_dynamic/incremental/main.py`; the bare source tree does not
contain its compiled Cython extensions. The separate `sources/DoubleAdapt/` API
release is retained as a reference and is not the native runner's import target.

`sources/doubleadapt-constraints.txt` records the core compatibility constraints.
`sources/doubleadapt-environment.lock.txt` is a historical environment freeze,
**not a portable installation lock**: its `pyqlib` entry contains an absolute
`file://` path to the former checkout. For a fresh installation, replace that
entry in a working copy with the staged `sources/DoubleAdapt-qlib/` path and build
the fork using the recorded constraints and CPU PyTorch wheel source. A clean
installation has not been revalidated by this cleanup.

The monthly runner uses `.venv-tra/bin/python`, a separately recorded modern
PyTorch/Qlib environment with `higher`, pandas, NumPy, PyArrow, SciPy and
statsmodels. Staging also restores the historical
`sources/tra-environment.lock.txt`; it has the same absolute Qlib `file://`
reference and is not a portable installation lock. Use the linked environment
locally or resolve and validate a new environment before training. Do not install into an archived environment.

## Inputs

The native runner requires `data/crowd_data/` and
`data/doubleadapt-data-receipt.json`. `scripts/prepare_doubleadapt_data.py`
validates and unpacks the dated author-linked `data/qlib-2023-06-01.tar.gz`;
it does not download that archive. It rejects an existing receipt or nonempty
extraction target.

The monthly runner requires `data/cz-monthly/panel.parquet` and its
`manifest.json`; evaluation also requires `data/factors-monthly/ff5-umd.csv`.
Local staging reuses these inputs from the former ordered-reproductions workspace.
To rebuild in a new workspace prepared with `--fresh-inputs`, run:

```bash
mkdir -p results
.venv-tra/bin/python scripts/preflight_cz.py
.venv-tra/bin/python scripts/build_cz_monthly.py
.venv-tra/bin/python scripts/verify_cz_panel.py
.venv-tra/bin/python scripts/prepare_monthly_factors.py
```

The first three commands read shared `../crsp-monthly/` and
`../Chen_Zimmermann_Factor_Zoo_2026_10/`. The factor script downloads the official
monthly FF5 and momentum archives when absent; its receipts pin the resulting
vintage. The panel and factor builders refuse to replace completed inputs.

## Native original-task run

In a fresh workspace, run:

```bash
.venv-doubleadapt/bin/python scripts/run_doubleadapt_original.py --preflight
.venv-doubleadapt/bin/python scripts/run_doubleadapt_original.py
```

This is the one-seed GRU/CSI300 README configuration. Preflight builds native
Qlib tasks without fitting; the full command trains and independently recomputes
prediction metrics. Outputs live in `results/doubleadapt-original/`. The full
command refuses to overwrite an existing `summary.json`; preflight writes its
own protocol/status files, so use a separate workspace instead of running it
on a historical replay.

## Monthly checks and saved-result replay

The causality/capacity preflight can run without a passed native gate. It loads
the monthly panel and exercises the model with synthetic labels, without
training on the real monthly observations:

```bash
.venv-tra/bin/python scripts/run_doubleadapt_cz.py --preflight
.venv-tra/bin/python scripts/verify_evaluation.py
```

For saved-result verification, prepare a separate replay workspace from the
collection root and restore copies of generated outputs:

```bash
python _replication/run.py prepare --project doubleadapt --workspace .replication-local/doubleadapt-replay --reuse-results
cd .replication-local/doubleadapt-replay/doubleadapt
.venv-tra/bin/python scripts/verify_doubleadapt_cz_results.py
```

That verifier recomputes score coverage, portfolio weights, realized returns,
costs and FF5+UMD HAC inference from saved scores and the source panel. To
regenerate the evaluation into a new directory without fitting a model:

```bash
.venv-tra/bin/python scripts/evaluate_cz.py --scores results/doubleadapt-cz/scores.csv --output results/doubleadapt-cz/evaluation-replay --cost-bps 10 --hac-lags 6
```

Monthly training requires `results/doubleadapt-original/gate.json` with
`status: passed` and SHA-256 hashes matching that run's `summary.json` and
`protocol.json`. The retained gate has
`status: unresolved_configuration_and_performance`, so replaying it does not
unlock training. There is no automated gate-review command in this folder.
Resolve and document the native configuration/performance review before a new
full monthly experiment; do not change the status merely to bypass the check.

Once a valid review exists, `scripts/run_doubleadapt_cz.py --device cpu` or
`--device cuda` is the direct trainer. The optional
`scripts/run_doubleadapt_cz_pipeline.py` hard-codes CUDA, uses
`.venv-tra/bin/python`, and requires the legacy `queue.json` structure as well as
the same passed gate; it is not a portable CPU launch command. Both preserve an
existing monthly `scores.csv` rather than overwrite it.

## Month–firm–return table

From the clean source folder, with pandas, NumPy and PyArrow installed:

```bash
python ../_replication/month_firm_returns.py --project doubleadapt --data-root ..
```

This joins the saved monthly score with the corresponding calendar-month CRSP
return by PERMNO and realization month. A score is not a calibrated return
forecast. Tables and provenance are written outside the code folder under
`.replication-local/tables/`.

Historical notes and generated runs remain under
`.replication-local/artifacts/ordered-reproductions/` at the collection root.
Author licenses remain with their source directories.
