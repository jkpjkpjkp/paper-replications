# Diffusion-DFL — portfolio replication

Alternative CRSP data and reduced training budget; the original WIKI snapshot and full published compute specification are distinct. The source retains synthetic and power-scheduling implementations, but this guide runs the local portfolio experiment.

## Prepare the runnable workspace

From this source folder:

```bash
python ../_replication/run.py prepare --project "diffusion-dfl"
cd ../.replication-local/workspace/"diffusion-dfl"
```

The launcher copies code and links existing local inputs. Raw data stay shared at
the root. Runs write to the staged workspace, keeping this source folder clean.
Use `--workspace /path/to/new-run` for an independent run; use `--fresh-inputs` when
rebuilding prepared data, and `--reuse-results` only to restore historical generated
outputs for replay. Historical README/status notes live under
`../.replication-local/artifacts/diffusion-dfl/`.

## Environment

Python 3.11; install requirements-reproduction.txt. The local environment is reused when available.

If staging restored `.venv`, use its Python without reinstalling packages into the
archived environment. For a new installation, create an environment outside the
source folder and activate it before running the commands below.

```bash
python -m pip install -r requirements-reproduction.txt
```

## Inputs

Shared CRSP daily Parquet partitions for 2004–2016 and the historical January 2004 S&P 500 membership receipt. Cached data/crsp_panel.npz is reused locally. To rebuild, first obtain the membership snapshot with reproduction/fetch_membership.py using your existing WRDS access, then run reproduction/prepare.py.

## Run and verify

Run from the staged project directory using the selected environment:

```bash
python -m pytest -q reproduction/test_numerics.py
python reproduction/audit_release.py
python reproduction/run.py --output local --seeds 0 1 2 --pretrain 100 --epochs 10
python reproduction/report.py --output local
```

## Outputs and reruns

`results/local/` contains method/seed outputs and REPORT.txt. The output name is a run identifier; choose a new one when changing settings.

Raw inputs, generated tables, checkpoints, environments and historical progress
reports are local artifacts and are excluded from the source release. Existing
third-party notices and licenses remain with the implementation. No new license
has been assigned to third-party work.

## Month–firm–return export

From the clean source folder, use an environment with NumPy, pandas >=2.1 and PyArrow:

```bash
python ../_replication/month_firm_returns.py --project diffusion-dfl --data-root ..
```

Compounded one-session CRSP CIZ total simple returns aligned to target_dates. CRSP PERMNO security identifier.

The generated Parquet table and provenance JSON are written under
`.replication-local/tables/` at the collection root. Daily forward predictions are
not relabeled as monthly forecasts; only same-month aligned scores are included.
