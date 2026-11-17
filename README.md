# Monthly return and volatility coforecasting

A model-family comparison rather than an exact replication of one paper. Train 1995–2004, validate 2005–2009, test 2010–2025 with annual refits. The monthly outcome is the exact next calendar month’s CRSP total simple return, not a 21-session proxy.

## Original-paper component check

The 2026-11-17 fresh NGBoost Yacht benchmark ran all 20 splits from Duan et al.
(2020), Tables 1 and 3, using the paper's validation-NLL selection rule. Local
NLL is 0.411059 versus 0.20 ± 0.26; RMSE is 0.622483 versus 0.50 ± 0.20.
Both meet a predeclared published-error-band tolerance, with worse local means;
this is approximate component agreement, not exact replication or validation
of the full CRSP study. The current author script's MSE-selection diagnostic
fails the NLL comparison (1.951417). The paper's uncertainty labels and current
author code also differ. See [the full report](NGBOOST_YACHT_REPRODUCTION.txt)
for scope, caveats, reproducible commands and all evidence.

`scripts/reproduce_ngboost_yacht.py` uses `NGB.paper_benchmark()` with depth-3
trees, learning rate 0.01 and raw target units. The monthly model defaults and
historical forecasts are unchanged.

## Prepare the runnable workspace

From this source folder:

```bash
python ../_replication/run.py prepare --project "monthly-coforecast"
cd ../.replication-local/workspace/"monthly-coforecast"
```

The launcher copies code and links existing local inputs. Raw data stay shared at
the root. Runs write to the staged workspace, keeping this source folder clean.
Use `--workspace /path/to/new-run` for an independent run; use `--fresh-inputs` when
rebuilding prepared data, and `--reuse-results` only to restore historical generated
outputs for replay. Historical README/status notes live under
`../.replication-local/artifacts/monthly-coforecast/`.

## Environment

Python 3.11 and requirements.lock.txt; cap BLAS/OpenMP threads when running multiple models.

If staging restored `.venv`, use its Python without reinstalling packages into the
archived environment. For a new installation, create an environment outside the
source folder and activate it before running the commands below.

```bash
python -m pip install -r requirements.lock.txt
```

## Inputs

Shared ../crsp-monthly/parquet and ../crsp-daily/parquet. The prepared data/panel.parquet is reused locally. To rebuild from raw data use --fresh-inputs in a separate workspace and run python scripts/prepare.py. Never put another copy of CRSP into this folder.

## Run and verify

Run from the staged project directory using the selected environment:

```bash
mkdir -p results
python scripts/verify.py
python scripts/run.py --stage tune
python scripts/run.py --stage test
python scripts/scaled_cmk.py
python scripts/auxiliary_controls.py
python scripts/evaluate.py
```

## Outputs and reruns

`results/predictions-YEAR.parquet` contains held-out forecasts; results also contains score/portfolio reports. Completed annual files are reused. New designs require a new workspace.

Raw inputs, generated tables, checkpoints, environments and historical progress
reports are local artifacts and are excluded from the source release. Existing
third-party notices and licenses remain with the implementation. No new license
has been assigned to third-party work.

## Month–firm–return export

From the clean source folder, use an environment with NumPy, pandas >=2.1 and PyArrow:

```bash
python ../_replication/month_firm_returns.py --project monthly-coforecast --data-root ..
```

Actual next calendar-month CRSP total simple return, keyed by realization month; noncontiguous or unavailable labels remain null. CRSP PERMNO security identifier.

The generated Parquet table and provenance JSON are written under
`.replication-local/tables/` at the collection root. Daily forward predictions are
not relabeled as monthly forecasts; only same-month aligned scores are included.
