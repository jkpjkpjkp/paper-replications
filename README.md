# Stable Alpha / CASH — author-code checks

The synthetic check is not a reproduced paper score. Original feature formulas, frequency, data and full experimental mapping remain required. A monthly firm table cannot be constructed from synthetic checks.

## Prepare the runnable workspace

From this source folder:

```bash
python ../_replication/run.py prepare --project "stable-alpha"
cd ../.replication-local/workspace/"stable-alpha"
```

The launcher copies code and links existing local inputs. Raw data stay shared at
the root. Runs write to the staged workspace, keeping this source folder clean.
Use `--workspace /path/to/new-run` for an independent run; use `--fresh-inputs` when
rebuilding prepared data, and `--reuse-results` only to restore historical generated
outputs for replay. Historical README/status notes live under
`../.replication-local/artifacts/stable-alpha/`.

## Environment

Python 3.12 with PyTorch, NumPy and pandas. The local engineering check used PyTorch 2.5.1; preserve the source revision pinned in scripts/verify_author_pipeline.py.

If staging restored `.venv`, use its Python without reinstalling packages into the
archived environment. For a new installation, create an environment outside the
source folder and activate it before running the commands below.

## Inputs

The engineering check creates a tiny synthetic fixture. Empirical execution requires caller-supplied features, targets, raw realized returns and timestamps, with a balanced asset axis; see upstream/config.py for the configuration contract. No original empirical dataset is invented.

## Run and verify

Run from the staged project directory using the selected environment:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 FORCE_CPU=1 python scripts/verify_author_pipeline.py --output results/engineering-repeat
```

## Outputs and reruns

`results/engineering-repeat/` contains the synthetic software/gradient/timing verification. Use an output directory that does not already exist. For real inputs run python upstream/run_cash_pipeline.py --config /path/to/config.json --dry-run, then the same command without --dry-run.

Raw inputs, generated tables, checkpoints, environments and historical progress
reports are local artifacts and are excluded from the source release. Existing
third-party notices and licenses remain with the implementation. No new license
has been assigned to third-party work.
