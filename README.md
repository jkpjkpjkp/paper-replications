# Online conformal prediction — reproduction

Reproduces the implemented online conformal experiments. Coverage and interval-width outputs are not firm returns; no month–firm–return table is defined for these prediction intervals.

## Prepare the runnable workspace

From this source folder:

```bash
python ../_replication/run.py prepare --project "liu26y"
cd ../.replication-local/workspace/"liu26y"
```

The launcher copies code and links existing local inputs. Raw data stay shared at
the root. Runs write to the staged workspace, keeping this source folder clean.
Use `--workspace /path/to/new-run` for an independent run; use `--fresh-inputs` when
rebuilding prepared data, and `--reuse-results` only to restore historical generated
outputs for replay. Historical README/status notes live under
`../.replication-local/artifacts/liu26y/`.

## Environment

Python 3.12.13; bootstrap.sh installs requirements.lock.txt and verifies the pinned author repository and PID baseline.

If staging restored `.venv`, use its Python without reinstalling packages into the
archived environment. For a new installation, create an environment outside the
source folder and activate it before running the commands below.

The bootstrap command below creates and installs `.venv` when it is absent.
It also synchronizes an existing environment, so skip it when using a working
archived environment supplied by staging.

## Inputs

The released upstream financial-score datasets are inputs. Existing copies under data/, upstream/ and reference/ are linked by staging. On a new machine bootstrap.sh downloads pinned code and input artifacts. config.json declares the experiment.

## Run and verify

Run from the staged project directory using the selected environment:

```bash
test -x .venv/bin/python || bash bootstrap.sh
bash run.sh
```

## Outputs and reruns

`results/`, `logs/`, and generated figures. run.sh prepares data, runs tests/test_ocp.py, reproduces, plots, and validates in that order.

Raw inputs, generated tables, checkpoints, environments and historical progress
reports are local artifacts and are excluded from the source release. Existing
third-party notices and licenses remain with the implementation. No new license
has been assigned to third-party work.
