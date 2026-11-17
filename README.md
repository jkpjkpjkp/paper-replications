# Label horizon paradox — US minute experiment

The code implements the disclosed equations with documented choices for unavailable features and settings. Original CSI minute inputs and several backbones/ablations are not available; the daily CSI results are not the original minute experiment. Intraday forward labels are not calendar-month returns.

## Prepare the runnable workspace

From this source folder:

```bash
python ../_replication/run.py prepare --project "song26l"
cd ../.replication-local/workspace/"song26l"
```

The launcher copies code and links existing local inputs. Raw data stay shared at
the root. Runs write to the staged workspace, keeping this source folder clean.
Use `--workspace /path/to/new-run` for an independent run; use `--fresh-inputs` when
rebuilding prepared data, and `--reuse-results` only to restore historical generated
outputs for replay. Historical README/status notes live under
`../.replication-local/artifacts/song26l/`.

## Environment

Use the Python environment in reproduction/.venv or install reproduction/requirements.lock into a compatible environment. The documented execution uses float64 and two CPU threads.

If staging restored `reproduction/.venv`, use its Python without reinstalling packages into the
archived environment. For a new installation, create an environment outside the
source folder and activate it before running the commands below.

## Inputs

Shared ../song26l-us/bars and support contain the audited minute inputs and mappings. They stay in place and are never copied into this paper folder. reproduction/protocol.json fixes the active US specification. The csi/ implementation is a separate daily-data adaptation.

## Run and verify

Run from the staged project directory using the selected environment:

```bash
cd reproduction
.venv/bin/python -m unittest test_reproduction test_backbones test_supervisor test_paper_protocol -v
.venv/bin/python check_hypergradient.py
.venv/bin/python check_protocol.py
cd ..
reproduction/.venv/bin/python resume.py --device cpu --backbones LSTM,GRU,TCN
```

## Outputs and reruns

`reproduction/results/` stores model fits, predictions and receipts. `reproduction/.venv/bin/python status.py` inspects the actual processes and checkpoints. Matching interrupted fits resume; mismatched source/config hashes fail.

Raw inputs, generated tables, checkpoints, environments and historical progress
reports are local artifacts and are excluded from the source release. Existing
third-party notices and licenses remain with the implementation. No new license
has been assigned to third-party work.
