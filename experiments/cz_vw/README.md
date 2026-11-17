# PRISM-VQ with CZ202 and value-weighted portfolios

Re-run the separate 202-characteristic adaptation of the released PRISM-VQ model.
It uses dated S&P 500 membership, the previous-month NYSE20 size screen, stage-one
seed 42 and stage-two seed 0. `protocol.json` fixes the training/validation/test
splits, daily input construction, execution delay, costs and FF5+UMD inference.
This is a large-cap adaptation, not an all-US-stock or exact original-paper run.

## Inputs and environment

Prepare the parent PRISM-VQ workspace as described in [its guide](../../README.md).
It must contain the checked S&P 500 base panel, shared CRSP data, the pinned
Chen–Zimmermann 2026.10 characteristics, shared daily factors and `protocol.txt`.
Prepared arrays and source metadata are linked from the local archive. Use
`--fresh-inputs` to rebuild them from existing raw sources.

Run all commands from the **prism-vq workspace root**:

```bash
uv pip install --python .venv/bin/python -r experiments/cz_vw/requirements-extra.txt
.venv/bin/python -u experiments/cz_vw/run.py
```

The coordinator builds inputs, performs preflight checks, trains both stages,
evaluates portfolios and verifies outputs. It resumes matching checkpoints.
Do not modify fingerprinted training/settings files under an existing run;
prepare a new workspace for a changed experiment.

## Outputs and checks

Outputs stay under `experiments/cz_vw/` in the workspace. Stage logs and checkpoints
record progress; final evidence is `REPORT.txt`, `analysis/verification.json`
with `passed=true`, and `.complete`. The runner does not need a systemd service.
No persistent service is installed by these instructions.

The primary spread is capitalization-weighted D10−D1, +100%/−100% exposure, with
signals at t, execution at t+1 close and CRSP payoff at t+2. The monthly input
features do not make the native ranking target a calendar-month return. Calendar
month–firm returns should be exported separately through the parent table command.
