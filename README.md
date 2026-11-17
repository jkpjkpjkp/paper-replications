# Expanding the Chaos

Re-run the independent SDENO/F-SPDENO implementation for Brownian reconstruction,
OU and Heston dynamics, and the stochastic one-dimensional Φ⁴ equation from
[Shi et al., ICML 2026](https://proceedings.mlr.press/v306/shi26i.html). The inspected
author release did not contain experiment code. This folder implements the
published method with explicit reconstruction choices; successful execution does
not establish agreement with the published benchmark accuracy.

## Prepare and install

From the collection root:

```bash
python _replication/run.py prepare --project PDE
cd .replication-local/workspace/PDE
```

The launcher stages code and links available archived inputs/environment. It
keeps generated results out of this source folder. Original notes, logs and
artifacts are preserved at `.replication-local/artifacts/PDE/` in the collection.
The default working directory has fresh results; `--reuse-results` is for replay.

The recorded runtime uses Python 3.11 and PyTorch 2.4.1. When no existing `.venv`
is supplied, install the pinned dependencies in the working directory:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
```

Data are simulated by `chaos/data.py` using `configs/`. No raw equity panel is
needed. The CPU commands work without a GPU. CUDA speeds up the SPDE runs.

## Re-run the finance experiments

```bash
.venv/bin/python scripts/run_finance.py --windows 50 60 70 75 80 --seeds 0 1 2 --device cpu
.venv/bin/python scripts/finance_report.py
```

This runs five Heston training windows for three seeds plus full-horizon
supervised controls. The latter use observations throughout the time interval
and remain separate from extrapolation results.

## Re-run the other included experiments

```bash
.venv/bin/python scripts/reproduce.py --experiments brownian ou heston --seeds 0 1 2 --device cpu
.venv/bin/python scripts/reproduce.py --experiments ou --seeds 0 1 2 --sde-full-horizon --device cpu
.venv/bin/python scripts/reproduce.py --experiments phi4 --seeds 0 1 2 --device cuda
.venv/bin/python scripts/reproduce.py --experiments phi4 --phi-order 1 --seeds 0 --device cuda
.venv/bin/python scripts/report.py
.venv/bin/python scripts/verify.py
```

Replace `cuda` with `cpu` if needed. Completed runs with `metrics.json` are
skipped; `--force` replaces them. Interrupted runs restart training rather than
resume automatically. `--epochs` allows a short engineering check, but writes
into the usual experiment directory: use a separate working directory for such
checks. The old README referred to a test suite that is not present in this
folder; use the saved-prediction and metric audit in `scripts/verify.py`.

## Inference and outputs

```bash
.venv/bin/python scripts/predict.py results/ou_seed0/model.pt new_noise.npz prediction.npz
```

OU input contains `dw` with shape `(batch,100)`. Heston uses independent Brownian
increments of shape `(batch,100,2)` before correlation. Φ⁴ input contains `dw`
with shape `(batch,50,128)`, `u0` with shape `(batch,128)`, and `qstd` with shape
`(128,)`. SDE predictions have shape `(batch,101,state_dim)` and SPDE predictions
have shape `(batch,51,128)`, including the initial state.

| Output | Contents |
| --- | --- |
| `results/<experiment>/metrics.json` | Settings, selected validation epoch, errors and history |
| `results/<experiment>/model.pt` | Trained model |
| `results/<experiment>/predictions.npz` | Saved predictions for independent replay |
| `report/finance.html` | Heston training-window and forecasting-control report |
| `report/index.html`, `report/*summary.*` | Other reports and seed summaries |
| `verification.json` | Recomputed metrics and checkpoint/prediction consistency |

The solution-operator experiment supplies future noise. Setting future
increments to zero changes it to an unknown-future-noise forecasting control.
The linearized Φ⁴ solver uses known coefficients and full noise; it is a
numerical reference, not a trained competitor. Navier–Stokes, fMRI, image
sampling and the remaining paper applications are not implemented here.
Month–firm–return tables do not apply to these simulated paths.

A fresh three-seed first-chaos capacity diagnostic is available through
`scripts/diagnose_phi4.py`; it saves fitted coefficients and held-out predictions
under `.agent/diagnostics/`. It demonstrates that the local simulated fixed-IC
problem can be learned accurately by a linear Fourier map. It is a diagnostic,
not the published F-SPDENO or a paper baseline, and does not change the failed
replication assessment. See `.agent/PHI4_CAPACITY_AUDIT_20261117.txt`.
