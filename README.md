# Error Propagation in Dynamic Programming

Re-run an independent implementation of Algorithm 1 from Della Vecchia and
Filipović, *Error Propagation in Dynamic Programming: From Stochastic Control to
American Option Pricing*, [arXiv:2509.20239v2](https://arxiv.org/abs/2509.20239v2).
The implementation uses full kernel ridge regression and a direct Nyström solve;
it does not reproduce the authors' FALKON solver or claim identical published
numbers.

## Prepare a working directory

Run from the collection root (the directory containing `MDP Pricing/`):

```bash
python _replication/run.py prepare --project 'MDP Pricing'
cd '.replication-local/workspace/MDP Pricing'
```

The launcher copies code to the working directory and links existing local
inputs and the archived environment. Training and reports are written there,
keeping this source directory clean. Historical results and the previous README
are preserved in `.replication-local/artifacts/MDP Pricing/` at the collection
root. The default working directory starts with fresh results; use the launcher's
`--reuse-results` option only to replay the historical results.

## Environment and inputs

Use Python 3.11 or later, NumPy, SciPy, Matplotlib and threadpoolctl. The recorded
Python 3.11 environment is pinned in `requirements.lock.txt`. If the launcher did
not supply an environment, create one inside the working directory:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
uv pip install --python .venv/bin/python -e . --no-deps
```

All market paths are simulated from the parameters in `configs/`; no raw stock
data or credentials are needed. The report compares against transcribed targets
in `sources/paper_results.json`. Full source-integrity verification additionally
uses the public PDFs identified in `sources/source_manifest.json`. The launcher
reuses existing local copies. If the PDFs are absent, obtain the pinned versions:

```bash
.venv/bin/python scripts/fetch_sources.py
```

## Re-run

```bash
export OPENBLAS_NUM_THREADS=2
make test
make reproduce
```

`make test` performs fast checks of kernel regularization, coincident inputs,
Black–Scholes limits and the exercise-value bound. The historical README named
a `tests/` directory that was absent from this folder; it is not presented as an
available test suite. The complete experiment and artifact audit is separate.

The `reproduce` target executes these steps in order:

```bash
.venv/bin/python scripts/run_suite.py
.venv/bin/python scripts/run_diagnostics.py
.venv/bin/python -m scripts.run_extensions sensitivities
.venv/bin/python -m scripts.run_extensions convergence
.venv/bin/python -m scripts.run_surfaces
.venv/bin/python scripts/report.py
.venv/bin/python scripts/verify.py
```

The main suite contains two option payoffs, dimensions 2/5/10/20/40 and ten seeds
per cell. Sensitivities change the time grid and Monte Carlo innovation sharing.
The finite-state diagnostic checks error decompositions and contraction
identities. Convergence and surface experiments are additional reconstructions,
not recovered settings for every paper figure.

Completed compatible runs are reused. The runner rejects changed configuration
hashes in an existing results directory. Use a separate working directory for a
changed design; do not mix exploratory and main results.

## Outputs and checks

| Path in the working directory | Contents |
| --- | --- |
| `results/main/summary.csv` | Primary option-price comparisons and uncertainty |
| `results/main/runs/` | Per-seed fits and stage diagnostics |
| `results/ten_steps/`, `results/independent_mc/` | Sensitivity runs |
| `results/convergence/` | Sample-size sweep |
| `results/diagnostics.json` | Finite-state and Monte Carlo checks |
| `results/surfaces/` | Fitted kernels and surface arrays |
| `report.html`, `figures/` | Generated report and figures |
| `results/verification.json` | Run counts, numerical checks and source integrity |
| `results/artifact_manifest.json` | Source, configuration and result hashes |

The main grid uses nine exercise intervals, while a ten-interval sensitivity
addresses the cited benchmark's convention. Unreported sample counts and other
method choices are specified in the retained configurations. Passing the checks
establishes consistency of this implementation, not equality to all paper
results. A month–firm–return table is not applicable: these are simulated option
pricing and finite-state experiments without dated firm observations.
