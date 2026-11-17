# Sparse portfolio optimization with transaction costs

Re-run the independent STCO/PSGA implementation of Yao and Zhang, *A Linearly
Convergent Proximal Subgradient Algorithm for Sparse Portfolio Optimization with
Transaction Cost*, [ICML 2026](https://openreview.net/forum?id=yZAo4TPqhE).
The [author source](https://github.com/Ting221/STCO) is pinned to
`a2e04730cbdfb6ccb0017456d4f835a4d7446ec3`.

## Current numerical assessment

As of 2026-11-17, grade 2 is scoped to a fresh BCRP MSCI baseline experiment:
all three Table 5 transaction-rate cells match two-decimal rounding. The main
STCO/PSGA reconstruction still misses all 24 declared Table 5 wealth cells
within 1%, and nine BCRP cells on other datasets miss rounding. See
[STATUS.txt](STATUS.txt) and [BENCHMARK_REPRODUCTION.txt](BENCHMARK_REPRODUCTION.txt)
for the complete comparison, independent solver checks and remaining gaps.
The historical main-method report remains archived without alteration.

## Prepare and install

From the collection root:

```bash
python _replication/run.py prepare --project proximal-portfolio
cd .replication-local/workspace/proximal-portfolio
```

The launcher stages source and links existing local inputs/environment. Outputs
are generated in this working directory. Original status, results and the old
README are preserved at `.replication-local/artifacts/proximal-portfolio/` in the
collection. Use the launcher's `--reuse-results` for historical result replay.

The lock file records Python 3.14 on Linux. If no `.venv` was supplied:

```bash
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
uv pip install --python .venv/bin/python --no-deps -e .
```

For another supported Python/platform (Python >=3.11), resolve compatible
packages from `pyproject.toml` with `uv pip install --python .venv/bin/python
-e '.[test]'`. Record those versions because they differ from the original lock.
MATLAB and Octave are not required.

## Inputs

The four author matrices are reused at `upstream/STCO-code/`:

| File | Shape: sessions × assets |
| --- | --- |
| `nyse-o.mat` | 5,651 × 36 |
| `nyse-n.mat` | 6,431 × 23 |
| `tse.mat` | 1,259 × 88 |
| `msci.mat` | 1,043 × 24 |

The launcher links the archived matrices without copying them. For a fresh
installation, obtain them from the pinned author source. The report also needs
`sources/paper.pdf`, its extracted `paper.txt`, and the transcribed
`paper_targets.csv`; these existing files are restored in the working directory.
`paper_risk_targets.csv` is regenerated from the paper text. Training never uses
paper comparison values to fit the portfolio.

## Re-run in order

```bash
mkdir -p results
.venv/bin/python -m pytest -q > results/tests.log
cat results/tests.log
.venv/bin/python scripts/reproduce.py --sweep --windows 4 5
.venv/bin/python scripts/validate.py
.venv/bin/python scripts/sensitivity.py
.venv/bin/python scripts/plot_results.py
.venv/bin/python scripts/extract_risk_targets.py
.venv/bin/python scripts/build_report.py
.venv/bin/python scripts/register_evidence.py
```

The test log is an input to `build_report.py`, so retain the redirection above.
Omit `--sweep` for only the three table transaction rates. `--datasets MSCI`
runs one complete dataset. `--direct` disables the event-skipping optimization
and can be much slower. Choose a distinct `--output` for a changed experiment;
the downstream report commands use the standard main suite location.

## Model and accounting conventions

The paper parameters are `K=10`, `c=8e-6`, `alpha=.0025`, `rho=.618`,
`lambda=10*gamma`, relative-step tolerance `1e-8`, and iteration cap `1e8`.
Each daily optimization starts from uniform weights and uses only earlier
observations. STCO2 uses window 4 as the main reconstructed choice and reports
window 5 separately. Stable top-K ties favor the lower asset index.

The main solver uses an exact scalar dual breakpoint solution to the convex
proximal subproblem. Event skipping accelerates repeated fixed-step iterates;
tests compare it with direct iteration, ADMM and a conic oracle. The author
Algorithm 2's ambiguous anchor is not silently substituted for the fixed outer
anchor used by the mathematical reference.

Wealth follows `theta + gamma*||h-theta*w||_1 = 1` and grows by
`theta*(w @ price_relative)`. Initial purchase costs are included; there is no
terminal liquidation. Paper-specific return conventions are retained in
`paper_*` metrics, while net Sharpe and drawdown use `gross*theta-1`.

## Outputs and checks

| Output | Contents |
| --- | --- |
| `results/main/metrics.csv` | Wealth, risk, sparsity, feasibility and solver diagnostics |
| `results/main/paper_comparison.csv` | Explicit computed-versus-published comparisons |
| `results/main/runs/*.npz` | Daily weights, wealth, gross returns and transaction fractions |
| `results/validation/` | Direct-iterate and small-problem oracle checks |
| `results/sensitivity/` | K and penalty sensitivity |
| `figures/`, `REPORT.txt` | Generated figures and report |

UBAH, BEST and BCRP are implemented references; the other published competitor
numbers are comparison targets, not computed methods. Convergence is the
relative-step condition, not proof of global optimality. These benchmark
matrices do not provide reliable calendar dates and firm identifiers, so a
month–firm–return table is not applicable without an external verified mapping.
