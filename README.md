# Distributionally robust portfolio optimization with Bayesian ambiguity sets

Re-run the portfolio application in Section 4.2 and Appendix E.2 of
[arXiv:2411.16829v2](https://arxiv.org/abs/2411.16829v2). This implementation
compares BASPE, BASPP and BDRO over the original DowJones weekly dataset. It uses
analytic gradients and SLSQP with independent conic validation, rather than the
paper's MOSEK timing setup. It does not claim bitwise agreement with author runs.

## Prepare a working directory

From the collection root:

```bash
python _replication/run.py prepare --project DRO
cd .replication-local/workspace/DRO
```

The launcher copies source and links existing data and the archived environment.
Historical outputs and the previous README are preserved in
`.replication-local/artifacts/Bayesian/DRO/` at the collection root. Generated
outputs remain in the working directory. Use `--reuse-results` when preparing a
historical replay, instead of a fresh output directory.

## Environment and inputs

The recorded environment uses Python 3.14.8. `requirements.lock.txt` records its
exact packages; `requirements.txt` supports resolving versions on another
Python >=3.11 installation. If no `.venv` was supplied:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

No commercial solver or upstream package installation is required. The optional
conic checks use CVXPY, Clarabel and ECOS from the requirements.

The runner reads existing Bruni et al. supplementary data under `data/`:

- one `DowJones.mat` containing `Assets_Returns` (1,363 × 28) and
  `Index_Returns` (1,363 values);
- `OutofSamplePortReturns_MeanVar_DowJones_List.txt`, the supplied baseline;
- `authors_reference.json`, transcribed saved author outputs for comparison.

The launcher links these existing inputs, without copying the raw archive. For
a new installation, obtain `mmc2.zip` from the
[original supplementary material](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4959918/supplementaryFiles)
and extract the DowJones dataset and baseline under `data/`. The previous README
referred to `fetch_data.py`, which is not present; no missing downloader is
advertised as a command here. The original author notebook is pinned at commit
`448ec4897ff588c8bed318afa13b8858d7e383fa` in
[PatrickOHara/mis-dro-code](https://github.com/PatrickOHara/mis-dro-code).

## Re-run and validate

```bash
.venv/bin/python run_portfolio.py --workers 4
.venv/bin/python validate_results.py
.venv/bin/python plot_results.py results/full
```

The default experiment uses 109 rolling windows, 16 radii and three methods.
For window `j`, training rows are `12*j : 12*j+52` and testing rows are
`12*j+52 : 12*j+64`, with zero-based indexing. It produces 5,232 optimized
portfolios and 1,308 out-of-sample weeks. Seeds are the window index and samples
are reused across radii within each method/window. Every radius is reported.

For a smaller computation with a separate output directory:

```bash
.venv/bin/python run_portfolio.py --windows 2 --workers 2 --output results/check
.venv/bin/python validate_results.py --output results/check
```

The runner checkpoints completed windows and rejects a changed code/data/config
fingerprint in the same output directory. Choose a new `--output` after changing
settings. There is no local `tests/` suite in this folder despite its historical
README reference; `validate_results.py` performs the portfolio audit and conic
cross-checks. `--skip-conic` omits the latter and should be recorded as reduced
validation.

## Outputs and interpretation

| Output | Contents |
| --- | --- |
| `results/full/windows/` | Window settings, weights, returns, timings and optimality gaps |
| `results/full/weekly_returns.csv` | Recomputed weekly returns |
| `results/full/weights.csv`, `diagnostics.csv` | Positions and solver diagnostics |
| `results/full/summary.csv` | Mean, variance and summed weekly returns |
| `results/full/authors*_comparison.csv` | Numerical comparisons with the source notebook |
| `results/full/validation.json`, `conic_validation.csv` | Feasibility, accounting and solver checks |
| `results/full/figure*.png` | Generated comparison figures |

The optimizer enforces the simplex and a convex first-order gap of at most
`1e-7`; the audit recomputes returns and compares independently solved conic
objectives. The paper's cumulative-return figures use a sum of decimal weekly
returns, not compounded wealth. BASPE's grid is the radius above its offset;
BASPP follows the finite empirical posterior predictive used in the released
code. Original random draws and solver behavior are not fully recoverable.

The anonymous weekly array does not include a verified calendar/firm mapping.
It is therefore not converted into a calendar-month firm-return table.
