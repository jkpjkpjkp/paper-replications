# Arboreal Neural Network

Independent replication of [Arboreal Neural Network,
ICML 2026](https://openreview.net/forum?id=NDM1ltvBOf): compile XGBoost trees into
ArborCells, jointly fine-tune thresholds and leaves, and export symbolic rules.
The main workflow runs eleven public tabular benchmarks. A separate monthly
U.S. equity experiment is implemented in `finance/`.

The reconstructed runs do not match the published numerical results.

A 2026-11-17 probe uses the public PyTorch Frame 80/10/10 split and a
broader validation search. It narrows the MagicTelescope gap but leaves grade 1.
See [the attempt report](.agent/RESOLUTION_ATTEMPT_20261117.txt) and
[STATUS.txt](STATUS.txt) for fresh and historical comparisons. The original
launcher defaults remain the historical protocol for reproducibility; the new
probe is a separately recorded experiment. [WRONG.txt](WRONG.txt) gives a verified
counterexample to universal soft/hard ranking equivalence.

## Prepare a working directory

From the collection root:

```bash
python _replication/run.py prepare --project arboreal
cd .replication-local/workspace/arboreal
```

The launcher stages code and links existing data and the archived environment.
It creates fresh output directories by default. Use `--reuse-results` for a
historical replay. The original README, protocols, status, data and outputs are
preserved in `.replication-local/artifacts/arboreal/` at the collection root.
Raw data are reused in place rather than copied into the source package.

## Install

Use Python 3.12 and the pinned CPU PyTorch environment. If `.venv` was not supplied
by the launcher:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match -r requirements.lock.txt
uv pip install --python .venv/bin/python --no-deps -e .
```

## Re-run the public tabular benchmarks

The data builder reuses prepared datasets under `data/`; missing public inputs
are downloaded using the source URLs in `arboreal/data.py`.

```bash
.venv/bin/python -m pytest -q tests/test_model.py tests/test_data.py
.venv/bin/python scripts/prepare_data.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python scripts/run_suite.py --threads 2 --jobs 3
.venv/bin/python scripts/audit_alignment.py
.venv/bin/python scripts/report.py
.venv/bin/python scripts/verify.py
.venv/bin/python scripts/finalize.py
```

Defaults are eleven datasets, seeds 0–4, 30 epochs and patience 8. Completed runs
are reused only under the same suite settings; changed configurations are
rejected. `--jobs 1` lowers concurrent CPU demand. For a small plumbing check:

```bash
.venv/bin/python scripts/run_suite.py --datasets MagicTelescope --seeds 0 \
  --epochs 5 --patience 3 --output artifacts/quick
.venv/bin/python scripts/report.py --root artifacts/quick
.venv/bin/python scripts/verify.py --root artifacts/quick
```

Use a separate output directory for changed settings. The quick check does not
estimate full benchmark performance.

## Re-run the monthly finance extension

This is a separate fixed-design application of the model. It reads existing
licensed `crsp-monthly/` inputs and the cached French monthly-factor ZIPs at
`finance/sources/`. The launcher links them. To use a different shared input
location, export `REPLICATION_DATA_ROOT=/path/to/shared/inputs`.

Install `finance/requirements.lock.txt` into a compatible working environment
before this extension if those packages are not already available, then run:

```bash
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
.venv/bin/python -m finance.data
.venv/bin/python -m finance.train --jobs 2
.venv/bin/python -m finance.backtest
.venv/bin/python -m finance.analyze
.venv/bin/python -m finance.verify
.venv/bin/python -m finance.report
.venv/bin/python -m pytest -q tests/test_finance.py
```

`finance/config.json` retains the original frozen specification. Runtime source
resolution uses the shared data root, while the original config bytes remain
unchanged for provenance checks. The archived `finance/PROTOCOL.md` and
`finance/protocol_initial.sha256` are required for historical protocol
verification and are restored by the launcher. If rebuilding a modified design,
create a distinct experiment and record a new protocol rather than reusing old
checkpoint or verification hashes.

## Outputs and validation

| Path in the working directory | Contents |
| --- | --- |
| `artifacts/benchmark/metrics.csv`, `summary.csv` | Per-seed and aggregate benchmark metrics |
| `artifacts/benchmark/<dataset>/seed*/` | Trees, ArborCell checkpoints, rules, histories and predictions |
| `artifacts/benchmark/verification.json` | Re-scored predictions and independently evaluated rules |
| `artifacts/benchmark/REPORT.md` | Generated benchmark report |
| `finance/results/predictions.parquet` | Monthly firm-level model forecasts |
| `finance/results/holdings.parquet`, `returns.parquet` | Positions and portfolio returns |
| `finance/results/verification.json` | Timing, accounting, exposure and covariance checks |
| `finance/REPORT.md` | Generated finance report |

Only `model.f` and `model.v` are trainable. Hard routing is checked against
XGBoost; finite-temperature soft predictions need not match it. Finance returns
use +100% long and −100% short exposure per unit of NAV. Monthly firm returns are
applicable to the finance extension; tabular benchmark rows without dates and
security IDs must not be relabeled as month–firm observations. Generated data
and tables stay in the working area, outside this source folder.

The implementation is MIT licensed. Third-party source and datasets retain their
own licenses and attribution.

## Month–firm–return export

From the clean source folder, use an environment with NumPy, pandas >=2.1 and PyArrow:

```bash
python ../_replication/month_firm_returns.py --project arboreal --data-root ..
```

Monthly CRSP total return joined to held-out ArbNN ensemble scores. CRSP PERMNO security identifier.

The generated Parquet table and provenance JSON are written under
`.replication-local/tables/` at the collection root. Daily forward predictions are
not relabeled as monthly forecasts; only same-month aligned scores are included.
