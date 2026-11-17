# A Set-Sequence Model for Time Series: finance audit

Re-run the implemented target extraction, experimental run plan and provisional
CRSP cohort audit for [arXiv:2505.11243v2](https://arxiv.org/abs/2505.11243v2).
This folder does not contain a Set-Sequence trainer or trained model. Its
published targets and audit outputs must not be described as reproduced scores.

## Prepare and install

From the collection root:

```bash
python _replication/run.py prepare --project 'Set Sequence'
cd '.replication-local/workspace/Set Sequence'
```

Existing inputs are linked, not copied. Generated audits go into the working
directory. The previous README, status and reports are preserved at
`.replication-local/artifacts/Set Sequence/` in the collection.

Use Python 3.12. If no `.venv` was supplied by the launcher:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
```

## Inputs

`configs/paper_protocol.json` and `configs/equity_features.json` record the
reconstructed specification. The target extraction and run-plan scripts do not
need external data. The cohort audit reads shared inputs at the parent of this
working directory:

- `crsp-daily/parquet/1994.parquet` through `2021.parquet`;
- `factors-daily/ff3_daily.parquet`.

Use `--data-root /path/to/shared/inputs` to select another existing input root.
Do not copy the CRSP archives into this paper folder. The original CoreLogic
mortgage panel and the exact author equity preprocessing are not supplied.

## Re-run the implemented work

```bash
.venv/bin/python scripts/build_targets.py
.venv/bin/python scripts/build_run_plan.py
.venv/bin/python scripts/audit_local_data.py --data-root .. --out results/data_audit
```

The cohort audit reads input Parquet files without changing them. Its annual
cohorts use stated assumptions and compare completeness filters; they are not
verified author cohorts. It does not train a model. There is no local `tests/`
directory despite the historical README reference to one.

| Output | Meaning |
| --- | --- |
| `targets/published_values.csv` | Transcribed paper targets; reproduced cells remain blank |
| `targets/inventory.json` | Target coverage and unavailable curve series |
| `configs/equity_required_fits.csv` | Proposed fit grid, not completed fits |
| `configs/run_plan_summary.json` | Scope and estimated compute |
| `results/data_audit/` | Candidate cohorts and input audit |

Review the audit's flags and source hashes before using a cohort. The helper
`portfolio_metrics` in `scripts/metrics.py` requires explicit annualization,
initial holdings and short-fraction conventions; it is not a complete trainer
or evaluator. No month–firm–return result table is claimed for this audit-only
folder. A trained, dated firm-level prediction output is not available.
