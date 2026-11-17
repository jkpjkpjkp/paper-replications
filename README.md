# AlphaSAGE, GP and XGBoost replication

Rerun the CSI300 experiment based on [AlphaSAGE](https://arxiv.org/html/2509.25055v2), including the adaptive AlphaSAGE combination and the released GP/XGBoost comparators. The corrected implementation uses substitute community data and explicit integration repairs. It does not establish an exact numerical or full-paper replication. The historical seed-0 failure remains in `STATUS.txt` and the archive; a smoke run is not a replacement paper comparison.

## Prepare a run workspace

From this source folder:

```bash
python ../_replication/run.py prepare --project alphasage --workspace ../.replication-local/runs/alphasage-corrected-new --fresh-inputs
cd ../.replication-local/runs/alphasage-corrected-new/alphasage
mkdir -p sources results runtime
```

The launcher copies code and links the existing shared inputs/environment. Use a new workspace name for each experiment. `--fresh-inputs` prevents writes through historical cache links. Previous runs, caches, status notes and the old README are kept under `.replication-local/artifacts/alphasage/` at the collection root. New outputs stay in the workspace. Replaying old raw-data outputs requires the archived implementation and its original protocol, not the corrected loader.

## Environment and shared input

Use Python 3.11. If the launcher did not link a working environment:

```bash
uv venv --python 3.11 .venv
uv pip sync --python .venv/bin/python --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match sources/requirements.lock.txt
```

The shared Qlib provider is `../csi/qlib/cn_data`; no raw data are copied into this paper folder. `reproduction/common.py` reads this relative path. The snapshot is revision `ee0fa6b2299cb5e927d2df491888c84f6c9c027a` of `joshuaxql/qlib_data`, archive SHA-256 `50d6fbb35634eb858f0e665e4811cbcbdcaccfba1d1e0084e8b7227deae17757`. It is not the authors' exact historical data.

The snapshot contains **raw prices**, despite the earlier loader's assumption. Adapter version 3 multiplies OHLC/VWAP by the stored cumulative factor and converts volume from 100-share lots to adjusted shares (`volume * 100 / factor`). It retains the snapshot's factor anchor without rebasing to a future endpoint. Observed prices with missing or invalid factors stop the run. Calendar, membership and consumed feature files are hashed into the cache identity and each run's `data-provenance.json`.

Feature eligibility uses historical membership and the currently observed close. Forward-return endpoints use full adjusted close histories, so an index exit does not erase an otherwise observed return. Missing outcomes remain missing. Runtime expression guards prevent nonmembers from entering nested operations or normalization; cross-sectional ranks use average ties. GFN consistency loss averages over eligible members. These are documented local corrections, not a claim that the original experiment used these exact semantics.

The source in `upstream/` is pinned to `517467a34909512a92d6e3139df54891957560de` from [BerkinChen/AlphaSAGE](https://github.com/BerkinChen/AlphaSAGE). The wrapper records that provenance and source hashes even when Git metadata is absent. The pinned files are unchanged. Training patches and runtime copies are saved within each experiment directory; `reproduction/integration.py` contains the local expression and pool corrections.

## Verify and run the corrected comparison

Run the regression checks before starting the full experiments:

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MLFLOW_DISABLE_AGENT_HINT=1
.venv/bin/python reproduction/verify_data.py > results/data-verification.log 2>&1
.venv/bin/python reproduction/verify.py > results/verification.log 2>&1
.venv/bin/python reproduction/verify_baselines.py > results/baseline-verification.log 2>&1
.venv/bin/python reproduction/run_corrected.py --seeds 0 1 2 --jobs 2
```

The runner freezes source hashes and budgets before training, runs each AlphaSAGE seed for 10,000 episodes, runs seed-0 GP and both XGBoost round variants, and writes `results/corrected-pipeline.json`, `results/comparison.csv` and `results/seed-summary.json`. `--jobs 2` bounds concurrent processes to two, each using two CPU threads. Seed SD is sample standard deviation, not a confidence interval. Existing partial and completed output directories are rejected.

For an individual seed-0 comparison, the equivalent manual commands are:

The report reads the standard directory names below, so use these names in a fresh workspace:

```bash
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MLFLOW_DISABLE_AGENT_HINT=1
.venv/bin/python reproduction/verify.py
.venv/bin/python reproduction/train.py --episodes 10000 --seed 0 --capacity 50 --update-freq 64 --log-freq 500 --output results/csi300-seed0
.venv/bin/python reproduction/evaluate.py --pool results/csi300-seed0/pool.json --output results/csi300-seed0/adaptive
.venv/bin/python reproduction/train_gp.py --seed 0 --population 1000 --generations 40 --output results/gp-csi300-seed0
.venv/bin/python reproduction/train_xgboost.py --seed 0 --rounds 100 --output results/xgboost-csi300-seed0
.venv/bin/python reproduction/verify_baselines.py
.venv/bin/python reproduction/report.py
```

Use fresh output directories for manual variants; the standard report will not automatically collect arbitrary names. It rejects missing, incomplete, mixed-date or reduced-budget summaries. The report's main table is the seed-0 comparison; the multi-seed runner writes seed aggregation separately. `run_remaining.py` is the older seed-0 continuation helper; use `run_corrected.py` for the corrected batch.

## Outputs and validation

AlphaSAGE writes policy weights, formulas, training progress and adaptive predictions under `results/csi300-seed0/`. GP writes its selected expression and predictions under `results/gp-csi300-seed0/`. XGBoost saves both the intended 100-round model and the notebook-equivalent first ten rounds under `results/xgboost-csi300-seed0/`.

`results/comparison.csv`, `results/report.json` and `results/RUN_REPORT.txt` summarize the completed runs. Reporting does not overwrite this README. Verification covers adjustment identities, full returns after exits, future-column invariance, cache invalidation, protected outputs, paired masks, tied ranks, causal OLS updates, target offsets and baseline construction. Historical audit scripts describe the old artifacts and must not be used to certify the corrected pipeline without adapting their assumptions.

## Timing and return tables

Training defaults to 2020-11-30 to purge forward targets, validation is 2021 and test is 2022–2024. All training drivers and the batch runner accept `--train-end 2020-12-31` to reproduce the nominal paper boundary explicitly; its final labels extend into January 2021. Use that option consistently for all methods in a separate workspace. Do not select between the two modes using test performance. Adaptive regression uses only labels available before each prediction. The native label spans twenty trading days.

Return/Sharpe/drawdown proxies are disabled by default. The optional `--legacy-return-proxy` flag on individual evaluation/baseline commands preserves the old diagnostic only: it uses overlapping forward outcomes and future-availability filtering and is not an executable backtest. A funded execution ledger remains separate unfinished work.

A calendar-month table must use actual realization-month returns. The separately retained CZ202 adaptation has monthly model scores and can be exported with the collection's month–firm–return tooling; its protocol is distinct from this CSI300 run. No raw price or feature panel is bundled with the source release.

## Export a month–firm–return table

This exports the separately saved monthly CZ202 adaptation and its model scores, not the native twenty-trading-day CSI label. Run from the **source folder**, using the separate shared table environment:

```bash
python -m venv ../.replication-local/table-env
../.replication-local/table-env/bin/python -m pip install -r ../_replication/requirements-tables.txt
../.replication-local/table-env/bin/python ../_replication/month_firm_returns.py --project alphasage --data-root ..
```

The environment can be reused for every paper. Existing inputs are read in place from the collection or its local archive; no raw data are copied. Output is `../.replication-local/tables/alphasage/cz-monthly/month_firm_return.parquet`, with `provenance.json` recording input hashes, source timing, coverage and validation. `month` is the realization month (`YYYY-MM`), `firm` retains the source security identifier, `return` is a simple decimal return, and `return_observed` distinguishes missing outcomes. A `prediction_score` is included only for a matching monthly prediction. Detailed definitions are in `../_replication/table_specs.json`.
