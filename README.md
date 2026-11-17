# AlphaQCM replication

Run the corrected release-based IQN experiment for [AlphaQCM: Alpha Discovery in Finance with Distributional Reinforcement Learning](https://proceedings.mlr.press/v267/zhu25ag.html). The local adapter fixes identified implementation defects and evaluates on a shared community CSI300 snapshot. It does not establish a complete multi-seed or multi-market replication of the paper.

## Prepare the run directory

From this source folder:

```bash
python ../_replication/run.py prepare --project alphaqcm
cd ../.replication-local/workspace/alphaqcm
mkdir -p sources results
```

The launcher copies code into the printed workspace and links existing inputs and the archived environment. Historical outputs, source evidence, caches and the old status README are preserved at `.replication-local/artifacts/alphaqcm/` under the collection root. New results are written in the run workspace. Use `--reuse-results` during preparation only for historical replay.

## Install and supply inputs

Use Python 3.11. If no working environment was linked, create one:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python .venv/bin/python -r requirements.txt
```

For CPU execution, install a compatible CPU PyTorch build and set `device` to `cpu` in a copy of the configuration. The direct package requirements are in `requirements.txt`; `sources/requirements-resolved.lock.txt` records the historical effective environment. The local archived environment used shared packages, so a standalone installation is preferable when moving to another machine.

| Required input | Path |
| --- | --- |
| Shared Qlib CSI data | `../csi/qlib/cn_data` |
| Snapshot receipt used for hash verification | Existing files under `../csi/` |
| Author source and Git provenance | `upstream/`, commit `dffc695f9ea54e6a54e4dac86ee357be314fead4` |
| Paper snapshot for run provenance | `sources/paper.pdf`, restored from the local archive or obtained from the paper link |

The JSON `provider` setting is resolved relative to this experiment directory, so the default reuses the shared data. The community snapshot is revision `ee0fa6b2299cb5e927d2df491888c84f6c9c027a` of `joshuaxql/qlib_data`, with archive SHA-256 `50d6fbb35634eb858f0e665e4811cbcbdcaccfba1d1e0084e8b7227deae17757`. It differs from the authors' Baostock data vintage.

## Run in order

```bash
.venv/bin/python -m pytest test_replication.py -q
.venv/bin/python verify_data.py
.venv/bin/python run.py --config configs/pilot-csi300.json --output results/new-pilot
.venv/bin/python evaluate.py --run results/new-pilot --output results/new-evaluation
```

The pilot configuration currently uses 50,000 interaction steps. For the paper's P=10 step budget, use a separate result directory:

```bash
.venv/bin/python run.py --config configs/paper-csi300-p10.json --output results/new-paper-budget
.venv/bin/python evaluate.py --run results/new-paper-budget --output results/new-paper-evaluation
```

The P=20/50/100 configuration files provide the other declared capacity/budget combinations. `--seed` and `--steps` override the JSON settings. Output directories must not already exist. Saved network weights do not include replay-buffer, optimizer or RNG state, so they are not complete training-resume checkpoints.

To test the **paper's stated method**, use the separate protocol below. It implements the incremental reward and unpenalized pool fit in Algorithm 1, the Table B.5 action set, two hidden layers per head, and the losses and replay priorities in Appendix E. Read [PAPER_PROTOCOL.txt](.agent/PAPER_PROTOCOL.txt) for the remaining data substitution, unspecified conventions, and nominal split dates whose forward labels overlap split boundaries. A single fixed pool and seed is not the paper's validation-tuned ten-seed comparison.

```bash
.venv/bin/python -m pytest test_replication.py test_paper_protocol.py -q
.venv/bin/python run.py --config configs/paper-method-csi300-p10.json --output results/new-paper-method
.venv/bin/python evaluate.py --run results/new-paper-method --output results/new-paper-method-evaluation
```

The archived environment's `reused-runtime.pth` may still refer to its old location. In the current collection, prefix these commands with `PYTHONPATH=/home/jkp/data/.replication-local/artifacts/prism-vq/.venv/lib/python3.11/site-packages` to reuse the relocated runtime without modifying the archive, or install the standalone environment above.

## Outputs and interpretation

Each training directory contains `manifest.json`, histories/progress, `selected_pool.json`, `final_pool.json`, model weights and `results.json`. Validation selects the reported pool; the final-step pool is a separate diagnostic. The evaluation directory contains `predictions.npz`, `daily_metrics.csv` and `metrics.json` and independently reloads formulas with a restricted parser.

New runs also snapshot their source and configuration and record PyTorch peak allocated/reserved GPU memory in `results.json`. These allocator values exclude CUDA context memory and other processes.

Training covers 2010–2019, validation 2020 and test 2021–2022. The corrected-release configurations purge training and validation endpoints to prevent their twenty-trading-day labels crossing the split; the explicit paper-method configuration retains the paper's nominal endpoints. Primary IC uses finite paired observations; released zero-fill metrics are reported separately. The label is `close[t+20]/close[t]-1`, not a calendar-month return, and this native experiment supplies no execution backtest. Do not convert its overlapping labels into monthly portfolio returns by relabeling dates.

The original MIT license is retained in `upstream/`. Scientific deviations and the historical measured results remain in the archived README rather than this rerun guide.

## Export a month–firm–return table

This exports calendar-month price returns from the shared CSI300 Qlib inputs. It does not claim a monthly AlphaQCM forecast or portfolio backtest; the native twenty-trading-day scores are omitted. Run from the **source folder**, using the separate shared table environment:

```bash
python -m venv ../.replication-local/table-env
../.replication-local/table-env/bin/python -m pip install -r ../_replication/requirements-tables.txt
../.replication-local/table-env/bin/python ../_replication/month_firm_returns.py --project alphaqcm --data-root ..
```

The environment can be reused for every paper. Existing inputs are read in place from the collection or its local archive; no raw data are copied. Output is `../.replication-local/tables/alphaqcm/csi300-price-inputs/month_firm_return.parquet`, with `provenance.json` recording input hashes, source timing, coverage and validation. `month` is the realization month (`YYYY-MM`), `firm` retains the source security identifier, `return` is a simple decimal return, and `return_observed` distinguishes missing outcomes. A `prediction_score` is included only for a matching monthly prediction. Detailed definitions are in `../_replication/table_specs.json`.
