# Risk-Aware Utility Re-Ranking for Financial Asset Recommendation

Complete execution of the released recommendation algorithm and the separate literal-paper-equation implementation across 16 slices and 29,090 users. The released algorithm matches its 24 reported reference cells, while the paper equations differ. All 368 method-slice evaluations are retained as distinct scopes.

## Setup

From the collection root, stage the code and link the existing inputs:

```bash
python _replication/run.py prepare --project 'risk-aware-utility-reranking'
cd '.replication-local/workspace/risk-aware-utility-reranking'
```

The command prints the run directory. Run everything below there. Source folders stay free of data and generated artifacts. Inputs are linked from shared storage and `.replication-local/artifacts/paper-reproduction-audit/workspaces/388-risk-aware-utility-reranking`; preparation does not duplicate them. A clean checkout still requires these external inputs. Keep their release filenames and directory layout when supplying them.

Use Python 3.11 and a compatible existing environment, or create an isolated environment **inside the run directory**:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
```

Skip environment creation and installation when using a compatible environment already provided by preparation. The requirements record the retained execution environment where known; upstream dependency files remain with the corresponding source. Dependency installation was not repeated during this cleanup.

## Inputs

The retained `release/data` FARTrans artifact: price, transaction, profile, feature, candidate and split parquet/JSONL files, plus prepared `inputs` if needed. The expanded input tree is about 16.7 GB; preparation links it in place. Code is under `upstream/`; do not duplicate the raw archive or extracted data.

Inputs and historical outcomes are not redistributed by this source folder. Preserve the retained upstream license files and the separate terms for datasets and model weights.

## Rerun

Execute the stages sequentially:

```bash
python run_paper_equations.py
python run_release_artifacts.py --probe
python run_release_artifacts.py
python native_upstream.py
python verify_outputs.py
```

Use a new `--workspace` when you need a completely independent run. To inspect saved outcomes, prepare a separate workspace with `--reuse-results`; this restores historical output copies and may trigger the runners' resume/skip behavior. Use the default fresh results for a new experiment. Acquisition/preparation helpers are included, but the commands above reuse the retained inputs. If rebuilding inputs, prepare a new workspace with `--fresh-inputs` so generated files cannot overwrite linked archives.

## Outputs and validation

`results/paper-equations/`, `results/released-algorithm/` and `results/native-upstream/`; numerical tables, independent validations and figures.

The probe checks parity before the full release algorithm. `verify_outputs.py` independently rebuilds labels, allocations and aggregate metrics from source inputs and cross-checks both scopes. Native ROI is not a calendar-time self-financing NAV return. Upstream code lacks an explicit license in the retained snapshot; data and paper licenses do not license that code.

The cleanup checked source syntax and command paths. It did not start training or rerun the complete experiments. The original reports, logs, environments and outcome files remain in the archive for comparison; they are not substitutes for results from your new run.

## Month–firm–return table

No direct month–firm–return table is supplied. The input contract has date, asset_id, close_eur, close_raw and currency; asset metadata describes products through asset_type/name/market/sector/industry. It does not provide a verified firm key or dividend/coupon/corporate-action-consistent total-return field. Model ROI is a costed, overlapping six-month recommendation-period metric, not a calendar-month realized return. Producing a separate price-return panel would require an explicit asset-to-firm mapping and return definition; no such mapping or units are inferred here.
