# Feature or Penalty? A Placement Diagnostic for Graph-Risk Signals in Learned Portfolio Decisions

Complete declared corrected experiment: 140 new models (100 original-required, 40 auxiliary), 140 released-checkpoint evaluations and 20 constant-score sensitivity evaluations. Exact original replication remains partial: current provider vintages differ, checkpoint lineage is unresolved, and corrected training purges labels, lags VIX and uses exact incentive compatibility.

## Setup

From the collection root, stage the code and link the existing inputs:

```bash
python _replication/run.py prepare --project 'graph-risk-placement'
cd '.replication-local/workspace/graph-risk-placement'
```

The command prints the run directory. Run everything below there. Source folders stay free of data and generated artifacts. Inputs are linked from shared storage and `.replication-local/artifacts/paper-reproduction-audit/workspaces/412`; preparation does not duplicate them. A clean checkout still requires these external inputs. Keep their release filenames and directory layout when supplying them.

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

Retained `data/raw` provider snapshots and read-only factor links, `data/manifest.json`, `sources/source-manifest.json`, and upstream released checkpoints/configurations. Existing raw snapshots are reused. `prepare_data.py` is available for intentional fresh provider acquisition, which changes vintage and is not part of the default replay.

Inputs and historical outcomes are not redistributed by this source folder. Preserve the retained upstream license files and the separate terms for datasets and model weights.

## Rerun

Execute the stages sequentially:

```bash
python audit_release.py
python run_frozen.py
python train_corrected.py
python finalize.py
```

Use a new `--workspace` when you need a completely independent run. To inspect saved outcomes, prepare a separate workspace with `--reuse-results`; this restores historical output copies and may trigger the runners' resume/skip behavior. Use the default fresh results for a new experiment. Acquisition/preparation helpers are included, but the commands above reuse the retained inputs. If rebuilding inputs, prepare a new workspace with `--fresh-inputs` so generated files cannot overwrite linked archives.

## Outputs and validation

`results/release_audit/`, `results/fresh_frozen/`, `results/corrected_training/`, `results/final_audit/` and `figures/`. Training retains 250-epoch cap and early stopping; completed seed/model markers allow resumption.

Finalization verifies checkpoint hashes, feasibility, future-prefix invariance, node exposure identities and 288 risk comparisons. These principal-agent contract outcomes do not define an investable daily portfolio return; no month-firm table is inferred.

The cleanup checked source syntax and command paths. It did not start training or rerun the complete experiments. The original reports, logs, environments and outcome files remain in the archive for comparison; they are not substitutes for results from your new run.

## Month–firm–return table

Not applicable to a firm table: inputs are 49 industry portfolio series and 20 ETFs, not individual firms. The model uses 252-day state windows and overlapping 20-day episodes to estimate principal-agent contract risk/payoff, not calendar-month firm returns. Real dates in the raw snapshots do not turn simulated contract payoffs or seed-level risk comparisons into realized monthly security returns.
