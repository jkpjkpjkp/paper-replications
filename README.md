# FraudGNAM: Inherently Interpretable Spectral GNN for Graph Fraud Detection

Partial original reproduction. One of ten full main-method seeds was complete at the archived report checkpoint; the native runner requests all ten. Published auxiliary datasets, baselines, ablations and interpretability suites are not claimed complete. The CPU/DGL 1.1.3 path differs from the paper’s CUDA/DGL 2.4 environment.

## Setup

From the collection root, stage the code and link the existing inputs:

```bash
python _replication/run.py prepare --project 'fraudgnam'
cd '.replication-local/workspace/fraudgnam'
```

The command prints the run directory. Run everything below there. Source folders stay free of data and generated artifacts. Inputs are linked from shared storage and `.replication-local/artifacts/paper-reproduction-audit/workspaces/429-fraudgnam`; preparation does not duplicate them. A clean checkout still requires these external inputs. Keep their release filenames and directory layout when supplying them.

Use Python 3.11 and a compatible existing environment, or create an isolated environment **inside the run directory**:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
```

Skip environment creation and installation when using a compatible environment already provided by preparation. The requirements record the retained execution environment where known; upstream dependency files remain with the corresponding source. Dependency installation was not repeated during this cleanup.

Set `DGLBACKEND=pytorch`. The recorded CPU DGL environment is an explicit compatibility adaptation; the original upstream requirements are retained separately.

## Inputs

The exact retained T-Finance graph at `data/tfinance`, source checksum manifests and upstream reference results under `official/results`. It has 39,357 nodes, about 42.48 million edges, 10 features and 1,803 fraud labels. Reuse it in place; full runs need substantial memory (the archived run used a 12 GiB limit).

Inputs and historical outcomes are not redistributed by this source folder. Preserve the retained upstream license files and the separate terms for datasets and model weights.

## Rerun

Execute the stages sequentially:

```bash
python audit_data.py
python run_tfinance_main.py
python validate_main.py
```

Use a new `--workspace` when you need a completely independent run. To inspect saved outcomes, prepare a separate workspace with `--reuse-results`; this restores historical output copies and may trigger the runners' resume/skip behavior. Use the default fresh results for a new experiment. Acquisition/preparation helpers are included, but the commands above reuse the retained inputs. If rebuilding inputs, prepare a new workspace with `--fresh-inputs` so generated files cannot overwrite linked archives.

## Outputs and validation

`results/data-audit.json`, `results/local-main/trial*/` and main summary/validation files; each completed trial saves predictions, model state and metrics for resumption.

Input auditing verifies split masks; output validation reconstructs scores and checks component predictions. Released label-dependent configuration and class-weight differences remain disclosed; this is not certified leak-free. Fraud classification has no firm-month realized-return target.

The cleanup checked source syntax and command paths. It did not start training or rerun the complete experiments. The original reports, logs, environments and outcome files remain in the archive for comparison; they are not substitutes for results from your new run.

## Month–firm–return table

Not applicable: anonymized T-Finance account nodes have fraud labels, with no disclosed calendar dates or tradable-firm mapping. Fraud scores are not return predictions.
