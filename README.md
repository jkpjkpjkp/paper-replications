# Healthcare Insurance Fraud Detection via Continual Fiedler Vector Graph Model

Released seed-10, 10% timeline execution: 364 graphs, including 14 history graphs and 350 online tasks. This is a complete execution of that released entry point, but only a partial reproduction of the paper. The released architecture differs from the paper; evaluation retains active dropout, and the original multi-seed claims are not reproduced.

## Setup

From the collection root, stage the code and link the existing inputs:

```bash
python _replication/run.py prepare --project 'healthcare-insurance-fraud'
cd '.replication-local/workspace/healthcare-insurance-fraud'
```

The command prints the run directory. Run everything below there. Source folders stay free of data and generated artifacts. Inputs are linked from shared storage and `.replication-local/artifacts/paper-reproduction-audit/workspaces/103-healthcare-insurance-fraud`; preparation does not duplicate them. A clean checkout still requires these external inputs. Keep their release filenames and directory layout when supplying them.

Use Python 3.11 and a compatible existing environment, or create an isolated environment **inside the run directory**:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
```

Skip environment creation and installation when using a compatible environment already provided by preparation. The requirements record the retained execution environment where known; upstream dependency files remain with the corresponding source. Dependency installation was not repeated during this cleanup.

## Inputs

The released medical graph file `official/Dataset/0.1/dataset.dat` and checksum metadata `sources/data-manifest.json`. Preparation links the retained file; do not copy it into this source folder.

Inputs and historical outcomes are not redistributed by this source folder. Preserve the retained upstream license files and the separate terms for datasets and model weights.

## Rerun

Execute the stages sequentially:

```bash
python run_released.py --run-id released-seed10-cpu
python audit_outputs.py --run-id released-seed10-cpu
```

Use a new `--workspace` when you need a completely independent run. To inspect saved outcomes, prepare a separate workspace with `--reuse-results`; this restores historical output copies and may trigger the runners' resume/skip behavior. Use the default fresh results for a new experiment. Acquisition/preparation helpers are included, but the commands above reuse the retained inputs. If rebuilding inputs, prepare a new workspace with `--fresh-inputs` so generated files cannot overwrite linked archives.

## Outputs and validation

`results/released-seed10-cpu/`: predictions, four model state dictionaries, metrics, logs and independent output audit. Use a new `--run-id` for an additional run because existing run folders are rejected.

The output audit reconstructs AUC, F1 and accuracy from saved predictions, checks the input checksum and inspects model diagnostics. No calendar-dated firm returns exist for this medical classification task.

The cleanup checked source syntax and command paths. It did not start training or rerun the complete experiments. The original reports, logs, environments and outcome files remain in the archive for comparison; they are not substitutes for results from your new run.

## Month–firm–return table

Not applicable: medical-claims graphs contain fraud labels and model probabilities, not traded-firm identities, calendar months or realized returns.
