# Corporate customer-graph anomaly study

Chronological graph anomaly scores use disclosed relationships and filing dates. No stock return is used by the scoring algorithm; a month–firm–return output is not applicable to this task.

## Representative parent-paper number

The corporate study is an independent extension. Its `paper29` reference points to
EB-GAD, publicly available as [arXiv:2609.38424v1](https://arxiv.org/html/2609.38424v1).
On 2026-11-17, independent re-evaluation of archived local LOF baseline scores
gave Elliptic AUROC **45.0857%**, matching Table 1's **45.1%** within rounding.
Three companion AUROC/AUPRC cells also match. This is a scoped baseline result;
EB-GAD and the corporate extension remain unfinished. See
[REPRESENTATIVE_NUMBER.txt](REPRESENTATIVE_NUMBER.txt) for protocol, limitations,
evidence and commands. Fresh fitting was deferred by resource admission.

## Prepare the runnable workspace

From this source folder:

```bash
python ../_replication/run.py prepare --project "corporate-graph-anomalies"
cd ../.replication-local/workspace/"corporate-graph-anomalies"
```

The launcher copies code and links existing local inputs. Raw data stay shared at
the root. Runs write to the staged workspace, keeping this source folder clean.
Use `--workspace /path/to/new-run` for an independent run; use `--fresh-inputs` when
rebuilding prepared data, and `--reuse-results` only to restore historical generated
outputs for replay. Historical README/status notes live under
`../.replication-local/artifacts/corporate-graph-anomalies/`.

## Environment

Python 3 with NumPy, pandas, PyArrow, SciPy, scikit-learn and threadpoolctl. WRDS acquisition additionally uses the existing shared downloader environment and authorized credentials.

If staging restored `.venv`, use its Python without reinstalling packages into the
archived environment. For a new installation, create an environment outside the
source folder and activate it before running the commands below.

## Inputs

Reuse raw disclosed customer relationships and existing Compustat/CRSP identifiers. download_relations.py acquires inputs, prepare_relations.py constructs data/edges.parquet and data/matched.parquet. Local staging restores these cached inputs; inspect analysis_config.json and study-plan.json before changing the design.

## Run and verify

Run from the staged project directory using the selected environment:

```bash
mkdir -p results
python prepare_relations.py
python score_graphs.py
```

## Outputs and reruns

`data/` preparation audits and `scores/` graph-anomaly outputs. inspect_wrds.py is an optional catalogue check; it is not a training stage.

Raw inputs, generated tables, checkpoints, environments and historical progress
reports are local artifacts and are excluded from the source release. Existing
third-party notices and licenses remain with the implementation. No new license
has been assigned to third-party work.
