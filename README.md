# Lookahead factor models: author source

This folder contains the LFM Quant author implementation associated with [Uncertainty-Aware Lookahead Factor Models for Quantitative Investing](https://proceedings.mlr.press/v119/chauhan20a.html). The pinned repository revision is `ac6f47c9a36f681920314423e2502f3654c5b592` from `lakshaykc/lfm_quant`.

The working reproduction adapter and rerun guide are in [`reproduction/README.md`](reproduction/README.md). Start there: it provides a Python 3.11/TensorFlow 2.15 runtime, explicit source repairs, tests and a synthetic integration check. This original checkout is retained as the input to `reproduction/scripts/prepare_runtime.py`.

## Code and data contract

| Path | Purpose |
| --- | --- |
| `scripts/lfm_quant.py` | Original training and inference entry point |
| `scripts/data_processing.py` | Monthly company-panel dataset implementation |
| `scripts/dataset_api.md` | Dataset contract notes |
| `scripts/models/` | Forecasting architectures |
| `scripts/utils/post_processing.py` | Prediction and uncertainty combination |

The original README's `config/system-test.conf` and `open-dataset.dat` are absent from the release. Do not run those example paths or treat them as supplied inputs. The available `Dataset` implementation expects a space-delimited monthly company panel with dates, `gvkey`, market capitalization, configured financial fields and auxiliary fields. The original data-generation path has a private `cdrs` dependency; the local adapter loads it only when that path is requested.

## Reproduce the implemented work

From the collection root:

```bash
python _replication/run.py prepare --project lfm-quant
cd .replication-local/workspace/lfm-quant
# Follow the environment setup in this workspace's README.md if .venv is absent.
bash run_smoke.sh
```

The smoke run generates a labelled synthetic fixture and verifies code integration. It is not a paper investment-return experiment. A full empirical run additionally needs authorized historical Compustat Snapshot vintages or the authors' processed panel, the exact field mapping/universe, and the portfolio construction. Existing current/restated Compustat files cannot establish that historical snapshot. No month–firm–return result is claimed for this incomplete empirical experiment.

The code was released by Euclidean Technologies; the author README describes it as MIT-licensed. The original attribution and text are preserved in the local archive. No new license is assigned to third-party material during this cleanup.
