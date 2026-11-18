# Kronos — pretrained replication

The current run targets the paper's **Kronos-base CSI300 investment simulation
(Table 10)**: annualized excess return 0.1911 and information ratio 1.3782.
It uses the released weights without fine-tuning, a 90-session context, 10-session
forecast, temperature 0.6, top-p 0.9, and 10 Monte Carlo paths. It evaluates all
66,338 eligible stock-day windows across 317 historical CSI300 constituents and
225 signal dates, with independent sampling seeds 0, 1, and 2.

This is a reconstruction using the community Qlib snapshot. The test period is
2024-07-01 through 2025-06-05: the start is specified by the paper, and the end
comes from the authors' released demo configuration. The original data snapshot
and complete backtest specification are unavailable, so agreement would not
establish an exact author-data replay. See `STATUS.txt` and `BLOCKER.txt`.

## First completed result

Seed 0 gives AER **13.7190%/year** and IR **1.205927**, versus the paper's
19.11%/year and 1.3782. Seeds 1/2 are still running. A separate opening-price
limit sensitivity gives 11.7693%/year and 1.050141. The demo-derived primary
limit filter can use closing information for opening trades; see `WRONG.txt`.
This is a material mismatch in the reconstructed experiment, recorded as grade 1.

## Current run

Artifacts: `/mnt/data/home/jkp/replications/kronos-base-20261117/`.
The directory contains frozen source, checkpoint hashes, input hashes, the full
protocol, numerical validation, resumable prediction shards, and live progress.
Weights are pinned to base revision `2b554741eca47781b64468546e77fef3e85130e6`
and tokenizer revision `0e0117387f39004a9016484a186a908917e22426`.

The float32 cache shares historical attention across sampled paths. The current run holds the
model on GPU. Initial shards used blockwise CPU transfers while GPU memory was
constrained; per-shard device provenance is saved. Both the
cached logits and end-to-end sampled forecasts were checked against the released
implementation. No weights are quantized. The original upstream code is unchanged.

```bash
systemctl --user status kronos-base-20261117.service
cat /mnt/data/home/jkp/replications/kronos-base-20261117/progress.json
tail -n 10 /mnt/data/home/jkp/replications/kronos-base-20261117/run.log
```

`scripts/run_pretrained.sh` resumes this frozen run. It obtains an exclusive
lock and skips completed shards. The service retries transient failures up to
five times within ten minutes. If it fails, inspect `failure.json` and `run.log`
before restarting. Running it manually while the service is active is rejected
by the lock. It reuses existing environments without modifying their packages.

Each completed seed automatically produces `seed-N/signals.csv`,
`seed-N/daily-report.csv`, and `seed-N/metrics.json`; `summary.json` reports seed
means, seed SD, paper targets and errors. Qlib uses the previous day's signal,
next-session open execution, top 50/drop 5, a five-session minimum holding period,
and 0.15% buy/sell costs. Primary annualization uses Qlib's 238-day default;
252-day sensitivity is also saved. The other implementation assumptions are
explicit in `protocol.json`. A running job is not a completed replication.

## Historical reduced pilot

The older `scripts/run_kronos.py` performs reduced pretrained inference, not a
matched paper replication. That older runner resolves the latest Hugging Face
revision at the start of each run and records it. It has no saved-revision replay
option, so a later run may use different model weights.

## Prepare the runnable workspace

From this source folder:

```bash
python ../_replication/run.py prepare --project "Kronos"
cd ../.replication-local/workspace/"Kronos"
```

The launcher copies code and links existing local inputs. Raw data stay shared at
the root. Runs write to the staged workspace, keeping this source folder clean.
Use `--workspace /path/to/new-run` for an independent run; use `--fresh-inputs` when
rebuilding prepared data, and `--reuse-results` only to restore historical generated
outputs for replay. Historical README/status notes live under
`../.replication-local/artifacts/LLM/alpha-research-reproduction/`.

## Environment

Python 3.11, PyTorch, pandas, NumPy, Qlib and huggingface_hub. Use the archived inference environment locally; upstream/kronos/requirements.txt describes the released package requirements.

If staging restored `.venv`, use its Python without reinstalling packages into the
archived environment. For a new installation, create an environment outside the
source folder and activate it before running the commands below.

## Inputs

Shared ../csi/qlib/cn_data. scripts/run_kronos.py obtains NeoQuasar tokenizer/small/base checkpoints from Hugging Face and records their resolved revisions. Model weights remain outside the public source. The current runner calls the Hugging Face model-info API even when weights are cached, so network access is required. Reusing an exact recorded revision requires adding an explicit revision selection before running; the current CLI cannot do that.

## Run and verify

Run from the staged project directory using the selected environment:

```bash
OMP_NUM_THREADS=4 python scripts/run_kronos.py
```

## Outputs and reruns

`results/kronos/` contains inference outputs, metrics and resolved checkpoint revisions. The current script writes these outputs again on rerun; use a new workspace to preserve an earlier inference record.

Raw inputs, generated tables, checkpoints, environments and historical progress
reports are local artifacts and are excluded from the source release. Existing
third-party notices and licenses remain with the implementation. No new license
has been assigned to third-party work.

## Monthly return applicability

The retained predictions contain 1–5-session outcomes at four sparse 2026 forecast origins. They do not supply complete security-month return paths or calendar-month endpoints. No monthly table is made by compounding these overlapping horizons. A separate monthly panel would need independently aligned full-month source prices/returns; the sparse prediction windows alone are insufficient.
