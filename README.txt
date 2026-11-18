TransAlpha: independent model reconstruction and daily CSI adaptation
===================================================================

Paper: Xiao Yang, TransAlpha: Lightweight Design Empowers Stock Return
Forecasting. IJCAI 2026, pp. 5209-5217. DOI: 10.24963/ijcai.2026/580.
https://ijcai.org/proceedings/2026/580

The architecture and five-term loss are reconstructed from the paper.
The completed local experiments are a DAILY ADAPTATION, not a replication
of the paper's private intraday results. See results/REPORT.txt for measured
results and sources/reconstruction_notes.txt for every material assumption.
The results report is generated after fitting and contains an explicit
completion count; checkpoints or queued runs alone do not establish completion.

Run from this directory:
  bash reproduce.sh

The script creates an isolated Python 3.11 environment from the pinned lock
file if needed. It verifies local source bytes, builds all three universes,
tests the implementation, trains three fixed seeds, writes a report, and
audits saved predictions and portfolio accounting. Runs already complete
under an identical code/data/configuration fingerprint are reused.

Individual commands:
  .venv/bin/python scripts/prepare_data.py
  .venv/bin/python scripts/prepare_execution.py
  .venv/bin/python -m pytest -q
  CUBLAS_WORKSPACE_CONFIG=:4096:8 .venv/bin/python scripts/train.py --device cuda
  .venv/bin/python scripts/report.py
  .venv/bin/python scripts/audit.py

Use --universes csi300 --variants transalpha,linear --seeds 0 to select a
smaller experiment. The default suite has 27 fits: three seeds each for
TransAlpha and a linear control on CSI300/500/1000, plus three seeds each
for no-CSD, no-TAG, and no-SP ablations on CSI300. No run is selected using
the 2024 test outcomes. There is no undisclosed hyperparameter search.
CSI500/1000 linear controls use CPU; the neural fits and CSI300 controls
use CUDA. The default runner makes the same device assignments. Minor
numerical differences across hardware and library versions remain possible.

Local data requirements:
  /home/jkp/data/csi/qlib/cn_data
  /home/jkp/data/csi/download_manifest.json
The immutable community snapshot is joshuaxql/qlib_data at revision
ee0fa6b2299cb5e927d2df491888c84f6c9c027a. Historical membership intervals
are applied on each decision date; future labels never decide forecast
eligibility. Source-file SHA256s and derived-array SHA256s are saved.
The independent build does not reuse another experiment's label-filtered
feature cache. Qlib's Alpha158 expressions are stored in each manifest.

Model API, including the paper's 400-dimensional input:
  import torch
  from transalpha import TransAlpha, hybrid_loss
  model = TransAlpha(features=400, hidden=64)
  x = torch.randn(300, 16, 400)  # ONE synchronized cross section
  prediction = model(x)
  target = torch.randn(300)     # next-period returns, supplied by dataset
  loss, components = hybrid_loss(prediction, target, model.parameters())
  loss.backward()

Batched input is [dates, stocks, lookback, factors]; padding is supported
by a [dates, stocks] boolean mask. Never mix unrelated dates along the
stock axis. Never run CSD separately on arbitrary chunks of a cross section.
Missing historical observations require dropping the entire stock window,
with stock IDs aligned across all 16 observations; rows are not reshaped
after independently deleting observations on each day.

Exact intraday replication remains pending:
The current representative target is CSI300 Table 3, specified separately in
.agent/ORIGINAL_EXPERIMENT.json (2026-11-18). It has no completed local run.
See sources/paper-setting-recheck-20261118.json for the renewed input search.
The paper says processed anonymous HDF5 factors are provided, but has no
download/code URL. A user-Chrome search on 2026-11-18 located the official
repository at https://github.com/Barnett888/TransAlpha. Its only commit,
3fb748e453610f5272c9978e769735e3cf07dcbe, contains a 69-byte README, license
and two Git metadata files; there is no runnable code, data or download link.
There are no tags, releases or issues. See sources/official-repository-20261118.json
for the pinned file inventory and browser receipts. This supersedes the older
search's failure to locate the repository. Reproducing Table 3 still needs the exact
400-factor 15-minute panel, aligned stock IDs/returns, preprocessing,
chosen hyperparameters, and precise long/short backtest rules. The daily
source cannot reconstruct these inputs. CSI2000 and full-market experiments
from the paper are not reconstructed here. The linear control is not a
claim to reproduce the paper's DLinear or its other published baselines.

Files:
  config.json                         Frozen daily experiment protocol
  transalpha/model.py                 CSD, TAG, SP, output head, ablations
  transalpha/loss.py                  Literal five-term paper loss
  transalpha/data.py                  Raw-data build, normalization, windows
  transalpha/evaluation.py            IC, rank IC and explicit fee accounting
  scripts/train.py                   Training/checkpointing/test forecasts
  results/<universe>/<variant>-seed*/ Checkpoints, forecasts, daily CSVs
  results/summary.csv                 Equal-seed aggregate measurements
  results/statistics.json             Date-block uncertainty and paired control
  results/REPORT.txt                  Results and limits
  results/overview.png                Measured daily IC and net Sharpe
  results/audit.json                  Independent replay and artifact checks
  requirements.lock.txt               Full environment lock

CZ202 monthly US adaptation (2026-11-17)
======================================
Run bash reproduce_cz202.sh for fixed seeds 0/1/2, TransAlpha and a linear control.
Use --seeds 0 --variants transalpha for a single fit. The 202-feature source
panel uses the lagged NYSE20 size screen and 2009-2019 / 2020-2021 / 2022-2025
splits. Evaluation is value-weighted D10-D1, 10bp costs and FF5+UMD HAC inference.
See DATA.txt for inputs, exact missingness counts, history imputation, source
vintage limitations and missing original-paper inputs. CZ outputs are separate
at results/cz202/; the original daily protocol and its 27 results are preserved.
A completed CZ adaptation does not establish an original intraday paper match.

CZ202 transfer assessment (2026-11-18)
====================================
The six completed fits were independently re-audited. TransAlpha has lower
IC and net Sharpe than the linear control in all three seed pairs. Paired
month-block confidence intervals include zero, so the 48-month test period
does not establish clear superiority or inferiority. The fixed CZ202 setup
does not reproduce an observed TransAlpha advantage. No new model fitting or
test tuning was performed for this assessment.
See results/cz202/TRANSFER_REPORT.txt and transfer_analysis.json for the
comparison, bootstrap sensitivity, paper-reference context and limitations.
To regenerate the comparison after the six-run audit:
  /home/jkp/data/.replication-local/artifacts/ordered-reproductions/.venv-tra/bin/python scripts/analyze_cz202_transfer.py
