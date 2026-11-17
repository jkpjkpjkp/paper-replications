STABLE: best-effort empirical reconstruction

Latest follow-up (2026-11-17): PAPER_SETTINGS_AUDIT.txt checks the paper and
public reviewer/author discussion. The isolated experiment at
experiments/us_paper_settings_20261117/ corrects documented US setting
differences and adds seeds 20261120 and 20261121. See its REPORT.txt for results
and protocol.json for remaining assumptions. Its revised setup must not be
pooled with the original three US seeds. Original inputs/checkpoints are intact.

Repeat and verify the follow-up on its frozen prepared inputs:

  bash experiments/us_paper_settings_20261117/reproduce.sh

Source/data fingerprints control checkpoint reuse. The script also publishes
the verified follow-up and historical results into the current project status.
Original pre-follow-up assessments are preserved under evidence/before_paper_settings_*.

Original four-market experiment follows:

The completed empirical run trains 12 models: four markets x three seeds,
200 epochs each. It evaluates held-out October 2024–March 2025 portfolio and
forecast performance. The original equation audit is preserved separately.

This is an independent reconstruction with substitute stock universes and
explicit repairs. It is not the unavailable author implementation, and it
does not certify the paper's reported financial results.

Read REPORT.txt for measured results, fallback frequency, and limitations.
STATUS.json is the machine-readable current status. The full protocol was
fixed before test evaluation in protocol.json.

Re-run on the frozen prepared data using the existing environment:

  bash reproduce.sh

The script runs equation checks, eight empirical checks, all model runs,
forecast scoring, independent ledger verification, and report generation.
Completed checkpoints are reused only when their source/data fingerprint
matches. The original run's training source is preserved in
evidence/training_source.py; run_empirical.py additionally corrects the DTW
scoring definition. A fresh run may therefore retrain the original saved
checkpoints. Training and portfolio decisions are unchanged by that scoring
correction. It was applied to all frozen generated paths before reporting.

To use an isolated environment instead:

  python3 -m venv .venv
  .venv/bin/pip install -r requirements-empirical.txt
  STABLE_PYTHON="$PWD/.venv/bin/python" bash reproduce.sh

The measured environment used PyTorch 2.4.1+cu121 and an RTX 4060. CPU execution
is supported but slower. See results/best_effort/environment.json.

To regenerate data from the existing read-only local WRDS archives:

  /home/jkp/data/limt/.venv/bin/python download_macro.py
  /home/jkp/data/limt/.venv/bin/python prepare_data.py

Fresh FRED downloads can contain revisions; the saved panels and CSV hashes
are the exact inputs used here. Source archive locations are recorded in
data/source_files.json. The prepared panels are sufficient for retraining.
The source PDF/table inputs used by verify.py remain under
/home/jkp/data/paper-reproduction-audit/workspaces/105-stable.

Main files:
  diffusion_model.py             Shared 1D U-Net, learnable guidance, DDIM
  run_empirical.py               Causal features, training and backtesting
  prepare_data.py                Substitute-universe selection and panels
  test_empirical.py              Chronology, gradients, accounting, DTW checks
  summarize_empirical.py         Independent ledger checks and final report
  results/best_effort/portfolio_metrics.csv      Every seed and cost scenario
  results/best_effort/forecast_metrics.csv       Per-seed MSE and sampled DTW
  results/best_effort/training_summary.csv       Epoch/checkpoint evidence
  results/best_effort/allocation_diagnostics.csv Fallbacks and repairs
  results/best_effort/verification.json          Completed-run validation
  checkpoints/                  All 12 learned model checkpoints
  data/*_universe.csv            Exact substitute stock identities and order
  EQUATION_REPORT.txt           Preserved original numerical investigation

Per-run weights CSV columns 0,1,... follow the corresponding universe CSV
row order. Saved generated paths use [rebalance, sample, horizon day, stock].
Daily ledgers include turnover and exposure drift. Ten-basis-point costs are
charged per unit of absolute turnover; borrow/financing charges are omitted.

This is STABLE by Soun et al., unrelated to the separate Stable Alpha/CASH
project. Scientific-critical-thinking methods attribution is in REPORT.txt.
