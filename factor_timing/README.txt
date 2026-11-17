FACTOR COMBINATIONS AND EARNINGS TIMING

New analysis, kept separate from prior timing-only research.
Results: results/report.html; results/summary.txt; results/paired_inference.csv.

From /home/jkp/data/earnings-timing:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python factor_timing/research.py prepare
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python factor_timing/research.py singles
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python factor_timing/research.py combos
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python factor_timing/research.py train
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python factor_timing/research.py evaluate
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python factor_timing/research.py report
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python factor_timing/research.py validate

The supplied Chen-Zimmermann folder has factor portfolio returns only. Stock
signals were downloaded separately from the official Hugging Face dataset:
https://huggingface.co/datasets/OpenSourceAP/CrossSection/resolve/Data-Release-2026.10/Firm%20Level%20Characteristics/Predictors.parquet
data/download_receipt.json records size and verified published SHA-256.
The data are unsigned; multiply by SignalDoc.Sign before month-wise ranking.
Monthly yyyymm signals are used only in the following calendar month. They
cannot provide contemporaneous announcement-day factor updates.

Universe and event information inherit coverage5/results_regular65. All
screening/combination/choice decisions use 1985-2004 only. Models freeze before
the 2005-2014 validation and 2015-2025 holdout evaluation. Earlier related
holdouts were already inspected, so this is retrospective research.

Primary family: all constituent predictors published by 2004.
Secondary family: all supplied predictors regardless of publication date.
Exact final execution uses each cost scenario's own NAV/holdings and counts
all bought/sold dollars. 5 bp means 0.0005 one-way, plus 1% annual short borrow.
No impact, financing, rebate, measured borrow availability, or terminal close.

Output lineage:
  design.json: frozen definitions before screening
  input_audit.json: signal receipt and checksums for source stock-day panels
  single_screen*: all individual predictors x five timing choices
  combo_screen*: training-ranked pair and equal-weight multi-factor candidates
  exact_train*: cost-specific exact execution for screen shortlist
  frozen_choices.json: choices before later return evaluation
  final*: exact final daily arrays, model specifications and statistics
  extra_month_lag*, liquid*: primary-strategy sensitivities
  paired_inference.csv: HAC20/60, Holm across frozen event candidates
  primary_robustness.csv: paired block-bootstrap and monthly HAC3
  daily_returns.csv, annual_returns.csv: reproducible return series
  validation.json: accounting and original timing-rule reproduction checks

Sources and procedural-method citation are included in the HTML report.
