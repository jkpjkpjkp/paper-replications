EARNINGS ANNOUNCEMENT TIMING RESEARCH

View results/report.html for findings, definitions, limitations, and references.
View results/earnings_timing.png for the holdout timing pattern and performance.

Intermediate-data cleanup (2026-11-03): generated stock-day panels, feature
and horizon caches, superseded fits, and runtime logs have been removed.
Source code, raw factor inputs, reports, result tables/arrays, audit records,
and the local Python environment are retained. See CLEANUP.txt for the removed
file groups and dependency order to rebuild them before further analysis.

Reproduce from /home/jkp/data/earnings-timing:
  .venv/bin/python analyze.py
  .venv/bin/python validate.py

Dependencies and exact installed versions are in requirements.txt.

Raw inputs are read-only. This analysis uses:
  ../crsp-daily/data/crsp_a_stock/dsf_v2/1982.parquet through 2025.parquet
  ../ibes/all-parquet/ibes/act_epsus/part-*.parquet

The older ../crsp-daily/parquet directory ends in 1976 and is not the return
input. Completed raw CRSP receipts and the completed EPS-actuals manifest
are checked against actual row counts, byte sizes, and SHA-256 checksums.

Output inventory:
  results/input_audit.json: exact input coverage/checksums and source manifest
  results/event_audit.json: event linkage, information states, forecast quality
  results/linked_events.parquet: linked observations, history, availability
  results/trading_calendar.csv: market-wide session index
  results/panel/YYYY.parquet: lagged-screen stock returns and signal clocks
  results/panel/market_YYYY.parquet: broad local EW/VW common-stock markets
  results/daily_timing_bins.csv: calendar-time returns in six past-event bins
  results/timing_bin_statistics.csv: bin means, intervals, HAC inference
  results/daily_strategies.csv: positions counts, gross/net P&L and turnover
  results/strategy_statistics.csv: subperiod returns and market controls
  results/annual_strategy_returns.csv: yearly compounded returns
  results/breakeven_costs.csv: implied maximum trading cost after 1% borrow
  results/inference_robustness.csv: alternative daily HAC lags/monthly results
  results/validation_checks.json: accounting and chronology validation

The regular=True screen counts the current announcement and three preceding
known announcements within 400 days. Only the latest economically dated event
known at the previous close is usable; late old records cannot rewind the clock.

The fields with DIAGNOSTIC_ONLY in their names use future actual dates only
to assess the forecast. They are never included in the signal-state table.
The seasonal pre-announcement variant was added after inspecting the first
holdout run and forecast diagnostics; its reported results are exploratory.
Predicted dates beyond the observed trading calendar cannot define a trading
session distance and are excluded from pre-announcement signals.

Stock-day panel columns include:
  abnormal_ew = stock total return - local equal-weight market return
  abnormal_vw = stock total return - local value-weight market return
  age = trading sessions since the latest known earnings announcement
  to_predicted = sessions until the median-gap forecast date
  to_seasonal_predicted = sessions until the prior-year forecast date
  nearest_predicted_signed_clock = age for a nearer past event, or a negative
    distance to a nearer predicted future event (ties select the past event)
There is no realized-future nearest-announcement signal.

Return conventions:
  gross = equal-weight long return minus equal-weight short underlying return
  net_1bp_1pct_borrow = gross - 0.0001 * turnover - active * 0.01 / 252
  net_10bps_1pct_borrow = gross - 0.001 * turnover - active * 0.01 / 252
  net_25bps_2pct_borrow = gross - 0.0025 * turnover - active * 0.02 / 252
  turnover = sum of absolute dollar position changes / gross-path equity

Turnover measures all dollars bought and sold, rather than half the sum.
The net scenarios apply costs to the same gross-path turnover; they do not
simulate the second-order effects of cost deductions on future share counts.
No terminal liquidation charge is imposed. Missing held-stock returns are
treated as zero and counted. Disappearing names are closed at their last mark.
Borrow and transaction costs are scenarios, not measured historical fees.

The zero-net long/short spread has 200% gross exposure on $1 capital. The
annual mean is 252 times the daily arithmetic mean; annual_geometric is the
compounded growth rate. These differ. Market-controlled intercepts regress
on raw locally constructed VW market returns without an official risk-free
series, and are not standard CAPM/Fama-French alphas. Financing and collateral
rebates are not modeled. Exact-CUSIP retrospective linkage and incomplete
historical availability versions remain limitations.

This is retrospective chronological validation, not live out-of-sample
performance. The published cycle hypothesis itself predates only part of
the holdout, so splitting dates does not remove publication-related hindsight.
